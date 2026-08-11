from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


BASE_VIEW_COUNT = 14
MID_VIEW_COUNT = 28
MASTER_VIEW_COUNT = 56
VIEW_COUNTS = (BASE_VIEW_COUNT, MID_VIEW_COUNT, MASTER_VIEW_COUNT)
LAYOUT_VERSION = "cube_surface_nested_fps_antipodal_14_28_56_cpu_master_v3"
TARGET_MODE = "edge_scale_normalized_laplacian"


def _expand(path: Path) -> Path:
    return path.expanduser().resolve()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _resolve_dataset_path(dataset_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (dataset_dir / path).resolve()


def _camera_center(camera: Any) -> np.ndarray:
    rotation = np.asarray(camera.rotation, dtype=np.float64)
    translation = np.asarray(camera.translation, dtype=np.float64)
    return -rotation.T @ translation


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def _canonical_antipode(direction: np.ndarray) -> np.ndarray:
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
    for value in direction:
        if abs(float(value)) > 1e-12:
            return direction if value > 0.0 else -direction
    return direction


def _fibonacci_pair_candidates(count: int = 16384) -> np.ndarray:
    if count < 100:
        raise ValueError("candidate count is too small")
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    unique: dict[tuple[float, float, float], np.ndarray] = {}
    for idx in range(count):
        z = 1.0 - 2.0 * (idx + 0.5) / count
        radius = math.sqrt(max(0.0, 1.0 - z * z))
        phi = idx * golden_angle
        direction = np.array(
            [radius * math.cos(phi), z, radius * math.sin(phi)],
            dtype=np.float64,
        )
        canonical = _canonical_antipode(direction)
        key = tuple(float(v) for v in np.round(canonical, 12))
        unique.setdefault(key, canonical)
    return np.stack(list(unique.values()), axis=0)


def _greedy_farthest_antipodal_pairs(
    base_directions: np.ndarray,
    num_pairs: int,
    *,
    candidate_count: int = 16384,
) -> list[np.ndarray]:
    existing = _normalize_rows(np.asarray(base_directions, dtype=np.float64))
    candidates = _fibonacci_pair_candidates(candidate_count)
    selected: list[np.ndarray] = []
    for _ in range(num_pairs):
        # Existing directions contain antipodal partners, so the closest-view cosine
        # for one member of a candidate pair also controls the opposite member.
        closest_cos = np.max(candidates @ existing.T, axis=1)
        best_index = int(np.argmin(closest_cos))
        direction = candidates[best_index]
        selected.append(direction.copy())
        existing = np.vstack([existing, direction[None, :], -direction[None, :]])
    return selected


def _cube_surface_center(direction: np.ndarray, half_extent: float) -> np.ndarray:
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
    denom = float(np.max(np.abs(direction)))
    if denom <= 1e-12:
        raise ValueError("invalid camera direction")
    return direction * (float(half_extent) / denom)


def _minimum_angular_separation_degrees(directions: np.ndarray) -> float:
    directions = _normalize_rows(np.asarray(directions, dtype=np.float64))
    cosine = np.clip(directions @ directions.T, -1.0, 1.0)
    np.fill_diagonal(cosine, -1.0)
    nearest = np.max(cosine, axis=1)
    return float(np.degrees(np.arccos(np.clip(np.max(nearest), -1.0, 1.0))))


def _build_nested_cameras(base_cameras: list[Any], deps: dict[str, Any]) -> tuple[list[Any], dict[str, Any]]:
    if len(base_cameras) != BASE_VIEW_COUNT:
        raise ValueError(f"Expected {BASE_VIEW_COUNT} base cameras, got {len(base_cameras)}")

    centers = np.stack([_camera_center(camera) for camera in base_cameras], axis=0)
    half_extent = float(np.max(np.abs(centers)))
    if not np.isfinite(half_extent) or half_extent <= 1.0:
        raise ValueError(f"Invalid inferred cube half extent: {half_extent}")
    base_directions = _normalize_rows(centers)

    # Add 7 antipodal pairs to 14 -> 28, then 14 more pairs to 28 -> 56.
    pair_directions = _greedy_farthest_antipodal_pairs(base_directions, num_pairs=21)
    all_cameras = list(base_cameras)
    added_records: list[dict[str, Any]] = []
    target = np.zeros(3, dtype=np.float64)
    intrinsics = np.asarray(base_cameras[0].intrinsics, dtype=np.float64)
    image_size = tuple(base_cameras[0].image_size)

    for pair_index, direction in enumerate(pair_directions):
        level = 1 if pair_index < 7 else 2
        level_pair_index = pair_index if level == 1 else pair_index - 7
        for sign_name, signed_direction in (("pos", direction), ("neg", -direction)):
            center = _cube_surface_center(signed_direction, half_extent)
            rotation, translation = deps["look_at_world_to_camera"](center, target)
            name = f"nested_l{level}_pair_{level_pair_index:02d}_{sign_name}"
            camera = deps["Camera"](
                intrinsics=intrinsics.copy(),
                rotation=rotation,
                translation=translation,
                image_size=image_size,
                name=name,
            )
            all_cameras.append(camera)
            added_records.append(
                {
                    "name": name,
                    "level": level,
                    "pair_index": int(level_pair_index),
                    "sign": sign_name,
                    "direction": signed_direction.tolist(),
                    "center": center.tolist(),
                }
            )

    if len(all_cameras) != MASTER_VIEW_COUNT:
        raise AssertionError(f"Nested camera builder produced {len(all_cameras)} cameras")

    all_centers = np.stack([_camera_center(camera) for camera in all_cameras], axis=0)
    all_directions = _normalize_rows(all_centers)
    nesting = {
        "layout_version": LAYOUT_VERSION,
        "counts": list(VIEW_COUNTS),
        "subset_rule": "prefix",
        "indices": {str(count): list(range(count)) for count in VIEW_COUNTS},
        "cube_half_extent": half_extent,
        "minimum_angular_separation_degrees": {
            str(count): _minimum_angular_separation_degrees(all_directions[:count])
            for count in VIEW_COUNTS
        },
        "base_14_names": [camera.name for camera in base_cameras],
        "added": added_records,
        "centers_56": all_centers.tolist(),
    }
    return all_cameras, nesting


def _link_or_copy(src: Path, dst: Path, *, force: bool) -> None:
    src = src.resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not force:
            return
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _save_rendered_views(
    output_dir: Path,
    cameras: list[Any],
    rendered_views: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    start_index: int,
) -> None:
    image_dir = output_dir / "images"
    mask_dir = output_dir / "masks"
    depth_dir = output_dir / "depth"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    if len(cameras) != len(rendered_views):
        raise ValueError("camera/rendered-view count mismatch")
    for offset, rendered_view in enumerate(rendered_views):
        index = start_index + offset
        rgb, mask, depth = rendered_view
        Image.fromarray(rgb).save(image_dir / f"{index:04d}.png")
        Image.fromarray((mask.astype(np.uint8) * 255)).save(mask_dir / f"{index:04d}.png")
        np.save(depth_dir / f"{index:04d}.npy", depth)


def _paths_for_count(root: Path, count: int) -> tuple[list[Path], list[Path], list[Path]]:
    images = [root / "images" / f"{index:04d}.png" for index in range(count)]
    masks = [root / "masks" / f"{index:04d}.png" for index in range(count)]
    depths = [root / "depth" / f"{index:04d}.npy" for index in range(count)]
    return images, masks, depths


def _validate_base_render(base_dataset_path: Path, payload: dict[str, Any]) -> None:
    config = payload.get("config", {})
    if int(config.get("num_views", -1)) != BASE_VIEW_COUNT:
        raise ValueError(f"{base_dataset_path}: base dataset is not 14-view")
    if str(config.get("trajectory")) != "cube_surface":
        raise ValueError(f"{base_dataset_path}: expected cube_surface base trajectory")
    if bool(config.get("normalize_mesh", True)):
        raise ValueError(f"{base_dataset_path}: expected normalize_mesh=false")
    if int(config.get("width", 0)) != 960 or int(config.get("height", 0)) != 960:
        raise ValueError(f"{base_dataset_path}: expected 960x960 base render")
    if abs(float(config.get("fov_degrees", 0.0)) - 90.0) > 1e-9:
        raise ValueError(f"{base_dataset_path}: expected 90 degree FOV")
    if str(config.get("backend")) != "opengl":
        raise ValueError(f"{base_dataset_path}: base actual backend must be opengl")
    if str(config.get("requested_backend")) != "opengl":
        raise ValueError(f"{base_dataset_path}: base requested backend must be opengl")


def _render_master_56(
    model_id: str,
    gt_path: Path,
    base_dataset_path: Path,
    master_dir: Path,
    deps: dict[str, Any],
    *,
    force: bool,
) -> tuple[list[Any], dict[str, Any], str, dict[str, Any]]:
    payload = _read_json(base_dataset_path)
    _validate_base_render(base_dataset_path, payload)
    reconstruction = deps["load_reconstruction_input"](base_dataset_path)
    base_cameras = list(reconstruction.cameras)
    all_cameras, nesting = _build_nested_cameras(base_cameras, deps)

    if force and master_dir.exists():
        shutil.rmtree(master_dir)
    master_dir.mkdir(parents=True, exist_ok=True)

    gt_mesh = deps["load_mesh"](gt_path).ensure_normals()
    base_config = payload["config"]
    render_config = deps["SyntheticRenderConfig"](
        num_views=MASTER_VIEW_COUNT,
        width=int(base_config["width"]),
        height=int(base_config["height"]),
        trajectory="nested_cube_surface",
        fov_degrees=float(base_config["fov_degrees"]),
        render_mode=str(base_config.get("render_mode", "lit")),
        backend="cpu",
        normalize_mesh=False,
        opengl_context_backend=str(base_config.get("opengl_context_backend", "egl")),
        cube_half_extent=float(base_config.get("cube_half_extent", nesting["cube_half_extent"])),
        antialiasing=str(base_config.get("antialiasing", "msaa4")),
        camera_layout_version=LAYOUT_VERSION,
        backface_culling=bool(base_config.get("backface_culling", False)),
        front_face_winding=str(base_config.get("front_face_winding", "ccw")),
    )

    # Render every observation with the same deterministic CPU reference backend.
    # This avoids the renderer-domain confound caused by mixing historic OpenGL
    # views with CPU fallback views when EGL is unavailable on the current node.
    print(f"  rendering all {MASTER_VIEW_COUNT} views with strict CPU reference for {model_id}", flush=True)
    rendered = [
        deps["render_mesh_view"](gt_mesh, camera, render_config)
        for camera in all_cameras
    ]
    _save_rendered_views(
        master_dir,
        all_cameras,
        rendered,
        start_index=0,
    )
    actual_backend = "cpu"

    mesh_path = master_dir / "mesh.obj"
    _link_or_copy(gt_path, mesh_path, force=force)
    images, masks, depths = _paths_for_count(master_dir, MASTER_VIEW_COUNT)
    cameras_path = master_dir / "cameras.json"
    dataset_path = master_dir / "dataset.json"
    deps["write_cameras_json"](
        cameras_path, all_cameras, images, masks, depths, master_dir
    )
    deps["write_dataset_json"](
        dataset_path,
        cameras=all_cameras,
        cameras_path=cameras_path,
        image_paths=images,
        mask_paths=masks,
        depth_paths=depths,
        mesh_path=mesh_path,
        source_mesh_path=gt_path,
        out_dir=master_dir,
        config=render_config,
        actual_backend=actual_backend,
    )

    # Diagnostic only: compare the rerendered first 14 against the old OpenGL base.
    # The new 14/28/56 experiment remains internally clean even if code/version
    # differences make the rerender non-identical to the historic 14-view files.
    base_dir = base_dataset_path.parent
    rgb_equal = True
    mask_equal = True
    depth_max_abs_diff = 0.0
    depth_finite_pattern_equal = True
    for key, subdir, suffix in (
        ("image_paths", "images", ".png"),
        ("mask_paths", "masks", ".png"),
        ("depth_paths", "depth", ".npy"),
    ):
        values = payload.get(key)
        if not isinstance(values, list) or len(values) != BASE_VIEW_COUNT:
            raise ValueError(f"{base_dataset_path}: invalid {key}")
        for index, value in enumerate(values):
            old_path = _resolve_dataset_path(base_dir, str(value))
            new_path = master_dir / subdir / f"{index:04d}{suffix}"
            if key == "image_paths":
                old = np.asarray(Image.open(old_path).convert("RGB"))
                new = np.asarray(Image.open(new_path).convert("RGB"))
                rgb_equal = rgb_equal and bool(np.array_equal(old, new))
            elif key == "mask_paths":
                old = np.asarray(Image.open(old_path))
                new = np.asarray(Image.open(new_path))
                mask_equal = mask_equal and bool(np.array_equal(old, new))
            else:
                old = np.load(old_path)
                new = np.load(new_path)
                old_finite = np.isfinite(old)
                new_finite = np.isfinite(new)
                depth_finite_pattern_equal = depth_finite_pattern_equal and bool(
                    np.array_equal(old_finite, new_finite)
                )
                both = old_finite & new_finite
                if np.any(both):
                    depth_max_abs_diff = max(
                        depth_max_abs_diff,
                        float(np.max(np.abs(old[both] - new[both]))),
                    )

    base14_rerender_check = {
        "rgb_pixel_equal": bool(rgb_equal),
        "mask_pixel_equal": bool(mask_equal),
        "depth_finite_pattern_equal": bool(depth_finite_pattern_equal),
        "depth_max_abs_diff_on_common_finite": float(depth_max_abs_diff),
    }
    _write_json(master_dir / "base14_rerender_check.json", base14_rerender_check)
    _write_json(master_dir / "nested_camera_layout.json", nesting)
    return all_cameras, nesting, actual_backend, base14_rerender_check


def _materialize_subset_dataset(
    master_dir: Path,
    subset_dir: Path,
    count: int,
    all_cameras: list[Any],
    gt_path: Path,
    deps: dict[str, Any],
    *,
    actual_backend: str,
    force: bool,
) -> Path:
    if count not in VIEW_COUNTS:
        raise ValueError(f"Unsupported subset count {count}")
    if subset_dir.resolve() == master_dir.resolve():
        return master_dir / "dataset.json"
    if force and subset_dir.exists():
        shutil.rmtree(subset_dir)
    subset_dir.mkdir(parents=True, exist_ok=True)

    for index in range(count):
        for subdir, suffix in (("images", ".png"), ("masks", ".png"), ("depth", ".npy")):
            _link_or_copy(
                master_dir / subdir / f"{index:04d}{suffix}",
                subset_dir / subdir / f"{index:04d}{suffix}",
                force=force,
            )
    _link_or_copy(gt_path, subset_dir / "mesh.obj", force=force)

    master_payload = _read_json(master_dir / "dataset.json")
    master_config = master_payload["config"]
    config = deps["SyntheticRenderConfig"](
        num_views=count,
        width=int(master_config["width"]),
        height=int(master_config["height"]),
        trajectory="nested_cube_surface",
        fov_degrees=float(master_config["fov_degrees"]),
        render_mode=str(master_config.get("render_mode", "lit")),
        backend="cpu",
        normalize_mesh=False,
        opengl_context_backend=str(master_config.get("opengl_context_backend", "egl")),
        cube_half_extent=float(master_config["cube_half_extent"]),
        antialiasing=str(master_config.get("antialiasing", "msaa4")),
        camera_layout_version=LAYOUT_VERSION,
        backface_culling=bool(master_config.get("backface_culling", False)),
        front_face_winding=str(master_config.get("front_face_winding", "ccw")),
    )
    cameras = all_cameras[:count]
    images, masks, depths = _paths_for_count(subset_dir, count)
    cameras_path = subset_dir / "cameras.json"
    dataset_path = subset_dir / "dataset.json"
    deps["write_cameras_json"](
        cameras_path, cameras, images, masks, depths, subset_dir
    )
    deps["write_dataset_json"](
        dataset_path,
        cameras=cameras,
        cameras_path=cameras_path,
        image_paths=images,
        mask_paths=masks,
        depth_paths=depths,
        mesh_path=subset_dir / "mesh.obj",
        source_mesh_path=gt_path,
        out_dir=subset_dir,
        config=config,
        actual_backend=actual_backend,
    )
    return dataset_path


def _prepared_source(
    model_id: str,
    split: str,
    count: int,
    dataset_path: Path,
    gt_path: Path,
    output_root: Path,
    deps: dict[str, Any],
) -> dict[str, Any]:
    source = deps["prepare_same_topology_sample"](
        dataset_path,
        gt_path,
        gt_path,
        image_size=960,
        target_mode=TARGET_MODE,
        extra_metadata={
            "dataset_family": "sofa50_nested_views_14_28_56_cpu_v4",
            "source_sample_id": model_id,
            "source_split": split,
            "render_geometry_role": "gt_mesh",
            "render_source_path": str(gt_path),
            "nested_view_count": count,
            "nested_master_view_count": MASTER_VIEW_COUNT,
            "nested_subset_rule": "prefix",
            "camera_layout_version": LAYOUT_VERSION,
            "base_14_camera_poses_reused_exactly": True,
            "base_14_observations_reused_exactly": False,
            "all_views_rendered_in_single_cpu_master": True,
            "renderer_fallback_allowed": False,
            "renderer_backend_control": "all_nested_views_cpu_reference",
        },
    )
    source["sample_id"] = model_id
    source.pop("images", None)
    render_payload = _read_json(dataset_path)
    image_paths = [
        _resolve_dataset_path(dataset_path.parent, str(value))
        for value in render_payload["image_paths"]
    ]
    source["image_paths"] = [
        path.resolve().relative_to(output_root.resolve()).as_posix() for path in image_paths
    ]
    source["prepared_storage_format"] = "lazy_image_paths_v1"
    source["source_image_size"] = [960, 960]
    source["prepared_image_size"] = 960
    return source


def _attach_renderer_visibility(
    sample: dict[str, Any],
    result: Any,
    count: int,
    artifact_path: Path,
    *,
    graph_role: str,
    backend: str,
) -> None:
    arrays = {
        "frustum_valid": result.frustum_valid[:count].copy(),
        "visibility_backface_only": result.backface_visible[:count].copy(),
        "visibility_occlusion_only": result.occlusion_visible[:count].copy(),
        "visibility_backface_and_occlusion": (
            result.backface_and_occlusion_visible[:count].copy()
        ),
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(artifact_path, **arrays)
    for name in (
        "visibility_backface_only",
        "visibility_occlusion_only",
        "visibility_backface_and_occlusion",
    ):
        sample[name] = torch.from_numpy(arrays[name])
    sample["visibility"] = sample["visibility_backface_and_occlusion"]
    visible_counts = arrays["visibility_backface_and_occlusion"].sum(axis=0)
    sample["metadata"].update(
        {
            "renderer_visibility_recompute_required": False,
            "visibility_policy": "renderer_native_current_query_graph_and_view_subset",
            "renderer_visibility": {
                "definition": "depth_tested_face_id_incident_face_neighborhood",
                "artifact_path": str(artifact_path),
                "backend": backend,
                "front_face_winding": "ccw",
                "neighborhood_radius": 1,
                "depth_image_used": False,
                "graph_role": graph_role,
                "view_count": count,
                "mean_visible_views_per_vertex": float(visible_counts.mean()),
                "zero_visible_vertex_ratio": float(np.mean(visible_counts == 0)),
            },
        }
    )


def _prepare_gt_query_sample(
    source: dict[str, Any],
    model_id: str,
    count: int,
    prepared_path: Path,
    visibility_result: Any,
    visibility_artifact_path: Path,
    visibility_backend: str,
    deps: dict[str, Any],
) -> None:

    sample = deps["prepare_gt_query_sample_from_prepared"](
        source,
        target_mode=TARGET_MODE,
    )
    sample["sample_id"] = model_id
    sample["metadata"].update(
        {
            "training_eligible": True,
            "nested_view_count": count,
            "nested_master_view_count": MASTER_VIEW_COUNT,
            "nested_subset_rule": "prefix",
            "camera_layout_version": LAYOUT_VERSION,
            "base_14_camera_poses_reused_exactly": True,
            "base_14_observations_reused_exactly": False,
            "all_views_rendered_in_single_cpu_master": True,
            "renderer_fallback_allowed": False,
            "renderer_backend_control": "all_nested_views_cpu_reference",
        }
    )
    sample["metadata"].pop("visibility_artifact", None)
    sample["metadata"].pop("renderer_visibility_artifact", None)
    _attach_renderer_visibility(
        sample,
        visibility_result,
        count,
        visibility_artifact_path,
        graph_role="gt_query_training",
        backend=visibility_backend,
    )
    deps["save_prepared_sample"](sample, prepared_path)


def _prepare_expanded_inference_sample(
    source: dict[str, Any],
    model_id: str,
    split: str,
    count: int,
    expanded_path: Path,
    gt_path: Path,
    prepared_path: Path,
    visibility_result: Any,
    visibility_artifact_path: Path,
    visibility_backend: str,
    deps: dict[str, Any],
) -> None:
    expanded_mesh = deps["load_mesh"](expanded_path).ensure_normals()
    gt_mesh = deps["load_mesh"](gt_path).ensure_normals()
    vertices = torch.as_tensor(expanded_mesh.vertices, dtype=torch.float32)
    faces = torch.as_tensor(expanded_mesh.faces, dtype=torch.long)
    laplacian_data = deps["build_uniform_laplacian_data"](
        expanded_mesh.faces, expanded_mesh.num_vertices
    )
    initial = torch.as_tensor(
        deps["apply_uniform_laplacian"](expanded_mesh.vertices, laplacian_data),
        dtype=torch.float32,
    )
    center = 0.5 * (vertices.amin(dim=0) + vertices.amax(dim=0))
    scale = torch.linalg.vector_norm(vertices - center, dim=-1).amax().reshape(())
    sample = {
        "sample_id": model_id,
        "image_paths": list(source["image_paths"]),
        "prepared_storage_format": "lazy_image_paths_v1",
        "source_image_size": [960, 960],
        "prepared_image_size": 960,
        "intrinsics": source["intrinsics"].clone(),
        "extrinsics": source["extrinsics"].clone(),
        "vertices": vertices,
        "faces": faces,
        "vertex_normals": torch.as_tensor(expanded_mesh.normals, dtype=torch.float32),
        "initial_laplacian": initial,
        "laplacian_target": initial.clone(),
        "target_confidence": torch.ones(expanded_mesh.num_vertices, dtype=torch.float32),
        "visibility": None,
        "gt_vertices": torch.as_tensor(gt_mesh.vertices, dtype=torch.float32),
        "gt_faces": torch.as_tensor(gt_mesh.faces, dtype=torch.long),
        "position_normalization_center": center,
        "position_normalization_scale": scale,
        "metadata": {
            "dataset_family": "sofa50_nested_views_14_28_56_cpu_v4",
            "source_sample_id": model_id,
            "source_split": split,
            "usage_role": "frozen_model_inference_and_reconstruction_evaluation_only",
            "training_eligible": False,
            "query_geometry_role": "expanded_initial_raw",
            "query_mesh_path": str(expanded_path),
            "gt_evaluation_mesh_path": str(gt_path),
            "render_geometry_role": "gt_mesh",
            "nested_view_count": count,
            "nested_master_view_count": MASTER_VIEW_COUNT,
            "nested_subset_rule": "prefix",
            "camera_layout_version": LAYOUT_VERSION,
            "target_field_role": "schema_required_identity_placeholder_not_training_supervision",
            "target_constructor": "uniform_laplacian_of_expanded_initial_raw_identity_reference",
            "gt_oracle_target_used": False,
            "P_target_oracle_used": False,
        },
    }
    deps["attach_target_scaling"](
        sample,
        TARGET_MODE,
        1e-12,
        edge_scale_source="expanded_initial_raw_graph_identity_reference",
    )
    _attach_renderer_visibility(
        sample,
        visibility_result,
        count,
        visibility_artifact_path,
        graph_role="expanded_initial_raw_inference",
        backend=visibility_backend,
    )
    deps["save_prepared_sample"](sample, prepared_path)


def _manifest(
    samples: list[dict[str, str]], count: int, *, expanded: bool
) -> dict[str, Any]:
    return {
        "format_version": (
            "sofa50_expanded_inference_nested_views_v4"
            if expanded
            else "sofa50_gt_query_nested_views_v4"
        ),
        "dataset_role": (
            "expanded_raw_frozen_model_inference" if expanded else "gt_query_training"
        ),
        "training_eligible": not expanded,
        "view_count": count,
        "master_view_count": MASTER_VIEW_COUNT,
        "nested_subset_rule": "prefix",
        "camera_layout_version": LAYOUT_VERSION,
        "target_mode": TARGET_MODE,
        "samples": samples,
    }


def _compute_master_visibility(
    mesh: Any,
    cameras: list[Any],
    backend: str,
    deps: dict[str, Any],
) -> Any:
    if backend == "cuda":
        from sofa50_refinement.gpu_visibility import (
            compute_renderer_visibility_cuda,
        )

        return compute_renderer_visibility_cuda(
            mesh,
            cameras,
            image_size=960,
            neighborhood_radius=1,
            front_face_winding="ccw",
        )
    config = deps["SyntheticRenderConfig"](
        num_views=len(cameras),
        width=960,
        height=960,
        backend=backend,
        normalize_mesh=False,
        antialiasing="none",
        backface_culling=False,
        front_face_winding="ccw",
    )
    return deps["compute_renderer_visibility"](
        mesh, cameras, config, neighborhood_radius=1
    )


def _prepared_contract_complete(path: Path, artifact_path: Path, count: int) -> bool:
    if not path.is_file() or not artifact_path.is_file():
        return False
    try:
        sample = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, TypeError, ValueError):
        return False
    if not isinstance(sample, dict):
        return False
    vertices = sample.get("vertices")
    visibility = sample.get("visibility_backface_and_occlusion")
    image_paths = sample.get("image_paths")
    return (
        isinstance(vertices, torch.Tensor)
        and isinstance(visibility, torch.Tensor)
        and isinstance(image_paths, list)
        and len(image_paths) == count
        and tuple(visibility.shape) == (count, len(vertices))
    )


def _load_base_records(base_manifest: Path, selected_ids: set[str] | None) -> list[dict[str, str]]:
    payload = _read_json(base_manifest)
    records: list[dict[str, str]] = []
    for item in payload.get("samples", []):
        sample_id = str(item.get("sample_id", ""))
        split = str(item.get("split", ""))
        if not sample_id or not split:
            raise ValueError(f"Invalid base manifest record: {item}")
        if selected_ids is not None and sample_id not in selected_ids:
            continue
        records.append({"sample_id": sample_id, "split": split})
    if selected_ids is not None:
        found = {record["sample_id"] for record in records}
        missing = sorted(selected_ids - found)
        if missing:
            raise ValueError("Requested model IDs absent from base manifest: " + ", ".join(missing))
    if not records:
        raise ValueError("No samples selected")
    return records


def _dependencies(downstream_root: Path) -> dict[str, Any]:
    source = downstream_root / "src"
    if not source.is_dir():
        raise FileNotFoundError(f"Downstream source tree not found: {source}")
    sys.path.insert(0, str(source))
    from mlr.coarse_lap_oracle import apply_uniform_laplacian, build_uniform_laplacian_data
    from mlr.data import Camera, Mesh
    from mlr.datasets import load_reconstruction_input
    from mlr.io import load_mesh
    from mlr.learned_laplacian.dataset import save_prepared_sample
    from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset
    from mlr.learned_laplacian.multi_trainer import _prepare_object_static
    from mlr.learned_laplacian.query_training import validate_gt_query_contract
    from mlr.learned_laplacian.renderer_visibility import compute_renderer_visibility
    from mlr.learned_laplacian.sample_io import (
        _attach_target_scaling,
        prepare_gt_query_sample_from_prepared,
        prepare_same_topology_sample,
    )
    from mlr.synthetic import (
        SyntheticRenderConfig,
        _write_cameras_json,
        _write_dataset_json,
        look_at_world_to_camera,
        render_mesh_view,
    )

    return {
        "Camera": Camera,
        "Mesh": Mesh,
        "SyntheticRenderConfig": SyntheticRenderConfig,
        "load_reconstruction_input": load_reconstruction_input,
        "load_mesh": load_mesh,
        "look_at_world_to_camera": look_at_world_to_camera,
        "render_mesh_view": render_mesh_view,
        "write_cameras_json": _write_cameras_json,
        "write_dataset_json": _write_dataset_json,
        "prepare_same_topology_sample": prepare_same_topology_sample,
        "prepare_gt_query_sample_from_prepared": prepare_gt_query_sample_from_prepared,
        "save_prepared_sample": save_prepared_sample,
        "attach_target_scaling": _attach_target_scaling,
        "build_uniform_laplacian_data": build_uniform_laplacian_data,
        "apply_uniform_laplacian": apply_uniform_laplacian,
        "compute_renderer_visibility": compute_renderer_visibility,
        "PreparedMeshDataset": PreparedMeshDataset,
        "validate_gt_query_contract": validate_gt_query_contract,
        "prepare_object_static": _prepare_object_static,
    }


def _validate_manifests(
    gt_manifest_paths: dict[int, Path],
    expanded_manifest_paths: dict[int, Path],
    training_config: dict[str, Any],
    deps: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    inference_config = copy.deepcopy(training_config)
    inference_config.setdefault("query_training", {})["enabled"] = False
    for count in VIEW_COUNTS:
        gt_payload = _read_json(gt_manifest_paths[count])
        splits = sorted({str(item["split"]) for item in gt_payload["samples"]})
        count_result: dict[str, Any] = {}
        for split in splits:
            gt_dataset = deps["PreparedMeshDataset"].from_manifest(
                gt_manifest_paths[count], split
            )
            expanded_dataset = deps["PreparedMeshDataset"].from_manifest(
                expanded_manifest_paths[count], split
            )
            if gt_dataset.sample_ids != expanded_dataset.sample_ids:
                raise ValueError(
                    f"views_{count}/{split}: GT and expanded sample order differs"
                )
            for index, sample_id in enumerate(gt_dataset.sample_ids):
                gt_sample = gt_dataset.load_static(index)
                deps["validate_gt_query_contract"](gt_sample)
                expected_gt = (count, len(gt_sample["vertices"]))
                if tuple(gt_sample["visibility_backface_and_occlusion"].shape) != expected_gt:
                    raise ValueError(f"views_{count}/{sample_id}: GT visibility mismatch")
                deps["prepare_object_static"](
                    gt_sample,
                    training_config,
                    keep_image_payload=True,
                    keep_projection=True,
                )
                expanded_sample = expanded_dataset.load_static(index)
                if expanded_sample.get("metadata", {}).get("training_eligible") is not False:
                    raise ValueError(f"views_{count}/{sample_id}: expanded sample role mismatch")
                expected_expanded = (count, len(expanded_sample["vertices"]))
                if (
                    tuple(expanded_sample["visibility_backface_and_occlusion"].shape)
                    != expected_expanded
                ):
                    raise ValueError(
                        f"views_{count}/{sample_id}: expanded visibility mismatch"
                    )
                deps["prepare_object_static"](
                    expanded_sample,
                    inference_config,
                    keep_image_payload=True,
                    keep_projection=True,
                )
                for value in gt_sample["image_paths"]:
                    path = Path(value)
                    if not path.is_absolute():
                        path = gt_manifest_paths[count].parent / path
                    if not path.is_file():
                        raise FileNotFoundError(
                            f"views_{count}/{sample_id}: missing image {path}"
                        )
            count_result[split] = len(gt_dataset)
        result[str(count)] = count_result
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare strictly nested Sofa50 14/28/56-view GT-query datasets with one strict CPU-reference 56-view master render. "
            "The original 14 camera poses are preserved, but all 56 observations are rerendered together."
        )
    )
    parser.add_argument(
        "--refinement-root",
        type=Path,
        default=Path("~/sofa_mesh/sofa50_refinement"),
    )
    parser.add_argument(
        "--base-multiview-root",
        type=Path,
        default=Path("~/sofa_mesh/sofa50_refinement/multiview_960"),
    )
    parser.add_argument(
        "--downstream-root",
        type=Path,
        default=Path("~/multiview-laplacian-refinement"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("~/sofa_mesh/sofa50_refinement/multiview_nested_14_28_56_cpu_v3"),
    )
    parser.add_argument(
        "--model-id",
        action="append",
        default=[],
        help="Repeat to prepare selected models only. Omit to prepare every sample in the base manifest.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--visibility-backend",
        choices=("cuda", "cpu", "opengl"),
        default="cuda",
    )
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument(
        "--training-config",
        type=Path,
        help=(
            "Config used for direct downstream contract validation. Default: "
            "<downstream-root>/configs/learned_laplacian/"
            "train_sofa50_50mesh_2000epoch_absolute_h2_confidence.json"
        ),
    )
    args = parser.parse_args()

    refinement_root = _expand(args.refinement_root)
    base_root = _expand(args.base_multiview_root)
    downstream_root = _expand(args.downstream_root)
    output_root = _expand(args.output_root)
    base_manifest = base_root / "gt_query_manifest.json"
    if not base_manifest.is_file():
        raise FileNotFoundError(f"Base 14-view manifest not found: {base_manifest}")

    selected_ids = set(args.model_id) if args.model_id else None
    records = _load_base_records(base_manifest, selected_ids)
    deps = _dependencies(downstream_root)
    training_config_path = (
        _expand(args.training_config)
        if args.training_config is not None
        else downstream_root
        / "configs"
        / "learned_laplacian"
        / "train_sofa50_50mesh_2000epoch_absolute_h2_confidence.json"
    )
    if not training_config_path.is_file():
        raise FileNotFoundError(f"Missing downstream training config: {training_config_path}")
    training_config = _read_json(training_config_path)
    output_root.mkdir(parents=True, exist_ok=True)

    gt_manifest_records: dict[int, list[dict[str, str]]] = {
        count: [] for count in VIEW_COUNTS
    }
    expanded_manifest_records: dict[int, list[dict[str, str]]] = {
        count: [] for count in VIEW_COUNTS
    }
    summaries: list[dict[str, Any]] = []
    reference_layout: dict[str, Any] | None = None

    for sample_index, record in enumerate(records, start=1):
        model_id = record["sample_id"]
        split = record["split"]
        print(f"[{sample_index}/{len(records)}] {model_id}", flush=True)
        gt_path = refinement_root / "models" / model_id / "gt_mesh.obj"
        expanded_path = refinement_root / "models" / model_id / "expanded_initial_raw.obj"
        base_dataset_path = base_root / "rendered" / model_id / "dataset.json"
        if not gt_path.is_file():
            raise FileNotFoundError(f"Missing GT mesh: {gt_path}")
        if not base_dataset_path.is_file():
            raise FileNotFoundError(f"Missing base 14-view render: {base_dataset_path}")
        if not expanded_path.is_file():
            raise FileNotFoundError(f"Missing expanded inference mesh: {expanded_path}")

        model_render_root = output_root / "rendered" / model_id
        master_dir = model_render_root / "views_56"
        render_existing = {
            count: (master_dir if count == MASTER_VIEW_COUNT else model_render_root / f"views_{count}") / "dataset.json"
            for count in VIEW_COUNTS
        }
        render_complete = (
            not args.force
            and all(path.is_file() for path in render_existing.values())
            and (master_dir / "nested_camera_layout.json").is_file()
        )
        if render_complete:
            print("  existing nested render complete; reusing", flush=True)
            reconstruction_56 = deps["load_reconstruction_input"](master_dir / "dataset.json")
            all_cameras = list(reconstruction_56.cameras)
            nesting = _read_json(master_dir / "nested_camera_layout.json")
            actual_backend = str(_read_json(master_dir / "dataset.json")["config"]["backend"])
            if actual_backend != "cpu":
                raise ValueError(f"{model_id}: existing master backend is {actual_backend}, expected cpu")
            check_path = master_dir / "base14_rerender_check.json"
            base14_rerender_check = _read_json(check_path) if check_path.is_file() else None
        else:
            all_cameras, nesting, actual_backend, base14_rerender_check = _render_master_56(
                model_id,
                gt_path,
                base_dataset_path,
                master_dir,
                deps,
                force=args.force,
            )
        if reference_layout is None:
            reference_layout = nesting
            _write_json(output_root / "nested_camera_layout.json", nesting)
        else:
            ref_centers = np.asarray(reference_layout["centers_56"], dtype=np.float64)
            cur_centers = np.asarray(nesting["centers_56"], dtype=np.float64)
            if not np.allclose(ref_centers, cur_centers, rtol=0.0, atol=1e-10):
                raise ValueError(f"{model_id}: camera layout differs from the first sample")

        gt_prepared_paths = {
            count: output_root / "prepared_gt_query" / f"views_{count}" / f"{model_id}.pt"
            for count in VIEW_COUNTS
        }
        expanded_prepared_paths = {
            count: output_root
            / "prepared_expanded_inference"
            / f"views_{count}"
            / f"{model_id}.pt"
            for count in VIEW_COUNTS
        }
        gt_visibility_paths = {
            count: output_root
            / "renderer_visibility"
            / "gt_query"
            / f"views_{count}"
            / f"{model_id}.npz"
            for count in VIEW_COUNTS
        }
        expanded_visibility_paths = {
            count: output_root
            / "renderer_visibility"
            / "expanded_inference"
            / f"views_{count}"
            / f"{model_id}.npz"
            for count in VIEW_COUNTS
        }
        needs_prepared_contract = args.force or any(
            not _prepared_contract_complete(
                gt_prepared_paths[count], gt_visibility_paths[count], count
            )
            or not _prepared_contract_complete(
                expanded_prepared_paths[count], expanded_visibility_paths[count], count
            )
            for count in VIEW_COUNTS
        )
        gt_visibility_result = None
        expanded_visibility_result = None
        if needs_prepared_contract:
            print("  computing renderer visibility on GT and expanded graphs", flush=True)
            gt_visibility_result = _compute_master_visibility(
                deps["load_mesh"](gt_path).ensure_normals(),
                all_cameras,
                args.visibility_backend,
                deps,
            )
            expanded_visibility_result = _compute_master_visibility(
                deps["load_mesh"](expanded_path).ensure_normals(),
                all_cameras,
                args.visibility_backend,
                deps,
            )

        dataset_paths: dict[int, Path] = {}
        for count in VIEW_COUNTS:
            subset_dir = master_dir if count == MASTER_VIEW_COUNT else model_render_root / f"views_{count}"
            dataset_path = _materialize_subset_dataset(
                master_dir,
                subset_dir,
                count,
                all_cameras,
                gt_path,
                deps,
                actual_backend=actual_backend,
                force=args.force,
            )
            dataset_paths[count] = dataset_path
            gt_prepared_path = gt_prepared_paths[count]
            expanded_prepared_path = expanded_prepared_paths[count]
            gt_complete = _prepared_contract_complete(
                gt_prepared_path, gt_visibility_paths[count], count
            )
            expanded_complete = _prepared_contract_complete(
                expanded_prepared_path, expanded_visibility_paths[count], count
            )
            if args.force or not gt_complete or not expanded_complete:
                if gt_visibility_result is None or expanded_visibility_result is None:
                    raise AssertionError("visibility results were not prepared")
                source = _prepared_source(
                    model_id,
                    split,
                    count,
                    dataset_path,
                    gt_path,
                    output_root,
                    deps,
                )
            if args.force or not gt_complete:
                gt_prepared_path.parent.mkdir(parents=True, exist_ok=True)
                _prepare_gt_query_sample(
                    source,
                    model_id,
                    count,
                    gt_prepared_path,
                    gt_visibility_result,
                    gt_visibility_paths[count],
                    args.visibility_backend,
                    deps,
                )
            if args.force or not expanded_complete:
                expanded_prepared_path.parent.mkdir(parents=True, exist_ok=True)
                _prepare_expanded_inference_sample(
                    source,
                    model_id,
                    split,
                    count,
                    expanded_path,
                    gt_path,
                    expanded_prepared_path,
                    expanded_visibility_result,
                    expanded_visibility_paths[count],
                    args.visibility_backend,
                    deps,
                )
            gt_manifest_records[count].append(
                {
                    "sample_id": model_id,
                    "split": split,
                    "path": gt_prepared_path.relative_to(output_root).as_posix(),
                }
            )
            expanded_manifest_records[count].append(
                {
                    "sample_id": model_id,
                    "split": split,
                    "path": expanded_prepared_path.relative_to(output_root).as_posix(),
                }
            )

        summaries.append(
            {
                "sample_id": model_id,
                "split": split,
                "gt_mesh": str(gt_path),
                "expanded_mesh": str(expanded_path),
                "base_14_dataset": str(base_dataset_path),
                "dataset_paths": {str(k): str(v) for k, v in dataset_paths.items()},
                "actual_backend_for_all_56": actual_backend,
                "renderer_fallback_allowed": False,
                "base14_rerender_check": base14_rerender_check,
                "camera_layout_version": LAYOUT_VERSION,
                "minimum_angular_separation_degrees": nesting[
                    "minimum_angular_separation_degrees"
                ],
                "reused_exact_base_camera_indices": list(range(BASE_VIEW_COUNT)),
                "all_views_rerendered": list(range(MASTER_VIEW_COUNT)),
            }
        )

    gt_manifest_paths: dict[int, Path] = {}
    expanded_manifest_paths: dict[int, Path] = {}
    for count in VIEW_COUNTS:
        gt_manifest_path = output_root / f"gt_query_views_{count}_manifest.json"
        expanded_manifest_path = (
            output_root / f"expanded_inference_views_{count}_manifest.json"
        )
        _write_json(
            gt_manifest_path,
            _manifest(gt_manifest_records[count], count, expanded=False),
        )
        _write_json(
            expanded_manifest_path,
            _manifest(expanded_manifest_records[count], count, expanded=True),
        )
        gt_manifest_paths[count] = gt_manifest_path
        expanded_manifest_paths[count] = expanded_manifest_path

    validation = None
    if not args.skip_validation:
        print("Validating manifests with downstream training and inference loaders...", flush=True)
        validation = _validate_manifests(
            gt_manifest_paths,
            expanded_manifest_paths,
            training_config,
            deps,
        )

    summary = {
        "format_version": "sofa50_nested_views_summary_v4",
        "camera_layout_version": LAYOUT_VERSION,
        "counts": list(VIEW_COUNTS),
        "nesting": "views_14 is prefix of views_28, which is prefix of views_56",
        "base_14_camera_poses_reused_exactly": True,
        "base_14_observations_reused_exactly": False,
        "all_56_observations_rerendered_with_strict_cpu_reference": True,
        "renderer_fallback_allowed": False,
        "views_rendered_per_model": MASTER_VIEW_COUNT,
        "same_gt_graph_target_across_view_groups": True,
        "target_mode": TARGET_MODE,
        "renderer_visibility_recompute_required": False,
        "renderer_visibility": {
            "backend": args.visibility_backend,
            "front_face_winding": "ccw",
            "neighborhood_radius": 1,
            "training_config": str(training_config_path),
        },
        "base_manifest": str(base_manifest),
        "output_root": str(output_root),
        "gt_query_manifests": {
            str(count): str(path) for count, path in gt_manifest_paths.items()
        },
        "expanded_inference_manifests": {
            str(count): str(path) for count, path in expanded_manifest_paths.items()
        },
        "downstream_contract_validation": validation,
        "samples": summaries,
    }
    _write_json(output_root / "summary.json", summary)
    print(json.dumps({
        "status": "passed",
        "sample_count": len(records),
        "output_root": str(output_root),
        "gt_query_manifests": {
            str(count): str(path) for count, path in gt_manifest_paths.items()
        },
        "expanded_inference_manifests": {
            str(count): str(path) for count, path in expanded_manifest_paths.items()
        },
        "summary": str(output_root / "summary.json"),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
