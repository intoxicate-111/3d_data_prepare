from __future__ import annotations

import copy
import gc
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


RENDER_SPEC = {
    "image_size": 960,
    "views": 14,
    "trajectory": "cube_surface",
    "fov_degrees": 90.0,
    "cube_half_extent": 1.5,
    "camera_layout_version": "unit_sphere_cube_surface_faces6_corners8_v1",
    "camera_convention": "right-handed CV world-to-camera, +Z forward, +X right, +Y down",
}


def prepare_multiview_dataset(
    refinement_root: str | Path,
    downstream_root: str | Path,
    output_root: str | Path | None = None,
    *,
    backend: str = "opengl",
    force: bool = False,
    expected_count: int = 2,
) -> dict[str, Any]:
    """Render and validate a two-sample trial or the complete Sofa50 dataset.

    Rendering and prepared-sample construction deliberately use the downstream
    repository implementation. Geometry files below ``refinement_root/models`` are only
    read, and every rendered observation is produced from ``gt_mesh.obj``.
    """

    refinement_root = Path(refinement_root).expanduser().resolve()
    downstream_root = Path(downstream_root).expanduser().resolve()
    output_root = Path(output_root or refinement_root / "multiview_960").expanduser().resolve()
    deps = _downstream_dependencies(downstream_root)
    source_manifest_path = refinement_root / "manifest.json"
    source_manifest = _read_json(source_manifest_path)
    samples = [item for item in source_manifest.get("samples", []) if item.get("status") == "valid"]
    if expected_count not in {2, 50}:
        raise ValueError("expected_count must be 2 for trial or 50 for full generation")
    if len(samples) != expected_count:
        raise ValueError(
            f"Expected exactly {expected_count} valid Sofa50 samples, found {len(samples)}."
        )
    if backend not in {"cpu", "opengl", "cuda"}:
        raise ValueError("backend must be cpu, opengl, or cuda")

    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    geometry_hashes_before = _geometry_hashes(samples)
    for index, item in enumerate(samples, start=1):
        model_id = str(item["model_id"])
        split = str(item["split"])
        print(
            f"[{index}/{expected_count}] preparing formal GT renders for {model_id}",
            flush=True,
        )
        records.append(
            _prepare_one(
                item,
                split,
                refinement_root,
                output_root,
                deps,
                backend=backend,
                force=force,
            )
        )
    geometry_hashes_after = _geometry_hashes(samples)
    if geometry_hashes_before != geometry_hashes_after:
        raise RuntimeError("A GT or expanded geometry file changed during rendering.")

    gt_manifest = _prepared_manifest(
        records,
        output_root,
        role="gt_query_training",
        prepared_key="gt_prepared_path",
        format_version=(
            "sofa50_gt_query_trial_manifest_v1"
            if expected_count == 2
            else "sofa50_gt_query_manifest_v1"
        ),
        training_eligible=True,
    )
    expanded_manifest = _prepared_manifest(
        records,
        output_root,
        role="expanded_raw_frozen_model_inference",
        prepared_key="expanded_prepared_path",
        format_version=(
            "sofa50_expanded_inference_trial_manifest_v1"
            if expected_count == 2
            else "sofa50_expanded_inference_manifest_v1"
        ),
        training_eligible=False,
    )
    gt_manifest_path = output_root / "gt_query_manifest.json"
    expanded_manifest_path = output_root / "expanded_inference_manifest.json"
    _write_json(gt_manifest_path, gt_manifest)
    _write_json(expanded_manifest_path, expanded_manifest)

    loader_audit = _validate_downstream_loaders(
        gt_manifest_path, expanded_manifest_path, downstream_root, deps
    )
    audit = {
        "status": "passed",
        "scope": "two_sample_trial" if expected_count == 2 else "full_sofa50",
        "sample_count": expected_count,
        "source_refinement_manifest": str(source_manifest_path),
        "downstream_repository": str(downstream_root),
        "downstream_git_branch": _git_branch(downstream_root),
        "render_spec": RENDER_SPEC,
        "geometry_unchanged": True,
        "geometry_sha256": geometry_hashes_after,
        "loader_validation": loader_audit,
        "samples": records,
    }
    audit_path = output_root / "FIELD_SHAPE_AUDIT.json"
    _write_json(audit_path, audit)
    observations_manifest = {
        "format_version": "sofa50_multiview_observations_v1",
        "scope": "two_sample_trial" if expected_count == 2 else "full_sofa50",
        "render_source": "gt_mesh.obj",
        "geometry_modified": False,
        "render_spec": RENDER_SPEC,
        "gt_query_manifest": str(gt_manifest_path),
        "expanded_inference_manifest": str(expanded_manifest_path),
        "field_shape_audit": str(audit_path),
        "samples": records,
    }
    observations_path = output_root / "observations_manifest.json"
    _write_json(observations_path, observations_manifest)
    report_path = output_root / "REPORT.md"
    report_path.write_text(_report(audit, observations_path), encoding="utf-8")
    result = {
        "status": "passed",
        "sample_count": expected_count,
        "output_root": str(output_root),
        "observations_manifest": str(observations_path),
        "gt_query_manifest": str(gt_manifest_path),
        "expanded_inference_manifest": str(expanded_manifest_path),
        "field_shape_audit": str(audit_path),
        "report": str(report_path),
        "full_50_completed": expected_count == 50,
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


def _prepare_one(
    item: dict[str, Any],
    split: str,
    refinement_root: Path,
    output_root: Path,
    deps: dict[str, Any],
    *,
    backend: str,
    force: bool,
) -> dict[str, Any]:
    model_id = str(item["model_id"])
    model_dir = refinement_root / "models" / model_id
    gt_path = model_dir / "gt_mesh.obj"
    expanded_path = model_dir / "expanded_initial_raw.obj"
    if not gt_path.is_file() or not expanded_path.is_file():
        raise FileNotFoundError(f"Missing refinement geometry for {model_id}")
    render_dir = output_root / "rendered" / model_id
    gt_prepared_path = output_root / "prepared_gt_query" / f"{model_id}.pt"
    expanded_prepared_path = output_root / "prepared_expanded_inference" / f"{model_id}.pt"
    visibility_path = render_dir / "visibility.npz"
    required = (
        render_dir / "dataset.json",
        render_dir / "cameras.json",
        gt_prepared_path,
        expanded_prepared_path,
        visibility_path,
    )

    if force or not all(path.is_file() for path in required):
        rendered = deps["generate_synthetic_dataset_from_mesh"](
            gt_path,
            render_dir,
            config=deps["SyntheticRenderConfig"](
                num_views=RENDER_SPEC["views"],
                width=RENDER_SPEC["image_size"],
                height=RENDER_SPEC["image_size"],
                trajectory=RENDER_SPEC["trajectory"],
                fov_degrees=RENDER_SPEC["fov_degrees"],
                render_mode="lit",
                backend=backend,
                normalize_mesh=False,
                cube_half_extent=RENDER_SPEC["cube_half_extent"],
                antialiasing="msaa4",
            ),
        )
        gt_mesh = deps["load_mesh"](gt_path).ensure_normals()
        expanded_mesh = deps["load_mesh"](expanded_path).ensure_normals()
        reconstruction = deps["load_reconstruction_input"](rendered.dataset_path)
        masks = deps["load_masks"](reconstruction.mask_paths)
        if masks is None:
            raise RuntimeError("The downstream render dataset did not produce masks.")
        gt_mask_visibility = deps["mask_visibility"](
            gt_mesh.vertices, reconstruction.cameras, masks, (1.0, 1.0)
        )
        expanded_mask_visibility = deps["mask_visibility"](
            expanded_mesh.vertices, reconstruction.cameras, masks, (1.0, 1.0)
        )
        depth_paths = [render_dir / "depth" / f"{view:04d}.npy" for view in range(14)]
        depth_maps = [np.load(path) for path in depth_paths]
        depth_tolerance = 0.01 * float(
            np.linalg.norm(gt_mesh.vertices.max(axis=0) - gt_mesh.vertices.min(axis=0))
        )
        gt_depth_visibility = _depth_visibility(
            gt_mesh.vertices, reconstruction.cameras, masks, depth_maps, depth_tolerance
        )
        expanded_depth_visibility = _depth_visibility(
            expanded_mesh.vertices,
            reconstruction.cameras,
            masks,
            depth_maps,
            depth_tolerance,
        )
        np.savez_compressed(
            visibility_path,
            gt_mask_support=gt_mask_visibility,
            gt_depth_visible=gt_depth_visibility,
            expanded_mask_support=expanded_mask_visibility,
            expanded_depth_visible=expanded_depth_visibility,
            depth_tolerance=np.asarray(depth_tolerance, dtype=np.float64),
        )

        relative_images = [
            path.resolve().relative_to(output_root).as_posix()
            for path in rendered.image_paths
        ]
        source = deps["prepare_same_topology_sample"](
            rendered.dataset_path,
            gt_path,
            gt_path,
            image_size=RENDER_SPEC["image_size"],
            target_mode="edge_scale_normalized_laplacian",
            extra_metadata={
                "dataset_family": "sofa50",
                "source_sample_id": model_id,
                "source_split": split,
                "render_geometry_role": "gt_mesh",
                "render_source_path": str(gt_path),
            },
        )
        source["sample_id"] = model_id
        source.pop("images", None)
        source["image_paths"] = relative_images
        source["prepared_storage_format"] = "lazy_image_paths_v1"
        source["source_image_size"] = [960, 960]
        source["prepared_image_size"] = 960
        gt_sample = deps["prepare_gt_query_sample_from_prepared"](
            source, target_mode="edge_scale_normalized_laplacian"
        )
        gt_sample["metadata"].update(
            {
                "visibility_policy": "projection_validity_recomputed_for_perturbed_gt_queries",
                "visibility_artifact": str(visibility_path),
                "training_eligible": True,
            }
        )
        deps["save_prepared_sample"](gt_sample, gt_prepared_path)

        expanded_sample = _expanded_inference_sample(
            source,
            expanded_mesh,
            gt_mesh,
            expanded_mask_visibility,
            model_id,
            split,
            expanded_path,
            gt_path,
            visibility_path,
            deps,
        )
        deps["save_prepared_sample"](expanded_sample, expanded_prepared_path)

    return _audit_one(
        item,
        split,
        output_root,
        render_dir,
        gt_path,
        expanded_path,
        gt_prepared_path,
        expanded_prepared_path,
        visibility_path,
        deps,
    )


def _expanded_inference_sample(
    source: dict[str, Any],
    expanded_mesh: Any,
    gt_mesh: Any,
    visibility: np.ndarray,
    model_id: str,
    split: str,
    expanded_path: Path,
    gt_path: Path,
    visibility_path: Path,
    deps: dict[str, Any],
) -> dict[str, Any]:
    vertices = torch.as_tensor(expanded_mesh.vertices, dtype=torch.float32)
    faces = torch.as_tensor(expanded_mesh.faces, dtype=torch.long)
    lap_data = deps["build_uniform_laplacian_data"](
        expanded_mesh.faces, expanded_mesh.num_vertices
    )
    initial = torch.as_tensor(
        deps["apply_uniform_laplacian"](expanded_mesh.vertices, lap_data),
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
        # The generic downstream loader requires target-shaped fields. They are
        # deliberately an identity/no-op reconstruction reference, not GT oracle
        # supervision, and this manifest is explicitly inference-only.
        "laplacian_target": initial.clone(),
        "target_confidence": torch.ones(expanded_mesh.num_vertices, dtype=torch.float32),
        "visibility": torch.as_tensor(visibility, dtype=torch.bool),
        "gt_vertices": torch.as_tensor(gt_mesh.vertices, dtype=torch.float32),
        "gt_faces": torch.as_tensor(gt_mesh.faces, dtype=torch.long),
        "position_normalization_center": center,
        "position_normalization_scale": scale,
        "metadata": {
            "dataset_family": "sofa50",
            "source_sample_id": model_id,
            "source_split": split,
            "usage_role": "frozen_model_inference_and_reconstruction_evaluation_only",
            "training_eligible": False,
            "query_geometry_role": "expanded_initial_raw",
            "query_mesh_path": str(expanded_path),
            "gt_evaluation_mesh_path": str(gt_path),
            "render_geometry_role": "gt_mesh",
            "camera_convention": RENDER_SPEC["camera_convention"],
            "visibility_policy": "downstream_mask_support_on_expanded_query_vertices",
            "visibility_artifact": str(visibility_path),
            "target_field_role": "schema_required_identity_placeholder_not_training_supervision",
            "target_constructor": "uniform_laplacian_of_expanded_initial_raw_identity_reference",
            "gt_oracle_target_used": False,
            "P_target_oracle_used": False,
        },
    }
    deps["attach_target_scaling"](
        sample,
        "edge_scale_normalized_laplacian",
        1e-12,
        edge_scale_source="expanded_initial_raw_graph_identity_reference",
    )
    return sample


def _depth_visibility(
    vertices: np.ndarray,
    cameras: list[Any],
    masks: list[np.ndarray],
    depths: list[np.ndarray],
    tolerance: float,
) -> np.ndarray:
    result = np.zeros((len(cameras), len(vertices)), dtype=bool)
    for view, (camera, mask, depth_map) in enumerate(zip(cameras, masks, depths, strict=True)):
        pixels, vertex_depth = camera.project(vertices)
        x = np.rint(pixels[:, 0]).astype(np.int64)
        y = np.rint(pixels[:, 1]).astype(np.int64)
        valid = (
            (vertex_depth > 1e-8)
            & (x >= 0)
            & (x < mask.shape[1])
            & (y >= 0)
            & (y < mask.shape[0])
        )
        indices = np.flatnonzero(valid)
        sampled_depth = depth_map[y[indices], x[indices]]
        result[view, indices] = (
            mask[y[indices], x[indices]]
            & np.isfinite(sampled_depth)
            & (np.abs(sampled_depth - vertex_depth[indices]) <= tolerance)
        )
    return result


def _audit_one(
    item: dict[str, Any],
    split: str,
    output_root: Path,
    render_dir: Path,
    gt_path: Path,
    expanded_path: Path,
    gt_prepared_path: Path,
    expanded_prepared_path: Path,
    visibility_path: Path,
    deps: dict[str, Any],
) -> dict[str, Any]:
    model_id = str(item["model_id"])
    dataset = deps["load_reconstruction_input"](render_dir / "dataset.json")
    gt_mesh = deps["load_mesh"](gt_path).ensure_normals()
    expanded_mesh = deps["load_mesh"](expanded_path).ensure_normals()
    if len(dataset.image_paths) != 14 or len(dataset.cameras) != 14:
        raise ValueError(f"{model_id}: downstream dataset does not contain 14 views")
    rgb_shapes, mask_shapes, depth_shapes = [], [], []
    finite_depth_inside = []
    background_depth_inf = []
    for view in range(14):
        with Image.open(dataset.image_paths[view]) as image:
            rgb_shapes.append([image.height, image.width, len(image.getbands())])
        mask_path = render_dir / "masks" / f"{view:04d}.png"
        depth_path = render_dir / "depth" / f"{view:04d}.npy"
        with Image.open(mask_path) as image:
            mask = np.asarray(image.convert("L")) > 0
            mask_shapes.append(list(mask.shape))
        depth = np.load(depth_path)
        depth_shapes.append(list(depth.shape))
        finite_depth_inside.append(bool(np.isfinite(depth[mask]).all() and mask.any()))
        background_depth_inf.append(bool(np.isinf(depth[~mask]).all()))
    if set(map(tuple, rgb_shapes)) != {(960, 960, 3)}:
        raise ValueError(f"{model_id}: RGB output is not uniformly [960,960,3]")
    if set(map(tuple, mask_shapes)) != {(960, 960)} or set(map(tuple, depth_shapes)) != {
        (960, 960)
    }:
        raise ValueError(f"{model_id}: mask/depth output shape mismatch")
    if not all(finite_depth_inside) or not all(background_depth_inf):
        raise ValueError(f"{model_id}: depth/mask finite-value contract failed")

    intrinsics = np.stack([camera.intrinsics for camera in dataset.cameras])
    extrinsics = np.stack([_extrinsic(camera) for camera in dataset.cameras])
    projection_checks = {}
    for role, mesh in (("gt", gt_mesh), ("expanded_initial_raw", expanded_mesh)):
        torch_projection = deps["project_vertices"](
            torch.as_tensor(mesh.vertices, dtype=torch.float32),
            torch.as_tensor(intrinsics, dtype=torch.float32),
            torch.as_tensor(extrinsics, dtype=torch.float32),
            (960, 960),
        )
        numpy_pixels = np.stack([camera.project(mesh.vertices)[0] for camera in dataset.cameras])
        max_error = float(
            np.max(np.abs(torch_projection.pixels.detach().cpu().numpy() - numpy_pixels))
        )
        if max_error > 2e-3:
            raise ValueError(f"{model_id}: torch/numpy projection mismatch {max_error}")
        projection_checks[role] = {
            "vertices": int(mesh.num_vertices),
            "faces": int(mesh.num_faces),
            "pixels_shape": list(torch_projection.pixels.shape),
            "depth_shape": list(torch_projection.depth.shape),
            "in_frame_fraction": float(torch_projection.valid.float().mean().item()),
            "numpy_torch_max_pixel_error": max_error,
        }
    with np.load(visibility_path) as visibility:
        visibility_shapes = {name: list(visibility[name].shape) for name in visibility.files}
        visibility_fractions = {
            name: float(np.mean(visibility[name]))
            for name in visibility.files
            if visibility[name].ndim == 2
        }
    expected_visibility = {
        "gt_mask_support": [14, gt_mesh.num_vertices],
        "gt_depth_visible": [14, gt_mesh.num_vertices],
        "expanded_mask_support": [14, expanded_mesh.num_vertices],
        "expanded_depth_visible": [14, expanded_mesh.num_vertices],
    }
    for name, shape in expected_visibility.items():
        if visibility_shapes.get(name) != shape:
            raise ValueError(f"{model_id}: {name} shape mismatch")

    render_payload = _read_json(render_dir / "dataset.json")
    if render_payload.get("source_mesh_path") != str(gt_path):
        raise ValueError(f"{model_id}: render source is not gt_mesh.obj")
    return {
        "model_id": model_id,
        "split": split,
        "render_source_role": "gt_mesh",
        "gt_mesh_path": str(gt_path),
        "expanded_initial_raw_path": str(expanded_path),
        "render_dataset_path": str(render_dir / "dataset.json"),
        "cameras_path": str(render_dir / "cameras.json"),
        "rgb_glob": str(render_dir / "images" / "*.png"),
        "mask_glob": str(render_dir / "masks" / "*.png"),
        "depth_glob": str(render_dir / "depth" / "*.npy"),
        "visibility_path": str(visibility_path),
        "gt_prepared_path": str(gt_prepared_path),
        "expanded_prepared_path": str(expanded_prepared_path),
        "view_names": [camera.name for camera in dataset.cameras],
        "rgb_shape_per_view": [960, 960, 3],
        "mask_shape_per_view": [960, 960],
        "depth_shape_per_view": [960, 960],
        "intrinsics_shape": list(intrinsics.shape),
        "extrinsics_shape": list(extrinsics.shape),
        "visibility_shapes": visibility_shapes,
        "visibility_fractions": visibility_fractions,
        "projection": projection_checks,
        "render_backend_actual": render_payload["config"]["backend"],
        "camera_convention": RENDER_SPEC["camera_convention"],
        "checks_passed": True,
    }


def _validate_downstream_loaders(
    gt_manifest_path: Path,
    expanded_manifest_path: Path,
    downstream_root: Path,
    deps: dict[str, Any],
) -> dict[str, Any]:
    config = _read_json(
        downstream_root / "configs" / "learned_laplacian" / "train_gt_query_50_960.json"
    )
    inference_config = copy.deepcopy(config)
    inference_config.setdefault("query_training", {})["enabled"] = False
    model = deps["build_model"](inference_config, None, True).eval()
    records = []
    manifest_items = _read_json(gt_manifest_path)["samples"]
    present_splits = [
        split
        for split in ("train", "validation", "test")
        if any(item["split"] == split for item in manifest_items)
    ]
    for split in present_splits:
        gt_dataset = deps["PreparedMeshDataset"].from_manifest(gt_manifest_path, split)
        expanded_dataset = deps["PreparedMeshDataset"].from_manifest(
            expanded_manifest_path, split
        )
        if gt_dataset.sample_ids != expanded_dataset.sample_ids:
            raise ValueError(
                f"GT and expanded manifests do not preserve model-ID order in {split}."
            )
        for index, model_id in enumerate(gt_dataset.sample_ids):
            gt_static = gt_dataset.load_static(index)
            deps["validate_gt_query_contract"](gt_static)
            gt_prepared = deps["prepare_object_static"](
                gt_static, config, keep_image_payload=True, keep_projection=True
            )
            gt_loaded = deps["prepare_item_for_use"](
                gt_prepared,
                config,
                torch.device("cpu"),
                cache_on_device=False,
                decode_images=True,
            )
            gt_image_shape = list(gt_loaded.sample["images"].shape)
            if tuple(gt_image_shape) != (14, 3, 960, 960):
                raise ValueError(f"{model_id}: GT training loader image tensor shape mismatch")
            gt_vertices_shape = list(gt_static["vertices"].shape)
            gt_faces_shape = list(gt_static["faces"].shape)
            del gt_loaded, gt_prepared
            gc.collect()

            expanded_static = expanded_dataset.load_static(index)
            if expanded_static.get("metadata", {}).get("training_eligible") is not False:
                raise ValueError(f"{model_id}: expanded sample is not marked inference-only")
            expanded_prepared = deps["prepare_object_static"](
                expanded_static,
                inference_config,
                keep_image_payload=True,
                keep_projection=True,
            )
            expanded_loaded = deps["prepare_item_for_use"](
                expanded_prepared,
                inference_config,
                torch.device("cpu"),
                cache_on_device=False,
                decode_images=True,
            )
            expanded_image_shape = list(expanded_loaded.sample["images"].shape)
            if tuple(expanded_image_shape) != (14, 3, 960, 960):
                raise ValueError(f"{model_id}: expanded inference loader image shape mismatch")
            with torch.no_grad():
                prediction = model(expanded_loaded.sample).predicted_laplacian
            if tuple(prediction.shape) != tuple(expanded_static["vertices"].shape):
                raise ValueError(f"{model_id}: expanded inference prediction shape mismatch")
            if not torch.isfinite(prediction).all():
                raise ValueError(f"{model_id}: expanded inference produced non-finite output")
            records.append(
                {
                    "model_id": model_id,
                    "split": split,
                    "gt_training_static_load": "passed",
                    "gt_query_contract": "passed",
                    "gt_training_decoded_images_shape": gt_image_shape,
                    "gt_vertices_shape": gt_vertices_shape,
                    "gt_faces_shape": gt_faces_shape,
                    "expanded_inference_static_load": "passed",
                    "expanded_inference_decoded_images_shape": expanded_image_shape,
                    "expanded_vertices_shape": list(expanded_static["vertices"].shape),
                    "expanded_faces_shape": list(expanded_static["faces"].shape),
                    "expanded_zero_image_model_forward": "passed",
                    "prediction_shape": list(prediction.shape),
                }
            )
            del expanded_loaded, expanded_prepared, prediction
            gc.collect()
    return {
        "status": "passed",
        "config_path": str(
            downstream_root
            / "configs"
            / "learned_laplacian"
            / "train_gt_query_50_960.json"
        ),
        "gt_query_training_loader": "PreparedMeshDataset + actual multi_trainer preparation",
        "expanded_query_inference_loader": (
            "PreparedMeshDataset + actual multi_trainer preparation with query_training disabled"
        ),
        "expanded_forward_check": "untrained architecture, zero RGB features; interface/shape only",
        "records": records,
    }


def _prepared_manifest(
    records: list[dict[str, Any]],
    output_root: Path,
    *,
    role: str,
    prepared_key: str,
    format_version: str,
    training_eligible: bool,
) -> dict[str, Any]:
    return {
        "format_version": format_version,
        "dataset_role": role,
        "training_eligible": training_eligible,
        "render_source": "gt_mesh.obj",
        "render_spec": RENDER_SPEC,
        "samples": [
            {
                "sample_id": record["model_id"],
                "path": Path(record[prepared_key]).relative_to(output_root).as_posix(),
                "split": record["split"],
            }
            for record in records
        ],
    }


def _downstream_dependencies(root: Path) -> dict[str, Any]:
    source = root / "src"
    if not source.is_dir():
        raise FileNotFoundError(f"Downstream source tree not found: {source}")
    sys.path.insert(0, str(source))
    from mlr.coarse_lap_oracle import apply_uniform_laplacian, build_uniform_laplacian_data
    from mlr.datasets import load_masks, load_reconstruction_input
    from mlr.io import load_mesh
    from mlr.learned_laplacian.dataset import save_prepared_sample
    from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset
    from mlr.learned_laplacian.multi_trainer import (
        _build_model,
        _prepare_item_for_use,
        _prepare_object_static,
    )
    from mlr.learned_laplacian.projection import project_vertices
    from mlr.learned_laplacian.query_training import validate_gt_query_contract
    from mlr.learned_laplacian.sample_io import (
        _attach_target_scaling,
        _mask_visibility,
        prepare_gt_query_sample_from_prepared,
        prepare_same_topology_sample,
    )
    from mlr.synthetic import (
        SyntheticRenderConfig,
        generate_synthetic_dataset_from_mesh,
    )

    return {
        "SyntheticRenderConfig": SyntheticRenderConfig,
        "generate_synthetic_dataset_from_mesh": generate_synthetic_dataset_from_mesh,
        "load_mesh": load_mesh,
        "load_masks": load_masks,
        "load_reconstruction_input": load_reconstruction_input,
        "prepare_same_topology_sample": prepare_same_topology_sample,
        "prepare_gt_query_sample_from_prepared": prepare_gt_query_sample_from_prepared,
        "save_prepared_sample": save_prepared_sample,
        "mask_visibility": _mask_visibility,
        "attach_target_scaling": _attach_target_scaling,
        "build_uniform_laplacian_data": build_uniform_laplacian_data,
        "apply_uniform_laplacian": apply_uniform_laplacian,
        "project_vertices": project_vertices,
        "PreparedMeshDataset": PreparedMeshDataset,
        "validate_gt_query_contract": validate_gt_query_contract,
        "prepare_object_static": _prepare_object_static,
        "prepare_item_for_use": _prepare_item_for_use,
        "build_model": _build_model,
    }


def _extrinsic(camera: Any) -> np.ndarray:
    value = np.eye(4, dtype=np.float64)
    value[:3, :3] = camera.rotation
    value[:3, 3] = camera.translation
    return value


def _geometry_hashes(samples: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    result = {}
    for item in samples:
        model_id = str(item["model_id"])
        result[model_id] = {
            "gt_mesh_obj": _file_sha256(Path(item["gt_obj"])),
            "gt_mesh_npz": _file_sha256(Path(item["gt_npz"])),
            "expanded_initial_raw_obj": _file_sha256(Path(item["expanded_initial_raw_obj"])),
            "expanded_initial_raw_npz": _file_sha256(Path(item["expanded_initial_raw_npz"])),
        }
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git_branch(root: Path) -> str:
    head = (root / ".git" / "HEAD").read_text(encoding="utf-8").strip()
    return head.rsplit("/", 1)[-1] if head.startswith("ref:") else head


def _report(audit: dict[str, Any], observations_path: Path) -> str:
    sample_count = int(audit["sample_count"])
    lines = [
        f"# Sofa50 final {'trial' if sample_count == 2 else 'full'} multiview audit",
        "",
        f"Status: **passed** for {sample_count} Sofa50 samples.",
        "",
        f"- Observations manifest: `{observations_path}`",
        f"- Downstream repository: `{audit['downstream_repository']}` on `{audit['downstream_git_branch']}`",
        "- Render source: `gt_mesh.obj` only; GT/coarse/expanded geometry was not modified.",
        "- RGB: 14 x `[960, 960, 3]` PNG; mask: 14 x `[960, 960]` PNG; depth: 14 x `[960, 960]` NPY.",
        "- Cameras: intrinsics `[14,3,3]`, extrinsics `[14,4,4]`, right-handed CV world-to-camera.",
        "- GT-query manifest is training eligible and uses direct GT-graph supervision.",
        "- Expanded manifest is inference-only; its schema-required target is an identity reference and is not GT/oracle supervision.",
        "- Saved visibility includes downstream mask support and depth-consistent visibility for GT and expanded query vertices.",
        "",
        "## Samples",
        "",
    ]
    for sample in audit["samples"]:
        lines.extend(
            [
                f"- `{sample['model_id']}`: GT {sample['projection']['gt']['vertices']} vertices; "
                f"expanded {sample['projection']['expanded_initial_raw']['vertices']} vertices; "
                f"projection max error {max(sample['projection']['gt']['numpy_torch_max_pixel_error'], sample['projection']['expanded_initial_raw']['numpy_torch_max_pixel_error']):.6g}px.",
            ]
        )
    return "\n".join(lines) + "\n"
