from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


VARIANT_COUNT = 5
VIEW_COUNT = 14
TARGET_MODE = "edge_scale_normalized_laplacian"
FORMAT_VERSION = "sofa50_synthetic_current_v1"


def _dependencies(downstream_root: Path) -> dict[str, Any]:
    source_root = downstream_root / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from mlr.coarse_lap_oracle import (
        apply_uniform_laplacian,
        build_uniform_laplacian_data,
    )
    from mlr.data import Camera, Mesh
    from mlr.learned_laplacian.dataset import save_prepared_sample, validate_sample
    from mlr.learned_laplacian.graph_layers import faces_to_edge_index
    from mlr.learned_laplacian.renderer_visibility import (
        compute_renderer_visibility,
        visibility_statistics,
    )
    from mlr.learned_laplacian.target_scaling import (
        edge_scale_statistics,
        incident_edge_length_and_valid_mask,
        normalize_laplacian_by_edge_scale,
    )
    from mlr.synthetic import SyntheticRenderConfig

    return {
        "apply_uniform_laplacian": apply_uniform_laplacian,
        "build_uniform_laplacian_data": build_uniform_laplacian_data,
        "Camera": Camera,
        "Mesh": Mesh,
        "save_prepared_sample": save_prepared_sample,
        "validate_sample": validate_sample,
        "faces_to_edge_index": faces_to_edge_index,
        "compute_renderer_visibility": compute_renderer_visibility,
        "visibility_statistics": visibility_statistics,
        "edge_scale_statistics": edge_scale_statistics,
        "incident_edge_length_and_valid_mask": incident_edge_length_and_valid_mask,
        "normalize_laplacian_by_edge_scale": normalize_laplacian_by_edge_scale,
        "SyntheticRenderConfig": SyntheticRenderConfig,
    }


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _variant_seed(object_id: str, base_seed: int, variant_index: int) -> int:
    digest = hashlib.sha256(
        f"{object_id}:{base_seed}:{variant_index}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little", signed=False) % (2**63 - 1)


def _smooth_scalar_field(
    values: torch.Tensor,
    edge_index: torch.Tensor,
    iterations: int,
    neighbor_weight: float,
) -> torch.Tensor:
    result = values
    source, destination = edge_index
    degree = torch.zeros_like(result)
    degree.index_add_(0, destination, torch.ones_like(result[source]))
    for _ in range(iterations):
        neighbor_sum = torch.zeros_like(result)
        neighbor_sum.index_add_(0, destination, result[source])
        neighbor_mean = neighbor_sum / degree.clamp_min(1.0)
        result = (1.0 - neighbor_weight) * result + neighbor_weight * neighbor_mean
    result = result - result.mean()
    standard_deviation = result.std(unbiased=False)
    if not torch.isfinite(standard_deviation) or float(standard_deviation) <= 1e-12:
        raise ValueError("Perturbation field has zero or invalid variance.")
    return (result / standard_deviation).clamp(-3.0, 3.0)


def _topology_change(
    initial: np.ndarray, current: np.ndarray, faces: np.ndarray
) -> dict[str, int | float]:
    before = np.cross(
        initial[faces[:, 1]] - initial[faces[:, 0]],
        initial[faces[:, 2]] - initial[faces[:, 0]],
    )
    after = np.cross(
        current[faces[:, 1]] - current[faces[:, 0]],
        current[faces[:, 2]] - current[faces[:, 0]],
    )
    before_area = 0.5 * np.linalg.norm(before, axis=1)
    after_area = 0.5 * np.linalg.norm(after, axis=1)
    return {
        "flipped_faces": int(np.sum(np.einsum("ij,ij->i", before, after) < 0.0)),
        "new_degenerate_faces": int(
            np.sum((after_area <= 5e-15) & (before_area > 5e-15))
        ),
        "minimum_triangle_area": float(after_area.min(initial=np.inf)),
    }


def _bad_topology_faces(
    initial: np.ndarray, current: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, dict[str, int | float]]:
    before = np.cross(
        initial[faces[:, 1]] - initial[faces[:, 0]],
        initial[faces[:, 2]] - initial[faces[:, 0]],
    )
    after = np.cross(
        current[faces[:, 1]] - current[faces[:, 0]],
        current[faces[:, 2]] - current[faces[:, 0]],
    )
    before_area = 0.5 * np.linalg.norm(before, axis=1)
    after_area = 0.5 * np.linalg.norm(after, axis=1)
    flipped = np.einsum("ij,ij->i", before, after) < 0.0
    newly_degenerate = (after_area <= 5e-15) & (before_area > 5e-15)
    return flipped | newly_degenerate, {
        "flipped_faces": int(flipped.sum()),
        "new_degenerate_faces": int(newly_degenerate.sum()),
        "minimum_triangle_area": float(after_area.min(initial=np.inf)),
    }


def perturb_current_vertices(
    gt_vertices: torch.Tensor,
    gt_normals: torch.Tensor,
    faces: torch.Tensor,
    *,
    seed: int,
    perturb_std_h: float,
    smooth_iterations: int,
    neighbor_weight: float,
    deps: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    if perturb_std_h <= 0:
        raise ValueError("perturb_std_h must be positive.")
    edge_index = deps["faces_to_edge_index"](faces, int(gt_vertices.shape[0]))
    local_h, valid = deps["incident_edge_length_and_valid_mask"](
        gt_vertices, edge_index
    )
    if not bool(valid.all()):
        raise ValueError("Synthetic current generation requires no isolated vertices.")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    scalar = torch.randn(gt_vertices.shape[0], generator=generator)
    scalar = _smooth_scalar_field(
        scalar, edge_index, smooth_iterations, neighbor_weight
    )
    requested = (
        scalar[:, None]
        * local_h[:, None]
        * float(perturb_std_h)
        * torch.nn.functional.normalize(gt_normals.float(), dim=-1, eps=1e-8)
    )
    gt_np = gt_vertices.detach().cpu().double().numpy()
    faces_np = faces.detach().cpu().numpy().astype(np.int64)
    realised = requested.clone()
    damping = torch.ones(gt_vertices.shape[0], dtype=torch.float32)
    locally_damped = torch.zeros(gt_vertices.shape[0], dtype=torch.bool)
    topology: dict[str, Any] | None = None
    current = gt_vertices.float()
    for iteration in range(32):
        current = gt_vertices.float() + realised
        bad_faces, topology = _bad_topology_faces(
            gt_np,
            current.detach().cpu().double().numpy(),
            faces_np,
        )
        if not np.any(bad_faces):
            break
        bad_vertices = np.unique(faces_np[bad_faces].reshape(-1))
        bad_vertices_t = torch.as_tensor(bad_vertices, dtype=torch.long)
        if iteration < 16:
            realised[bad_vertices_t] *= 0.5
            damping[bad_vertices_t] *= 0.5
        else:
            realised[bad_vertices_t] = 0.0
            damping[bad_vertices_t] = 0.0
        locally_damped[bad_vertices_t] = True
    current = gt_vertices.float() + realised
    bad_faces, topology = _bad_topology_faces(
        gt_np,
        current.detach().cpu().double().numpy(),
        faces_np,
    )
    if np.any(bad_faces):
        raise RuntimeError("Could not generate a topology-valid perturbed current mesh.")
    realised_offset = current - gt_vertices.float()
    ratio = torch.linalg.vector_norm(realised_offset, dim=-1) / local_h.clamp_min(1e-12)
    diagnostics = {
        "seed": int(seed),
        "requested_perturb_std_h": float(perturb_std_h),
        "realised_global_scale": 1.0,
        "locally_damped_vertex_ratio": float(locally_damped.float().mean()),
        "minimum_local_damping": float(damping.min()),
        "mean_offset_over_h": float(ratio.mean()),
        "median_offset_over_h": float(ratio.median()),
        "p95_offset_over_h": float(torch.quantile(ratio, 0.95)),
        "max_offset_over_h": float(ratio.max()),
        **topology,
    }
    return current, diagnostics


def _cameras(sample: Mapping[str, Any], deps: Mapping[str, Any]) -> list[Any]:
    image_size = int(sample.get("prepared_image_size", 960))
    result = []
    for index in range(VIEW_COUNT):
        extrinsic = sample["extrinsics"][index].detach().cpu().double().numpy()
        result.append(
            deps["Camera"](
                intrinsics=sample["intrinsics"][index].detach().cpu().double().numpy(),
                rotation=extrinsic[:3, :3],
                translation=extrinsic[:3, 3],
                image_size=(image_size, image_size),
                name=f"view_{index:02d}",
            )
        )
    return result


def _attach_visibility(
    sample: dict[str, Any],
    current_mesh: Any,
    source: Mapping[str, Any],
    *,
    backend: str,
    artifact_path: Path,
    deps: Mapping[str, Any],
) -> dict[str, Any]:
    image_size = int(source.get("prepared_image_size", 960))
    config = deps["SyntheticRenderConfig"](
        width=image_size,
        height=image_size,
        num_views=VIEW_COUNT,
        backend=backend,
        normalize_mesh=False,
        backface_culling=False,
        front_face_winding="ccw",
        antialiasing="none",
    )
    result = deps["compute_renderer_visibility"](
        current_mesh,
        _cameras(source, deps),
        config,
        neighborhood_radius=1,
    )
    arrays = {
        "visibility_frustum": result.frustum_valid,
        "visibility_backface_only": result.backface_visible,
        "visibility_occlusion_only": result.occlusion_visible,
        "visibility_backface_and_occlusion": result.backface_and_occlusion_visible,
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(artifact_path, **arrays)
    sample["visibility"] = torch.as_tensor(
        result.backface_and_occlusion_visible, dtype=torch.bool
    )
    sample["visibility_backface_only"] = torch.as_tensor(
        result.backface_visible, dtype=torch.bool
    )
    sample["visibility_occlusion_only"] = torch.as_tensor(
        result.occlusion_visible, dtype=torch.bool
    )
    sample["visibility_backface_and_occlusion"] = sample["visibility"]
    return {
        "definition": "depth_tested_face_id_incident_face_neighborhood",
        "backend": result.backend,
        "front_face_winding": result.front_face_winding,
        "neighborhood_radius": 1,
        "depth_image_used": False,
        "graph_role": "synthetic_current_query_training",
        "artifact_path": str(artifact_path),
        **deps["visibility_statistics"](result),
    }


def build_current_sample(
    source: Mapping[str, Any],
    *,
    object_id: str,
    split: str,
    variant_index: int,
    base_seed: int,
    perturb_std_h: float,
    smooth_iterations: int,
    neighbor_weight: float,
    output_root: Path,
    source_root: Path,
    visibility_backend: str | None,
    visibility_artifact: Path | None,
    deps: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if int(source["intrinsics"].shape[0]) != VIEW_COUNT:
        raise ValueError(f"Expected exactly {VIEW_COUNT} source views.")
    gt_vertices = source["gt_vertices"].detach().cpu().float()
    gt_faces = source["gt_faces"].detach().cpu().long()
    if not torch.equal(source["faces"].detach().cpu().long(), gt_faces):
        raise ValueError("Source GT-query sample faces must equal gt_faces.")
    gt_normals = deps["Mesh"](
        gt_vertices.double().numpy(), gt_faces.numpy()
    ).ensure_normals().normals
    seed = _variant_seed(object_id, base_seed, variant_index)
    current_vertices, perturbation = perturb_current_vertices(
        gt_vertices,
        torch.as_tensor(gt_normals, dtype=torch.float32),
        gt_faces,
        seed=seed,
        perturb_std_h=perturb_std_h,
        smooth_iterations=smooth_iterations,
        neighbor_weight=neighbor_weight,
        deps=deps,
    )
    faces_np = gt_faces.numpy()
    current_mesh = deps["Mesh"](
        current_vertices.double().numpy(), faces_np.copy()
    ).ensure_normals()
    laplacian_data = deps["build_uniform_laplacian_data"](
        faces_np, int(current_vertices.shape[0])
    )
    initial_raw = torch.as_tensor(
        deps["apply_uniform_laplacian"](current_mesh.vertices, laplacian_data),
        dtype=torch.float32,
    )
    target_raw = torch.as_tensor(
        deps["apply_uniform_laplacian"](
            gt_vertices.double().numpy(), laplacian_data
        ),
        dtype=torch.float32,
    )
    edge_index = deps["faces_to_edge_index"](gt_faces, int(gt_vertices.shape[0]))
    local_h, valid_scale = deps["incident_edge_length_and_valid_mask"](
        current_vertices, edge_index
    )
    target_normalized = deps["normalize_laplacian_by_edge_scale"](
        target_raw,
        local_h,
        eps=1e-12,
        valid_scale_mask=valid_scale,
    )
    center = 0.5 * (current_vertices.amin(dim=0) + current_vertices.amax(dim=0))
    position_scale = torch.linalg.vector_norm(
        current_vertices - center, dim=-1
    ).amax()
    if not torch.isfinite(position_scale) or float(position_scale) <= 1e-12:
        raise ValueError("Current mesh has invalid position-normalization scale.")
    sample_id = f"{object_id}__v{variant_index:02d}"
    image_paths = []
    for value in source["image_paths"]:
        absolute = Path(value)
        if not absolute.is_absolute():
            absolute = source_root / absolute
        image_paths.append(os.path.relpath(absolute.resolve(), output_root))
    sample: dict[str, Any] = {
        "sample_id": sample_id,
        "image_paths": image_paths,
        "prepared_storage_format": "lazy_image_paths_v1",
        "source_image_size": list(source["source_image_size"]),
        "prepared_image_size": int(source["prepared_image_size"]),
        "intrinsics": source["intrinsics"].detach().cpu().float().clone(),
        "extrinsics": source["extrinsics"].detach().cpu().float().clone(),
        "vertices": current_vertices,
        "faces": gt_faces,
        "vertex_normals": torch.as_tensor(current_mesh.normals, dtype=torch.float32),
        "initial_laplacian": initial_raw,
        "laplacian_target": target_raw,
        "raw_laplacian_target": target_raw,
        "normalized_laplacian_target": target_normalized,
        "target_confidence": torch.ones(gt_vertices.shape[0], dtype=torch.float32),
        "target_positions": gt_vertices,
        "gt_vertices": gt_vertices,
        "gt_faces": gt_faces,
        "position_normalization_center": center,
        "position_normalization_scale": position_scale,
        "local_edge_length": local_h,
        "local_edge_scale": local_h.square(),
        "valid_scale_mask": valid_scale,
        "metadata": {
            "dataset_family": FORMAT_VERSION,
            "dataset_role": "synthetic_current_query_training",
            "training_eligible": True,
            "object_id": object_id,
            "source_sample_id": str(source["sample_id"]),
            "source_split": split,
            "variant_index": int(variant_index),
            "variant_seed": int(seed),
            "view_count": VIEW_COUNT,
            "input_resolution": int(source["prepared_image_size"]),
            "current_graph_source": "deterministic_smooth_normal_perturbation_of_gt_topology",
            "proxy_definition": "P_proxy=source_gt_vertices_with_exact_same_topology",
            "target_constructor": "delta_target=L_current@P_proxy",
            "normalization": "delta_target_hat=delta_target/(h_current^2+1e-12)",
            "laplacian_target_mode": TARGET_MODE,
            "edge_scale_definition": "square_of_mean_incident_edge_length",
            "edge_scale_source": "synthetic_current_graph",
            "edge_scale_epsilon": 1e-12,
            "operator_type": "uniform",
            "query_training_mode": "fixed_synthetic_current_graph_v1",
            "training_geometry_source": "synthetic_current_vertices_and_current_faces",
            "initial_laplacian_input": "L_current@C",
            "position_normalization": "bbox_center_max_radius",
            "perturbation": perturbation,
            "edge_scale_statistics": deps["edge_scale_statistics"](local_h),
        },
    }
    if visibility_backend is None:
        for name in (
            "visibility",
            "visibility_backface_only",
            "visibility_occlusion_only",
            "visibility_backface_and_occlusion",
        ):
            sample[name] = source[name].detach().cpu().clone()
        sample["metadata"]["renderer_visibility"] = {
            "recomputed_on_current_graph": False,
            "debug_only": True,
        }
    else:
        if visibility_artifact is None:
            raise ValueError("visibility_artifact is required when visibility is recomputed.")
        sample["metadata"]["renderer_visibility"] = _attach_visibility(
            sample,
            current_mesh,
            source,
            backend=visibility_backend,
            artifact_path=visibility_artifact,
            deps=deps,
        )
        sample["metadata"]["renderer_visibility"]["recomputed_on_current_graph"] = True
    validated = deps["validate_sample"](sample)
    raw_check = torch.as_tensor(
        deps["apply_uniform_laplacian"](
            gt_vertices.double().numpy(), laplacian_data
        ),
        dtype=torch.float32,
    )
    max_raw_error = float(torch.max(torch.abs(raw_check - target_raw)))
    roundtrip = target_normalized * (local_h.square() + 1e-12)[:, None]
    max_roundtrip_error = float(torch.max(torch.abs(roundtrip - target_raw)))
    oracle = {
        "sample_id": sample_id,
        "object_id": object_id,
        "split": split,
        "variant_index": int(variant_index),
        "vertex_count": int(gt_vertices.shape[0]),
        "face_count": int(gt_faces.shape[0]),
        "max_abs_Lc_Pproxy_target_error": max_raw_error,
        "max_abs_h2_roundtrip_error": max_roundtrip_error,
        "all_finite": bool(
            torch.isfinite(current_vertices).all()
            and torch.isfinite(target_normalized).all()
        ),
        "target_contract_pass": bool(max_raw_error <= 1e-7),
        "normalization_roundtrip_pass": bool(max_roundtrip_error <= 1e-5),
        "perturbation": perturbation,
    }
    return validated, oracle


def _oracle_from_saved_sample(
    sample: Mapping[str, Any], deps: Mapping[str, Any]
) -> dict[str, Any]:
    faces = sample["faces"].detach().cpu().numpy().astype(np.int64)
    proxy = sample["target_positions"].detach().cpu().double().numpy()
    operator = deps["build_uniform_laplacian_data"](faces, len(proxy))
    expected = torch.as_tensor(
        deps["apply_uniform_laplacian"](proxy, operator), dtype=torch.float32
    )
    target_raw = sample["raw_laplacian_target"].detach().cpu().float()
    target_normalized = sample["normalized_laplacian_target"].detach().cpu().float()
    local_h = sample["local_edge_length"].detach().cpu().float()
    roundtrip = target_normalized * (local_h.square() + 1e-12)[:, None]
    max_raw_error = float(torch.max(torch.abs(expected - target_raw)))
    max_roundtrip_error = float(torch.max(torch.abs(roundtrip - target_raw)))
    metadata = dict(sample.get("metadata", {}))
    return {
        "sample_id": str(sample["sample_id"]),
        "object_id": str(metadata["object_id"]),
        "split": str(metadata["source_split"]),
        "variant_index": int(metadata["variant_index"]),
        "vertex_count": int(sample["vertices"].shape[0]),
        "face_count": int(sample["faces"].shape[0]),
        "max_abs_Lc_Pproxy_target_error": max_raw_error,
        "max_abs_h2_roundtrip_error": max_roundtrip_error,
        "all_finite": bool(
            torch.isfinite(sample["vertices"]).all()
            and torch.isfinite(target_normalized).all()
        ),
        "target_contract_pass": bool(max_raw_error <= 1e-7),
        "normalization_roundtrip_pass": bool(max_roundtrip_error <= 1e-5),
        "perturbation": dict(metadata["perturbation"]),
    }


def _generate_object(task: Mapping[str, Any]) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    downstream_root = Path(str(task["downstream_root"]))
    source_manifest = Path(str(task["source_manifest"]))
    output_root = Path(str(task["output_root"]))
    record = dict(task["record"])
    deps = _dependencies(downstream_root)
    object_id = str(record["sample_id"])
    split = str(record["split"])
    source_path = Path(str(record["path"]))
    if not source_path.is_absolute():
        source_path = source_manifest.parent / source_path
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    print(
        f"[{task['object_index']}/{task['object_count']}] {object_id} split={split}",
        flush=True,
    )
    output_records: list[dict[str, str]] = []
    oracle_rows: list[dict[str, Any]] = []
    for variant_index in range(VARIANT_COUNT):
        print(f"  {object_id} variant={variant_index:02d}", flush=True)
        relative = Path("prepared") / split / object_id / f"variant_{variant_index:02d}.pt"
        prepared_path = output_root / relative
        artifact = (
            output_root
            / "renderer_visibility"
            / split
            / object_id
            / f"variant_{variant_index:02d}.npz"
        )
        if bool(task["resume"]) and prepared_path.is_file() and artifact.is_file():
            saved = torch.load(prepared_path, map_location="cpu", weights_only=False)
            deps["validate_sample"](saved)
            output_records.append(
                {"path": relative.as_posix(), "sample_id": str(saved["sample_id"]), "split": split}
            )
            oracle_rows.append(_oracle_from_saved_sample(saved, deps))
            print(f"    reuse={relative}", flush=True)
            continue
        sample, oracle = build_current_sample(
            source,
            object_id=object_id,
            split=split,
            variant_index=variant_index,
            base_seed=int(task["seed"]),
            perturb_std_h=float(task["perturb_std_h"]),
            smooth_iterations=int(task["smooth_iterations"]),
            neighbor_weight=float(task["neighbor_weight"]),
            output_root=output_root,
            source_root=source_manifest.parent,
            visibility_backend=(
                None
                if bool(task["debug_reuse_source_visibility"])
                else str(task["visibility_backend"])
            ),
            visibility_artifact=artifact,
            deps=deps,
        )
        deps["save_prepared_sample"](sample, prepared_path)
        output_records.append(
            {"path": relative.as_posix(), "sample_id": sample["sample_id"], "split": split}
        )
        oracle_rows.append(oracle)
    return output_records, oracle_rows


def generate_dataset(args: argparse.Namespace) -> dict[str, Any]:
    downstream_root = args.downstream_root.expanduser().resolve()
    source_manifest = args.source_manifest.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    manifest = _read_json(source_manifest)
    records = list(manifest.get("samples", []))
    if len(records) != 50:
        raise ValueError(f"Expected 50 Sofa50 objects, found {len(records)}.")
    split_counts = {
        split: sum(record.get("split") == split for record in records)
        for split in ("train", "validation", "test")
    }
    if split_counts != {"train": 40, "validation": 5, "test": 5}:
        raise ValueError(f"Unexpected object-level split counts: {split_counts}")
    if output_root.exists() and any(output_root.iterdir()) and not (args.overwrite or args.resume):
        raise FileExistsError(
            f"Output directory is non-empty: {output_root}; pass --overwrite explicitly."
        )
    output_root.mkdir(parents=True, exist_ok=True)
    selected = records[: args.limit] if args.limit is not None else records
    tasks = [
        {
            "record": record,
            "object_index": object_index,
            "object_count": len(selected),
            "downstream_root": str(downstream_root),
            "source_manifest": str(source_manifest),
            "output_root": str(output_root),
            "seed": int(args.seed),
            "perturb_std_h": float(args.perturb_std_h),
            "smooth_iterations": int(args.smooth_iterations),
            "neighbor_weight": float(args.neighbor_weight),
            "visibility_backend": args.visibility_backend,
            "debug_reuse_source_visibility": bool(args.debug_reuse_source_visibility),
            "resume": bool(args.resume),
        }
        for object_index, record in enumerate(selected, start=1)
    ]
    if args.workers == 1 or len(tasks) == 1:
        results = [_generate_object(task) for task in tasks]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = list(executor.map(_generate_object, tasks))
    output_records = [record for records_out, _ in results for record in records_out]
    oracle_rows = [row for _, rows_out in results for row in rows_out]
    per_object_variants: dict[str, list[str]] = {}
    for record in output_records:
        object_id = record["sample_id"].rsplit("__v", 1)[0]
        per_object_variants.setdefault(object_id, []).append(record["sample_id"])
    for object_id, variant_ids in per_object_variants.items():
        if len(variant_ids) != VARIANT_COUNT:
            raise AssertionError(f"Object {object_id} does not have five variants.")
    output_split_counts = {
        split: sum(record["split"] == split for record in output_records)
        for split in ("train", "validation", "test")
    }
    expected_multiplier = VARIANT_COUNT
    if args.limit is None and output_split_counts != {
        "train": 40 * expected_multiplier,
        "validation": 5 * expected_multiplier,
        "test": 5 * expected_multiplier,
    }:
        raise AssertionError(f"Unexpected variant split counts: {output_split_counts}")
    manifest_out = {
        "format_version": FORMAT_VERSION,
        "dataset_role": "synthetic_current_query_training_and_shared_test_evaluation",
        "source_manifest": str(source_manifest),
        "object_split_counts": split_counts,
        "variant_split_counts": output_split_counts,
        "variants_per_object": VARIANT_COUNT,
        "object_level_split_enforced": True,
        "view_count": VIEW_COUNT,
        "target": "delta_target_hat=(L_current@P_proxy)/(h_current^2+1e-12)",
        "samples": output_records,
    }
    oracle_summary = {
        "format_version": FORMAT_VERSION,
        "sample_count": len(oracle_rows),
        "all_target_contracts_pass": all(row["target_contract_pass"] for row in oracle_rows),
        "all_normalization_roundtrips_pass": all(
            row["normalization_roundtrip_pass"] for row in oracle_rows
        ),
        "maximum_target_error": max(
            (row["max_abs_Lc_Pproxy_target_error"] for row in oracle_rows), default=0.0
        ),
        "maximum_roundtrip_error": max(
            (row["max_abs_h2_roundtrip_error"] for row in oracle_rows), default=0.0
        ),
        "per_sample": oracle_rows,
    }
    _write_json(output_root / "manifest.json", manifest_out)
    _write_json(output_root / "oracle_validation.json", oracle_summary)
    _write_json(
        output_root / "generation_config.json",
        {
            "source_manifest": str(source_manifest),
            "downstream_root": str(downstream_root),
            "output_root": str(output_root),
            "seed": int(args.seed),
            "variants_per_object": VARIANT_COUNT,
            "perturb_std_h": float(args.perturb_std_h),
            "smooth_iterations": int(args.smooth_iterations),
            "neighbor_weight": float(args.neighbor_weight),
            "visibility_backend": (
                "source_debug_only" if args.debug_reuse_source_visibility else args.visibility_backend
            ),
            "workers": int(args.workers),
            "resume": bool(args.resume),
            "limit": args.limit,
        },
    )
    if not oracle_summary["all_target_contracts_pass"]:
        raise RuntimeError("At least one L_current @ P_proxy target contract failed.")
    if not oracle_summary["all_normalization_roundtrips_pass"]:
        raise RuntimeError("At least one h^2 normalization round-trip failed.")
    return {"manifest": manifest_out, "oracle": oracle_summary}


def build_parser() -> argparse.ArgumentParser:
    home = Path.home()
    parser = argparse.ArgumentParser(
        description="Generate five fixed synthetic-current C-query variants per Sofa50 object."
    )
    parser.add_argument(
        "--downstream-root",
        type=Path,
        default=home / "multiview-laplacian-refinement",
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=(
            home
            / "sofa_mesh"
            / "sofa50_refinement"
            / "multiview_nested_14_28_56_cpu_v3"
            / "gt_query_views_14_manifest.json"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=home / "sofa_mesh" / "sofa50_synthetic_current",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--perturb-std-h", type=float, default=0.15)
    parser.add_argument("--smooth-iterations", type=int, default=5)
    parser.add_argument("--neighbor-weight", type=float, default=0.65)
    parser.add_argument("--visibility-backend", choices=("opengl", "cpu"), default="cpu")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--debug-reuse-source-visibility",
        action="store_true",
        help="Smoke-test only; final data must recompute current-graph visibility.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive.")
    if args.workers < 1:
        raise ValueError("--workers must be positive.")
    result = generate_dataset(args)
    print(
        json.dumps(
            {
                "samples": len(result["manifest"]["samples"]),
                "variant_split_counts": result["manifest"]["variant_split_counts"],
                "target_contract": result["oracle"]["all_target_contracts_pass"],
                "normalization_roundtrip": result["oracle"][
                    "all_normalization_roundtrips_pass"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
