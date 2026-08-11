#!/usr/bin/env python3
from __future__ import annotations

"""Combine Sofa50 nested 28-view observations with GT-adaptive query graphs."""

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


FORMAT_VERSION = "sofa50_view28_gt_adaptive_v1"
VIEW_COUNT = 28
VISIBILITY_FIELDS = (
    "visibility",
    "visibility_backface_only",
    "visibility_occlusion_only",
    "visibility_backface_and_occlusion",
)
OBSERVATION_FIELDS = (
    "intrinsics",
    "extrinsics",
    "source_image_size",
    "prepared_image_size",
    "prepared_storage_format",
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def records_by_id(manifest: Mapping[str, Any], label: str) -> tuple[list[str], dict[str, dict[str, str]]]:
    items = manifest.get("samples")
    if not isinstance(items, list) or not items:
        raise ValueError(f"{label}: manifest has no samples")
    order: list[str] = []
    records: dict[str, dict[str, str]] = {}
    for item in items:
        if not isinstance(item, Mapping):
            raise ValueError(f"{label}: sample record is not an object")
        sample_id = str(item.get("sample_id", ""))
        split = str(item.get("split", ""))
        path = str(item.get("path", ""))
        if not sample_id or not split or not path:
            raise ValueError(f"{label}: incomplete record {item}")
        if sample_id in records:
            raise ValueError(f"{label}: duplicate sample_id {sample_id}")
        order.append(sample_id)
        records[sample_id] = {"sample_id": sample_id, "split": split, "path": path}
    return order, records


def resolve_record(manifest_path: Path, record: Mapping[str, str]) -> Path:
    path = Path(record["path"])
    return path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()


def relative_image_paths(sample: Mapping[str, Any], source_root: Path, output_root: Path) -> list[str]:
    values = sample.get("image_paths")
    if not isinstance(values, list) or len(values) != VIEW_COUNT:
        raise ValueError(f"Expected {VIEW_COUNT} lazy image paths")
    result: list[str] = []
    for value in values:
        path = Path(str(value))
        resolved = path.resolve() if path.is_absolute() else (source_root / path).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Missing 28-view image: {resolved}")
        result.append(Path(os.path.relpath(resolved, output_root)).as_posix())
    return result


def merge_observations(
    adaptive: Mapping[str, Any],
    view28: Mapping[str, Any],
    *,
    image_paths: list[str],
) -> dict[str, Any]:
    if adaptive.get("sample_id") != view28.get("sample_id"):
        raise ValueError("Adaptive and view28 sample IDs differ")
    result = copy.copy(dict(adaptive))
    for field in OBSERVATION_FIELDS:
        value = view28.get(field)
        result[field] = value.clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
    result["image_paths"] = list(image_paths)
    result.pop("images", None)
    result.pop("image_path_root", None)
    for field in VISIBILITY_FIELDS:
        result.pop(field, None)
    metadata = copy.deepcopy(dict(adaptive.get("metadata", {})))
    view_metadata = view28.get("metadata", {})
    if isinstance(view_metadata, Mapping):
        for key in (
            "dataset_path",
            "dataset_family",
            "render_geometry_role",
            "render_source_path",
            "nested_view_count",
            "nested_master_view_count",
            "nested_subset_rule",
            "camera_layout_version",
            "base_14_camera_poses_reused_exactly",
            "base_14_observations_reused_exactly",
            "all_views_rendered_in_single_cpu_master",
            "renderer_fallback_allowed",
            "renderer_backend_control",
        ):
            if key in view_metadata:
                metadata[key] = copy.deepcopy(view_metadata[key])
    metadata.update(
        {
            "ablation": "view_count_plus_query_resolution_combo_v1",
            "combination_arm": "views_28_gt_adaptive",
            "query_graph_variant": "gt_adaptive",
            "observation_view_count": VIEW_COUNT,
            "visibility_policy": "renderer_native_current_query_graph_and_view_subset",
            "renderer_visibility_recompute_required": False,
        }
    )
    result["metadata"] = metadata
    return result


def cameras_and_mesh(sample: Mapping[str, Any], deps: Mapping[str, Any]) -> tuple[list[Any], Any]:
    intrinsics = sample["intrinsics"].detach().cpu().numpy()
    extrinsics = sample["extrinsics"].detach().cpu().numpy()
    image_size = int(sample["prepared_image_size"])
    cameras = [
        deps["Camera"](
            intrinsics=intrinsics[index],
            rotation=extrinsics[index, :3, :3],
            translation=extrinsics[index, :3, 3],
            image_size=(image_size, image_size),
            name=f"view_{index:04d}",
        )
        for index in range(len(intrinsics))
    ]
    mesh = deps["Mesh"](
        sample["vertices"].detach().cpu().numpy(),
        sample["faces"].detach().cpu().numpy(),
    ).ensure_normals()
    return cameras, mesh


def attach_visibility(
    sample: dict[str, Any],
    result: Any,
    artifact_path: Path,
    output_root: Path,
) -> None:
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        artifact_path,
        frustum_valid=result.frustum_valid,
        visibility_backface_only=result.backface_visible,
        visibility_occlusion_only=result.occlusion_visible,
        visibility_backface_and_occlusion=result.backface_and_occlusion_visible,
    )
    sample["visibility_backface_only"] = torch.from_numpy(result.backface_visible)
    sample["visibility_occlusion_only"] = torch.from_numpy(result.occlusion_visible)
    sample["visibility_backface_and_occlusion"] = torch.from_numpy(result.backface_and_occlusion_visible)
    sample["visibility"] = sample["visibility_backface_and_occlusion"]
    combined = result.backface_and_occlusion_visible
    visible_per_vertex = combined.sum(axis=0)
    metadata = dict(sample["metadata"])
    metadata["renderer_visibility"] = {
        "definition": "depth_tested_face_id_incident_face_neighborhood",
        "artifact_path": artifact_path.relative_to(output_root).as_posix(),
        "backend": "cuda",
        "front_face_winding": "ccw",
        "neighborhood_radius": 1,
        "depth_image_used": False,
        "graph": "gt_adaptive",
        "view_count": VIEW_COUNT,
        "mean_visible_views_per_vertex": float(visible_per_vertex.mean()),
        "zero_visible_vertex_ratio": float(np.mean(visible_per_vertex == 0)),
    }
    sample["metadata"] = metadata


def dependencies(downstream_root: Path) -> dict[str, Any]:
    source = downstream_root / "src"
    if not source.is_dir():
        raise FileNotFoundError(f"Downstream source tree not found: {source}")
    sys.path.insert(0, str(source))
    from mlr.data import Camera, Mesh
    from mlr.learned_laplacian.dataset import save_prepared_sample
    from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset
    from mlr.learned_laplacian.multi_trainer import _prepare_object_static
    from mlr.learned_laplacian.query_training import validate_gt_query_contract
    from sofa50_refinement.gpu_visibility import compute_renderer_visibility_cuda

    return {
        "Camera": Camera,
        "Mesh": Mesh,
        "PreparedMeshDataset": PreparedMeshDataset,
        "prepare_object_static": _prepare_object_static,
        "validate_gt_query_contract": validate_gt_query_contract,
        "save_prepared_sample": save_prepared_sample,
        "compute_renderer_visibility_cuda": compute_renderer_visibility_cuda,
    }


def validate_output(
    manifest_path: Path,
    config: Mapping[str, Any],
    deps: Mapping[str, Any],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    reference_ids: set[str] = set()
    for split in ("train", "validation", "test"):
        dataset = deps["PreparedMeshDataset"].from_manifest(manifest_path, split)
        vertex_counts: list[int] = []
        for index in range(len(dataset)):
            sample = dataset.load_static(index)
            deps["validate_gt_query_contract"](sample)
            vertices = int(sample["vertices"].shape[0])
            visibility = sample.get("visibility_backface_and_occlusion")
            if not isinstance(visibility, torch.Tensor) or tuple(visibility.shape) != (VIEW_COUNT, vertices):
                raise ValueError(f"{sample['sample_id']}: visibility shape does not match 28xN")
            deps["prepare_object_static"](
                sample,
                config,
                keep_image_payload=True,
                keep_projection=True,
            )
            root = Path(sample["_dataset_root"])
            for value in sample["image_paths"]:
                path = Path(value)
                resolved = path if path.is_absolute() else root / path
                if not resolved.is_file():
                    raise FileNotFoundError(f"{sample['sample_id']}: missing image {resolved}")
            if sample["sample_id"] in reference_ids:
                raise ValueError(f"Duplicate sample across splits: {sample['sample_id']}")
            reference_ids.add(sample["sample_id"])
            vertex_counts.append(vertices)
        output[split] = {
            "samples": len(dataset),
            "min_vertices": min(vertex_counts),
            "max_vertices": max(vertex_counts),
        }
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--views-manifest", required=True, type=Path)
    parser.add_argument("--adaptive-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--downstream-root", required=True, type=Path)
    parser.add_argument("--training-config", required=True, type=Path)
    parser.add_argument("--model-id", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU visibility fallback is disabled")
    views_manifest = args.views_manifest.expanduser().resolve()
    adaptive_manifest = args.adaptive_manifest.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    downstream_root = args.downstream_root.expanduser().resolve()
    config_path = args.training_config.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    deps = dependencies(downstream_root)

    view_order, view_records = records_by_id(read_json(views_manifest), "views_28")
    adaptive_order, adaptive_records = records_by_id(read_json(adaptive_manifest), "gt_adaptive")
    if view_order != adaptive_order:
        raise ValueError("views_28 and gt_adaptive sample ID order differs")
    for sample_id in view_order:
        if view_records[sample_id]["split"] != adaptive_records[sample_id]["split"]:
            raise ValueError(f"{sample_id}: split differs between source manifests")
    requested = set(args.model_id)
    selected = [sample_id for sample_id in view_order if not requested or sample_id in requested]
    missing = requested - set(selected)
    if missing:
        raise ValueError("Unknown --model-id values: " + ", ".join(sorted(missing)))

    output_records: list[dict[str, str]] = []
    per_sample: list[dict[str, Any]] = []
    for index, sample_id in enumerate(selected, start=1):
        output_path = output_root / "prepared" / f"{sample_id}.pt"
        artifact_path = output_root / "renderer_visibility" / f"{sample_id}.npz"
        view_path = resolve_record(views_manifest, view_records[sample_id])
        adaptive_path = resolve_record(adaptive_manifest, adaptive_records[sample_id])
        print(f"[{index}/{len(selected)}] {sample_id}", flush=True)
        if args.force or not output_path.is_file() or not artifact_path.is_file():
            view28 = torch.load(view_path, map_location="cpu", weights_only=False)
            adaptive = torch.load(adaptive_path, map_location="cpu", weights_only=False)
            image_paths = relative_image_paths(view28, views_manifest.parent, output_root)
            sample = merge_observations(adaptive, view28, image_paths=image_paths)
            cameras, mesh = cameras_and_mesh(sample, deps)
            result = deps["compute_renderer_visibility_cuda"](
                mesh,
                cameras,
                image_size=int(sample["prepared_image_size"]),
                neighborhood_radius=1,
                front_face_winding="ccw",
            )
            attach_visibility(sample, result, artifact_path, output_root)
            deps["validate_gt_query_contract"](sample)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            deps["save_prepared_sample"](sample, output_path)
        stored = torch.load(output_path, map_location="cpu", weights_only=False)
        vertices = int(stored["vertices"].shape[0])
        output_records.append(
            {
                "sample_id": sample_id,
                "split": view_records[sample_id]["split"],
                "path": output_path.relative_to(output_root).as_posix(),
            }
        )
        per_sample.append(
            {
                "sample_id": sample_id,
                "split": view_records[sample_id]["split"],
                "vertices": vertices,
                "faces": int(stored["faces"].shape[0]),
                "views": len(stored["image_paths"]),
                "visibility_shape": list(stored["visibility"].shape),
                "view_source": str(view_path),
                "graph_source": str(adaptive_path),
            }
        )

    manifest_path = output_root / "manifest.json"
    manifest = {
        "format_version": FORMAT_VERSION,
        "dataset_role": "gt_query_training_view_query_combo_ablation",
        "training_eligible": True,
        "combination_arm": "views_28_gt_adaptive",
        "view_count": VIEW_COUNT,
        "query_graph_variant": "gt_adaptive",
        "target_mode": "edge_scale_normalized_laplacian",
        "samples": output_records,
    }
    write_json(manifest_path, manifest)
    validation = None
    if not requested:
        validation = validate_output(manifest_path, read_json(config_path), deps)
    summary = {
        "format_version": FORMAT_VERSION,
        "manifest": str(manifest_path),
        "source_manifests": {
            "views_28": str(views_manifest),
            "gt_adaptive": str(adaptive_manifest),
        },
        "controls": {
            "observations_reused_from_views_28": True,
            "query_graph_and_targets_reused_from_gt_adaptive": True,
            "renderer_visibility_recomputed_on_gt_adaptive_for_28_views": True,
            "visibility_backend": "cuda",
            "cpu_visibility_fallback_allowed": False,
        },
        "sample_count": len(output_records),
        "validation": validation,
        "samples": per_sample,
    }
    write_json(output_root / "summary.json", summary)
    print(json.dumps({"status": "passed", "manifest": str(manifest_path), "validation": validation}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
