#!/usr/bin/env python3
from __future__ import annotations

"""Attach nested 28-view observations to all fixed synthetic-current variants."""

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


VIEW_COUNT = 28
FORMAT_VERSION = "sofa50_synthetic_current_28view_v1"
FROZEN_FIELDS = (
    "vertices",
    "faces",
    "gt_vertices",
    "gt_faces",
    "target_positions",
    "initial_laplacian",
    "raw_laplacian_target",
    "normalized_laplacian_target",
    "local_edge_length",
    "valid_scale_mask",
)
VISIBILITY_FIELDS = (
    "visibility",
    "visibility_backface_only",
    "visibility_occlusion_only",
    "visibility_backface_and_occlusion",
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def resolve_record(manifest_path: Path, record: Mapping[str, Any]) -> Path:
    path = Path(str(record["path"]))
    return path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()


def records(manifest: Mapping[str, Any], label: str) -> list[dict[str, Any]]:
    value = manifest.get("samples")
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label}: manifest has no samples")
    return [dict(item) for item in value if isinstance(item, Mapping)]


def dependencies(downstream_root: Path) -> dict[str, Any]:
    sys.path.insert(0, str(downstream_root / "src"))
    from mlr.data import Camera, Mesh
    from mlr.learned_laplacian.dataset import save_prepared_sample
    from mlr.learned_laplacian.local_query_jitter import validate_local_query_jitter_contract
    from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset
    from mlr.learned_laplacian.multi_trainer import _prepare_object_static
    from sofa50_refinement.gpu_visibility import compute_renderer_visibility_cuda

    return {
        "Camera": Camera,
        "Mesh": Mesh,
        "save": save_prepared_sample,
        "validate_contract": validate_local_query_jitter_contract,
        "Dataset": PreparedMeshDataset,
        "prepare": _prepare_object_static,
        "visibility": compute_renderer_visibility_cuda,
    }


def image_paths(view_sample: Mapping[str, Any], source_root: Path, output_root: Path) -> list[str]:
    values = view_sample.get("image_paths")
    if not isinstance(values, list) or len(values) != VIEW_COUNT:
        raise ValueError(f"Expected {VIEW_COUNT} image paths")
    output = []
    for value in values:
        path = Path(str(value))
        resolved = path.resolve() if path.is_absolute() else (source_root / path).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        output.append(Path(os.path.relpath(resolved, output_root)).as_posix())
    return output


def merge(
    current: Mapping[str, Any],
    view_sample: Mapping[str, Any],
    paths: list[str],
) -> dict[str, Any]:
    sample = copy.copy(dict(current))
    for field in ("intrinsics", "extrinsics", "source_image_size", "prepared_image_size"):
        value = view_sample[field]
        sample[field] = value.clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
    sample["prepared_storage_format"] = "lazy_image_paths_v1"
    sample["image_paths"] = paths
    sample.pop("images", None)
    sample.pop("image_path_root", None)
    for field in VISIBILITY_FIELDS:
        sample.pop(field, None)
    metadata = copy.deepcopy(dict(current.get("metadata", {})))
    metadata.update(
        {
            "view_count": VIEW_COUNT,
            "observation_view_count": VIEW_COUNT,
            "observation_source": "nested_views_28",
            "runtime_query_jitter_eligible": True,
            "graph_proxy_target_policy": "frozen_from_synthetic_current_source",
            "visibility_policy": "recomputed_on_each_current_graph_for_28_views",
        }
    )
    sample["metadata"] = metadata
    return sample


def cameras_mesh(sample: Mapping[str, Any], deps: Mapping[str, Any]) -> tuple[list[Any], Any]:
    intrinsics = sample["intrinsics"].detach().cpu().numpy()
    extrinsics = sample["extrinsics"].detach().cpu().numpy()
    size = int(sample["prepared_image_size"])
    cameras = [
        deps["Camera"](
            intrinsics=intrinsics[index],
            rotation=extrinsics[index, :3, :3],
            translation=extrinsics[index, :3, 3],
            image_size=(size, size),
            name=f"view_{index:04d}",
        )
        for index in range(VIEW_COUNT)
    ]
    mesh = deps["Mesh"](
        sample["vertices"].detach().cpu().numpy(),
        sample["faces"].detach().cpu().numpy(),
    ).ensure_normals()
    return cameras, mesh


def attach_visibility(
    sample: dict[str, Any], result: Any, artifact: Path, output_root: Path
) -> None:
    artifact.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        artifact,
        frustum_valid=result.frustum_valid,
        visibility_backface_only=result.backface_visible,
        visibility_occlusion_only=result.occlusion_visible,
        visibility_backface_and_occlusion=result.backface_and_occlusion_visible,
    )
    sample["visibility_backface_only"] = torch.from_numpy(result.backface_visible)
    sample["visibility_occlusion_only"] = torch.from_numpy(result.occlusion_visible)
    combined = torch.from_numpy(result.backface_and_occlusion_visible)
    sample["visibility_backface_and_occlusion"] = combined
    sample["visibility"] = combined
    visible = result.backface_and_occlusion_visible.sum(axis=0)
    metadata = dict(sample["metadata"])
    metadata["renderer_visibility"] = {
        "definition": "depth_tested_face_id_incident_face_neighborhood",
        "artifact_path": artifact.relative_to(output_root).as_posix(),
        "backend": "cuda",
        "front_face_winding": "ccw",
        "neighborhood_radius": 1,
        "depth_image_used": False,
        "graph": "fixed_synthetic_current_variant",
        "view_count": VIEW_COUNT,
        "mean_visible_views_per_vertex": float(visible.mean()),
        "zero_visible_vertex_ratio": float(np.mean(visible == 0)),
    }
    sample["metadata"] = metadata


def assert_frozen(source: Mapping[str, Any], output: Mapping[str, Any]) -> None:
    for field in FROZEN_FIELDS:
        if not torch.equal(source[field], output[field]):
            raise AssertionError(f"Frozen tensor changed: {field}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-manifest", required=True, type=Path)
    parser.add_argument("--views-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--downstream-root", required=True, type=Path)
    parser.add_argument("--training-config", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU visibility fallback is disabled")
    current_manifest = args.current_manifest.expanduser().resolve()
    views_manifest = args.views_manifest.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    downstream_root = args.downstream_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    deps = dependencies(downstream_root)
    current_records = records(read_json(current_manifest), "synthetic current")
    view_records = {
        str(item["sample_id"]): item
        for item in records(read_json(views_manifest), "views 28")
    }
    config = read_json(args.training_config.expanduser().resolve())

    output_records = []
    diagnostics = []
    for index, record in enumerate(current_records, start=1):
        current_path = resolve_record(current_manifest, record)
        current = torch.load(current_path, map_location="cpu", weights_only=True)
        object_id = str(current["metadata"]["object_id"])
        if object_id not in view_records:
            raise ValueError(f"No 28-view source for object {object_id}")
        view_record = view_records[object_id]
        if str(record["split"]) != str(view_record["split"]):
            raise ValueError(f"{record['sample_id']}: split differs from view source")
        view_path = resolve_record(views_manifest, view_record)
        output_path = output_root / "prepared" / str(record["split"]) / object_id / (
            f"variant_{int(current['metadata']['variant_index']):02d}.pt"
        )
        artifact = output_root / "renderer_visibility" / str(record["split"]) / object_id / (
            f"variant_{int(current['metadata']['variant_index']):02d}.npz"
        )
        print(f"[{index}/{len(current_records)}] {record['sample_id']}", flush=True)
        if args.force or not output_path.is_file() or not artifact.is_file():
            view_sample = torch.load(view_path, map_location="cpu", weights_only=True)
            sample = merge(
                current,
                view_sample,
                image_paths(view_sample, views_manifest.parent, output_root),
            )
            cameras, mesh = cameras_mesh(sample, deps)
            visibility = deps["visibility"](
                mesh,
                cameras,
                image_size=int(sample["prepared_image_size"]),
                neighborhood_radius=1,
                front_face_winding="ccw",
            )
            attach_visibility(sample, visibility, artifact, output_root)
            assert_frozen(current, sample)
            deps["validate_contract"](sample)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            deps["save"](sample, output_path)
        stored = torch.load(output_path, map_location="cpu", weights_only=True)
        assert_frozen(current, stored)
        deps["validate_contract"](stored)
        if tuple(stored["visibility"].shape) != (VIEW_COUNT, len(stored["vertices"])):
            raise ValueError(f"{record['sample_id']}: invalid visibility shape")
        output_records.append(
            {
                "sample_id": str(record["sample_id"]),
                "split": str(record["split"]),
                "path": output_path.relative_to(output_root).as_posix(),
            }
        )
        diagnostics.append(
            {
                "sample_id": str(record["sample_id"]),
                "object_id": object_id,
                "split": str(record["split"]),
                "variant_index": int(current["metadata"]["variant_index"]),
                "vertices": int(stored["vertices"].shape[0]),
                "views": len(stored["image_paths"]),
                "all_frozen_tensors_exact": True,
            }
        )

    manifest_path = output_root / "manifest.json"
    write_json(
        manifest_path,
        {
            "format_version": FORMAT_VERSION,
            "dataset_role": "fixed_synthetic_current_28view_runtime_jitter_ablation",
            "object_level_split_enforced": True,
            "view_count": VIEW_COUNT,
            "variants_per_object": 5,
            "samples": output_records,
        },
    )
    split_counts = {
        split: sum(item["split"] == split for item in output_records)
        for split in ("train", "validation", "test")
    }
    validation = {}
    for split in ("train", "validation", "test"):
        dataset = deps["Dataset"].from_manifest(manifest_path, split)
        for item_index in range(len(dataset)):
            deps["prepare"](
                dataset.load_static(item_index), config,
                keep_image_payload=True, keep_projection=True,
            )
        validation[split] = {"samples": len(dataset), "training_contract": "passed"}
    write_json(
        output_root / "summary.json",
        {
            "format_version": FORMAT_VERSION,
            "manifest": str(manifest_path),
            "source_manifests": {
                "synthetic_current": str(current_manifest),
                "views_28": str(views_manifest),
            },
            "controls": {
                "graph_proxy_targets_reused_bit_exact": True,
                "observations_reused_from_nested_views_28": True,
                "visibility_recomputed_per_current_variant": True,
                "visibility_backend": "cuda",
                "cpu_fallback_allowed": False,
            },
            "sample_count": len(output_records),
            "split_counts": split_counts,
            "validation": validation,
            "samples": diagnostics,
        },
    )
    print(json.dumps({"status": "passed", "manifest": str(manifest_path), "split_counts": split_counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
