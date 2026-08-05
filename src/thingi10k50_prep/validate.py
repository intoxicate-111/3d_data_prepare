from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from mlr.datasets import load_reconstruction_input
from mlr.io import load_mesh
from mlr.coarse_lap_oracle import apply_uniform_laplacian, build_uniform_laplacian_data
from mlr.learned_laplacian.dataset import load_prepared_sample
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset, validate_disjoint_splits

from .io_utils import write_csv


def _load_npz_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return np.asarray(data["vertices"]), np.asarray(data["faces"])


def _check_mesh(vertices: np.ndarray, faces: np.ndarray) -> list[str]:
    issues: list[str] = []
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        issues.append("invalid_vertex_shape")
    if faces.ndim != 2 or faces.shape[1] != 3:
        issues.append("invalid_face_shape")
    if not np.isfinite(vertices).all():
        issues.append("non_finite_vertices")
    if np.min(faces) < 0 or np.max(faces) >= len(vertices):
        issues.append("invalid_face_indices")
    repeated_idx = (faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2]) | (faces[:, 0] == faces[:, 2])
    if np.any(repeated_idx):
        issues.append("repeated_indices_in_face")
    tri = vertices[faces]
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) * 0.5
    if np.any(area <= 0):
        issues.append("non_positive_triangle_area")
    if len(np.unique(faces)) != len(vertices):
        issues.append("unreferenced_vertices")
    return issues


def validate_dataset(data_root: str | Path) -> None:
    root = Path(data_root)
    manifest_path = root / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    rows: list[dict[str, Any]] = []
    for file_id in manifest["file_id"].astype(int).tolist():
        model_dir = root / "models" / str(file_id)
        issues: list[str] = []
        try:
            gt_v, gt_f = _load_npz_mesh(model_dir / "gt_mesh.npz")
            coarse_v, coarse_f = _load_npz_mesh(model_dir / "coarse_mesh.npz")
            exp_v, exp_f = _load_npz_mesh(model_dir / "expanded_mesh.npz")

            issues.extend(_check_mesh(gt_v, gt_f))
            issues.extend(_check_mesh(coarse_v, coarse_f))
            issues.extend(_check_mesh(exp_v, exp_f))

            if np.max(np.abs(gt_v)) > 1.01:
                issues.append("normalized_gt_out_of_expected_range")

            targets = np.load(model_dir / "targets.npz")
            lap = sp.load_npz(model_dir / "laplacian.npz")
            delta = np.load(model_dir / "laplacian_targets.npz")["laplacian_target"]
            target_positions = targets["target_positions"]
            distances = targets["surface_distance"]
            closest_faces = targets["closest_face_indices"]
            bary = targets["closest_barycentric_coordinates"]
            valid_mask = targets["valid_mask"]

            if target_positions.shape != (len(exp_v), 3):
                issues.append("target_positions_shape_mismatch")
            if delta.shape != (len(exp_v), 3):
                issues.append("laplacian_target_shape_mismatch")
            if lap.shape != (len(exp_v), len(exp_v)):
                issues.append("laplacian_shape_mismatch")
            if not np.isfinite(distances).all():
                issues.append("non_finite_projection_distance")
            if not np.isfinite(target_positions).all():
                issues.append("non_finite_target_positions")
            if np.any(closest_faces < 0) or np.any(closest_faces >= len(gt_f)):
                issues.append("invalid_closest_face_indices")
            if np.any(np.abs(bary.sum(axis=1) - 1.0) > 1e-5):
                issues.append("barycentric_sum_mismatch")
            if not np.isfinite(bary).all():
                issues.append("non_finite_barycentric_coordinates")
            if valid_mask.shape != (len(exp_v),):
                issues.append("valid_mask_shape_mismatch")
            if valid_mask.dtype.kind not in "bu" or not np.all(np.isin(valid_mask, [0, 1])):
                issues.append("invalid_valid_mask")
            expected_valid = (
                np.isfinite(target_positions).all(axis=1)
                & np.isfinite(distances)
                & (closest_faces >= 0)
                & (closest_faces < len(gt_f))
                & np.isfinite(bary).all(axis=1)
            )
            if not np.array_equal(valid_mask.astype(bool), expected_valid) or not np.all(expected_valid):
                issues.append("valid_mask_semantics_mismatch")
            if not np.isfinite(lap.data).all():
                issues.append("non_finite_laplacian")

            dataset_json = model_dir / "views" / "dataset.json"
            if not dataset_json.exists():
                issues.append("missing_views_dataset_json")
            else:
                dataset = load_reconstruction_input(dataset_json)
                expected_views = int(manifest.loc[manifest["file_id"] == file_id, "views_count"].iloc[0])
                if len(dataset.image_paths) != expected_views:
                    issues.append("unexpected_view_count")
                if not (model_dir / "views" / "cameras.json").exists():
                    issues.append("missing_cameras_json")
                if not (model_dir / "views" / "mesh.obj").exists():
                    issues.append("missing_view_mesh_obj")

            for mesh_name in ("gt_mesh.obj", "coarse_mesh.obj", "expanded_mesh.obj"):
                try:
                    load_mesh(model_dir / mesh_name)
                except Exception as exc:  # noqa: BLE001
                    issues.append(f"load_mesh_failed:{mesh_name}:{exc}")

            downstream_data = build_uniform_laplacian_data(exp_f, len(exp_v))
            probes = np.random.default_rng(1949 + file_id).standard_normal((len(exp_v), 3))
            if not np.allclose(lap @ probes, apply_uniform_laplacian(probes, downstream_data), rtol=0.0, atol=1e-12):
                issues.append("laplacian_contract_mismatch")
            expected_delta = apply_uniform_laplacian(target_positions, downstream_data)
            if not np.allclose(delta, expected_delta, rtol=0.0, atol=2e-7):
                issues.append("laplacian_target_contract_mismatch")

            prepared = load_prepared_sample(root / "prepared" / f"thingi10k_{file_id}.pt")
            required_extra = (
                "raw_laplacian_target", "normalized_laplacian_target", "valid_scale_mask",
                "local_edge_length", "local_edge_scale", "target_positions", "gt_vertices", "gt_faces",
            )
            if any(name not in prepared for name in required_extra):
                issues.append("prepared_sample_missing_extended_fields")
            if prepared["sample_id"] != f"thingi10k_{file_id}":
                issues.append("prepared_sample_id_mismatch")
            if not np.allclose(prepared["raw_laplacian_target"].numpy(), delta, atol=2e-7, rtol=0.0):
                issues.append("prepared_raw_target_mismatch")
        except Exception as exc:  # noqa: BLE001
            issues.append(f"validation_exception:{exc}")

        rows.append(
            {
                "file_id": file_id,
                "status": "valid" if not issues else "invalid",
                "issues": "|".join(sorted(set(issues))),
            }
        )

    write_csv(root / "reports" / "validation_summary.csv", rows)
    invalid = [r for r in rows if r["status"] != "valid"]
    if invalid:
        raise RuntimeError(f"Validation failed for {len(invalid)} models")

    split = json.loads((root / "split.json").read_text(encoding="utf-8"))

    split_names = ("train", "validation", "test")

    expected_split_counts = {
        name: len(split[name])
        for name in split_names
    }

    expected_total = sum(expected_split_counts.values())

    if len(manifest) != expected_total:
        raise RuntimeError(
            f"Expected {expected_total} manifest rows, "
            f"found {len(manifest)}"
        )


    prepared_manifest = root / "prepared_manifest.json"

    datasets = [
        PreparedMeshDataset.from_manifest(
            prepared_manifest,
            split_name,
        )
        for split_name in split_names
    ]

    validate_disjoint_splits(*datasets)

    actual_split_counts = {
        split_name: len(dataset)
        for split_name, dataset in zip(split_names, datasets)
    }

    if actual_split_counts != expected_split_counts:
        raise RuntimeError(
            "Prepared manifest split counts do not match split.json: "
            f"expected={expected_split_counts}, "
            f"actual={actual_split_counts}"
        )

    report = {"total_models": len(rows), "valid_models": len(rows), "invalid_models": 0}
    (root / "reports" / "validation_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
