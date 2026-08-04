from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

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
            if target_positions.shape != (len(exp_v), 3):
                issues.append("target_positions_shape_mismatch")
            if delta.shape != (len(exp_v), 3):
                issues.append("laplacian_target_shape_mismatch")
            if lap.shape != (len(exp_v), len(exp_v)):
                issues.append("laplacian_shape_mismatch")
            if not np.isfinite(distances).all():
                issues.append("non_finite_projection_distance")
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

    report = {"total_models": len(rows), "valid_models": len(rows), "invalid_models": 0}
    (root / "reports" / "validation_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

