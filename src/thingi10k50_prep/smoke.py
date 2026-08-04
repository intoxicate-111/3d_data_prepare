from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from mlr.datasets import load_reconstruction_input
from mlr.io import load_mesh, save_mesh
from mlr.laplacian import build_uniform_laplacian as mlr_build_uniform_laplacian
from mlr.oracle import OracleBaselineConfig, run_oracle_baselines


def run_smoke_test(data_root: str | Path) -> None:
    root = Path(data_root)
    split = json.loads((root / "split.json").read_text(encoding="utf-8"))
    picks = {
        "train": split["train"][0],
        "val": split["val"][0],
        "test": split["test"][0],
    }

    mlr_repo_root = Path("/home/zhou_c_WMGDS.WMG.WARWICK.AC.UK/multiview-laplacian-refinement")
    downstream_commit = subprocess.check_output(["git", "-C", str(mlr_repo_root), "rev-parse", "HEAD"], text=True).strip()

    results: list[dict[str, object]] = []
    for split_name, file_id in picks.items():
        model_dir = root / "models" / str(file_id)
        expanded = np.load(model_dir / "expanded_mesh.npz")["vertices"]
        lap = sp.load_npz(model_dir / "laplacian.npz")
        targets = np.load(model_dir / "targets.npz")["target_positions"]
        delta = np.load(model_dir / "laplacian_targets.npz")["laplacian_target"]

        pred = lap @ targets
        l2 = float(np.linalg.norm(pred - delta) / np.sqrt(delta.size))

        dataset_json = model_dir / "views" / "dataset.json"
        if not dataset_json.exists():
            raise FileNotFoundError(f"Missing downstream views dataset: {dataset_json}")
        dataset = load_reconstruction_input(dataset_json)
        gt_mesh = load_mesh(model_dir / "gt_mesh.obj")
        coarse_mesh = load_mesh(model_dir / "coarse_mesh.obj")
        expanded_mesh = load_mesh(model_dir / "expanded_mesh.obj")

        expected_lap = mlr_build_uniform_laplacian(expanded_mesh.faces, len(expanded_mesh.vertices))
        np.testing.assert_allclose(lap.toarray(), expected_lap, rtol=0.0, atol=1e-12)

        expected_delta = expected_lap @ targets
        np.testing.assert_allclose(delta, expected_delta, rtol=0.0, atol=1e-12)

        if expanded.shape[0] != lap.shape[0]:
            raise RuntimeError(f"Shape mismatch for {split_name}/{file_id}")

        oracle = run_oracle_baselines(
            gt_mesh,
            gt_mesh.vertices,
            config=OracleBaselineConfig(operator_type="uniform", num_iters=2, learning_rate=1e-3),
        )
        oracle_result = oracle["position_plus_laplacian"]
        if not np.isfinite(oracle_result.history[-1]["loss"]):
            raise RuntimeError(f"Non-finite oracle loss for {split_name}/{file_id}")
        save_mesh(oracle_result.mesh, root / "reports" / f"smoke_{split_name}_{file_id}.obj")

        if len(dataset.image_paths) != 40:
            raise RuntimeError(f"Unexpected view count for {split_name}/{file_id}")
        if coarse_mesh.vertices.shape[0] == 0 or expanded_mesh.vertices.shape[0] == 0:
            raise RuntimeError(f"Empty coarse/expanded geometry for {split_name}/{file_id}")

        results.append(
            {
                "split": split_name,
                "file_id": int(file_id),
                "rmse_like": l2,
                "views": len(dataset.image_paths),
                "downstream_commit": downstream_commit,
                "oracle_loss": float(oracle_result.history[-1]["loss"]),
            }
        )

    (root / "reports" / "smoke_test.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

