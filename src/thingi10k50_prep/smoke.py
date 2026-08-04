from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp


def run_smoke_test(data_root: str | Path) -> None:
    root = Path(data_root)
    split = json.loads((root / "split.json").read_text(encoding="utf-8"))
    picks = {
        "train": split["train"][0],
        "val": split["val"][0],
        "test": split["test"][0],
    }

    results = []
    for split_name, file_id in picks.items():
        model_dir = root / "models" / str(file_id)
        expanded = np.load(model_dir / "expanded_mesh.npz")["vertices"]
        lap = sp.load_npz(model_dir / "laplacian.npz")
        targets = np.load(model_dir / "targets.npz")["target_positions"]
        delta = np.load(model_dir / "laplacian_targets.npz")["laplacian_target"]

        pred = lap @ targets
        l2 = float(np.linalg.norm(pred - delta) / np.sqrt(delta.size))
        results.append({"split": split_name, "file_id": int(file_id), "rmse_like": l2})

        if expanded.shape[0] != lap.shape[0]:
            raise RuntimeError(f"Shape mismatch for {split_name}/{file_id}")

    (root / "reports" / "smoke_test.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

