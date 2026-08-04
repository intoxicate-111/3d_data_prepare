from __future__ import annotations

import importlib
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import DownstreamConfig


@dataclass(frozen=True)
class DownstreamRuntime:
    root: Path
    url: str
    branch: str
    sha: str


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def validate_downstream(config: DownstreamConfig) -> DownstreamRuntime:
    root = Path(config.repo_root).expanduser().resolve()
    if not (root / ".git").exists():
        raise RuntimeError(f"Downstream is not a git repository: {root}")
    actual_url = _git(root, "remote", "get-url", "origin")
    actual_branch = _git(root, "branch", "--show-current")
    if actual_url != config.repo_url:
        raise RuntimeError(f"Downstream origin mismatch: expected {config.repo_url!r}, got {actual_url!r}")
    if actual_branch != config.branch:
        raise RuntimeError(f"Downstream branch mismatch: expected {config.branch!r}, got {actual_branch!r}")

    required = (
        "src/mlr/learned_laplacian/dataset.py",
        "src/mlr/learned_laplacian/sample_io.py",
        "src/mlr/learned_laplacian/target_scaling.py",
        "src/mlr/learned_laplacian/model.py",
        "src/mlr/learned_laplacian/trainer.py",
        "src/mlr/learned_laplacian/evaluation.py",
        "src/mlr/learned_laplacian/multi_trainer.py",
    )
    missing = [rel for rel in required if not (root / rel).is_file()]
    if missing:
        raise RuntimeError(f"Downstream learned pipeline is incomplete: {missing}")

    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    contracts = {
        "mlr.learned_laplacian.dataset": ("load_prepared_sample", "save_prepared_sample", "validate_sample"),
        "mlr.learned_laplacian.sample_io": ("prepare_single_object_sample",),
        "mlr.learned_laplacian.target_scaling": (
            "normalize_laplacian_by_edge_scale", "denormalize_laplacian_by_edge_scale"
        ),
        "mlr.learned_laplacian.model": ("LearnedLaplacianModel",),
        "mlr.learned_laplacian.trainer": ("train_single_object",),
        "mlr.learned_laplacian.multi_trainer": ("train_multi_object",),
        "mlr.learned_laplacian.evaluation": ("reconstruct_and_evaluate",),
    }
    for module_name, names in contracts.items():
        module = importlib.import_module(module_name)
        absent = [name for name in names if not hasattr(module, name)]
        if absent:
            raise RuntimeError(f"Downstream contract missing {module_name}: {absent}")
    return DownstreamRuntime(root, actual_url, actual_branch, _git(root, "rev-parse", "HEAD"))
