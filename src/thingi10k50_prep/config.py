from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class StratumConfig:
    name: str
    min_faces: int
    max_faces: int
    count: int


@dataclass(frozen=True)
class SplitConfig:
    train: int
    val: int
    test: int


@dataclass(frozen=True)
class DownstreamConfig:
    repo_url: str
    repo_root: str
    branch: str


@dataclass(frozen=True)
class PreparedSampleConfig:
    directory: str
    manifest: str
    image_size: int
    target_mode: str
    edge_scale_epsilon: float


@dataclass(frozen=True)
class PrepareConfig:
    seed: int
    cache_dir: str
    output_root: str
    log_file: str
    force: bool
    max_faces: int
    min_vertices: int
    min_faces: int
    known_corrupt_ids: list[int]
    strata: list[StratumConfig]
    split: SplitConfig
    normalization_mode: str
    coarse_target_vertices: int
    coarse_min_vertices: int
    subdivision_steps: int
    views_count: int
    views_width: int
    views_height: int
    downstream: DownstreamConfig
    prepared_samples: PreparedSampleConfig


def load_config(path: str | Path) -> PrepareConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    strata = [StratumConfig(**item) for item in raw["strata"]]
    split = SplitConfig(**raw["split"])
    downstream = DownstreamConfig(**raw["downstream"])
    prepared_samples = PreparedSampleConfig(**raw["prepared_samples"])
    return PrepareConfig(
        seed=int(raw["seed"]),
        cache_dir=str(raw["cache_dir"]),
        output_root=str(raw["output_root"]),
        log_file=str(raw.get("log_file", "prepare.log")),
        force=bool(raw.get("force", False)),
        max_faces=int(raw["max_faces"]),
        min_vertices=int(raw["min_vertices"]),
        min_faces=int(raw["min_faces"]),
        known_corrupt_ids=[int(x) for x in raw.get("known_corrupt_ids", [])],
        strata=strata,
        split=split,
        normalization_mode=str(raw["normalization"]["mode"]),
        coarse_target_vertices=int(raw["coarse_mesh"]["target_vertices"]),
        coarse_min_vertices=int(raw["coarse_mesh"]["min_vertices"]),
        subdivision_steps=int(raw["subdivision"]["steps"]),
        views_count=int(raw["views"]["count"]),
        views_width=int(raw["views"]["width"]),
        views_height=int(raw["views"]["height"]),
        downstream=downstream,
        prepared_samples=prepared_samples,
    )
