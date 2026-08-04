from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm import tqdm

from .config import load_config
from .downstream import validate_downstream
from .io_utils import ensure_dir, write_csv, write_json
from .prepare import _build_prepared_sample, _write_contract_report


def finalize_cached_dataset(config_path: str | Path) -> None:
    cfg = load_config(config_path)
    runtime = validate_downstream(cfg.downstream)
    from mlr.datasets import load_reconstruction_input
    from mlr.io import load_mesh
    from mlr.learned_laplacian.dataset import load_prepared_sample, save_prepared_sample
    from mlr.synthetic import SyntheticRenderConfig, generate_synthetic_dataset

    root = Path(cfg.output_root)
    manifest = pd.read_csv(root / "manifest.csv")
    if len(manifest) != 50:
        raise RuntimeError(f"Cached manifest must contain 50 models, found {len(manifest)}")
    split = json.loads((root / "split.json").read_text(encoding="utf-8"))
    if "val" in split:
        split["validation"] = split.pop("val")
    id_to_split = {int(file_id): name for name, ids in split.items() for file_id in ids}
    if sorted(len(split[name]) for name in ("train", "validation", "test")) != [5, 5, 40]:
        raise RuntimeError("Split must contain 40/5/5 train/validation/test models")

    prepared_root = root / cfg.prepared_samples.directory
    ensure_dir(prepared_root)
    records: list[dict[str, str]] = []
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    render_cfg = SyntheticRenderConfig(
        num_views=cfg.views_count, width=cfg.views_width, height=cfg.views_height,
        trajectory="sphere", min_elevation_degrees=-60.0, max_elevation_degrees=60.0,
        render_mode="lit", backend="cpu", normalize_mesh=False,
    )
    for source_row in tqdm(manifest.to_dict(orient="records"), desc="Finalizing cached models"):
        file_id = int(source_row["file_id"])
        model_dir = root / "models" / str(file_id)
        try:
            views_dir = model_dir / "views"
            dataset_json = views_dir / "dataset.json"
            needs_render = True
            if dataset_json.exists():
                try:
                    existing = load_reconstruction_input(dataset_json)
                    needs_render = len(existing.image_paths) != cfg.views_count
                    if not needs_render and existing.image_paths:
                        from PIL import Image
                        needs_render = Image.open(existing.image_paths[0]).size != (cfg.views_width, cfg.views_height)
                except Exception:  # noqa: BLE001
                    needs_render = True
            if needs_render:
                generate_synthetic_dataset(
                    load_mesh(model_dir / "gt_mesh.obj"), views_dir, render_cfg,
                    source_mesh_path=model_dir / "gt_mesh.obj",
                )
            reconstruction = load_reconstruction_input(dataset_json)
            if len(reconstruction.image_paths) != cfg.views_count:
                raise RuntimeError("rendered view count mismatch")

            with np.load(model_dir / "targets.npz") as archive:
                targets = {name: archive[name] for name in archive.files}
            laplacian = sp.load_npz(model_dir / "laplacian.npz")
            sample_id = f"thingi10k_{file_id}"
            sample = _build_prepared_sample(
                dataset_json, model_dir / "expanded_mesh.obj", model_dir / "gt_mesh.obj",
                targets, laplacian, cfg.prepared_samples.image_size, sample_id,
                {
                    "dataset_path": str(dataset_json),
                    "coarse_mesh_path": str(model_dir / "expanded_mesh.obj"),
                    "gt_mesh_path": str(model_dir / "gt_mesh.obj"),
                    "operator_type": "uniform",
                    "target_constructor": "precomputed_closest_surface_on_prediction_graph",
                    "laplacian_target_mode": cfg.prepared_samples.target_mode,
                    "edge_scale_epsilon": cfg.prepared_samples.edge_scale_epsilon,
                    "file_id": file_id, "split": id_to_split[file_id],
                    "downstream_url": runtime.url, "downstream_branch": runtime.branch,
                    "downstream_sha": runtime.sha,
                },
            )
            sample_path = prepared_root / f"{sample_id}.pt"
            save_prepared_sample(sample, sample_path)
            reloaded = load_prepared_sample(sample_path)
            if reloaded["sample_id"] != sample_id:
                raise RuntimeError("saved sample ID mismatch")
            np.savez_compressed(
                model_dir / "laplacian_targets.npz",
                laplacian_target=reloaded["raw_laplacian_target"].numpy().astype(np.float32),
            )
            row = dict(source_row)
            row.update(
                split=id_to_split[file_id], views_count=cfg.views_count,
                views_resolution=f"{cfg.views_width}x{cfg.views_height}", validation_status="valid",
                failure_reason="",
            )
            rows.append(row)
            records.append(
                {"path": str(sample_path.relative_to(root)), "split": id_to_split[file_id], "sample_id": sample_id}
            )
            metrics_path = model_dir / "metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics["manifest_row"] = row
            metrics["validation_status"] = "valid"
            write_json(metrics_path, metrics)
        except Exception as exc:  # noqa: BLE001
            failures.append({"file_id": file_id, "reason": f"finalization_failure: {exc}"})
            write_csv(root / "failed_models.csv", failures)

    if len(rows) != 50:
        raise RuntimeError(f"Expected 50 finalized models, got {len(rows)}; failures={failures}")
    write_csv(root / "manifest.csv", rows)
    write_json(root / "manifest.json", rows)
    write_json(root / cfg.prepared_samples.manifest, {
        "samples": records,
        "downstream": {**asdict(runtime), "root": str(runtime.root)},
    })
    write_json(root / "split.json", split)
    write_csv(root / "split.csv", [
        {"file_id": file_id, "split": name} for name, ids in split.items() for file_id in ids
    ])
    write_csv(root / "failed_models.csv", failures)
    write_json(root / "config.yaml.resolved.json", asdict(cfg))
    _write_contract_report(root, runtime)
