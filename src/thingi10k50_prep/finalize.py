from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from tqdm import tqdm

from .config import load_config
from .downstream import validate_downstream
from .io_utils import ensure_dir, write_csv, write_json
from .prepare import _build_prepared_sample, _expected_model_count, _mesh_checksum, _write_contract_report
from .rendering import generate_configured_synthetic_dataset
from .rendering import CUBE_SURFACE_VIEW_NAMES, VIEW_LAYOUT_VERSION


def finalize_cached_dataset(config_path: str | Path) -> None:
    cfg = load_config(config_path)

    expected_total = _expected_model_count(cfg)

    expected_split_counts = {
        "train": cfg.split.train,
        "validation": cfg.split.val,
        "test": cfg.split.test,
    }

    runtime = validate_downstream(cfg.downstream)
    from mlr.datasets import load_reconstruction_input
    from mlr.io import load_mesh
    from mlr.learned_laplacian.dataset import save_prepared_sample
    from mlr.synthetic import SyntheticRenderConfig

    root = Path(cfg.output_root)
    manifest = pd.read_csv(root / "manifest.csv")
    if len(manifest) != expected_total:
        raise RuntimeError(f"Cached manifest must contain {expected_total} models, found {len(manifest)}")
    split = json.loads((root / "split.json").read_text(encoding="utf-8"))
    if "val" in split:
        split["validation"] = split.pop("val")
    id_to_split = {int(file_id): name for name, ids in split.items() for file_id in ids}
    actual_split_counts = {
        name: len(split[name])
        for name in (
            "train",
            "validation",
            "test",
        )
    }

    if actual_split_counts != expected_split_counts:
        raise RuntimeError(
            "Split counts do not match config: "
            f"expected={expected_split_counts}, "
            f"actual={actual_split_counts}"
        )

    prepared_root = root / cfg.prepared_samples.directory
    ensure_dir(prepared_root)
    records: list[dict[str, str]] = []
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    render_cfg = SyntheticRenderConfig(
        num_views=cfg.views_count, width=cfg.views_width, height=cfg.views_height,
        trajectory=cfg.views_trajectory, min_elevation_degrees=-60.0, max_elevation_degrees=60.0,
        fov_degrees=cfg.views_fov_degrees, render_mode=cfg.views_render_mode,
        backend=cfg.views_backend, normalize_mesh=False,
        opengl_context_backend=cfg.views_opengl_context_backend,
        cube_half_extent=cfg.views_cube_half_extent,
        antialiasing=cfg.views_antialiasing,
        camera_layout_version=VIEW_LAYOUT_VERSION,
    )
    for source_row in tqdm(manifest.to_dict(orient="records"), desc="Finalizing cached models"):
        file_id = int(source_row["file_id"])
        model_dir = root / "models" / str(file_id)
        try:
            normalization = json.loads((model_dir / "normalization.json").read_text(encoding="utf-8"))
            if normalization.get("normalization_mode") != cfg.normalization_mode:
                raise RuntimeError(
                    "geometry cache normalization mode is stale; rerun prepare before finalize"
                )
            if float(normalization.get("normalization_epsilon", -1.0)) != cfg.normalization_epsilon:
                raise RuntimeError(
                    "geometry cache normalization epsilon is stale; rerun prepare before finalize"
                )
            views_dir = model_dir / "views"
            dataset_json = views_dir / "dataset.json"
            needs_render = True
            if dataset_json.exists():
                try:
                    existing = load_reconstruction_input(dataset_json)
                    needs_render = len(existing.image_paths) != cfg.views_count
                    render_metadata = json.loads(dataset_json.read_text(encoding="utf-8")).get("config", {})
                    needs_render = needs_render or render_metadata.get("trajectory") != cfg.views_trajectory
                    needs_render = needs_render or render_metadata.get(
                        "requested_backend", render_metadata.get("backend")
                    ) != cfg.views_backend
                    expected_render_metadata = {
                        "opengl_context_backend": cfg.views_opengl_context_backend,
                        "cube_half_extent": cfg.views_cube_half_extent,
                        "fov_degrees": cfg.views_fov_degrees,
                        "render_mode": cfg.views_render_mode,
                        "antialiasing": cfg.views_antialiasing,
                        "camera_layout_version": VIEW_LAYOUT_VERSION,
                    }
                    needs_render = needs_render or any(
                        render_metadata.get(key) != value
                        for key, value in expected_render_metadata.items()
                    )
                    gt_mesh = load_mesh(model_dir / "gt_mesh.obj")
                    needs_render = needs_render or render_metadata.get(
                        "normalized_mesh_checksum"
                    ) != _mesh_checksum(gt_mesh.vertices, gt_mesh.faces)
                    if not needs_render and existing.image_paths:
                        from PIL import Image
                        needs_render = Image.open(existing.image_paths[0]).size != (cfg.views_width, cfg.views_height)
                except Exception:  # noqa: BLE001
                    needs_render = True
            if needs_render:
                generate_configured_synthetic_dataset(
                    load_mesh(model_dir / "gt_mesh.obj"), views_dir, render_cfg,
                    source_mesh_path=model_dir / "gt_mesh.obj",
                )
            reconstruction = load_reconstruction_input(dataset_json)
            render_metadata = json.loads(dataset_json.read_text(encoding="utf-8")).get("config", {})
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
                    "normalization_mode": cfg.normalization_mode,
                    "normalization_center": normalization["normalization_center"],
                    "normalization_scale": normalization["normalization_scale"],
                    "view_count": cfg.views_count,
                    "view_trajectory": cfg.views_trajectory,
                    "view_backend": cfg.views_backend,
                    "opengl_context_backend": cfg.views_opengl_context_backend,
                    "cube_half_extent": cfg.views_cube_half_extent,
                    "fov_degrees": cfg.views_fov_degrees,
                    "view_width": cfg.views_width,
                    "view_height": cfg.views_height,
                    "view_layout_version": VIEW_LAYOUT_VERSION,
                    "view_names": list(CUBE_SURFACE_VIEW_NAMES),
                    "prepared_storage_format": cfg.prepared_samples.storage_format,
                    "downstream_url": runtime.url, "downstream_branch": runtime.branch,
                    "downstream_sha": runtime.sha,
                },
                dataset_root=root,
            )
            sample_path = prepared_root / f"{sample_id}.pt"
            save_prepared_sample(sample, sample_path)
            raw_saved = torch.load(sample_path, map_location="cpu", weights_only=False)
            if raw_saved["sample_id"] != sample_id:
                raise RuntimeError("saved sample ID mismatch")
            if "images" in raw_saved or raw_saved.get("prepared_storage_format") != "lazy_image_paths_v1":
                raise RuntimeError("saved sample does not use lazy_image_paths_v1 storage")
            np.savez_compressed(
                model_dir / "laplacian_targets.npz",
                laplacian_target=sample["raw_laplacian_target"].numpy().astype(np.float32),
            )
            row = dict(source_row)
            row.update(
                split=id_to_split[file_id], views_count=cfg.views_count,
                views_resolution=f"{cfg.views_width}x{cfg.views_height}", validation_status="valid",
                normalization_mode=cfg.normalization_mode,
                views_trajectory=cfg.views_trajectory,
                views_backend=render_metadata.get("backend", cfg.views_backend),
                cube_half_extent=cfg.views_cube_half_extent, fov_degrees=cfg.views_fov_degrees,
                view_layout_version=VIEW_LAYOUT_VERSION,
                prepared_image_size=cfg.prepared_samples.image_size,
                prepared_storage_format=cfg.prepared_samples.storage_format,
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

    if len(rows) != expected_total:
        raise RuntimeError(
            f"Expected {expected_total} finalized models, "
            f"got {len(rows)}; failures={failures}"
        )
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
