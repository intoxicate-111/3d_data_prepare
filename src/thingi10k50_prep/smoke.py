from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from mlr.learned_laplacian.dataset import load_prepared_sample, validate_sample
from mlr.learned_laplacian.evaluation import reconstruct_and_evaluate
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset, validate_disjoint_splits
from mlr.learned_laplacian.multi_trainer import train_multi_object
from mlr.learned_laplacian.target_scaling import denormalize_laplacian_by_edge_scale
from mlr.learned_laplacian.trainer import train_single_object


def _small_real_sample(sample: dict, size: int = 64) -> dict:
    """Keep real geometry/camera data while making the CPU smoke test bounded."""
    result = dict(sample)
    old_h, old_w = sample["images"].shape[-2:]
    result["images"] = functional.interpolate(
        sample["images"][:1], size=(size, size), mode="bilinear", align_corners=False
    )
    result["intrinsics"] = sample["intrinsics"][:1].clone()
    result["intrinsics"][:, 0, :] *= size / old_w
    result["intrinsics"][:, 1, :] *= size / old_h
    result["extrinsics"] = sample["extrinsics"][:1]
    if sample.get("visibility") is not None:
        result["visibility"] = sample["visibility"][:1]
    return validate_sample(result)


def _training_config(seed: int) -> dict:
    return {
        "seed": seed,
        "device": "cpu",
        "input_mode": "coarse_plus_multiview",
        "target_mode": "edge_scale_normalized_laplacian",
        "target_scaling": {"method": "square_of_mean_incident_edge_length", "epsilon": 1e-12},
        "image_encoder": {"feature_dim": 4},
        "model": {"hidden_dim": 8, "num_graph_layers": 1, "dropout": 0.0},
        "training": {"steps": 1, "learning_rate": 1e-4, "loss": "huber", "huber_delta": 0.01},
        "multi_object_training": {
            "epochs": 1, "gradient_accumulation_meshes": 1,
            "validation_every_epochs": 1, "shuffle": False,
        },
    }


def run_smoke_test(data_root: str | Path) -> None:
    root = Path(data_root)
    manifest_path = root / "prepared_manifest.json"
    datasets = {
        name: PreparedMeshDataset.from_manifest(manifest_path, name)
        for name in ("train", "validation", "test")
    }
    validate_disjoint_splits(*datasets.values())
    reports = root / "reports" / "learned_smoke"
    reports.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []

    for index, split_name in enumerate(("train", "validation", "test")):
        record = datasets[split_name].records[0]
        loaded = load_prepared_sample(record.path)
        sample = _small_real_sample(loaded)
        trained = train_single_object(
            sample, _training_config(900 + index), output_dir=reports / split_name,
            device_override="cpu", progress=False,
        )
        trained.model.eval()
        with torch.no_grad():
            normalized = trained.model(sample).predicted_laplacian
            raw = denormalize_laplacian_by_edge_scale(normalized, sample["local_edge_length"])
        if not torch.isfinite(raw).all() or not np.isfinite(trained.final_loss):
            raise FloatingPointError(f"Non-finite learned smoke result for {split_name}")
        evaluation = reconstruct_and_evaluate(
            sample, raw, reports / split_name / "reconstruction",
            {"operator_type": "uniform", "num_iters": 1, "learning_rate": 1e-3,
             "dense_vertex_limit": 5000, "chamfer_samples": 64, "metric_seed": 77},
            normalized_prediction=normalized,
        )
        if not evaluation["reconstruction"]["all_finite"]:
            raise FloatingPointError(f"Non-finite reconstruction for {split_name}")
        results.append({
            "split": split_name, "sample_id": sample["sample_id"],
            "vertices": int(sample["vertices"].shape[0]), "initial_loss": trained.initial_loss,
            "final_loss": trained.final_loss, "evaluation": evaluation,
        })

    train_samples = [_small_real_sample(datasets["train"][i]) for i in range(2)]
    if train_samples[0]["vertices"].shape[0] == train_samples[1]["vertices"].shape[0]:
        raise RuntimeError("Multi-mesh smoke requires two variable-topology meshes")
    validation_samples = [_small_real_sample(datasets["validation"][0])]
    multi = train_multi_object(
        train_samples, validation_samples, _training_config(1201),
        output_dir=reports / "multi_object", device_override="cpu",
        input_mode_override="coarse_only", zero_images=True, progress=False,
    )
    if not np.isfinite(multi.final_train_loss) or not np.isfinite(multi.final_validation_loss):
        raise FloatingPointError("Non-finite multi-object smoke result")
    payload = {
        "single_object": results,
        "multi_object": {
            "train_sample_ids": [sample["sample_id"] for sample in train_samples],
            "validation_sample_ids": [sample["sample_id"] for sample in validation_samples],
            "vertex_counts": [int(sample["vertices"].shape[0]) for sample in train_samples],
            "optimizer_steps": multi.optimizer_steps,
            "final_train_loss": multi.final_train_loss,
            "final_validation_loss": multi.final_validation_loss,
        },
    }
    (root / "reports" / "smoke_test.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
