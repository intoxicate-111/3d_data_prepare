from __future__ import annotations

import json

import numpy as np
import trimesh

from future2000_prep.pipeline import (
    INFERENCE_FORMAT_VERSION,
    SUPER_CATEGORIES,
    MeshData,
    _balanced_splits,
    _save_mesh,
    _simplify_raw_componentwise,
    validate_future2000_inference,
)
from thingi10k50_prep.mesh_ops import midpoint_subdivide


def test_balanced_2000_split_is_category_stratified_and_reproducible() -> None:
    selected = {
        category: [
            {"model_id": f"{category}-{index:03d}"}
            for index in range(250)
        ]
        for category in SUPER_CATEGORIES
    }

    first = _balanced_splits(selected, 20260806)
    second = _balanced_splits(selected, 20260806)

    assert first == second
    assert {split: len(values) for split, values in first.items()} == {
        "train": 1600,
        "validation": 200,
        "test": 200,
    }
    assert len({model_id for values in first.values() for model_id in values}) == 2000
    for category in SUPER_CATEGORIES:
        assert sum(value.startswith(f"{category}-") for value in first["train"]) == 200
        assert sum(value.startswith(f"{category}-") for value in first["validation"]) == 25
        assert sum(value.startswith(f"{category}-") for value in first["test"]) == 25


def test_inference_query_is_exact_raw_coarse_midpoint_subdivision(tmp_path) -> None:
    source = trimesh.creation.icosphere(subdivisions=2)
    gt = MeshData(np.asarray(source.vertices), np.asarray(source.faces))
    coarse, _ = _simplify_raw_componentwise(gt, target_vertices=40, min_vertices=4)
    expanded_arrays, mapping = midpoint_subdivide(coarse.vertices, coarse.faces, steps=1)
    expanded = MeshData(expanded_arrays.vertices, expanded_arrays.faces)
    model_dir = tmp_path / "models" / "sample"
    gt_files = _save_mesh(model_dir / "gt_mesh", gt)
    coarse_files = _save_mesh(model_dir / "coarse_raw", coarse)
    expanded_files = _save_mesh(model_dir / "expanded_initial_raw", expanded)
    mapping_path = model_dir / "subdivision_mapping_raw.npz"
    np.savez_compressed(mapping_path, **mapping)
    record = {
        "model_id": "sample",
        "split": "train",
        "gt_npz": gt_files["npz"],
        "coarse_raw_npz": coarse_files["npz"],
        "expanded_initial_raw_npz": expanded_files["npz"],
        "subdivision_mapping_raw_npz": str(mapping_path),
    }
    (model_dir / "metadata.json").write_text(
        json.dumps(
            {
                "format_version": INFERENCE_FORMAT_VERSION,
                "gt_guided_query_modification": False,
                "oracle_target_generated": False,
                "coarse_generation_policy": (
                    "componentwise_direct_qem_then_component_local_cleanup"
                ),
            }
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"samples": [record]}), encoding="utf-8")

    result = validate_future2000_inference(manifest_path, expected_count=1)

    assert result["valid_count"] == 1
    assert result["exact_midpoint_reconstruction_count"] == 1
