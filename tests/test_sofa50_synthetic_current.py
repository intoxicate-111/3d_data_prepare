from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch


def _load_script():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "prepare_sofa50_synthetic_current.py"
    )
    spec = importlib.util.spec_from_file_location("prepare_sofa50_synthetic_current", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generator = _load_script()


def _source() -> dict:
    vertices = torch.tensor(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    faces = torch.tensor(
        [
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
        ],
        dtype=torch.long,
    )
    visibility = torch.ones((14, len(vertices)), dtype=torch.bool)
    return {
        "sample_id": "object-a",
        "image_paths": [f"rendered/images/{index:04d}.png" for index in range(14)],
        "prepared_storage_format": "lazy_image_paths_v1",
        "source_image_size": [960, 960],
        "prepared_image_size": 960,
        "intrinsics": torch.eye(3).repeat(14, 1, 1),
        "extrinsics": torch.eye(4).repeat(14, 1, 1),
        "vertices": vertices.clone(),
        "faces": faces.clone(),
        "gt_vertices": vertices.clone(),
        "gt_faces": faces.clone(),
        "visibility": visibility.clone(),
        "visibility_backface_only": visibility.clone(),
        "visibility_occlusion_only": visibility.clone(),
        "visibility_backface_and_occlusion": visibility.clone(),
    }


def test_current_target_uses_current_graph_and_proxy(tmp_path: Path) -> None:
    downstream = Path(__file__).resolve().parents[2] / "multiview-laplacian-refinement"
    deps = generator._dependencies(downstream)
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    sample, oracle = generator.build_current_sample(
        _source(),
        object_id="object-a",
        split="train",
        variant_index=0,
        base_seed=7,
        perturb_std_h=0.15,
        smooth_iterations=2,
        neighbor_weight=0.5,
        output_root=output_root,
        source_root=source_root,
        visibility_backend=None,
        visibility_artifact=None,
        deps=deps,
    )

    lap = deps["build_uniform_laplacian_data"](
        sample["faces"].numpy(), len(sample["vertices"])
    )
    expected = deps["apply_uniform_laplacian"](
        sample["target_positions"].double().numpy(), lap
    )
    np.testing.assert_allclose(sample["raw_laplacian_target"].numpy(), expected, atol=1e-7)
    assert not torch.equal(sample["vertices"], sample["target_positions"])
    assert torch.count_nonzero(sample["initial_laplacian"]).item() > 0
    assert oracle["target_contract_pass"]
    assert oracle["normalization_roundtrip_pass"]
    assert sample["metadata"]["source_split"] == "train"
    assert sample["metadata"]["variant_index"] == 0


def test_five_variants_are_deterministic_and_distinct(tmp_path: Path) -> None:
    downstream = Path(__file__).resolve().parents[2] / "multiview-laplacian-refinement"
    deps = generator._dependencies(downstream)
    source = _source()
    variants = []
    for index in range(5):
        sample, _ = generator.build_current_sample(
            source,
            object_id="object-a",
            split="validation",
            variant_index=index,
            base_seed=7,
            perturb_std_h=0.15,
            smooth_iterations=2,
            neighbor_weight=0.5,
            output_root=tmp_path / "output",
            source_root=tmp_path / "source",
            visibility_backend=None,
            visibility_artifact=None,
            deps=deps,
        )
        variants.append(sample)
    assert len({sample["sample_id"] for sample in variants}) == 5
    assert len({sample["metadata"]["variant_seed"] for sample in variants}) == 5
    assert all(sample["metadata"]["source_split"] == "validation" for sample in variants)
    assert not torch.equal(variants[0]["vertices"], variants[1]["vertices"])

    repeated, _ = generator.build_current_sample(
        source,
        object_id="object-a",
        split="validation",
        variant_index=0,
        base_seed=7,
        perturb_std_h=0.15,
        smooth_iterations=2,
        neighbor_weight=0.5,
        output_root=tmp_path / "output",
        source_root=tmp_path / "source",
        visibility_backend=None,
        visibility_artifact=None,
        deps=deps,
    )
    assert torch.equal(variants[0]["vertices"], repeated["vertices"])
