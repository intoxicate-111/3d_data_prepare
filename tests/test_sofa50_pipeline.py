from __future__ import annotations

import json

import numpy as np
import trimesh

from sofa50_prep.pipeline import (
    _clean_mesh,
    _is_sofa,
    _orient_and_normalize,
    _split_ids,
    prepare_sofa50,
)


def test_sofa_classification_prefers_official_super_category() -> None:
    assert _is_sofa({"super-category": "Sofa", "category": "Loveseat Sofa"})
    assert not _is_sofa(
        {
            "super-category": "Pier/Stool",
            "category": "Footstool / Sofastool / Bed End Stool / Stool",
        }
    )
    assert not _is_sofa({"super-category": "Bed", "category": "Couch Bed"})


def test_clean_mesh_removes_duplicate_and_degenerate_faces() -> None:
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [np.nan, 0, 0]],
        dtype=np.float64,
    )
    faces = np.array(
        [[0, 1, 2], [2, 1, 0], [0, 0, 1], [0, 1, 4], [0, 2, 3]],
        dtype=np.int64,
    )
    cleaned, operations = _clean_mesh(
        trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    )

    assert np.isfinite(cleaned.vertices).all()
    assert len(cleaned.faces) == 2
    assert "removed_duplicate_faces" in operations
    assert "removed_repeated_index_faces" in operations
    assert "removed_faces_using_non_finite_vertices" in operations


def test_orientation_normalizes_maximum_extent() -> None:
    mesh = trimesh.creation.box(extents=(4.0, 1.5, 2.0))
    normalized, center, scale, _ = _orient_and_normalize(
        mesh, source_up_axis="z", target_max_extent=2.0
    )

    np.testing.assert_allclose(np.max(normalized.extents), 2.0)
    np.testing.assert_allclose(normalized.bounds.mean(axis=0), np.zeros(3), atol=1e-12)
    assert len(center) == 3
    assert scale == 0.5


def test_fixed_sofa50_split_is_disjoint_and_reproducible() -> None:
    ids = [f"sofa-{index:03d}" for index in range(50)]
    first = _split_ids(ids, seed=20260806)
    second = _split_ids(ids, seed=20260806)

    assert first == second
    assert {name: len(values) for name, values in first.items()} == {
        "train": 40,
        "validation": 5,
        "test": 5,
    }
    assert len(set(first["train"]) | set(first["validation"]) | set(first["test"])) == 50


def test_prepare_sofa_pipeline_end_to_end_on_multicomponent_meshes(tmp_path) -> None:
    future = tmp_path / "downloads" / "3D-FUTURE-model"
    future.mkdir(parents=True)
    records = []
    for index in range(5):
        model_id = f"synthetic-sofa-{index}"
        records.append(
            {
                "model_id": model_id,
                "super-category": "Sofa",
                "category": "Three-seat / Multi-seat Sofa",
            }
        )
        model_dir = future / model_id
        model_dir.mkdir()
        width = 3.2 + 0.25 * index
        depth = 1.15 + 0.12 * index
        seat = trimesh.creation.box(extents=(width, depth, 0.35))
        seat.apply_translation((0.0, 0.0, 0.55))
        back = trimesh.creation.box(extents=(width, 0.22 + 0.03 * index, 1.1))
        back.apply_translation((0.0, 0.5 * depth, 1.15))
        left_arm = trimesh.creation.box(extents=(0.25, depth, 0.75))
        left_arm.apply_translation((-0.5 * width, 0.0, 0.85))
        right_arm = left_arm.copy()
        right_arm.apply_translation((width, 0.0, 0.0))
        sofa = trimesh.util.concatenate((seat, back, left_arm, right_arm))
        sofa.export(model_dir / "raw_model.obj")
    (future / "model_info.json").write_text(json.dumps(records), encoding="utf-8")

    result = prepare_sofa50(
        data_root=tmp_path,
        count=3,
        seed=20260806,
        source_up_axis="z",
    )

    assert result["raw_sofas_found"] == 5
    assert result["validation"]["valid_meshes"] == 3
    assert (tmp_path / "sofa50" / "contact_sheet.png").is_file()
    assert len(list((tmp_path / "all_sofas").glob("*/raw_model.obj"))) == 5
    assert len(list((tmp_path / "sofa50").glob("*/mesh.obj"))) == 3
