from __future__ import annotations

import json

import numpy as np
import trimesh

from sofa50_refinement.pipeline import (
    MeshData,
    _load_npz_mesh,
    _mesh_issues,
    _parameters,
    _process_one,
    _stable_triangle_correspondence,
    _topology_safe_target,
    _validate_model,
    build_provenance_targets,
    simplify_with_provenance,
)
from thingi10k50_prep.mesh_ops import midpoint_subdivide


def test_provenance_targets_align_with_midpoint_order_and_faces() -> None:
    source = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    gt = MeshData(
        np.asarray(source.vertices, dtype=np.float64),
        np.asarray(source.faces, dtype=np.int64),
    )
    coarse, collapses, original_to_coarse, _ = simplify_with_provenance(
        gt, target_vertices=300, min_vertices=100
    )
    expanded_arrays, subdivision = midpoint_subdivide(coarse.vertices, coarse.faces, steps=1)
    expanded = MeshData(expanded_arrays.vertices, expanded_arrays.faces)
    p_target, mapping, diagnostics = build_provenance_targets(
        gt, coarse, original_to_coarse, expanded, subdivision
    )

    assert len(collapses) > 0
    assert p_target.shape == expanded.vertices.shape
    assert np.isfinite(p_target).all()
    assert diagnostics["mapping_coverage"] == 1.0
    assert np.all(np.isin(mapping["expanded_mapping_kind"], [0, 1, 2, 3]))
    np.testing.assert_allclose(
        mapping["expanded_gt_weights"].sum(axis=1), 1.0, atol=2e-6, rtol=0.0
    )
    np.testing.assert_array_equal(expanded.faces, expanded.faces.copy())
    safe_target, alpha, _ = _topology_safe_target(expanded, p_target)
    assert alpha >= 0.95
    assert not _mesh_issues(MeshData(safe_target, expanded.faces))


def test_thin_triangle_correspondence_is_reconstructible() -> None:
    triangle = np.asarray(
        [
            [-0.88787812, 0.31478441, 0.26878771],
            [0.86925519, 0.31423566, 0.26878771],
            [-0.48902702, 0.31465983, 0.26878771],
        ],
        dtype=np.float64,
    )
    query = np.asarray([0.19011408, 0.31444776, 0.26878771], dtype=np.float64)
    point, weights = _stable_triangle_correspondence(query, triangle)

    assert np.all(weights >= 0.0)
    np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(point, weights @ triangle, atol=1e-12, rtol=0.0)


def test_pipeline_query_is_raw_subdivision_without_registration_leakage(tmp_path) -> None:
    model_id = "synthetic-sofa-like"
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    model_dir = source_root / model_id
    model_dir.mkdir(parents=True)
    source = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    source.export(model_dir / "mesh.obj")
    parameters = _parameters(
        source_root,
        coarse_target_vertices=300,
        coarse_min_vertices=100,
        subdivision_steps=1,
        seed=20260806,
    )

    row, _ = _process_one(
        model_id,
        "train",
        source_root,
        output_root,
        parameters,
        force=True,
    )
    saved_dir = output_root / "models" / model_id
    coarse_raw = _load_npz_mesh(saved_dir / "coarse_raw.npz")
    coarse_registered = _load_npz_mesh(
        saved_dir / "coarse_registered_oracle.npz"
    )
    expanded_raw = _load_npz_mesh(saved_dir / "expanded_initial_raw.npz")
    target_oracle = _load_npz_mesh(saved_dir / "P_target_oracle.npz")
    metadata = json.loads((saved_dir / "metadata.json").read_text(encoding="utf-8"))
    reconstructed_raw, _ = midpoint_subdivide(
        coarse_raw.vertices, coarse_raw.faces, steps=1
    )
    reconstructed_registered, _ = midpoint_subdivide(
        coarse_registered.vertices, coarse_registered.faces, steps=1
    )

    assert row["status"] == "valid"
    np.testing.assert_allclose(
        reconstructed_raw.vertices,
        expanded_raw.vertices,
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_array_equal(reconstructed_raw.faces, expanded_raw.faces)
    np.testing.assert_array_equal(expanded_raw.faces, target_oracle.faces)
    assert expanded_raw.vertices.shape == target_oracle.vertices.shape
    assert metadata["usage_contract"]["may_connect_to_training_loader"] is False
    assert metadata["usage_contract"]["training_target"] == (
        "delta_gt = L(gt_faces) @ gt_vertices"
    )
    assert (
        metadata["usage_contract"][
            "expanded_graph_oracle_laplacian_is_training_supervision"
        ]
        is False
    )
    if not np.allclose(coarse_raw.vertices, coarse_registered.vertices):
        assert not np.allclose(
            expanded_raw.vertices,
            reconstructed_registered.vertices,
            atol=1e-12,
            rtol=0.0,
        )
    assert not _validate_model(saved_dir)
