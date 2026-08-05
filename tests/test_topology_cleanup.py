from __future__ import annotations

import numpy as np

from thingi10k50_prep.mesh_ops import midpoint_subdivide, normalize_vertices, remove_unreferenced_vertices


def test_remove_unreferenced_vertices_remaps_faces() -> None:
    vertices = np.arange(15, dtype=np.float64).reshape(5, 3)
    faces = np.array([[0, 2, 4]], dtype=np.int64)
    compact, old_to_new, final_to_old = remove_unreferenced_vertices(vertices, faces)
    np.testing.assert_array_equal(compact.faces, [[0, 1, 2]])
    np.testing.assert_array_equal(final_to_old, [0, 2, 4])
    np.testing.assert_array_equal(old_to_new, [0, -1, 1, -1, 2])


def test_midpoint_mapping_uses_final_compacted_indices() -> None:
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [8, 8, 8]], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    expanded, mapping = midpoint_subdivide(vertices, faces, steps=1)
    assert len(np.unique(expanded.faces)) == len(expanded.vertices)
    assert mapping["parent_edges"].min() >= 0
    assert mapping["parent_edges"].max() < len(expanded.vertices)
    assert mapping["new_vertex_indices"].max() < len(expanded.vertices)


def test_unit_sphere_normalization_is_centered_bounded_and_reversible() -> None:
    vertices = np.array(
        [[8.0, -3.0, 2.0], [12.0, -3.0, 2.0], [9.0, 5.0, -1.0], [11.0, 1.0, 7.0]],
        dtype=np.float64,
    )
    normalized, metadata = normalize_vertices(vertices, epsilon=1e-12)
    radii = np.linalg.norm(normalized, axis=1)
    assert radii.max() <= 1.0 + 1e-6
    assert radii.max() >= 1.0 - 1e-5
    np.testing.assert_allclose(
        0.5 * (normalized.min(axis=0) + normalized.max(axis=0)), np.zeros(3), atol=1e-7
    )
    restored = normalized * metadata["normalization_scale"] + metadata["normalization_center"]
    np.testing.assert_allclose(restored, vertices, atol=1e-6)
