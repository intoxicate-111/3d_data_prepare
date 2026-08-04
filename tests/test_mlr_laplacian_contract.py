from __future__ import annotations

import numpy as np
import pytest

from thingi10k50_prep.mesh_ops import build_uniform_laplacian


@pytest.mark.parametrize(
    ("faces", "num_vertices"),
    [
        (np.array([[0, 1, 2]], dtype=np.int64), 3),
        (np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64), 4),
        (np.array([[0, 1, 2], [2, 3, 4], [4, 5, 6]], dtype=np.int64), 7),
    ],
)
def test_sparse_uniform_laplacian_matches_mlr(faces: np.ndarray, num_vertices: int) -> None:
    from mlr.laplacian import build_uniform_laplacian as mlr_build_uniform_laplacian

    our_laplacian = build_uniform_laplacian(num_vertices, faces)
    expected = mlr_build_uniform_laplacian(faces, num_vertices)

    np.testing.assert_allclose(our_laplacian.toarray(), expected, rtol=0.0, atol=1e-12)
