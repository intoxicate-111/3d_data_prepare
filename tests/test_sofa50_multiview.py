from __future__ import annotations

import numpy as np

from sofa50_refinement.multiview import _depth_visibility


class _Camera:
    def project(self, points):
        return points[:, :2], points[:, 2]


def test_depth_visibility_requires_mask_finite_depth_and_depth_agreement() -> None:
    vertices = np.asarray(
        [[1.0, 1.0, 2.0], [2.0, 2.0, 5.0], [3.0, 3.0, 2.0], [-1.0, 0.0, 1.0]]
    )
    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 1] = True
    mask[2, 2] = True
    mask[3, 3] = True
    depth = np.full((4, 4), np.inf)
    depth[1, 1] = 2.0
    depth[2, 2] = 1.0
    depth[3, 3] = 2.0

    visible = _depth_visibility(vertices, [_Camera()], [mask], [depth], tolerance=0.01)

    assert visible.shape == (1, 4)
    assert visible.tolist() == [[True, False, True, False]]
