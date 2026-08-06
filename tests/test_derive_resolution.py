from __future__ import annotations

import numpy as np

from thingi10k50_prep.derive import _scale_camera


def test_scale_camera_updates_intrinsics_and_image_size() -> None:
    camera = {
        "name": "test",
        "intrinsics": [[1000.0, 0.0, 960.0], [0.0, 800.0, 540.0], [0.0, 0.0, 1.0]],
        "image_size": [1920, 1080],
        "extrinsics": np.eye(4).tolist(),
    }

    scaled = _scale_camera(camera, 0.5, 0.5, 960, 540)

    np.testing.assert_allclose(
        scaled["intrinsics"],
        [[500.0, 0.0, 480.0], [0.0, 400.0, 270.0], [0.0, 0.0, 1.0]],
    )
    assert scaled["image_size"] == [960, 540]
    assert camera["image_size"] == [1920, 1080]
