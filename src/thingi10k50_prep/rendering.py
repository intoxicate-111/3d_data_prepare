from __future__ import annotations

import math
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np


def _create_cube_cameras(
    mesh: Any,
    num_views: int,
    image_size: tuple[int, int],
    trajectory: str = "cube",
    radius_scale: float = 2.5,
    elevation_degrees: float = 20.0,
    min_elevation_degrees: float = -60.0,
    max_elevation_degrees: float = 60.0,
    fov_degrees: float = 50.0,
) -> list[Any]:
    del elevation_degrees, min_elevation_degrees, max_elevation_degrees
    if trajectory != "cube":
        raise ValueError(f"Unsupported local camera trajectory: {trajectory}")
    if num_views != 14:
        raise ValueError(f"Cube camera trajectory requires exactly 14 views, got {num_views}")

    from mlr.data import Camera
    from mlr.synthetic import look_at_world_to_camera

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    target = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    extent = np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))
    radius = max(1e-3, radius_scale * extent)
    width, height = image_size
    focal = 0.5 * width / math.tan(math.radians(fov_degrees) * 0.5)
    intrinsics = np.array(
        [
            [focal, 0.0, width * 0.5],
            [0.0, focal, height * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    face_directions = [
        np.array(direction, dtype=np.float64)
        for direction in (
            (1, 0, 0), (-1, 0, 0),
            (0, 1, 0), (0, -1, 0),
            (0, 0, 1), (0, 0, -1),
        )
    ]
    corner_directions = [
        np.array(direction, dtype=np.float64)
        for direction in product((-1, 1), repeat=3)
    ]
    directions = face_directions + corner_directions

    cameras = []
    for index, direction in enumerate(directions):
        direction /= np.linalg.norm(direction)
        center = target + radius * direction
        rotation, translation = look_at_world_to_camera(center, target)
        cameras.append(
            Camera(
                intrinsics=intrinsics,
                rotation=rotation,
                translation=translation,
                image_size=image_size,
                name=f"view_{index:04d}",
            )
        )
    return cameras


def generate_configured_synthetic_dataset(
    mesh: Any,
    out_dir: str | Path,
    config: Any,
    source_mesh_path: Path,
) -> Any:
    """Use the downstream writer/rasterizer with the local 14-view cube trajectory."""
    import mlr.synthetic as synthetic

    if config.trajectory != "cube":
        return synthetic.generate_synthetic_dataset(
            mesh=mesh,
            out_dir=out_dir,
            config=config,
            source_mesh_path=source_mesh_path,
        )

    original_camera_factory = synthetic.create_synthetic_cameras
    synthetic.create_synthetic_cameras = _create_cube_cameras
    try:
        return synthetic.generate_synthetic_dataset(
            mesh=mesh,
            out_dir=out_dir,
            config=config,
            source_mesh_path=source_mesh_path,
        )
    finally:
        synthetic.create_synthetic_cameras = original_camera_factory
