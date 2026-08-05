from __future__ import annotations

from pathlib import Path
from typing import Any


VIEW_LAYOUT_VERSION = "unit_sphere_cube_surface_faces6_corners8_v1"
CUBE_SURFACE_VIEW_NAMES = (
    "pos_x", "neg_x", "pos_y", "neg_y", "pos_z", "neg_z",
    "neg_x_neg_y_neg_z", "neg_x_neg_y_pos_z", "neg_x_pos_y_neg_z",
    "neg_x_pos_y_pos_z", "pos_x_neg_y_neg_z", "pos_x_neg_y_pos_z",
    "pos_x_pos_y_neg_z", "pos_x_pos_y_pos_z",
)


def generate_configured_synthetic_dataset(
    mesh: Any,
    out_dir: str | Path,
    config: Any,
    source_mesh_path: Path,
) -> Any:
    from mlr.synthetic import generate_synthetic_dataset

    return generate_synthetic_dataset(
        mesh=mesh,
        out_dir=out_dir,
        config=config,
        source_mesh_path=source_mesh_path,
    )
