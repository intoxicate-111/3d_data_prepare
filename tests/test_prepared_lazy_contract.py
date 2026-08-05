from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import torch
import trimesh

from mlr.data import Mesh
from mlr.io import save_mesh
from mlr.learned_laplacian.dataset import load_prepared_sample, save_prepared_sample
from mlr.synthetic import SyntheticRenderConfig, generate_synthetic_dataset

from thingi10k50_prep.config import load_config
from thingi10k50_prep.mesh_ops import build_uniform_laplacian
from thingi10k50_prep.prepare import (
    _build_prepared_sample,
    _prepared_cache_issues,
    _render_cache_issues,
)
from thingi10k50_prep.rendering import VIEW_LAYOUT_VERSION
from thingi10k50_prep.smoke import _small_real_sample


def test_build_prepared_sample_is_lazy_and_uses_prediction_graph_target(tmp_path) -> None:
    root = tmp_path / "dataset"
    views_dir = root / "models" / "1" / "views"
    prepared_dir = root / "prepared"
    prepared_dir.mkdir(parents=True)
    sphere = trimesh.creation.icosphere(subdivisions=1, radius=0.8)
    vertices = np.asarray(sphere.vertices, dtype=np.float64)
    faces = np.asarray(sphere.faces, dtype=np.int64)
    mesh = Mesh(vertices, faces)
    mesh_path = root / "models" / "1" / "mesh.obj"
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    save_mesh(mesh, mesh_path)
    rendered = generate_synthetic_dataset(
        mesh,
        views_dir,
        SyntheticRenderConfig(
            num_views=14,
            width=20,
            height=16,
            trajectory="cube_surface",
            fov_degrees=90.0,
            backend="cpu",
            normalize_mesh=False,
        ),
        source_mesh_path=mesh_path,
    )
    laplacian = build_uniform_laplacian(len(vertices), faces)
    target_positions = vertices + np.array([0.01, -0.02, 0.03])
    sample = _build_prepared_sample(
        rendered.dataset_path,
        mesh_path,
        mesh_path,
        {"target_positions": target_positions, "valid_mask": np.ones(len(vertices), dtype=bool)},
        laplacian,
        image_size=12,
        sample_id="thingi10k_1",
        metadata={
            "normalization_mode": "unit_sphere_max_radius",
            "view_layout_version": VIEW_LAYOUT_VERSION,
            "edge_scale_epsilon": 1e-12,
        },
        dataset_root=root,
    )
    sample_path = prepared_dir / "thingi10k_1.pt"
    save_prepared_sample(sample, sample_path)
    raw = torch.load(sample_path, map_location="cpu", weights_only=False)

    assert "images" not in raw
    assert len(raw["image_paths"]) == 14
    assert all(not value.startswith("/") for value in raw["image_paths"])
    assert raw["intrinsics"].shape == (14, 3, 3)
    assert raw["extrinsics"].shape == (14, 4, 4)
    torch.testing.assert_close(
        raw["raw_laplacian_target"],
        torch.as_tensor(laplacian @ target_positions, dtype=torch.float32),
    )
    small = _small_real_sample(load_prepared_sample(sample_path), size=8)
    assert small["images"].shape == (1, 3, 8, 8)
    assert len(small["image_paths"]) == 1


def test_cache_contract_changes_are_reported() -> None:
    cfg = load_config("configs/thingi10k50.yaml")
    checksum = "mesh-sha"
    render_metadata = {
        "trajectory": cfg.views_trajectory,
        "backend": "opengl",
        "requested_backend": cfg.views_backend,
        "opengl_context_backend": cfg.views_opengl_context_backend,
        "cube_half_extent": cfg.views_cube_half_extent,
        "fov_degrees": cfg.views_fov_degrees,
        "render_mode": cfg.views_render_mode,
        "antialiasing": cfg.views_antialiasing,
        "camera_layout_version": VIEW_LAYOUT_VERSION,
        "normalized_mesh_checksum": checksum,
        "width": cfg.views_width,
        "height": cfg.views_height,
    }
    assert _render_cache_issues(render_metadata, cfg, checksum) == []
    for field, replacement in (
        ("camera_layout_version", "old-layout"),
        ("width", cfg.views_width + 1),
        ("normalized_mesh_checksum", "old-mesh"),
    ):
        changed = dict(render_metadata)
        changed[field] = replacement
        assert _render_cache_issues(changed, cfg, checksum)

    raw = {
        "prepared_storage_format": cfg.prepared_samples.storage_format,
        "image_paths": [f"images/{index}.png" for index in range(cfg.views_count)],
        "intrinsics": torch.zeros(cfg.views_count, 3, 3),
        "extrinsics": torch.zeros(cfg.views_count, 4, 4),
        "metadata": {
            "normalization_mode": cfg.normalization_mode,
            "view_layout_version": VIEW_LAYOUT_VERSION,
        },
    }
    assert _prepared_cache_issues(raw, cfg) == []
    assert _prepared_cache_issues(
        {**raw, "prepared_storage_format": "embedded_images_v1"}, cfg
    )
    changed_metadata = dict(raw)
    changed_metadata["metadata"] = {
        **raw["metadata"],
        "normalization_mode": "old-normalization",
    }
    assert _prepared_cache_issues(changed_metadata, cfg)

    changed_cfg = replace(cfg, views_count=cfg.views_count + 1)
    assert _prepared_cache_issues(raw, changed_cfg)
