from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import trimesh


def _load_script(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_nested = _load_script("prepare_sofa50_nested_views_14_28_56")
_ablation = _load_script("prepare_sofa50_query_resolution_ablation")
_combo = _load_script("prepare_sofa50_view28_gt_adaptive")
_attach_renderer_visibility = _nested._attach_renderer_visibility
_manifest = _nested._manifest
_prepared_contract_complete = _nested._prepared_contract_complete
_prepare_expanded_inference_sample = _nested._prepare_expanded_inference_sample
adaptive_subdivide_by_vertex_area = _ablation.adaptive_subdivide_by_vertex_area
build_variants = _ablation.build_variants
represented_vertex_area = _ablation.represented_vertex_area
triangle_areas = _ablation.triangle_areas
merge_observations = _combo.merge_observations


def _uneven_planar_mesh() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [4.0, 4.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.2, 0.2, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [
            [0, 1, 4],
            [1, 2, 4],
            [2, 3, 4],
            [3, 0, 4],
        ],
        dtype=np.int64,
    )
    return vertices, faces


def test_adaptive_subdivision_enforces_represented_vertex_area_limit() -> None:
    vertices, faces = _uneven_planar_mesh()
    initial_vertex_area, _ = represented_vertex_area(vertices, faces)
    threshold = float(initial_vertex_area.max()) / 5.0

    refined_vertices, refined_faces, history = adaptive_subdivide_by_vertex_area(
        vertices,
        faces,
        threshold,
        max_iters=8,
        max_vertices=10_000,
    )

    refined_vertex_area, _ = represented_vertex_area(refined_vertices, refined_faces)
    assert refined_vertex_area.max() <= threshold * (1.0 + 1e-12)
    np.testing.assert_allclose(
        triangle_areas(refined_vertices, refined_faces).sum(),
        triangle_areas(vertices, faces).sum(),
        rtol=0.0,
        atol=1e-12,
    )
    assert history[0]["oversized_vertices"] > 0
    assert history[-1]["oversized_vertices"] == 0
    assert len(refined_vertices) > len(vertices)


def test_adaptive_variant_uses_uniform_reference_vertex_area() -> None:
    vertices, faces = _uneven_planar_mesh()
    variants, adaptive = build_variants(
        vertices,
        faces,
        adaptive_reference="sub1",
        adaptive_area_scale=1.0,
        adaptive_max_iters=8,
        max_vertices=10_000,
    )

    reference_max = represented_vertex_area(*variants["gt_sub1"])[0].max()
    adaptive_max = represented_vertex_area(*variants["gt_adaptive"])[0].max()
    assert adaptive["reference_max_represented_vertex_area"] == reference_max
    assert adaptive_max <= reference_max * (1.0 + 1e-12)


def test_nested_visibility_is_sliced_and_attached_for_training(tmp_path) -> None:
    views = 56
    vertices = 5
    base = np.arange(views * vertices).reshape(views, vertices) % 3 == 0
    result = SimpleNamespace(
        frustum_valid=np.ones((views, vertices), dtype=bool),
        backface_visible=base,
        occlusion_visible=~base,
        backface_and_occlusion_visible=base,
    )
    sample = {"metadata": {}}
    artifact = tmp_path / "visibility.npz"

    _attach_renderer_visibility(
        sample,
        result,
        28,
        artifact,
        graph_role="gt_query_training",
        backend="cuda",
    )

    assert tuple(sample["visibility_backface_and_occlusion"].shape) == (28, vertices)
    assert torch.equal(sample["visibility"], sample["visibility_backface_and_occlusion"])
    assert sample["metadata"]["renderer_visibility_recompute_required"] is False
    with np.load(artifact) as payload:
        assert payload["visibility_backface_and_occlusion"].shape == (28, vertices)


def test_view28_adaptive_combo_replaces_only_observation_contract() -> None:
    adaptive_vertices = torch.randn(7, 3)
    adaptive_target = torch.randn(7, 3)
    adaptive = {
        "sample_id": "sample",
        "vertices": adaptive_vertices,
        "faces": torch.tensor([[0, 1, 2]], dtype=torch.long),
        "laplacian_target": adaptive_target,
        "intrinsics": torch.eye(3).repeat(14, 1, 1),
        "extrinsics": torch.eye(4).repeat(14, 1, 1),
        "image_paths": [f"old/{index}.png" for index in range(14)],
        "source_image_size": [960, 960],
        "prepared_image_size": 960,
        "prepared_storage_format": "lazy_image_paths_v1",
        "visibility": torch.ones((14, 7), dtype=torch.bool),
        "visibility_backface_and_occlusion": torch.ones((14, 7), dtype=torch.bool),
        "metadata": {"query_graph_variant": "gt_adaptive", "adaptive_reference": "sub2"},
    }
    view28 = {
        "sample_id": "sample",
        "intrinsics": torch.arange(28 * 9, dtype=torch.float32).reshape(28, 3, 3),
        "extrinsics": torch.eye(4).repeat(28, 1, 1),
        "source_image_size": [960, 960],
        "prepared_image_size": 960,
        "prepared_storage_format": "lazy_image_paths_v1",
        "metadata": {"nested_view_count": 28, "camera_layout_version": "nested-v1"},
    }
    image_paths = [f"nested/{index}.png" for index in range(28)]

    combined = merge_observations(adaptive, view28, image_paths=image_paths)

    assert torch.equal(combined["vertices"], adaptive_vertices)
    assert torch.equal(combined["laplacian_target"], adaptive_target)
    assert torch.equal(combined["intrinsics"], view28["intrinsics"])
    assert combined["image_paths"] == image_paths
    assert "visibility" not in combined
    assert "visibility_backface_and_occlusion" not in combined
    assert combined["metadata"]["query_graph_variant"] == "gt_adaptive"
    assert combined["metadata"]["adaptive_reference"] == "sub2"
    assert combined["metadata"]["nested_view_count"] == 28
    assert combined["metadata"]["combination_arm"] == "views_28_gt_adaptive"


def test_nested_manifests_declare_training_and_inference_roles() -> None:
    records = [{"sample_id": "a", "split": "train", "path": "a.pt"}]
    training = _manifest(records, 14, expanded=False)
    inference = _manifest(records, 14, expanded=True)

    assert training["training_eligible"] is True
    assert training["dataset_role"] == "gt_query_training"
    assert inference["training_eligible"] is False
    assert inference["dataset_role"] == "expanded_raw_frozen_model_inference"


def test_nested_prepared_contract_requires_visibility_artifact(tmp_path) -> None:
    sample_path = tmp_path / "sample.pt"
    artifact = tmp_path / "visibility.npz"
    torch.save(
        {
            "vertices": torch.zeros((3, 3)),
            "image_paths": [f"{index}.png" for index in range(14)],
            "visibility_backface_and_occlusion": torch.zeros((14, 3), dtype=torch.bool),
        },
        sample_path,
    )
    assert not _prepared_contract_complete(sample_path, artifact, 14)
    np.savez_compressed(artifact, visibility=np.zeros((14, 3), dtype=bool))
    assert _prepared_contract_complete(sample_path, artifact, 14)


def test_nested_expanded_sample_enters_downstream_inference_preparation(tmp_path) -> None:
    downstream_root = Path(__file__).resolve().parents[2] / "multiview-laplacian-refinement"
    deps = _nested._dependencies(downstream_root)
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    gt_path = tmp_path / "gt.obj"
    expanded_path = tmp_path / "expanded.obj"
    mesh.export(gt_path)
    mesh.export(expanded_path)
    count = 14
    source = {
        "image_paths": [f"images/{index:04d}.png" for index in range(count)],
        "intrinsics": torch.eye(3).repeat(count, 1, 1),
        "extrinsics": torch.eye(4).repeat(count, 1, 1),
    }
    shape = (56, len(mesh.vertices))
    visibility = np.ones(shape, dtype=bool)
    result = SimpleNamespace(
        frustum_valid=visibility,
        backface_visible=visibility,
        occlusion_visible=visibility,
        backface_and_occlusion_visible=visibility,
    )
    prepared_path = tmp_path / "expanded.pt"
    artifact_path = tmp_path / "expanded_visibility.npz"

    _prepare_expanded_inference_sample(
        source,
        "sample",
        "validation",
        count,
        expanded_path,
        gt_path,
        prepared_path,
        result,
        artifact_path,
        "cuda",
        deps,
    )

    sample = torch.load(prepared_path, map_location="cpu", weights_only=False)
    config = json.loads(
        (
            downstream_root
            / "configs"
            / "learned_laplacian"
            / "train_sofa50_50mesh_2000epoch_absolute_h2_confidence.json"
        ).read_text(encoding="utf-8")
    )
    config["query_training"]["enabled"] = False
    prepared = deps["prepare_object_static"](
        sample,
        config,
        keep_image_payload=True,
        keep_projection=True,
    )

    assert prepared.sample["num_views"] == count
    assert tuple(prepared.sample["visibility"].shape) == (count, len(mesh.vertices))
    assert sample["metadata"]["training_eligible"] is False
