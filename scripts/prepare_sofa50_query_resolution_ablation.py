from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


GROUPS = ("gt", "gt_sub1", "gt_sub2", "gt_adaptive")
TARGET_MODE = "edge_scale_normalized_laplacian"
GRAPH_BOUND_VISIBILITY_FIELDS = (
    "visibility",
    "visibility_backface_only",
    "visibility_occlusion_only",
    "visibility_backface_and_occlusion",
)


def triangle_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    a = vertices[faces[:, 0]]
    b = vertices[faces[:, 1]]
    c = vertices[faces[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def represented_vertex_area(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    face_area = triangle_areas(vertices, faces)
    vertex_area = np.zeros(len(vertices), dtype=np.float64)
    contribution = face_area / 3.0
    np.add.at(vertex_area, faces[:, 0], contribution)
    np.add.at(vertex_area, faces[:, 1], contribution)
    np.add.at(vertex_area, faces[:, 2], contribution)
    return vertex_area, face_area


def gini(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x) & (x >= 0)]
    if x.size == 0 or float(x.sum()) <= 0:
        return 0.0
    x = np.sort(x)
    n = x.size
    ranks = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(ranks * x) / (n * np.sum(x))) - (n + 1.0) / n)


def mesh_stats(vertices: np.ndarray, faces: np.ndarray) -> dict[str, float | int]:
    vertex_area, face_area = represented_vertex_area(vertices, faces)
    total_area = float(face_area.sum())
    top_k = max(1, int(math.ceil(0.10 * len(vertex_area))))
    top_share = (
        float(np.partition(vertex_area, len(vertex_area) - top_k)[-top_k:].sum() / total_area)
        if total_area > 0
        else 0.0
    )
    sum_sq = float(np.square(vertex_area).sum())
    effective_fraction = (
        float((vertex_area.sum() ** 2) / (len(vertex_area) * sum_sq)) if sum_sq > 0 else 0.0
    )
    positive_face_area = face_area[face_area > 0]
    return {
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "total_area": total_area,
        "degenerate_faces": int(np.count_nonzero(face_area <= 1e-18)),
        "max_face_area": float(face_area.max(initial=0.0)),
        "p99_face_area": float(np.quantile(positive_face_area, 0.99))
        if positive_face_area.size
        else 0.0,
        "median_face_area": float(np.median(positive_face_area))
        if positive_face_area.size
        else 0.0,
        "represented_area_gini": gini(vertex_area),
        "top10_vertex_surface_area_share": top_share,
        "effective_vertex_fraction_by_area": effective_fraction,
        "max_represented_area": float(vertex_area.max(initial=0.0)),
        "median_represented_area": float(np.median(vertex_area)),
    }


def _unique_sorted_edges(faces: np.ndarray) -> np.ndarray:
    edges = np.concatenate(
        (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0
    )
    edges = np.sort(edges, axis=1)
    return np.unique(edges, axis=0)


def _edge_keys(edges: np.ndarray, base: int) -> np.ndarray:
    return edges[:, 0].astype(np.int64) * np.int64(base) + edges[:, 1].astype(np.int64)


def split_marked_edges(
    vertices: np.ndarray,
    faces: np.ndarray,
    marked_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Conformingly split a set of existing mesh edges.

    Every marked edge gets exactly one midpoint shared by all incident faces. Faces
    with 0/1/2/3 marked edges are retriangulated into 1/2/3/4 triangles, so there
    are no T-junctions. New vertices lie exactly on the original piecewise-linear
    surface.
    """

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if len(marked_edges) == 0:
        return vertices.copy(), faces.copy()

    marked_edges = np.asarray(marked_edges, dtype=np.int64)
    marked_edges = np.sort(marked_edges, axis=1)
    marked_edges = np.unique(marked_edges, axis=0)
    if marked_edges.min() < 0 or marked_edges.max() >= len(vertices):
        raise ValueError("marked edge index is outside the vertex array")

    midpoint_vertices = 0.5 * (
        vertices[marked_edges[:, 0]] + vertices[marked_edges[:, 1]]
    )
    new_vertices = np.concatenate((vertices, midpoint_vertices), axis=0)

    # Numeric edge keys let us find each face-edge midpoint without a Python dict.
    base = len(vertices) + 1
    marked_keys = _edge_keys(marked_edges, base)
    order = np.argsort(marked_keys)
    marked_keys = marked_keys[order]
    midpoint_indices = len(vertices) + order

    a = faces[:, 0]
    b = faces[:, 1]
    c = faces[:, 2]

    def lookup(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lo = np.minimum(u, v)
        hi = np.maximum(u, v)
        keys = lo.astype(np.int64) * np.int64(base) + hi.astype(np.int64)
        pos = np.searchsorted(marked_keys, keys)
        found = pos < len(marked_keys)
        safe = np.minimum(pos, len(marked_keys) - 1)
        found &= marked_keys[safe] == keys
        mid = np.full(len(keys), -1, dtype=np.int64)
        mid[found] = midpoint_indices[safe[found]]
        return found, mid

    has_ab, m_ab = lookup(a, b)
    has_bc, m_bc = lookup(b, c)
    has_ca, m_ca = lookup(c, a)
    code = has_ab.astype(np.uint8) + 2 * has_bc.astype(np.uint8) + 4 * has_ca.astype(np.uint8)

    chunks: list[np.ndarray] = []

    def add(mask: np.ndarray, *triangles: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        if not np.any(mask):
            return
        for x, y, z in triangles:
            chunks.append(np.stack((x[mask], y[mask], z[mask]), axis=1))

    mask = code == 0
    add(mask, (a, b, c))

    mask = code == 1  # AB
    add(mask, (a, m_ab, c), (m_ab, b, c))

    mask = code == 2  # BC
    add(mask, (b, m_bc, a), (m_bc, c, a))

    mask = code == 4  # CA
    add(mask, (c, m_ca, b), (m_ca, a, b))

    mask = code == 3  # AB + BC
    add(mask, (a, m_ab, c), (m_ab, m_bc, c), (m_ab, b, m_bc))

    mask = code == 6  # BC + CA
    add(mask, (b, m_bc, a), (m_bc, m_ca, a), (m_bc, c, m_ca))

    mask = code == 5  # CA + AB
    add(mask, (c, m_ca, b), (m_ca, m_ab, b), (m_ca, a, m_ab))

    mask = code == 7  # AB + BC + CA
    add(
        mask,
        (a, m_ab, m_ca),
        (m_ab, b, m_bc),
        (m_ca, m_bc, c),
        (m_ab, m_bc, m_ca),
    )

    new_faces = np.concatenate(chunks, axis=0)
    return new_vertices, new_faces


def uniform_subdivide(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    return split_marked_edges(vertices, faces, _unique_sorted_edges(faces))


def adaptive_subdivide_by_vertex_area(
    vertices: np.ndarray,
    faces: np.ndarray,
    max_vertex_area: float,
    *,
    max_iters: int,
    max_vertices: int | None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    if not np.isfinite(max_vertex_area) or max_vertex_area <= 0:
        raise ValueError("max_vertex_area must be finite and positive")
    current_v = np.asarray(vertices, dtype=np.float64).copy()
    current_f = np.asarray(faces, dtype=np.int64).copy()
    history: list[dict[str, Any]] = []

    for iteration in range(max_iters + 1):
        vertex_area, face_area = represented_vertex_area(current_v, current_f)
        selected_vertices = np.flatnonzero(
            vertex_area > max_vertex_area * (1.0 + 1e-12)
        )
        selected_mask = np.zeros(len(current_v), dtype=bool)
        selected_mask[selected_vertices] = True
        selected_faces = np.flatnonzero(np.any(selected_mask[current_f], axis=1))
        history.append(
            {
                "iteration": int(iteration),
                "vertices": int(len(current_v)),
                "faces": int(len(current_f)),
                "max_represented_vertex_area": float(vertex_area.max(initial=0.0)),
                "max_face_area": float(face_area.max(initial=0.0)),
                "oversized_vertices": int(len(selected_vertices)),
                "incident_faces_split": int(len(selected_faces)),
            }
        )
        if len(selected_vertices) == 0:
            return current_v, current_f, history
        if iteration == max_iters:
            break

        # Split every edge of every face incident to an oversized vertex.  This
        # reduces the represented area (one third of incident face area) at the
        # selected vertex while keeping the triangulation conforming.
        marked = _unique_sorted_edges(current_f[selected_faces])
        projected_vertices = len(current_v) + len(marked)
        if max_vertices is not None and projected_vertices > max_vertices:
            raise RuntimeError(
                f"adaptive subdivision would exceed --max-vertices={max_vertices}: "
                f"{len(current_v)} + {len(marked)} = {projected_vertices}"
            )
        current_v, current_f = split_marked_edges(current_v, current_f, marked)

    final_max = float(represented_vertex_area(current_v, current_f)[0].max(initial=0.0))
    raise RuntimeError(
        f"adaptive subdivision did not reach max_vertex_area={max_vertex_area:.8g} "
        f"within {max_iters} iterations; final max={final_max:.8g}"
    )


def save_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# GT-derived query graph for Sofa50 resolution ablation\n")
        for x, y, z in vertices:
            handle.write(f"v {x:.10g} {y:.10g} {z:.10g}\n")
        for a, b, c in faces:
            handle.write(f"f {a + 1} {b + 1} {c + 1}\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_base_sample_path(multiview_root: Path, item: dict[str, Any]) -> Path:
    value = Path(str(item["path"]))
    return value if value.is_absolute() else (multiview_root / value).resolve()


def absolutize_image_paths(sample: dict[str, Any], multiview_root: Path) -> None:
    paths = sample.get("image_paths")
    if not isinstance(paths, list) or not paths:
        raise ValueError(
            "Expected existing Sofa50 GT-query samples to use lazy image_paths. "
            "Regenerate the canonical multiview dataset first."
        )
    resolved = []
    for value in paths:
        path = Path(str(value))
        if not path.is_absolute():
            path = multiview_root / path
        resolved.append(str(path.resolve()))
    sample["image_paths"] = resolved
    sample["prepared_storage_format"] = "lazy_image_paths_v1"


def build_variants(
    gt_vertices: np.ndarray,
    gt_faces: np.ndarray,
    *,
    adaptive_reference: str,
    adaptive_area_scale: float,
    adaptive_max_iters: int,
    max_vertices: int | None,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    if adaptive_reference not in {"sub1", "sub2"}:
        raise ValueError("adaptive_reference must be sub1 or sub2")
    if adaptive_area_scale <= 0:
        raise ValueError("adaptive_area_scale must be positive")

    gt = (gt_vertices.copy(), gt_faces.copy())
    sub1 = uniform_subdivide(*gt)
    if max_vertices is not None and len(sub1[0]) > max_vertices:
        raise RuntimeError(
            f"GT-sub1 has {len(sub1[0])} vertices, exceeding --max-vertices={max_vertices}"
        )
    sub2 = uniform_subdivide(*sub1)
    if max_vertices is not None and len(sub2[0]) > max_vertices:
        raise RuntimeError(
            f"GT-sub2 has {len(sub2[0])} vertices, exceeding --max-vertices={max_vertices}"
        )

    reference_mesh = sub1 if adaptive_reference == "sub1" else sub2
    reference_max_area = float(
        represented_vertex_area(*reference_mesh)[0].max(initial=0.0)
    )
    threshold = reference_max_area * adaptive_area_scale
    adaptive_v, adaptive_f, adaptive_history = adaptive_subdivide_by_vertex_area(
        *gt,
        threshold,
        max_iters=adaptive_max_iters,
        max_vertices=max_vertices,
    )

    variants = {
        "gt": gt,
        "gt_sub1": sub1,
        "gt_sub2": sub2,
        "gt_adaptive": (adaptive_v, adaptive_f),
    }
    adaptive_info = {
        "reference": adaptive_reference,
        "reference_max_represented_vertex_area": reference_max_area,
        "area_scale": float(adaptive_area_scale),
        "threshold": float(threshold),
        "history": adaptive_history,
    }
    return variants, adaptive_info


def prepare_variant_sample(
    base_sample: dict[str, Any],
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    group: str,
    adaptive_info: dict[str, Any] | None,
    visibility_backend: str,
    visibility_neighborhood_radius: int,
    visibility_artifact_path: Path,
    deps: dict[str, Any],
) -> dict[str, Any]:
    source = copy.copy(base_sample)
    source["metadata"] = dict(base_sample.get("metadata", {}))

    # Renderer visibility is indexed by graph vertices. Reusing it after
    # subdivision is invalid; it is recomputed below on every current graph.
    removed_visibility_fields = []
    for name in GRAPH_BOUND_VISIBILITY_FIELDS:
        if name in source:
            removed_visibility_fields.append(name)
            source.pop(name, None)
    source["metadata"].pop("visibility_artifact", None)
    source["metadata"].pop("renderer_visibility_artifact", None)

    source["gt_vertices"] = torch.as_tensor(vertices, dtype=torch.float32)
    source["gt_faces"] = torch.as_tensor(faces, dtype=torch.long)
    source["metadata"].update(
        {
            "ablation": "gt_query_resolution_and_surface_coverage_v1",
            "query_graph_variant": group,
            "query_graph_surface": "same_piecewise_linear_gt_surface",
            "cross_graph_target_transfer": False,
            "source_graph_visibility_discarded": sorted(removed_visibility_fields),
            "renderer_visibility_recompute_required": False,
            "visibility_policy": "renderer_native_current_query_graph",
        }
    )

    sample = deps["prepare_gt_query_sample_from_prepared"](
        source,
        target_mode=TARGET_MODE,
    )

    # Defensive cleanup before attaching visibility for the current graph.
    for name in GRAPH_BOUND_VISIBILITY_FIELDS:
        sample.pop(name, None)
    sample["visibility"] = None

    renderer_result = _compute_renderer_visibility(
        sample,
        backend=visibility_backend,
        neighborhood_radius=visibility_neighborhood_radius,
        deps=deps,
    )
    visibility_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        visibility_artifact_path,
        frustum_valid=renderer_result.frustum_valid,
        visibility_backface_only=renderer_result.backface_visible,
        visibility_occlusion_only=renderer_result.occlusion_visible,
        visibility_backface_and_occlusion=(
            renderer_result.backface_and_occlusion_visible
        ),
    )
    sample["visibility_backface_only"] = torch.from_numpy(
        renderer_result.backface_visible
    )
    sample["visibility_occlusion_only"] = torch.from_numpy(
        renderer_result.occlusion_visible
    )
    sample["visibility_backface_and_occlusion"] = torch.from_numpy(
        renderer_result.backface_and_occlusion_visible
    )
    sample["visibility"] = sample["visibility_backface_and_occlusion"]

    metadata = dict(sample.get("metadata", {}))
    metadata.update(
        {
            "ablation": "gt_query_resolution_and_surface_coverage_v1",
            "query_graph_variant": group,
            "query_graph_surface": "same_piecewise_linear_gt_surface",
            "target_recomputed_on_current_graph": True,
            "h2_normalization_recomputed_on_current_graph": True,
            "cross_graph_target_transfer": False,
            "renderer_visibility_recompute_required": False,
            "visibility_policy": "renderer_native_current_query_graph",
            "renderer_visibility": {
                "definition": "depth_tested_face_id_incident_face_neighborhood",
                "artifact_path": str(visibility_artifact_path),
                "backend": visibility_backend,
                "front_face_winding": "ccw",
                "neighborhood_radius": int(visibility_neighborhood_radius),
                "depth_image_used": False,
                "graph": group,
            },
        }
    )
    if group == "gt_sub1":
        metadata["uniform_midpoint_subdivision_steps"] = 1
    elif group == "gt_sub2":
        metadata["uniform_midpoint_subdivision_steps"] = 2
    elif group == "gt_adaptive" and adaptive_info is not None:
        metadata.update(
            {
                "adaptive_subdivision_criterion": "represented_vertex_area",
                "adaptive_reference": adaptive_info["reference"],
                "adaptive_max_represented_vertex_area_reference": adaptive_info[
                    "reference_max_represented_vertex_area"
                ],
                "adaptive_max_represented_vertex_area_scale": adaptive_info["area_scale"],
                "adaptive_max_represented_vertex_area_threshold": adaptive_info["threshold"],
                "adaptive_iterations": len(adaptive_info["history"]) - 1,
            }
        )
    sample["metadata"] = metadata
    deps["validate_gt_query_contract"](sample)
    return sample


def _compute_renderer_visibility(
    sample: dict[str, Any],
    *,
    backend: str,
    neighborhood_radius: int,
    deps: dict[str, Any],
) -> Any:
    image_size = int(sample["prepared_image_size"])
    intrinsics = sample["intrinsics"].detach().cpu().numpy()
    extrinsics = sample["extrinsics"].detach().cpu().numpy()
    cameras = [
        deps["Camera"](
            intrinsics=intrinsics[index],
            rotation=extrinsics[index, :3, :3],
            translation=extrinsics[index, :3, 3],
            image_size=(image_size, image_size),
            name=f"view_{index:04d}",
        )
        for index in range(len(intrinsics))
    ]
    mesh = deps["Mesh"](
        sample["vertices"].detach().cpu().numpy(),
        sample["faces"].detach().cpu().numpy(),
    ).ensure_normals()
    if backend == "cuda":
        from sofa50_refinement.gpu_visibility import (
            compute_renderer_visibility_cuda,
        )

        return compute_renderer_visibility_cuda(
            mesh,
            cameras,
            image_size=image_size,
            neighborhood_radius=neighborhood_radius,
            front_face_winding="ccw",
        )
    config = deps["SyntheticRenderConfig"](
        num_views=len(cameras),
        width=image_size,
        height=image_size,
        backend=backend,
        normalize_mesh=False,
        antialiasing="none",
        backface_culling=False,
        front_face_winding="ccw",
    )
    return deps["compute_renderer_visibility"](
        mesh,
        cameras,
        config,
        neighborhood_radius=neighborhood_radius,
    )


def make_manifest(
    records: list[dict[str, Any]],
    output_root: Path,
    group: str,
) -> dict[str, Any]:
    return {
        "format_version": "sofa50_gt_query_resolution_ablation_v2",
        "dataset_role": "gt_query_training_resolution_ablation",
        "training_eligible": True,
        "query_graph_variant": group,
        "target_mode": TARGET_MODE,
        "render_observations": "reused_from_existing_sofa50_multiview",
        "samples": [
            {
                "sample_id": record["sample_id"],
                "split": record["split"],
                "path": Path(record["prepared_paths"][group])
                .relative_to(output_root)
                .as_posix(),
            }
            for record in records
        ],
    }


def prepared_training_contract_complete(path: Path, artifact_path: Path) -> bool:
    if not path.is_file() or not artifact_path.is_file():
        return False
    try:
        sample = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError, TypeError):
        return False
    if not isinstance(sample, dict):
        return False
    visibility = sample.get("visibility_backface_and_occlusion")
    image_paths = sample.get("image_paths")
    vertices = sample.get("vertices")
    return (
        isinstance(visibility, torch.Tensor)
        and isinstance(image_paths, list)
        and isinstance(vertices, torch.Tensor)
        and tuple(visibility.shape) == (len(image_paths), len(vertices))
    )


def validate_manifests(
    manifest_paths: dict[str, Path], training_config: dict[str, Any], deps: dict[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    reference_ids: dict[str, tuple[str, ...]] = {}
    for group, manifest_path in manifest_paths.items():
        payload = read_json(manifest_path)
        splits = sorted({str(item["split"]) for item in payload["samples"]})
        group_result: dict[str, Any] = {}
        for split in splits:
            dataset = deps["PreparedMeshDataset"].from_manifest(manifest_path, split)
            ids = dataset.sample_ids
            reference_ids.setdefault(split, ids)
            if ids != reference_ids[split]:
                raise ValueError(f"{group}: sample-ID order differs for split {split}")
            vertex_counts = []
            for index in range(len(dataset)):
                sample = dataset.load_static(index)
                deps["validate_gt_query_contract"](sample)
                field = sample.get("visibility_backface_and_occlusion")
                expected_shape = (len(sample["image_paths"]), len(sample["vertices"]))
                if not isinstance(field, torch.Tensor) or tuple(field.shape) != expected_shape:
                    raise ValueError(
                        f"{group}/{sample['sample_id']}: renderer visibility shape mismatch"
                    )
                deps["prepare_object_static"](
                    sample,
                    training_config,
                    keep_image_payload=True,
                    keep_projection=True,
                )
                missing_images = [
                    value for value in sample["image_paths"] if not Path(value).is_file()
                ]
                if missing_images:
                    raise FileNotFoundError(
                        f"{group}/{sample['sample_id']}: missing image {missing_images[0]}"
                    )
                vertex_counts.append(int(sample["vertices"].shape[0]))
            group_result[split] = {
                "samples": len(dataset),
                "min_vertices": min(vertex_counts),
                "max_vertices": max(vertex_counts),
            }
        result[group] = group_result
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare four Sofa50 GT-query graph-resolution groups using the same existing "
            "RGB/camera observations: GT, GT-sub1, GT-sub2, and area-adaptive GT."
        )
    )
    parser.add_argument(
        "--multiview-root",
        type=Path,
        default=Path("~/sofa_mesh/sofa50_refinement/multiview_960"),
        help="Existing canonical Sofa50 multiview root containing gt_query_manifest.json.",
    )
    parser.add_argument(
        "--downstream-root",
        type=Path,
        default=Path("~/multiview-laplacian-refinement"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Default: <multiview-root>/query_resolution_ablation_v2",
    )
    parser.add_argument(
        "--adaptive-reference",
        choices=("sub1", "sub2"),
        default="sub2",
        help=(
            "Use the maximum represented vertex area of this uniform-subdivision "
            "group as the adaptive coverage threshold reference."
        ),
    )
    parser.add_argument(
        "--adaptive-area-scale",
        type=float,
        default=1.0,
        help=(
            "Adaptive maximum represented vertex area = the selected reference "
            "maximum times this scale."
        ),
    )
    parser.add_argument("--adaptive-max-iters", type=int, default=12)
    parser.add_argument(
        "--max-vertices",
        type=int,
        help="Optional safety cap applied to sub1, sub2, and adaptive meshes.",
    )
    parser.add_argument(
        "--model-id",
        action="append",
        dest="model_ids",
        help="Prepare only this model ID; repeat for a subset.",
    )
    parser.add_argument("--write-objs", action="store_true")
    parser.add_argument(
        "--visibility-backend",
        choices=("cuda", "cpu", "opengl"),
        default="cuda",
    )
    parser.add_argument("--visibility-neighborhood-radius", type=int, default=1)
    parser.add_argument(
        "--training-config",
        type=Path,
        help=(
            "Config used for direct downstream training-contract validation. Default: "
            "<downstream-root>/configs/learned_laplacian/"
            "train_sofa50_50mesh_2000epoch_absolute_h2_confidence.json"
        ),
    )
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    multiview_root = args.multiview_root.expanduser().resolve()
    downstream_root = args.downstream_root.expanduser().resolve()
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else multiview_root / "query_resolution_ablation_v2"
    )
    if args.adaptive_max_iters < 1:
        raise ValueError("--adaptive-max-iters must be positive")
    if args.max_vertices is not None and args.max_vertices < 1:
        raise ValueError("--max-vertices must be positive")
    if args.visibility_neighborhood_radius < 0:
        raise ValueError("--visibility-neighborhood-radius must be non-negative")

    base_manifest_path = multiview_root / "gt_query_manifest.json"
    if not base_manifest_path.is_file():
        raise FileNotFoundError(f"Missing canonical GT-query manifest: {base_manifest_path}")
    base_manifest = read_json(base_manifest_path)
    items = list(base_manifest.get("samples", []))
    if not items:
        raise ValueError("Canonical GT-query manifest contains no samples")

    requested = set(args.model_ids or [])
    if requested:
        items = [item for item in items if str(item.get("sample_id")) in requested]
        found = {str(item.get("sample_id")) for item in items}
        missing = sorted(requested - found)
        if missing:
            raise ValueError("Unknown --model-id values: " + ", ".join(missing))

    deps = _load_downstream_dependencies(downstream_root)
    training_config_path = (
        args.training_config.expanduser().resolve()
        if args.training_config is not None
        else downstream_root
        / "configs"
        / "learned_laplacian"
        / "train_sofa50_50mesh_2000epoch_absolute_h2_confidence.json"
    )
    if not training_config_path.is_file():
        raise FileNotFoundError(f"Missing downstream training config: {training_config_path}")
    training_config = read_json(training_config_path)
    output_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    flat_stats: list[dict[str, Any]] = []

    for sample_index, item in enumerate(items, start=1):
        model_id = str(item["sample_id"])
        split = str(item["split"])
        print(f"[{sample_index}/{len(items)}] {model_id}", flush=True)
        base_path = resolve_base_sample_path(multiview_root, item)
        base_sample = torch.load(base_path, map_location="cpu", weights_only=False)
        if not isinstance(base_sample, dict):
            raise TypeError(f"{model_id}: base prepared sample is not a dict")
        absolutize_image_paths(base_sample, multiview_root)

        gt_vertices_tensor = base_sample.get("gt_vertices")
        gt_faces_tensor = base_sample.get("gt_faces")
        if not isinstance(gt_vertices_tensor, torch.Tensor) or not isinstance(
            gt_faces_tensor, torch.Tensor
        ):
            raise ValueError(f"{model_id}: canonical sample has no GT graph tensors")
        gt_vertices = gt_vertices_tensor.detach().cpu().numpy().astype(np.float64, copy=True)
        gt_faces = gt_faces_tensor.detach().cpu().numpy().astype(np.int64, copy=True)

        variants, adaptive_info = build_variants(
            gt_vertices,
            gt_faces,
            adaptive_reference=args.adaptive_reference,
            adaptive_area_scale=args.adaptive_area_scale,
            adaptive_max_iters=args.adaptive_max_iters,
            max_vertices=args.max_vertices,
        )
        print(
            "  query vertices: "
            + ", ".join(f"{name}={len(variants[name][0])}" for name in GROUPS)
            + f"; adaptive threshold={adaptive_info['threshold']:.6g}",
            flush=True,
        )

        prepared_paths: dict[str, str] = {}
        per_group_stats: dict[str, Any] = {}
        gt_total_area = mesh_stats(*variants["gt"])["total_area"]
        for group in GROUPS:
            vertices, faces = variants[group]
            stats = mesh_stats(vertices, faces)
            stats["total_area_relative_error_vs_gt"] = (
                abs(float(stats["total_area"]) - float(gt_total_area))
                / max(abs(float(gt_total_area)), 1e-30)
            )
            if stats["total_area_relative_error_vs_gt"] > 1e-10:
                raise RuntimeError(
                    f"{model_id}/{group}: subdivision changed total surface area by "
                    f"{stats['total_area_relative_error_vs_gt']:.3e}"
                )
            per_group_stats[group] = stats

            prepared_path = output_root / "prepared" / group / f"{model_id}.pt"
            prepared_paths[group] = str(prepared_path)
            visibility_artifact_path = (
                output_root / "renderer_visibility" / group / f"{model_id}.npz"
            )
            if args.force or not prepared_training_contract_complete(
                prepared_path, visibility_artifact_path
            ):
                sample = prepare_variant_sample(
                    base_sample,
                    vertices,
                    faces,
                    group=group,
                    adaptive_info=adaptive_info if group == "gt_adaptive" else None,
                    visibility_backend=args.visibility_backend,
                    visibility_neighborhood_radius=args.visibility_neighborhood_radius,
                    visibility_artifact_path=visibility_artifact_path,
                    deps=deps,
                )
                prepared_path.parent.mkdir(parents=True, exist_ok=True)
                deps["save_prepared_sample"](sample, prepared_path)

            if args.write_objs:
                obj_path = output_root / "meshes" / group / f"{model_id}.obj"
                if args.force or not obj_path.is_file():
                    save_obj(obj_path, vertices, faces)

            flat_stats.append(
                {
                    "model_id": model_id,
                    "split": split,
                    "group": group,
                    **stats,
                }
            )

        adaptive_stats = per_group_stats["gt_adaptive"]
        sub2_stats = per_group_stats["gt_sub2"]
        record = {
            "sample_id": model_id,
            "split": split,
            "base_prepared_path": str(base_path),
            "prepared_paths": prepared_paths,
            "stats": per_group_stats,
            "adaptive": {
                **adaptive_info,
                "vertex_count_ratio_vs_sub2": float(adaptive_stats["vertices"])
                / float(sub2_stats["vertices"]),
                "face_count_ratio_vs_sub2": float(adaptive_stats["faces"])
                / float(sub2_stats["faces"]),
            },
        }
        records.append(record)

    manifest_paths: dict[str, Path] = {}
    for group in GROUPS:
        path = output_root / f"{group}_manifest.json"
        write_json(path, make_manifest(records, output_root, group))
        manifest_paths[group] = path

    csv_path = output_root / "mesh_stats.csv"
    if flat_stats:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat_stats[0].keys()))
            writer.writeheader()
            writer.writerows(flat_stats)

    validation = None
    if not args.skip_validation:
        print("Validating all four manifests with downstream static loaders...", flush=True)
        validation = validate_manifests(manifest_paths, training_config, deps)

    summary = {
        "format_version": "sofa50_query_resolution_ablation_summary_v2",
        "base_manifest": str(base_manifest_path),
        "multiview_root": str(multiview_root),
        "downstream_root": str(downstream_root),
        "output_root": str(output_root),
        "target_mode": TARGET_MODE,
        "groups": list(GROUPS),
        "control": {
            "same_rgb_camera_observations": True,
            "same_piecewise_linear_gt_surface": True,
            "target_recomputed_on_each_current_graph": True,
            "h2_normalization_recomputed_on_each_current_graph": True,
            "cross_graph_target_interpolation": False,
            "gt_query_initial_laplacian": "zero",
        },
        "adaptive_policy": {
            "criterion": "max_represented_vertex_area",
            "reference": args.adaptive_reference,
            "area_scale": args.adaptive_area_scale,
            "interpretation": (
                "adaptive mesh splits faces incident to vertices whose represented area "
                "exceeds the selected uniform-reference maximum represented vertex area"
            ),
        },
        "renderer_visibility": {
            "backend": args.visibility_backend,
            "front_face_winding": "ccw",
            "neighborhood_radius": args.visibility_neighborhood_radius,
            "training_config": str(training_config_path),
        },
        "manifests": {name: str(path) for name, path in manifest_paths.items()},
        "mesh_stats_csv": str(csv_path),
        "validation": validation,
        "samples": records,
    }
    summary_path = output_root / "summary.json"
    write_json(summary_path, summary)

    print(
        json.dumps(
            {
                "status": "passed",
                "samples": len(records),
                "output_root": str(output_root),
                "manifests": {name: str(path) for name, path in manifest_paths.items()},
                "mesh_stats_csv": str(csv_path),
                "summary": str(summary_path),
            },
            indent=2,
        ),
        flush=True,
    )


def _load_downstream_dependencies(downstream_root: Path) -> dict[str, Any]:
    """Reuse exactly the same downstream imports as the canonical Sofa50 preparer."""
    # Import lazily so the pure subdivision helpers can be unit-tested without the
    # downstream repository being present.
    try:
        from sofa50_refinement.multiview import _downstream_dependencies
    except ImportError as exc:
        raise ImportError(
            "Could not import sofa50_refinement. Run this script from the 3d_data_prepare "
            "environment (for example after `pip install -e .`)."
        ) from exc
    deps = _downstream_dependencies(downstream_root)
    source = downstream_root / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from mlr.data import Camera, Mesh
    from mlr.learned_laplacian.multi_trainer import _prepare_object_static
    from mlr.learned_laplacian.renderer_visibility import compute_renderer_visibility
    from mlr.synthetic import SyntheticRenderConfig

    deps.update(
        {
            "Camera": Camera,
            "Mesh": Mesh,
            "SyntheticRenderConfig": SyntheticRenderConfig,
            "compute_renderer_visibility": compute_renderer_visibility,
            "prepare_object_static": _prepare_object_static,
        }
    )
    return deps


if __name__ == "__main__":
    main()
