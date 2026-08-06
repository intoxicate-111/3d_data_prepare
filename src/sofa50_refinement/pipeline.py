from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fast_simplification
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import trimesh
from PIL import Image, ImageOps
from scipy.spatial import cKDTree
from tqdm import tqdm

from thingi10k50_prep.mesh_ops import midpoint_subdivide


DATASET_FORMAT = "sofa50_refinement_final_v1"
TARGET_CONSTRUCTOR = "collapse_provenance_oracle_diagnostic_v2"


@dataclass(frozen=True)
class MeshData:
    vertices: np.ndarray
    faces: np.ndarray


def _json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _load_obj(path: Path) -> MeshData:
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError("mesh scene has no geometry")
        loaded = loaded.to_mesh()
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"unsupported mesh type: {type(loaded).__name__}")
    return MeshData(
        np.asarray(loaded.vertices, dtype=np.float64),
        np.asarray(loaded.faces, dtype=np.int64),
    )


def _save_mesh(base: Path, mesh: MeshData) -> dict[str, str]:
    base.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        base.with_suffix(".npz"),
        # Keep source/correspondence precision.  Some valid Sofa faces are very
        # thin and lose their barycentric reconstruction contract at float32.
        vertices=np.asarray(mesh.vertices, dtype=np.float64),
        faces=np.asarray(mesh.faces, dtype=np.int64),
    )
    value = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False).export(file_type="obj")
    base.with_suffix(".obj").write_text(value, encoding="utf-8")
    return {
        "obj": str(base.with_suffix(".obj").resolve()),
        "npz": str(base.with_suffix(".npz").resolve()),
    }


def _load_npz_mesh(path: Path) -> MeshData:
    payload = np.load(path)
    return MeshData(
        np.asarray(payload["vertices"], dtype=np.float64),
        np.asarray(payload["faces"], dtype=np.int64),
    )


def _checksum(mesh: MeshData) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(mesh.vertices, dtype=np.float64).tobytes())
    digest.update(np.asarray(mesh.faces, dtype=np.int64).tobytes())
    return digest.hexdigest()


def _mesh_issues(mesh: MeshData) -> list[str]:
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    issues: list[str] = []
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        issues.append("invalid_vertices_shape")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        issues.append("invalid_faces_shape")
        return issues
    if not np.isfinite(vertices).all():
        issues.append("non_finite_vertices")
    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        issues.append("out_of_range_face_indices")
        return issues
    repeated = (
        (faces[:, 0] == faces[:, 1])
        | (faces[:, 1] == faces[:, 2])
        | (faces[:, 0] == faces[:, 2])
    )
    if np.any(repeated):
        issues.append("repeated_vertex_face")
    triangles = vertices[faces]
    double_area = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    if np.any(~np.isfinite(double_area)) or np.any(double_area <= 0):
        issues.append("degenerate_face")
    if len(np.unique(faces)) != len(vertices):
        issues.append("unreferenced_vertices")
    return issues


def _unique_edges(faces: np.ndarray) -> np.ndarray:
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    return np.unique(np.sort(edges, axis=1), axis=0)


def simplify_with_provenance(
    gt: MeshData,
    target_vertices: int,
    min_vertices: int,
) -> tuple[MeshData, np.ndarray, np.ndarray, dict[str, Any]]:
    vertex_components, component_faces = _vertex_components(gt)
    ratio = min(1.0, target_vertices / len(gt.vertices))
    coarse_vertices: list[np.ndarray] = []
    coarse_faces: list[np.ndarray] = []
    collapse_history: list[np.ndarray] = []
    original_to_coarse = np.full(len(gt.vertices), -1, dtype=np.int64)
    component_records: list[dict[str, Any]] = []
    vertex_offset = 0
    for component in sorted(component_faces):
        face_indices = component_faces[component]
        global_faces = gt.faces[face_indices]
        global_vertices = np.flatnonzero(vertex_components == component)
        global_to_local = np.full(len(gt.vertices), -1, dtype=np.int64)
        global_to_local[global_vertices] = np.arange(len(global_vertices), dtype=np.int64)
        local = MeshData(gt.vertices[global_vertices], global_to_local[global_faces])
        requested_vertices = max(4, int(round(len(local.vertices) * ratio)))
        requested_faces = max(4, int(round(len(local.faces) * ratio)))
        requested_faces = min(requested_faces, max(len(local.faces) - 1, 0))
        if (
            len(local.vertices) <= requested_vertices
            or len(local.faces) <= 4
            or requested_faces <= 0
        ):
            local_coarse = local
            local_collapses = np.zeros((0, 2), dtype=np.int64)
            local_mapping = np.arange(len(local.vertices), dtype=np.int64)
        else:
            _, _, local_collapses = fast_simplification.simplify(
                np.asarray(local.vertices, dtype=np.float64),
                np.asarray(local.faces, dtype=np.int64),
                target_count=requested_faces,
                agg=7.0,
                return_collapses=True,
            )
            replay_vertices, replay_faces, local_mapping = (
                fast_simplification.replay_simplification(
                    np.asarray(local.vertices, dtype=np.float32),
                    np.asarray(local.faces, dtype=np.int32),
                    np.asarray(local_collapses, dtype=np.int32),
                )
            )
            local_coarse = MeshData(
                np.asarray(replay_vertices, dtype=np.float64),
                np.asarray(replay_faces, dtype=np.int64),
            )
        coarse_vertices.append(local_coarse.vertices)
        coarse_faces.append(local_coarse.faces + vertex_offset)
        original_to_coarse[global_vertices] = np.asarray(local_mapping) + vertex_offset
        if len(local_collapses):
            collapse_history.append(global_vertices[np.asarray(local_collapses, dtype=np.int64)])
        component_records.append(
            {
                "component": int(component),
                "gt_vertices": int(len(local.vertices)),
                "gt_faces": int(len(local.faces)),
                "coarse_vertices": int(len(local_coarse.vertices)),
                "coarse_faces": int(len(local_coarse.faces)),
                "collapses": int(len(local_collapses)),
            }
        )
        vertex_offset += len(local_coarse.vertices)
    coarse = MeshData(
        np.vstack(coarse_vertices),
        np.vstack(coarse_faces).astype(np.int64),
    )
    collapses = (
        np.vstack(collapse_history).astype(np.int64)
        if collapse_history
        else np.zeros((0, 2), dtype=np.int64)
    )
    issues = _mesh_issues(coarse)
    if issues:
        raise RuntimeError(f"coarse mesh invalid after collapse replay: {issues}")
    if len(coarse.vertices) < min_vertices:
        raise RuntimeError(
            f"coarse mesh has {len(coarse.vertices)} vertices; minimum is {min_vertices}"
        )
    original_to_coarse = np.asarray(original_to_coarse, dtype=np.int64)
    if original_to_coarse.shape != (len(gt.vertices),):
        raise RuntimeError("collapse replay returned an invalid original-to-coarse map")
    if np.any(original_to_coarse < 0) or np.any(original_to_coarse >= len(coarse.vertices)):
        raise RuntimeError("collapse replay mapping contains invalid coarse indices")
    return coarse, np.asarray(collapses, dtype=np.int64), original_to_coarse, {
        "componentwise": True,
        "gt_components": int(len(component_faces)),
        "requested_target_vertices": int(target_vertices),
        "actual_collapses": int(len(collapses)),
        "simplification_skipped": bool(len(collapses) == 0),
        "components": component_records,
    }


def _clusters(
    gt_vertices: np.ndarray,
    coarse_vertices: np.ndarray,
    original_to_coarse: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.lexsort((np.arange(len(original_to_coarse)), original_to_coarse))
    counts = np.bincount(original_to_coarse, minlength=len(coarse_vertices))
    if np.any(counts == 0):
        raise RuntimeError("collapse replay produced an empty coarse cluster")
    offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    representatives = np.empty(len(coarse_vertices), dtype=np.int64)
    for coarse_index in range(len(coarse_vertices)):
        members = order[offsets[coarse_index] : offsets[coarse_index + 1]]
        squared = np.sum(
            (gt_vertices[members] - coarse_vertices[coarse_index]) ** 2,
            axis=1,
        )
        representatives[coarse_index] = int(members[int(np.argmin(squared))])
    return offsets, order.astype(np.int64), representatives


def _provenance_face_groups(
    gt_faces: np.ndarray,
    original_to_coarse: np.ndarray,
) -> tuple[dict[int, np.ndarray], dict[tuple[int, int], np.ndarray]]:
    mapped_faces = original_to_coarse[gt_faces]
    vertex_groups: dict[int, list[int]] = {}
    edge_groups: dict[tuple[int, int], list[int]] = {}
    for face_index, mapped in enumerate(mapped_faces):
        values = sorted({int(value) for value in mapped})
        for coarse_index in values:
            vertex_groups.setdefault(coarse_index, []).append(face_index)
        for first_index in range(len(values)):
            for second_index in range(first_index + 1, len(values)):
                key = (values[first_index], values[second_index])
                edge_groups.setdefault(key, []).append(face_index)
    return (
        {key: np.asarray(value, dtype=np.int64) for key, value in vertex_groups.items()},
        {key: np.asarray(value, dtype=np.int64) for key, value in edge_groups.items()},
    )


def _closest_on_provenance_faces(
    query: np.ndarray,
    gt: MeshData,
    candidate_faces: np.ndarray,
) -> tuple[np.ndarray, int, np.ndarray, float]:
    triangles = gt.vertices[gt.faces[candidate_faces]]
    queries = np.repeat(np.asarray(query, dtype=np.float64)[None, :], len(triangles), axis=0)
    closest = trimesh.triangles.closest_point(triangles, queries)
    squared = np.sum((closest - query) ** 2, axis=1)
    local_index = int(np.argmin(squared))
    face_index = int(candidate_faces[local_index])
    triangle = gt.vertices[gt.faces[face_index]]
    point, barycentric = _stable_triangle_correspondence(query, triangle)
    return point, face_index, barycentric, float(np.linalg.norm(point - query))


def _stable_triangle_correspondence(
    query: np.ndarray, triangle: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return a reconstructible closest point and simplex weights.

    `trimesh.triangles.points_to_barycentric` uses a dot-product formula which
    is ill-conditioned on the extremely thin (but valid) triangles present in
    a few Sofa meshes.  Enumerating the triangle interior, its edges and its
    vertices keeps the stored point exactly tied to the stored weights.
    """
    query = np.asarray(query, dtype=np.float64)
    triangle = np.asarray(triangle, dtype=np.float64)
    candidates: list[tuple[np.ndarray, np.ndarray]] = []
    for vertex_index in range(3):
        weights = np.zeros(3, dtype=np.float64)
        weights[vertex_index] = 1.0
        candidates.append((triangle[vertex_index], weights))
    for first, second in ((0, 1), (1, 2), (2, 0)):
        edge = triangle[second] - triangle[first]
        denominator = float(np.dot(edge, edge))
        parameter = 0.0 if denominator == 0.0 else float(
            np.clip(np.dot(query - triangle[first], edge) / denominator, 0.0, 1.0)
        )
        weights = np.zeros(3, dtype=np.float64)
        weights[first] = 1.0 - parameter
        weights[second] = parameter
        candidates.append((weights @ triangle, weights))
    basis = np.column_stack((triangle[0] - triangle[2], triangle[1] - triangle[2]))
    solution, _, rank, _ = np.linalg.lstsq(basis, query - triangle[2], rcond=None)
    interior_weights = np.asarray(
        [solution[0], solution[1], 1.0 - solution.sum()], dtype=np.float64
    )
    if rank == 2 and np.all(interior_weights >= -1e-10):
        interior_weights = np.maximum(interior_weights, 0.0)
        interior_weights /= interior_weights.sum()
        candidates.append((interior_weights @ triangle, interior_weights))
    point, weights = min(
        candidates,
        key=lambda candidate: float(np.sum((candidate[0] - query) ** 2)),
    )
    return np.asarray(point, dtype=np.float64), np.asarray(weights, dtype=np.float64)


def _vertex_components(mesh: MeshData) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    parent = np.arange(len(mesh.vertices), dtype=np.int64)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    for first, second in _unique_edges(mesh.faces):
        root_first, root_second = find(int(first)), find(int(second))
        if root_first != root_second:
            if root_first < root_second:
                parent[root_second] = root_first
            else:
                parent[root_first] = root_second
    roots = np.asarray([find(index) for index in range(len(parent))], dtype=np.int64)
    _, labels = np.unique(roots, return_inverse=True)
    face_components = labels[mesh.faces]
    if np.any(face_components != face_components[:, [0]]):
        raise RuntimeError("GT face spans multiple connected-component labels")
    grouped: dict[int, list[int]] = {}
    for face_index, component in enumerate(face_components[:, 0]):
        grouped.setdefault(int(component), []).append(face_index)
    return labels.astype(np.int64), {
        key: np.asarray(values, dtype=np.int64) for key, values in grouped.items()
    }


def build_provenance_targets(
    gt: MeshData,
    coarse: MeshData,
    original_to_coarse: np.ndarray,
    expanded: MeshData,
    subdivision: dict[str, np.ndarray],
    max_local_distance: float = 0.05,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    offsets, cluster_vertices, representatives = _clusters(
        gt.vertices, coarse.vertices, original_to_coarse
    )
    target = np.full((len(expanded.vertices), 3), np.nan, dtype=np.float64)
    kind = np.full(len(expanded.vertices), 255, dtype=np.uint8)
    gt_face_indices = np.full(len(expanded.vertices), -1, dtype=np.int64)
    gt_indices = np.full((len(expanded.vertices), 3), -1, dtype=np.int64)
    gt_weights = np.zeros((len(expanded.vertices), 3), dtype=np.float64)
    candidate_counts = np.zeros(len(expanded.vertices), dtype=np.int64)
    vertex_faces, edge_faces = _provenance_face_groups(gt.faces, original_to_coarse)
    gt_components, component_faces = _vertex_components(gt)
    coarse_components = np.empty(len(coarse.vertices), dtype=np.int64)
    for coarse_index in range(len(coarse.vertices)):
        members = cluster_vertices[offsets[coarse_index] : offsets[coarse_index + 1]]
        components = np.unique(gt_components[members])
        if len(components) != 1:
            raise RuntimeError(
                f"coarse cluster {coarse_index} spans {len(components)} GT components"
            )
        coarse_components[coarse_index] = int(components[0])
    expanded_components = np.full(len(expanded.vertices), -1, dtype=np.int64)
    component_fallback_count = 0

    pre_to_final = np.asarray(subdivision["pre_compaction_to_final"], dtype=np.int64)
    final_to_pre = np.asarray(subdivision["final_to_pre_compaction"], dtype=np.int64)
    for coarse_index in range(len(coarse.vertices)):
        final_index = int(pre_to_final[coarse_index])
        if final_index < 0:
            continue
        candidates = vertex_faces.get(coarse_index)
        if candidates is None or len(candidates) == 0:
            raise RuntimeError(f"coarse vertex {coarse_index} has no provenance GT faces")
        point, face_index, barycentric, distance = _closest_on_provenance_faces(
            expanded.vertices[final_index], gt, candidates
        )
        component = int(coarse_components[coarse_index])
        expanded_components[final_index] = component
        mapping_kind = 0
        if distance > max_local_distance:
            point, face_index, barycentric, distance = _closest_on_provenance_faces(
                expanded.vertices[final_index], gt, component_faces[component]
            )
            mapping_kind = 2
            component_fallback_count += 1
        target[final_index] = point
        kind[final_index] = mapping_kind
        gt_face_indices[final_index] = face_index
        gt_indices[final_index] = gt.faces[face_index]
        gt_weights[final_index] = barycentric
        candidate_counts[final_index] = len(candidates)

    missing: list[tuple[int, int]] = []
    parent_edges = np.asarray(subdivision["parent_edges"], dtype=np.int64)
    children = np.asarray(subdivision["new_vertex_indices"], dtype=np.int64)
    for parents, child in zip(parent_edges, children, strict=True):
        parent_pre = final_to_pre[parents]
        if np.any(parent_pre >= len(coarse.vertices)):
            raise RuntimeError("midpoint parent is not a coarse vertex for one-step subdivision")
        key = tuple(sorted((int(parent_pre[0]), int(parent_pre[1]))))
        parent_components = np.unique(coarse_components[parent_pre])
        if len(parent_components) != 1:
            raise RuntimeError(f"coarse edge {key} spans multiple GT components")
        component = int(parent_components[0])
        expanded_components[child] = component
        candidates = edge_faces.get(key)
        if candidates is None or len(candidates) == 0:
            missing.append(key)
            continue
        point, face_index, barycentric, distance = _closest_on_provenance_faces(
            expanded.vertices[child], gt, candidates
        )
        mapping_kind = 1
        if distance > max_local_distance:
            point, face_index, barycentric, distance = _closest_on_provenance_faces(
                expanded.vertices[child], gt, component_faces[component]
            )
            mapping_kind = 3
            component_fallback_count += 1
        target[child] = point
        kind[child] = mapping_kind
        gt_face_indices[child] = face_index
        gt_indices[child] = gt.faces[face_index]
        gt_weights[child] = barycentric
        candidate_counts[child] = len(candidates)

    if missing:
        raise RuntimeError(
            f"{len(missing)} expanded midpoint edges lack a collapse-provenance GT boundary"
        )
    unmapped = np.flatnonzero(~np.isfinite(target).all(axis=1))
    if len(unmapped):
        raise RuntimeError(f"{len(unmapped)} expanded vertices have no provenance target")
    if np.any(expanded_components < 0):
        raise RuntimeError("expanded component mapping is incomplete")
    if not np.allclose(gt_weights.sum(axis=1), 1.0, atol=2e-6, rtol=0.0):
        raise RuntimeError("provenance target weights do not sum to one")

    cluster_sizes = np.diff(offsets)
    diagnostics = {
        "mapping_coverage": 1.0,
        "coarse_cluster_size_min": int(cluster_sizes.min()),
        "coarse_cluster_size_mean": float(cluster_sizes.mean()),
        "coarse_cluster_size_max": int(cluster_sizes.max()),
        "coarse_local_face_candidates_min": int(candidate_counts[np.isin(kind, [0, 2])].min()),
        "coarse_local_face_candidates_mean": float(candidate_counts[np.isin(kind, [0, 2])].mean()),
        "coarse_local_face_candidates_max": int(candidate_counts[np.isin(kind, [0, 2])].max()),
        "midpoint_local_face_candidates_min": int(candidate_counts[np.isin(kind, [1, 3])].min()),
        "midpoint_local_face_candidates_mean": float(candidate_counts[np.isin(kind, [1, 3])].mean()),
        "midpoint_local_face_candidates_max": int(candidate_counts[np.isin(kind, [1, 3])].max()),
        "coarse_local_projection_count": int(np.sum(kind == 0)),
        "midpoint_local_projection_count": int(np.sum(kind == 1)),
        "coarse_component_fallback_count": int(np.sum(kind == 2)),
        "midpoint_component_fallback_count": int(np.sum(kind == 3)),
        "component_fallback_count": component_fallback_count,
        "gt_connected_components": int(len(component_faces)),
        "max_provenance_local_distance": max_local_distance,
    }
    mapping = {
        "original_gt_to_coarse": original_to_coarse.astype(np.int64),
        "coarse_cluster_offsets": offsets,
        "coarse_cluster_gt_vertices": cluster_vertices,
        "coarse_representative_gt_vertex": representatives,
        "gt_vertex_component": gt_components,
        "coarse_gt_component": coarse_components,
        "expanded_gt_component": expanded_components,
        "expanded_mapping_kind": kind,
        "expanded_gt_face_index": gt_face_indices,
        "expanded_gt_vertex_indices": gt_indices,
        "expanded_gt_weights": gt_weights.astype(np.float64),
        "expanded_boundary_candidate_count": candidate_counts,
    }
    return target.astype(np.float64), mapping, diagnostics


def _topology_safe_target(
    expanded: MeshData,
    surface_correspondence: np.ndarray,
    minimum_alpha: float = 0.95,
) -> tuple[np.ndarray, float, int]:
    initial = np.asarray(expanded.vertices, dtype=np.float64)
    projected = np.asarray(surface_correspondence, dtype=np.float64)

    def collapsed_count(vertices: np.ndarray) -> int:
        triangles = vertices[expanded.faces]
        double_area = np.linalg.norm(
            np.cross(
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 0],
            ),
            axis=1,
        )
        return int(np.sum(~np.isfinite(double_area) | (double_area <= 0)))

    projected_collapsed = collapsed_count(projected)
    for alpha in (1.0, 0.9999, 0.999, 0.995, 0.99, 0.98, 0.95):
        target = initial + alpha * (projected - initial)
        if collapsed_count(target) == 0:
            if alpha < minimum_alpha:
                break
            return target.astype(np.float64), float(alpha), projected_collapsed
    raise RuntimeError(
        "local surface correspondence cannot preserve expanded connectivity "
        f"above alpha={minimum_alpha}; projected collapsed faces={projected_collapsed}"
    )


def register_coarse_to_gt_component(
    gt: MeshData,
    coarse_qem: MeshData,
    original_to_coarse: np.ndarray,
) -> tuple[MeshData, dict[str, np.ndarray], dict[str, Any]]:
    offsets, cluster_vertices, _ = _clusters(
        gt.vertices, coarse_qem.vertices, original_to_coarse
    )
    gt_components, component_faces = _vertex_components(gt)
    registered_surface = np.empty_like(coarse_qem.vertices, dtype=np.float64)
    coarse_components = np.empty(len(coarse_qem.vertices), dtype=np.int64)
    face_indices = np.empty(len(coarse_qem.vertices), dtype=np.int64)
    gt_indices = np.empty((len(coarse_qem.vertices), 3), dtype=np.int64)
    gt_weights = np.empty((len(coarse_qem.vertices), 3), dtype=np.float64)
    candidate_counts = np.empty(len(coarse_qem.vertices), dtype=np.int64)
    for coarse_index in range(len(coarse_qem.vertices)):
        members = cluster_vertices[offsets[coarse_index] : offsets[coarse_index + 1]]
        components = np.unique(gt_components[members])
        if len(components) != 1:
            raise RuntimeError(
                f"coarse cluster {coarse_index} spans {len(components)} GT components"
            )
        component = int(components[0])
        candidates = component_faces[component]
        point, face_index, barycentric, _ = _closest_on_provenance_faces(
            coarse_qem.vertices[coarse_index], gt, candidates
        )
        registered_surface[coarse_index] = point
        coarse_components[coarse_index] = component
        face_indices[coarse_index] = face_index
        gt_indices[coarse_index] = gt.faces[face_index]
        gt_weights[coarse_index] = barycentric
        candidate_counts[coarse_index] = len(candidates)
    registered_vertices, alpha, collapsed_faces = _topology_safe_target(
        coarse_qem, registered_surface
    )
    registered = MeshData(registered_vertices.astype(np.float64), coarse_qem.faces.copy())
    issues = _mesh_issues(registered)
    if issues:
        raise RuntimeError(f"component-registered coarse mesh is invalid: {issues}")
    displacement = np.linalg.norm(registered_vertices - coarse_qem.vertices, axis=1)
    mapping = {
        "qem_vertices": np.asarray(coarse_qem.vertices, dtype=np.float64),
        "registered_surface_positions": registered_surface.astype(np.float64),
        "registered_vertices": registered_vertices.astype(np.float64),
        "coarse_gt_component": coarse_components,
        "coarse_gt_face_index": face_indices,
        "coarse_gt_vertex_indices": gt_indices,
        "coarse_gt_barycentric_weights": gt_weights.astype(np.float64),
        "component_candidate_face_count": candidate_counts,
        "topology_safe_blend_alpha": np.asarray(alpha, dtype=np.float64),
    }
    diagnostics = {
        "topology_safe_blend_alpha": alpha,
        "surface_registration_collapsed_faces": collapsed_faces,
        "displacement": _stats(displacement),
        "gt_connected_components": int(len(component_faces)),
    }
    return registered, mapping, diagnostics


def _point_to_surface(points: np.ndarray, surface: MeshData) -> np.ndarray:
    mesh = trimesh.Trimesh(surface.vertices, surface.faces, process=False)
    _, distances, _ = trimesh.proximity.closest_point(mesh, points)
    return np.asarray(distances, dtype=np.float64)


def _stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
        "rms": float(np.sqrt(np.mean(values**2))),
    }


def _chamfer(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    a_to_b = cKDTree(b).query(a, workers=-1)[0]
    b_to_a = cKDTree(a).query(b, workers=-1)[0]
    return {
        "mean_l2": float(0.5 * (a_to_b.mean() + b_to_a.mean())),
        "mean_squared_l2": float(0.5 * (np.mean(a_to_b**2) + np.mean(b_to_a**2))),
        "a_to_b_mean": float(a_to_b.mean()),
        "b_to_a_mean": float(b_to_a.mean()),
    }


def _pair_diagnostics(first: MeshData, second: MeshData) -> dict[str, Any]:
    return {
        "first_vertices_to_second_surface": _stats(
            _point_to_surface(first.vertices, second)
        ),
        "second_vertices_to_first_surface": _stats(
            _point_to_surface(second.vertices, first)
        ),
        "vertex_chamfer": _chamfer(first.vertices, second.vertices),
    }


def _diagnostics(
    gt: MeshData,
    coarse_raw: MeshData,
    expanded_initial_raw: MeshData,
    coarse_registered_oracle: MeshData,
    p_target_oracle: MeshData,
) -> dict[str, Any]:
    displacement = np.linalg.norm(
        p_target_oracle.vertices - expanded_initial_raw.vertices, axis=1
    )
    return {
        "paired_expanded_initial_raw_to_P_target_oracle_displacement": _stats(
            displacement
        ),
        "distances": {
            "gt_vs_coarse_raw": _pair_diagnostics(gt, coarse_raw),
            "gt_vs_expanded_initial_raw": _pair_diagnostics(
                gt, expanded_initial_raw
            ),
            "gt_vs_coarse_registered_oracle": _pair_diagnostics(
                gt, coarse_registered_oracle
            ),
            "gt_vs_P_target_oracle": _pair_diagnostics(gt, p_target_oracle),
            "expanded_initial_raw_vs_P_target_oracle": _pair_diagnostics(
                expanded_initial_raw, p_target_oracle
            ),
        },
    }


def _comparison_obj(path: Path, initial: MeshData, target: MeshData) -> None:
    extent = max(float(np.max(np.ptp(initial.vertices, axis=0))), 1.0)
    left = initial.vertices.copy()
    right = target.vertices.copy()
    left[:, 0] -= extent * 0.65
    right[:, 0] += extent * 0.65
    lines = ["o expanded_initial_raw"]
    lines.extend(f"v {x:.9g} {y:.9g} {z:.9g}" for x, y, z in left)
    lines.extend(f"f {a+1} {b+1} {c+1}" for a, b, c in initial.faces)
    lines.append("o P_target_oracle")
    lines.extend(f"v {x:.9g} {y:.9g} {z:.9g}" for x, y, z in right)
    offset = len(left) + 1
    lines.extend(f"f {a+offset} {b+offset} {c+offset}" for a, b, c in target.faces)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _preview(path: Path, model_id: str, gt: MeshData, expanded: MeshData, target: MeshData) -> None:
    figure = plt.figure(figsize=(12, 4), facecolor="white")
    for index, (title, mesh, color) in enumerate(
        (("GT", gt, "#777777"), ("Expanded initial raw", expanded, "#3274a1"), ("P target oracle", target, "#d06b35"))
    ):
        axis = figure.add_subplot(1, 3, index + 1, projection="3d")
        count = min(len(mesh.vertices), 5000)
        sample = np.linspace(0, len(mesh.vertices) - 1, count, dtype=np.int64)
        points = mesh.vertices[sample]
        axis.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.35, c=color)
        axis.set_title(title)
        axis.set_box_aspect((1, 1, 1))
        axis.view_init(elev=20, azim=-65)
        axis.set_axis_off()
    figure.suptitle(model_id, fontsize=9)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _visual_contact_sheet(output_root: Path, model_ids: list[str]) -> None:
    width, height = 900, 340
    columns = 2
    rows = math.ceil(len(model_ids) / columns)
    sheet = Image.new("RGB", (columns * width, rows * height), "white")
    for index, model_id in enumerate(model_ids):
        image = Image.open(output_root / "visualizations" / f"{model_id}.png").convert("RGB")
        image = ImageOps.contain(image, (width, height))
        x = (index % columns) * width + (width - image.width) // 2
        y = (index // columns) * height + (height - image.height) // 2
        sheet.paste(image, (x, y))
    sheet.save(output_root / "visualizations" / "contact_sheet.png", optimize=True)


def _read_splits(source_root: Path) -> tuple[dict[str, list[str]], dict[str, str]]:
    splits: dict[str, list[str]] = {}
    owner: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        ids = [
            line.strip()
            for line in (source_root / f"{split}.txt").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        splits[split] = ids
        for model_id in ids:
            if model_id in owner:
                raise RuntimeError(f"model {model_id} appears in multiple splits")
            owner[model_id] = split
    return splits, owner


def _parameters(
    source_root: Path,
    coarse_target_vertices: int,
    coarse_min_vertices: int,
    subdivision_steps: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "format_version": DATASET_FORMAT,
        "source_root": str(source_root.resolve()),
        "source_mesh": "mesh.obj",
        "coarse_algorithm": "componentwise_fast_simplification_qem_with_collapse_replay",
        "coarse_target_vertices": coarse_target_vertices,
        "coarse_min_vertices": coarse_min_vertices,
        "simplifier_aggressiveness": 7.0,
        "coarse_generation_policy": "componentwise_to_prevent_cross_component_collapses",
        "model_query_coarse": "raw_qem_simplifier_output",
        "model_query_expanded": "midpoint_subdivide_of_raw_qem_coarse_only",
        "coarse_registration": "oracle_diagnostic_only_not_query_input",
        "dataset_primary_use": "frozen_model_inference_and_reconstruction_evaluation",
        "training_loader_compatible": False,
        "training_supervision_contract": "GT_only_delta_gt_equals_L_gt_faces_at_gt_vertices",
        "oracle_target_role": "diagnostic_upper_bound_only_not_training_supervision",
        "subdivision_algorithm": "project_existing_midpoint_subdivide",
        "subdivision_steps": subdivision_steps,
        "target_constructor": TARGET_CONSTRUCTOR,
        "max_provenance_local_distance": 0.05,
        "max_component_correspondence_displacement": 0.25,
        "global_nearest_surface_used_for_correspondence": False,
        "nearest_surface_used_for_diagnostics_only": True,
        "seed": seed,
    }


def _cache_valid(model_dir: Path, source_checksum: str, parameters: dict[str, Any]) -> bool:
    metadata_path = model_dir / "metadata.json"
    if not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    required = (
        "gt_mesh.obj", "gt_mesh.npz", "coarse_raw.obj", "coarse_raw.npz",
        "expanded_initial_raw.obj", "expanded_initial_raw.npz",
        "P_target_oracle.obj", "P_target_oracle.npz",
        "correspondence_oracle.npz", "subdivision_mapping_raw.npz",
        "coarse_registered_oracle.obj", "coarse_registered_oracle.npz",
        "coarse_registration_oracle.npz", "surface_target_oracle.npz",
        "diagnostics.json",
    )
    return (
        metadata.get("status") == "valid"
        and metadata.get("source_checksum") == source_checksum
        and metadata.get("parameters") == parameters
        and all((model_dir / name).is_file() for name in required)
    )


def _row_from_metadata(metadata: dict[str, Any], metadata_path: Path) -> dict[str, Any]:
    counts = metadata["counts"]
    files = metadata["files"]
    return {
        "model_id": metadata["model_id"],
        "split": metadata["split"],
        "status": metadata["status"],
        "failure_reason": metadata.get("failure_reason", ""),
        "warning_count": len(metadata.get("warnings", [])),
        "warnings": "|".join(metadata.get("warnings", [])),
        "gt_vertices": counts["gt_vertices"],
        "gt_faces": counts["gt_faces"],
        "coarse_raw_vertices": counts["coarse_raw_vertices"],
        "coarse_raw_faces": counts["coarse_raw_faces"],
        "expanded_initial_raw_vertices": counts["expanded_initial_raw_vertices"],
        "expanded_initial_raw_faces": counts["expanded_initial_raw_faces"],
        "P_target_oracle_vertices": counts["P_target_oracle_vertices"],
        "P_target_oracle_faces": counts["P_target_oracle_faces"],
        "gt_obj": files["gt_mesh"]["obj"],
        "gt_npz": files["gt_mesh"]["npz"],
        "coarse_raw_obj": files["coarse_raw"]["obj"],
        "coarse_raw_npz": files["coarse_raw"]["npz"],
        "expanded_initial_raw_obj": files["expanded_initial_raw"]["obj"],
        "expanded_initial_raw_npz": files["expanded_initial_raw"]["npz"],
        "P_target_oracle_obj": files["P_target_oracle"]["obj"],
        "P_target_oracle_npz": files["P_target_oracle"]["npz"],
        "correspondence_oracle_npz": files["correspondence_oracle"],
        "coarse_registered_oracle_obj": files["coarse_registered_oracle"]["obj"],
        "coarse_registered_oracle_npz": files["coarse_registered_oracle"]["npz"],
        "coarse_registration_oracle_npz": files["coarse_registration_oracle"],
        "surface_target_oracle_npz": files["surface_target_oracle"],
        "subdivision_mapping_raw_npz": files["subdivision_mapping_raw"],
        "diagnostics_json": files["diagnostics"],
        "comparison_obj": files["comparison_obj"],
        "metadata_json": str(metadata_path.resolve()),
        "target_constructor": metadata["parameters"]["target_constructor"],
        "subdivision_steps": metadata["parameters"]["subdivision_steps"],
        "coarse_target_vertices": metadata["parameters"]["coarse_target_vertices"],
    }


def _process_one(
    model_id: str,
    split: str,
    source_root: Path,
    output_root: Path,
    parameters: dict[str, Any],
    force: bool,
) -> tuple[dict[str, Any], tuple[MeshData, MeshData, MeshData] | None]:
    source_path = source_root / model_id / "mesh.obj"
    gt = _load_obj(source_path)
    source_issues = _mesh_issues(gt)
    if source_issues:
        raise RuntimeError(f"source GT mesh invalid: {source_issues}")
    source_checksum = _checksum(gt)
    model_dir = output_root / "models" / model_id
    metadata_path = model_dir / "metadata.json"
    if not force and _cache_valid(model_dir, source_checksum, parameters):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return _row_from_metadata(metadata, metadata_path), None

    model_dir.mkdir(parents=True, exist_ok=True)
    # A previously failed run may have left this marker behind.  The current
    # attempt owns the status from this point onward.
    (model_dir / "failure.json").unlink(missing_ok=True)
    coarse_raw, collapses, original_to_coarse, simplify_meta = simplify_with_provenance(
        gt,
        int(parameters["coarse_target_vertices"]),
        int(parameters["coarse_min_vertices"]),
    )
    # The model query path ends here and is fixed before any GT-guided
    # registration or correspondence is computed.
    expanded_arrays, subdivision = midpoint_subdivide(
        coarse_raw.vertices,
        coarse_raw.faces,
        steps=int(parameters["subdivision_steps"]),
    )
    expanded_initial_raw = MeshData(
        np.asarray(expanded_arrays.vertices, dtype=np.float64),
        np.asarray(expanded_arrays.faces, dtype=np.int64),
    )

    # Oracle-only path.  Registered coarse geometry is retained for diagnostics
    # but is never used to generate the query mesh above.
    coarse_registered_oracle, coarse_registration, coarse_registration_diagnostics = (
        register_coarse_to_gt_component(gt, coarse_raw, original_to_coarse)
    )
    surface_target_oracle, correspondence, mapping_diagnostics = build_provenance_targets(
        gt,
        coarse_raw,
        original_to_coarse,
        expanded_initial_raw,
        subdivision,
        max_local_distance=float(parameters["max_provenance_local_distance"]),
    )
    surface_displacement = np.linalg.norm(
        np.asarray(surface_target_oracle)
        - np.asarray(expanded_initial_raw.vertices),
        axis=1,
    )
    if float(surface_displacement.max()) > float(
        parameters["max_component_correspondence_displacement"]
    ) + 1e-6:
        raise RuntimeError(
            "provenance/component-constrained correspondence exceeds maximum displacement: "
            f"{float(surface_displacement.max())}"
        )
    p_target_oracle, target_alpha, projected_collapsed_faces = _topology_safe_target(
        expanded_initial_raw, surface_target_oracle
    )
    correspondence["surface_target_oracle_positions"] = surface_target_oracle
    correspondence["topology_safe_blend_alpha"] = np.asarray(target_alpha, dtype=np.float64)
    mapping_diagnostics["topology_safe_blend_alpha"] = target_alpha
    mapping_diagnostics["surface_correspondence_collapsed_faces"] = projected_collapsed_faces
    mapping_diagnostics["surface_displacement_max"] = float(surface_displacement.max())
    warnings: list[str] = []
    if float(surface_displacement.max()) > 0.1:
        warnings.append("large_but_component_constrained_displacement")
    if target_alpha < 1.0:
        warnings.append("topology_safe_target_blend_applied")
    target_oracle = MeshData(
        p_target_oracle.astype(np.float64), expanded_initial_raw.faces.copy()
    )
    issues = {
        "gt": _mesh_issues(gt),
        "coarse_raw": _mesh_issues(coarse_raw),
        "expanded_initial_raw": _mesh_issues(expanded_initial_raw),
        "coarse_registered_oracle": _mesh_issues(coarse_registered_oracle),
        "P_target_oracle": _mesh_issues(target_oracle),
    }
    structural_issues = [f"{name}:{issue}" for name, values in issues.items() for issue in values]
    if len(expanded_initial_raw.vertices) != len(p_target_oracle):
        structural_issues.append("raw_query_oracle_target_vertex_count_mismatch")
    if not np.array_equal(expanded_initial_raw.faces, target_oracle.faces):
        structural_issues.append("raw_query_oracle_target_connectivity_mismatch")
    if structural_issues:
        raise RuntimeError(f"generated sample failed structural validation: {structural_issues}")

    files = {
        "gt_mesh": _save_mesh(model_dir / "gt_mesh", gt),
        "coarse_raw": _save_mesh(model_dir / "coarse_raw", coarse_raw),
        "expanded_initial_raw": _save_mesh(
            model_dir / "expanded_initial_raw", expanded_initial_raw
        ),
        "coarse_registered_oracle": _save_mesh(
            model_dir / "coarse_registered_oracle", coarse_registered_oracle
        ),
        "P_target_oracle": _save_mesh(
            model_dir / "P_target_oracle", target_oracle
        ),
    }
    np.savez_compressed(
        model_dir / "surface_target_oracle.npz",
        vertices=np.asarray(surface_target_oracle, dtype=np.float64),
        faces=expanded_initial_raw.faces.astype(np.int64),
    )
    np.savez_compressed(
        model_dir / "correspondence_oracle.npz",
        collapse_history=collapses,
        **correspondence,
    )
    np.savez_compressed(
        model_dir / "coarse_registration_oracle.npz", **coarse_registration
    )
    np.savez_compressed(model_dir / "subdivision_mapping_raw.npz", **subdivision)
    diagnostic_payload = _diagnostics(
        gt,
        coarse_raw,
        expanded_initial_raw,
        coarse_registered_oracle,
        target_oracle,
    )
    diagnostic_payload["mapping"] = mapping_diagnostics
    diagnostic_payload["coarse_registration"] = coarse_registration_diagnostics
    _json(model_dir / "diagnostics.json", diagnostic_payload)
    _comparison_obj(
        model_dir / "comparison_oracle.obj", expanded_initial_raw, target_oracle
    )

    files.update(
        {
            "correspondence_oracle": str(
                (model_dir / "correspondence_oracle.npz").resolve()
            ),
            "coarse_registration_oracle": str(
                (model_dir / "coarse_registration_oracle.npz").resolve()
            ),
            "surface_target_oracle": str(
                (model_dir / "surface_target_oracle.npz").resolve()
            ),
            "subdivision_mapping_raw": str(
                (model_dir / "subdivision_mapping_raw.npz").resolve()
            ),
            "diagnostics": str((model_dir / "diagnostics.json").resolve()),
            "comparison_obj": str((model_dir / "comparison_oracle.obj").resolve()),
        }
    )
    metadata = {
        "model_id": model_id,
        "split": split,
        "status": "valid",
        "failure_reason": "",
        "warnings": warnings,
        "source_path": str(source_path.resolve()),
        "source_checksum": source_checksum,
        "parameters": parameters,
        "counts": {
            "gt_vertices": int(len(gt.vertices)),
            "gt_faces": int(len(gt.faces)),
            "coarse_raw_vertices": int(len(coarse_raw.vertices)),
            "coarse_raw_faces": int(len(coarse_raw.faces)),
            "expanded_initial_raw_vertices": int(len(expanded_initial_raw.vertices)),
            "expanded_initial_raw_faces": int(len(expanded_initial_raw.faces)),
            "coarse_registered_oracle_vertices": int(
                len(coarse_registered_oracle.vertices)
            ),
            "coarse_registered_oracle_faces": int(
                len(coarse_registered_oracle.faces)
            ),
            "P_target_oracle_vertices": int(len(target_oracle.vertices)),
            "P_target_oracle_faces": int(len(target_oracle.faces)),
        },
        "connectivity": {
            "expanded_initial_raw_and_P_target_oracle_faces_identical": True,
            "expanded_initial_raw_vertex_order_matches_P_target_oracle": True,
            "expanded_initial_raw_source": "coarse_raw",
            "registered_coarse_used_for_query": False,
        },
        "file_roles": {
            "coarse_raw": "model_input; direct component-wise QEM simplifier output",
            "expanded_initial_raw": "model_query; midpoint subdivision of coarse_raw only",
            "P_target_oracle": "oracle_diagnostic_upper_bound_only; never training supervision",
            "correspondence_oracle": "oracle_diagnostic_construction_only",
            "coarse_registered_oracle": "oracle_diagnostic_only; never a model query input",
            "coarse_registration_oracle": "oracle_diagnostic_mapping_only",
            "surface_target_oracle": "unblended_oracle_surface_target_only",
        },
        "usage_contract": {
            "primary_dataset_use": "frozen_model_inference_and_reconstruction_evaluation",
            "may_connect_to_training_loader": False,
            "training_uses_only_GT_mesh": True,
            "training_target": "delta_gt = L(gt_faces) @ gt_vertices",
            "inference_path": "coarse_raw -> expanded_initial_raw -> frozen_model_prediction -> Laplacian_integration/reconstruction",
            "P_target_oracle_role": "diagnostic_upper_bound_only",
            "expanded_graph_oracle_laplacian_is_training_supervision": False,
        },
        "simplification": simplify_meta,
        "mapping_diagnostics": mapping_diagnostics,
        "files": files,
        "limitations": [
            "P_target_oracle is a GT-guided proxy induced by collapse provenance, not a proven semantic correspondence or bijective dense registration.",
            "Collapse provenance is only a target-search constraint.",
            "Coarse targets are projected only within GT faces incident to their collapse cluster.",
            "Midpoint targets are projected only within GT faces spanning both parent collapse clusters.",
            "coarse_registered_oracle is computed only after expanded_initial_raw is fixed and never enters the model query path.",
            "Low-confidence local mappings may search only within the same provenance-matched GT connected component; this is recorded with mapping kinds 2/3.",
            "If local surface correspondences collapse a target face, one global displacement alpha is line-searched; the unblended surface positions and alpha are saved.",
            "Nearest-surface queries are diagnostic only and do not define correspondence.",
        ],
    }
    _json(metadata_path, metadata)
    return _row_from_metadata(metadata, metadata_path), (
        gt,
        expanded_initial_raw,
        target_oracle,
    )


def _validate_model(model_dir: Path) -> list[str]:
    issues: list[str] = []
    try:
        gt = _load_npz_mesh(model_dir / "gt_mesh.npz")
        coarse_raw = _load_npz_mesh(model_dir / "coarse_raw.npz")
        coarse_registered_oracle = _load_npz_mesh(
            model_dir / "coarse_registered_oracle.npz"
        )
        expanded_initial_raw = _load_npz_mesh(
            model_dir / "expanded_initial_raw.npz"
        )
        target_oracle = _load_npz_mesh(model_dir / "P_target_oracle.npz")
        for name, mesh in (
            ("gt", gt),
            ("coarse_raw", coarse_raw),
            ("coarse_registered_oracle", coarse_registered_oracle),
            ("expanded_initial_raw", expanded_initial_raw),
            ("P_target_oracle", target_oracle),
        ):
            issues.extend(f"{name}:{value}" for value in _mesh_issues(mesh))
        if coarse_raw.vertices.shape != coarse_registered_oracle.vertices.shape:
            issues.append("coarse_registration_shape_mismatch")
        if not np.array_equal(coarse_raw.faces, coarse_registered_oracle.faces):
            issues.append("coarse_registration_faces_mismatch")
        if target_oracle.vertices.shape != expanded_initial_raw.vertices.shape:
            issues.append("P_target_oracle_shape_mismatch")
        if not np.array_equal(target_oracle.faces, expanded_initial_raw.faces):
            issues.append("P_target_oracle_faces_mismatch")

        # Explicit leakage test: the saved query must be exactly reproducible
        # from raw QEM coarse and must not come from the registered oracle coarse.
        reconstructed_arrays, reconstructed_mapping = midpoint_subdivide(
            coarse_raw.vertices, coarse_raw.faces, steps=1
        )
        if not np.array_equal(
            np.asarray(reconstructed_arrays.faces, dtype=np.int64),
            expanded_initial_raw.faces,
        ):
            issues.append("raw_query_faces_not_reconstructed_from_coarse_raw")
        if not np.allclose(
            np.asarray(reconstructed_arrays.vertices, dtype=np.float64),
            expanded_initial_raw.vertices,
            atol=0.0,
            rtol=0.0,
        ):
            issues.append("raw_query_vertices_not_reconstructed_from_coarse_raw")
        registered_arrays, _ = midpoint_subdivide(
            coarse_registered_oracle.vertices,
            coarse_registered_oracle.faces,
            steps=1,
        )
        registration_displacement = np.linalg.norm(
            coarse_raw.vertices - coarse_registered_oracle.vertices, axis=1
        )
        if float(registration_displacement.max()) > 1e-12 and np.allclose(
            expanded_initial_raw.vertices,
            np.asarray(registered_arrays.vertices, dtype=np.float64),
            atol=1e-12,
            rtol=0.0,
        ):
            issues.append("gt_registered_coarse_leaked_into_query")

        correspondence = np.load(model_dir / "correspondence_oracle.npz")
        if len(correspondence["expanded_mapping_kind"]) != len(
            expanded_initial_raw.vertices
        ):
            issues.append("correspondence_length_mismatch")
        if not np.all(np.isin(correspondence["expanded_mapping_kind"], [0, 1, 2, 3])):
            issues.append("invalid_mapping_kind")
        if not np.allclose(
            correspondence["expanded_gt_weights"].sum(axis=1), 1.0, atol=2e-6, rtol=0.0
        ):
            issues.append("mapping_weight_sum_mismatch")
        face_indices = np.asarray(correspondence["expanded_gt_face_index"], dtype=np.int64)
        gt_indices = np.asarray(correspondence["expanded_gt_vertex_indices"], dtype=np.int64)
        weights = np.asarray(correspondence["expanded_gt_weights"], dtype=np.float64)
        if face_indices.shape != (len(expanded_initial_raw.vertices),):
            issues.append("mapping_face_index_shape_mismatch")
        elif np.any(face_indices < 0) or np.any(face_indices >= len(gt.faces)):
            issues.append("mapping_face_index_out_of_range")
        else:
            if gt_indices.shape != (len(expanded_initial_raw.vertices), 3):
                issues.append("mapping_gt_vertex_index_shape_mismatch")
            elif not np.array_equal(gt_indices, gt.faces[face_indices]):
                issues.append("mapping_face_vertex_indices_mismatch")
            if weights.shape != (
                len(expanded_initial_raw.vertices),
                3,
            ) or not np.isfinite(weights).all():
                issues.append("mapping_weights_invalid")
            else:
                if np.any(weights < -1e-10):
                    issues.append("mapping_weights_outside_simplex")
                reconstructed_surface = np.einsum(
                    "ni,nij->nj", weights, gt.vertices[gt.faces[face_indices]]
                )
                saved_surface = np.asarray(
                    correspondence["surface_target_oracle_positions"],
                    dtype=np.float64,
                )
                if not np.allclose(reconstructed_surface, saved_surface, atol=3e-6, rtol=0.0):
                    issues.append("surface_target_oracle_reconstruction_mismatch")
                saved_surface_file = np.load(model_dir / "surface_target_oracle.npz")
                if not np.allclose(
                    saved_surface_file["vertices"], saved_surface, atol=0.0, rtol=0.0
                ) or not np.array_equal(
                    saved_surface_file["faces"], expanded_initial_raw.faces
                ):
                    issues.append("surface_target_oracle_file_mismatch")
                alpha = float(correspondence["topology_safe_blend_alpha"])
                reconstructed_target = expanded_initial_raw.vertices + alpha * (
                    saved_surface - expanded_initial_raw.vertices
                )
                if not np.allclose(
                    reconstructed_target,
                    target_oracle.vertices,
                    atol=3e-6,
                    rtol=0.0,
                ):
                    issues.append("P_target_oracle_reconstruction_mismatch")
        original_to_coarse = np.asarray(
            correspondence["original_gt_to_coarse"], dtype=np.int64
        )
        if original_to_coarse.shape != (len(gt.vertices),):
            issues.append("original_gt_to_coarse_shape_mismatch")
        elif np.any(original_to_coarse < 0) or np.any(
            original_to_coarse >= len(coarse_raw.vertices)
        ):
            issues.append("original_gt_to_coarse_out_of_range")
        collapse_history = np.asarray(correspondence["collapse_history"], dtype=np.int64)
        if collapse_history.ndim != 2 or collapse_history.shape[1] != 2:
            issues.append("collapse_history_shape_mismatch")
        elif len(collapse_history) and (
            np.any(collapse_history < 0) or np.any(collapse_history >= len(gt.vertices))
        ):
            issues.append("collapse_history_out_of_range")

        registration = np.load(model_dir / "coarse_registration_oracle.npz")
        if not np.allclose(
            registration["qem_vertices"], coarse_raw.vertices, atol=1e-7, rtol=0.0
        ):
            issues.append("saved_qem_vertices_mismatch")
        if not np.allclose(
            registration["registered_vertices"],
            coarse_registered_oracle.vertices,
            atol=1e-7,
            rtol=0.0,
        ):
            issues.append("saved_registered_vertices_mismatch")
        registration_faces = np.asarray(registration["coarse_gt_face_index"], dtype=np.int64)
        registration_weights = np.asarray(
            registration["coarse_gt_barycentric_weights"], dtype=np.float64
        )
        if registration_faces.shape != (len(coarse_raw.vertices),) or np.any(
            (registration_faces < 0) | (registration_faces >= len(gt.faces))
        ):
            issues.append("coarse_registration_face_indices_invalid")
        if registration_weights.shape != (len(coarse_raw.vertices), 3) or not np.allclose(
            registration_weights.sum(axis=1), 1.0, atol=2e-6, rtol=0.0
        ):
            issues.append("coarse_registration_weights_invalid")
        elif not np.isfinite(registration_weights).all() or np.any(
            registration_weights < -1e-10
        ):
            issues.append("coarse_registration_weights_outside_simplex")

        subdivision = np.load(model_dir / "subdivision_mapping_raw.npz")
        for key in (
            "parent_edges",
            "new_vertex_indices",
            "pre_compaction_to_final",
            "final_to_pre_compaction",
        ):
            if not np.array_equal(subdivision[key], reconstructed_mapping[key]):
                issues.append(f"raw_subdivision_mapping_mismatch:{key}")
        parent_edges = np.asarray(subdivision["parent_edges"], dtype=np.int64)
        children = np.asarray(subdivision["new_vertex_indices"], dtype=np.int64)
        if parent_edges.ndim != 2 or parent_edges.shape[1] != 2 or len(parent_edges) != len(children):
            issues.append("subdivision_mapping_shape_mismatch")
        elif (
            np.any(parent_edges < 0)
            or np.any(parent_edges >= len(expanded_initial_raw.vertices))
            or np.any(children < 0)
            or np.any(children >= len(expanded_initial_raw.vertices))
        ):
            issues.append("subdivision_mapping_out_of_range")
        metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("status") != "valid":
            issues.append("metadata_not_valid")
        if metadata.get("parameters", {}).get("global_nearest_surface_used_for_correspondence") is not False:
            issues.append("invalid_correspondence_contract")
        roles = metadata.get("file_roles", {})
        if "model_query" not in roles.get("expanded_initial_raw", ""):
            issues.append("expanded_initial_raw_role_missing")
        if "oracle" not in roles.get("coarse_registered_oracle", ""):
            issues.append("registered_coarse_oracle_role_missing")
        if "never training supervision" not in roles.get("P_target_oracle", ""):
            issues.append("P_target_oracle_training_exclusion_missing")
        if metadata.get("connectivity", {}).get("registered_coarse_used_for_query") is not False:
            issues.append("registered_coarse_query_contract_invalid")
        usage = metadata.get("usage_contract", {})
        if usage.get("may_connect_to_training_loader") is not False:
            issues.append("training_loader_exclusion_missing")
        if usage.get("training_target") != "delta_gt = L(gt_faces) @ gt_vertices":
            issues.append("GT_only_training_target_contract_missing")
        if usage.get("expanded_graph_oracle_laplacian_is_training_supervision") is not False:
            issues.append("expanded_oracle_laplacian_training_exclusion_missing")
    except Exception as error:  # noqa: BLE001
        issues.append(f"validation_exception:{error}")
    return sorted(set(issues))


def validate_refinement_dataset(output_root: str | Path) -> dict[str, Any]:
    output_root = Path(output_root).expanduser().resolve()
    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    results: dict[str, list[str]] = {}
    for sample in manifest["samples"]:
        if sample["status"] != "valid":
            continue
        issues = _validate_model(output_root / "models" / sample["model_id"])
        if issues:
            results[sample["model_id"]] = issues
    report = {
        "sample_count": len(manifest["samples"]),
        "valid_samples": sum(sample["status"] == "valid" for sample in manifest["samples"]),
        "failed_samples": sum(sample["status"] != "valid" for sample in manifest["samples"]),
        "validated_samples": sum(sample["status"] == "valid" for sample in manifest["samples"]),
        "invalid_after_validation": len(results),
        "issues": results,
    }
    _json(output_root / "validation.json", report)
    if results:
        raise RuntimeError(f"refinement validation failed for {len(results)} samples")
    return report


def _write_report(
    output_root: Path,
    source_root: Path,
    parameters: dict[str, Any],
    rows: list[dict[str, Any]],
    validation: dict[str, Any],
) -> None:
    valid = [row for row in rows if row["status"] == "valid"]
    failed = [row for row in rows if row["status"] != "valid"]
    split_counts = {
        split: sum(row["split"] == split and row["status"] == "valid" for row in rows)
        for split in ("train", "validation", "test")
    }
    diagnostics = {
        row["model_id"]: json.loads(Path(row["diagnostics_json"]).read_text(encoding="utf-8"))
        for row in valid
    }
    topology_blends = [
        (model_id, values["mapping"]["topology_safe_blend_alpha"])
        for model_id, values in diagnostics.items()
        if values["mapping"]["topology_safe_blend_alpha"] < 1.0
    ]
    large_displacement = [
        row["model_id"]
        for row in valid
        if "large_but_component_constrained_displacement" in row.get("warnings", "")
    ]
    maximum_displacement, maximum_displacement_model = max(
        (
            values["mapping"]["surface_displacement_max"],
            model_id,
        )
        for model_id, values in diagnostics.items()
    )
    total_component_fallbacks = sum(
        values["mapping"]["component_fallback_count"] for values in diagnostics.values()
    )
    fallback_samples = sum(
        values["mapping"]["component_fallback_count"] > 0
        for values in diagnostics.values()
    )
    count_range = lambda key: (  # noqa: E731
        min(row[key] for row in valid),
        max(row[key] for row in valid),
    )
    text = f"""# Sofa50 frozen-model inference and reconstruction evaluation data

## Result

- Source: `{source_root}` (Sofa50 only; Thingi10K is not read)
- Output: `{output_root}`
- Valid samples: {len(valid)}
- Failed samples: {len(failed)}
- Valid splits: {split_counts}
- Validation issues: {validation['invalid_after_validation']}
- Raw coarse vertex range: {count_range('coarse_raw_vertices')}
- Raw expanded query vertex range: {count_range('expanded_initial_raw_vertices')}
- Raw expanded query face range: {count_range('expanded_initial_raw_faces')}

## Generation contract

- Query coarse mesh: direct deterministic component-wise QEM output. Component-wise processing is a coarse-generation policy that prevents cross-component collapses.
- Requested coarse vertices: {parameters['coarse_target_vertices']} (minimum {parameters['coarse_min_vertices']}).
- Query expanded mesh: project `midpoint_subdivide(raw_coarse)`, {parameters['subdivision_steps']} step, before any GT-guided operation.
- Optional oracle diagnostic: `{TARGET_CONSTRUCTOR}`.
- Local provenance threshold: {parameters['max_provenance_local_distance']}; mappings beyond it fall back only within the same collapse-lineage GT component.
- Maximum allowed component-constrained displacement: {parameters['max_component_correspondence_displacement']}.
- `P_target_oracle[i]` matches `expanded_initial_raw[i]`, but is diagnostic/upper-bound only and is never training supervision.
- Registered coarse geometry is oracle/diagnostic-only and never generates query vertices.
- Global nearest-surface projection is not used to create correspondence.
- Nearest-surface queries are used only for diagnostics.
- This inference dataset does not generate or store a training Laplacian target.
- GT, mapping arrays and `P_target_oracle` are stored as float64; faces and indices are int64.

## Training boundary

- Do not connect this inference manifest to a training loader.
- Learned Laplacian training uses GT meshes only.
- The training target is `delta_gt = L(gt_faces) @ gt_vertices`, on the GT mesh's own connectivity.
- Training does not read `coarse_raw`, `expanded_initial_raw`, `P_target_oracle`, or an expanded-graph oracle Laplacian.

## Recorded limitations and warnings

- This is a deterministic GT-aligned proxy induced by real QEM collapse provenance, not a bijective dense registration.
- Same-component fallback was used for {total_component_fallbacks} vertices across {fallback_samples} samples; unconstrained/global fallback was never used.
- A topology-safe alpha below 1 was applied to {len(topology_blends)} samples to preserve every expanded face; the unblended oracle surface positions remain in `correspondence_oracle.npz` and `surface_target_oracle.npz`.
- {len(large_displacement)} sample(s) carry `large_but_component_constrained_displacement` warnings: {large_displacement}.
- Maximum correspondence displacement is {maximum_displacement:.9f} in `{maximum_displacement_model}` (hard limit {parameters['max_component_correspondence_displacement']}).

## Mapping

Each raw coarse vertex owns a cluster of original GT vertices obtained by replaying the
saved edge-collapse sequence. Its target is restricted to GT faces incident to that
cluster. Each new midpoint vertex is restricted to GT faces that span both parent
clusters. The selected GT face and barycentric weights are saved explicitly.

Collapse provenance is only an oracle target-search constraint; it is not a proven
semantic correspondence. `P_target_oracle` is a GT-guided proxy, not a bijective
dense registration. Query geometry is fixed before this oracle path begins.

## Files

- `manifest.json`, `manifest.csv`: unified sample inventory and paths.
- `train.txt`, `validation.txt`, `test.txt`: fixed Sofa50 split IDs copied into the output.
- `models/<model_id>/gt_mesh.*`
- `models/<model_id>/coarse_raw.*`
- `models/<model_id>/expanded_initial_raw.*`
- `models/<model_id>/P_target_oracle.*`
- `models/<model_id>/correspondence_oracle.npz`
- `models/<model_id>/subdivision_mapping_raw.npz`
- `models/<model_id>/coarse_registered_oracle.*`, `coarse_registration_oracle.npz`
- `models/<model_id>/surface_target_oracle.npz`
- `models/<model_id>/metadata.json`, `diagnostics.json`, `comparison_oracle.obj`
- `visualizations/`: selected PNG comparisons.
- `validation.json`: independent structural validation.

## Inference handoff

Load `coarse_raw.npz`, generate or load `expanded_initial_raw.npz`, apply the fully
trained frozen model to that raw expanded query, then run Laplacian integration or
reconstruction. `P_target_oracle.npz` may be loaded only for oracle diagnostics or
upper-bound evaluation; it must not define a training loss or target.
"""
    (output_root / "REPORT.md").write_text(text, encoding="utf-8")


def prepare_refinement_dataset(
    source_root: str | Path = "~/sofa_mesh/sofa50",
    output_root: str | Path = "~/sofa_mesh/sofa50_refinement",
    coarse_target_vertices: int = 3500,
    coarse_min_vertices: int = 1000,
    subdivision_steps: int = 1,
    seed: int = 20260806,
    force: bool = False,
    model_ids: list[str] | None = None,
) -> dict[str, Any]:
    source_root = Path(source_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if "thingi10k" in str(source_root).casefold():
        raise ValueError("Sofa refinement preparation refuses a Thingi10K source path")
    if subdivision_steps != 1:
        raise ValueError("provenance target construction currently supports exactly one subdivision step")
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "models").mkdir(exist_ok=True)
    (output_root / "visualizations").mkdir(exist_ok=True)
    source_splits, owner = _read_splits(source_root)
    if len(owner) != 50:
        raise RuntimeError(f"Sofa50 split files contain {len(owner)} unique models, expected 50")
    is_full_sofa50 = model_ids is None
    if model_ids is None:
        selected_ids = [
            model_id
            for split in ("train", "validation", "test")
            for model_id in source_splits[split]
        ]
    else:
        selected_ids = list(dict.fromkeys(model_ids))
        unknown = [model_id for model_id in selected_ids if model_id not in owner]
        if unknown:
            raise ValueError(f"requested model IDs are not in Sofa50 splits: {unknown}")
        if not selected_ids:
            raise ValueError("model_ids must contain at least one Sofa50 model")
    selected = set(selected_ids)
    splits = {
        split: [model_id for model_id in source_splits[split] if model_id in selected]
        for split in ("train", "validation", "test")
    }
    split_files: dict[str, str] = {}
    for split, split_model_ids in splits.items():
        split_path = output_root / f"{split}.txt"
        split_path.write_text("\n".join(split_model_ids) + "\n", encoding="utf-8")
        split_files[split] = str(split_path.resolve())
    parameters = _parameters(
        source_root,
        coarse_target_vertices,
        coarse_min_vertices,
        subdivision_steps,
        seed,
    )

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    preview_payload: list[tuple[str, MeshData, MeshData, MeshData]] = []
    ordered_ids = [
        model_id
        for split in ("train", "validation", "test")
        for model_id in splits[split]
    ]
    for model_id in tqdm(ordered_ids, desc="Preparing Sofa50 refinement meshes"):
        try:
            row, preview = _process_one(
                model_id, owner[model_id], source_root, output_root, parameters, force
            )
            rows.append(row)
            if preview is not None and len(preview_payload) < 6:
                preview_payload.append((model_id, *preview))
        except Exception as error:  # noqa: BLE001
            failure = {
                "model_id": model_id,
                "split": owner[model_id],
                "status": "invalid",
                "failure_reason": str(error),
            }
            failures.append(failure)
            rows.append(failure)
            failure_dir = output_root / "models" / model_id
            failure_dir.mkdir(parents=True, exist_ok=True)
            (failure_dir / "metadata.json").unlink(missing_ok=True)
            _json(
                failure_dir / "failure.json",
                {
                    **failure,
                    "source_path": str((source_root / model_id / "mesh.obj").resolve()),
                    "parameters": parameters,
                },
            )

    generated_previews = {model_id: (gt, expanded, target) for model_id, gt, expanded, target in preview_payload}
    preview_ids = [row["model_id"] for row in rows if row["status"] == "valid"][:6]
    for model_id in preview_ids:
        preview_path = output_root / "visualizations" / f"{model_id}.png"
        if preview_path.is_file() and not force:
            continue
        if model_id in generated_previews:
            gt, expanded, target = generated_previews[model_id]
        else:
            model_dir = output_root / "models" / model_id
            gt = _load_npz_mesh(model_dir / "gt_mesh.npz")
            expanded = _load_npz_mesh(model_dir / "expanded_initial_raw.npz")
            target = _load_npz_mesh(model_dir / "P_target_oracle.npz")
        _preview(preview_path, model_id, gt, expanded, target)
    if preview_ids:
        _visual_contact_sheet(output_root, preview_ids)
    rows.sort(key=lambda row: (("train", "validation", "test").index(row["split"]), row["model_id"]))
    pd.DataFrame(rows).to_csv(output_root / "manifest.csv", index=False)
    manifest = {
        "dataset": "Sofa50 frozen-model inference and reconstruction evaluation",
        "sample_scope": "full_sofa50" if is_full_sofa50 else "explicit_trial_subset",
        "source_root": str(source_root),
        "output_root": str(output_root),
        "parameters": parameters,
        "file_roles": {
            "coarse_raw": "model_input_direct_qem_output",
            "expanded_initial_raw": "model_query_midpoint_subdivision_of_coarse_raw",
            "P_target_oracle": "oracle_diagnostic_upper_bound_only_not_training_supervision",
            "correspondence_oracle": "oracle_diagnostic_construction_only",
            "coarse_registered_oracle": "oracle_diagnostic_only_never_query",
        },
        "usage_contract": {
            "primary_use": "frozen_model_inference_and_reconstruction_evaluation",
            "may_connect_to_training_loader": False,
            "training_data": "GT_mesh_only_from_a_separate_GT_loader_or_manifest",
            "training_target": "delta_gt = L(gt_faces) @ gt_vertices",
            "inference_path": "coarse_raw -> expanded_initial_raw -> frozen_model_prediction -> Laplacian_integration/reconstruction",
            "P_target_oracle_role": "diagnostic_upper_bound_only_not_training_supervision",
            "expanded_graph_oracle_laplacian_is_training_supervision": False,
        },
        "split_counts": {name: len(values) for name, values in splits.items()},
        "split_files": split_files,
        "success_count": sum(row["status"] == "valid" for row in rows),
        "failure_count": len(failures),
        "samples": rows,
        "failures": failures,
    }
    _json(output_root / "manifest.json", manifest)
    validation = validate_refinement_dataset(output_root)
    _write_report(output_root, source_root, parameters, rows, validation)
    return {
        "manifest": str((output_root / "manifest.json").resolve()),
        "report": str((output_root / "REPORT.md").resolve()),
        "output_root": str(output_root),
        "success_count": manifest["success_count"],
        "failure_count": manifest["failure_count"],
        "validation": validation,
    }
