from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp
import trimesh
from trimesh.triangles import points_to_barycentric


@dataclass(frozen=True)
class MeshArrays:
    vertices: np.ndarray
    faces: np.ndarray


def _as_mesh(vertices: np.ndarray, faces: np.ndarray) -> trimesh.Trimesh:
    return trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )


def extract_vertices_faces(entry: dict[str, Any]) -> MeshArrays:
    vertex_keys = ("vertices", "verts", "v")
    # The official ``npz`` variant calls triangle indices ``facets`` while
    # other variants and exported meshes commonly use one of the other names.
    face_keys = ("faces", "facets", "triangles", "f")
    vertices = next((entry[k] for k in vertex_keys if k in entry), None)
    faces = next((entry[k] for k in face_keys if k in entry), None)

    if vertices is not None and faces is not None:
        return MeshArrays(np.asarray(vertices), np.asarray(faces))

    path_keys = ("npz_path", "path", "file_path", "mesh_path")
    data_path = next((entry[k] for k in path_keys if k in entry), None)
    if data_path:
        data = np.load(data_path)
        file_vertices = next((data[k] for k in vertex_keys if k in data), None)
        file_faces = next((data[k] for k in face_keys if k in data), None)
        if file_vertices is not None and file_faces is not None:
            return MeshArrays(np.asarray(file_vertices), np.asarray(file_faces))

    raise ValueError(f"Unable to extract vertices/faces from entry keys={sorted(entry.keys())}")


def mesh_cleanup(vertices: np.ndarray, faces: np.ndarray) -> tuple[MeshArrays, list[str]]:
    operations: list[str] = []
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)

    mask_valid_face_shape = (f.ndim == 2) and (f.shape[1] == 3)
    if not mask_valid_face_shape:
        raise ValueError(f"Faces must be triangular [F,3], got shape {f.shape}")

    repeated_idx = (f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 0] == f[:, 2])
    if np.any(repeated_idx):
        f = f[~repeated_idx]
        operations.append("removed_repeated_index_faces")

    unique_faces, unique_idx = np.unique(f, axis=0, return_index=True)
    if unique_faces.shape[0] != f.shape[0]:
        f = unique_faces[np.argsort(unique_idx)]
        operations.append("removed_duplicate_faces")

    tri = v[f]
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) * 0.5
    tol = max(np.linalg.norm(v.max(axis=0) - v.min(axis=0)) * 1e-14, 1e-16)
    keep = area > tol
    if not np.all(keep):
        f = f[keep]
        operations.append("removed_zero_area_faces")

    mesh = _as_mesh(v, f)
    before = len(mesh.vertices)
    mesh.remove_unreferenced_vertices()
    if len(mesh.vertices) != before:
        operations.append("removed_unreferenced_vertices")

    diag = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
    eps = max(diag * 1e-10, 1e-14)
    before = len(mesh.vertices)
    # trimesh 5.x expresses merge tolerance as decimal precision rather than
    # accepting the older ``eps`` keyword.
    digits_vertex = max(0, int(np.ceil(-np.log10(eps))))
    mesh.merge_vertices(digits_vertex=digits_vertex)
    if len(mesh.vertices) != before:
        operations.append("merged_near_duplicate_vertices")

    # Merging can turn formerly valid triangles into repeated-index or
    # zero-area faces, so minimally clean faces once more afterwards.
    f = np.asarray(mesh.faces, dtype=np.int64)
    repeated_idx = (f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 0] == f[:, 2])
    if np.any(repeated_idx):
        mesh.update_faces(~repeated_idx)
        operations.append("removed_post_merge_repeated_index_faces")

    f = np.asarray(mesh.faces, dtype=np.int64)
    tri = np.asarray(mesh.vertices)[f]
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) * 0.5
    if np.any(area <= tol):
        mesh.update_faces(area > tol)
        operations.append("removed_post_merge_zero_area_faces")
    mesh.remove_unreferenced_vertices()

    if not mesh.is_winding_consistent:
        mesh.fix_normals()
        operations.append("fixed_face_orientation")

    return MeshArrays(np.asarray(mesh.vertices), np.asarray(mesh.faces)), operations


def normalize_vertices(vertices: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    bounds_min = vertices.min(axis=0)
    bounds_max = vertices.max(axis=0)
    center = (bounds_min + bounds_max) * 0.5
    extents = bounds_max - bounds_min
    longest = float(np.max(extents))
    scale = 2.0 / longest if longest > 0 else 1.0

    normalized = (vertices - center) * scale
    nmin = normalized.min(axis=0)
    nmax = normalized.max(axis=0)
    transform = {
        "original_bounds_min": bounds_min.tolist(),
        "original_bounds_max": bounds_max.tolist(),
        "translation_center": center.tolist(),
        "uniform_scale": scale,
        "normalized_bounds_min": nmin.tolist(),
        "normalized_bounds_max": nmax.tolist(),
        "inverse": {"scale": 1.0 / scale if scale != 0 else 1.0, "translation_center": center.tolist()},
    }
    return normalized.astype(np.float32), transform


def simplify_mesh(vertices: np.ndarray, faces: np.ndarray, target_vertices: int, min_vertices: int) -> MeshArrays:
    mesh = _as_mesh(vertices, faces)
    if len(mesh.vertices) <= target_vertices:
        return MeshArrays(np.asarray(mesh.vertices), np.asarray(mesh.faces))

    target_vertices = max(target_vertices, min_vertices)
    ratio = float(target_vertices) / float(len(mesh.vertices))
    target_faces = int(max(len(mesh.faces) * ratio, min_vertices * 2))

    if not hasattr(mesh, "simplify_quadric_decimation"):
        raise RuntimeError("trimesh.simplify_quadric_decimation is unavailable in this environment")

    # In trimesh 5.x the first positional argument is a fraction; pass the
    # requested absolute face count explicitly.
    simplified = mesh.simplify_quadric_decimation(face_count=target_faces)
    if simplified is None or len(simplified.vertices) < min_vertices:
        raise RuntimeError("Simplification failed or produced too few vertices")
    return MeshArrays(np.asarray(simplified.vertices), np.asarray(simplified.faces))


def midpoint_subdivide(vertices: np.ndarray, faces: np.ndarray, steps: int) -> tuple[MeshArrays, dict[str, np.ndarray]]:
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    all_parent_edges: list[np.ndarray] = []
    all_new_indices: list[np.ndarray] = []

    for _ in range(steps):
        edge_to_mid: dict[tuple[int, int], int] = {}
        parent_edges: list[tuple[int, int]] = []
        new_indices: list[int] = []
        new_vertices = v.tolist()
        new_faces: list[list[int]] = []

        def midpoint_index(a: int, b: int) -> int:
            key = (a, b) if a < b else (b, a)
            if key in edge_to_mid:
                return edge_to_mid[key]
            idx = len(new_vertices)
            new_vertices.append(((v[key[0]] + v[key[1]]) * 0.5).tolist())
            edge_to_mid[key] = idx
            parent_edges.append(key)
            new_indices.append(idx)
            return idx

        for a, b, c in f:
            ab = midpoint_index(int(a), int(b))
            bc = midpoint_index(int(b), int(c))
            ca = midpoint_index(int(c), int(a))
            new_faces.extend(
                [
                    [int(a), ab, ca],
                    [int(b), bc, ab],
                    [int(c), ca, bc],
                    [ab, bc, ca],
                ]
            )

        v = np.asarray(new_vertices, dtype=np.float64)
        f = np.asarray(new_faces, dtype=np.int64)
        all_parent_edges.append(np.asarray(parent_edges, dtype=np.int64))
        all_new_indices.append(np.asarray(new_indices, dtype=np.int64))

    mapping = {
        "parent_edges": np.concatenate(all_parent_edges, axis=0) if all_parent_edges else np.zeros((0, 2), dtype=np.int64),
        "new_vertex_indices": np.concatenate(all_new_indices, axis=0)
        if all_new_indices
        else np.zeros((0,), dtype=np.int64),
    }
    # At float32 precision a midpoint on a very short edge can coincide with
    # an endpoint. Remove only faces that collapse during that conversion.
    v_out = v.astype(np.float32)
    f_out = f.astype(np.int64)
    tri = v_out[f_out]
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) * 0.5
    repeated_idx = (
        (f_out[:, 0] == f_out[:, 1])
        | (f_out[:, 1] == f_out[:, 2])
        | (f_out[:, 0] == f_out[:, 2])
    )
    f_out = f_out[(area > 0) & ~repeated_idx]
    return MeshArrays(v_out, f_out), mapping


def compute_surface_targets(
    expanded_vertices: np.ndarray, gt_vertices: np.ndarray, gt_faces: np.ndarray
) -> dict[str, np.ndarray]:
    gt_mesh = _as_mesh(gt_vertices, gt_faces)
    closest_points, distances, face_ids = trimesh.proximity.closest_point(gt_mesh, expanded_vertices)

    valid_face = face_ids >= 0
    triangles = gt_mesh.triangles[np.clip(face_ids, 0, len(gt_mesh.faces) - 1)]
    bary = points_to_barycentric(triangles, closest_points)

    vnormals = gt_mesh.vertex_normals
    tri_vid = gt_mesh.faces[np.clip(face_ids, 0, len(gt_mesh.faces) - 1)]
    interpolated_normals = (
        vnormals[tri_vid[:, 0]] * bary[:, [0]]
        + vnormals[tri_vid[:, 1]] * bary[:, [1]]
        + vnormals[tri_vid[:, 2]] * bary[:, [2]]
    )
    normal_norm = np.linalg.norm(interpolated_normals, axis=1, keepdims=True)
    normal_norm[normal_norm == 0] = 1.0
    interpolated_normals = interpolated_normals / normal_norm

    valid = np.isfinite(closest_points).all(axis=1) & np.isfinite(distances) & valid_face
    return {
        "target_positions": closest_points.astype(np.float32),
        "target_displacements": (closest_points - expanded_vertices).astype(np.float32),
        "closest_face_indices": face_ids.astype(np.int64),
        "closest_barycentric_coordinates": bary.astype(np.float32),
        "surface_distance": distances.astype(np.float32),
        "target_normals": interpolated_normals.astype(np.float32),
        "valid_mask": valid.astype(np.uint8),
    }


def build_uniform_laplacian(num_vertices: int, faces: np.ndarray) -> sp.csr_matrix:
    f = np.asarray(faces, dtype=np.int64)
    edges = np.vstack(
        [
            f[:, [0, 1]],
            f[:, [1, 2]],
            f[:, [2, 0]],
        ]
    )
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)

    row = np.concatenate([edges[:, 0], edges[:, 1]])
    col = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.ones(len(row), dtype=np.float64)
    adjacency = sp.csr_matrix((data, (row, col)), shape=(num_vertices, num_vertices))
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    inv_degree = np.zeros_like(degree, dtype=np.float64)
    nonzero = degree > 0
    inv_degree[nonzero] = 1.0 / degree[nonzero]
    laplacian = sp.eye(num_vertices, format="csr", dtype=np.float64) - sp.diags(inv_degree) @ adjacency
    return laplacian.tocsr()
