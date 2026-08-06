from __future__ import annotations

import hashlib
import io
import json
import math
import random
import shutil
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image, ImageDraw, ImageFont, ImageOps
from tqdm import tqdm


SOFA_WORDS = ("sofa", "couch", "settee")
PREFERRED_WORDS = (
    "two seat",
    "two-seat",
    "2 seat",
    "2-seat",
    "three seat",
    "three-seat",
    "3 seat",
    "3-seat",
    "loveseat",
    "multi-seat",
    "multiple seat",
)
EXCLUDED_WORDS = (
    "l-shaped",
    "l shaped",
    "corner sofa",
    "sectional",
    "u-shaped",
    "u shaped",
    "chaise",
    "recliner",
    "reclining",
    "lazy sofa",
    "single seat",
    "single-seat",
    "one seat",
    "one-seat",
    "armchair",
    "footstool",
    "sofastool",
    "bed end stool",
    "sofa bed",
    "sofabed",
    "couch bed",
)
MESH_NAMES = ("raw_model.obj", "normalized_model.obj")
VISUAL_EXCLUSIONS = {
    # Official previews reveal geometry/subtypes which are not reliable in metadata.
    "06cf411f-a021-4bf1-b417-26ca819591f7": "heavy_ornamental_design",
    "3db020fe-44a8-427d-840e-8ab9cb1ca01e": "near_duplicate_ornamental_sofa",
    "5f41adee-8f09-4431-b246-6cf93d78f4d8": "curved_sofa",
    "0893af5b-1c13-4996-b1ad-78329f5c2fd2": "chaise_like_asymmetry",
    "6ee12c70-d29e-408b-999e-19c015833ed6": "heavy_ornamental_design",
    "07be7bf4-c318-4055-9272-c47e5ab532bf": "near_duplicate_sofa",
    "2403c725-cec6-46ba-8d43-b7b87834a71a": "blanket_obscures_seat",
    # High-ranked reserve candidates rejected during the same visual review.
    "14132991-8e49-45ed-82d0-f48ba5903b8f": "curved_sofa",
    "40d2c9d3-d44e-46c9-bfcc-2a4555acf11d": "near_duplicate_ornamental_sofa",
    "431e3530-40b1-404a-bf3b-d6a2fee00a65": "curved_sofa",
    "3e6c45ba-9989-4e6f-bc3a-63285ae10d6d": "heavy_ornamental_design",
    "4c9ac6a1-2643-4484-952c-395892f5bee8": "heavy_ornamental_design",
    "cbfb77b6-92f7-4bb6-9160-eaa820606773": "curved_sofa",
    "0dbd0e22-0e25-439a-8472-fa3cd394c9ad": "missing_distinct_armrests",
    "251be3a1-e684-4e30-a6fe-2ec57cc75cbc": "corner_like_geometry",
    "76ed78f5-36a5-4bfa-b6d9-797182d70be7": "heavy_ornamental_design",
    "dd49b740-7632-4928-adf1-3fd853decb30": "daybed_like_wood_frame",
    "e53f180e-c706-4fba-a222-b2845c1ee8e5": "daybed_like_wood_frame",
    "7df23cf5-1e4d-4c6c-b680-84c80e92830d": "near_duplicate_sofa",
}


@dataclass(frozen=True)
class Candidate:
    model_id: str
    category: str
    source_path: str
    vertex_count: int
    face_count: int
    connected_components: int
    width_depth_ratio: float
    width_height_ratio: float
    quality_score: float
    geometry_hash: str
    shape_descriptor: tuple[float, ...]
    known_issues: tuple[str, ...]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _find_archive(downloads: Path) -> Path | None:
    archives = sorted(downloads.glob("*3D-FUTURE*model*.zip"))
    if not archives:
        archives = sorted(downloads.glob("*.zip"))
    return archives[0] if archives else None


def _extract_model_info(archive: Path, downloads: Path) -> Path:
    with zipfile.ZipFile(archive) as bundle:
        members = [
            name for name in bundle.namelist() if name.rstrip("/").endswith("/model_info.json")
        ]
        if not members:
            raise FileNotFoundError(f"Archive has no model_info.json: {archive}")
        member = min(members, key=lambda value: (value.count("/"), len(value)))
        parts = Path(member).parts
        if any(part in {"", ".", ".."} for part in parts):
            raise ValueError(f"Unsafe model_info archive path: {member}")
        target = downloads.joinpath(*parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file():
            with bundle.open(member) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
        return target.parent


def _find_future_root(downloads: Path) -> Path:
    roots = sorted({path.parent for path in downloads.rglob("model_info.json")})
    roots = [root for root in roots if any((root / child).is_dir() for child in root.iterdir())]
    if roots:
        return roots[0]

    archive = _find_archive(downloads)
    if archive is not None:
        return _extract_model_info(archive, downloads)

    raise FileNotFoundError(
        "Official 3D-FUTURE data is missing. Place 3D-FUTURE-model.zip under "
        f"{downloads}, or extract it there so that 3D-FUTURE-model/model_info.json "
        "and <model_id>/raw_model.obj are present."
    )


def _metadata_entries(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        values = payload
    elif isinstance(payload, dict):
        values = next(
            (
                payload[key]
                for key in ("models", "model_info", "furniture", "data")
                if isinstance(payload.get(key), list)
            ),
            [],
        )
    else:
        values = []
    entries = [dict(value) for value in values if isinstance(value, dict)]
    if not entries:
        raise ValueError(f"No model records found in {path}")
    return entries


def _model_id(entry: dict[str, Any]) -> str:
    for key in ("model_id", "modelId", "jid", "id", "uid"):
        value = entry.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _category_text(entry: dict[str, Any]) -> str:
    values: list[str] = []
    for key, value in entry.items():
        normalized = str(key).lower().replace("_", "-")
        if any(word in normalized for word in ("category", "title", "type", "name")):
            if isinstance(value, (str, int, float)):
                values.append(str(value))
    return " | ".join(values)


def _is_sofa(entry: dict[str, Any]) -> bool:
    super_category = str(
        entry.get("super-category", entry.get("super_category", ""))
    ).strip()
    if super_category:
        return super_category.casefold() == "sofa"
    text = _category_text(entry).lower()
    return any(word in text for word in SOFA_WORDS) and "sofastool" not in text


def _mesh_for_model(root: Path, model_id: str) -> Path | None:
    folder = root / model_id
    for name in MESH_NAMES:
        path = folder / name
        if path.is_file():
            return path
    return None


def _copy_raw_mesh_from_archive(
    bundle: zipfile.ZipFile,
    archive_names: set[str],
    model_root_name: str,
    model_id: str,
    destination: Path,
) -> bool:
    candidates = (
        f"{model_root_name}/{model_id}/raw_model.obj",
        f"{model_root_name}/{model_id}/normalized_model.obj",
    )
    member = next((name for name in candidates if name in archive_names), None)
    if member is None:
        suffixes = tuple(f"/{model_id}/{Path(name).name}" for name in candidates)
        member = next(
            (name for name in archive_names if name.endswith(suffixes)),
            None,
        )
    if member is None:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file():
        with bundle.open(member) as source, destination.open("wb") as output:
            shutil.copyfileobj(source, output)
    return True


def _load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError("scene contains no mesh geometry")
        mesh = loaded.to_mesh()
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise TypeError(f"Unsupported mesh type: {type(loaded).__name__}")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError("mesh is empty")
    return trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices, dtype=np.float64),
        faces=np.asarray(mesh.faces, dtype=np.int64),
        process=False,
    )


def _clean_mesh(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, list[str]]:
    operations: list[str] = []
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces are not triangular: {faces.shape}")

    valid_index = np.all((faces >= 0) & (faces < len(vertices)), axis=1)
    if not np.all(valid_index):
        faces = faces[valid_index]
        operations.append("removed_invalid_face_indices")
    finite_vertices = np.isfinite(vertices).all(axis=1)
    valid_finite = finite_vertices[faces].all(axis=1)
    if not np.all(valid_finite):
        faces = faces[valid_finite]
        operations.append("removed_faces_using_non_finite_vertices")

    repeated = (
        (faces[:, 0] == faces[:, 1])
        | (faces[:, 1] == faces[:, 2])
        | (faces[:, 0] == faces[:, 2])
    )
    if np.any(repeated):
        faces = faces[~repeated]
        operations.append("removed_repeated_index_faces")

    canonical = np.sort(faces, axis=1)
    _, first = np.unique(canonical, axis=0, return_index=True)
    keep_unique = np.zeros(len(faces), dtype=bool)
    keep_unique[np.sort(first)] = True
    if not np.all(keep_unique):
        faces = faces[keep_unique]
        operations.append("removed_duplicate_faces")

    if len(faces) == 0:
        raise ValueError("no faces remain after index cleanup")
    tri = vertices[faces]
    diagonal = float(np.linalg.norm(np.ptp(vertices[finite_vertices], axis=0)))
    area_tolerance = max(diagonal * diagonal * 1e-14, 1e-18)
    areas = np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
    ) * 0.5
    keep_area = np.isfinite(areas) & (areas > area_tolerance)
    if not np.all(keep_area):
        faces = faces[keep_area]
        operations.append("removed_degenerate_faces")

    result = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    before = len(result.vertices)
    result.remove_unreferenced_vertices()
    if len(result.vertices) != before:
        operations.append("removed_isolated_vertices")
    before = len(result.vertices)
    result.merge_vertices(merge_tex=True, merge_norm=True, digits_vertex=10)
    if len(result.vertices) != before:
        operations.append("merged_duplicate_vertices")

    faces = np.asarray(result.faces, dtype=np.int64)
    repeated = (
        (faces[:, 0] == faces[:, 1])
        | (faces[:, 1] == faces[:, 2])
        | (faces[:, 0] == faces[:, 2])
    )
    if np.any(repeated):
        result.update_faces(~repeated)
        operations.append("removed_post_merge_degenerate_faces")
    result.remove_unreferenced_vertices()
    if not result.is_winding_consistent:
        result.fix_normals(multibody=True)
        operations.append("fixed_face_winding_and_normals")
    return result, operations


def _remove_remote_fragments(
    mesh: trimesh.Trimesh,
) -> tuple[trimesh.Trimesh, int, list[str]]:
    components = list(mesh.split(only_watertight=False))
    if len(components) <= 1:
        return mesh, 0, []
    largest = max(components, key=lambda item: len(item.faces))
    diagonal = max(float(np.linalg.norm(largest.extents)), 1e-12)
    lower, upper = largest.bounds
    kept: list[trimesh.Trimesh] = []
    removed = 0
    total_faces = max(len(mesh.faces), 1)
    for component in components:
        centroid = np.asarray(component.centroid)
        outside = np.maximum(np.maximum(lower - centroid, centroid - upper), 0.0)
        distance = float(np.linalg.norm(outside))
        face_fraction = len(component.faces) / total_faces
        tiny = face_fraction < 0.002 and len(component.faces) < 128
        if component is not largest and tiny and distance > 0.35 * diagonal:
            removed += 1
        else:
            kept.append(component)
    if not kept:
        kept = [largest]
    result = trimesh.util.concatenate(kept)
    operations = ["removed_remote_tiny_fragments"] if removed else []
    return result, removed, operations


def _to_z_up(vertices: np.ndarray, source_up_axis: str) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    if source_up_axis == "z":
        return vertices.copy()
    if source_up_axis == "y":
        return np.column_stack((vertices[:, 0], -vertices[:, 2], vertices[:, 1]))
    if source_up_axis == "x":
        return np.column_stack((-vertices[:, 1], -vertices[:, 2], vertices[:, 0]))
    raise ValueError("source_up_axis must be x, y, or z")


def _orient_and_normalize(
    mesh: trimesh.Trimesh,
    source_up_axis: str,
    target_max_extent: float,
) -> tuple[trimesh.Trimesh, list[float], float, list[str]]:
    vertices = _to_z_up(np.asarray(mesh.vertices), source_up_axis)
    operations = [f"converted_{source_up_axis}_up_to_z_up"] if source_up_axis != "z" else []
    extents = np.ptp(vertices, axis=0)
    if extents[1] > extents[0]:
        vertices = np.column_stack((vertices[:, 1], -vertices[:, 0], vertices[:, 2]))
        operations.append("rotated_horizontal_axes_to_make_width_x")

    z = vertices[:, 2]
    high = z >= np.quantile(z, 0.65)
    if np.any(high) and float(np.mean(vertices[high, 1])) < float(np.mean(vertices[:, 1])):
        vertices[:, 1] *= -1.0
        operations.append("flipped_front_to_negative_y")

    lower = vertices.min(axis=0)
    upper = vertices.max(axis=0)
    center = (lower + upper) * 0.5
    maximum = float(np.max(upper - lower))
    if not np.isfinite(maximum) or maximum <= 1e-12:
        raise ValueError("mesh has an invalid bounding box")
    multiplier = target_max_extent / maximum
    vertices = (vertices - center) * multiplier
    normalized = trimesh.Trimesh(vertices=vertices, faces=mesh.faces, process=False)
    return normalized, center.tolist(), multiplier, operations


def _geometry_hash(mesh: trimesh.Trimesh) -> str:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    centered = vertices - (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
    scale = max(float(np.max(np.ptp(centered, axis=0))), 1e-12)
    quantized = np.round(centered / scale, decimals=5)
    order = np.lexsort((quantized[:, 2], quantized[:, 1], quantized[:, 0]))
    digest = hashlib.sha256()
    digest.update(np.asarray(quantized[order], dtype=np.float32).tobytes())
    digest.update(np.asarray([len(mesh.faces)], dtype=np.int64).tobytes())
    return digest.hexdigest()


def _shape_descriptor(mesh: trimesh.Trimesh) -> tuple[float, ...]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
    centered = vertices - (lower + upper) * 0.5
    maximum = max(float(np.max(upper - lower)), 1e-12)
    points = centered / maximum
    descriptor: list[float] = []
    for axis in range(3):
        histogram, _ = np.histogram(points[:, axis], bins=12, range=(-0.5, 0.5))
        descriptor.extend((histogram / max(histogram.sum(), 1)).tolist())
    radii = np.linalg.norm(points, axis=1)
    histogram, _ = np.histogram(radii, bins=12, range=(0.0, math.sqrt(3.0) * 0.5))
    descriptor.extend((histogram / max(histogram.sum(), 1)).tolist())
    descriptor.extend((np.ptp(points, axis=0)).tolist())
    return tuple(float(value) for value in descriptor)


def _candidate_score(
    category: str,
    mesh: trimesh.Trimesh,
    width_depth_ratio: float,
    width_height_ratio: float,
) -> tuple[float, list[str]]:
    text = category.lower()
    issues: list[str] = []
    score = 0.0
    if any(word in text for word in PREFERRED_WORDS):
        score += 30.0
    if any(word in text for word in EXCLUDED_WORDS):
        score -= 100.0
        issues.append("excluded_sofa_subtype_keyword")
    if 1.45 <= width_depth_ratio <= 4.5:
        score += 24.0 - abs(width_depth_ratio - 2.4) * 3.0
    else:
        score -= 35.0
        issues.append("unusual_width_depth_ratio")
    if 1.3 <= width_height_ratio <= 5.5:
        score += 14.0
    else:
        score -= 20.0
        issues.append("unusual_width_height_ratio")
    faces = len(mesh.faces)
    if 10_000 <= faces <= 50_000:
        score += 20.0
    elif 2_000 <= faces <= 200_000:
        score += 8.0
    else:
        score -= 12.0
        issues.append("source_face_count_outside_preferred_range")
    components = len(mesh.split(only_watertight=False))
    if components <= 64:
        score += 8.0
    else:
        score -= min(20.0, (components - 64) * 0.1)
        issues.append("many_connected_components")
    return score, issues


def _evaluate_candidate(
    model_id: str,
    category: str,
    path: Path,
    source_up_axis: str,
) -> Candidate:
    mesh, _ = _clean_mesh(_load_mesh(path))
    mesh, _, _ = _remove_remote_fragments(mesh)
    oriented, _, _, _ = _orient_and_normalize(mesh, source_up_axis, 2.0)
    extents = np.asarray(oriented.extents)
    width_depth = float(extents[0] / max(extents[1], 1e-12))
    width_height = float(extents[0] / max(extents[2], 1e-12))
    score, issues = _candidate_score(category, oriented, width_depth, width_height)
    return Candidate(
        model_id=model_id,
        category=category,
        source_path=str(path.resolve()),
        vertex_count=int(len(oriented.vertices)),
        face_count=int(len(oriented.faces)),
        connected_components=int(len(oriented.split(only_watertight=False))),
        width_depth_ratio=width_depth,
        width_height_ratio=width_height,
        quality_score=float(score),
        geometry_hash=_geometry_hash(oriented),
        shape_descriptor=_shape_descriptor(oriented),
        known_issues=tuple(issues),
    )


def _near_duplicate(a: Candidate, b: Candidate) -> bool:
    if a.geometry_hash == b.geometry_hash:
        return True
    if abs(a.width_depth_ratio - b.width_depth_ratio) > 0.06:
        return False
    if abs(a.width_height_ratio - b.width_height_ratio) > 0.08:
        return False
    descriptor_distance = float(
        np.linalg.norm(np.asarray(a.shape_descriptor) - np.asarray(b.shape_descriptor))
    )
    face_ratio = max(a.face_count, b.face_count) / max(min(a.face_count, b.face_count), 1)
    return descriptor_distance < 0.035 and face_ratio < 1.15


def _select_candidates(candidates: Iterable[Candidate], count: int) -> list[Candidate]:
    ranked = sorted(candidates, key=lambda item: (-item.quality_score, item.model_id))
    selected: list[Candidate] = []
    for candidate in ranked:
        if candidate.model_id in VISUAL_EXCLUSIONS:
            continue
        if "excluded_sofa_subtype_keyword" in candidate.known_issues:
            continue
        if any(_near_duplicate(candidate, previous) for previous in selected):
            continue
        selected.append(candidate)
        if len(selected) == count:
            return selected
    raise RuntimeError(
        f"Only {len(selected)} distinct usable Sofa candidates remain; requested {count}. "
        "Review all_sofas/candidates.json or provide more official models."
    )


def _load_cached_candidates(
    candidates_path: Path,
    failures_path: Path,
    eligible_ids: set[str],
) -> tuple[list[Candidate], list[dict[str, str]]] | None:
    if not candidates_path.is_file() or not failures_path.is_file():
        return None
    try:
        candidate_payload = json.loads(candidates_path.read_text(encoding="utf-8"))
        failures = json.loads(failures_path.read_text(encoding="utf-8"))
        candidates = [Candidate(**item) for item in candidate_payload]
        cached_ids = {candidate.model_id for candidate in candidates}
        cached_ids.update(str(item["model_id"]) for item in failures)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if cached_ids != eligible_ids:
        return None
    return candidates, failures


def _prune_generated_model_dirs(root: Path, selected_ids: set[str]) -> None:
    for path in root.iterdir():
        if not path.is_dir() or path.name in selected_ids:
            continue
        info_path = path / "info.json"
        if not info_path.is_file():
            continue
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if info.get("processing_success") is True and info.get("model_id") == path.name:
            shutil.rmtree(path)


def _prune_non_sofa_raw_dirs(
    all_sofas: Path,
    all_metadata_ids: set[str],
    sofa_ids: set[str],
) -> None:
    for path in all_sofas.iterdir():
        if (
            path.is_dir()
            and path.name in all_metadata_ids
            and path.name not in sofa_ids
            and (path / "raw_model.obj").is_file()
        ):
            shutil.rmtree(path)


def _process_selected(
    candidate: Candidate,
    output_root: Path,
    source_up_axis: str,
    target_max_extent: float,
    target_faces: int,
    max_faces: int,
) -> dict[str, Any]:
    model_dir = output_root / candidate.model_id
    info_path = model_dir / "info.json"
    if info_path.is_file() and (model_dir / "mesh.obj").is_file() and (model_dir / "mesh.ply").is_file():
        existing = json.loads(info_path.read_text(encoding="utf-8"))
        if (
            existing.get("processing_success") is True
            and existing.get("pipeline_version") == 2
        ):
            return existing

    model_dir.mkdir(parents=True, exist_ok=True)
    mesh, operations = _clean_mesh(_load_mesh(Path(candidate.source_path)))
    mesh, removed_fragments, fragment_operations = _remove_remote_fragments(mesh)
    operations.extend(fragment_operations)
    if len(mesh.faces) > max_faces:
        simplified = mesh.simplify_quadric_decimation(face_count=target_faces)
        if simplified is None or len(simplified.faces) == 0:
            raise RuntimeError("quadric simplification failed")
        mesh, simplify_cleanup = _clean_mesh(simplified)
        operations.append(f"simplified_to_approximately_{target_faces}_faces")
        operations.extend(simplify_cleanup)

    mesh, center, scale, orientation_operations = _orient_and_normalize(
        mesh, source_up_axis, target_max_extent
    )
    operations.extend(orientation_operations)
    mesh, final_cleanup = _clean_mesh(mesh)
    operations.extend(final_cleanup)
    components = len(mesh.split(only_watertight=False))
    known_issues = [
        issue
        for issue in candidate.known_issues
        if issue != "source_face_count_outside_preferred_range"
    ]
    if len(mesh.faces) < 10_000:
        known_issues.append("below_recommended_10000_faces")
    if len(mesh.faces) > 50_000:
        known_issues.append("above_recommended_50000_faces")
    mesh.export(model_dir / "mesh.obj", file_type="obj")
    mesh.export(model_dir / "mesh.ply", file_type="ply", encoding="binary")
    info = {
        "model_id": candidate.model_id,
        "pipeline_version": 2,
        "source_path": candidate.source_path,
        "vertex_count": int(len(mesh.vertices)),
        "face_count": int(len(mesh.faces)),
        "connected_components": int(components),
        "normalization_center": [float(value) for value in center],
        "normalization_scale": float(scale),
        "processing_success": True,
        "known_issues": sorted(set(known_issues)),
        "category": candidate.category,
        "orientation": "z_up_front_negative_y_width_x",
        "target_max_extent": float(target_max_extent),
        "processing_operations": list(dict.fromkeys(operations)),
    }
    _write_json(info_path, info)
    return info


def _split_ids(ids: list[str], seed: int) -> dict[str, list[str]]:
    shuffled = sorted(ids)
    random.Random(seed).shuffle(shuffled)
    if len(ids) == 50:
        train_count, validation_count = 40, 5
    else:
        validation_count = max(1, round(len(ids) * 0.1))
        test_count = max(1, round(len(ids) * 0.1))
        train_count = len(ids) - validation_count - test_count
    validation_end = train_count + validation_count
    return {
        "train": shuffled[:train_count],
        "validation": shuffled[train_count:validation_end],
        "test": shuffled[validation_end:],
    }


def _write_splits(root: Path, splits: dict[str, list[str]]) -> None:
    for name in ("train", "validation", "test"):
        (root / f"{name}.txt").write_text(
            "".join(f"{model_id}\n" for model_id in splits[name]), encoding="utf-8"
        )


def _preview_contact_sheet(
    root: Path,
    model_ids: list[str],
    archive_path: Path,
) -> bool:
    cell_width, image_height, label_height = 400, 360, 44
    columns = 5
    rows = math.ceil(len(model_ids) / columns)
    sheet = Image.new(
        "RGB",
        (cell_width * columns, (image_height + label_height) * rows),
        "white",
    )
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    with zipfile.ZipFile(archive_path) as bundle:
        names = bundle.namelist()
        preview_names = {
            Path(name).parent.name: name
            for name in names
            if name.endswith("/image.jpg")
        }
        if not all(model_id in preview_names for model_id in model_ids):
            return False
        for index, model_id in enumerate(model_ids):
            with bundle.open(preview_names[model_id]) as source:
                preview = Image.open(io.BytesIO(source.read())).convert("RGB")
            preview = ImageOps.contain(preview, (image_height, image_height))
            column, row = index % columns, index // columns
            x = column * cell_width + (cell_width - preview.width) // 2
            y = row * (image_height + label_height)
            sheet.paste(preview, (x, y))
            draw = ImageDraw.Draw(sheet)
            label = f"{model_id[:18]}\n{model_id[18:]}"
            draw.multiline_text(
                (column * cell_width + 8, y + image_height + 3),
                label,
                fill="black",
                font=font,
                spacing=0,
            )
    sheet.save(root / "contact_sheet.png", optimize=True)
    return True


def _contact_sheet(
    root: Path,
    model_ids: list[str],
    archive_path: Path | None = None,
) -> None:
    if archive_path is not None and _preview_contact_sheet(root, model_ids, archive_path):
        return
    columns = 5
    rows = math.ceil(len(model_ids) / columns)
    figure = plt.figure(figsize=(15, rows * 3), facecolor="white")
    for index, model_id in enumerate(model_ids):
        axis = figure.add_subplot(rows, columns, index + 1, projection="3d")
        mesh = _load_mesh(root / model_id / "mesh.obj")
        faces = np.asarray(mesh.faces)
        if len(faces) > 1800:
            sample = np.linspace(0, len(faces) - 1, 1800, dtype=np.int64)
            faces = faces[sample]
        triangles = np.asarray(mesh.vertices)[faces]
        collection = Poly3DCollection(
            triangles,
            linewidths=0.0,
            facecolors="#86a9c4",
            edgecolors="none",
            alpha=1.0,
        )
        axis.add_collection3d(collection)
        axis.set_xlim(-1.05, 1.05)
        axis.set_ylim(-1.05, 1.05)
        axis.set_zlim(-1.05, 1.05)
        axis.set_box_aspect((1.0, 1.0, 1.0))
        axis.view_init(elev=18, azim=-62)
        axis.set_axis_off()
        axis.set_title(model_id, fontsize=6, pad=0)
    figure.tight_layout(pad=0.3)
    figure.savefig(root / "contact_sheet.png", dpi=140, bbox_inches="tight")
    plt.close(figure)


def validate_sofa50(root: str | Path, expected_count: int = 50) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    split_ids = {
        name: [line.strip() for line in (root / f"{name}.txt").read_text().splitlines() if line.strip()]
        for name in ("train", "validation", "test")
    }
    all_ids = [model_id for values in split_ids.values() for model_id in values]
    issues: dict[str, list[str]] = {}
    if len(all_ids) != expected_count or len(set(all_ids)) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} unique split IDs, found {len(all_ids)} entries and "
            f"{len(set(all_ids))} unique IDs"
        )
    if expected_count == 50 and {name: len(values) for name, values in split_ids.items()} != {
        "train": 40,
        "validation": 5,
        "test": 5,
    }:
        raise RuntimeError("Expected split sizes train=40, validation=5, test=5")

    for model_id in all_ids:
        model_issues: list[str] = []
        model_dir = root / model_id
        for filename in ("mesh.obj", "mesh.ply", "info.json"):
            if not (model_dir / filename).is_file():
                model_issues.append(f"missing_{filename}")
        try:
            mesh = _load_mesh(model_dir / "mesh.obj")
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.faces)
            if faces.ndim != 2 or faces.shape[1] != 3:
                model_issues.append("non_triangular_faces")
            if not np.isfinite(vertices).all():
                model_issues.append("non_finite_vertices")
            if not np.all((faces >= 0) & (faces < len(vertices))):
                model_issues.append("invalid_face_indices")
            if not np.isfinite(mesh.extents).all() or np.any(mesh.extents <= 0):
                model_issues.append("invalid_bounding_box")
            if float(np.max(mesh.extents)) > 2.0001:
                model_issues.append("normalization_extent_exceeded")
            info = json.loads((model_dir / "info.json").read_text(encoding="utf-8"))
            if info.get("model_id") != model_id or info.get("processing_success") is not True:
                model_issues.append("invalid_info_json")
        except Exception as error:  # noqa: BLE001
            model_issues.append(f"load_failure:{error}")
        if model_issues:
            issues[model_id] = model_issues
    report = {
        "expected_count": expected_count,
        "actual_count": len(all_ids),
        "unique_count": len(set(all_ids)),
        "split_counts": {name: len(values) for name, values in split_ids.items()},
        "valid_meshes": expected_count - len(issues),
        "invalid_meshes": len(issues),
        "issues": issues,
    }
    _write_json(root / "validation.json", report)
    if issues:
        raise RuntimeError(f"Sofa50 validation failed for {len(issues)} models")
    return report


def prepare_sofa50(
    data_root: str | Path,
    count: int = 50,
    seed: int = 20260806,
    source_up_axis: str = "y",
    target_max_extent: float = 2.0,
    target_faces: int = 40_000,
    max_faces: int = 50_000,
) -> dict[str, Any]:
    data_root = Path(data_root).expanduser().resolve()
    downloads = data_root / "downloads"
    all_sofas = data_root / "all_sofas"
    output_root = data_root / "sofa50"
    downloads.mkdir(parents=True, exist_ok=True)
    all_sofas.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    future_root = _find_future_root(downloads)
    entries = _metadata_entries(future_root / "model_info.json")
    sofa_entries = [entry for entry in entries if _is_sofa(entry)]
    sofa_ids = {_model_id(entry) for entry in sofa_entries if _model_id(entry)}
    _prune_non_sofa_raw_dirs(
        all_sofas,
        {_model_id(entry) for entry in entries if _model_id(entry)},
        sofa_ids,
    )
    copied: list[tuple[str, str, Path]] = []
    seen_ids: set[str] = set()
    archive_path = _find_archive(downloads)
    bundle = zipfile.ZipFile(archive_path) if archive_path is not None else None
    archive_names = set(bundle.namelist()) if bundle is not None else set()
    try:
        for entry in tqdm(sofa_entries, desc="Collecting raw Sofa meshes"):
            model_id = _model_id(entry)
            if not model_id or model_id in seen_ids:
                continue
            seen_ids.add(model_id)
            category = _category_text(entry)
            destination = all_sofas / model_id / "raw_model.obj"
            source = _mesh_for_model(future_root, model_id)
            if source is not None:
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    shutil.copy2(source, destination)
            elif bundle is None or not _copy_raw_mesh_from_archive(
                bundle,
                archive_names,
                future_root.name,
                model_id,
                destination,
            ):
                continue
            copied.append((model_id, category, destination))
    finally:
        if bundle is not None:
            bundle.close()
    if len(copied) < count:
        raise RuntimeError(
            f"Found only {len(copied)} readable raw Sofa paths in official 3D-FUTURE data; "
            f"need at least {count}."
        )

    eligible = [
        item
        for item in copied
        if not any(word in item[1].lower() for word in EXCLUDED_WORDS)
    ]
    excluded = [
        {"model_id": model_id, "category": category}
        for model_id, category, _ in copied
        if any(word in category.lower() for word in EXCLUDED_WORDS)
    ]
    _write_json(all_sofas / "excluded_subtypes.json", excluded)
    candidates_path = all_sofas / "candidates.json"
    failures_path = all_sofas / "load_failures.json"
    cached = _load_cached_candidates(
        candidates_path,
        failures_path,
        {model_id for model_id, _, _ in eligible},
    )
    if cached is None:
        candidates = []
        failures: list[dict[str, str]] = []
        for model_id, category, path in tqdm(eligible, desc="Evaluating Sofa candidates"):
            try:
                candidates.append(_evaluate_candidate(model_id, category, path, source_up_axis))
            except Exception as error:  # noqa: BLE001
                failures.append({"model_id": model_id, "reason": str(error)})
        _write_json(candidates_path, [asdict(candidate) for candidate in candidates])
        _write_json(failures_path, failures)
    else:
        candidates, failures = cached
    selected = _select_candidates(candidates, count)
    selected_ids = [candidate.model_id for candidate in selected]
    _prune_generated_model_dirs(output_root, set(selected_ids))

    infos: list[dict[str, Any]] = []
    for candidate in tqdm(selected, desc="Processing selected Sofa meshes"):
        infos.append(
            _process_selected(
                candidate,
                output_root,
                source_up_axis,
                target_max_extent,
                target_faces,
                max_faces,
            )
        )
    splits = _split_ids(selected_ids, seed)
    _write_splits(output_root, splits)
    _contact_sheet(output_root, selected_ids, archive_path)
    report = validate_sofa50(output_root, expected_count=count)
    _write_json(
        output_root / "selection.json",
        {
            "seed": seed,
            "source": "3D-FUTURE-model",
            "source_root": str(future_root),
            "raw_sofas_found": len(copied),
            "candidate_meshes_loaded": len(candidates),
            "selected_ids": selected_ids,
            "visual_exclusions": VISUAL_EXCLUSIONS,
            "splits": splits,
            "processing": {
                "source_up_axis": source_up_axis,
                "output_orientation": "z_up_front_negative_y_width_x",
                "target_max_extent": target_max_extent,
                "target_faces": target_faces,
                "max_faces": max_faces,
                "smoothing": False,
                "watertight_conversion": False,
                "voxelization": False,
            },
            "known_issues": {
                info["model_id"]: info["known_issues"]
                for info in infos
                if info["known_issues"]
            },
            "validation": report,
        },
    )
    return {
        "all_sofas": str(all_sofas),
        "raw_sofas_found": len(copied),
        "output_root": str(output_root),
        "selected_ids": selected_ids,
        "splits": splits,
        "validation": report,
    }
