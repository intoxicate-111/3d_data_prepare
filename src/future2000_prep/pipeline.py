from __future__ import annotations

import json
import random
import zipfile
from pathlib import Path
from typing import Any

import fast_simplification
import numpy as np
import trimesh
from tqdm import tqdm

from sofa50_prep.pipeline import (
    Candidate,
    _clean_mesh,
    _copy_raw_mesh_from_archive,
    _evaluate_candidate,
    _find_archive,
    _metadata_entries,
    _model_id,
    _near_duplicate,
    _process_selected,
)
from sofa50_refinement.pipeline import (
    MeshData,
    _load_npz_mesh,
    _load_obj,
    _mesh_issues,
    _save_mesh,
    _vertex_components,
)
from thingi10k50_prep.mesh_ops import midpoint_subdivide


SUPER_CATEGORIES = (
    "Cabinet/Shelf/Desk",
    "Sofa",
    "Lighting",
    "Chair",
    "Others",
    "Bed",
    "Table",
    "Pier/Stool",
)
FORMAT_VERSION = "3d_future_2000_final_v2"
INFERENCE_FORMAT_VERSION = "3d_future_2000_inference_v3"


def prepare_future2000(
    downloads: str | Path,
    output_root: str | Path,
    *,
    count: int = 2000,
    seed: int = 20260806,
    source_up_axis: str = "y",
    target_max_extent: float = 1.0,
    target_faces: int = 40_000,
    max_faces: int = 50_000,
    per_category: int | None = None,
) -> dict[str, Any]:
    """Select an equal-size, deterministic multi-category 3D-FUTURE GT set."""

    downloads = Path(downloads).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    raw_root = output_root / "raw"
    gt_root = output_root / "gt"
    output_root.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(exist_ok=True)
    gt_root.mkdir(exist_ok=True)
    if per_category is None:
        if count % len(SUPER_CATEGORIES):
            raise ValueError("count must be divisible by the eight 3D-FUTURE super-categories")
        per_category = count // len(SUPER_CATEGORIES)
    if per_category * len(SUPER_CATEGORIES) != count:
        raise ValueError("per_category * 8 must equal count")

    archive_path = _find_archive(downloads)
    if archive_path is None:
        raise FileNotFoundError(f"No official 3D-FUTURE model zip found under {downloads}")
    model_info = downloads / "3D-FUTURE-model" / "model_info.json"
    if not model_info.is_file():
        raise FileNotFoundError(f"Missing extracted model_info.json: {model_info}")
    entries = _metadata_entries(model_info)
    grouped = {category: [] for category in SUPER_CATEGORIES}
    for entry in entries:
        category = str(entry.get("super-category", "")).strip()
        if category in grouped and _model_id(entry):
            grouped[category].append(entry)
    availability = {category: len(values) for category, values in grouped.items()}
    insufficient = {key: value for key, value in availability.items() if value < per_category}
    if insufficient:
        raise RuntimeError(f"Insufficient metadata records for balanced selection: {insufficient}")

    selected: dict[str, list[dict[str, Any]]] = {category: [] for category in SUPER_CATEGORIES}
    failures: list[dict[str, str]] = []
    global_ids: set[str] = set()
    with zipfile.ZipFile(archive_path) as bundle:
        archive_names = set(bundle.namelist())
        for category_index, category in enumerate(SUPER_CATEGORIES):
            candidates = sorted(grouped[category], key=_model_id)
            random.Random(seed + category_index * 104729).shuffle(candidates)
            accepted_candidates: list[Candidate] = []
            progress = tqdm(total=per_category, desc=f"Selecting {category}")
            for entry in candidates:
                if len(accepted_candidates) == per_category:
                    break
                model_id = _model_id(entry)
                if model_id in global_ids:
                    continue
                raw_path = raw_root / model_id / "raw_model.obj"
                if not raw_path.is_file() and not _copy_raw_mesh_from_archive(
                    bundle,
                    archive_names,
                    "3D-FUTURE-model",
                    model_id,
                    raw_path,
                ):
                    failures.append({"model_id": model_id, "category": category, "reason": "mesh_missing"})
                    continue
                try:
                    candidate = _evaluate_candidate(model_id, category, raw_path, source_up_axis)
                    if candidate.vertex_count < 128 or candidate.face_count < 128:
                        raise ValueError("mesh_too_small")
                    if candidate.connected_components > 512:
                        raise ValueError("too_many_connected_components")
                    if any(_near_duplicate(candidate, previous) for previous in accepted_candidates):
                        raise ValueError("near_duplicate_within_category")
                    # The Sofa pipeline caches by its own pipeline version.  The
                    # generic set also makes the render-safe target extent part
                    # of the cache key so a parameter change cannot silently
                    # retain geometry at an older scale.
                    gt_info = gt_root / model_id / "info.json"
                    if gt_info.is_file():
                        cached_info = json.loads(gt_info.read_text(encoding="utf-8"))
                        if not np.isclose(
                            float(cached_info.get("target_max_extent", -1.0)),
                            target_max_extent,
                        ):
                            gt_info.unlink()
                    info = _process_selected(
                        candidate,
                        gt_root,
                        source_up_axis,
                        target_max_extent,
                        target_faces,
                        max_faces,
                    )
                    info = _repair_gt_roundtrip_if_needed(gt_root, model_id, info)
                    accepted_candidates.append(candidate)
                    global_ids.add(model_id)
                    selected[category].append(
                        {
                            "model_id": model_id,
                            "super_category": category,
                            "category": str(entry.get("category", "")),
                            "style": str(entry.get("style", "")),
                            "theme": str(entry.get("theme", "")),
                            "material": str(entry.get("material", "")),
                            "gt_mesh": str((gt_root / model_id / "mesh.obj").resolve()),
                            "vertex_count": int(info["vertex_count"]),
                            "face_count": int(info["face_count"]),
                            "geometry_hash": candidate.geometry_hash,
                        }
                    )
                    progress.update(1)
                except Exception as error:  # noqa: BLE001
                    failures.append(
                        {"model_id": model_id, "category": category, "reason": str(error)}
                    )
            progress.close()
            if len(accepted_candidates) != per_category:
                raise RuntimeError(
                    f"Only selected {len(accepted_candidates)}/{per_category} usable {category} models"
                )

    splits = _balanced_splits(selected, seed)
    _write_splits(gt_root, splits)
    records = [item for category in SUPER_CATEGORIES for item in selected[category]]
    split_by_id = {
        model_id: split for split, values in splits.items() for model_id in values
    }
    for record in records:
        record["split"] = split_by_id[record["model_id"]]
    records.sort(key=lambda item: (("train", "validation", "test").index(item["split"]), item["model_id"]))
    manifest = {
        "format_version": FORMAT_VERSION,
        "source": "official_3D-FUTURE-model",
        "source_archive": str(archive_path),
        "seed": seed,
        "selection_policy": "equal_250_per_official_super_category",
        "count": count,
        "per_category": per_category,
        "category_availability": availability,
        "category_counts": {category: len(selected[category]) for category in SUPER_CATEGORIES},
        "split_counts": {key: len(value) for key, value in splits.items()},
        "split_policy": "per_category_80_10_10",
        "normalization_policy": "center_bbox_then_scale_bbox_max_extent",
        "target_max_extent": target_max_extent,
        "render_fit_rationale": (
            "extent 1.0 keeps every centered bbox within the downstream cube-camera frusta"
        ),
        "gt_root": str(gt_root),
        "samples": records,
        "failure_count_during_candidate_search": len(failures),
        "candidate_failures": failures,
    }
    _write_json(output_root / "selection_manifest.json", manifest)
    validation = validate_future2000(gt_root, output_root / "selection_manifest.json", count)
    return {
        "output_root": str(output_root),
        "gt_root": str(gt_root),
        "manifest": str((output_root / "selection_manifest.json").resolve()),
        "count": count,
        "validation": validation,
    }


def prepare_future2000_inference(
    gt_root: str | Path,
    output_root: str | Path,
    *,
    coarse_target_vertices: int = 3500,
    coarse_min_vertices: int = 32,
    force: bool = False,
) -> dict[str, Any]:
    """Build leakage-free raw coarse/expanded inference queries without oracle targets."""

    gt_root = Path(gt_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    models_root = output_root / "models"
    models_root.mkdir(exist_ok=True)
    splits = _read_splits(gt_root)
    owner = {model_id: split for split, values in splits.items() for model_id in values}
    ordered = [model_id for split in ("train", "validation", "test") for model_id in splits[split]]
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for model_id in tqdm(ordered, desc="Preparing 3D-FUTURE-2000 inference meshes"):
        model_dir = models_root / model_id
        metadata_path = model_dir / "metadata.json"
        if metadata_path.is_file() and not force:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("format_version") == INFERENCE_FORMAT_VERSION:
                rows.append(metadata["manifest_record"])
                continue
        try:
            gt = _load_obj(gt_root / model_id / "mesh.obj")
            issues = _mesh_issues(gt)
            if issues:
                gt, cleanup = _clean_mesh_data(gt)
                _persist_repaired_gt(gt_root, model_id, gt, cleanup)
                _update_selection_counts(gt_root.parent / "selection_manifest.json", model_id, gt)
            coarse, simplify_meta = _simplify_raw_componentwise(
                gt, target_vertices=coarse_target_vertices, min_vertices=coarse_min_vertices
            )
            expanded_arrays, subdivision = midpoint_subdivide(
                coarse.vertices, coarse.faces, steps=1
            )
            expanded = MeshData(expanded_arrays.vertices, expanded_arrays.faces)
            if _mesh_issues(coarse) or _mesh_issues(expanded):
                raise ValueError("coarse or expanded mesh failed structural validation")
            model_dir.mkdir(parents=True, exist_ok=True)
            gt_files = _save_mesh(model_dir / "gt_mesh", gt)
            coarse_files = _save_mesh(model_dir / "coarse_raw", coarse)
            expanded_files = _save_mesh(model_dir / "expanded_initial_raw", expanded)
            mapping_path = model_dir / "subdivision_mapping_raw.npz"
            np.savez_compressed(mapping_path, **subdivision)
            info = json.loads((gt_root / model_id / "info.json").read_text(encoding="utf-8"))
            record = {
                "model_id": model_id,
                "split": owner[model_id],
                "status": "valid",
                "super_category": info.get("category", ""),
                "gt_obj": gt_files["obj"],
                "gt_npz": gt_files["npz"],
                "gt_vertices": len(gt.vertices),
                "gt_faces": len(gt.faces),
                "coarse_raw_obj": coarse_files["obj"],
                "coarse_raw_npz": coarse_files["npz"],
                "coarse_raw_vertices": len(coarse.vertices),
                "coarse_raw_faces": len(coarse.faces),
                "expanded_initial_raw_obj": expanded_files["obj"],
                "expanded_initial_raw_npz": expanded_files["npz"],
                "expanded_initial_raw_vertices": len(expanded.vertices),
                "expanded_initial_raw_faces": len(expanded.faces),
                "subdivision_mapping_raw_npz": str(mapping_path.resolve()),
            }
            _write_json(
                metadata_path,
                {
                    "format_version": INFERENCE_FORMAT_VERSION,
                    "model_id": model_id,
                    "usage_role": "frozen_model_inference_query",
                    "gt_guided_query_modification": False,
                    "oracle_target_generated": False,
                    "coarse_generation_policy": "componentwise_direct_qem_then_component_local_cleanup",
                    "simplification": simplify_meta,
                    "manifest_record": record,
                },
            )
            rows.append(record)
        except Exception as error:  # noqa: BLE001
            failures.append({"model_id": model_id, "split": owner[model_id], "reason": str(error)})
    if failures:
        _write_json(output_root / "failures.json", failures)
        raise RuntimeError(f"Inference generation failed for {len(failures)} samples")
    for split, values in splits.items():
        (output_root / f"{split}.txt").write_text("\n".join(values) + "\n", encoding="utf-8")
    manifest = {
        "format_version": INFERENCE_FORMAT_VERSION,
        "dataset": "3D-FUTURE-2000 training and inference geometry",
        "source_root": str(gt_root),
        "output_root": str(output_root),
        "split_counts": {key: len(value) for key, value in splits.items()},
        "success_count": len(rows),
        "failure_count": 0,
        "usage_contract": {
            "training_geometry": "gt_mesh_on_gt_connectivity_only",
            "training_target": "delta_gt = L(gt_faces) @ gt_vertices",
            "inference_path": "coarse_raw -> expanded_initial_raw -> frozen model -> reconstruction",
            "oracle_target_generated": False,
        },
        "samples": rows,
        "failures": [],
    }
    _write_json(output_root / "failures.json", [])
    _write_json(output_root / "manifest.json", manifest)
    validation = validate_future2000_inference(
        output_root / "manifest.json", expected_count=len(ordered)
    )
    return {
        "manifest": str((output_root / "manifest.json").resolve()),
        "success_count": len(rows),
        "validation": validation,
    }


def validate_future2000_inference(
    manifest_path: str | Path, expected_count: int = 2000
) -> dict[str, Any]:
    """Validate every saved query and reconstruct midpoint subdivision exactly."""

    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = manifest.get("samples", [])
    issues: dict[str, list[str]] = {}
    ids = [str(record.get("model_id", "")) for record in samples]
    if len(ids) != expected_count or len(set(ids)) != expected_count:
        raise RuntimeError("inference manifest does not contain the expected unique sample count")
    for record in tqdm(samples, desc="Validating 3D-FUTURE-2000 inference meshes"):
        model_id = str(record["model_id"])
        model_issues: list[str] = []
        try:
            gt = _load_npz_mesh(Path(record["gt_npz"]))
            coarse = _load_npz_mesh(Path(record["coarse_raw_npz"]))
            expanded = _load_npz_mesh(Path(record["expanded_initial_raw_npz"]))
            for role, mesh in (("gt", gt), ("coarse_raw", coarse), ("expanded_initial_raw", expanded)):
                model_issues.extend(f"{role}:{issue}" for issue in _mesh_issues(mesh))
            reconstructed_arrays, reconstructed_mapping = midpoint_subdivide(
                coarse.vertices, coarse.faces, steps=1
            )
            if not np.array_equal(reconstructed_arrays.faces, expanded.faces):
                model_issues.append("expanded_faces_not_midpoint_subdivision")
            if not np.array_equal(reconstructed_arrays.vertices, expanded.vertices):
                model_issues.append("expanded_vertices_not_midpoint_subdivision")
            saved_mapping = np.load(record["subdivision_mapping_raw_npz"])
            if set(saved_mapping.files) != set(reconstructed_mapping):
                model_issues.append("subdivision_mapping_fields_mismatch")
            else:
                for key, expected in reconstructed_mapping.items():
                    if not np.array_equal(saved_mapping[key], expected):
                        model_issues.append(f"subdivision_mapping_value_mismatch:{key}")
            metadata_path = Path(record["coarse_raw_npz"]).parent / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("format_version") != INFERENCE_FORMAT_VERSION:
                model_issues.append("stale_format_version")
            if metadata.get("gt_guided_query_modification") is not False:
                model_issues.append("gt_guided_query_modification_not_false")
            if metadata.get("oracle_target_generated") is not False:
                model_issues.append("oracle_target_generated_not_false")
            if metadata.get("coarse_generation_policy") != (
                "componentwise_direct_qem_then_component_local_cleanup"
            ):
                model_issues.append("unexpected_coarse_generation_policy")
            forbidden = {
                "P_target_oracle.npz",
                "coarse_registered_oracle.npz",
                "surface_target_oracle.npz",
            }
            if any((metadata_path.parent / name).exists() for name in forbidden):
                model_issues.append("oracle_artifact_present")
        except Exception as error:  # noqa: BLE001
            model_issues.append(f"validation_exception:{error}")
        if model_issues:
            issues[model_id] = model_issues
    result = {
        "format_version": INFERENCE_FORMAT_VERSION,
        "expected_count": expected_count,
        "valid_count": expected_count - len(issues),
        "invalid_count": len(issues),
        "exact_midpoint_reconstruction_count": expected_count - len(issues),
        "gt_guided_query_modification": False,
        "oracle_target_generated": False,
        "issues": issues,
    }
    _write_json(manifest_path.parent / "validation.json", result)
    if issues:
        raise RuntimeError(f"Inference validation failed for {len(issues)} samples")
    return result


def validate_future2000(
    gt_root: str | Path, manifest_path: str | Path, expected_count: int = 2000
) -> dict[str, Any]:
    gt_root = Path(gt_root).resolve()
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    splits = _read_splits(gt_root)
    ids = [model_id for split in ("train", "validation", "test") for model_id in splits[split]]
    issues: dict[str, list[str]] = {}
    if len(ids) != expected_count or len(set(ids)) != expected_count:
        raise RuntimeError("split IDs are not exactly the requested unique sample count")
    expected_splits = {"train": 1600, "validation": 200, "test": 200}
    if expected_count == 2000 and {key: len(value) for key, value in splits.items()} != expected_splits:
        raise RuntimeError("expected split sizes 1600/200/200")
    category_by_id = {item["model_id"]: item["super_category"] for item in manifest["samples"]}
    for model_id in ids:
        model_issues = []
        try:
            mesh = trimesh.load(gt_root / model_id / "mesh.obj", force="mesh", process=False)
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.faces)
            if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
                model_issues.append("invalid_vertices")
            if faces.ndim != 2 or faces.shape[1] != 3 or np.any(faces < 0) or np.any(faces >= len(vertices)):
                model_issues.append("invalid_faces")
            elif len(faces):
                triangles = vertices[faces]
                double_area = np.linalg.norm(
                    np.cross(
                        triangles[:, 1] - triangles[:, 0],
                        triangles[:, 2] - triangles[:, 0],
                    ),
                    axis=1,
                )
                if np.any(~np.isfinite(double_area)) or np.any(double_area <= 0):
                    model_issues.append("degenerate_faces")
            extents = np.ptp(vertices, axis=0)
            target_extent = float(manifest.get("target_max_extent", 1.0))
            if not np.isfinite(extents).all() or float(extents.max()) > target_extent + 1e-5:
                model_issues.append("normalization_extent_exceeded")
            if not (gt_root / model_id / "info.json").is_file():
                model_issues.append("missing_info")
        except Exception as error:  # noqa: BLE001
            model_issues.append(f"load_failure:{error}")
        if model_issues:
            issues[model_id] = model_issues
    category_split_counts = {
        category: {
            split: sum(category_by_id.get(model_id) == category for model_id in splits[split])
            for split in ("train", "validation", "test")
        }
        for category in SUPER_CATEGORIES
    }
    result = {
        "expected_count": expected_count,
        "valid_count": expected_count - len(issues),
        "invalid_count": len(issues),
        "split_counts": {key: len(value) for key, value in splits.items()},
        "category_split_counts": category_split_counts,
        "issues": issues,
    }
    _write_json(gt_root / "validation.json", result)
    if issues:
        raise RuntimeError(f"GT validation failed for {len(issues)} samples")
    return result


def _clean_mesh_data(mesh: MeshData) -> tuple[MeshData, list[str]]:
    value = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)
    cleaned, operations = _clean_mesh(value)
    result = MeshData(
        np.asarray(cleaned.vertices, dtype=np.float64),
        np.asarray(cleaned.faces, dtype=np.int64),
    )
    issues = _mesh_issues(result)
    if issues:
        raise RuntimeError(f"mesh remains invalid after cleanup: {issues}")
    return result, operations


def _simplify_raw_componentwise(
    gt: MeshData,
    *,
    target_vertices: int,
    min_vertices: int,
) -> tuple[MeshData, dict[str, Any]]:
    """Direct component-wise QEM for inference; no provenance replay is needed."""

    vertex_components, component_faces = _vertex_components(gt)
    ratio = min(1.0, target_vertices / len(gt.vertices))
    vertices_out: list[np.ndarray] = []
    faces_out: list[np.ndarray] = []
    component_records: list[dict[str, Any]] = []
    offset = 0
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
            simplified = local
            skipped = True
        else:
            result_vertices, result_faces = fast_simplification.simplify(
                np.asarray(local.vertices, dtype=np.float64),
                np.asarray(local.faces, dtype=np.int64),
                target_count=requested_faces,
                agg=7.0,
            )
            simplified = MeshData(
                np.asarray(result_vertices, dtype=np.float64),
                np.asarray(result_faces, dtype=np.int64),
            )
            skipped = False
        cleanup: list[str] = []
        invalid_qem_fallback = False
        if _mesh_issues(simplified):
            try:
                simplified, cleanup = _clean_mesh_data(simplified)
            except (TypeError, ValueError):
                # QEM can return an empty result for a tiny otherwise-valid
                # disconnected component (for example five vertices/nine
                # faces). Preserve that component instead of deleting it.
                simplified = local
                cleanup = ["invalid_qem_fallback_to_unsimplified_component"]
                invalid_qem_fallback = True
        vertices_out.append(simplified.vertices)
        faces_out.append(simplified.faces + offset)
        component_records.append(
            {
                "component": int(component),
                "gt_vertices": int(len(local.vertices)),
                "gt_faces": int(len(local.faces)),
                "coarse_vertices": int(len(simplified.vertices)),
                "coarse_faces": int(len(simplified.faces)),
                "simplification_skipped": skipped,
                "invalid_qem_fallback": invalid_qem_fallback,
                "cleanup_operations": cleanup,
            }
        )
        offset += len(simplified.vertices)
    coarse = MeshData(
        np.vstack(vertices_out),
        np.vstack(faces_out).astype(np.int64),
    )
    issues = _mesh_issues(coarse)
    if issues:
        raise RuntimeError(f"direct component-wise QEM produced invalid coarse mesh: {issues}")
    if len(coarse.vertices) < min_vertices:
        raise RuntimeError(
            f"coarse mesh has {len(coarse.vertices)} vertices; minimum is {min_vertices}"
        )
    return coarse, {
        "method": "fast_simplification_direct_componentwise_qem",
        "componentwise": True,
        "gt_components": int(len(component_faces)),
        "requested_target_vertices": int(target_vertices),
        "actual_coarse_vertices": int(len(coarse.vertices)),
        "components": component_records,
    }


def _persist_repaired_gt(
    gt_root: Path,
    model_id: str,
    mesh: MeshData,
    operations: list[str],
) -> dict[str, Any]:
    model_dir = gt_root / model_id
    value = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)
    value.export(model_dir / "mesh.obj", file_type="obj")
    value.export(model_dir / "mesh.ply", file_type="ply", encoding="binary")
    info_path = model_dir / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["vertex_count"] = int(len(mesh.vertices))
    info["face_count"] = int(len(mesh.faces))
    existing = list(info.get("processing_operations", []))
    info["processing_operations"] = list(dict.fromkeys(existing + operations + ["validated_obj_roundtrip"]))
    _write_json(info_path, info)
    return info


def _repair_gt_roundtrip_if_needed(
    gt_root: Path,
    model_id: str,
    info: dict[str, Any],
) -> dict[str, Any]:
    mesh = _load_obj(gt_root / model_id / "mesh.obj")
    if not _mesh_issues(mesh):
        return info
    cleaned, operations = _clean_mesh_data(mesh)
    repaired = _persist_repaired_gt(gt_root, model_id, cleaned, operations)
    roundtrip = _load_obj(gt_root / model_id / "mesh.obj")
    issues = _mesh_issues(roundtrip)
    if issues:
        raise RuntimeError(f"GT OBJ remains invalid after round-trip repair: {issues}")
    return repaired


def _update_selection_counts(path: Path, model_id: str, mesh: MeshData) -> None:
    if not path.is_file():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    for record in payload.get("samples", []):
        if record.get("model_id") == model_id:
            record["vertex_count"] = int(len(mesh.vertices))
            record["face_count"] = int(len(mesh.faces))
            break
    _write_json(path, payload)


def _balanced_splits(
    selected: dict[str, list[dict[str, Any]]], seed: int
) -> dict[str, list[str]]:
    result = {"train": [], "validation": [], "test": []}
    for index, category in enumerate(SUPER_CATEGORIES):
        ids = sorted(item["model_id"] for item in selected[category])
        random.Random(seed + index * 65537).shuffle(ids)
        train_count = round(len(ids) * 0.8)
        validation_count = round(len(ids) * 0.1)
        result["train"].extend(ids[:train_count])
        result["validation"].extend(ids[train_count : train_count + validation_count])
        result["test"].extend(ids[train_count + validation_count :])
    for index, split in enumerate(("train", "validation", "test")):
        random.Random(seed + index).shuffle(result[split])
    return result


def _read_splits(root: Path) -> dict[str, list[str]]:
    return {
        split: [line.strip() for line in (root / f"{split}.txt").read_text().splitlines() if line.strip()]
        for split in ("train", "validation", "test")
    }


def _write_splits(root: Path, splits: dict[str, list[str]]) -> None:
    for split, values in splits.items():
        (root / f"{split}.txt").write_text("\n".join(values) + "\n", encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
