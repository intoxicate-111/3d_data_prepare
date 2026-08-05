from __future__ import annotations

import hashlib
import inspect
import json
import random
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import thingi10k
import torch
import trimesh
from tqdm import tqdm
from PIL import Image

from .config import PrepareConfig
from .downstream import DownstreamRuntime, validate_downstream
from .io_utils import ensure_dir, save_mesh_npz, setup_logger, write_csv, write_json
from .mesh_ops import (
    build_uniform_laplacian,
    compute_surface_targets,
    extract_vertices_faces,
    mesh_cleanup,
    midpoint_subdivide,
    normalize_vertices,
    simplify_mesh,
)
from .rendering import CUBE_SURFACE_VIEW_NAMES, VIEW_LAYOUT_VERSION, generate_configured_synthetic_dataset


def _entry_int(entry: dict[str, Any], keys: tuple[str, ...], default: int = -1) -> int:
    for key in keys:
        if key in entry and entry[key] is not None:
            try:
                return int(entry[key])
            except (TypeError, ValueError):
                continue
    return default


def _entry_str(entry: dict[str, Any], keys: tuple[str, ...], default: str = "") -> str:
    for key in keys:
        if key in entry and entry[key] is not None:
            return str(entry[key])
    return default


def _mesh_metrics(vertices: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    bbox = mesh.bounds
    watertight = bool(mesh.is_watertight)
    return {
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "components": int(len(mesh.split(only_watertight=False))),
        "watertight": watertight,
        "bbox_min": bbox[0].tolist(),
        "bbox_max": bbox[1].tolist(),
    }


def _save_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    path.write_text(mesh.export(file_type="obj"), encoding="utf-8")


def _mesh_checksum(vertices: np.ndarray, faces: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(vertices, dtype=np.float64).tobytes())
    digest.update(np.asarray(faces, dtype=np.int64).tobytes())
    return digest.hexdigest()


def _expected_geometry_contract(cfg: PrepareConfig) -> dict[str, Any]:
    return {
        "normalization_mode": cfg.normalization_mode,
        "normalization_epsilon": cfg.normalization_epsilon,
        "coarse_target_vertices": cfg.coarse_target_vertices,
        "coarse_min_vertices": cfg.coarse_min_vertices,
        "subdivision_steps": cfg.subdivision_steps,
        "target_constructor": "precomputed_closest_surface_on_prediction_graph",
        "laplacian_operator": "uniform",
    }


def _render_cache_issues(
    render_metadata: dict[str, Any],
    cfg: PrepareConfig,
    normalized_mesh_checksum: str,
) -> list[str]:
    issues: list[str] = []
    if render_metadata.get("trajectory") != cfg.views_trajectory:
        issues.append("unexpected_view_trajectory")
    if render_metadata.get("requested_backend", render_metadata.get("backend")) != cfg.views_backend:
        issues.append("unexpected_render_backend")
    expected = {
        "opengl_context_backend": cfg.views_opengl_context_backend,
        "cube_half_extent": cfg.views_cube_half_extent,
        "fov_degrees": cfg.views_fov_degrees,
        "render_mode": cfg.views_render_mode,
        "antialiasing": cfg.views_antialiasing,
        "camera_layout_version": VIEW_LAYOUT_VERSION,
    }
    issues.extend(
        f"unexpected_render_{key}"
        for key, expected_value in expected.items()
        if render_metadata.get(key) != expected_value
    )
    if render_metadata.get("normalized_mesh_checksum") != normalized_mesh_checksum:
        issues.append("render_mesh_checksum_changed")
    if (
        render_metadata.get("width") != cfg.views_width
        or render_metadata.get("height") != cfg.views_height
    ):
        issues.append("unexpected_view_resolution")
    return issues


def _prepared_cache_issues(raw_prepared: dict[str, Any], cfg: PrepareConfig) -> list[str]:
    issues: list[str] = []
    if "images" in raw_prepared:
        issues.append("prepared_sample_embeds_images")
    if raw_prepared.get("prepared_storage_format") != cfg.prepared_samples.storage_format:
        issues.append("prepared_sample_storage_contract_changed")
    if len(raw_prepared.get("image_paths", [])) != cfg.views_count:
        issues.append("prepared_sample_image_paths_changed")
    if tuple(raw_prepared.get("intrinsics", torch.empty(0)).shape) != (cfg.views_count, 3, 3):
        issues.append("prepared_sample_intrinsics_changed")
    if tuple(raw_prepared.get("extrinsics", torch.empty(0)).shape) != (cfg.views_count, 4, 4):
        issues.append("prepared_sample_extrinsics_changed")
    metadata = raw_prepared.get("metadata", {})
    if metadata.get("normalization_mode") != cfg.normalization_mode:
        issues.append("prepared_sample_normalization_changed")
    if metadata.get("view_layout_version") != VIEW_LAYOUT_VERSION:
        issues.append("prepared_sample_view_layout_changed")
    return issues


def _build_prepared_sample(
    dataset_json: Path,
    expanded_obj: Path,
    gt_obj: Path,
    targets: dict[str, np.ndarray],
    laplacian: sp.csr_matrix,
    image_size: int,
    sample_id: str,
    metadata: dict[str, Any],
    dataset_root: Path,
) -> dict[str, Any]:
    """Assemble existing projections through the downstream loader/scaling contract."""
    from mlr.datasets import load_masks, load_reconstruction_input
    from mlr.io import load_mesh
    from mlr.learned_laplacian.dataset import validate_sample
    from mlr.learned_laplacian.sample_io import _mask_visibility, _resize_mask

    reconstruction = load_reconstruction_input(dataset_json)
    render_metadata = json.loads(dataset_json.read_text(encoding="utf-8")).get("config", {})
    metadata = dict(metadata)
    metadata["view_backend"] = render_metadata.get("backend", metadata.get("view_backend"))
    prediction_mesh = load_mesh(expanded_obj).ensure_normals()
    gt_mesh = load_mesh(gt_obj).ensure_normals()
    source_sizes = [Image.open(path).size for path in reconstruction.image_paths]
    if not source_sizes or len(set(source_sizes)) != 1:
        raise ValueError("All sample images must have one consistent non-empty source size")
    source_width, source_height = source_sizes[0]
    scale_xy = (image_size / source_width, image_size / source_height)
    intrinsics: list[np.ndarray] = []
    extrinsics: list[np.ndarray] = []
    for camera in reconstruction.cameras:
        scaled = camera.intrinsics.copy()
        scaled[0, :] *= scale_xy[0]
        scaled[1, :] *= scale_xy[1]
        intrinsics.append(scaled)
        extrinsic = np.eye(4, dtype=np.float64)
        extrinsic[:3, :3] = camera.rotation
        extrinsic[:3, 3] = camera.translation
        extrinsics.append(extrinsic)
    masks = load_masks(reconstruction.mask_paths)
    visibility = None
    if masks is not None:
        resized_masks = [_resize_mask(mask, (image_size, image_size)) for mask in masks]
        visibility = _mask_visibility(
            prediction_mesh.vertices, reconstruction.cameras, resized_masks, scale_xy
        )
    relative_image_paths = [
        path.resolve().relative_to(dataset_root.resolve()).as_posix()
        for path in reconstruction.image_paths
    ]
    raw_target = laplacian @ targets["target_positions"]
    sample = {
        "sample_id": sample_id,
        "image_paths": relative_image_paths,
        "source_image_size": [source_width, source_height],
        "prepared_image_size": image_size,
        "prepared_storage_format": "lazy_image_paths_v1",
        "intrinsics": torch.as_tensor(np.stack(intrinsics), dtype=torch.float32),
        "extrinsics": torch.as_tensor(np.stack(extrinsics), dtype=torch.float32),
        "vertices": torch.as_tensor(prediction_mesh.vertices, dtype=torch.float32),
        "faces": torch.as_tensor(prediction_mesh.faces, dtype=torch.long),
        "vertex_normals": torch.as_tensor(prediction_mesh.normals, dtype=torch.float32),
        "initial_laplacian": torch.as_tensor(laplacian @ prediction_mesh.vertices, dtype=torch.float32),
        "laplacian_target": torch.as_tensor(raw_target, dtype=torch.float32),
        "raw_laplacian_target": torch.as_tensor(raw_target, dtype=torch.float32),
        "target_confidence": torch.as_tensor(targets["valid_mask"], dtype=torch.float32),
        "visibility": None if visibility is None else torch.as_tensor(visibility, dtype=torch.bool),
        "target_positions": torch.as_tensor(targets["target_positions"], dtype=torch.float32),
        "gt_vertices": torch.as_tensor(gt_mesh.vertices, dtype=torch.float32),
        "gt_faces": torch.as_tensor(gt_mesh.faces, dtype=torch.long),
        "metadata": metadata,
    }
    return validate_sample(sample)


def _preview(path: Path, gt_v: np.ndarray, coarse_v: np.ndarray, expanded_v: np.ndarray) -> None:
    fig = plt.figure(figsize=(9, 3))
    axes = [fig.add_subplot(1, 3, i + 1, projection="3d") for i in range(3)]
    samples = [
        ("Ground truth", gt_v),
        ("Coarse", coarse_v),
        ("Expanded", expanded_v),
    ]
    for ax, (title, points) in zip(axes, samples):
        subset = points[:: max(len(points) // 3000, 1)]
        ax.scatter(subset[:, 0], subset[:, 1], subset[:, 2], s=0.5)
        ax.set_title(title)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_contract_report(root: Path, runtime: DownstreamRuntime) -> None:
    reports = root / "reports"
    ensure_dir(reports)
    contract = reports / "existing_pipeline_contract.md"

    try:
        data_prep_sha = subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:  # noqa: BLE001
        data_prep_sha = "unknown"
    payload = (
        "# Existing pipeline contract\n\n"
        f"- Repository URL: `{runtime.url}`\n"
        f"- Repository root: `{runtime.root}`\n"
        f"- Active branch: `{runtime.branch}`\n"
        f"- Active commit: `{runtime.sha}`\n"
        f"- Data-preparation commit: `{data_prep_sha}`\n"
        "- Entry point: `mlr.cli:main` via `mlr` console script\n"
        "- Directory structure expected by this project: `views/images`, `views/masks`, `views/depth`, `views/cameras.json`, `views/dataset.json`, `views/mesh.obj`\n"
        "- Normalization convention: AABB-center + maximum-radius-to-unit-sphere, with renderer `normalize_mesh=False`\n"
        "- Coarse mesh target: 3500 vertices\n"
        "- Midpoint subdivision steps: 1\n"
        "- Laplacian type: downstream uniform operator `I - D^-1 A`\n"
        "- Target quantity: `delta_target = L_exp @ target_positions`\n"
        "- Multi-view inputs: downstream-compatible synthetic renderer from `mlr.synthetic`\n"
        "- Train/validation/test expectations are defined by the active preparation config\n"
    )
    contract.write_text(payload, encoding="utf-8")


def _candidate_reason(vertices: np.ndarray, faces: np.ndarray, cfg: PrepareConfig) -> str | None:
    if vertices.size == 0 or faces.size == 0:
        return "empty_vertices_or_faces"
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        return "invalid_vertex_shape"
    if faces.ndim != 2 or faces.shape[1] != 3:
        return "non_triangle_faces"
    if not np.isfinite(vertices).all():
        return "non_finite_vertices"
    if np.min(faces) < 0 or np.max(faces) >= len(vertices):
        return "invalid_face_indices"
    if len(vertices) < cfg.min_vertices or len(faces) < cfg.min_faces:
        return "below_minimum_complexity"
    if len(faces) > cfg.max_faces:
        return "too_many_faces"
    return None


def _post_cleanup_reason(vertices: np.ndarray, faces: np.ndarray, cfg: PrepareConfig) -> str | None:
    if vertices.size == 0 or faces.size == 0:
        return "empty_vertices_or_faces_after_cleanup"
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        return "invalid_vertex_shape_after_cleanup"
    if faces.ndim != 2 or faces.shape[1] != 3:
        return "non_triangle_faces_after_cleanup"
    if not np.isfinite(vertices).all():
        return "non_finite_vertices_after_cleanup"
    if np.min(faces) < 0 or np.max(faces) >= len(vertices):
        return "invalid_face_indices_after_cleanup"
    repeated_idx = (faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2]) | (faces[:, 0] == faces[:, 2])
    if np.any(repeated_idx):
        return "repeated_indices_in_face_after_cleanup"
    tri = vertices[faces]
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) * 0.5
    if np.any(area <= 0):
        return "non_positive_triangle_area_after_cleanup"
    if len(vertices) < cfg.min_vertices or len(faces) < cfg.min_faces:
        return "below_minimum_complexity_after_cleanup"
    return None


def _stratum_for_face_count(face_count: int, strata: list[dict[str, Any]]) -> str:
    for s in strata:
        if s["min_faces"] <= face_count <= s["max_faces"]:
            return str(s["name"])
    return "out_of_range"


def _expected_model_count(cfg: PrepareConfig) -> int:
    stratum_total = sum(stratum.count for stratum in cfg.strata)
    split_total = cfg.split.train + cfg.split.val + cfg.split.test

    if stratum_total != split_total:
        raise ValueError(
            "Stratum and split totals must match: "
            f"strata={stratum_total}, split={split_total}"
        )

    return stratum_total


def _group_key(row: dict[str, Any]) -> tuple[str, int]:
    thing_id = int(row.get("thing_id", -1))
    if thing_id > 0:
        return ("thing", thing_id)
    return ("file", int(row["file_id"]))


def _sample_stratified(valid_rows: list[dict[str, Any]], cfg: PrepareConfig) -> list[dict[str, Any]]:
    rng = random.Random(cfg.seed)
    by_stratum: dict[str, list[dict[str, Any]]] = {s.name: [] for s in cfg.strata}
    for row in valid_rows:
        if row["stratum"] in by_stratum:
            by_stratum[row["stratum"]].append(row)

    target_count = _expected_model_count(cfg)
    candidate_counts = {name: len(rows) for name, rows in by_stratum.items()}
    if len(valid_rows) < target_count or any(candidate_counts[s.name] < s.count for s in cfg.strata):
        raise RuntimeError(
            "Insufficient candidates for stratified sampling: "
            f"target_count={target_count}, valid_rows={len(valid_rows)}, "
            f"stratum_candidates={candidate_counts}"
        )

    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()
    used_positive_things: set[int] = set()
    remaining_by_stratum: dict[str, list[dict[str, Any]]] = {}

    # First pass: satisfy each stratum with as many distinct positive thing IDs
    # as possible. Missing/non-positive thing IDs are independent file groups.
    for s in cfg.strata:
        pool = by_stratum[s.name].copy()
        rng.shuffle(pool)
        pick: list[dict[str, Any]] = []
        for row in pool:
            file_id = int(row["file_id"])
            if file_id in selected_ids:
                continue
            thing_id = int(row.get("thing_id", -1))
            if thing_id > 0 and thing_id in used_positive_things:
                continue
            pick.append(row)
            selected_ids.add(file_id)
            if thing_id > 0:
                used_positive_things.add(thing_id)
            if len(pick) == s.count:
                break
        selected.extend(pick)
        remaining_by_stratum[s.name] = pool

    # Second pass: fill each stratum quota with other file IDs, even when their
    # positive thing ID has already been represented.
    for s in cfg.strata:
        current = sum(1 for row in selected if row["stratum"] == s.name)
        for row in remaining_by_stratum[s.name]:
            if current == s.count:
                break
            file_id = int(row["file_id"])
            if file_id in selected_ids:
                continue
            selected.append(row)
            selected_ids.add(file_id)
            current += 1

    if len(selected) != target_count:
        raise RuntimeError(
            "Unable to satisfy stratified sampling exactly: "
            f"target_count={target_count}, selected={len(selected)}, "
            f"valid_rows={len(valid_rows)}, stratum_candidates={candidate_counts}"
        )
    return selected


def _assert_group_aware_split(split: dict[str, list[int]], rows_by_id: dict[int, dict[str, Any]]) -> None:
    expected_names = ("train", "validation", "test")
    id_sets = {name: set(split[name]) for name in expected_names}
    if any(id_sets[a] & id_sets[b] for i, a in enumerate(expected_names) for b in expected_names[i + 1 :]):
        raise RuntimeError("Split file IDs are not mutually exclusive")

    positive_things = {
        name: {
            int(rows_by_id[file_id].get("thing_id", -1))
            for file_id in ids
            if int(rows_by_id[file_id].get("thing_id", -1)) > 0
        }
        for name, ids in id_sets.items()
    }
    if any(
        positive_things[a] & positive_things[b]
        for i, a in enumerate(expected_names)
        for b in expected_names[i + 1 :]
    ):
        raise RuntimeError("Positive thing_id leakage detected between splits")

    for name in ("validation", "test"):
        ids = [
            int(rows_by_id[file_id].get("thing_id", -1))
            for file_id in split[name]
            if int(rows_by_id[file_id].get("thing_id", -1)) > 0
        ]
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"Split {name} contains more than one file for a positive thing_id")


def _make_split(selected: list[dict[str, Any]], cfg: PrepareConfig) -> dict[str, list[int]]:
    expected_total = _expected_model_count(cfg)
    if len(selected) != expected_total:
        raise RuntimeError(f"Split input must contain exactly {expected_total} rows, found {len(selected)}")

    rng = random.Random(cfg.seed)
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in selected:
        groups.setdefault(_group_key(row), []).append(row)

    ranked_groups = list(groups.items())
    rng.shuffle(ranked_groups)
    ranked_groups.sort(key=lambda item: len(item[1]))
    held_out_count = cfg.split.val + cfg.split.test
    if len(ranked_groups) < held_out_count:
        raise RuntimeError(
            "Unable to construct group-aware split: "
            f"available_groups={len(ranked_groups)}, available_train_files=0, "
            f"required_train={cfg.split.train}, required_validation={cfg.split.val}, "
            f"required_test={cfg.split.test}"
        )

    held_out_groups = ranked_groups[:held_out_count]
    train_groups = ranked_groups[held_out_count:]
    available_train_files = sum(len(rows) for _, rows in train_groups)
    if available_train_files < cfg.split.train:
        raise RuntimeError(
            "Unable to construct exact group-aware split without thing_id leakage: "
            f"available_groups={len(ranked_groups)}, "
            f"available_train_files={available_train_files}, "
            f"required_train={cfg.split.train}, required_validation={cfg.split.val}, "
            f"required_test={cfg.split.test}"
        )

    held_out_rows = [rows[0] for _, rows in held_out_groups]
    validation = [int(row["file_id"]) for row in held_out_rows[: cfg.split.val]]
    test = [int(row["file_id"]) for row in held_out_rows[cfg.split.val :]]
    train_pool = [row for _, rows in train_groups for row in rows]
    rng.shuffle(train_pool)
    train = [int(row["file_id"]) for row in train_pool[: cfg.split.train]]
    split = {"train": train, "validation": validation, "test": test}
    _assert_group_aware_split(split, {int(row["file_id"]): row for row in selected})
    return split


def _replacement_candidate(
    failed: dict[str, Any], failed_split: str, valid_rows: list[dict[str, Any]],
    selected_by_id: dict[int, dict[str, Any]], id_to_split: dict[int, str], current_scan_index: int,
) -> dict[str, Any] | None:
    """Pick a deterministic later candidate that preserves split group isolation."""
    positive_things_by_split = {
        name: {
            int(selected_by_id[file_id].get("thing_id", -1))
            for file_id, split_name in id_to_split.items()
            if split_name == name and int(selected_by_id[file_id].get("thing_id", -1)) > 0
        }
        for name in ("train", "validation", "test")
    }

    def group_is_allowed(row: dict[str, Any]) -> bool:
        thing_id = int(row.get("thing_id", -1))
        if thing_id <= 0:
            return True
        if failed_split == "train":
            return thing_id not in positive_things_by_split["validation"] | positive_things_by_split["test"]
        other_splits = {"train", "validation", "test"} - {failed_split}
        return all(thing_id not in positive_things_by_split[name] for name in other_splits) and thing_id not in positive_things_by_split[failed_split]

    eligible = [
        row for row in valid_rows
        if int(row["file_id"]) not in selected_by_id
        and int(row.get("scan_index", -1)) > current_scan_index
        and group_is_allowed(row)
    ]
    preferred = [row for row in eligible if row["stratum"] == failed["stratum"]]
    pool = preferred
    if not pool:
        return None
    return min(pool, key=lambda row: (int(row["scan_index"]), int(row["file_id"])))


def prepare_dataset(cfg: PrepareConfig) -> None:
    expected_count = _expected_model_count(cfg)
    runtime = validate_downstream(cfg.downstream)
    from mlr.datasets import load_reconstruction_input
    from mlr.io import load_mesh
    from mlr.learned_laplacian.dataset import save_prepared_sample, validate_sample
    from mlr.synthetic import SyntheticRenderConfig

    output_root = Path(cfg.output_root)
    ensure_dir(output_root)
    logger = setup_logger(output_root / cfg.log_file)
    logger.info(
        "Starting preparation for %s models (train=%s, validation=%s, test=%s)",
        expected_count, cfg.split.train, cfg.split.val, cfg.split.test,
    )
    _write_contract_report(output_root, runtime)

    thingi10k.init(variant="npz", cache_dir=cfg.cache_dir)
    dataset_fn = thingi10k.dataset
    dataset_help = inspect.getdoc(dataset_fn) or "No help text available"
    write_json(output_root / "reports" / "dataset_api_summary.json", {"dataset_help": dataset_help})

    valid_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []

    logger.info("Scanning candidates from thingi10k.dataset()")
    dataset_entries = list(thingi10k.dataset())
    for scan_index, entry in enumerate(tqdm(dataset_entries, desc="Scanning candidates")):
        file_id = _entry_int(entry, ("file_id", "id", "model_id"))
        thing_id = _entry_int(entry, ("thing_id", "thing"))
        if file_id in cfg.known_corrupt_ids:
            rejected_rows.append({"file_id": file_id, "thing_id": thing_id, "reason": "known_corrupt_id"})
            continue
        try:
            arrays = extract_vertices_faces(entry)
            reason = _candidate_reason(arrays.vertices, arrays.faces, cfg)
            if reason:
                rejected_rows.append({"file_id": file_id, "thing_id": thing_id, "reason": reason})
                continue
            valid_rows.append(
                {
                    "file_id": file_id,
                    "thing_id": thing_id,
                    "license": _entry_str(entry, ("license", "licence")),
                    "author": _entry_str(entry, ("author", "designer")),
                    "category": _entry_str(entry, ("category",)),
                    "tags": json.dumps(entry.get("tags", [])),
                    "source_variant": "npz",
                    "orig_vertices": int(len(arrays.vertices)),
                    "orig_faces": int(len(arrays.faces)),
                    "open_closed": "closed"
                    if trimesh.Trimesh(vertices=arrays.vertices, faces=arrays.faces, process=False).is_watertight
                    else "open",
                    "stratum": _stratum_for_face_count(int(len(arrays.faces)), [asdict(s) for s in cfg.strata]),
                    "scan_index": scan_index,
                }
            )
        except Exception as exc:  # noqa: BLE001
            failed_rows.append({"file_id": file_id, "thing_id": thing_id, "reason": f"candidate_load_failure: {exc}"})

    write_csv(output_root / "selection_candidates.csv", valid_rows)
    write_csv(output_root / "rejected_models.csv", rejected_rows)
    write_csv(output_root / "failed_models.csv", failed_rows)

    selected = _sample_stratified(valid_rows, cfg)
    split = _make_split(selected, cfg)
    split_csv = [{"file_id": file_id, "split": split_name} for split_name, ids in split.items() for file_id in ids]
    write_json(output_root / "split.json", split)
    write_csv(output_root / "split.csv", split_csv)

    manifest_rows: list[dict[str, Any]] = []
    prepared_records: list[dict[str, Any]] = []
    model_root = output_root / "models"
    prepared_root = output_root / cfg.prepared_samples.directory
    ensure_dir(model_root)
    ensure_dir(prepared_root)

    id_to_split = {fid: s for s, ids in split.items() for fid in ids}
    selected_by_id = {int(row["file_id"]): row for row in selected}
    logger.info(
        "Processing %s selected models (train=%s, validation=%s, test=%s)",
        expected_count, cfg.split.train, cfg.split.val, cfg.split.test,
    )
    for scan_index, entry in enumerate(tqdm(dataset_entries, desc="Preparing models")):
        file_id = _entry_int(entry, ("file_id", "id", "model_id"))
        if file_id not in selected_by_id:
            continue

        out_dir = model_root / str(file_id)
        status_file = out_dir / "metrics.json"
        if status_file.exists() and not cfg.force:
            metrics = json.loads(status_file.read_text(encoding="utf-8"))
            if metrics.get("validation_status") == "valid":
                from .validate import _check_mesh, _load_npz_mesh

                output_issues: list[str] = []
                expected_geometry_contract = _expected_geometry_contract(cfg)
                if metrics.get("geometry_contract") != expected_geometry_contract:
                    output_issues.append("geometry_contract_changed")
                for mesh_name in ("gt_mesh.npz", "coarse_mesh.npz", "expanded_mesh.npz"):
                    mesh_v, mesh_f = _load_npz_mesh(out_dir / mesh_name)
                    output_issues.extend(_check_mesh(mesh_v, mesh_f))

                views_dataset = out_dir / "views" / "dataset.json"
                prepared_path = prepared_root / f"thingi10k_{file_id}.pt"
                if not views_dataset.exists():
                    output_issues.append("missing_views_dataset")
                else:
                    try:
                        reconstruction = load_reconstruction_input(views_dataset)
                        render_metadata = json.loads(views_dataset.read_text(encoding="utf-8")).get("config", {})
                        if len(reconstruction.image_paths) != cfg.views_count:
                            output_issues.append("unexpected_view_count")
                        gt_mesh = load_mesh(out_dir / "gt_mesh.obj")
                        expected_mesh_checksum = _mesh_checksum(gt_mesh.vertices, gt_mesh.faces)
                        output_issues.extend(
                            _render_cache_issues(render_metadata, cfg, expected_mesh_checksum)
                        )
                    except Exception as exc:  # noqa: BLE001
                        output_issues.append(f"invalid_views_dataset:{exc}")
                if not prepared_path.exists():
                    output_issues.append("missing_prepared_sample")
                elif metrics.get("manifest_row", {}).get("prepared_storage_format") != cfg.prepared_samples.storage_format:
                    output_issues.append("prepared_storage_format_changed")
                else:
                    try:
                        raw_prepared = torch.load(
                            prepared_path, map_location="cpu", weights_only=False
                        )
                        output_issues.extend(_prepared_cache_issues(raw_prepared, cfg))
                    except Exception as exc:  # noqa: BLE001
                        output_issues.append(f"invalid_prepared_sample:{exc}")
                if not output_issues:
                    logger.info("Skipping already valid model %s", file_id)
                    cached_manifest_row = dict(metrics["manifest_row"])
                    cached_manifest_row["split"] = id_to_split[file_id]
                    manifest_rows.append(cached_manifest_row)
                    prepared_records.append(
                        {"path": str(prepared_path.relative_to(output_root)), "split": id_to_split[file_id],
                         "sample_id": f"thingi10k_{file_id}"}
                    )
                    continue
                logger.info("Regenerating invalid cached model %s: %s", file_id, sorted(set(output_issues)))

        try:
            ensure_dir(out_dir)
            arrays = extract_vertices_faces(entry)
            source_v = np.asarray(arrays.vertices, dtype=np.float32)
            source_f = np.asarray(arrays.faces, dtype=np.int64)

            save_mesh_npz(out_dir / "source_mesh.npz", source_v, source_f)
            _save_obj(out_dir / "source_mesh.obj", source_v, source_f)
            write_json(out_dir / "source_metadata.json", entry)

            clean, cleanup_ops = mesh_cleanup(source_v, source_f)
            cleanup_reason = _post_cleanup_reason(clean.vertices, clean.faces, cfg)
            if cleanup_reason is not None:
                raise RuntimeError(cleanup_reason)
            gt_norm_v, norm = normalize_vertices(
                clean.vertices,
                mode=cfg.normalization_mode,
                epsilon=cfg.normalization_epsilon,
            )
            gt_f = clean.faces.astype(np.int64)

            save_mesh_npz(out_dir / "gt_mesh.npz", gt_norm_v, gt_f)
            _save_obj(out_dir / "gt_mesh.obj", gt_norm_v, gt_f)
            write_json(out_dir / "normalization.json", norm)

            coarse = simplify_mesh(
                gt_norm_v,
                gt_f,
                target_vertices=cfg.coarse_target_vertices,
                min_vertices=cfg.coarse_min_vertices,
            )
            save_mesh_npz(out_dir / "coarse_mesh.npz", coarse.vertices, coarse.faces)
            _save_obj(out_dir / "coarse_mesh.obj", coarse.vertices, coarse.faces)

            expanded, mapping = midpoint_subdivide(coarse.vertices, coarse.faces, steps=cfg.subdivision_steps)
            save_mesh_npz(out_dir / "expanded_mesh.npz", expanded.vertices, expanded.faces)
            _save_obj(out_dir / "expanded_mesh.obj", expanded.vertices, expanded.faces)
            np.savez_compressed(out_dir / "subdivision_mapping.npz", **mapping)

            targets = compute_surface_targets(expanded.vertices, gt_norm_v, gt_f)
            np.savez_compressed(out_dir / "targets.npz", **targets)

            lap = build_uniform_laplacian(len(expanded.vertices), expanded.faces)
            sp.save_npz(out_dir / "laplacian.npz", lap)
            delta_target = lap @ targets["target_positions"]
            np.savez_compressed(out_dir / "laplacian_targets.npz", laplacian_target=delta_target.astype(np.float32))

            _preview(out_dir / "preview.png", gt_norm_v, coarse.vertices, expanded.vertices)
            views_dir = out_dir / "views"
            ensure_dir(views_dir)
            render_cfg = SyntheticRenderConfig(
                num_views=cfg.views_count,
                width=cfg.views_width,
                height=cfg.views_height,
                trajectory=cfg.views_trajectory,
                min_elevation_degrees=-60.0,
                max_elevation_degrees=60.0,
                fov_degrees=cfg.views_fov_degrees,
                render_mode=cfg.views_render_mode,
                backend=cfg.views_backend,
                normalize_mesh=False,
                opengl_context_backend=cfg.views_opengl_context_backend,
                cube_half_extent=cfg.views_cube_half_extent,
                antialiasing=cfg.views_antialiasing,
                camera_layout_version=VIEW_LAYOUT_VERSION,
            )
            rendered = generate_configured_synthetic_dataset(
                mesh=load_mesh(out_dir / "gt_mesh.obj"),
                out_dir=views_dir,
                config=render_cfg,
                source_mesh_path=out_dir / "gt_mesh.obj",
            )
            mesh = load_mesh(out_dir / "gt_mesh.obj")
            rendered_mesh = load_mesh(views_dir / "mesh.obj")
            if not np.allclose(rendered_mesh.vertices, mesh.vertices, atol=1e-5):
                raise RuntimeError("Rendered mesh does not match prepared normalized GT mesh")
            dataset_json = views_dir / "dataset.json"
            dataset = load_reconstruction_input(dataset_json)
            render_metadata = json.loads(dataset_json.read_text(encoding="utf-8")).get("config", {})
            if len(dataset.image_paths) != cfg.views_count:
                raise RuntimeError("Unexpected number of rendered views")

            for rel_path in ["images", "masks", "depth"]:
                ensure_dir(views_dir / rel_path)

            write_json(out_dir / "views_dataset.json", {"path": str(dataset_json), "views": len(dataset.image_paths)})

            prepared = _build_prepared_sample(
                dataset_json, out_dir / "expanded_mesh.obj", out_dir / "gt_mesh.obj",
                targets, lap, cfg.prepared_samples.image_size, f"thingi10k_{file_id}",
                {
                    "dataset_path": str(dataset_json),
                    "coarse_mesh_path": str(out_dir / "expanded_mesh.obj"),
                    "gt_mesh_path": str(out_dir / "gt_mesh.obj"),
                    "operator_type": "uniform",
                    "target_constructor": "precomputed_closest_surface_on_prediction_graph",
                    "laplacian_target_mode": cfg.prepared_samples.target_mode,
                    "edge_scale_epsilon": cfg.prepared_samples.edge_scale_epsilon,
                    "file_id": int(file_id),
                    "split": id_to_split[file_id],
                    "normalization": norm,
                    "normalization_mode": cfg.normalization_mode,
                    "normalization_center": norm["normalization_center"],
                    "normalization_scale": norm["normalization_scale"],
                    "view_count": cfg.views_count,
                    "view_trajectory": cfg.views_trajectory,
                    "view_backend": cfg.views_backend,
                    "opengl_context_backend": cfg.views_opengl_context_backend,
                    "cube_half_extent": cfg.views_cube_half_extent,
                    "fov_degrees": cfg.views_fov_degrees,
                    "view_width": cfg.views_width,
                    "view_height": cfg.views_height,
                    "view_layout_version": VIEW_LAYOUT_VERSION,
                    "view_names": list(CUBE_SURFACE_VIEW_NAMES),
                    "prepared_storage_format": cfg.prepared_samples.storage_format,
                    "downstream_url": runtime.url,
                    "downstream_branch": runtime.branch,
                    "downstream_sha": runtime.sha,
                },
                dataset_root=output_root,
            )
            prepared_path = prepared_root / f"thingi10k_{file_id}.pt"
            save_prepared_sample(prepared, prepared_path)
            raw_target = prepared["raw_laplacian_target"].detach().cpu().numpy()
            np.savez_compressed(out_dir / "laplacian_targets.npz", laplacian_target=raw_target.astype(np.float32))
            prepared_records.append(
                {"path": str(prepared_path.relative_to(output_root)), "split": id_to_split[file_id],
                 "sample_id": prepared["sample_id"]}
            )

            dist = targets["surface_distance"]
            manifest_row = {
                "file_id": file_id,
                "thing_id": selected_by_id[file_id].get("thing_id", -1),
                "author": selected_by_id[file_id].get("author", ""),
                "license": selected_by_id[file_id].get("license", ""),
                "category": selected_by_id[file_id].get("category", ""),
                "tags": selected_by_id[file_id].get("tags", "[]"),
                "source_variant": "npz",
                "split": id_to_split[file_id],
                "open_closed": selected_by_id[file_id].get("open_closed", "unknown"),
                "original_vertices": int(len(source_v)),
                "original_faces": int(len(source_f)),
                "cleaned_vertices": int(len(gt_norm_v)),
                "cleaned_faces": int(len(gt_f)),
                "coarse_vertices": int(len(coarse.vertices)),
                "coarse_faces": int(len(coarse.faces)),
                "expanded_vertices": int(len(expanded.vertices)),
                "expanded_faces": int(len(expanded.faces)),
                "normalization_scale": float(norm["normalization_scale"]),
                "normalization_mode": cfg.normalization_mode,
                "cleanup_operations": "|".join(cleanup_ops),
                "subdivision_steps": cfg.subdivision_steps,
                "laplacian_type": "uniform",
                "views_count": cfg.views_count,
                "views_resolution": f"{cfg.views_width}x{cfg.views_height}",
                "views_trajectory": cfg.views_trajectory,
                "views_backend": render_metadata.get("backend", cfg.views_backend),
                "cube_half_extent": cfg.views_cube_half_extent,
                "fov_degrees": cfg.views_fov_degrees,
                "view_layout_version": VIEW_LAYOUT_VERSION,
                "prepared_image_size": cfg.prepared_samples.image_size,
                "prepared_storage_format": cfg.prepared_samples.storage_format,
                "distance_mean": float(np.mean(dist)),
                "distance_median": float(np.median(dist)),
                "distance_p95": float(np.quantile(dist, 0.95)),
                "distance_max": float(np.max(dist)),
                "validation_status": "valid",
                "failure_reason": "",
                "random_seed": cfg.seed,
                "script_checksum": _self_checksum(),
            }
            metrics = {
                "source_metrics": _mesh_metrics(source_v, source_f),
                "gt_metrics": _mesh_metrics(gt_norm_v, gt_f),
                "coarse_metrics": _mesh_metrics(coarse.vertices, coarse.faces),
                "expanded_metrics": _mesh_metrics(expanded.vertices, expanded.faces),
                "geometry_contract": _expected_geometry_contract(cfg),
                "validation_status": "valid",
                "manifest_row": manifest_row,
            }
            write_json(out_dir / "metrics.json", metrics)
            manifest_rows.append(manifest_row)
        except Exception as exc:  # noqa: BLE001
            failed_rows.append({"file_id": file_id, "reason": f"processing_failure: {exc}"})
            prepared_records = [
                record for record in prepared_records
                if record.get("sample_id") != f"thingi10k_{file_id}"
            ]
            manifest_rows = [row for row in manifest_rows if int(row["file_id"]) != file_id]
            failed_row = selected_by_id[file_id]
            failed_split = id_to_split.pop(file_id)
            split[failed_split].remove(file_id)
            selected_by_id.pop(file_id)
            replacement = _replacement_candidate(
                failed_row, failed_split, valid_rows, selected_by_id, id_to_split, scan_index
            )
            if replacement is not None:
                replacement_id = int(replacement["file_id"])
                split[failed_split].append(replacement_id)
                id_to_split[replacement_id] = failed_split
                selected_by_id[replacement_id] = replacement
                _assert_group_aware_split(split, selected_by_id)
                actual_counts = {name: len(ids) for name, ids in split.items()}
                required_counts = {
                    "train": cfg.split.train,
                    "validation": cfg.split.val,
                    "test": cfg.split.test,
                }
                if actual_counts != required_counts:
                    raise RuntimeError(
                        "Replacement changed exact split counts: "
                        f"required={required_counts}, actual={actual_counts}"
                    )
                logger.warning(
                    "Replacing failed model %s with %s in split %s", file_id, replacement_id, failed_split
                )
            else:
                logger.error("No eligible later replacement for failed model %s", file_id)
            write_csv(output_root / "failed_models.csv", failed_rows)

    if len(manifest_rows) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} prepared models, "
            f"got {len(manifest_rows)}"
        )
    if len(prepared_records) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} prepared manifest records, got {len(prepared_records)}"
        )

    required_split_counts = {
        "train": cfg.split.train,
        "validation": cfg.split.val,
        "test": cfg.split.test,
    }
    actual_split_counts = {name: len(ids) for name, ids in split.items()}
    if actual_split_counts != required_split_counts:
        raise RuntimeError(
            "Final split counts do not match config: "
            f"required={required_split_counts}, actual={actual_split_counts}"
        )
    _assert_group_aware_split(split, selected_by_id)

    write_csv(output_root / "manifest.csv", manifest_rows)
    write_json(output_root / "manifest.json", manifest_rows)
    write_json(
        output_root / cfg.prepared_samples.manifest,
        {"samples": prepared_records, "downstream": {**asdict(runtime), "root": str(runtime.root)}},
    )
    write_csv(output_root / "failed_models.csv", failed_rows)
    split_csv = [{"file_id": file_id, "split": name} for name, ids in split.items() for file_id in ids]
    write_json(output_root / "split.json", split)
    write_csv(output_root / "split.csv", split_csv)
    write_json(output_root / "config.yaml.resolved.json", asdict(cfg))
    _write_preparation_report(output_root, manifest_rows, valid_rows, rejected_rows, failed_rows, cfg)


def _self_checksum() -> str:
    source = inspect.getsource(prepare_dataset)
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _write_preparation_report(
    output_root: Path,
    manifest_rows: list[dict[str, Any]],
    valid_rows: list[dict[str, Any]],
    rejected_rows: list[dict[str, Any]],
    failed_rows: list[dict[str, Any]],
    cfg: PrepareConfig,
) -> None:
    distances = np.asarray([row["distance_mean"] for row in manifest_rows], dtype=np.float64)
    report = (
        "# Preparation report\n\n"
        f"- Seed: `{cfg.seed}`\n"
        f"- Cache dir: `{cfg.cache_dir}`\n"
        f"- Output root: `{cfg.output_root}`\n"
        f"- Candidates accepted: `{len(valid_rows)}`\n"
        f"- Candidates rejected: `{len(rejected_rows)}`\n"
        f"- Model failures: `{len(failed_rows)}`\n"
        f"- Final models: `{len(manifest_rows)}`\n"
        f"- Configured models: `{_expected_model_count(cfg)}`\n"
        f"- Split counts: `train={cfg.split.train}, validation={cfg.split.val}, test={cfg.split.test}`\n"
        f"- Projection distance mean over models: `{float(np.mean(distances)):.6f}`\n"
        f"- Projection distance worst mean: `{float(np.max(distances)):.6f}`\n"
    )
    report_path = output_root / "reports" / "preparation_report.md"
    ensure_dir(report_path.parent)
    report_path.write_text(report, encoding="utf-8")
