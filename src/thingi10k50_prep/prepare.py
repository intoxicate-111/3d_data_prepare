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


def _build_prepared_sample(
    dataset_json: Path,
    expanded_obj: Path,
    gt_obj: Path,
    targets: dict[str, np.ndarray],
    laplacian: sp.csr_matrix,
    image_size: int,
    sample_id: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Assemble existing projections through the downstream loader/scaling contract."""
    from mlr.datasets import load_masks, load_reconstruction_input
    from mlr.io import load_mesh
    from mlr.learned_laplacian.dataset import validate_sample
    from mlr.learned_laplacian.sample_io import _load_images, _mask_visibility

    reconstruction = load_reconstruction_input(dataset_json)
    prediction_mesh = load_mesh(expanded_obj).ensure_normals()
    gt_mesh = load_mesh(gt_obj).ensure_normals()
    images, scale_xy = _load_images(reconstruction.image_paths, image_size)
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
        visibility = _mask_visibility(prediction_mesh.vertices, reconstruction.cameras, masks, scale_xy)
    raw_target = laplacian @ targets["target_positions"]
    sample = {
        "sample_id": sample_id,
        "images": images,
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
        "- Normalization convention: AABB-center + longest-side-to-2.0, with renderer `normalize_mesh=False` for already normalized meshes\n"
        "- Coarse mesh target: 3500 vertices\n"
        "- Midpoint subdivision steps: 1\n"
        "- Laplacian type: downstream uniform operator `I - D^-1 A`\n"
        "- Target quantity: `delta_target = L_exp @ target_positions`\n"
        "- Multi-view inputs: downstream-compatible synthetic renderer from `mlr.synthetic`\n"
        "- Train/validation/test expectations: prepared manifest with 40/5/5 disjoint samples\n"
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
    repeated_idx = (faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2]) | (faces[:, 0] == faces[:, 2])
    if np.any(repeated_idx):
        return "repeated_indices_in_face"
    tri = vertices[faces]
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) * 0.5
    if np.any(area <= 0):
        return "non_positive_triangle_area"
    return None


def _stratum_for_face_count(face_count: int, strata: list[dict[str, Any]]) -> str:
    for s in strata:
        if s["min_faces"] <= face_count <= s["max_faces"]:
            return str(s["name"])
    return "out_of_range"


def _sample_stratified(valid_rows: list[dict[str, Any]], cfg: PrepareConfig) -> list[dict[str, Any]]:
    random.seed(cfg.seed)
    by_stratum: dict[str, list[dict[str, Any]]] = {s.name: [] for s in cfg.strata}
    for row in valid_rows:
        by_stratum[row["stratum"]].append(row)

    selected: list[dict[str, Any]] = []
    used_things: set[int] = set()
    for s in cfg.strata:
        pool = by_stratum[s.name].copy()
        random.shuffle(pool)
        pick: list[dict[str, Any]] = []
        for row in pool:
            thing_id = int(row.get("thing_id", -1))
            if thing_id > 0 and thing_id in used_things:
                continue
            pick.append(row)
            if thing_id > 0:
                used_things.add(thing_id)
            if len(pick) == s.count:
                break
        selected.extend(pick)

    if len(selected) < 50:
        remaining = [r for r in valid_rows if r["file_id"] not in {x["file_id"] for x in selected}]
        random.shuffle(remaining)
        for row in remaining:
            thing_id = int(row.get("thing_id", -1))
            if thing_id > 0 and thing_id in used_things:
                continue
            selected.append(row)
            if thing_id > 0:
                used_things.add(thing_id)
            if len(selected) == 50:
                break

    if len(selected) != 50:
        raise RuntimeError(f"Unable to sample 50 models; got {len(selected)} valid selections")
    return selected


def _make_split(selected: list[dict[str, Any]], cfg: PrepareConfig) -> dict[str, list[int]]:
    random.seed(cfg.seed)
    shuffled = selected.copy()
    random.shuffle(shuffled)
    train = [int(x["file_id"]) for x in shuffled[: cfg.split.train]]
    val = [int(x["file_id"]) for x in shuffled[cfg.split.train : cfg.split.train + cfg.split.val]]
    test = [int(x["file_id"]) for x in shuffled[cfg.split.train + cfg.split.val :]]
    return {"train": train, "validation": val, "test": test}


def _replacement_candidate(
    failed: dict[str, Any], valid_rows: list[dict[str, Any]], selected_ids: set[int],
    used_thing_ids: set[int], current_scan_index: int,
) -> dict[str, Any] | None:
    """Pick a deterministic later candidate, preferring the failed slot's stratum."""
    eligible = [
        row for row in valid_rows
        if int(row["file_id"]) not in selected_ids
        and int(row.get("scan_index", -1)) > current_scan_index
        and (int(row.get("thing_id", -1)) <= 0 or int(row["thing_id"]) not in used_thing_ids)
    ]
    preferred = [row for row in eligible if row["stratum"] == failed["stratum"]]
    pool = preferred or eligible
    if not pool:
        return None
    return min(pool, key=lambda row: (int(row["scan_index"]), int(row["file_id"])))


def prepare_dataset(cfg: PrepareConfig) -> None:
    runtime = validate_downstream(cfg.downstream)
    from mlr.datasets import load_reconstruction_input
    from mlr.io import load_mesh
    from mlr.learned_laplacian.dataset import save_prepared_sample, validate_sample
    from mlr.synthetic import SyntheticRenderConfig, generate_synthetic_dataset

    output_root = Path(cfg.output_root)
    ensure_dir(output_root)
    logger = setup_logger(output_root / cfg.log_file)
    logger.info("Starting Thingi10K50 preparation")
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
    logger.info("Processing selected models")
    used_thing_ids = {int(row["thing_id"]) for row in selected if int(row.get("thing_id", -1)) > 0}
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
                for mesh_name in ("gt_mesh.npz", "coarse_mesh.npz", "expanded_mesh.npz"):
                    mesh_v, mesh_f = _load_npz_mesh(out_dir / mesh_name)
                    output_issues.extend(_check_mesh(mesh_v, mesh_f))

                views_dataset = out_dir / "views" / "dataset.json"
                prepared_path = prepared_root / f"thingi10k_{file_id}.pt"
                if not views_dataset.exists():
                    output_issues.append("missing_views_dataset")
                if not prepared_path.exists():
                    output_issues.append("missing_prepared_sample")
                if not output_issues:
                    logger.info("Skipping already valid model %s", file_id)
                    manifest_rows.append(metrics["manifest_row"])
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
            gt_norm_v, norm = normalize_vertices(clean.vertices)
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
                trajectory="sphere",
                min_elevation_degrees=-60.0,
                max_elevation_degrees=60.0,
                render_mode="lit",
                backend="cpu",
                normalize_mesh=False,
            )
            rendered = generate_synthetic_dataset(
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
                    "downstream_url": runtime.url,
                    "downstream_branch": runtime.branch,
                    "downstream_sha": runtime.sha,
                },
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
                "normalization_scale": float(norm["uniform_scale"]),
                "cleanup_operations": "|".join(cleanup_ops),
                "subdivision_steps": cfg.subdivision_steps,
                "laplacian_type": "uniform",
                "views_count": cfg.views_count,
                "views_resolution": f"{cfg.views_width}x{cfg.views_height}",
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
                "validation_status": "valid",
                "manifest_row": manifest_row,
            }
            write_json(out_dir / "metrics.json", metrics)
            manifest_rows.append(manifest_row)
        except Exception as exc:  # noqa: BLE001
            failed_rows.append({"file_id": file_id, "reason": f"processing_failure: {exc}"})
            failed_row = selected_by_id[file_id]
            replacement = _replacement_candidate(
                failed_row, valid_rows, set(selected_by_id), used_thing_ids, scan_index
            )
            if replacement is not None:
                failed_split = id_to_split.pop(file_id)
                split[failed_split].remove(file_id)
                replacement_id = int(replacement["file_id"])
                split[failed_split].append(replacement_id)
                id_to_split[replacement_id] = failed_split
                selected_by_id[replacement_id] = replacement
                replacement_thing = int(replacement.get("thing_id", -1))
                if replacement_thing > 0:
                    used_thing_ids.add(replacement_thing)
                logger.warning(
                    "Replacing failed model %s with %s in split %s", file_id, replacement_id, failed_split
                )
            else:
                logger.error("No eligible later replacement for failed model %s", file_id)
            write_csv(output_root / "failed_models.csv", failed_rows)

    if len(manifest_rows) != 50:
        raise RuntimeError(f"Expected 50 prepared models, got {len(manifest_rows)}")

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
        f"- Projection distance mean over models: `{float(np.mean(distances)):.6f}`\n"
        f"- Projection distance worst mean: `{float(np.max(distances)):.6f}`\n"
    )
    report_path = output_root / "reports" / "preparation_report.md"
    ensure_dir(report_path.parent)
    report_path.write_text(report, encoding="utf-8")
