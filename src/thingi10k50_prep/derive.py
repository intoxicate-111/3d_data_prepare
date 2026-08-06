from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
from PIL import Image
from tqdm import tqdm

from .io_utils import ensure_dir, write_csv, write_json
from .prepare import _build_prepared_sample


DERIVATION_VERSION = "resize_prepared_dataset_v1"


def _scale_camera(
    camera: dict[str, Any],
    scale_x: float,
    scale_y: float,
    width: int,
    height: int,
) -> dict[str, Any]:
    result = copy.deepcopy(camera)
    intrinsics = np.asarray(result["intrinsics"], dtype=np.float64)
    if intrinsics.shape != (3, 3):
        raise ValueError(f"Camera intrinsics must have shape (3, 3), got {intrinsics.shape}")
    intrinsics[0, :] *= scale_x
    intrinsics[1, :] *= scale_y
    result["intrinsics"] = intrinsics.tolist()
    result["image_size"] = [width, height]
    return result


def _resize_depth(source: Path, destination: Path, width: int, height: int) -> None:
    depth = np.load(source)
    if depth.ndim != 2:
        raise ValueError(f"Depth image must be two-dimensional: {source}")
    original_dtype = depth.dtype
    resized = Image.fromarray(depth.astype(np.float32, copy=False), mode="F").resize(
        (width, height), resample=Image.Resampling.NEAREST
    )
    ensure_dir(destination.parent)
    np.save(destination, np.asarray(resized, dtype=original_dtype))


def _resize_png(
    source: Path,
    destination: Path,
    width: int,
    height: int,
    resample: Image.Resampling,
) -> None:
    ensure_dir(destination.parent)
    with Image.open(source) as image:
        image.resize((width, height), resample=resample).save(destination)


def _copy_model_geometry(source: Path, destination: Path) -> None:
    ensure_dir(destination)
    for item in source.iterdir():
        if item.name in {"views", "metrics.json", "views_dataset.json"}:
            continue
        target = destination / item.name
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def _rewrite_views(
    source_model: Path,
    destination_model: Path,
    width: int,
    height: int,
) -> Path:
    source_views = source_model / "views"
    destination_views = destination_model / "views"
    dataset_path = source_views / "dataset.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    cameras = dataset.get("cameras")
    if not isinstance(cameras, list) or not cameras:
        raise ValueError(f"No cameras found in {dataset_path}")

    first_image_path = source_views / dataset["image_paths"][0]
    with Image.open(first_image_path) as image:
        source_width, source_height = image.size
    if source_width <= 0 or source_height <= 0:
        raise ValueError(f"Invalid source image size in {first_image_path}")
    scale_x = width / source_width
    scale_y = height / source_height

    image_paths = dataset.get("image_paths", [])
    mask_paths = dataset.get("mask_paths", [])
    depth_paths = dataset.get("depth_paths", [])
    if not (len(image_paths) == len(mask_paths) == len(depth_paths) == len(cameras)):
        raise ValueError(f"View path and camera counts disagree in {dataset_path}")

    for image_rel, mask_rel, depth_rel in zip(
        image_paths, mask_paths, depth_paths, strict=True
    ):
        _resize_png(
            source_views / image_rel,
            destination_views / image_rel,
            width,
            height,
            Image.Resampling.LANCZOS,
        )
        _resize_png(
            source_views / mask_rel,
            destination_views / mask_rel,
            width,
            height,
            Image.Resampling.NEAREST,
        )
        _resize_depth(
            source_views / depth_rel,
            destination_views / depth_rel,
            width,
            height,
        )

    scaled_cameras = [
        _scale_camera(camera, scale_x, scale_y, width, height) for camera in cameras
    ]
    dataset["cameras"] = scaled_cameras
    dataset["source_mesh_path"] = str((destination_model / "gt_mesh.obj").resolve())
    dataset.setdefault("config", {})["width"] = width
    dataset["config"]["height"] = height
    dataset["config"]["derived_from_resolution"] = [source_width, source_height]
    dataset["config"]["derivation_version"] = DERIVATION_VERSION

    ensure_dir(destination_views)
    shutil.copy2(source_views / "mesh.obj", destination_views / "mesh.obj")
    write_json(destination_views / "cameras.json", scaled_cameras)
    destination_dataset = destination_views / "dataset.json"
    write_json(destination_dataset, dataset)
    write_json(
        destination_model / "views_dataset.json",
        {"path": str(destination_dataset.resolve()), "views": len(scaled_cameras)},
    )
    return destination_dataset


def _updated_metadata(
    metadata: dict[str, Any],
    source_root: Path,
    destination_root: Path,
    destination_model: Path,
    width: int,
    height: int,
) -> dict[str, Any]:
    result = copy.deepcopy(metadata)
    for key, filename in (
        ("dataset_path", "views/dataset.json"),
        ("coarse_mesh_path", "expanded_mesh.obj"),
        ("gt_mesh_path", "gt_mesh.obj"),
    ):
        result[key] = str((destination_model / filename).resolve())
    result["view_width"] = width
    result["view_height"] = height
    result["derived_from_dataset"] = str(source_root.resolve())
    result["derivation_version"] = DERIVATION_VERSION
    result["derivation_method"] = "rgb_lanczos_mask_depth_nearest"
    result["destination_dataset"] = str(destination_root.resolve())
    return result


def derive_resized_dataset(
    source_root: str | Path,
    destination_root: str | Path,
    width: int,
    height: int,
) -> None:
    source_root = Path(source_root).resolve()
    destination_root = Path(destination_root).resolve()
    if width <= 0 or height <= 0:
        raise ValueError("Target width and height must be positive")
    if width != height:
        raise ValueError("Prepared-sample derivation currently requires a square target resolution")
    if source_root == destination_root:
        raise ValueError("Source and destination roots must differ")
    if not (source_root / "prepared_manifest.json").is_file():
        raise FileNotFoundError(f"Missing source prepared manifest: {source_root}")
    if destination_root.exists() and any(destination_root.iterdir()):
        raise FileExistsError(f"Destination is not empty: {destination_root}")
    ensure_dir(destination_root)

    source_rows = json.loads((source_root / "manifest.json").read_text(encoding="utf-8"))
    prepared_manifest = json.loads(
        (source_root / "prepared_manifest.json").read_text(encoding="utf-8")
    )
    if len(source_rows) != len(prepared_manifest.get("samples", [])):
        raise ValueError("Source manifest and prepared manifest counts disagree")

    for name in (
        "selection_candidates.csv",
        "rejected_models.csv",
        "failed_models.csv",
        "split.json",
        "split.csv",
    ):
        source = source_root / name
        if source.is_file():
            shutil.copy2(source, destination_root / name)

    destination_rows: list[dict[str, Any]] = []
    prepared_records_by_id = {
        str(record["sample_id"]): record for record in prepared_manifest["samples"]
    }
    for source_row in tqdm(source_rows, desc=f"Deriving {width}x{height} samples"):
        file_id = int(source_row["file_id"])
        sample_id = f"thingi10k_{file_id}"
        if sample_id not in prepared_records_by_id:
            raise ValueError(f"Missing prepared-manifest record for {sample_id}")
        source_model = source_root / "models" / str(file_id)
        destination_model = destination_root / "models" / str(file_id)
        _copy_model_geometry(source_model, destination_model)
        destination_dataset = _rewrite_views(
            source_model, destination_model, width, height
        )

        with np.load(destination_model / "targets.npz") as archive:
            targets = {name: archive[name] for name in archive.files}
        laplacian = sp.load_npz(destination_model / "laplacian.npz")
        source_sample_path = source_root / "prepared" / f"{sample_id}.pt"
        raw_sample = torch.load(source_sample_path, map_location="cpu", weights_only=False)
        metadata = _updated_metadata(
            dict(raw_sample.get("metadata", {})),
            source_root,
            destination_root,
            destination_model,
            width,
            height,
        )
        sample = _build_prepared_sample(
            destination_dataset,
            destination_model / "expanded_mesh.obj",
            destination_model / "gt_mesh.obj",
            targets,
            laplacian,
            width,
            sample_id,
            metadata,
            destination_root,
        )
        from mlr.learned_laplacian.dataset import save_prepared_sample

        save_prepared_sample(
            sample, destination_root / "prepared" / f"{sample_id}.pt"
        )

        destination_row = dict(source_row)
        destination_row["views_resolution"] = f"{width}x{height}"
        destination_row["prepared_image_size"] = width
        destination_row["derivation_version"] = DERIVATION_VERSION
        destination_row["derived_from_resolution"] = (
            f"{int(raw_sample['source_image_size'][0])}x"
            f"{int(raw_sample['source_image_size'][1])}"
        )
        destination_rows.append(destination_row)

        metrics = json.loads((source_model / "metrics.json").read_text(encoding="utf-8"))
        metrics["manifest_row"] = destination_row
        metrics["derivation"] = {
            "version": DERIVATION_VERSION,
            "source_dataset": str(source_root),
            "source_resolution": [
                int(raw_sample["source_image_size"][0]),
                int(raw_sample["source_image_size"][1]),
            ],
            "target_resolution": [width, height],
            "method": "rgb_lanczos_mask_depth_nearest",
        }
        write_json(destination_model / "metrics.json", metrics)

    write_csv(destination_root / "manifest.csv", destination_rows)
    write_json(destination_root / "manifest.json", destination_rows)
    write_json(destination_root / "prepared_manifest.json", prepared_manifest)

    resolved_path = source_root / "config.yaml.resolved.json"
    if resolved_path.is_file():
        resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
        resolved["output_root"] = str(destination_root)
        resolved["views_width"] = width
        resolved["views_height"] = height
        resolved.setdefault("prepared_samples", {})["image_size"] = width
        write_json(destination_root / "config.yaml.resolved.json", resolved)

    write_json(
        destination_root / "reports" / "derivation_report.json",
        {
            "version": DERIVATION_VERSION,
            "source_dataset": str(source_root),
            "destination_dataset": str(destination_root),
            "target_resolution": [width, height],
            "samples": len(destination_rows),
            "rgb_resampling": "lanczos",
            "mask_resampling": "nearest",
            "depth_resampling": "nearest",
            "camera_intrinsics_scaled": True,
            "prepared_visibility_recomputed": True,
        },
    )
