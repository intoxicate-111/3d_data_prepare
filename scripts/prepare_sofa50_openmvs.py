from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import sqlite3
import struct
import subprocess
import sys
from pathlib import Path
from typing import Sequence


SCRIPT_VERSION = "v6"
SIFT_PEAK_THRESHOLD = 0.002
DEFAULT_TEST_MODEL = "00859a4c-8945-4665-92e5-69ab8fba6593"


def expand(path: Path | str) -> Path:
    return Path(path).expanduser().resolve()


def run(
    command: Sequence[str],
    *,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> None:
    print("+", shlex.join([str(x) for x in command]), flush=True)
    subprocess.run(
        [str(x) for x in command],
        check=True,
        env=env,
        cwd=str(cwd) if cwd is not None else None,
    )


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def require_executable(path: Path, label: str) -> None:
    require_file(path, label)
    if not os.access(path, os.X_OK):
        raise PermissionError(f"{label} is not executable: {path}")


def with_prepend_path(env: dict[str, str], key: str, value: Path) -> dict[str, str]:
    result = dict(env)
    old = result.get(key, "")
    result[key] = str(value) if not old else f"{value}{os.pathsep}{old}"
    return result


def count_text_points3d(points_path: Path) -> int:
    count = 0
    with points_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#"):
                count += 1
    return count


def read_database_diagnostics(database_path: Path) -> dict[str, int | None]:
    result: dict[str, int | None] = {
        "database_images": None,
        "total_keypoints": None,
        "raw_matched_pairs": None,
        "total_raw_matches": None,
        "verified_matched_pairs": None,
        "total_inlier_matches": None,
    }
    if not database_path.exists():
        return result

    with sqlite3.connect(database_path) as connection:
        cursor = connection.cursor()
        queries = {
            "database_images": "SELECT COUNT(*) FROM images",
            "total_keypoints": "SELECT COALESCE(SUM(rows), 0) FROM keypoints",
            "raw_matched_pairs": "SELECT COUNT(*) FROM matches WHERE rows > 0",
            "total_raw_matches": "SELECT COALESCE(SUM(rows), 0) FROM matches",
            "verified_matched_pairs": "SELECT COUNT(*) FROM two_view_geometries WHERE rows > 0",
            "total_inlier_matches": "SELECT COALESCE(SUM(rows), 0) FROM two_view_geometries",
        }
        for key, query in queries.items():
            try:
                value = cursor.execute(query).fetchone()
                result[key] = int(value[0]) if value is not None else None
            except sqlite3.DatabaseError:
                result[key] = None
    return result


def _read_known_camera_text_model(
    sparse_dir: Path,
) -> tuple[dict[int, tuple[int, int, tuple[float, float, float, float]]], dict[str, int]]:
    cameras_path = sparse_dir / "cameras.txt"
    images_path = sparse_dir / "images.txt"
    require_file(cameras_path, "known-camera cameras.txt")
    require_file(images_path, "known-camera images.txt")

    cameras: dict[int, tuple[int, int, tuple[float, float, float, float]]] = {}
    for raw in cameras_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 8:
            raise ValueError(f"Expected PINHOLE camera row with 8 fields, got: {line}")
        camera_id = int(fields[0])
        model = fields[1]
        if model != "PINHOLE":
            raise ValueError(f"Expected PINHOLE known cameras, got {model} for camera {camera_id}")
        width = int(fields[2])
        height = int(fields[3])
        fx, fy, cx, cy = map(float, fields[4:8])
        cameras[camera_id] = (width, height, (fx, fy, cx, cy))

    image_to_camera: dict[str, int] = {}
    for raw in images_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        # Known-pose seed has only the IMAGE_ID/Q/T/CAMERA_ID/NAME line; the
        # following POINTS2D line is intentionally blank.
        if len(fields) < 10:
            continue
        camera_id = int(fields[8])
        image_name = fields[9]
        image_to_camera[image_name] = camera_id

    if not cameras or not image_to_camera:
        raise RuntimeError(f"Failed to parse known-camera model under {sparse_dir}")
    return cameras, image_to_camera


def _read_known_pose_rows(
    sparse_dir: Path,
) -> dict[str, tuple[int, int, tuple[float, float, float, float, float, float, float]]]:
    """Read renderer-known camera poses keyed by image filename.

    The input may be either a legacy 3-file COLMAP text model or a newer
    5-file model.  We only need the IMAGE_ID/Q/T/CAMERA_ID/NAME row from
    images.txt; POINTS2D is expected to be empty for this known-pose seed.
    """

    images_path = sparse_dir / "images.txt"
    require_file(images_path, "known-camera images.txt")

    rows: dict[
        str,
        tuple[int, int, tuple[float, float, float, float, float, float, float]],
    ] = {}
    for raw in images_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 10:
            # Empty POINTS2D rows are skipped here.
            continue
        try:
            image_id = int(fields[0])
            pose = tuple(float(value) for value in fields[1:8])
            camera_id = int(fields[8])
        except ValueError:
            # A non-pose row cannot occur in a clean known-pose seed, but
            # ignoring it here makes the parser tolerant to diagnostics text.
            continue
        if len(pose) != 7:
            raise RuntimeError(f"Invalid pose row in {images_path}: {line}")
        image_name = fields[9]
        if image_name in rows:
            raise RuntimeError(f"Duplicate image name in known-pose seed: {image_name}")
        rows[image_name] = (image_id, camera_id, pose)  # type: ignore[arg-type]

    if not rows:
        raise RuntimeError(f"Failed to parse known camera poses from {images_path}")
    return rows


def _fmt_float(value: float) -> str:
    return format(float(value), ".17g")


def align_known_camera_model_to_database(
    *,
    database_path: Path,
    sparse_dir: Path,
) -> dict[str, object]:
    """Rewrite the known-pose sparse seed using COLMAP database identities.

    COLMAP >= 3.12 stores rigs and frames in both the reconstruction and the
    database.  A legacy text model implicitly creates FRAME_ID=IMAGE_ID and a
    trivial RIG_ID=CAMERA_ID.  Feature extraction, however, may assign image /
    frame IDs in a different order from the renderer seed.  If only image IDs
    are transcribed later, the old frame IDs can collide with database frames
    belonging to different rigs and trigger:

        existing_frame.RigId() == frame.RigId()

    This function uses image filename as the correspondence key, keeps the
    renderer Q/T and exact intrinsics, but adopts the database IMAGE_ID,
    CAMERA_ID, FRAME_ID, and RIG_ID.  It writes an explicit 5-file COLMAP text
    model so point_triangulator no longer needs to infer rig/frame identities.
    """

    cameras, _image_to_known_camera = _read_known_camera_text_model(sparse_dir)
    known_poses = _read_known_pose_rows(sparse_dir)

    backup_dir = sparse_dir.parent / "sparse_renderer_seed"
    if backup_dir.exists():
        raise RuntimeError(
            f"Renderer seed backup already exists: {backup_dir}. "
            "Re-run this model with --force for a clean rebuild."
        )
    shutil.copytree(sparse_dir, backup_dir)

    with sqlite3.connect(database_path) as connection:
        cursor = connection.cursor()
        db_images = cursor.execute(
            "SELECT image_id, name, camera_id FROM images ORDER BY image_id"
        ).fetchall()
        if not db_images:
            raise RuntimeError(f"COLMAP database contains no images: {database_path}")

        # Resolve the exact frame and rig that COLMAP created for each image.
        # frame_data carries the image data_id and sensor_id; for this pipeline
        # every image is the reference camera of a trivial single-camera rig.
        frame_rows = cursor.execute(
            "SELECT i.image_id, i.name, i.camera_id, fd.frame_id, f.rig_id, "
            "       r.ref_sensor_id, r.ref_sensor_type "
            "FROM images AS i "
            "JOIN frame_data AS fd "
            "  ON fd.data_id = i.image_id AND fd.sensor_id = i.camera_id "
            "JOIN frames AS f ON f.frame_id = fd.frame_id "
            "JOIN rigs AS r ON r.rig_id = f.rig_id "
            "ORDER BY i.image_id"
        ).fetchall()

        if len(frame_rows) != len(db_images):
            raise RuntimeError(
                "Could not resolve exactly one COLMAP frame/rig for every database image: "
                f"images={len(db_images)}, resolved_frames={len(frame_rows)}"
            )

        db_image_by_name = {
            str(name): (int(image_id), int(camera_id))
            for image_id, name, camera_id in db_images
        }
        if set(db_image_by_name) != set(known_poses):
            missing_in_seed = sorted(set(db_image_by_name) - set(known_poses))
            missing_in_db = sorted(set(known_poses) - set(db_image_by_name))
            raise RuntimeError(
                "Renderer seed and COLMAP database image names differ. "
                f"missing_in_seed={missing_in_seed[:10]}, "
                f"missing_in_database={missing_in_db[:10]}"
            )

        frame_by_name: dict[str, tuple[int, int, int, int]] = {}
        rig_to_camera: dict[int, int] = {}
        for (
            image_id,
            name,
            camera_id,
            frame_id,
            rig_id,
            ref_sensor_id,
            _ref_sensor_type,
        ) in frame_rows:
            image_id = int(image_id)
            camera_id = int(camera_id)
            frame_id = int(frame_id)
            rig_id = int(rig_id)
            ref_sensor_id = int(ref_sensor_id)
            name = str(name)

            # The feature extractor used by this script creates one trivial rig
            # per camera.  Reject a non-trivial setup rather than silently write
            # an incorrect rig pose.
            if ref_sensor_id != camera_id:
                raise RuntimeError(
                    f"Database image {name!r} uses camera {camera_id}, but frame {frame_id} "
                    f"belongs to rig {rig_id} whose reference sensor is {ref_sensor_id}. "
                    "This script expects feature_extractor's default trivial camera rigs."
                )
            non_ref_sensor_count = int(
                cursor.execute(
                    "SELECT COUNT(*) FROM rig_sensors WHERE rig_id = ?", (rig_id,)
                ).fetchone()[0]
            )
            if non_ref_sensor_count != 0:
                raise RuntimeError(
                    f"Rig {rig_id} is non-trivial ({non_ref_sensor_count} non-reference sensors). "
                    "The Sofa50 known-pose pipeline expects one camera per rig."
                )

            previous_camera = rig_to_camera.get(rig_id)
            if previous_camera is not None and previous_camera != camera_id:
                raise RuntimeError(
                    f"Rig {rig_id} maps to multiple cameras: {previous_camera}, {camera_id}"
                )
            rig_to_camera[rig_id] = camera_id
            frame_by_name[name] = (image_id, camera_id, frame_id, rig_id)

    target_by_db_camera: dict[
        int, tuple[int, int, tuple[float, float, float, float]]
    ] = {}
    aligned_rows: list[
        tuple[
            int,
            int,
            int,
            int,
            str,
            tuple[float, float, float, float, float, float, float],
        ]
    ] = []

    changed_image_ids = 0
    changed_camera_ids = 0
    for name, (old_image_id, old_camera_id, pose) in known_poses.items():
        if old_camera_id not in cameras:
            raise RuntimeError(
                f"Known-pose image {name!r} references missing camera {old_camera_id}"
            )
        image_id, camera_id, frame_id, rig_id = frame_by_name[name]
        target = cameras[old_camera_id]
        previous_target = target_by_db_camera.get(camera_id)
        if previous_target is not None and previous_target != target:
            raise RuntimeError(
                f"Database camera {camera_id} is shared by renderer images with different "
                "intrinsics. Re-run feature extraction with per-image cameras."
            )
        target_by_db_camera[camera_id] = target
        changed_image_ids += int(old_image_id != image_id)
        changed_camera_ids += int(old_camera_id != camera_id)
        aligned_rows.append((image_id, camera_id, frame_id, rig_id, name, pose))

    aligned_rows.sort(key=lambda row: row[0])

    cameras_lines = [
        "# Camera list with one line of data per camera:",
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
        f"# Number of cameras: {len(target_by_db_camera)}",
    ]
    for camera_id in sorted(target_by_db_camera):
        width, height, params = target_by_db_camera[camera_id]
        cameras_lines.append(
            " ".join(
                [
                    str(camera_id),
                    "PINHOLE",
                    str(width),
                    str(height),
                    *(_fmt_float(value) for value in params),
                ]
            )
        )

    rigs_lines = [
        "# Rig calib list with one line of data per calib:",
        "#   RIG_ID, NUM_SENSORS, REF_SENSOR_TYPE, REF_SENSOR_ID, "
        "SENSORS[] as (SENSOR_TYPE, SENSOR_ID, HAS_POSE, [QW, QX, QY, QZ, TX, TY, TZ])",
        f"# Number of rigs: {len(rig_to_camera)}",
    ]
    for rig_id in sorted(rig_to_camera):
        rigs_lines.append(f"{rig_id} 1 CAMERA {rig_to_camera[rig_id]}")

    frames_lines = [
        "# Frame list with one line of data per frame:",
        "#   FRAME_ID, RIG_ID, RIG_FROM_WORLD[QW, QX, QY, QZ, TX, TY, TZ], "
        "NUM_DATA_IDS, DATA_IDS[] as (SENSOR_TYPE, SENSOR_ID, DATA_ID)",
        f"# Number of frames: {len(aligned_rows)}",
    ]
    images_lines = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
        f"# Number of images: {len(aligned_rows)}, mean observations per image: 0",
    ]

    for image_id, camera_id, frame_id, rig_id, name, pose in aligned_rows:
        pose_text = " ".join(_fmt_float(value) for value in pose)
        frames_lines.append(
            f"{frame_id} {rig_id} {pose_text} 1 CAMERA {camera_id} {image_id}"
        )
        images_lines.append(f"{image_id} {pose_text} {camera_id} {name}")
        images_lines.append("")

    (sparse_dir / "cameras.txt").write_text(
        "\n".join(cameras_lines) + "\n", encoding="utf-8"
    )
    (sparse_dir / "rigs.txt").write_text(
        "\n".join(rigs_lines) + "\n", encoding="utf-8"
    )
    (sparse_dir / "frames.txt").write_text(
        "\n".join(frames_lines) + "\n", encoding="utf-8"
    )
    (sparse_dir / "images.txt").write_text(
        "\n".join(images_lines) + "\n", encoding="utf-8"
    )
    (sparse_dir / "points3D.txt").write_text(
        "# 3D point list with one line of data per point:\n"
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as "
        "(IMAGE_ID, POINT2D_IDX)\n"
        "# Number of points: 0, mean track length: 0\n",
        encoding="utf-8",
    )

    # Reconstruction::Read prefers binary files when present.  Remove stale
    # binaries so the DB-aligned text files above are guaranteed to be used.
    removed_binary_files: list[str] = []
    for filename in ["rigs.bin", "cameras.bin", "frames.bin", "images.bin", "points3D.bin"]:
        path = sparse_dir / filename
        if path.exists():
            path.unlink()
            removed_binary_files.append(filename)

    return {
        "aligned_images": len(aligned_rows),
        "aligned_cameras": len(target_by_db_camera),
        "aligned_frames": len(aligned_rows),
        "aligned_rigs": len(rig_to_camera),
        "changed_image_ids": changed_image_ids,
        "changed_camera_ids": changed_camera_ids,
        "renderer_seed_backup": str(backup_dir),
        "removed_stale_binary_files": removed_binary_files,
    }


def sync_renderer_intrinsics_to_database(
    *,
    database_path: Path,
    sparse_dir: Path,
) -> dict[str, object]:
    """Replace COLMAP's auto-guessed DB intrinsics with renderer-known PINHOLE intrinsics.

    Feature extraction must run first so the database contains images, cameras,
    keypoints, and descriptors.  Matching runs only after this synchronization.
    """

    cameras, image_to_known_camera = _read_known_camera_text_model(sparse_dir)
    target_by_db_camera: dict[int, tuple[int, int, tuple[float, float, float, float]]] = {}
    mapped_images = 0
    before_fx: list[float] = []
    after_fx: list[float] = []

    with sqlite3.connect(database_path) as connection:
        cursor = connection.cursor()
        db_images = cursor.execute(
            "SELECT image_id, name, camera_id FROM images ORDER BY image_id"
        ).fetchall()
        if not db_images:
            raise RuntimeError(f"COLMAP database contains no images: {database_path}")

        for _image_id, name, db_camera_id in db_images:
            known_camera_id = image_to_known_camera.get(str(name))
            if known_camera_id is None:
                raise RuntimeError(
                    f"Database image {name!r} is missing from renderer-known images.txt"
                )
            target = cameras[known_camera_id]
            db_camera_id = int(db_camera_id)
            previous_target = target_by_db_camera.get(db_camera_id)
            if previous_target is not None and previous_target != target:
                raise RuntimeError(
                    f"Database camera {db_camera_id} is shared by images that require "
                    "different renderer intrinsics. Re-run feature extraction with "
                    "per-image cameras."
                )
            target_by_db_camera[db_camera_id] = target
            mapped_images += 1

        for db_camera_id, (width, height, params) in target_by_db_camera.items():
            row = cursor.execute(
                "SELECT model, params FROM cameras WHERE camera_id = ?",
                (db_camera_id,),
            ).fetchone()
            if row is None:
                raise RuntimeError(f"Database camera {db_camera_id} not found")
            model_id, old_blob = row
            if old_blob is not None and len(old_blob) >= 8:
                before_fx.append(struct.unpack("<d", bytes(old_blob[:8]))[0])
            blob = struct.pack("<4d", *params)
            cursor.execute(
                "UPDATE cameras "
                "SET width = ?, height = ?, params = ?, prior_focal_length = 1 "
                "WHERE camera_id = ?",
                (int(width), int(height), sqlite3.Binary(blob), db_camera_id),
            )
            after_fx.append(float(params[0]))

        connection.commit()

        # Verify the update really landed in SQLite before matching.
        for db_camera_id, (_width, _height, params) in target_by_db_camera.items():
            row = cursor.execute(
                "SELECT params FROM cameras WHERE camera_id = ?",
                (db_camera_id,),
            ).fetchone()
            if row is None or row[0] is None or len(row[0]) != 32:
                raise RuntimeError(f"Invalid PINHOLE params blob for DB camera {db_camera_id}")
            actual = struct.unpack("<4d", bytes(row[0]))
            if any(abs(a - b) > 1e-9 for a, b in zip(actual, params)):
                raise RuntimeError(
                    f"Renderer intrinsics sync verification failed for DB camera {db_camera_id}: "
                    f"expected={params}, actual={actual}"
                )

    return {
        "mapped_images": mapped_images,
        "updated_database_cameras": len(target_by_db_camera),
        "database_fx_before_min": min(before_fx) if before_fx else None,
        "database_fx_before_max": max(before_fx) if before_fx else None,
        "renderer_fx_after_min": min(after_fx) if after_fx else None,
        "renderer_fx_after_max": max(after_fx) if after_fx else None,
    }


def resolve_model_ids(render_root: Path, requested: list[str], run_all: bool) -> list[str]:
    if run_all:
        model_ids = sorted(
            path.name
            for path in render_root.iterdir()
            if path.is_dir() and (path / "dataset.json").is_file()
        )
        if not model_ids:
            raise FileNotFoundError(f"No rendered Sofa50 datasets found under {render_root}")
        return model_ids
    if requested:
        return requested
    return [DEFAULT_TEST_MODEL]


def prepare_known_camera_model(
    *,
    mlr_python: Path,
    downstream_root: Path,
    dataset_path: Path,
    scene_dir: Path,
    coarse_obj: Path,
) -> None:
    env = with_prepend_path(os.environ.copy(), "PYTHONPATH", downstream_root / "src")
    run(
        [
            str(mlr_python),
            "-m",
            "mlr.cli",
            "coarse-openmvs",
            "--dataset",
            str(dataset_path),
            "--scene-dir",
            str(scene_dir),
            "--out",
            str(coarse_obj),
            "--interface",
            "colmap",
            "--no-visibility",
            "--prepare-only",
        ],
        env=env,
        cwd=downstream_root,
    )


def convert_ply_to_obj(
    *,
    mlr_python: Path,
    downstream_root: Path,
    ply_path: Path,
    obj_path: Path,
) -> None:
    env = with_prepend_path(os.environ.copy(), "PYTHONPATH", downstream_root / "src")
    code = (
        "from mlr.io import load_mesh, save_mesh; "
        f"mesh = load_mesh({str(ply_path)!r}).ensure_normals(); "
        f"save_mesh(mesh, {str(obj_path)!r})"
    )
    run([str(mlr_python), "-c", code], env=env, cwd=downstream_root)


def reconstruct_one(
    *,
    model_id: str,
    render_root: Path,
    output_root: Path,
    downstream_root: Path,
    mlr_python: Path,
    colmap_bin: Path,
    colmap_runtime_lib_dir: Path | None,
    openmvs_bin_dir: Path,
    resolution_level: int,
    force: bool,
) -> dict[str, object]:
    dataset_path = render_root / model_id / "dataset.json"
    require_file(dataset_path, "dataset.json")

    model_dir = output_root / "models" / model_id
    scene_dir = model_dir / "scene"
    coarse_obj = model_dir / "coarse.obj"
    coarse_ply = model_dir / "coarse.ply"
    diagnostics_path = model_dir / "diagnostics.json"

    if model_dir.exists() and force:
        shutil.rmtree(model_dir)
    elif coarse_obj.exists():
        print(f"[skip] {model_id}: {coarse_obj} already exists", flush=True)
        return {
            "script_version": SCRIPT_VERSION,
            "model_id": model_id,
            "status": "skipped_existing",
            "coarse_obj": str(coarse_obj),
        }
    elif model_dir.exists():
        raise RuntimeError(
            f"Incomplete output directory already exists: {model_dir}. "
            "Re-run with --force to rebuild this model cleanly."
        )

    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {model_id} ===", flush=True)
    print("[1/8] Prepare renderer-known COLMAP camera model", flush=True)
    prepare_known_camera_model(
        mlr_python=mlr_python,
        downstream_root=downstream_root,
        dataset_path=dataset_path,
        scene_dir=scene_dir,
        coarse_obj=coarse_obj,
    )

    colmap_root = scene_dir / "colmap"
    images_dir = colmap_root / "images"
    sparse_seed = colmap_root / "sparse"
    database_path = colmap_root / "database.db"
    sparse_triangulated = colmap_root / "sparse_triangulated"
    sparse_triangulated_txt = colmap_root / "sparse_triangulated_txt"
    sparse_known_cameras = colmap_root / "sparse_known_cameras"

    require_file(sparse_seed / "cameras.txt", "known-camera cameras.txt")
    require_file(sparse_seed / "images.txt", "known-camera images.txt")
    require_file(sparse_seed / "points3D.txt", "known-camera points3D.txt")

    colmap_env = os.environ.copy()
    if colmap_runtime_lib_dir is not None:
        colmap_env = with_prepend_path(colmap_env, "LD_LIBRARY_PATH", colmap_runtime_lib_dir)

    print("[2/8] COLMAP CPU SIFT feature extraction", flush=True)
    run(
        [
            str(colmap_bin),
            "feature_extractor",
            "--database_path",
            str(database_path),
            "--image_path",
            str(images_dir),
            "--ImageReader.camera_model",
            "PINHOLE",
            "--SiftExtraction.use_gpu",
            "0",
            "--SiftExtraction.peak_threshold",
            str(SIFT_PEAK_THRESHOLD),

            "--SiftExtraction.estimate_affine_shape",
            "1",
            "--SiftExtraction.domain_size_pooling",
            "1",
        ],
        env=colmap_env,
    )

    print("[3/8] Align known-pose sparse IDs/frames/rigs to COLMAP database", flush=True)
    identity_alignment = align_known_camera_model_to_database(
        database_path=database_path,
        sparse_dir=sparse_seed,
    )
    print(
        "COLMAP identity alignment:",
        json.dumps(identity_alignment, sort_keys=True),
        flush=True,
    )

    print("      Synchronize renderer exact intrinsics into COLMAP database", flush=True)
    intrinsics_sync = sync_renderer_intrinsics_to_database(
        database_path=database_path,
        sparse_dir=sparse_seed,
    )
    print("Renderer intrinsics sync:", json.dumps(intrinsics_sync, sort_keys=True), flush=True)

    print("[4/8] COLMAP exhaustive matching with renderer intrinsics", flush=True)
    run(
    [
            str(colmap_bin),
            "exhaustive_matcher",
            "--database_path",
            str(database_path),
            "--SiftMatching.use_gpu",
            "0",
            "--SiftMatching.guided_matching",
            "1",
        ],
        env=colmap_env,
    )

    db_diag = read_database_diagnostics(database_path)
    print(
        "COLMAP matching diagnostics: "
        f"images={db_diag['database_images']}, "
        f"keypoints={db_diag['total_keypoints']}, "
        f"raw_pairs={db_diag['raw_matched_pairs']}, "
        f"raw_matches={db_diag['total_raw_matches']}, "
        f"verified_pairs={db_diag['verified_matched_pairs']}, "
        f"inlier_matches={db_diag['total_inlier_matches']}",
        flush=True,
    )

    if not db_diag.get("total_inlier_matches"):
        diagnostics = {
            "script_version": SCRIPT_VERSION,
            "model_id": model_id,
            "status": "failed_zero_verified_matches",
            "dataset": str(dataset_path),
            "database": str(database_path),
            "known_camera_sparse": str(sparse_seed),
            "identity_alignment": identity_alignment,
            "intrinsics_sync": intrinsics_sync,
            "sift_peak_threshold": SIFT_PEAK_THRESHOLD,
            **db_diag,
            "resolution_level": resolution_level,
        }
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
        raise RuntimeError(
            f"COLMAP exhaustive matching produced 0 verified matches for {model_id}. "
            f"Triangulation/OpenMVS were not started. See {diagnostics_path}."
        )

    sparse_triangulated.mkdir(parents=True, exist_ok=True)
    print("[5/8] COLMAP fixed-camera point triangulation", flush=True)
    run(
        [
            str(colmap_bin),
            "point_triangulator",
            "--database_path",
            str(database_path),
            "--image_path",
            str(images_dir),
            "--input_path",
            str(sparse_seed),
            "--output_path",
            str(sparse_triangulated),
            "--clear_points",
            "1",
            "--refine_intrinsics",
            "0",
            "--Mapper.tri_ignore_two_view_tracks",
            "0",
        ],
        env=colmap_env,
    )

    sparse_triangulated_txt.mkdir(parents=True, exist_ok=True)
    run(
        [
            str(colmap_bin),
            "model_converter",
            "--input_path",
            str(sparse_triangulated),
            "--output_path",
            str(sparse_triangulated_txt),
            "--output_type",
            "TXT",
        ],
        env=colmap_env,
    )

    points_txt = sparse_triangulated_txt / "points3D.txt"
    require_file(points_txt, "triangulated points3D.txt")
    num_sparse_points = count_text_points3d(points_txt)
    print(
        "COLMAP triangulation diagnostics: "
        f"verified_pairs={db_diag['verified_matched_pairs']}, "
        f"inlier_matches={db_diag['total_inlier_matches']}, "
        f"triangulated_points={num_sparse_points}",
        flush=True,
    )

    diagnostics: dict[str, object] = {
        "script_version": SCRIPT_VERSION,
        "model_id": model_id,
        "dataset": str(dataset_path),
        "database": str(database_path),
        "known_camera_sparse": str(sparse_seed),
        "triangulated_sparse": str(sparse_triangulated),
        "triangulated_sparse_text": str(sparse_triangulated_txt),
        "identity_alignment": identity_alignment,
        "intrinsics_sync": intrinsics_sync,
        "sift_peak_threshold": SIFT_PEAK_THRESHOLD,
        "num_sparse_points": num_sparse_points,
        **db_diag,
        "resolution_level": resolution_level,
    }

    if num_sparse_points <= 0:
        diagnostics["status"] = "failed_empty_sparse_triangulation"
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
        raise RuntimeError(
            f"COLMAP triangulation produced 0 points for {model_id}. "
            f"See {diagnostics_path}. OpenMVS was not started."
        )

    # InterfaceCOLMAP in the existing MLR pipeline expects <colmap_root>/sparse.
    # Preserve the original renderer-known camera-only model, then place the
    # successfully triangulated model at the expected sparse/ location.
    print("[6/8] Promote triangulated sparse model for OpenMVS", flush=True)
    shutil.move(str(sparse_seed), str(sparse_known_cameras))
    shutil.copytree(sparse_triangulated, sparse_seed)

    interface_colmap = openmvs_bin_dir / "InterfaceCOLMAP"
    densify = openmvs_bin_dir / "DensifyPointCloud"
    reconstruct = openmvs_bin_dir / "ReconstructMesh"
    for path, label in [
        (interface_colmap, "InterfaceCOLMAP"),
        (densify, "DensifyPointCloud"),
        (reconstruct, "ReconstructMesh"),
    ]:
        require_executable(path, label)

    scene_mvs = scene_dir / "scene.mvs"
    dense_mvs = scene_dir / "scene_dense.mvs"

    print("[7/8] OpenMVS InterfaceCOLMAP + DensifyPointCloud", flush=True)
    run(
        [
            str(interface_colmap),
            "-w",
            str(scene_dir),
            "-i",
            str(colmap_root),
            "-o",
            str(scene_mvs),
            "--image-folder",
            str(images_dir),
        ]
    )
    run(
        [
            str(densify),
            "-w",
            str(scene_dir),
            "-i",
            str(scene_mvs),
            "-o",
            str(dense_mvs),
            "--resolution-level",
            str(resolution_level),
        ]
    )

    print("[8/8] OpenMVS ReconstructMesh + OBJ conversion", flush=True)
    run(
        [
            str(reconstruct),
            "-w",
            str(scene_dir),
            "-i",
            str(dense_mvs),
            "-o",
            str(coarse_ply),
            "--export-type",
            "ply",
        ]
    )
    require_file(coarse_ply, "OpenMVS coarse PLY")

    convert_ply_to_obj(
        mlr_python=mlr_python,
        downstream_root=downstream_root,
        ply_path=coarse_ply,
        obj_path=coarse_obj,
    )
    require_file(coarse_obj, "coarse OBJ")

    diagnostics.update(
        {
            "status": "ok",
            "known_camera_sparse_backup": str(sparse_known_cameras),
            "openmvs_sparse": str(sparse_seed),
            "scene_mvs": str(scene_mvs),
            "dense_mvs": str(dense_mvs),
            "coarse_ply": str(coarse_ply),
            "coarse_obj": str(coarse_obj),
        }
    )
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    print(json.dumps(diagnostics, indent=2), flush=True)
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare Sofa50 real coarse meshes from 1920 multi-view renders using "
            "renderer-known camera poses, renderer-intrinsics-synchronized COLMAP "
            "matching/fixed-pose triangulation, and OpenMVS densification/meshing. "
            f"Script {SCRIPT_VERSION}."
        )
    )
    parser.add_argument(
        "--refinement-root",
        type=Path,
        default=Path("~/sofa_mesh/sofa50_refinement"),
    )
    parser.add_argument(
        "--downstream-root",
        type=Path,
        default=Path("~/multiview-laplacian-refinement"),
    )
    parser.add_argument(
        "--render-root",
        type=Path,
        default=None,
        help="Default: <refinement-root>/multiview_1920/rendered",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Default: <refinement-root>/openmvs_coarse_v6",
    )
    parser.add_argument(
        "--mlr-python",
        type=Path,
        default=Path("~/miniconda3/envs/test/bin/python"),
        help="Python interpreter that can run `python -m mlr.cli`.",
    )
    parser.add_argument(
        "--colmap-bin",
        type=Path,
        default=Path("~/vcpkg_colmap_cli/installed/x64-linux/tools/colmap/colmap"),
    )
    parser.add_argument(
        "--colmap-runtime-lib-dir",
        type=Path,
        default=Path("~/miniconda3/envs/colmap_vcpkg_cli/lib"),
        help=(
            "Prepended to LD_LIBRARY_PATH only for COLMAP subprocesses. "
            "Use --no-colmap-runtime-lib-dir to disable."
        ),
    )
    parser.add_argument(
        "--no-colmap-runtime-lib-dir",
        action="store_true",
    )
    parser.add_argument(
        "--openmvs-bin-dir",
        type=Path,
        default=Path("~/OpenMVS-v2.4.0/bin"),
    )
    parser.add_argument(
        "--model-id",
        action="append",
        default=[],
        help=(
            "Sofa model id to process. Repeat for multiple models. "
            f"If omitted, defaults to the current test model {DEFAULT_TEST_MODEL}."
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process every rendered model containing dataset.json.",
    )
    parser.add_argument(
        "--resolution-level",
        type=int,
        default=2,
        help="OpenMVS DensifyPointCloud resolution level (existing baseline: 2).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete the selected model output directory before rebuilding.",
    )
    args = parser.parse_args()

    refinement_root = expand(args.refinement_root)
    downstream_root = expand(args.downstream_root)
    render_root = expand(args.render_root) if args.render_root else refinement_root / "multiview_1920" / "rendered"
    output_root = expand(args.output_root) if args.output_root else refinement_root / "openmvs_coarse_v6"
    mlr_python = expand(args.mlr_python)
    colmap_bin = expand(args.colmap_bin)
    openmvs_bin_dir = expand(args.openmvs_bin_dir)
    colmap_runtime_lib_dir = None
    if not args.no_colmap_runtime_lib_dir:
        colmap_runtime_lib_dir = expand(args.colmap_runtime_lib_dir)

    require_executable(mlr_python, "MLR Python")
    require_executable(colmap_bin, "COLMAP")
    if colmap_runtime_lib_dir is not None and not colmap_runtime_lib_dir.is_dir():
        raise FileNotFoundError(
            f"COLMAP runtime lib directory not found: {colmap_runtime_lib_dir}"
        )
    if not downstream_root.is_dir():
        raise FileNotFoundError(f"Downstream repository not found: {downstream_root}")
    if not render_root.is_dir():
        raise FileNotFoundError(f"Render root not found: {render_root}")

    # Smoke-test the exact COLMAP binary/environment before touching a model.
    colmap_env = os.environ.copy()
    if colmap_runtime_lib_dir is not None:
        colmap_env = with_prepend_path(colmap_env, "LD_LIBRARY_PATH", colmap_runtime_lib_dir)
    run([str(colmap_bin), "-h"], env=colmap_env)

    model_ids = resolve_model_ids(render_root, args.model_id, args.all)
    print(f"Processing {len(model_ids)} model(s): {model_ids}", flush=True)

    results: list[dict[str, object]] = []
    for index, model_id in enumerate(model_ids, start=1):
        print(f"\n[{index}/{len(model_ids)}] {model_id}", flush=True)
        results.append(
            reconstruct_one(
                model_id=model_id,
                render_root=render_root,
                output_root=output_root,
                downstream_root=downstream_root,
                mlr_python=mlr_python,
                colmap_bin=colmap_bin,
                colmap_runtime_lib_dir=colmap_runtime_lib_dir,
                openmvs_bin_dir=openmvs_bin_dir,
                resolution_level=args.resolution_level,
                force=args.force,
            )
        )

    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "summary.json"
    summary_payload = {"script_version": SCRIPT_VERSION, "results": results}
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(f"\nWrote summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()