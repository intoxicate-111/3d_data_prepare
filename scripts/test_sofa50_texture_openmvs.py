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


SCRIPT_VERSION = "texture-test-v6-48view-overlapping-rings-v5-texture-diagnostics"
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


def read_colmap_points3d_quality(points_path: Path) -> dict[str, object]:
    """Summarize COLMAP sparse-point reprojection errors and track lengths."""
    import math
    import statistics

    errors: list[float] = []
    track_lengths: list[int] = []
    with points_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 8:
                continue
            try:
                error = float(fields[7])
            except ValueError:
                continue
            track_length = max(0, (len(fields) - 8) // 2)
            if math.isfinite(error):
                errors.append(error)
            track_lengths.append(track_length)

    def percentile(values: list[float], q: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        if len(ordered) == 1:
            return float(ordered[0])
        pos = (len(ordered) - 1) * q
        lo = int(pos)
        hi = min(lo + 1, len(ordered) - 1)
        alpha = pos - lo
        return float((1.0 - alpha) * ordered[lo] + alpha * ordered[hi])

    track2 = sum(length == 2 for length in track_lengths)
    track3 = sum(length == 3 for length in track_lengths)
    track4plus = sum(length >= 4 for length in track_lengths)
    return {
        "num_points": len(track_lengths),
        "reprojection_error_mean": float(statistics.fmean(errors)) if errors else None,
        "reprojection_error_median": float(statistics.median(errors)) if errors else None,
        "reprojection_error_p95": percentile(errors, 0.95),
        "reprojection_error_max": max(errors) if errors else None,
        "track_length_mean": float(statistics.fmean(track_lengths)) if track_lengths else None,
        "track_length_median": float(statistics.median(track_lengths)) if track_lengths else None,
        "track_length_2": track2,
        "track_length_3": track3,
        "track_length_ge4": track4plus,
        "track_length_2_fraction": (track2 / len(track_lengths)) if track_lengths else None,
        "track_length_ge4_fraction": (track4plus / len(track_lengths)) if track_lengths else None,
    }


def surface_distance_diagnostic(
    *,
    mlr_python: Path,
    downstream_root: Path,
    source_path: Path,
    source_kind: str,
    gt_mesh_path: Path,
    output_path: Path,
    max_points: int = 512,
) -> dict[str, object]:
    """Sample reconstructed points and measure exact nearest distance to GT triangles.

    Diagnostic only: GT is never passed to COLMAP/OpenMVS.
    """
    code = r'''import json
import sys
from pathlib import Path
import numpy as np
from mlr.io import load_mesh
from mlr.gt_laplacian import closest_points_on_mesh

kind = sys.argv[1]
source = Path(sys.argv[2])
gt_path = Path(sys.argv[3])
out = Path(sys.argv[4])
max_points = int(sys.argv[5])

if kind == "colmap_points3d":
    rows = []
    for raw in source.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) >= 4:
            rows.append([float(fields[1]), float(fields[2]), float(fields[3])])
    points = np.asarray(rows, dtype=np.float64).reshape((-1, 3))
elif kind in {"ply", "obj"}:
    points = np.asarray(load_mesh(source).vertices, dtype=np.float64)
else:
    raise ValueError(f"unsupported source kind: {kind}")

gt = load_mesh(gt_path)
num_points = int(points.shape[0])
if num_points == 0:
    payload = {"status": "empty", "source_points": 0, "sampled_points": 0}
else:
    sample_count = min(num_points, max_points)
    if sample_count == num_points:
        sample = points
    else:
        indices = np.linspace(0, num_points - 1, sample_count, dtype=np.int64)
        sample = points[indices]
    result = closest_points_on_mesh(sample, gt.vertices, gt.faces)
    d = np.asarray(result.distances, dtype=np.float64)
    payload = {
        "status": "ok",
        "source_points": num_points,
        "sampled_points": int(sample.shape[0]),
        "gt_vertices": int(gt.num_vertices),
        "gt_faces": int(gt.num_faces),
        "distance_mean": float(np.mean(d)),
        "distance_median": float(np.median(d)),
        "distance_p90": float(np.percentile(d, 90)),
        "distance_p95": float(np.percentile(d, 95)),
        "distance_p99": float(np.percentile(d, 99)),
        "distance_max": float(np.max(d)),
    }
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload))
'''
    env = with_prepend_path(os.environ.copy(), "PYTHONPATH", downstream_root / "src")
    try:
        subprocess.run(
            [
                str(mlr_python),
                "-c",
                code,
                source_kind,
                str(source_path),
                str(gt_mesh_path),
                str(output_path),
                str(max_points),
            ],
            check=True,
            env=env,
            cwd=str(downstream_root),
        )
        return json.loads(output_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[diagnostic warning] surface-distance diagnostic failed for {source_path}: {exc}", flush=True)
        return {"status": "failed", "error": str(exc), "source": str(source_path)}


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



def resolve_sofa_model_ids(
    sofa_root: Path,
    requested: list[str],
    run_all: bool,
) -> list[str]:
    """Resolve Sofa50 model IDs from the canonical <sofa-root>/<id>/mesh.obj layout."""
    if requested:
        return requested
    if run_all:
        selection_path = sofa_root / "selection.json"
        require_file(selection_path, "Sofa50 selection.json")
        payload = json.loads(selection_path.read_text(encoding="utf-8"))
        splits = payload.get("splits", {})
        model_ids: list[str] = []
        seen: set[str] = set()
        for split in ("train", "validation", "test"):
            values = splits.get(split, [])
            if not isinstance(values, list):
                raise ValueError(f"Invalid split {split!r} in {selection_path}")
            for value in values:
                model_id = str(value)
                if model_id and model_id not in seen:
                    seen.add(model_id)
                    model_ids.append(model_id)
        if not model_ids:
            raise RuntimeError(f"No Sofa50 IDs found in {selection_path}")
        return model_ids
    return [DEFAULT_TEST_MODEL]


def sofa_mesh_path(sofa_root: Path, model_id: str) -> Path:
    """Canonical Sofa50 mesh path used by prepare_sofa50_gt_query.py."""
    path = sofa_root / model_id / "mesh.obj"
    require_file(path, f"Sofa50 mesh for {model_id}")
    return path


def render_textured_dataset(
    *,
    script_path: Path,
    mlr_python: Path,
    downstream_root: Path,
    mesh_path: Path,
    render_dir: Path,
    image_size: int,
    texture_smoothing_steps: int,
    texture_seed: int,
    force: bool,
) -> dict[str, object]:
    """Run the renderer stage inside the MLR Python environment."""
    if render_dir.exists() and force:
        shutil.rmtree(render_dir)
    dataset_path = render_dir / "dataset.json"
    render_diag_path = render_dir / "texture_render_diagnostics.json"
    if dataset_path.is_file() and render_diag_path.is_file() and not force:
        print(f"[render skip] Existing textured dataset: {dataset_path}", flush=True)
        return json.loads(render_diag_path.read_text(encoding="utf-8"))
    if render_dir.exists():
        raise RuntimeError(
            f"Incomplete texture render directory exists: {render_dir}. "
            "Re-run with --force to rebuild it cleanly."
        )

    env = with_prepend_path(os.environ.copy(), "PYTHONPATH", downstream_root / "src")
    run(
        [
            str(mlr_python),
            str(script_path),
            "--_render-stage",
            "--_render-mesh",
            str(mesh_path),
            "--_render-out",
            str(render_dir),
            "--_render-image-size",
            str(image_size),
            "--_texture-smoothing-steps",
            str(texture_smoothing_steps),
            "--_texture-seed",
            str(texture_seed),
            "--_downstream-root",
            str(downstream_root),
        ],
        env=env,
        cwd=downstream_root,
    )
    require_file(dataset_path, "textured dataset.json")
    require_file(render_diag_path, "texture render diagnostics")
    return json.loads(render_diag_path.read_text(encoding="utf-8"))


def _render_stage_cli(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Internal Sofa50 texture renderer stage")
    parser.add_argument("--_render-mesh", required=True, type=Path)
    parser.add_argument("--_render-out", required=True, type=Path)
    parser.add_argument("--_render-image-size", required=True, type=int)
    parser.add_argument("--_texture-smoothing-steps", required=True, type=int)
    parser.add_argument("--_texture-seed", required=True, type=int)
    parser.add_argument("--_downstream-root", required=True, type=Path)
    args = parser.parse_args(argv)
    _render_textured_dataset_impl(
        mesh_path=expand(args._render_mesh),
        render_dir=expand(args._render_out),
        image_size=args._render_image_size,
        texture_smoothing_steps=args._texture_smoothing_steps,
        texture_seed=args._texture_seed,
        downstream_root=expand(args._downstream_root),
    )


def _mesh_unique_edges(faces):
    import numpy as np

    faces = np.asarray(faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must have shape [F,3], got {faces.shape}")
    edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
    )
    edges = np.sort(edges, axis=1)
    return np.unique(edges, axis=0)


def _smooth_vertex_field(values, edges, steps: int, neighbor_weight: float = 0.58):
    """Small graph diffusion used only to turn vertex noise into a C0 surface texture."""
    import numpy as np

    field = np.asarray(values, dtype=np.float64).copy()
    edges = np.asarray(edges, dtype=np.int64)
    if field.ndim == 1:
        field = field[:, None]
    if steps < 0:
        raise ValueError("texture smoothing steps must be non-negative")
    if not (0.0 <= neighbor_weight <= 1.0):
        raise ValueError("neighbor_weight must be in [0,1]")

    n = field.shape[0]
    degree = np.zeros(n, dtype=np.float64)
    np.add.at(degree, edges[:, 0], 1.0)
    np.add.at(degree, edges[:, 1], 1.0)
    safe_degree = np.maximum(degree, 1.0)

    for _ in range(steps):
        neighbor_sum = np.zeros_like(field)
        np.add.at(neighbor_sum, edges[:, 0], field[edges[:, 1]])
        np.add.at(neighbor_sum, edges[:, 1], field[edges[:, 0]])
        neighbor_mean = neighbor_sum / safe_degree[:, None]
        field = (1.0 - neighbor_weight) * field + neighbor_weight * neighbor_mean
    return field


def _robust_unit_interval(values):
    import numpy as np

    values = np.asarray(values, dtype=np.float64)
    lo = np.percentile(values, 1.0, axis=0, keepdims=True)
    hi = np.percentile(values, 99.0, axis=0, keepdims=True)
    unit = (values - lo) / np.maximum(hi - lo, 1e-9)
    return np.clip(unit, 0.0, 1.0)


def _make_smoothed_random_vertex_texture(vertices, faces, texture_seed: int, smoothing_steps: int):
    """Create a continuous, aperiodic, surface-attached texture at mesh vertices.

    Random values exist only at shared mesh vertices. Two graph-smoothed random
    fields provide local and broader structure. During rendering, RGB is
    perspective-correctly interpolated across each triangle, so there are no
    floor/step/hash/checker discontinuities and the same surface point receives
    the same RGB from every camera.
    """
    import numpy as np

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must have shape [N,3], got {vertices.shape}")
    if smoothing_steps < 0:
        raise ValueError("smoothing_steps must be non-negative")

    edges = _mesh_unique_edges(faces)
    rng = np.random.default_rng(int(texture_seed))
    n = vertices.shape[0]

    # Luminance is explicit because COLMAP SIFT and OpenMVS patch matching must
    # still see strong structure after RGB -> grayscale conversion.
    lum_local = _smooth_vertex_field(
        rng.standard_normal((n, 1)), edges, smoothing_steps, neighbor_weight=0.58
    )
    lum_broad = _smooth_vertex_field(
        rng.standard_normal((n, 1)), edges, smoothing_steps + 4, neighbor_weight=0.62
    )
    lum = _robust_unit_interval(0.76 * lum_local + 0.24 * lum_broad)[:, 0]

    chroma_local = _smooth_vertex_field(
        rng.standard_normal((n, 3)), edges, smoothing_steps + 1, neighbor_weight=0.56
    )
    chroma_broad = _smooth_vertex_field(
        rng.standard_normal((n, 3)), edges, smoothing_steps + 5, neighbor_weight=0.62
    )
    chroma = _robust_unit_interval(0.72 * chroma_local + 0.28 * chroma_broad) - 0.5

    # Strong luminance contrast for feature/patch support, modest chroma for
    # uniqueness. Keep away from clipping so resampling remains well behaved.
    base = 0.12 + 0.76 * lum
    rgb = np.stack(
        [
            base + 0.20 * chroma[:, 0] - 0.06 * chroma[:, 1],
            base + 0.17 * chroma[:, 1] + 0.05 * chroma[:, 2],
            base + 0.18 * chroma[:, 2] - 0.05 * chroma[:, 0],
        ],
        axis=1,
    )
    rgb = np.clip(rgb, 0.06, 0.94)
    rgb_u8 = np.round(rgb * 255.0).astype(np.uint8)

    edge_diff = np.abs(rgb[edges[:, 0]] - rgb[edges[:, 1]])
    stats = {
        "vertex_count": int(n),
        "edge_count": int(edges.shape[0]),
        "smoothing_steps": int(smoothing_steps),
        "vertex_rgb_std": [float(x) for x in np.std(rgb, axis=0)],
        "vertex_luminance_std": float(np.std(lum)),
        "neighbor_rgb_absdiff_mean": float(np.mean(edge_diff)),
        "neighbor_rgb_absdiff_p95": float(np.percentile(edge_diff, 95.0)),
    }
    return rgb_u8, stats


def _create_overlapping_48_ring_cameras(
    *,
    mesh,
    image_size: int,
    radius_scale: float = 1.8,
    fov_degrees: float = 90.0,
):
    """Create 48 overlapping cameras as 3 staggered rings x 16 azimuths.

    Elevations are -25, 0, +25 degrees. Adjacent cameras on a ring are 22.5
    degrees apart, and the middle ring is staggered by half a step (11.25 deg)
    so every view has nearby horizontal and vertical/diagonal neighbors.
    The radius is radius_scale times the maximum GT vertex radius around the
    bbox center. With the default 90 degree FOV at 1920px, fx remains 960.
    """
    import math
    import numpy as np
    from mlr.data import Camera
    from mlr.synthetic import look_at_world_to_camera

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    target = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    object_radius = float(np.max(np.linalg.norm(vertices - target[None, :], axis=1)))
    camera_radius = max(1e-3, float(radius_scale) * object_radius)
    focal = 0.5 * image_size / math.tan(math.radians(float(fov_degrees)) * 0.5)
    intrinsics = np.array(
        [[focal, 0.0, image_size * 0.5],
         [0.0, focal, image_size * 0.5],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    elevations_deg = (-25.0, 0.0, 25.0)
    views_per_ring = 16
    azimuth_step = 360.0 / views_per_ring
    cameras = []
    for ring_index, elevation_deg in enumerate(elevations_deg):
        # Stagger the center ring by half a horizontal step. This avoids three
        # cameras lining up on the same meridians and increases diagonal overlap.
        azimuth_offset_deg = 0.5 * azimuth_step if ring_index == 1 else 0.0
        elevation = math.radians(elevation_deg)
        for azimuth_index in range(views_per_ring):
            azimuth_deg = azimuth_offset_deg + azimuth_index * azimuth_step
            azimuth = math.radians(azimuth_deg)
            center = target + camera_radius * np.array(
                [
                    math.cos(elevation) * math.cos(azimuth),
                    math.sin(elevation),
                    math.cos(elevation) * math.sin(azimuth),
                ],
                dtype=np.float64,
            )
            rotation, translation = look_at_world_to_camera(center, target)
            name = f"ring{ring_index}_el{elevation_deg:+05.1f}_az{azimuth_deg:06.2f}"
            cameras.append(
                Camera(
                    intrinsics=intrinsics.copy(),
                    rotation=rotation,
                    translation=translation,
                    image_size=(image_size, image_size),
                    name=name,
                )
            )

    if len(cameras) != 48:
        raise RuntimeError(f"Expected exactly 48 cameras, got {len(cameras)}")
    return cameras, {
        "layout": "three_staggered_rings_16_each_v1",
        "views": 48,
        "views_per_ring": views_per_ring,
        "elevations_degrees": list(elevations_deg),
        "middle_ring_azimuth_offset_degrees": 0.5 * azimuth_step,
        "azimuth_step_degrees": azimuth_step,
        "fov_degrees": float(fov_degrees),
        "focal_pixels": float(focal),
        "target": target.tolist(),
        "object_radius": object_radius,
        "camera_radius_scale": float(radius_scale),
        "camera_radius": camera_radius,
    }


def _render_textured_dataset_impl(
    *,
    mesh_path: Path,
    render_dir: Path,
    image_size: int,
    texture_smoothing_steps: int,
    texture_seed: int,
    downstream_root: Path,
) -> None:
    """Render 48 overlapping Sofa50 views with the frozen V5 vertex texture."""
    if image_size < 64:
        raise ValueError("image_size must be at least 64")
    if texture_smoothing_steps < 0:
        raise ValueError("texture_smoothing_steps must be non-negative")

    sys.path.insert(0, str(downstream_root / "src"))
    import numpy as np
    from PIL import Image

    from mlr.io import load_mesh, save_mesh
    from mlr.synthetic import SyntheticRenderConfig, _write_cameras_json, _write_dataset_json

    mesh = load_mesh(mesh_path).ensure_normals()
    print(
        f"[texture render v6-48] {mesh_path} -> {render_dir}; "
        f"vertices={mesh.num_vertices}, faces={mesh.num_faces}, "
        f"views=48, size={image_size}, smoothing_steps={texture_smoothing_steps}",
        flush=True,
    )
    print(
        "[texture render v6-48] frozen V5 smoothed random shared-vertex texture; "
        "48 cameras = 3 staggered rings x 16 views at elevations -25/0/+25 deg; "
        "direct CPU perspective-correct rasterization; no EGL/OpenGL/CUDA dependency",
        flush=True,
    )

    render_dir.mkdir(parents=True, exist_ok=True)
    image_dir = render_dir / "images"
    mask_dir = render_dir / "masks"
    depth_dir = render_dir / "depth"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    # Config is metadata only here; the custom 48-view camera list is created
    # explicitly below. FOV stays at 90 degrees, so 1920px -> fx=960 as before.
    config = SyntheticRenderConfig(
        num_views=48,
        width=image_size,
        height=image_size,
        trajectory="custom_48_overlapping_rings",
        fov_degrees=90.0,
        render_mode="lit",
        backend="cpu",
        normalize_mesh=False,
        background_color=(0, 0, 0),
        antialiasing="none",
        backface_culling=False,
        front_face_winding="ccw",
    )
    cameras, camera_layout = _create_overlapping_48_ring_cameras(
        mesh=mesh,
        image_size=image_size,
        radius_scale=1.8,
        fov_degrees=90.0,
    )
    print("[texture render v6-48] camera layout:", json.dumps(camera_layout, sort_keys=True), flush=True)

    normalized_mesh_path = render_dir / "mesh.obj"
    save_mesh(mesh, normalized_mesh_path)

    # Frozen from V5: only camera count/layout changes in this experiment.
    vertex_texture_rgb, vertex_texture_stats = _make_smoothed_random_vertex_texture(
        mesh.vertices,
        mesh.faces,
        texture_seed=texture_seed,
        smoothing_steps=texture_smoothing_steps,
    )
    vertex_texture_path = render_dir / "vertex_texture_rgb.npy"
    np.save(vertex_texture_path, vertex_texture_rgb)

    image_paths: list[Path] = []
    mask_paths: list[Path] = []
    depth_paths: list[Path] = []
    for index, camera in enumerate(cameras):
        rgb, mask, depth = _render_vertex_texture_view_cpu_perspective(
            mesh=mesh,
            camera=camera,
            vertex_texture_rgb=vertex_texture_rgb,
        )
        image_path = image_dir / f"{index:04d}.png"
        mask_path = mask_dir / f"{index:04d}.png"
        depth_path = depth_dir / f"{index:04d}.npy"
        Image.fromarray(rgb).save(image_path)
        Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
        np.save(depth_path, depth)
        image_paths.append(image_path)
        mask_paths.append(mask_path)
        depth_paths.append(depth_path)
        print(
            f"[texture render v6-48] view {index + 1:02d}/48: "
            f"foreground_pixels={int(mask.sum())} -> {image_path}",
            flush=True,
        )

    cameras_path = render_dir / "cameras.json"
    dataset_path = render_dir / "dataset.json"
    _write_cameras_json(
        cameras_path, cameras, image_paths, mask_paths, depth_paths, render_dir
    )
    _write_dataset_json(
        dataset_path,
        cameras=cameras,
        cameras_path=cameras_path,
        image_paths=image_paths,
        mask_paths=mask_paths,
        depth_paths=depth_paths,
        mesh_path=normalized_mesh_path,
        source_mesh_path=mesh_path,
        out_dir=render_dir,
        config=config,
        actual_backend="cpu_smoothed_random_vertex_texture_v1_48view",
    )

    dataset_payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    dataset_payload["texture_test"] = {
        "mode": "smoothed_random_vertex_surface_texture_v1",
        "purpose": "48view_overlap_test_for_colmap_openmvs_dense_geometry",
        "texture_seed": int(texture_seed),
        "texture_smoothing_steps": int(texture_smoothing_steps),
        "texture_rasterizer": "direct_cpu_perspective_correct_vertex_rgb_v1",
        "vertex_texture_file": vertex_texture_path.name,
        "view_dependent_lighting": False,
        "hard_spatial_quantization": False,
        "periodic_world_texture": False,
        "topology_id_encoding": False,
        "camera_layout": camera_layout,
        "image_size": [image_size, image_size],
    }
    dataset_path.write_text(json.dumps(dataset_payload, indent=2) + "\n", encoding="utf-8")

    preview_path = render_dir / "texture_preview_grid.jpg"
    _make_texture_preview(image_paths, preview_path)

    gray_std: list[float] = []
    foreground_fraction: list[float] = []
    for image_path, mask_path in zip(image_paths, mask_paths, strict=True):
        rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32)
        mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8) > 127
        foreground_fraction.append(float(mask.mean()))
        if np.any(mask):
            gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
            gray_std.append(float(np.std(gray[mask])))
        else:
            gray_std.append(0.0)

    diagnostics = {
        "script_version": SCRIPT_VERSION,
        "mesh": str(mesh_path),
        "dataset": str(dataset_path),
        "render_dir": str(render_dir),
        "views": 48,
        "image_size": image_size,
        "trajectory": "custom_48_overlapping_rings",
        "camera_layout": camera_layout,
        "texture_mode": "smoothed_random_vertex_surface_texture_v1",
        "texture_seed": int(texture_seed),
        "texture_smoothing_steps": int(texture_smoothing_steps),
        "texture_rasterizer": "direct_cpu_perspective_correct_vertex_rgb_v1",
        "vertex_texture": str(vertex_texture_path),
        "vertex_texture_stats": vertex_texture_stats,
        "foreground_gray_std_min": min(gray_std),
        "foreground_gray_std_mean": float(sum(gray_std) / len(gray_std)),
        "foreground_fraction_min": min(foreground_fraction),
        "foreground_fraction_mean": float(sum(foreground_fraction) / len(foreground_fraction)),
        "preview": str(preview_path),
    }
    if diagnostics["foreground_gray_std_min"] < 10.0:
        raise RuntimeError(
            "Random vertex texture has insufficient intensity variation in at least one view: "
            f"gray_std_min={diagnostics['foreground_gray_std_min']:.3f}"
        )
    (render_dir / "texture_render_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(diagnostics, indent=2), flush=True)


def _barycentric_grid_for_triangle(grid_x, grid_y, triangle):
    import numpy as np
    a, b, c = triangle
    v0 = b - a
    v1 = c - a
    v2x = grid_x - a[0]
    v2y = grid_y - a[1]
    d00 = float(np.dot(v0, v0))
    d01 = float(np.dot(v0, v1))
    d11 = float(np.dot(v1, v1))
    d20 = v2x * v0[0] + v2y * v0[1]
    d21 = v2x * v1[0] + v2y * v1[1]
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-12:
        return None
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return np.stack([u, v, w], axis=2)


def _render_vertex_texture_view_cpu_perspective(*, mesh, camera, vertex_texture_rgb):
    """Software z-buffer with perspective-correct interpolation of shared vertex RGB."""
    import math
    import numpy as np

    width, height = camera.image_size
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=bool)
    depth = np.full((height, width), np.inf, dtype=np.float64)

    world_vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertex_rgb = np.asarray(vertex_texture_rgb, dtype=np.float64)
    if vertex_rgb.shape != (world_vertices.shape[0], 3):
        raise ValueError(
            f"vertex_texture_rgb must have shape {(world_vertices.shape[0], 3)}, got {vertex_rgb.shape}"
        )
    pixels, camera_z = camera.project(world_vertices)
    pixels = np.asarray(pixels, dtype=np.float64)
    camera_z = np.asarray(camera_z, dtype=np.float64)

    for face in faces:
        zf = camera_z[face]
        if np.any(zf <= 1e-8):
            continue
        pts = pixels[face]
        min_x = max(0, int(math.floor(float(np.min(pts[:, 0])))))
        max_x = min(width - 1, int(math.ceil(float(np.max(pts[:, 0])))))
        min_y = max(0, int(math.floor(float(np.min(pts[:, 1])))))
        max_y = min(height - 1, int(math.ceil(float(np.max(pts[:, 1])))))
        if min_x > max_x or min_y > max_y:
            continue

        xs = np.arange(min_x, max_x + 1, dtype=np.float64) + 0.5
        ys = np.arange(min_y, max_y + 1, dtype=np.float64) + 0.5
        grid_x, grid_y = np.meshgrid(xs, ys)
        bary = _barycentric_grid_for_triangle(grid_x, grid_y, pts)
        if bary is None:
            continue
        inside = np.all(bary >= -1e-8, axis=2)
        if not np.any(inside):
            continue

        inv_z = 1.0 / zf
        reciprocal_depth = np.tensordot(bary, inv_z, axes=([2], [0]))
        valid = inside & (reciprocal_depth > 1e-12)
        if not np.any(valid):
            continue
        fragment_z = np.full_like(reciprocal_depth, np.inf)
        fragment_z[valid] = 1.0 / reciprocal_depth[valid]

        depth_patch = depth[min_y : max_y + 1, min_x : max_x + 1]
        update = valid & (fragment_z < depth_patch)
        if not np.any(update):
            continue

        bary_update = bary[update]
        denom_update = reciprocal_depth[update]
        perspective_weights = (bary_update * inv_z[None, :]) / denom_update[:, None]
        triangle_rgb = vertex_rgb[face]
        colors = perspective_weights @ triangle_rgb
        colors = np.clip(np.rint(colors), 0.0, 255.0).astype(np.uint8)

        depth_patch[update] = fragment_z[update]
        mask_patch = mask[min_y : max_y + 1, min_x : max_x + 1]
        rgb_patch = rgb[min_y : max_y + 1, min_x : max_x + 1]
        mask_patch[update] = True
        rgb_patch[update] = colors

    return rgb, mask, depth


def _make_texture_preview(image_paths, output_path: Path) -> None:
    from PIL import Image, ImageOps, ImageDraw

    thumb = 240
    columns = 4
    rows = (len(image_paths) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * thumb, rows * thumb), (32, 32, 32))
    draw = ImageDraw.Draw(canvas)
    for index, image_path in enumerate(image_paths):
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((thumb - 8, thumb - 24))
        cell = Image.new("RGB", (thumb, thumb), (20, 20, 20))
        x = (thumb - image.width) // 2
        y = max(4, (thumb - 20 - image.height) // 2)
        cell.paste(image, (x, y))
        draw_cell = ImageDraw.Draw(cell)
        draw_cell.text((6, thumb - 18), f"view {index:02d}", fill=(230, 230, 230))
        col = index % columns
        row = index // columns
        canvas.paste(cell, (col * thumb, row * thumb))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def count_obj_vertices_faces(path: Path) -> tuple[int, int]:
    vertices = 0
    faces = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("v "):
                vertices += 1
            elif line.startswith("f "):
                faces += 1
    return vertices, faces

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
    dataset_payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    gt_mesh_rel = dataset_payload.get("mesh_path", "mesh.obj")
    gt_mesh_path = (dataset_path.parent / gt_mesh_rel).resolve()
    require_file(gt_mesh_path, "textured-render GT mesh copy")

    model_dir = output_root / "models" / model_id
    scene_dir = model_dir / "scene"
    coarse_obj = model_dir / "coarse.obj"
    coarse_ply = model_dir / "coarse.ply"
    diagnostics_path = model_dir / "diagnostics.json"

    if model_dir.exists() and force:
        shutil.rmtree(model_dir)
    elif coarse_obj.exists():
        print(f"[skip] {model_id}: {coarse_obj} already exists", flush=True)
        final_mesh_vertices, final_mesh_faces = count_obj_vertices_faces(coarse_obj)
        return {
            "script_version": SCRIPT_VERSION,
            "model_id": model_id,
            "status": "skipped_existing",
            "coarse_obj": str(coarse_obj),
            "final_mesh_vertices": final_mesh_vertices,
            "final_mesh_faces": final_mesh_faces,
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
            "guided_matching": True,
            "tri_ignore_two_view_tracks": False,
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
    sparse_quality = read_colmap_points3d_quality(points_txt)
    sparse_gt_surface = surface_distance_diagnostic(
        mlr_python=mlr_python,
        downstream_root=downstream_root,
        source_path=points_txt,
        source_kind="colmap_points3d",
        gt_mesh_path=gt_mesh_path,
        output_path=model_dir / "sparse_to_gt_surface.json",
    )
    print(
        "COLMAP triangulation diagnostics: "
        f"verified_pairs={db_diag['verified_matched_pairs']}, "
        f"inlier_matches={db_diag['total_inlier_matches']}, "
        f"triangulated_points={num_sparse_points}, "
        f"track2_fraction={sparse_quality.get('track_length_2_fraction')}, "
        f"reproj_median={sparse_quality.get('reprojection_error_median')}, "
        f"sparse_gt_median={sparse_gt_surface.get('distance_median')}",
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
        "guided_matching": True,
        "tri_ignore_two_view_tracks": False,
        "num_sparse_points": num_sparse_points,
        "sparse_quality": sparse_quality,
        "sparse_gt_surface": sparse_gt_surface,
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
    dense_ply = scene_dir / "scene_dense.ply"
    if dense_ply.is_file():
        dense_gt_surface = surface_distance_diagnostic(
            mlr_python=mlr_python,
            downstream_root=downstream_root,
            source_path=dense_ply,
            source_kind="ply",
            gt_mesh_path=gt_mesh_path,
            output_path=model_dir / "dense_to_gt_surface.json",
        )
    else:
        dense_gt_surface = {"status": "missing", "source": str(dense_ply)}
    diagnostics["dense_ply"] = str(dense_ply)
    diagnostics["dense_gt_surface"] = dense_gt_surface
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    print(
        "OpenMVS dense diagnostic: "
        f"points={dense_gt_surface.get('source_points')}, "
        f"gt_median={dense_gt_surface.get('distance_median')}, "
        f"gt_p95={dense_gt_surface.get('distance_p95')}",
        flush=True,
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
    if not coarse_ply.is_file():
        diagnostics["status"] = "failed_openmvs_no_mesh"
        diagnostics["coarse_ply"] = str(coarse_ply)
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
        raise RuntimeError(
            f"OpenMVS ReconstructMesh did not create {coarse_ply}. "
            f"See {diagnostics_path}; inspect the DensifyPointCloud/ReconstructMesh log above."
        )

    convert_ply_to_obj(
        mlr_python=mlr_python,
        downstream_root=downstream_root,
        ply_path=coarse_ply,
        obj_path=coarse_obj,
    )
    require_file(coarse_obj, "coarse OBJ")
    final_mesh_vertices, final_mesh_faces = count_obj_vertices_faces(coarse_obj)
    final_gt_surface = surface_distance_diagnostic(
        mlr_python=mlr_python,
        downstream_root=downstream_root,
        source_path=coarse_obj,
        source_kind="obj",
        gt_mesh_path=gt_mesh_path,
        output_path=model_dir / "final_mesh_to_gt_surface.json",
    )

    diagnostics.update(
        {
            "status": "ok",
            "known_camera_sparse_backup": str(sparse_known_cameras),
            "openmvs_sparse": str(sparse_seed),
            "scene_mvs": str(scene_mvs),
            "dense_mvs": str(dense_mvs),
            "coarse_ply": str(coarse_ply),
            "coarse_obj": str(coarse_obj),
            "final_mesh_vertices": final_mesh_vertices,
            "final_mesh_faces": final_mesh_faces,
            "final_mesh_gt_surface": final_gt_surface,
        }
    )
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    print(json.dumps(diagnostics, indent=2), flush=True)
    return diagnostics



def main() -> None:
    # The parent process can be any Python. Heavy renderer imports are isolated
    # in the MLR Python child process selected by --mlr-python.
    if len(sys.argv) >= 2 and sys.argv[1] == "--_render-stage":
        _render_stage_cli(sys.argv[2:])
        return

    parser = argparse.ArgumentParser(
        description=(
            "End-to-end Sofa50 test: 48-view overlapping-ring render with frozen V5 vertex texture -> "
            "COLMAP known-pose triangulation -> OpenMVS densification/mesh. "
            f"Script {SCRIPT_VERSION}."
        )
    )
    parser.add_argument(
        "--sofa-root",
        type=Path,
        default=Path("~/sofa_mesh/sofa50"),
        help="Canonical Sofa50 root containing <model-id>/mesh.obj and selection.json.",
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
        help=(
            "Textured 48-view datasets. Default: "
            "<refinement-root>/openmvs_texture_test_v6_48view/rendered"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "COLMAP/OpenMVS outputs. Default: "
            "<refinement-root>/openmvs_texture_test_v6_48view/reconstruction"
        ),
    )
    parser.add_argument(
        "--mlr-python",
        type=Path,
        default=Path("~/miniconda3/envs/test/bin/python"),
        help="Python interpreter containing the MLR package + Pillow/Numpy; no EGL/CUDA required for texture rendering.",
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
    )
    parser.add_argument("--no-colmap-runtime-lib-dir", action="store_true")
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
            "Sofa model ID. Repeat for several models. If omitted, uses "
            f"{DEFAULT_TEST_MODEL}."
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all IDs in Sofa50 selection.json (not recommended for the first test).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=1920,
        help="Square render size; 1920 reproduces the current OpenMVS input setup.",
    )
    parser.add_argument(
        "--texture-smoothing-steps",
        type=int,
        default=2,
        help=(
            "Graph-smoothing iterations for the random per-vertex texture. "
            "Default 2 keeps local feature structure while removing hard pixel/triangle-scale noise."
        ),
    )
    parser.add_argument("--texture-seed", type=int, default=17)
    parser.add_argument(
        "--resolution-level",
        type=int,
        default=2,
        help="OpenMVS DensifyPointCloud resolution level (kept equal to the current baseline).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete the selected textured render and reconstruction outputs before rebuilding.",
    )
    args = parser.parse_args()

    sofa_root = expand(args.sofa_root)
    refinement_root = expand(args.refinement_root)
    downstream_root = expand(args.downstream_root)
    render_root = (
        expand(args.render_root)
        if args.render_root
        else refinement_root / "openmvs_texture_test_v6_48view" / "rendered"
    )
    output_root = (
        expand(args.output_root)
        if args.output_root
        else refinement_root / "openmvs_texture_test_v6_48view" / "reconstruction"
    )
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
    if not sofa_root.is_dir():
        raise FileNotFoundError(f"Sofa50 root not found: {sofa_root}")
    if not downstream_root.is_dir():
        raise FileNotFoundError(f"Downstream repository not found: {downstream_root}")
    if args.image_size < 64:
        raise ValueError("--image-size must be >= 64")
    if args.texture_smoothing_steps < 0:
        raise ValueError("--texture-smoothing-steps must be non-negative")

    # Smoke-test the exact COLMAP binary/environment before rendering.
    colmap_env = os.environ.copy()
    if colmap_runtime_lib_dir is not None:
        colmap_env = with_prepend_path(colmap_env, "LD_LIBRARY_PATH", colmap_runtime_lib_dir)
    run([str(colmap_bin), "-h"], env=colmap_env)

    model_ids = resolve_sofa_model_ids(sofa_root, args.model_id, args.all)
    print(f"Processing {len(model_ids)} textured OpenMVS test model(s): {model_ids}", flush=True)
    print(
        "Fixed reconstruction settings: "
        "SIFT peak=0.002, guided_matching=1, tri_ignore_two_view_tracks=0, "
        f"OpenMVS resolution_level={args.resolution_level}",
        flush=True,
    )

    results: list[dict[str, object]] = []
    script_path = Path(__file__).resolve()
    for index, model_id in enumerate(model_ids, start=1):
        print(f"\n[{index}/{len(model_ids)}] {model_id}", flush=True)
        mesh_path = sofa_mesh_path(sofa_root, model_id)
        render_dir = render_root / model_id
        render_info = render_textured_dataset(
            script_path=script_path,
            mlr_python=mlr_python,
            downstream_root=downstream_root,
            mesh_path=mesh_path,
            render_dir=render_dir,
            image_size=args.image_size,
            texture_smoothing_steps=args.texture_smoothing_steps,
            texture_seed=args.texture_seed,
            force=args.force,
        )
        result = reconstruct_one(
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
        result["source_sofa_mesh"] = str(mesh_path)
        result["texture_render"] = render_info
        results.append(result)
        # Re-write the per-model diagnostics with the texture provenance included.
        diagnostics_path = output_root / "models" / model_id / "diagnostics.json"
        if diagnostics_path.parent.is_dir():
            diagnostics_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "summary.json"
    summary_payload = {
        "script_version": SCRIPT_VERSION,
        "experiment": "sofa50_48view_overlapping_rings_smoothed_random_vertex_texture_openmvs",
        "fixed_settings": {
            "image_size": args.image_size,
            "views": 48,
            "trajectory": "three_staggered_rings_16_each_v1",
            "elevations_degrees": [-25.0, 0.0, 25.0],
            "azimuth_step_degrees": 22.5,
            "middle_ring_azimuth_offset_degrees": 11.25,
            "fov_degrees": 90.0,
            "camera_radius_scale": 1.8,
            "texture_smoothing_steps": args.texture_smoothing_steps,
            "texture_seed": args.texture_seed,
            "sift_peak_threshold": SIFT_PEAK_THRESHOLD,
            "guided_matching": True,
            "tri_ignore_two_view_tracks": False,
            "resolution_level": args.resolution_level,
        },
        "results": results,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
