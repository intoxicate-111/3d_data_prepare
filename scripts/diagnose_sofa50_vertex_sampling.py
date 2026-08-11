from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import trimesh


ROLE_TO_FILENAME = {
    "gt": "gt_mesh.obj",
    "expanded": "expanded_initial_raw.obj",
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load_mesh(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if len(geometries) != 1:
            raise ValueError(
                f"Expected one mesh in {path}, found {len(geometries)} geometries."
            )
        loaded = geometries[0]
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type for {path}: {type(loaded)!r}")
    if loaded.vertices.ndim != 2 or loaded.vertices.shape[1] != 3:
        raise ValueError(f"Invalid vertex array in {path}: {loaded.vertices.shape}")
    if loaded.faces.ndim != 2 or loaded.faces.shape[1] != 3:
        raise ValueError(f"Expected triangular faces in {path}: {loaded.faces.shape}")
    return loaded


def _unique_edges(faces: np.ndarray) -> np.ndarray:
    edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
    ).astype(np.int64, copy=False)
    edges.sort(axis=1)
    return np.unique(edges, axis=0)


def _safe_percentile(values: np.ndarray, q: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.percentile(finite, q))


def _gini_nonnegative(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x) & (x >= 0.0)]
    if x.size == 0 or np.all(x == 0.0):
        return float("nan")
    x = np.sort(x)
    n = x.size
    index = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(index * x) / (n * np.sum(x))) - (n + 1.0) / n)


def _rankdata_average(values: np.ndarray) -> np.ndarray:
    # Minimal scipy-free average-rank implementation.
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = 0.5 * ((start + 1) + end)
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[mask], dtype=np.float64)
    y = np.asarray(y[mask], dtype=np.float64)
    if x.size < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    rx = _rankdata_average(x)
    ry = _rankdata_average(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def compute_vertex_sampling_metrics(mesh: trimesh.Trimesh) -> pd.DataFrame:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    n_vertices = len(vertices)

    tri = vertices[faces]
    face_area = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
    )

    represented_area = np.zeros(n_vertices, dtype=np.float64)
    incident_faces = np.zeros(n_vertices, dtype=np.int64)
    one_third_area = face_area / 3.0
    for corner in range(3):
        np.add.at(represented_area, faces[:, corner], one_third_area)
        np.add.at(incident_faces, faces[:, corner], 1)

    edges = _unique_edges(faces)
    edge_vec = vertices[edges[:, 1]] - vertices[edges[:, 0]]
    edge_len = np.linalg.norm(edge_vec, axis=1)
    edge_sum = np.zeros(n_vertices, dtype=np.float64)
    degree = np.zeros(n_vertices, dtype=np.int64)
    np.add.at(edge_sum, edges[:, 0], edge_len)
    np.add.at(edge_sum, edges[:, 1], edge_len)
    np.add.at(degree, edges[:, 0], 1)
    np.add.at(degree, edges[:, 1], 1)
    mean_edge_length = np.divide(
        edge_sum,
        degree,
        out=np.full(n_vertices, np.nan, dtype=np.float64),
        where=degree > 0,
    )

    # A local straight/flatness proxy: mean 1 - |n_i dot n_j| over unique one-ring edges.
    # Zero means neighboring vertex normals agree locally; larger values indicate stronger
    # normal change. abs(dot) reduces sensitivity to occasional winding sign flips.
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    normal_norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(
        normals,
        normal_norm,
        out=np.zeros_like(normals),
        where=normal_norm > 1e-12,
    )
    edge_dot = np.sum(normals[edges[:, 0]] * normals[edges[:, 1]], axis=1)
    edge_normal_change = 1.0 - np.clip(np.abs(edge_dot), 0.0, 1.0)
    normal_change_sum = np.zeros(n_vertices, dtype=np.float64)
    np.add.at(normal_change_sum, edges[:, 0], edge_normal_change)
    np.add.at(normal_change_sum, edges[:, 1], edge_normal_change)
    normal_variation = np.divide(
        normal_change_sum,
        degree,
        out=np.full(n_vertices, np.nan, dtype=np.float64),
        where=degree > 0,
    )

    positive_area = represented_area[represented_area > 0.0]
    mean_area = float(np.mean(positive_area)) if positive_area.size else float("nan")
    median_area = float(np.median(positive_area)) if positive_area.size else float("nan")
    area_over_mean = represented_area / mean_area if mean_area > 0.0 else np.full(n_vertices, np.nan)
    area_over_median = (
        represented_area / median_area if median_area > 0.0 else np.full(n_vertices, np.nan)
    )
    sampling_density_relative = np.divide(
        median_area,
        represented_area,
        out=np.full(n_vertices, np.nan, dtype=np.float64),
        where=represented_area > 0.0,
    )

    order = np.argsort(represented_area, kind="mergesort")
    area_percentile = np.empty(n_vertices, dtype=np.float64)
    if n_vertices > 1:
        area_percentile[order] = 100.0 * np.arange(n_vertices) / (n_vertices - 1)
    else:
        area_percentile[:] = 100.0

    return pd.DataFrame(
        {
            "vertex_index": np.arange(n_vertices, dtype=np.int64),
            "x": vertices[:, 0],
            "y": vertices[:, 1],
            "z": vertices[:, 2],
            "represented_area": represented_area,
            "area_over_mean": area_over_mean,
            "area_over_median": area_over_median,
            "sampling_density_relative": sampling_density_relative,
            "area_percentile": area_percentile,
            "mean_edge_length_h": mean_edge_length,
            "h_squared": mean_edge_length**2,
            "degree": degree,
            "incident_face_count": incident_faces,
            "normal_variation": normal_variation,
        }
    )


def summarize_metrics(
    model_id: str,
    role: str,
    mesh_path: Path,
    mesh: trimesh.Trimesh,
    frame: pd.DataFrame,
) -> dict[str, Any]:
    area = frame["represented_area"].to_numpy(dtype=np.float64)
    normal_variation = frame["normal_variation"].to_numpy(dtype=np.float64)
    positive = area[area > 0.0]
    total_area = float(np.sum(area))
    n = len(area)

    def top_share(frac: float) -> float:
        if n == 0 or total_area <= 0.0:
            return float("nan")
        k = max(1, int(math.ceil(frac * n)))
        return float(np.sort(area)[-k:].sum() / total_area)

    q20 = _safe_percentile(normal_variation, 20.0)
    q80 = _safe_percentile(normal_variation, 80.0)
    if not np.isfinite(q20) or not np.isfinite(q80) or q80 <= q20:
        flat = np.asarray([], dtype=np.float64)
        detailed = np.asarray([], dtype=np.float64)
    else:
        flat = area[np.isfinite(normal_variation) & (normal_variation <= q20)]
        detailed = area[np.isfinite(normal_variation) & (normal_variation >= q80)]
    flat_median = float(np.median(flat)) if flat.size else float("nan")
    detailed_median = float(np.median(detailed)) if detailed.size else float("nan")
    flat_to_detail_ratio = (
        flat_median / detailed_median
        if np.isfinite(flat_median) and np.isfinite(detailed_median) and detailed_median > 0.0
        else float("nan")
    )

    effective_vertex_count = (
        float(total_area**2 / np.sum(area**2)) if total_area > 0.0 and np.sum(area**2) > 0.0 else float("nan")
    )

    mean_area = float(np.mean(positive)) if positive.size else float("nan")
    median_area = float(np.median(positive)) if positive.size else float("nan")
    return {
        "model_id": model_id,
        "mesh_role": role,
        "mesh_path": str(mesh_path),
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "total_surface_area": total_area,
        "represented_area_mean": mean_area,
        "represented_area_median": median_area,
        "represented_area_p90": _safe_percentile(area, 90.0),
        "represented_area_p95": _safe_percentile(area, 95.0),
        "represented_area_p99": _safe_percentile(area, 99.0),
        "represented_area_max": float(np.max(area)),
        "max_over_median": float(np.max(area) / median_area) if median_area > 0.0 else float("nan"),
        "represented_area_cv": float(np.std(positive) / mean_area) if mean_area > 0.0 else float("nan"),
        "represented_area_gini": _gini_nonnegative(area),
        "top_1pct_vertices_surface_area_share": top_share(0.01),
        "top_10pct_vertices_surface_area_share": top_share(0.10),
        "top_20pct_vertices_surface_area_share": top_share(0.20),
        "vertices_area_gt_2x_mean_fraction": float(np.mean(area > (2.0 * mean_area))) if mean_area > 0.0 else float("nan"),
        "vertices_area_gt_4x_mean_fraction": float(np.mean(area > (4.0 * mean_area))) if mean_area > 0.0 else float("nan"),
        "effective_vertex_count_by_area": effective_vertex_count,
        "effective_vertex_fraction_by_area": effective_vertex_count / n if n > 0 else float("nan"),
        "area_vs_normal_variation_spearman": _spearman(area, normal_variation),
        "flat20_median_represented_area": flat_median,
        "detailed20_median_represented_area": detailed_median,
        "flat20_to_detailed20_area_ratio": flat_to_detail_ratio,
        "zero_area_vertices": int(np.sum(area <= 0.0)),
    }


def save_area_heatmap(
    mesh: trimesh.Trimesh,
    area: np.ndarray,
    path: Path,
    clip_low: float,
    clip_high: float,
) -> None:
    positive = area[area > 0.0]
    if positive.size == 0:
        raise ValueError("Cannot make heatmap: represented areas are all zero.")
    lo = float(np.percentile(positive, clip_low))
    hi = float(np.percentile(positive, clip_high))
    if hi <= lo:
        hi = lo + max(abs(lo), 1.0) * 1e-12
    safe_area = np.clip(area, max(lo, 1e-30), hi)
    log_lo = math.log10(max(lo, 1e-30))
    log_hi = math.log10(max(hi, 1e-30))
    normalized = (np.log10(safe_area) - log_lo) / max(log_hi - log_lo, 1e-12)
    rgba = (plt.colormaps["turbo"](normalized) * 255.0).astype(np.uint8)

    colored = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.faces),
        process=False,
    )
    colored.visual.vertex_colors = rgba
    path.parent.mkdir(parents=True, exist_ok=True)
    colored.export(path)


def save_histogram(frame: pd.DataFrame, path: Path, title: str) -> None:
    ratio = frame["area_over_median"].to_numpy(dtype=np.float64)
    ratio = ratio[np.isfinite(ratio) & (ratio > 0.0)]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(np.log10(ratio), bins=80)
    ax.axvline(0.0, linestyle="--", linewidth=1.0, label="median area")
    ax.axvline(math.log10(2.0), linestyle=":", linewidth=1.0, label="2x median")
    ax.axvline(math.log10(4.0), linestyle=":", linewidth=1.0, label="4x median")
    ax.set_xlabel("log10(represented area / median represented area)")
    ax.set_ylabel("vertex count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_area_vs_normal_variation(frame: pd.DataFrame, path: Path, title: str) -> None:
    area_ratio = frame["area_over_median"].to_numpy(dtype=np.float64)
    normal_variation = frame["normal_variation"].to_numpy(dtype=np.float64)
    mask = np.isfinite(area_ratio) & (area_ratio > 0.0) & np.isfinite(normal_variation)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(normal_variation[mask], np.log10(area_ratio[mask]), s=4, alpha=0.25)
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.set_xlabel("one-ring normal variation (lower = straighter/flatter)")
    ax.set_ylabel("log10(represented area / median)")
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _select_samples(manifest: dict[str, Any], requested_ids: list[str] | None) -> list[dict[str, Any]]:
    samples = manifest.get("samples", [])
    samples = [item for item in samples if item.get("status", "valid") == "valid"]
    if requested_ids:
        wanted = set(requested_ids)
        selected = [item for item in samples if str(item.get("model_id")) in wanted]
        missing = sorted(wanted - {str(item.get("model_id")) for item in selected})
        if missing:
            raise ValueError(f"Requested model IDs not found as valid samples: {missing}")
        return selected
    return samples


def _roles(mesh_role: str) -> list[str]:
    if mesh_role == "both":
        return ["gt", "expanded"]
    return [mesh_role]


def _finite_median(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else float("nan")


def write_report(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Sofa50 vertex-sampling diagnostic",
        "",
        "Interpretation: a vertex with large `represented_area` stands for a large surface patch,",
        "so high area concentration means sparse/non-uniform vertex sampling. The normal-variation",
        "comparison tests whether straighter/flatter regions systematically carry larger patches.",
        "",
    ]
    if summary.empty:
        lines.append("No meshes were processed.")
    else:
        for role, group in summary.groupby("mesh_role", sort=False):
            lines.extend(
                [
                    f"## {role}",
                    "",
                    f"Meshes: {len(group)}",
                    f"Median Gini: {_finite_median(group['represented_area_gini']):.4f}",
                    f"Median top-10% surface-area share: {_finite_median(group['top_10pct_vertices_surface_area_share']):.4f}",
                    f"Median effective vertex fraction: {_finite_median(group['effective_vertex_fraction_by_area']):.4f}",
                    f"Median flat20/detailed20 area ratio: {_finite_median(group['flat20_to_detailed20_area_ratio']):.4f}",
                    f"Median Spearman(area, normal variation): {_finite_median(group['area_vs_normal_variation_spearman']):.4f}",
                    "",
                    "Worst meshes by top-10% surface-area share:",
                    "",
                ]
            )
            worst = group.nlargest(min(10, len(group)), "top_10pct_vertices_surface_area_share")
            lines.append("| model_id | top10 share | Gini | effective fraction | flat/detail ratio | Spearman |")
            lines.append("|---|---:|---:|---:|---:|---:|")
            for _, row in worst.iterrows():
                lines.append(
                    f"| {row['model_id']} | {row['top_10pct_vertices_surface_area_share']:.4f} | "
                    f"{row['represented_area_gini']:.4f} | {row['effective_vertex_fraction_by_area']:.4f} | "
                    f"{row['flat20_to_detailed20_area_ratio']:.4f} | "
                    f"{row['area_vs_normal_variation_spearman']:.4f} |"
                )
            lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure Sofa50 per-vertex represented surface area to diagnose sparse/non-uniform "
            "mesh sampling, especially on straight/flat sofa regions."
        )
    )
    parser.add_argument(
        "--refinement-root",
        type=Path,
        default=Path("~/sofa_mesh/sofa50_refinement"),
        help="Sofa50 refinement root containing manifest.json and models/<id>/...",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Default: <refinement-root>/vertex_sampling_diagnostics",
    )
    parser.add_argument(
        "--mesh-role",
        choices=("gt", "expanded", "both"),
        default="gt",
        help="gt tests training-query tessellation; expanded tests current inference graph.",
    )
    parser.add_argument(
        "--model-id",
        action="append",
        dest="model_ids",
        help="Process only this model ID; repeat to select multiple models.",
    )
    parser.add_argument("--heatmap-clip-low", type=float, default=5.0)
    parser.add_argument("--heatmap-clip-high", type=float, default=99.0)
    args = parser.parse_args()

    refinement_root = args.refinement_root.expanduser().resolve()
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else refinement_root / "vertex_sampling_diagnostics"
    )
    manifest_path = refinement_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing refinement manifest: {manifest_path}")
    if not (0.0 <= args.heatmap_clip_low < args.heatmap_clip_high <= 100.0):
        raise ValueError("Heatmap percentiles must satisfy 0 <= low < high <= 100")

    manifest = _read_json(manifest_path)
    samples = _select_samples(manifest, args.model_ids)
    if not samples:
        raise ValueError("No valid Sofa50 samples selected.")

    output_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []

    for sample_index, sample in enumerate(samples, start=1):
        model_id = str(sample["model_id"])
        for role in _roles(args.mesh_role):
            mesh_path = refinement_root / "models" / model_id / ROLE_TO_FILENAME[role]
            if not mesh_path.is_file():
                raise FileNotFoundError(f"Missing {role} mesh for {model_id}: {mesh_path}")
            print(
                f"[{sample_index}/{len(samples)}] {model_id} role={role} loading {mesh_path}",
                flush=True,
            )
            mesh = _load_mesh(mesh_path)
            frame = compute_vertex_sampling_metrics(mesh)
            summary = summarize_metrics(model_id, role, mesh_path, mesh, frame)
            summaries.append(summary)

            mesh_out = output_root / model_id / role
            mesh_out.mkdir(parents=True, exist_ok=True)
            frame.to_csv(mesh_out / "per_vertex_sampling.csv", index=False)
            (mesh_out / "summary.json").write_text(
                json.dumps(summary, indent=2, allow_nan=True) + "\n", encoding="utf-8"
            )
            save_area_heatmap(
                mesh,
                frame["represented_area"].to_numpy(dtype=np.float64),
                mesh_out / "represented_area_heatmap.ply",
                args.heatmap_clip_low,
                args.heatmap_clip_high,
            )
            save_histogram(
                frame,
                mesh_out / "represented_area_hist.png",
                f"{model_id} {role}: represented-area distribution",
            )
            save_area_vs_normal_variation(
                frame,
                mesh_out / "area_vs_normal_variation.png",
                f"{model_id} {role}: sampling vs local straightness",
            )
            print(
                "  "
                f"top10_area_share={summary['top_10pct_vertices_surface_area_share']:.4f} "
                f"gini={summary['represented_area_gini']:.4f} "
                f"effective_fraction={summary['effective_vertex_fraction_by_area']:.4f} "
                f"flat/detail={summary['flat20_to_detailed20_area_ratio']:.4f} "
                f"rho(area,normal_variation)={summary['area_vs_normal_variation_spearman']:.4f}",
                flush=True,
            )

    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(output_root / "summary.csv", index=False)
    (output_root / "summary.json").write_text(
        json.dumps(summaries, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    write_report(summary_frame, output_root / "REPORT.md")

    print(
        json.dumps(
            {
                "status": "completed",
                "mesh_count": len(samples),
                "roles": _roles(args.mesh_role),
                "output_root": str(output_root),
                "summary_csv": str(output_root / "summary.csv"),
                "report": str(output_root / "REPORT.md"),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
