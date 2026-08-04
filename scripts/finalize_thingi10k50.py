from __future__ import annotations

import argparse
import importlib.metadata
import platform
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw
from pandas.errors import EmptyDataError


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _read_csv_allow_empty(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def _contact_sheet(root: Path, manifest: pd.DataFrame) -> None:
    previews: list[tuple[int, Image.Image]] = []
    for file_id in manifest["file_id"].astype(int):
        image = Image.open(root / "models" / str(file_id) / "preview.png").convert("RGB")
        image.thumbnail((480, 180))
        previews.append((file_id, image.copy()))

    columns = 5
    cell_width, cell_height = 480, 210
    rows = (len(previews) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (file_id, image) in enumerate(previews):
        x = (index % columns) * cell_width
        y = (index // columns) * cell_height
        sheet.paste(image, (x, y + 24))
        draw.text((x + 8, y + 5), str(file_id), fill="black")
    sheet.save(root / "reports" / "contact_sheet.png", optimize=True)


def finalize(root: Path, config: Path) -> None:
    manifest = pd.read_csv(root / "manifest.csv")
    rejected = pd.read_csv(root / "rejected_models.csv")
    failed = _read_csv_allow_empty(root / "failed_models.csv")
    shutil.copy2(config, root / "config.yaml")
    _contact_sheet(root, manifest)

    reasons = Counter(rejected.get("reason", pd.Series(dtype=str)).dropna().astype(str))
    reason_lines = "\n".join(f"  - `{reason}`: {count}" for reason, count in reasons.most_common())
    empty_views = sum(not any((root / "models" / str(fid) / "views").iterdir()) for fid in manifest["file_id"])
    report = f"""# Preparation report

- Environment: Conda `test`, Python `{platform.python_version()}`
- Packages: thingi10k `{_version('thingi10k')}`, NumPy `{_version('numpy')}`, SciPy `{_version('scipy')}`, pandas `{_version('pandas')}`, trimesh `{_version('trimesh')}`, fast-simplification `{_version('fast-simplification')}`, rtree `{_version('rtree')}`, networkx `{_version('networkx')}`
- Source: official Thingi10K `npz` variant
- Cache: `/home/zhou_c_WMGDS.WMG.WARWICK.AC.UK/thingi10k50/data/.cache/thingi10k`
- Output: `{root}`
- Seed: `20260804`
- Candidates accepted: `{len(pd.read_csv(root / 'selection_candidates.csv'))}`
- Candidates rejected: `{len(rejected)}`
- Final processing failures: `{len(failed)}`
- Final models: `{len(manifest)}` (`{(manifest.split == 'train').sum()}` train / `{(manifest.split == 'val').sum()}` validation / `{(manifest.split == 'test').sum()}` test)
- Original vertices: `{manifest.original_vertices.min()}` to `{manifest.original_vertices.max()}`
- Original faces: `{manifest.original_faces.min()}` to `{manifest.original_faces.max()}`
- Expanded vertices: `{manifest.expanded_vertices.min()}` to `{manifest.expanded_vertices.max()}`
- Expanded faces: `{manifest.expanded_faces.min()}` to `{manifest.expanded_faces.max()}`
- Mean projection distance over models: `{manifest.distance_mean.mean():.9f}`
- Worst per-model mean projection distance: `{manifest.distance_mean.max():.9f}`
- Worst individual projection distance: `{manifest.distance_max.max():.9f}`

## Rejection reasons

{reason_lines or '  - None'}

## Reproduction

```bash
PYTHONPATH=src ~/miniconda3/bin/conda run --no-capture-output -n test \\
  python scripts/prepare_thingi10k50.py --config configs/thingi10k50.yaml
PYTHONPATH=src ~/miniconda3/bin/conda run --no-capture-output -n test \\
  python scripts/validate_thingi10k50.py --data-root ~/thingi10k50/sample
PYTHONPATH=src ~/miniconda3/bin/conda run --no-capture-output -n test \\
  python scripts/smoke_test_thingi10k50.py --data-root ~/thingi10k50/sample
```

## Deviation

- `{empty_views}` model `views/` directories are empty. No existing experiment repository or compatible renderer was present, so geometry, projection targets, and Laplacian inputs were prepared, but fallback RGB/depth/mask/camera rendering was not fabricated without a consumer-defined camera convention.
"""
    (root / "reports" / "preparation_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize Thingi10K50 packaging artifacts")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--config", default="configs/thingi10k50.yaml", type=Path)
    args = parser.parse_args()
    finalize(args.data_root, args.config)


if __name__ == "__main__":
    main()
