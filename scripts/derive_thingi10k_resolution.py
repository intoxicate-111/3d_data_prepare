from __future__ import annotations

import argparse
from pathlib import Path

from thingi10k50_prep.derive import derive_resized_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive a standalone lower-resolution dataset from prepared Thingi10K data"
    )
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--height", required=True, type=int)
    args = parser.parse_args()
    derive_resized_dataset(args.source_root, args.output_root, args.width, args.height)


if __name__ == "__main__":
    main()
