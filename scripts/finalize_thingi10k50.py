from __future__ import annotations

import argparse
from pathlib import Path

from thingi10k50_prep.finalize import finalize_cached_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Complete cached Thingi10K geometry with views and .pt samples")
    parser.add_argument("--config", default="configs/thingi10k50.yaml", type=Path)
    args = parser.parse_args()
    finalize_cached_dataset(args.config)


if __name__ == "__main__":
    main()
