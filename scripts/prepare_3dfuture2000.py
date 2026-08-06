from __future__ import annotations

import argparse
import json
from pathlib import Path

from future2000_prep import prepare_future2000, prepare_future2000_inference


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a balanced 2000-model 3D-FUTURE dataset")
    parser.add_argument("--data-root", type=Path, default=Path("~/future2000"))
    parser.add_argument(
        "--downloads",
        type=Path,
        default=Path("~/sofa_mesh/downloads"),
    )
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument(
        "--target-max-extent",
        type=float,
        default=1.0,
        help="Maximum normalized GT bounding-box edge; 1.0 fits the downstream cube cameras",
    )
    parser.add_argument("--stage", choices=("all", "gt", "inference"), default="all")
    parser.add_argument("--force-inference", action="store_true")
    args = parser.parse_args()
    root = args.data_root.expanduser().resolve()
    result = {}
    if args.stage in {"all", "gt"}:
        result["gt"] = prepare_future2000(
            args.downloads,
            root,
            count=args.count,
            seed=args.seed,
            target_max_extent=args.target_max_extent,
        )
    if args.stage in {"all", "inference"}:
        result["inference"] = prepare_future2000_inference(
            root / "gt",
            root / "inference",
            force=args.force_inference,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
