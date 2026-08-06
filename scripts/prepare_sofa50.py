from __future__ import annotations

import argparse
import json
from pathlib import Path

from sofa50_prep import prepare_sofa50


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare 50 clean, normalized Sofa meshes from official 3D-FUTURE data"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--source-up-axis", choices=("x", "y", "z"), default="y")
    parser.add_argument("--target-max-extent", type=float, default=2.0)
    parser.add_argument("--target-faces", type=int, default=40_000)
    parser.add_argument("--max-faces", type=int, default=50_000)
    args = parser.parse_args()
    result = prepare_sofa50(
        data_root=args.data_root,
        count=args.count,
        seed=args.seed,
        source_up_axis=args.source_up_axis,
        target_max_extent=args.target_max_extent,
        target_faces=args.target_faces,
        max_faces=args.max_faces,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
