from __future__ import annotations

import argparse
import json
from pathlib import Path

from sofa50_refinement import prepare_refinement_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate leakage-free Sofa50 frozen-model inference queries and "
            "optional oracle diagnostics"
        )
    )
    parser.add_argument("--source-root", type=Path, default=Path("~/sofa_mesh/sofa50"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("~/sofa_mesh/sofa50_refinement")
    )
    parser.add_argument("--coarse-target-vertices", type=int, default=3500)
    parser.add_argument("--coarse-min-vertices", type=int, default=1000)
    parser.add_argument("--subdivision-steps", type=int, default=1, choices=(1,))
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--model-id",
        action="append",
        dest="model_ids",
        help="Process only this Sofa50 model ID; repeat for a trial subset",
    )
    args = parser.parse_args()

    result = prepare_refinement_dataset(
        source_root=args.source_root,
        output_root=args.output_root,
        coarse_target_vertices=args.coarse_target_vertices,
        coarse_min_vertices=args.coarse_min_vertices,
        subdivision_steps=args.subdivision_steps,
        seed=args.seed,
        force=args.force,
        model_ids=args.model_ids,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
