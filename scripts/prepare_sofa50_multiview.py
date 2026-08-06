from __future__ import annotations

import argparse
from pathlib import Path

from sofa50_refinement.multiview import prepare_multiview_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render a Sofa50 trial or the full final dataset and prepare downstream-compatible "
            "GT-query training and expanded-query inference manifests."
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
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--backend", choices=("opengl", "cpu", "cuda"), default="opengl")
    parser.add_argument(
        "--full-50",
        action="store_true",
        help="Require and process the complete 50-sample dataset instead of a 2-sample trial.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    prepare_multiview_dataset(
        args.refinement_root,
        args.downstream_root,
        args.output_root,
        backend=args.backend,
        force=args.force,
        expected_count=50 if args.full_50 else 2,
    )


if __name__ == "__main__":
    main()
