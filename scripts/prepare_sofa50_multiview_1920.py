from __future__ import annotations

import argparse
from pathlib import Path

from sofa50_refinement.multiview_1920 import prepare_multiview_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render the complete Sofa50 dataset at 1920x1920 and prepare "
            "downstream-compatible GT-query / expanded-query manifests."
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
        "--output-root",
        type=Path,
        default=Path("~/sofa_mesh/sofa50_refinement/multiview_1920"),
    )
    parser.add_argument(
        "--backend",
        choices=("opengl", "cpu", "cuda"),
        default="opengl",
    )
    parser.add_argument(
        "--full-50",
        action="store_true",
        help="Require and process all 50 Sofa samples; otherwise run the 2-sample trial.",
    )
    parser.add_argument(
        "--full-forward-validation",
        action="store_true",
        help=(
            "Decode 1920 images and run the downstream model during validation. "
            "Off by default because this is memory intensive and is not required "
            "for generating the 1920 render dataset."
        ),
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
        full_forward_validation=args.full_forward_validation,
    )


if __name__ == "__main__":
    main()
