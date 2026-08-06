from __future__ import annotations

import argparse
from pathlib import Path

from sofa50_refinement.multiview import prepare_multiview_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Render downstream-compatible 3D-FUTURE-2000 data")
    parser.add_argument("--data-root", type=Path, default=Path("~/future2000"))
    parser.add_argument(
        "--downstream-root",
        type=Path,
        default=Path("~/multiview-laplacian-refinement"),
    )
    parser.add_argument("--backend", choices=("opengl", "cpu", "cuda"), default="opengl")
    parser.add_argument("--expected-count", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = args.data_root.expanduser().resolve()
    prepare_multiview_dataset(
        root / "inference",
        args.downstream_root,
        args.output_root.expanduser().resolve() if args.output_root else root / "multiview_960",
        backend=args.backend,
        force=args.force,
        expected_count=args.expected_count,
        dataset_name="3d_future_2000",
        full_forward_validation=False,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
