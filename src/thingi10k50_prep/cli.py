from __future__ import annotations

import argparse

from .config import load_config


def prepare_main() -> None:
    parser = argparse.ArgumentParser(description="Prepare 50-model Thingi10K dataset")
    parser.add_argument("--config", default="configs/thingi10k50.yaml")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    from .prepare import prepare_dataset

    cfg = load_config(args.config)
    if args.output_root is not None:
        cfg = type(cfg)(**{**cfg.__dict__, "output_root": args.output_root})
    if args.seed is not None:
        cfg = type(cfg)(**{**cfg.__dict__, "seed": args.seed})
    if args.force:
        cfg = type(cfg)(**{**cfg.__dict__, "force": True})
    prepare_dataset(cfg)


def validate_main() -> None:
    parser = argparse.ArgumentParser(description="Validate prepared Thingi10K50 dataset")
    parser.add_argument("--data-root", default="data/thingi10k50")
    args = parser.parse_args()
    from .validate import validate_dataset

    validate_dataset(args.data_root)


def smoke_main() -> None:
    parser = argparse.ArgumentParser(description="Run data smoke test on prepared Thingi10k50")
    parser.add_argument("--data-root", default="data/thingi10k50")
    args = parser.parse_args()
    from .smoke import run_smoke_test

    run_smoke_test(args.data_root)
