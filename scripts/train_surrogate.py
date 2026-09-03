#!/usr/bin/env python3
"""Train one surrogate from a stored dataset, or all three.

A thin wrapper over ``glof train`` for batch schedulers that prefer a script to a
console entry point:

    python scripts/train_surrogate.py --config configs/production.yaml --which fno
    python scripts/train_surrogate.py --config configs/production.yaml --which all \
        --set training.fno.epochs=300 runtime.device=cuda

Datasets must exist first (``glof dataset --config ... --which all``); this script
does not generate them, so a scheduler can stage data on cheap CPU nodes and train
on GPU nodes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glof_pipeline.cli import main  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="path to a configuration YAML")
    parser.add_argument("--which", choices=["mgn", "fno", "downscaler", "all"], default="all")
    parser.add_argument("--set", nargs="*", default=[], metavar="key.path=value")
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    argv = ["train", "--config", arguments.config, "--which", arguments.which]
    if arguments.set:
        argv += ["--set", *arguments.set]
    raise SystemExit(main(argv))
