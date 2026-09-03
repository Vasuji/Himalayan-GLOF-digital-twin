"""Command-line interface.

    glof info                                   # resolved backends and versions
    glof run       --config configs/toy.yaml    # full end-to-end run
    glof dataset   --config configs/toy.yaml --which swe
    glof train     --config configs/toy.yaml --which fno
    glof benchmark --config configs/toy.yaml    # routing benchmark from checkpoints

Any configuration key can be overridden inline:

    glof run --config configs/toy.yaml --set runtime.seed=3 training.fno.epochs=5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from glof_pipeline import __version__
from glof_pipeline.backends import backend_report
from glof_pipeline.config import Config, parse_cli_overrides
from glof_pipeline.utils.runtime import environment_report, get_logger, set_seed

LOGGER = get_logger("glof.cli")


def _load_config(arguments: argparse.Namespace) -> Config:
    overrides = parse_cli_overrides(arguments.set or [])
    config = Config.load(arguments.config, overrides=overrides)
    import logging

    level = getattr(logging, str(config.get("runtime.log_level", "INFO")).upper(), logging.INFO)
    get_logger("glof", level=level)
    return config


def _command_info(_: argparse.Namespace) -> int:
    payload = {
        "version": __version__,
        "backends": backend_report(),
        "environment": environment_report(),
    }
    print(json.dumps(payload, indent=2))
    return 0


def _command_run(arguments: argparse.Namespace) -> int:
    from glof_pipeline.pipeline import run_pipeline

    config = _load_config(arguments)
    manifest = run_pipeline(config, reuse_checkpoints=arguments.reuse_checkpoints)
    print(f"\nRun complete. Manifest: {manifest['manifest_path']}")
    breach = manifest["artifacts"].get("breach", {})
    if breach.get("breached"):
        print(
            f"Predicted {breach['mechanism']} breach: peak {breach['peak_discharge_m3_per_s']:.0f} m3/s, "
            f"{breach['released_volume_m3'] / 1e6:.1f} Mm3 released."
        )
    else:
        print("No breach predicted for this forcing ensemble.")
    return 0


def _command_dataset(arguments: argparse.Namespace) -> int:
    from glof_pipeline.datasets.builders import (
        build_downscaling_dataset,
        build_moraine_dataset,
        build_swe_dataset,
    )
    from glof_pipeline.terrain.mesh_builder import build_moraine_graph
    from glof_pipeline.terrain.synthetic_dem import build_synthetic_valley

    config = _load_config(arguments)
    set_seed(int(config.get("runtime.seed")))
    rng = np.random.default_rng(int(config.get("runtime.seed")))
    terrain = build_synthetic_valley(config.get("domain.synthetic"))

    if arguments.which in ("moraine", "all"):
        graph = build_moraine_graph(terrain, int(config.get("surrogates.mgn.neighbours", 8)))
        cfg = config.get("datasets.moraine")
        build_moraine_dataset(terrain, graph, config.get("moraine"), int(cfg["n_scenarios"]), rng, cfg["path"])
    if arguments.which in ("swe", "all"):
        cfg = config.get("datasets.swe")
        build_swe_dataset(
            terrain, config.get("routing"), config.get("breach"),
            int(cfg["n_scenarios"]), int(cfg["frames_per_scenario"]), int(cfg["resolution"]),
            rng, cfg["path"],
        )
    if arguments.which in ("downscaling", "all"):
        cfg = config.get("datasets.downscaling")
        build_downscaling_dataset(terrain, config.get("atmosphere"), int(cfg["n_scenarios"]), rng, cfg["path"])
    return 0


def _command_train(arguments: argparse.Namespace) -> int:
    from glof_pipeline.terrain.mesh_builder import build_moraine_graph
    from glof_pipeline.terrain.synthetic_dem import build_synthetic_valley
    from glof_pipeline.utils.io_helpers import load_npz

    config = _load_config(arguments)
    set_seed(int(config.get("runtime.seed")))
    device = str(config.get("runtime.device", "cpu"))
    checkpoint_dir = Path(config.get("runtime.checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    terrain = build_synthetic_valley(config.get("domain.synthetic"))
    reports: dict[str, dict] = {}

    if arguments.which in ("mgn", "all"):
        from glof_pipeline.training.train_mgn import train_mgn

        graph = build_moraine_graph(terrain, int(config.get("surrogates.mgn.neighbours", 8)))
        dataset = load_npz(config.get("datasets.moraine.path"))
        train_cfg = dict(config.get("training.mgn"))
        train_cfg["val_fraction"] = float(config.get("datasets.moraine.val_fraction", 0.2))
        _, report = train_mgn(
            dataset, graph, config.get("surrogates.mgn"), train_cfg, device=device,
            checkpoint=checkpoint_dir / "mgn_moraine.pt", seed=int(config.get("runtime.seed")),
        )
        reports["mgn"] = {k: v for k, v in report.items() if k != "history"}

    if arguments.which in ("fno", "all"):
        from glof_pipeline.training.train_fno import train_fno

        dataset = load_npz(config.get("datasets.swe.path"))
        _, report = train_fno(
            dataset, config.get("surrogates.fno"), config.get("training.fno"), device=device,
            checkpoint=checkpoint_dir / "fno_swe.pt", seed=int(config.get("runtime.seed")),
            val_fraction=float(config.get("datasets.swe.val_fraction", 0.25)),
        )
        reports["fno"] = {k: v for k, v in report.items() if k != "history"}

    if arguments.which in ("downscaler", "all"):
        from glof_pipeline.training.train_downscaler import train_downscaler

        dataset = load_npz(config.get("datasets.downscaling.path"))
        _, report, _ = train_downscaler(
            dataset, config.get("downscaling.toy"), config.get("training.downscaler"),
            device=device, checkpoint=checkpoint_dir / "downscaler.pt",
            seed=int(config.get("runtime.seed")),
            val_fraction=float(config.get("datasets.downscaling.val_fraction", 0.2)),
        )
        reports["downscaler"] = {k: v for k, v in report.items() if k != "history"}

    print(json.dumps(reports, indent=2, default=str))
    return 0


def _command_benchmark(arguments: argparse.Namespace) -> int:
    from glof_pipeline.pipeline import Pipeline

    config = _load_config(arguments)
    pipeline = Pipeline(config, reuse_checkpoints=True)
    pipeline.build_terrain()
    pipeline.run_atmosphere()
    pipeline.run_downscaling()
    pipeline.run_mass_balance()
    pipeline.run_moraine()
    pipeline.run_breach()
    pipeline.run_routing()
    report = pipeline.run_evaluation()
    if report and "comparisons" in report:
        from glof_pipeline.evaluate.benchmark import benchmark_table

        print(benchmark_table(report))
    else:
        print("No benchmark produced (no breach was predicted).")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="glof", description="Himalayan GLOF digital twin")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--config", required=True, type=Path, help="path to a YAML configuration")
        subparser.add_argument(
            "--set", nargs="*", metavar="KEY=VALUE", help="inline configuration overrides"
        )

    info = subparsers.add_parser("info", help="report resolved backends and package versions")
    info.set_defaults(handler=_command_info)

    run_parser = subparsers.add_parser("run", help="execute the full pipeline")
    add_common(run_parser)
    run_parser.add_argument(
        "--reuse-checkpoints", action="store_true", help="skip training when a checkpoint exists"
    )
    run_parser.set_defaults(handler=_command_run)

    dataset_parser = subparsers.add_parser("dataset", help="build training datasets")
    add_common(dataset_parser)
    dataset_parser.add_argument(
        "--which", choices=["moraine", "swe", "downscaling", "all"], default="all"
    )
    dataset_parser.set_defaults(handler=_command_dataset)

    train_parser = subparsers.add_parser("train", help="train a surrogate from a stored dataset")
    add_common(train_parser)
    train_parser.add_argument("--which", choices=["mgn", "fno", "downscaler", "all"], default="all")
    train_parser.set_defaults(handler=_command_train)

    benchmark_parser = subparsers.add_parser("benchmark", help="routing benchmark from checkpoints")
    add_common(benchmark_parser)
    benchmark_parser.set_defaults(handler=_command_benchmark)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        return int(arguments.handler(arguments))
    except (RuntimeError, ValueError, FileNotFoundError, KeyError) as error:
        LOGGER.error("%s: %s", type(error).__name__, error)
        return 1


if __name__ == "__main__":
    sys.exit(main())
