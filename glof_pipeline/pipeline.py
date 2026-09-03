"""End-to-end orchestration of the GLOF digital twin.

Stage graph
-----------
1. terrain          - build or load the valley, mesh the moraine
2. atmosphere       - initial conditions, ensemble forecast
3. downscaling      - train (or load) the downscaler, generate 1 km fields
4. mass balance     - catchment melt and lake filling per ensemble member
5. moraine          - train (or load) the MeshGraphNet, evaluate stability, breach probability
6. breach           - Froehlich geometry, outflow hydrograph for the representative member
7. routing          - finite-volume solver and FNO surrogate on the same event
8. assimilation     - twin experiment demonstrating the observing network's value
9. evaluation       - measured benchmark and verification metrics
10. products        - figures, USD scene, run manifest

Every stage records its wall time and writes its outputs under
``runtime.output_dir``; the manifest ties them to the configuration hash, the
random seed and the resolved backends, which is what makes a run reproducible.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from glof_pipeline.assimilation.enkf import EnsembleKalmanFilter, StateSpec
from glof_pipeline.assimilation.sensors import build_sensor_network, synthesise_observations
from glof_pipeline.atmospheric.downscaling import (
    catchment_mean_series,
    downscale_forecast,
    lapse_correct,
)
from glof_pipeline.atmospheric.fetcher import fetch_initial_conditions
from glof_pipeline.atmospheric.forecaster import run_forecast
from glof_pipeline.backends import backend_report, resolve_tier
from glof_pipeline.config import Config
from glof_pipeline.datasets.builders import (
    build_downscaling_dataset,
    build_moraine_dataset,
    build_swe_dataset,
)
from glof_pipeline.evaluate.benchmark import benchmark_routing, benchmark_table, timing_breakdown
from glof_pipeline.hydrology.flood_router import route_flood
from glof_pipeline.hydrology.ice_mechanics import assess_moraine
from glof_pipeline.physics.breach import simulate_breach
from glof_pipeline.physics.mass_balance import integrate_catchment
from glof_pipeline.surrogates.fno_swe import FloodOperator
from glof_pipeline.surrogates.mgn_moraine import MoraineOperator
from glof_pipeline.terrain.dem_io import load_dem
from glof_pipeline.terrain.mesh_builder import build_moraine_graph
from glof_pipeline.terrain.synthetic_dem import (
    ValleyTerrain,
    build_synthetic_valley,
    delineate_from_dem,
)
from glof_pipeline.training.train_downscaler import train_downscaler
from glof_pipeline.training.train_fno import train_fno
from glof_pipeline.training.train_mgn import train_mgn
from glof_pipeline.utils.io_helpers import write_json
from glof_pipeline.utils.runtime import environment_report, get_logger, set_seed

LOGGER = get_logger("glof.pipeline")


class Pipeline:
    """Stateful runner so stages can be executed individually or as a whole."""

    def __init__(self, config: Config, reuse_checkpoints: bool = False):
        self.config = config
        self.tier = resolve_tier(str(config.get("runtime.tier")))
        self.device = str(config.get("runtime.device", "cpu"))
        self.seed = int(config.get("runtime.seed"))
        self.output_dir = Path(config.get("runtime.output_dir"))
        self.checkpoint_dir = Path(config.get("runtime.checkpoint_dir", "checkpoints"))
        self.reuse_checkpoints = reuse_checkpoints
        self.rng = np.random.default_rng(self.seed)
        self.timings: dict[str, float] = {}
        self.artifacts: dict[str, Any] = {}
        set_seed(self.seed)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # -- helpers ------------------------------------------------------------
    def _stage(self, name: str):
        pipeline = self

        class _Stage:
            def __enter__(self):
                LOGGER.info("--- stage: %s", name)
                self._start = time.perf_counter()
                return self

            def __exit__(self, *exc: object) -> None:
                pipeline.timings[name] = time.perf_counter() - self._start
                LOGGER.info("--- stage %s finished in %.2f s", name, pipeline.timings[name])

        return _Stage()

    # -- stages -------------------------------------------------------------
    def build_terrain(self) -> ValleyTerrain:
        with self._stage("terrain"):
            dem_path = self.config.get("domain.dem_path", None)
            if dem_path and Path(dem_path).is_file():
                z, dx, meta = load_dem(dem_path)
                LOGGER.info("Delineating from %s: %s raster at %.1f m", meta["source"], z.shape, dx)
                terrain = delineate_from_dem(
                    z, dx,
                    min_lake_depth_m=float(self.config.get("domain.min_lake_depth_m", 2.0)),
                    min_lake_area_m2=float(self.config.get("domain.min_lake_area_m2", 1.0e4)),
                    moraine_band_m=float(self.config.get("domain.moraine_band_m", 500.0)),
                    freeboard_m=float(self.config.get("domain.synthetic.freeboard_m")),
                    masks=self.config.get("domain.masks", None),
                )
            else:
                terrain = build_synthetic_valley(self.config.get("domain.synthetic"))
            graph = build_moraine_graph(terrain, int(self.config.get("surrogates.mgn.neighbours", 8)))
        self.terrain, self.graph = terrain, graph
        self.artifacts["terrain"] = {
            "shape": list(terrain.shape),
            "dx_m": terrain.dx,
            "crest_elevation_m": terrain.crest_elevation,
            "initial_lake_level_m": terrain.initial_lake_level,
            "initial_lake_volume_m3": terrain.meta["initial_lake_volume_m3"],
            "initial_lake_area_km2": terrain.meta["initial_lake_area_m2"] / 1e6,
            "moraine_nodes": graph.num_nodes,
            "moraine_edges": graph.num_edges,
        }
        return terrain

    def run_atmosphere(self):
        with self._stage("atmosphere"):
            atmospheric_cfg = self.config.get("atmosphere")
            initial_conditions = fetch_initial_conditions(atmospheric_cfg, tier=self.tier)
            forecast = run_forecast(
                atmospheric_cfg, initial_conditions, self.terrain.z, self.rng,
                self.output_dir / "forecast", tier=self.tier,
            )
        self.forecast = forecast
        self.artifacts["atmosphere"] = {
            "initial_conditions": {k: v for k, v in initial_conditions.items() if k != "source"},
            "members": getattr(forecast, "members", None),
            "source": getattr(forecast, "source", None),
        }
        return forecast

    def run_downscaling(self):
        with self._stage("downscaling"):
            dataset_cfg = self.config.get("datasets.downscaling")
            dataset = build_downscaling_dataset(
                self.terrain, self.config.get("atmosphere"),
                int(dataset_cfg["n_scenarios"]), self.rng, path=dataset_cfg.get("path"),
            )
            checkpoint = self.checkpoint_dir / "downscaler.pt"
            model, report, scaler = train_downscaler(
                dataset,
                self.config.get("downscaling.corrdiff"),
                self.config.get("training.downscaler"),
                device=self.device,
                checkpoint=checkpoint,
                seed=self.seed,
                val_fraction=float(dataset_cfg.get("val_fraction", 0.2)),
            )
            self.artifacts["downscaler_training"] = {
                k: v for k, v in report.items() if k != "history"
            }

            # Apply to the ensemble-mean coarse forecast; the generative spread is
            # sampled per member to carry downscaling uncertainty downstream.
            temperature, precipitation = self.forecast.ensemble_mean()
            samples = int(self.config.get("downscaling.samples"))
            fields = downscale_forecast(
                model, temperature, precipitation, self.terrain.z,
                samples=samples, device=self.device,
            )
            mean = np.asarray(scaler.mean)
            std = np.asarray(scaler.std)
            fine_temperature = fields["temperature_c"] * std[0] + mean[0]
            fine_precipitation = np.clip(fields["precipitation_mm_per_h"] * std[1] + mean[1], 0.0, None)

        self.downscaled = {
            "temperature_c": fine_temperature,
            "precipitation_mm_per_h": fine_precipitation,
        }
        self.artifacts["downscaling"] = {
            "samples": samples,
            "target_shape": list(self.terrain.shape),
            "mean_precipitation_mm_per_h": float(fine_precipitation.mean()),
            "max_precipitation_mm_per_h": float(fine_precipitation.max()),
        }
        return self.downscaled

    def run_mass_balance(self):
        with self._stage("mass_balance"):
            glaciology = self.config.get("glaciology")
            reference_elevation = float(self.config.get("atmosphere.target_region.reference_elevation_m"))
            lapse = float(self.config.get("atmosphere.toy_generator.lapse_rate_c_per_km"))
            time_h = self.forecast.time_h

            forcings = []
            for sample in range(self.downscaled["temperature_c"].shape[0]):
                temperature_field = lapse_correct(
                    self.downscaled["temperature_c"][sample], self.terrain.z,
                    reference_elevation, lapse,
                )
                temperature = catchment_mean_series(temperature_field)
                precipitation = catchment_mean_series(self.downscaled["precipitation_mm_per_h"][sample])
                forcings.append(integrate_catchment(time_h, temperature, precipitation, glaciology))
        self.forcings = forcings
        self.artifacts["mass_balance"] = {
            "n_members": len(forcings),
            "members": [f.as_summary() for f in forcings],
        }
        return forcings

    def run_moraine(self):
        with self._stage("moraine_surrogate"):
            moraine_cfg = self.config.get("moraine")
            dataset_cfg = self.config.get("datasets.moraine")
            checkpoint = self.checkpoint_dir / Path(self.config.get("surrogates.mgn.checkpoint")).name
            if self.reuse_checkpoints and checkpoint.is_file():
                operator = MoraineOperator.load(checkpoint, device=self.device)
                self.artifacts["mgn_training"] = {"reused_checkpoint": str(checkpoint)}
            else:
                dataset = build_moraine_dataset(
                    self.terrain, self.graph, moraine_cfg,
                    int(dataset_cfg["n_scenarios"]), self.rng, path=dataset_cfg.get("path"),
                )
                train_cfg = dict(self.config.get("training.mgn"))
                train_cfg["val_fraction"] = float(dataset_cfg.get("val_fraction", 0.2))
                operator, report = train_mgn(
                    dataset, self.graph, self.config.get("surrogates.mgn"), train_cfg,
                    device=self.device, checkpoint=checkpoint, seed=self.seed,
                )
                self.artifacts["mgn_training"] = {k: v for k, v in report.items() if k != "history"}

        with self._stage("moraine_assessment"):
            assessment = assess_moraine(
                self.terrain, self.graph, self.forcings, self.config.get("moraine"), operator=operator
            )
        self.moraine_operator = operator
        self.assessment = assessment
        self.artifacts["moraine"] = assessment.as_summary()
        return assessment

    def run_breach(self):
        with self._stage("breach"):
            representative = self.assessment.members[self.assessment.representative_member]
            if not representative.breached:
                LOGGER.info("No member breached; the twin reports a stable dam.")
                self.breach = None
                self.artifacts["breach"] = {"breached": False}
                return None
            index = int(np.nanargmax(representative.lake_level_m))
            breach = simulate_breach(
                self.terrain,
                float(representative.lake_level_m[index]),
                representative.mechanism,
                self.config.get("breach"),
                inflow_m3_per_s=float(np.max(self.forcings[representative.member].inflow_m3_per_s)),
            )
        self.breach = breach
        self.artifacts["breach"] = {
            **breach.as_summary(),
            "breached": True,
            "hydrograph_mass_balance_error": breach.mass_balance_error(),
        }
        return breach

    def run_routing(self):
        if self.breach is None:
            self.routing = {}
            self.artifacts["routing"] = {"skipped": "no breach predicted"}
            return {}

        with self._stage("fno_surrogate"):
            fno_cfg = self.config.get("surrogates.fno")
            dataset_cfg = self.config.get("datasets.swe")
            checkpoint = self.checkpoint_dir / Path(fno_cfg["checkpoint"]).name
            if self.reuse_checkpoints and checkpoint.is_file():
                operator = FloodOperator.load(checkpoint, device=self.device)
                self.artifacts["fno_training"] = {"reused_checkpoint": str(checkpoint)}
            else:
                dataset = build_swe_dataset(
                    self.terrain, self.config.get("routing"), self.config.get("breach"),
                    int(dataset_cfg["n_scenarios"]), int(dataset_cfg["frames_per_scenario"]),
                    int(dataset_cfg["resolution"]), self.rng, path=dataset_cfg.get("path"),
                )
                operator, report = train_fno(
                    dataset, fno_cfg, self.config.get("training.fno"),
                    device=self.device, checkpoint=checkpoint, seed=self.seed,
                    val_fraction=float(dataset_cfg.get("val_fraction", 0.25)),
                )
                self.artifacts["fno_training"] = {k: v for k, v in report.items() if k != "history"}

        with self._stage("routing"):
            receptors = self.terrain.receptor_rows(self.config.get("domain.receptors"))
            outcomes = route_flood(
                self.terrain, self.breach, self.config.get("routing"), receptors, operator=operator
            )
        self.flood_operator = operator
        self.routing = outcomes
        self.artifacts["routing"] = {name: o.as_summary() for name, o in outcomes.items()}
        return outcomes

    def run_assimilation(self):
        sensors_cfg = self.config.get("sensors")
        if not bool(sensors_cfg.get("enabled", True)):
            self.artifacts["assimilation"] = {"enabled": False}
            return None

        with self._stage("assimilation"):
            network = build_sensor_network(self.terrain, self.graph, sensors_cfg, self.rng)
            spec = StateSpec.from_config(sensors_cfg["assimilation"])
            filter_ = EnsembleKalmanFilter(spec, sensors_cfg["assimilation"], self.rng)

            truth_values = {
                "lake_level_m": float(self.terrain.initial_lake_level + 0.8),
                "pore_pressure_ratio": 0.42,
                "ddf_ice_mm_per_c_per_day": float(self.config.get("glaciology.ddf_ice_mm_per_c_per_day")) * 1.15,
            }
            truth = np.array([truth_values[name] for name in spec.names])
            background = truth + spec.prior_std * np.array([1.5, -1.2, 1.1])[: spec.size]

            filter_.initialise(background)
            prior_rmse = filter_.rmse_against(truth)
            trajectory = [prior_rmse]
            n_cycles = 4
            for _ in range(n_cycles):
                observations = synthesise_observations(network, truth_values, sensors_cfg, self.rng)
                filter_.analysis(observations)
                trajectory.append(filter_.rmse_against(truth))

        self.artifacts["assimilation"] = {
            "enabled": True,
            "network": network.counts(),
            "state_variables": spec.names,
            "prior_normalised_rmse": prior_rmse,
            "analysis_normalised_rmse": trajectory[-1],
            "rmse_trajectory": trajectory,
            "error_reduction": float(1.0 - trajectory[-1] / max(prior_rmse, 1e-12)),
            "ensemble_spread": filter_.spread().tolist(),
        }
        self.sensors = network
        return network

    def run_evaluation(self):
        with self._stage("evaluation"):
            if "solver" in self.routing:
                report = benchmark_routing(self.routing)
                self.artifacts["benchmark"] = report
                LOGGER.info("\n%s", benchmark_table(report))
            else:
                self.artifacts["benchmark"] = {"skipped": "no routing performed"}
        return self.artifacts.get("benchmark")

    def make_products(self):
        with self._stage("products"):
            visualization_cfg = self.config.get("visualization")
            produced: dict[str, str] = {}

            if bool(visualization_cfg.get("make_figures", True)):
                from glof_pipeline.visualization import plots

                figure_dir = self.output_dir / "figures"
                figure_dir.mkdir(parents=True, exist_ok=True)
                dpi = int(visualization_cfg.get("dpi", 150))

                figures = {
                    "terrain": plots.plot_terrain(self.terrain, getattr(self, "sensors", None)),
                    "forcing": plots.plot_forcing(self.forcings, self.forecast.time_h),
                    "stability": plots.plot_stability(self.assessment, self.terrain, self.graph),
                }
                if self.breach is not None:
                    figures["breach_hydrograph"] = plots.plot_breach_hydrograph(self.breach)
                if "solver" in self.routing:
                    figures["flood"] = plots.plot_flood_snapshots(self.terrain, self.routing["solver"])
                if "comparisons" in self.artifacts.get("benchmark", {}):
                    figures["benchmark"] = plots.plot_benchmark(self.artifacts["benchmark"])

                for name, figure in figures.items():
                    path = figure_dir / f"{name}.png"
                    figure.savefig(path, dpi=dpi)
                    produced[name] = str(path)

            if bool(visualization_cfg.get("export_usd", True)) and "solver" in self.routing:
                from glof_pipeline.visualization.usd_exporter import export_flood_to_usd

                outcome = self.routing["solver"]
                # Prefer the full depth history so the scene animates the wave rather
                # than cutting between two stills.
                depths = (
                    outcome.depth_sequence
                    if outcome.depth_sequence is not None
                    else np.stack([outcome.max_depth_m, outcome.final_depth_m])
                )
                path = export_flood_to_usd(
                    self.terrain.z, depths, outcome.time_s[: depths.shape[0]], self.terrain.dx,
                    visualization_cfg["usd_path"],
                    threshold_m=float(self.config.get("routing.inundation_threshold_m")),
                    stride=max(1, self.terrain.shape[0] // 64),
                    frames=int(visualization_cfg.get("usd_frames", 24)),
                )
                produced["usd"] = str(path)

                nucleus_url = self.config.get("visualization.omniverse.nucleus_url", None)
                if bool(self.config.get("visualization.omniverse.publish", False)) and nucleus_url:
                    from glof_pipeline.nvidia.omniverse import publish_to_nucleus

                    stem = str(path).removesuffix(".usda")
                    produced["nucleus"] = publish_to_nucleus(
                        {
                            "root": str(path),
                            "terrain": f"{stem}_terrain.usda",
                            "water": f"{stem}_water.usda",
                        },
                        str(nucleus_url),
                    )

        self.artifacts["products"] = produced
        return produced

    # -- driver -------------------------------------------------------------
    def run(self) -> dict[str, Any]:
        """Execute every stage and write the run manifest."""
        started = time.perf_counter()
        self.build_terrain()
        self.run_atmosphere()
        self.run_downscaling()
        self.run_mass_balance()
        self.run_moraine()
        self.run_breach()
        self.run_routing()
        self.run_assimilation()
        self.run_evaluation()
        self.make_products()

        manifest = {
            "tier": self.tier,
            "device": self.device,
            "seed": self.seed,
            "config_hash": self.config.hash(),
            "config_source": str(self.config.source),
            "config": self.config.to_dict(),
            "backends": backend_report(),
            "environment": environment_report(),
            "timings_s": self.timings,
            "total_wall_time_s": time.perf_counter() - started,
            "artifacts": self.artifacts,
        }
        path = write_json(self.output_dir / "run_manifest.json", manifest)
        LOGGER.info("\n%s", timing_breakdown(self.timings))
        LOGGER.info("Run manifest written to %s", path)
        manifest["manifest_path"] = str(path)
        return manifest


def run_pipeline(config: Config, reuse_checkpoints: bool = False) -> dict[str, Any]:
    """Convenience wrapper used by the CLI and the tests."""
    return Pipeline(config, reuse_checkpoints=reuse_checkpoints).run()
