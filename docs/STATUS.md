# Implementation and verification status

Component-level record of what is implemented, what has been verified by
execution, and what remains blocked on external data or hardware. Updated in the
same commit as the work it describes.

Markers:

- `[x]` implemented and verified by execution, with the result recorded below
- `[~]` implemented, not yet executed in this environment
- `[ ]` not implemented
- `[!]` blocked on data or software outside this repository

---

## Test status

```
73 passed, 1 skipped, 8 deselected in 21.08s
```

- The skipped test is the PhysicsNeMo MeshGraphNet forward pass, which requires
  `torch-scatter`. The wheel available for macOS does not load against torch 2.10,
  so this stage is verified on a CUDA host.
- The 8 deselected tests are the GPU tier (`-m gpu`): the end-to-end run, which
  trains the MeshGraphNet, the FNO and a two-stage CorrDiff. `pyproject.toml` sets
  `addopts = "-m 'not slow and not gpu'"` so they are never selected by default.

---

## Reference physics and terrain

- [x] Degree-day catchment mass balance with rain/snow partition, snowpack
      accounting and lake filling against a stage-storage curve.
- [x] Moraine stability: infinite-slope Mohr-Coulomb on effective stress, Dupuit
      phreatic surface, Terzaghi heave and backward-erosion piping criteria,
      thermal degradation of an ice core by cumulative positive degree-days.
- [x] Breach formation: Froehlich (2008) geometric regressions, trapezoidal
      broad-crested weir outflow, lake depletion against the DEM-derived
      stage-storage curve.
- [x] Well-balanced HLL finite-volume shallow-water solver with wet/dry front
      handling and semi-implicit Manning friction.
- [x] Synthetic moraine-dammed valley construction and raster-derived moraine
      graph with exact grid adjacency.
- [x] Delineation from a surveyed DEM: depression filling by morphological
      reconstruction (Soille & Ansoult 1990) locates the basin, its spill point
      sets the crest, and the moraine is the downstream band within a configurable
      distance. Survey masks supplied under `domain.masks` take precedence over
      the automatic result.

### Verified numerical results

| Property | Result |
|---|---|
| Lake at rest over a rough bed (well-balanced) | max abs momentum 2.5 × 10⁻¹⁴ |
| Closed-basin mass conservation | relative error < 10⁻¹⁰ |
| Ritter dam break, t = 20 s | mean L1 / upstream depth = 0.36 % |
| Breach hydrograph vs released volume | within 2 % |
| Toy-configuration breach probability (8 members) | 0.88, earliest failure at 54 h lead |
| Lake rise over a 7-day forecast | 2.5–18.8 m against a 10 m freeboard |
| SWE training set at 24 × 24 | max depth 0.33 m, monotonic across frames |

---

## PhysicsNeMo integration

- [x] `glof_pipeline/nvidia/` holds every NVIDIA import at module scope;
      `glof_pipeline/backends.py` is the single point of model construction.
- [x] Warp kernel-cache configuration. `physicsnemo.models` initialises Warp at
      import and requires a writable cache directory;
      `configure_warp_cache` sets `WARP_CACHE_PATH` and
      `warp.config.kernel_cache_dir`, the latter being required once
      `warp.config` has already been imported in the process.
- [x] `physicsnemo.models.fno.FNO`: constructed and run, `(1,4,24,24) → (1,3,24,24)`.
- [x] CorrDiff on the current diffusion API — `SongUNetPosEmbd`,
      `EDMPreconditioner`, `EDMNoiseScheduler`, `samplers.sample` with the Heun
      solver, and `MSEDSMLoss`. Verified: regression `(1,3,32,32) → (1,2,32,32)`,
      finite loss with populated gradients, sampling `(1,2,32,32)`. The APIs
      PhysicsNeMo 2.2.1 marks for removal are not used.
- [x] Checkpoints written through `physicsnemo.utils.checkpoint.save_checkpoint`;
      training logs through `physicsnemo.utils.logging.PythonLogger`.
- [x] Verification statistics from `physicsnemo.metrics.general`
      (`mse`, `relative_error`, `crps`/`kcrps`, `power_spectrum`, `calibration`)
      and `earth2studio.statistics`. The PhysicsNeMo kernel CRPS agrees with the
      empirical estimator within 0.05 on a 200-member Gaussian.
- [~] MeshGraphNet: construction verified; the forward pass requires
      `torch-scatter` and runs on a CUDA host.

---

## Earth-2 integration

- [x] Registries confirmed present in the installed 0.18.0: `models.px` (FCN3,
      SFNO, GraphCast variants, AIFS, Pangu24, Aurora, DLWP, DLESyM), `models.dx`
      (CorrDiff, CorrDiffTaiwan, CorrDiffCMIP6, CBottleSR), `data` (GFS, ARCO,
      IFS, HRRR, NCAR_ERA5, Random and others), `perturbation`, `io.ZarrBackend`,
      `statistics`.
- [x] Region safety: published CorrDiff checkpoints are trained per region, and
      `load_downscaler` refuses a region-foreign checkpoint for this domain unless
      the configuration names an explicit override. Covered by test.
- [x] Deterministic and ensemble workflow drivers wired through
      `earth2studio.run`, with the forecast written to a `ZarrBackend` store.

---

## Omniverse integration

- [x] Layered USD authoring: a static terrain sublayer and an animated water
      sublayer composed by a thin root, `UsdPreviewSurface` water and terrain
      materials, `UsdLux` sun and dome lighting, a framed review camera, and
      simulated seconds recorded in `customLayerData`. Verified against a real
      `pxr` stage (`usd-core` 0.26.8): the composed root opens, declares a default
      prim and two sublayers; both meshes carry valid quad topology; the water
      surface has time samples and the stage a non-degenerate time range.
- [~] Nucleus publishing via `omni.client`, which ships with Omniverse Kit rather
      than PyPI and is therefore unexecuted here.

---

## Pipeline, configuration and packaging

- [x] Ten-stage orchestrator writing `run_manifest.json` with the configuration
      hash, resolved configuration, NVIDIA component versions, per-stage wall
      times and each stage's numerical summary.
- [x] Configuration: four component YAMLs plus `toy` / `production` / `smoke`;
      recursive includes with cycle detection, deep merge, inline `--set`
      overrides, and a 12-character content hash. Layering verified: CorrDiff
      width resolves to 8 / 16 / 128 model channels across the three tiers.
- [x] Datasets generated from the reference physics; three training loops;
      ensemble Kalman assimilation with a synthetic in-situ network; moraine
      assessment; flood routing; measured surrogate-versus-solver benchmark;
      figures.
- [x] CLI (`glof info | run | dataset | train | benchmark`), Makefile, run and
      setup scripts, CUDA Dockerfile asserting the NVIDIA import at build time,
      docker-compose with GPU reservation, GitHub Actions workflow.
- [x] Documentation: `ARCHITECTURE.md`, `NVIDIA_INTEGRATION.md`, `PRODUCTION.md`,
      `GCP_GPU.md`, `VALIDATION.md`, `README.md`.

---

## Remaining work

- [ ] GPU-tier suite (`pytest -m gpu`) on a CUDA host: the end-to-end run.
- [ ] Production-scale dataset generation and surrogate training.

---

## Environment

`nvidia-physicsnemo` 2.2.1, `earth2studio` 0.18.0, `torch` 2.10.0,
`torch-geometric` 2.8.0 and `usd-core` 0.26.8 are installed and importable.
`torch-scatter` is not installed: the macOS wheel installs but fails to load
`_scatter_cpu.so` against torch 2.10, and an unusable extension also disables
`torch_geometric`'s use of it, so `glof_pipeline.backends` probes dependencies by
import rather than by module spec.

Local execution is limited to imports, constructor calls, small-tensor forward
passes and the CPU reference physics. Dataset generation at production
resolution, surrogate training, CorrDiff training and Earth-2 model loading run
on a CUDA host; see `docs/GCP_GPU.md`.

---

## Blocked on external data or resources

- [!] **A CorrDiff trained for the Hindu Kush Himalaya.** No published checkpoint
      covers this domain. Until one is trained, kilometre-scale fields are a
      structural demonstration rather than a forecast.
- [!] **Site data.** Lake bathymetry, surveyed moraine geotechnical parameters and
      a contemporaneous DEM. Values under `moraine.*` are literature defaults, and
      the toy dam geometry is set near its stability boundary for demonstration
      rather than measured.
- [!] **Hindcast validation** against South Lhonak (2023), Chorabari (2013) and
      Dig Tsho (1985), reporting arrival-time and peak-discharge errors.
- [!] **Probability calibration.** The reliability curve, Brier score and rank
      histogram are implemented; calibration requires an event archive.
- [!] **`torch-scatter` matching torch 2.10** for the MeshGraphNet forward pass.
- [!] **`omni.client`** for Nucleus publishing, distributed with Omniverse Kit.
