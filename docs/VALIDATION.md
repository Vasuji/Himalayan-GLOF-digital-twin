# Validation

Two questions, kept separate because conflating them is the main way a surrogate
flood model overstates its own credibility:

1. **Is the numerics correct?** Does the reference solver reproduce known
   solutions, and does the surrogate reproduce the reference solver? Answered
   below by executed benchmarks.
2. **Is the model right about the world?** Would it have predicted a real GLOF?
   **Not answered.** No hindcast has been run.

A speedup figure without (2) is a statement about arithmetic, not about floods.

## 1. Reference solver against analytic solutions

Executed; each row is a measured number.

| Test | What it checks | Result |
|---|---|---|
| Lake at rest over a rough bed | Well-balancedness: a still lake over irregular bathymetry must generate no motion | max abs momentum **2.5e-14**, max depth drift 3e-14 |
| Closed basin, wall boundaries | Discrete mass conservation | relative volume error **< 1e-10** |
| Ritter dam break, dry bed, t = 20 s | Shock and rarefaction against the exact solution | mean L1 / upstream depth **0.36 %**; numerical wet front lags the analytic tip, as expected for a first-order scheme |
| Wet/dry front on a slope | Positivity and finiteness at the margin | all depths finite and non-negative |
| Breach hydrograph vs released volume | Hydrograph integral equals lake volume change | agreement within **2 %** |
| Froehlich peak regression cross-check | Independent peak-discharge estimate | same order of magnitude as the routed peak |

Analytic identities checked in the unit tests: the cohesionless dry slope reduces
exactly to `tan(phi)/tan(beta)` (`rtol` 1e-12); the Terzaghi critical gradient
matches `(G_s - 1)/(1 + e)`; the Dupuit phreatic surface is linear in distance and
capped by the slab depth.

## 2. NVIDIA stack integration

Executed on CPU. Imports and tensor shapes only — no training, by instruction.

| Check | Result |
|---|---|
| `physicsnemo` 2.2.1, `earth2studio` 0.18.0, `torch` 2.10.0, `torch_geometric` 2.8.0 import | pass |
| `physicsnemo.models.{fno, meshgraphnet, diffusion_unets}` import after Warp cache redirection | pass |
| `earth2studio.models.px.FCN3`, `models.dx.CorrDiff`, `io.ZarrBackend` present | pass |
| Real `physicsnemo` FNO forward, `(1,4,24,24) -> (1,3,24,24)` | pass |
| CorrDiff regression stage on `SongUNetPosEmbd`, `(1,3,32,32) -> (1,2,32,32)` | pass |
| CorrDiff diffusion loss via `EDMPreconditioner` + `EDMNoiseScheduler`, backward pass | pass, finite loss, gradients populated |
| CorrDiff Heun sampling via `samplers.sample`, `(1,2,32,32)` | pass |
| PhysicsNeMo kernel CRPS vs the empirical-CDF estimator, 200-member Gaussian | agree within 0.05 |
| Foreign-domain CorrDiff refusal (`CorrDiffTaiwan` over the Himalaya) | raises, as intended |
| PhysicsNeMo MeshGraphNet forward | **not run locally**: needs `torch_scatter`, which compiles against torch and has no toolchain here. Construction succeeds; the forward pass raises from inside PhysicsNeMo. Deferred to the GPU host |

### Environment requirements established by testing

- `SongUNetPosEmbd` concatenates `N_grid_channels` positional-embedding channels
  onto its input, so `in_channels` must be declared as data channels **plus** grid
  channels. Declaring only the data channels fails at the first forward pass.
- `physicsnemo.models` triggers `warp.init()` at import, which raises
  `PermissionError` when the Warp kernel cache directory is not writable. Setting
  `WARP_CACHE_PATH` is insufficient once `warp.config` has been imported;
  `warp.config.kernel_cache_dir` must be set too.

## 3. Surrogate against the reference solver

`glof_pipeline/evaluate/benchmark.py` measures, per routing method: wall-clock
time, speedup against the solver, relative L2 of the depth envelope, peak depth,
and arrival-time error at each named receptor. The numbers are written into
`run_manifest.json`.

**These are measured at run time and deliberately not quoted here.** An earlier
draft of the manuscript carried a speedup/error table captioned as illustrative;
any such table must be replaced by `benchmark_table()` output from a run whose
configuration hash is cited alongside it.

## 4. Assimilation

The twin experiment in `pipeline.run_assimilation` generates observations from a
known truth, corrupts them with the configured instrument noise, drops a fraction
to represent telemetry loss, and reports the reduction in normalised state RMSE
from prior to posterior. It demonstrates that the filter uses information; it does
**not** demonstrate skill against real instruments.

## 5. What is required before operational use

In order of how much they limit the result:

1. **Hindcast validation.** South Lhonak (Sikkim, October 2023), Chorabari
   (Kedarnath, June 2013), Dig Tsho (Khumbu, August 1985). Report arrival-time
   and peak-discharge errors against documented observations, not
   surrogate-versus-solver agreement.
2. **A CorrDiff trained for the Hindu Kush Himalaya.** No published checkpoint
   covers this domain, and region-foreign checkpoints are refused in code.
3. **Site data.** Lake bathymetry, moraine geotechnical parameters from survey,
   and a DEM contemporaneous with the assessment. Present values are literature
   defaults; the toy dam geometry was tuned to sit near its stability boundary.
4. **Probability calibration.** Reliability curve and Brier score exist in
   `evaluate/metrics.py`; calibration needs an event archive.
5. **Institutional review** by the responsible national agency before any output
   reaches a person downstream.
