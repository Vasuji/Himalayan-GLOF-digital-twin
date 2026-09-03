# Architecture

## The two-tier idea

Every stage exists once and runs in one of two tiers, selected by
`runtime.tier`:

* **toy** — reference physics plus narrow PhysicsNeMo surrogates that train and
  run on a CPU laptop. Nothing is stubbed and no number is asserted: the melt is
  integrated, the stability field is solved, the shallow-water equations are
  advanced, the networks are trained.
* **production** — the same stage graph and the same PhysicsNeMo modules at
  production width, with NVIDIA Earth-2 driving the atmosphere.

The tiers differ in size and data source, never in implementation. PhysicsNeMo is
a hard requirement in both: every NVIDIA import lives at module scope in
`glof_pipeline/nvidia/`, and `glof_pipeline/backends.py` raises with an actionable
message rather than substituting anything. The versions of each NVIDIA component
are recorded in every run manifest.

## Stage graph

```
                    configs/*.yaml  (includes + deep merge -> content hash)
                              |
  terrain ------------------- v ------------------------------------------+
  synthetic valley or DEM; moraine meshed as a graph                      |
      |                                                                  |
      v                                                                  |
  atmosphere                                                             |
  Earth-2 prognostic + ensemble perturbation | toy stochastic generator   |
      |  coarse (~25 km) T2M, precipitation                               |
      v                                                                  |
  downscaling                                                            |
  PhysicsNeMo CorrDiff: regression stage + EDM residual diffusion         |
      |  fine (1 km) T2M, precipitation, S generative samples             |
      v                                                                  |
  mass balance                                                           |
  degree-day melt, rain/snow split, lake inflow, cumulative PDD           |
      |                                                                  |
      v                                                                  |
  moraine  <---- MeshGraphNet surrogate <---- moraine dataset <-----------+
  limit equilibrium per node; overtopping / piping / slope-failure        |
  triggers; breach probability across the ensemble                        |
      |                                                                  |
      v                                                                  |
  breach                                                                 |
  Froehlich (2008) geometry; trapezoidal weir outflow against the         |
  DEM stage-storage curve                                                |
      |  hydrograph                                                      |
      v                                                                  |
  routing  <----- FNO surrogate <----- SWE dataset <---------------------+
  well-balanced HLL finite volume (reference) and FNO rollout
      |
      v
  assimilation (EnKF over stage / piezometric / discharge observations)
      |
      v
  evaluation (measured benchmark, verification metrics)
      |
      v
  products (figures, animated USD scene, run manifest)
```

## Reference physics and why these choices

| Component | Model | Reason |
|---|---|---|
| Melt | Degree-day (temperature index) | The energy-balance terms are unobserved in most Himalayan catchments; the temperature-index approach is the operational standard (Hock, *J. Hydrol.* 282, 2003). |
| Slope stability | Infinite-slope Mohr-Coulomb with a Dupuit phreatic surface | Reduces exactly to `tan(phi)/tan(beta)` in the dry cohesionless limit, which is a closed-form unit test. |
| Internal erosion | Terzaghi heave **and** backward erosion along a concentrated leak path | Heave alone never triggers on a wide dam; the documented mechanism for ice-cored moraines is a conduit far shorter than the dam is wide. Both ratios are reported. |
| Breach geometry | Froehlich (2008) regressions on 74 embankment failures | The standard empirical basis; an independent 1995 peak-discharge regression is carried as a cross-check. |
| Breach outflow | Trapezoidal broad-crested weir against the stage-storage curve | Makes the integral of the hydrograph equal the released volume by construction. |
| Flood routing | First-order HLL Godunov with hydrostatic reconstruction (Audusse et al. 2004) and semi-implicit Manning friction | Well-balanced (lake at rest stays at rest to machine precision), positivity-preserving at wet/dry fronts, and verified against the Ritter solution. |

## Surrogate design decisions

**FNO learns the increment, not the next state.** `(h, hu, hv, z_bed)_t ->
(h, hu, hv)_{t+dt}` is predicted as `state + network(state)`, and training uses a
short autoregressive rollout so the model sees its own predictions. Both choices
are what keep long rollouts from drifting.

**The loss is not a plain MSE.** A pointwise L2 lets a spectral model damp the
high wavenumbers that carry the flood front, and gives it no reason to conserve
water. `surrogates/losses.py` adds a spectral term (relative L2 between amplitude
spectra) and a volume term.

**MeshGraphNet predicts `log(FS)`, not `FS`.** The factor of safety is a positive
ratio spanning more than an order of magnitude across the mesh; regressing it
directly makes the loss dominated by the stable interior rather than the
near-critical toe where the decision is made.

**The downscaler is two-stage.** A deterministic regression for the conditional
mean plus EDM residual diffusion (Karras et al. 2022), which is CorrDiff's
structure. The residual is where the orographically-forced extremes live, and a
GLOF is driven by extremes rather than by the mean. The smoke test asserts that
the diffusion stage restores variance the regression smooths away.

**The graph is raster-derived, not k-nearest-neighbour.** Grid adjacency over the
moraine footprint gives an exact, symmetric neighbour set with no spurious long
edges across the crest.

## Configuration

Component files (`atmospheric_cfg.yaml`, `hydrology_cfg.yaml`,
`swe_routing_cfg.yaml`, `sensors_cfg.yaml`) each own one section. `toy.yaml`
includes them; `production.yaml` and `smoke.yaml` include `toy.yaml` and
override. Includes are resolved recursively with cycle detection and deep-merged,
and the resulting tree carries a 12-character content hash written into every run
manifest. Any key is overridable inline:

```bash
glof run --config configs/toy.yaml --set runtime.seed=3 training.fno.epochs=5
```

## Reproducibility

`run_manifest.json` records the configuration hash and full resolved tree, the
seed, the resolved backends, package versions and CUDA availability, per-stage
wall times, and every stage's numerical summary. A figure without a manifest hash
is not a result.
