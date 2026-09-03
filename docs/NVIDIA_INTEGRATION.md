# NVIDIA integration map

Every PhysicsNeMo, Earth-2 and Omniverse API the project calls, and the file that
calls it. Signatures were read from the installed distributions
(`nvidia-physicsnemo` 2.2.1, `earth2studio` 0.18.0), not from memory or
documentation.

## Where the NVIDIA imports live

All of them are in `glof_pipeline/nvidia/`, at module scope. Nothing elsewhere in
the package imports `physicsnemo` or `earth2studio` directly. That is a
deliberate constraint: it makes the dependency surface greppable, and it means
`glof_pipeline.backends` is the single place where models are constructed.

| File | Imports |
|---|---|
| `nvidia/__init__.py` | `warp` (cache redirection only) |
| `nvidia/physicsnemo_models.py` | `physicsnemo`, `physicsnemo.models.fno.FNO`, `physicsnemo.models.meshgraphnet.MeshGraphNet` |
| `nvidia/corrdiff.py` | `physicsnemo.models.diffusion_unets.SongUNetPosEmbd`, `physicsnemo.diffusion.preconditioners.EDMPreconditioner`, `physicsnemo.diffusion.noise_schedulers.EDMNoiseScheduler`, `physicsnemo.diffusion.samplers.sample` |
| `nvidia/launch.py` | `physicsnemo.utils.checkpoint.{save_checkpoint, load_checkpoint}`, `physicsnemo.utils.logging.PythonLogger` |
| `nvidia/statistics.py` | `physicsnemo.metrics.general.crps.{crps, kcrps}`, `earth2studio.statistics.{crps, brier_score, spread_skill_ratio, rank_histogram}` |
| `nvidia/earth2.py` | `earth2studio.{run, data, models.px, models.dx, perturbation, io.ZarrBackend}` |
| `nvidia/omniverse.py` | `pxr.{Usd, UsdGeom, UsdShade, UsdLux, Gf, Sdf, Vt}`, optional `omni.client` |

## PhysicsNeMo

### Models

| API | Used for | Verified signature notes |
|---|---|---|
| `physicsnemo.models.fno.FNO` | Flood-routing surrogate: learns `(h, hu, hv, z_bed) -> (h, hu, hv)` over one output interval | Takes `in_channels`, `out_channels`, `dimension`, `num_fno_layers`, `num_fno_modes`, `latent_channels`, `decoder_layers`, `decoder_layer_size`, `padding`, `padding_type`, `activation_fn`, `coord_features` |
| `physicsnemo.models.meshgraphnet.MeshGraphNet` | Moraine mechanics: per-node `log(FS)` and pore-pressure ratio | `forward(node_features, edge_features, graph)` where `graph` is a **PyTorch Geometric `Data`**. DGL support was withdrawn upstream; `torch-geometric` is required |
| `physicsnemo.models.diffusion_unets.SongUNetPosEmbd` | Backbone for both CorrDiff stages | `(img_resolution, in_channels, out_channels, model_channels, channel_mult, num_blocks, attn_resolutions, dropout, ...)`; `forward(x, noise_labels, class_labels=None, ...)` |

### Diffusion — current API, not the legacy one

PhysicsNeMo 2.2.1 raises `LegacyFeatureWarning` for `EDMPrecond`,
`EDMPrecondSuperResolution`, `EDMPrecondSR`, `deterministic_sampler` and
`stochastic_sampler`, and states they will be removed. This project uses the
replacements:

| API | Role |
|---|---|
| `physicsnemo.diffusion.preconditioners.EDMPreconditioner` | `(model, sigma_data=0.5)`. EDM parameterisation; `forward(x, t, condition=None, **model_kwargs)` maps the time step to a noise level internally |
| `physicsnemo.diffusion.noise_schedulers.EDMNoiseScheduler` | `(sigma_min=0.002, sigma_max=80.0, rho=7.0, sigma_data=0.5, P_mean=-1.2, P_std=1.2)`. Supplies `sample_time`, `sigma`, `loss_weight`, `add_noise`, `timesteps` |
| `physicsnemo.diffusion.samplers.sample` | `(denoiser, xN, noise_scheduler, num_steps, solver='heun', ...)` |

**The conditioning bridge.** `EDMPreconditioner.forward` calls its wrapped model
as `model(c_in * x, c_noise, condition=condition)`, but `SongUNetPosEmbd.forward`
accepts `(x, noise_labels, ...)` and has no `condition` parameter. Passing the
UNet directly would therefore raise a `TypeError` on every conditioned call.
`nvidia/corrdiff.py` inserts `_ConditionedUNet`, which accepts `condition` and
concatenates it onto the channel axis — which is how CorrDiff conditions its
diffusion stage. A second wrapper, `_BoundDenoiser`, freezes the condition so the
sampler only has to supply `(x, t)`, avoiding any dependence on whether a given
release forwards keyword arguments through the solver.

### Utilities

| API | Used for |
|---|---|
| `physicsnemo.utils.checkpoint.save_checkpoint` | `(path, models, optimizer, scheduler, scaler, epoch, metadata)`. Mirrors every surrogate checkpoint in PhysicsNeMo's own format, so a checkpoint written on a laptop loads in a PhysicsNeMo GPU job without translation. `path` is a **directory** |
| `physicsnemo.utils.checkpoint.load_checkpoint` | Restores models plus optimiser and scheduler state; returns the stored epoch |
| `physicsnemo.utils.logging.PythonLogger` | Training-loop logging, so surrogate output matches PhysicsNeMo recipe output. Takes pre-formatted messages, so callers use f-strings rather than printf-style arguments |
| `physicsnemo.metrics.general.crps.crps` / `kcrps` | `(pred, obs, dim=0, method='kernel')` and `(pred, obs, dim=0, biased=True)`. Deterministic and kernel CRPS estimators |

### The Warp gotcha

`physicsnemo.models` imports `physicsnemo.nn`, which calls `warp.init()` at import
time. Warp creates a kernel cache under the user cache directory and raises
`PermissionError` when that path is not writable — sandboxes, hardened CI runners,
read-only containers. Two things are needed, and the second is easy to miss:

```python
os.environ["WARP_CACHE_PATH"] = str(path)   # for fresh subprocesses
import warp
warp.config.kernel_cache_dir = str(path)    # for an interpreter that already imported warp
```

The environment variable is only read when `warp.config` is first imported, so if
anything imported Warp earlier in the process it arrives too late.
`glof_pipeline.nvidia.configure_warp_cache` sets both and runs at package import.
Note that `warp.config.quiet` is deprecated in favour of
`warp.config.log_level = warp.LOG_WARNING`.

## Earth-2 (Earth2Studio)

| API | Used for |
|---|---|
| `earth2studio.models.px` | Prognostic models. Installed registry includes `FCN3`, `SFNO`, `GraphCastOperational`, `GraphCastSmall`, `AIFS`, `AIFSENS`, `Pangu24`, `Aurora`, `DLWP`, `DLESyM` |
| `earth2studio.models.dx` | Diagnostic downscalers: `CorrDiff`, `CorrDiffTaiwan`, `CorrDiffCMIP6`, `CorrDiffCosmoEra5`, `CBottleSR`, `CBottleInfill` |
| `earth2studio.data` | Initial conditions: `GFS`, `GFS_FX`, `ARCO`, `IFS`, `HRRR`, `NCAR_ERA5`, `CDS`, `CMIP6`, `GOES` |
| `earth2studio.perturbation` | `Zero`, `Gaussian`, `SphericalGaussian`, `CorrelatedSphericalGaussian`, `BredVector`, `HemisphericCentredBredVector`, `Brown`, `LaggedEnsemble`. Default here is `CorrelatedSphericalGaussian`, which respects the sphere's geometry instead of adding grid-space white noise |
| `earth2studio.run.deterministic` / `.ensemble` / `.diagnostic` | Workflow drivers |
| `earth2studio.io.ZarrBackend` | Forecast output store |
| `earth2studio.statistics` | `crps(ensemble_dimension, reduction_dimensions=None, weights=None, fair=False)`, `rmse(reduction_dimensions, ...)`, `brier_score(reduction_dimensions, thresholds, ...)`, `spread_skill_ratio(ensemble_dimension, reduction_dimensions, ...)`, `rank_histogram(...)`. All are **classes** invoked as `(x, x_coords, y, y_coords)` with `OrderedDict` coordinate systems |

### Region safety is enforced in code

`CorrDiffTaiwan` was trained on the Taiwan CWA domain and `CorrDiffCosmoEra5`
over central Europe. Applying either over the Hindu Kush Himalaya produces fields
that look plausible and are wrong. `nvidia/earth2.py::load_downscaler` raises
unless `downscaling.production_checkpoint` names an explicit override, so the
mistake cannot be made by leaving a default in place.

## Omniverse

| API | Used for |
|---|---|
| `pxr.Usd.Stage` | Three stages: static terrain sublayer, animated water sublayer, and a thin root that composes both via `subLayerPaths` |
| `pxr.UsdGeom.Mesh` | Quad heightfield meshes; the water mesh's `points` attribute is time-sampled |
| `pxr.UsdShade.Material` / `Shader` | `UsdPreviewSurface` water material (`ior` 1.333, low roughness, configurable opacity) and a matte terrain material |
| `pxr.UsdLux.DistantLight` / `DomeLight` | Sun and sky, so the scene renders without manual setup |
| `pxr.UsdGeom.Camera` | Review camera framed on the valley |
| `customLayerData` | Simulated seconds, frame count, grid size, cell size and depth threshold, so the animation is quantitative |
| `omni.client.copy` | Optional publishing to a Nucleus server. Ships with Omniverse Kit, **not** on PyPI; the code raises rather than silently skipping |

Layered composition is used because the terrain is static and the water animation
is not: a reviewer can reload or swap the water layer without re-reading the
terrain, and a production-resolution domain stays streamable.

## Dependency policy

PhysicsNeMo is a hard requirement of both tiers. `glof_pipeline/backends.py` is the
only place models are constructed, and it raises with a message naming the missing
component rather than substituting an alternative implementation. Consequently a
run either uses the NVIDIA stack throughout or does not proceed.

`run_manifest.json` records, under `backends`, the availability and installed
version of every NVIDIA component the run touched.

## Version constraints worth knowing

- PhysicsNeMo 2.2.1 requires **torch >= 2.10**. An older torch fails at import,
  not at call time.
- The PyPI distribution is `nvidia-physicsnemo`, imported as `physicsnemo`. The
  older `nvidia-modulus` name refers to pre-rename releases.
- MeshGraphNet requires **both** `torch-geometric` and `torch-scatter`.
  Construction succeeds with `torch-geometric` alone; message passing raises
  `ImportError: MeshGraphNet requires PyTorch Geometric and torch_scatter` from
  `physicsnemo/models/meshgraphnet/meshgraphnet.py`. `torch-scatter` compiles
  against the installed torch and requires a C++ toolchain; the macOS wheel
  installs but fails to load `_scatter_cpu.so` against torch 2.10, so the moraine
  surrogate runs on a CUDA host. `backends.build_meshgraphnet` validates both
  dependencies before construction and raises naming the missing one.
- `SongUNetPosEmbd` concatenates `N_grid_channels` positional-embedding channels
  onto its own input, so `in_channels` must be declared as **data channels plus
  grid channels**. Declaring only the data channels builds a first convolution
  that is `N_grid_channels` too narrow and raises
  `expected input[...] to have 3 channels, but got 7` at the first forward pass.
  CorrDiff's own configuration files follow the same convention
  (`img_in_channels` includes the grid channels); `nvidia/corrdiff.py` adds them
  in `_build_unet`.
- Constructor arguments are filtered against the live signature by
  `nvidia/_introspect.py`, so a renamed or retired argument surfaces as a logged
  note listing what was dropped rather than a `TypeError`. A genuinely missing
  required argument raises an error quoting the installed signature.
