# Running the production tier (NVIDIA Earth-2 + PhysicsNeMo)

The toy tier is a complete, self-consistent end-to-end model that runs on a CPU
laptop. The production tier replaces three of its components with the NVIDIA
stack. **It is not a drop-in swap** — the honest state of each component is
recorded below, because a downscaler or surrogate trained on the wrong domain
produces fields that look plausible and are wrong.

## 1. Environment

```bash
# CUDA host, Python >= 3.10
pip install -r requirements-nvidia.txt
export NGC_API_KEY=...        # only needed for gated model packages
glof info                     # confirm both backends resolve
```

Package names, verified against the shipped distributions:

| Purpose | PyPI distribution | Import name |
|---|---|---|
| Physics-ML framework | `nvidia-physicsnemo` (>= 2.2.1) | `physicsnemo` |
| Earth-2 inference | `earth2studio` (>= 0.18.0) | `earth2studio` |
| GNN backend | `torch-geometric`, `torch-scatter`, `torch-cluster` | — |

Two points to note:

* the distribution is **not** called `physicsnemo` on PyPI, and the older
  `nvidia-modulus` name refers to the pre-rename releases;
* PhysicsNeMo's graph models take a **PyTorch Geometric** `Data` object. DGL
  support was withdrawn upstream, so `pip install dgl` will not help.

## 2. What each component does in production

### Atmosphere — `earth2studio.models.px`

`atmosphere.prognostic_model` selects the global model. The registry in
earth2studio 0.18 includes `FCN3`, `SFNO`, `GraphCastOperational`,
`GraphCastSmall`, `AIFS`, `AIFSENS`, `Pangu24`, `Aurora`, `DLWP` and `DLESyM`.
Initial conditions come from `atmosphere.data_source` (`GFS`, `GFS_FX`, `ARCO`,
`IFS`, `HRRR`, `NCAR_ERA5`, `CDS`, `CMIP6`, `GOES`).

Ensembles use `earth2studio.run.ensemble` with a perturbation from
`earth2studio.perturbation`: `Zero`, `Gaussian`, `SphericalGaussian`,
`CorrelatedSphericalGaussian`, `BredVector`, `HemisphericCentredBredVector`,
`Brown`, `LaggedEnsemble`. The default here is `CorrelatedSphericalGaussian`,
which respects the sphere's geometry rather than adding grid-space white noise.

**Status: ready.** This component works as shipped.

### Downscaling — `earth2studio.models.dx`

**Status: requires a domain-specific checkpoint. This is the critical gap.**

The published CorrDiff checkpoints are trained per region: `CorrDiffTaiwan` on
the Taiwan CWA domain, `CorrDiffCosmoEra5` over central Europe. Neither is valid
over the Hindu Kush Himalaya, and using one would fabricate orographic
precipitation structure from the wrong mountain range.
`glof_pipeline/atmospheric/downscaling.py` therefore **refuses** to run those
checkpoints unless `downscaling.production_checkpoint` names an explicit
override.

To close the gap you must train a CorrDiff for this domain with PhysicsNeMo:

1. assemble a paired training archive — ERA5 or IFS coarse fields against a
   convection-permitting WRF/ICON hindcast over the target catchment, several
   years at hourly resolution;
2. train the regression (UNet) stage, then the diffusion (EDM-preconditioned)
   stage on the residual, following PhysicsNeMo's CorrDiff recipe;
3. package the checkpoint and point `downscaling.production_checkpoint` at it.

Until that exists, run the toy downscaler and treat the kilometre-scale fields
as a structural demonstration, not a forecast.

### Surrogates — `physicsnemo.models`

`glof_pipeline/backends.py` instantiates `physicsnemo.models.fno.FNO` and
`physicsnemo.models.meshgraphnet.MeshGraphNet` from the same configuration
blocks the toy tier uses, so `configs/production.yaml` only changes capacities
(32 Fourier modes, 64 latent channels, 15 message-passing steps).

**Status: ready, but the checkpoints must be retrained at production scale.** A
surrogate trained on 256 toy scenarios is not a surrogate for a 512x512 domain
with 20,000 scenarios. Weight transfer works for the FNO (identical
architecture); the MeshGraphNet normalisation layers differ between backends, so
retrain rather than port.

## 3. Terrain

`domain.dem_path` should point at a real raster (Copernicus GLO-30, ALOS AW3D30,
or a UAV survey). `glof_pipeline/terrain/dem_io.py` reads it, but lake and
moraine delineation from a survey raster needs site-specific masks — the
pipeline raises rather than guessing them. Supply them under `domain.masks`.

## 4. Suggested first production run

```bash
glof dataset --config configs/production.yaml --which all     # hours, GPU
glof train   --config configs/production.yaml --which all     # hours, GPU
glof run     --config configs/production.yaml --reuse-checkpoints
```

Inspect `outputs/production/run_manifest.json`: it records the configuration
hash, the resolved backends, package versions, per-stage wall times and the
measured surrogate-versus-solver benchmark. Any figure or table quoted from a
run should cite that hash.

## 5. Before this is used operationally

The following are prerequisites, not refinements:

1. **Site data.** Bathymetric survey of the lake, geotechnical parameters for the
   moraine (`moraine.*` are literature defaults, not measurements), and a DEM
   contemporaneous with the assessment.
2. **A domain-trained CorrDiff** (section 2).
3. **Hindcast validation** against documented events — South Lhonak (2023),
   Chorabari (2013), Dig Tsho (1985) — reporting arrival-time and peak-discharge
   errors, not only surrogate-versus-solver agreement.
4. **Calibrated probabilities.** `evaluate/metrics.py` provides the Brier score
   and reliability curve; a breach probability is only useful once it is shown to
   be calibrated.
5. **Institutional review.** Any warning product needs the responsible national
   agency in the loop.
