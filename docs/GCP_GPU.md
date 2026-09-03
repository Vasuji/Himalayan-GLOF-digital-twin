# Running on a Google Cloud GPU

Nothing GPU-intensive runs on a laptop. Locally the repository is limited to
imports, constructor calls, small-tensor forward passes and the CPU reference
physics. Dataset generation at production resolution, surrogate training, CorrDiff
training and Earth-2 model loading all belong here.

## 1. Choose a machine

| Work | Suggested machine | Notes |
|---|---|---|
| Surrogate training (FNO, MeshGraphNet) | `a2-highgpu-1g` (1×A100 40 GB) | Both surrogates fit comfortably; this is the cheapest useful tier |
| CorrDiff training for the Hindu Kush Himalaya | `a2-ultragpu-4g` or `a3-highgpu-8g` | Diffusion training is the expensive item — days, not hours |
| Earth-2 inference only (FCN3, SFNO, ensemble) | `g2-standard-16` (1×L4 24 GB) | Sufficient for global forecasting at 0.25 degrees |
| Reference-solver dataset generation | `c3-highcpu-88` (no GPU) | The shallow-water solver is CPU-bound and embarrassingly parallel over scenarios |

Boot disk: 200 GB balanced or better. Earth-2 model packages, a Zarr forecast
store and the SWE training set will consume most of it.

```bash
gcloud compute instances create glof-train \
  --zone=us-central1-a \
  --machine-type=a2-highgpu-1g \
  --maintenance-policy=TERMINATE \
  --image-family=common-cu126-ubuntu-2204-py310 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB --boot-disk-type=pd-balanced \
  --metadata="install-nvidia-driver=True"
```

The Deep Learning VM image ships the NVIDIA driver and CUDA, which avoids the
most common source of lost time. Verify before anything else:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 2. Install

```bash
git clone <your-fork> && cd codes
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip

# PhysicsNeMo 2.2.1 requires torch >= 2.10. Install torch for your CUDA first,
# otherwise pip may resolve a CPU wheel.
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements-nvidia.txt
pip install -e .

glof info      # must report physicsnemo, earth2studio and CUDA all present
```

Two environment variables matter:

```bash
export NGC_API_KEY=...                    # gated Earth-2 model packages
export WARP_CACHE_PATH="$PWD/.warp_cache" # see docs/NVIDIA_INTEGRATION.md
```

`glof info` printing `"physicsnemo": false` means the import failed rather than
the package being absent — check the torch version first.

## 3. Container route (reproducible, recommended for a cluster)

```bash
docker build -t glof-twin:latest .
docker run --gpus all --rm -it \
  -e NGC_API_KEY \
  -v "$PWD/data:/workspace/data" \
  -v "$PWD/outputs:/workspace/outputs" \
  -v "$PWD/checkpoints:/workspace/checkpoints" \
  glof-twin:latest glof info
```

The image is based on `nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04`. Mount `data`,
`outputs` and `checkpoints` so nothing that costs GPU-hours lives inside a
container layer.

## 4. Stage the data

The toy tier needs no external data — the valley, forcing and training sets are
generated from configuration and a seed. The production tier needs:

| Item | Destination | Source |
|---|---|---|
| DEM (30 m or better) | `data/raw/himalayas_dem_30m.tif` | Copernicus GLO-30, ALOS AW3D30, or a UAV survey |
| Lake bathymetry | `data/raw/` | Site survey. Without it the stage-storage curve is a guess |
| Historical events (for hindcasts) | `data/raw/events/` | Literature; see `docs/VALIDATION.md` |

Earth-2 initial conditions are fetched at run time by `earth2studio.data` and do
not need staging. If the instance has no egress, pre-download to a Zarr store and
point `atmosphere.data_source` at a local source instead.

```bash
gsutil -m cp -r gs://your-bucket/himalayas/raw data/
```

## 5. Run

Run the stages separately, not as one long job — dataset generation and training
have very different failure modes, and a crash three hours into a combined run
costs the dataset too.

```bash
# 5a. Datasets. CPU-bound; the SWE set dominates.
glof dataset --config configs/production.yaml --which all

# 5b. Train the surrogates. GPU.
glof train --config configs/production.yaml --which mgn
glof train --config configs/production.yaml --which fno
glof train --config configs/production.yaml --which downscaler

# 5c. Full pipeline against the trained checkpoints.
glof run --config configs/production.yaml --reuse-checkpoints
```

Any key can be overridden without editing a file, which is how to shrink a first
run to something that fails fast:

```bash
glof run --config configs/production.yaml \
  --set datasets.swe.n_scenarios=64 training.fno.epochs=5 \
        atmosphere.ensemble.members=8
```

Sanity-check the shapes with the smoke configuration on the GPU host **before**
launching a long job. It exercises every stage in under a minute:

```bash
glof run --config configs/smoke.yaml --set runtime.device=cuda
```

## 6. Read the results

`outputs/production/run_manifest.json` records the configuration hash, the full
resolved configuration, the resolved backends and their versions, CUDA
availability, per-stage wall times and every stage's numerical summary.

Check `backends` first. It records the availability and installed version of each
NVIDIA component the run touched (`physicsnemo`, `earth2studio`,
`torch_geometric`, `torch_scatter`, `omniverse_usd`). PhysicsNeMo is a hard
requirement, so a missing or unusable component aborts the run with a message
naming it.

## 7. Cost discipline

- Stop the instance rather than deleting it while iterating; the boot disk holds
  the Earth-2 model cache, which is slow to repopulate.
- Use a preemptible or Spot instance for dataset generation (checkpointable) but
  not for a multi-day CorrDiff training run.
- `glof benchmark --config configs/production.yaml` reports measured
  surrogate-versus-solver speedup from stored checkpoints, so the case for the
  surrogate can be made without re-running training.

## 8. What a GPU does not fix

The blockers in `docs/STATUS.md` are data and validation problems, not compute
problems. In particular, no published CorrDiff checkpoint covers the Hindu Kush
Himalaya, so kilometre-scale fields remain a structural demonstration until one is
trained for this domain — which is itself the largest GPU item on the list.
