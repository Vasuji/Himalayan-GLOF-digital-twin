#!/usr/bin/env bash
# Provision the production tier on a Google Cloud GPU instance.
# Run ON the instance, after `nvidia-smi` reports a healthy driver.
# See docs/GCP_GPU.md for machine-type guidance.
set -euo pipefail

CUDA_CHANNEL="${CUDA_CHANNEL:-cu126}"

command -v nvidia-smi >/dev/null || { echo "No NVIDIA driver found." >&2; exit 1; }
nvidia-smi

python -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip

# torch first, from the CUDA index: PhysicsNeMo 2.2.1 needs >=2.10, and letting pip
# resolve it transitively can pull a CPU-only wheel.
pip install torch --index-url "https://download.pytorch.org/whl/${CUDA_CHANNEL}"
pip install -r requirements-nvidia.txt
pip install -e ".[dev]"

export WARP_CACHE_PATH="$PWD/.warp_cache"
mkdir -p "$WARP_CACHE_PATH" data/raw data/processed outputs checkpoints

python -m glof_pipeline info
echo
echo "If 'physicsnemo' reads false above, the import failed rather than the package"
echo "being absent -- check the torch version first (needs >= 2.10)."
echo
echo "Shape-check every stage before launching a long job:"
echo "  python -m glof_pipeline run --config configs/smoke.yaml --set runtime.device=cuda"
