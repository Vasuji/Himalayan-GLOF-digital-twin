#!/usr/bin/env bash
# Create a Python environment for the digital twin.
#
#   ./scripts/setup_env.sh cpu    # imports, shapes, reference physics, unit tests
#   ./scripts/setup_env.sh gpu    # everything, including surrogate training
#
# PhysicsNeMo is a hard requirement of both tiers and is installed in both modes.
set -euo pipefail

MODE="${1:-cpu}"
VENV="${VENV:-.venv}"

python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel

if [[ "$MODE" == "gpu" ]]; then
  # torch first, from the CUDA index. PhysicsNeMo 2.2.1 requires torch>=2.10 and
  # transitive resolution can otherwise pick a wheel that fails at import.
  python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
  python -m pip install -r requirements-nvidia.txt
else
  python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
  python -m pip install nvidia-physicsnemo earth2studio torch-geometric usd-core
fi
python -m pip install -e ".[dev]"

# physicsnemo.models calls warp.init() at import; Warp aborts when its default
# cache directory is not writable (sandboxes, hardened CI, read-only containers).
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-$PWD/.warp_cache}"
mkdir -p "$WARP_CACHE_PATH"
echo "export WARP_CACHE_PATH=$WARP_CACHE_PATH" >> "$VENV/bin/activate"

echo
python -m glof_pipeline info
echo
echo "Environment ready. Activate it with: source $VENV/bin/activate"
if [[ "$MODE" == "cpu" ]]; then
  echo "Note: torch-scatter is not installed, so the PhysicsNeMo MeshGraphNet"
  echo "      forward pass is unavailable. Use the gpu mode for that."
fi
