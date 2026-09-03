#!/usr/bin/env bash
# Full toy-tier run on CPU: real physics, real training, no GPU and no downloads.
set -euo pipefail
cd "$(dirname "$0")/.."

export WARP_CACHE_PATH="${WARP_CACHE_PATH:-$PWD/.warp_cache}"
mkdir -p "$WARP_CACHE_PATH"

python -m glof_pipeline run --config configs/toy.yaml "$@"
echo
echo "Outputs in outputs/toy/ ; timings and backends in outputs/toy/run_manifest.json"
