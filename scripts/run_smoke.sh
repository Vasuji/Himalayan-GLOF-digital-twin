#!/usr/bin/env bash
# Smallest end-to-end run: every pipeline stage at reduced size. Use this to check
# that a change has not broken a stage-to-stage interface before spending GPU time.
set -euo pipefail
cd "$(dirname "$0")/.."

export WARP_CACHE_PATH="${WARP_CACHE_PATH:-$PWD/.warp_cache}"
mkdir -p "$WARP_CACHE_PATH"

python -m glof_pipeline info
python -m glof_pipeline run --config configs/smoke.yaml "$@"

MANIFEST=outputs/smoke/run_manifest.json
if [[ -f "$MANIFEST" ]]; then
  echo
  echo "NVIDIA components used by this run:"
  python - "$MANIFEST" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1]))
backends = manifest["backends"]
for name, value in backends["versions"].items():
    print(f"  {name:24s} {value}")
for component in ("physicsnemo", "earth2studio", "torch_geometric", "torch_scatter", "omniverse_usd"):
    print(f"  {component:24s} {'available' if backends.get(component) else 'MISSING'}")
print(f"  config hash: {manifest['config_hash']}")
PY
fi
