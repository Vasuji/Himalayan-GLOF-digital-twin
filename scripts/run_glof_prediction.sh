#!/usr/bin/env bash
# Operational-style run: forecast to arrival time, with the Omniverse scene.
#
#   ./scripts/run_glof_prediction.sh                       # toy tier, CPU
#   ./scripts/run_glof_prediction.sh configs/production.yaml
#
# Reuses stored checkpoints when present, so repeated runs cost a forecast and a
# routing pass rather than a full retrain. Generate datasets and train first:
#   glof dataset --config <cfg> --which all
#   glof train   --config <cfg> --which all
set -euo pipefail

CONFIG="${1:-configs/toy.yaml}"
cd "$(dirname "$0")/.."

# physicsnemo.models calls warp.init() at import and aborts if its cache is not
# writable; keep it inside the repository.
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-$PWD/.warp_cache}"
mkdir -p "$WARP_CACHE_PATH"

echo "Configuration: $CONFIG"
python -m glof_pipeline info
python -m glof_pipeline run --config "$CONFIG" --reuse-checkpoints

OUT=$(python - "$CONFIG" <<'PY'
import sys, yaml, pathlib
cfg = yaml.safe_load(pathlib.Path(sys.argv[1]).read_text()) or {}
print((cfg.get("runtime") or {}).get("output_dir", "outputs/toy"))
PY
)

python - "$OUT" <<'PY'
import json, pathlib, sys
manifest = json.loads((pathlib.Path(sys.argv[1]) / "run_manifest.json").read_text())
print(f"\nconfiguration hash : {manifest['config_hash']}")
print(f"total wall time    : {manifest['total_wall_time_s']:.1f} s")
print("\nNVIDIA components:")
for name, value in manifest["backends"]["versions"].items():
    print(f"  {name:24s} {value}")
moraine = manifest["artifacts"].get("moraine", {})
if moraine:
    print(f"\nbreach probability : {moraine.get('breach_probability')}")
    print(f"earliest breach    : {moraine.get('earliest_breach_h')} h")
for name, arrival in (manifest["artifacts"].get("routing", {}).get("arrivals") or {}).items():
    print(f"  arrival {name:22s} {arrival}")
PY
