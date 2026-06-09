#!/usr/bin/env bash
# Launch the 19-cell method x hookpoint x layer grid, one RunPod GPU pod per cell.
# DISK is sized to the on-disk memmap activation pool: union (2 models) ~109GB -> 220,
# single-model SAE ~55GB -> 140, DoM (no pool, incremental) -> 60.
set -uo pipefail
cd "$(dirname "$0")"
export RUNPOD_API_KEY="${RP_API_KEY_MATS}"
export RUN_SEED=7
export PY_SCRIPT=run_steer.py

CONFIGS=$(ls configs/grid_*.yaml | xargs -n1 basename | sed 's/\.yaml$//')
ok=0; fail=0
for cfg in $CONFIGS; do
  case "$cfg" in
    grid_dom_*)       disk=60  ;;
    grid_cs_union_*)  disk=220 ;;
    *)                disk=140 ;;
  esac
  pod="rs-$(echo "$cfg" | sed 's/^grid_//; s/_/-/g')"      # rs-fra-base-ln1-l0 etc.
  pod=$(echo "$pod" | tr 'A-Z' 'a-z')
  POD_NAME="$pod" CONFIG="$cfg" OUT_JSON="${cfg}_results.json" RUN_LOG="${pod}_run.log" \
    DISK="$disk" MIN_MEM=24 bash launch_pod_mts.sh >/tmp/launch_${pod}.out 2>&1
  if [ -f "/tmp/${pod}_pod_id.txt" ] && [ -s "/tmp/${pod}_pod_id.txt" ]; then
    echo "[OK ] $cfg -> $pod (disk=$disk) id=$(cat /tmp/${pod}_pod_id.txt)"; ok=$((ok+1))
  else
    echo "[FAIL] $cfg -> $pod : $(tail -1 /tmp/launch_${pod}.out)"; fail=$((fail+1))
  fi
  sleep 3
done
echo "=== launched ok=$ok fail=$fail ==="
