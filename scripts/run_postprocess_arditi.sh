#!/usr/bin/env bash
# Post-run pipeline for the Arditi-SAE × our-judge experiment.
#
# 1. SCP per-seed qualitative JSONs from each pod into a flat local layout.
# 2. Judge unjudged entries + per-(sae, em) cross-seed combine via
#    phase1_judge_and_combine.py (uses OPENAI_API_KEY_MATS).
# 3. Render our plots for the Arditi setup (seed-grid + box-plot).
#
# Pod SSH endpoints are pulled from the Runpod GraphQL API at runtime so we
# don't hardcode IP:port that change between pod cycles. Set RP_API_KEY_MATS
# and OPENAI_API_KEY_MATS in the environment before running.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL_ROOT="${LOCAL_ROOT:-/Users/dmitrymanning-coe/Documents/Research/Temporal Crosscoders/temp_xc/plots/2026-05-13_arditi/streams}"
FIG_DIR="${FIG_DIR:-$REPO_ROOT/figures/em_figures}"

declare -A POD_BY_SEED=(
  [42]="oqdxta0ckkcz31"
  [123]="1pykp6us6dmj08"
  [456]="w8o30uuu98w6xb"
)

resolve_ssh() {
  local pod_id="$1"
  /usr/bin/curl -sS -X POST -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${RP_API_KEY_MATS:?set RP_API_KEY_MATS}" \
    -d "{\"query\":\"query Pod(\$id: String!) { pod(input:{podId:\$id}) { runtime { ports { ip privatePort publicPort type isIpPublic } } } }\",\"variables\":{\"id\":\"$pod_id\"}}" \
    https://api.runpod.io/graphql | \
    python3 -c "
import json, sys
ports = json.load(sys.stdin)['data']['pod']['runtime']['ports']
for p in ports:
    if p['type'] == 'tcp' and p.get('isIpPublic'):
        print(p['ip'], p['publicPort']); break
"
}

mkdir -p "$LOCAL_ROOT"

echo "[1/3] SCP per-seed qualitative JSONs"
for seed in 42 123 456; do
  pod="${POD_BY_SEED[$seed]}"
  read -r ip port < <(resolve_ssh "$pod")
  dest="$LOCAL_ROOT/medical_$seed"
  mkdir -p "$dest"
  echo "  seed=$seed pod=$pod  $ip:$port  →  $dest/"
  scp -o StrictHostKeyChecking=no -P "$port" \
    "root@$ip:/workspace/results/qualitative_arditi_medical_evalseed${seed}.json" \
    "$dest/" 2>&1 | tail -1 || echo "  (no file yet for seed=$seed)"
done

echo
echo "[2/3] Judge + combine"
OPENAI_API_KEY="${OPENAI_API_KEY_MATS:?set OPENAI_API_KEY_MATS}" \
  python3 "$REPO_ROOT/phase1_judge_and_combine.py" --stream-root "$LOCAL_ROOT" | tail -20

echo
echo "[3/3] Plot"
combined=$(ls "$LOCAL_ROOT"/gpt4o_combined_L15_resid_post_andyrdt_qwen7b_*_medical.json 2>/dev/null | head -1)
if [[ -z "$combined" ]]; then
  echo "  no combined file produced; aborting plot step"
  exit 1
fi
mkdir -p "$FIG_DIR"
python3 "$REPO_ROOT/scripts/plot_arditi_seed_grid.py" \
  --combined "$combined" \
  --out-grid "$FIG_DIR/arditi_seed_grid_medical" \
  --out-box  "$FIG_DIR/arditi_box_medical"

echo
echo "done. inspect:"
echo "  $FIG_DIR/arditi_seed_grid_medical.png"
echo "  $FIG_DIR/arditi_box_medical.png"
