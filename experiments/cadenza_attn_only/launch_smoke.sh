#!/usr/bin/env bash
# Launch ONE H100 GPU pod for the Cadenza attention-only-A SMOKE GATE.
#
# This is the human-in-the-loop gate: the pod runs auto_start_gpu.sh with
# SMOKE=1 (tiny dataset subset + ~30 steps + small eval subset, pushing to
# throwaway `*-smoke` HF repos), then self-terminates. The HUMAN reviews the
# pod's run.log to confirm: fork checked out at the pinned SHA, LoRA attached
# q/k/v/o ONLY (the script asserts + prints the target-module list +
# trainable-param count), a few steps ran, and merge+push+eval are wired. ONLY
# after that confirmation do we hand the full run to the babysitter
# (launch_babysitter.sh).
#
# Does NOT launch a babysitter and does NOT start the full run.
#
# Required env:
#   RUNPOD_API_KEY   RunPod API key
#   HF_TOKEN         HF write token (dmanningcoe)
# Optional env:
#   BRANCH           default autoresearch/cadenza-attn-only
#   REPO_URL         default https://github.com/chainik1125/fra_proj.git
#   GPU_TYPE_IDS     pipe-separated H100 fallback list
#   IMAGE_GPU        cu124 PyTorch base
set -euo pipefail

BRANCH="${BRANCH:-autoresearch/cadenza-attn-only}"
REPO_URL="${REPO_URL:-https://github.com/chainik1125/fra_proj.git}"
GPU_TYPE_IDS="${GPU_TYPE_IDS:-NVIDIA H100 PCIe|NVIDIA H100 80GB HBM3|NVIDIA H100 NVL}"
IMAGE_GPU="${IMAGE_GPU:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"

: "${RUNPOD_API_KEY:?RUNPOD_API_KEY not set}"
: "${HF_TOKEN:?HF_TOKEN not set}"

GRAPHQL="https://api.runpod.io/graphql"
json_str() { python3 -c "import json,sys; print(json.dumps(sys.stdin.read()))"; }

make_inner() {
    cat <<EOF
#!/bin/bash
set -eo pipefail
exec > >(stdbuf -oL tee /workspace/run.log) 2>&1
echo "[\$(date -u +%H:%M:%S)] inner smoke bootstrap start"
cd /workspace
[ -d fra_proj ] || git clone --branch '$BRANCH' --single-branch '$REPO_URL' /workspace/fra_proj
cd /workspace/fra_proj
git fetch origin && git checkout '$BRANCH' && git pull --ff-only
HF_TOKEN='$HF_TOKEN' RUNPOD_API_KEY='$RUNPOD_API_KEY' RUNPOD_POD_ID="\$RUNPOD_POD_ID" \\
BRANCH='$BRANCH' SMOKE=1 \\
bash experiments/cadenza_attn_only/auto_start_gpu.sh
EOF
}

make_docker_args() {
    local b64="$1"
    cat <<EOF
bash -c "echo $b64 | base64 -d > /start_user.sh && chmod +x /start_user.sh && /start.sh & sleep 30 && /start_user.sh"
EOF
}

deploy() {
    local inner; inner=$(make_inner)
    local b64; b64=$(printf '%s' "$inner" | base64 | tr -d '\n')
    local cmd_json; cmd_json=$(make_docker_args "$b64" | json_str)
    local last_resp=""
    while IFS= read -r gpu_type; do
        [ -z "$gpu_type" ] && continue
        local input
        input=$(cat <<JSON
{
  "name": "cadenza-attn-A-smoke",
  "imageName": "$IMAGE_GPU",
  "cloudType": "SECURE",
  "gpuTypeId": "$gpu_type",
  "gpuCount": 1,
  "minVcpuCount": 8,
  "minMemoryInGb": 32,
  "containerDiskInGb": 120,
  "volumeInGb": 0,
  "dockerArgs": $cmd_json,
  "ports": "22/tcp",
  "startSsh": true
}
JSON
)
        local payload
        payload=$(python3 -c "
import json, sys
inp = json.loads(sys.argv[1])
q = '''mutation Deploy(\$input: PodFindAndDeployOnDemandInput!) {
  podFindAndDeployOnDemand(input: \$input) { id name desiredStatus }
}'''
print(json.dumps({'query': q, 'variables': {'input': inp}}))
" "$input")
        local resp; resp=$(curl -sS -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" \
            -H "Content-Type: application/json" -d "$payload" "$GRAPHQL")
        last_resp="$resp"
        local pid; pid=$(printf '%s' "$resp" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('data',{}).get('podFindAndDeployOnDemand',{}).get('id') or '')")
        if [ -n "$pid" ]; then
            echo "    [$gpu_type] pod_id=$pid" >&2
            echo "$pid"; return 0
        fi
    done < <(printf '%s' "$GPU_TYPE_IDS" | tr '|' '\n')
    echo "    FAILED on all GPU types. Last response:" >&2
    echo "    $last_resp" >&2
    return 1
}

LAUNCH_LOG="/tmp/cadenza_attn_smoke_$(date +%s).json"   # /tmp ONLY — has pod IDs
echo "[smoke] launching cadenza-attn-A-smoke ..."
PID=$(deploy) || { echo "ABORT"; exit 1; }
printf '{ "cadenza-attn-A-smoke": "%s" }\n' "$PID" > "$LAUNCH_LOG"

echo
echo "============================================================"
echo "[smoke] pod cadenza-attn-A-smoke = $PID  (SMOKE=1, self-terminates)"
echo "Review the gate by tailing the pod's run.log:"
echo "  runpodctl pod logs $PID    (or SSH and: tail -f /workspace/run.log)"
echo "Confirm: fork checked out at pinned SHA, LoRA target_modules = q/k/v/o ONLY (no gate/up/down),"
echo "  trainable-param count printed, ~30 steps ran, merge+push+eval wired."
echo "Smoke eval lands at: dmanningcoe/fra-phase1-steering-data ::"
echo "  cadenza_attn_only/variantA/eval_results_smoke.json (+ run_smoke.log)"
echo "ONLY after you confirm → launch the full durable run: launch_babysitter.sh"
echo "Launch log: $LAUNCH_LOG"
echo "============================================================"
