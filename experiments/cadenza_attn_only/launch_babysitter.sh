#!/usr/bin/env bash
# Launch ONE orchestrator pod for the Cadenza attention-only-A FULL run.
#
# This is the ONLY local action for the durable phase. The pod runs
# auto_start_cpu.sh → babysitter.py, the scripted DAG that LAUNCHES the H100
# trainer itself (full 1-epoch run), polls HF, relaunches on death/stall
# (4-launch hard cap), and on completion writes summary.md + terminates the GPU
# pod + self-terminates. After this command returns, LOCAL CAN CLOSE.
#
# Run this ONLY after the smoke gate (launch_smoke.sh) has been reviewed and
# confirmed by the human.
#
# NB: RunPod `computeType: CPU` returns SUPPLY_CONSTRAINT on this account
# (reference_runpod_api memory), so the orchestrator runs on a cheap GPU pod.
# It only polls HF + calls the RunPod API — any small secure GPU is fine.
#
# Required env:
#   RUNPOD_API_KEY   RunPod API key (also handed into the pod env)
#   HF_TOKEN         HF write token (dmanningcoe; also handed into the pod env)
# Optional env:
#   BRANCH           default autoresearch/cadenza-attn-only
#   REPO_URL         default https://github.com/chainik1125/fra_proj.git
#   BABY_GPU_IDS     pipe-separated cheap-GPU fallback list for the orchestrator
#   IMAGE_CPU        image for the orchestrator pod (has python3 + pip)
#   GPU_TYPE_IDS     H100 fallback the orchestrator uses for the TRAINER
#                    (forwarded into the pod env)
#   POLL_SEC STALL_TIMEOUT_SEC MAX_LAUNCHES GPU_CONTAINER_GB IMAGE_GPU
#                    orchestrator knobs (forwarded into the pod env)
set -euo pipefail

BRANCH="${BRANCH:-autoresearch/cadenza-attn-only}"
REPO_URL="${REPO_URL:-https://github.com/chainik1125/fra_proj.git}"
BABY_GPU_IDS="${BABY_GPU_IDS:-NVIDIA RTX A4000|NVIDIA RTX A5000|NVIDIA L4|NVIDIA L40S}"
IMAGE_CPU="${IMAGE_CPU:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"
# Trainer-side knobs forwarded into the orchestrator pod env (babysitter.py reads them):
GPU_TYPE_IDS="${GPU_TYPE_IDS:-NVIDIA H100 PCIe|NVIDIA H100 80GB HBM3|NVIDIA H100 NVL}"
IMAGE_GPU="${IMAGE_GPU:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"
POLL_SEC="${POLL_SEC:-120}"
STALL_TIMEOUT_SEC="${STALL_TIMEOUT_SEC:-5400}"
MAX_LAUNCHES="${MAX_LAUNCHES:-4}"
GPU_CONTAINER_GB="${GPU_CONTAINER_GB:-120}"

: "${RUNPOD_API_KEY:?RUNPOD_API_KEY not set}"
: "${HF_TOKEN:?HF_TOKEN not set}"

GRAPHQL="https://api.runpod.io/graphql"
json_str() { python3 -c "import json,sys; print(json.dumps(sys.stdin.read()))"; }

# Inner bootstrap forwards every orchestrator knob into the pod env, then runs
# auto_start_cpu.sh (which runs babysitter.py).
make_inner() {
    cat <<EOF
#!/bin/bash
set -eo pipefail
exec > >(stdbuf -oL tee /workspace/babysitter.log) 2>&1
echo "[\$(date -u +%H:%M:%S)] inner orchestrator bootstrap start"
cd /workspace
[ -d fra_proj ] || git clone --branch '$BRANCH' --single-branch '$REPO_URL' /workspace/fra_proj
cd /workspace/fra_proj
git fetch origin && git checkout '$BRANCH' && git pull --ff-only
HF_TOKEN='$HF_TOKEN' RUNPOD_API_KEY='$RUNPOD_API_KEY' RUNPOD_POD_ID="\$RUNPOD_POD_ID" \\
BRANCH='$BRANCH' REPO_URL='$REPO_URL' \\
GPU_TYPE_IDS='$GPU_TYPE_IDS' IMAGE_GPU='$IMAGE_GPU' \\
POLL_SEC='$POLL_SEC' STALL_TIMEOUT_SEC='$STALL_TIMEOUT_SEC' \\
MAX_LAUNCHES='$MAX_LAUNCHES' GPU_CONTAINER_GB='$GPU_CONTAINER_GB' \\
bash experiments/cadenza_attn_only/auto_start_cpu.sh
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
  "name": "cadenza-attn-baby",
  "imageName": "$IMAGE_CPU",
  "cloudType": "SECURE",
  "gpuTypeId": "$gpu_type",
  "gpuCount": 1,
  "minVcpuCount": 4,
  "minMemoryInGb": 16,
  "containerDiskInGb": 20,
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
    done < <(printf '%s' "$BABY_GPU_IDS" | tr '|' '\n')
    echo "    FAILED on all GPU types. Last response:" >&2
    echo "    $last_resp" >&2
    return 1
}

LAUNCH_LOG="/tmp/cadenza_attn_baby_$(date +%s).json"   # /tmp ONLY — has pod IDs
echo "[baby] launching cadenza-attn-baby (orchestrator) ..."
PID=$(deploy) || { echo "ABORT"; exit 1; }
printf '{ "cadenza-attn-baby": "%s" }\n' "$PID" > "$LAUNCH_LOG"

echo
echo "============================================================"
echo "[baby] orchestrator pod cadenza-attn-baby = $PID"
echo "It will launch the H100 trainer (full run), poll HF, relaunch on"
echo "death/stall (cap $MAX_LAUNCHES launches), then summary + terminate all + self-stop."
echo "LOCAL CAN CLOSE NOW. Watch progress on HF:"
echo "  - model:   https://huggingface.co/dmanningcoe/dolphin-llama3-8B-sleeper-attn-only-A"
echo "  - dataset: https://huggingface.co/datasets/dmanningcoe/fra-phase1-steering-data/tree/main/cadenza_attn_only/variantA"
echo "  - orchestrator log: runpodctl pod logs $PID"
echo "If cadenza_attn_only/variantA/status_stalled.json appears → human intervention."
echo "Launch log: $LAUNCH_LOG"
echo "============================================================"
