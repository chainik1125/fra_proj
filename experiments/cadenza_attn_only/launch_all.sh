#!/usr/bin/env bash
# Launch ONE H100 GPU training pod + ONE babysitter pod for the Cadenza
# attention-only-A campaign (Variant A, stage 1).
#
# Uses the POD_BOOTSTRAP_B64 + `/start.sh &` dockerArgs pattern (keeps the
# base image's default entrypoint running so sshd starts, then runs our
# inner bootstrap in the foreground).
#
# GPU pod   → runs experiments/cadenza_attn_only/auto_start_gpu.sh (MODE=gate:
#             smoke then, if it passes, the full 1-epoch run — same pod).
# Babysitter→ runs auto_start_cpu.sh → babysitter.py (polls HF, summary, stop).
#
# NOTE: RunPod `computeType: CPU` returns SUPPLY_CONSTRAINT on this account
# (see reference_runpod_api memory), so the babysitter runs on a cheap GPU
# pod instead. It only polls HF — any small secure GPU is fine.
#
# Required env:
#   RUNPOD_API_KEY   RunPod API key
#   HF_TOKEN         HF write token (dmanningcoe)
#
# Optional env:
#   BRANCH           default autoresearch/cadenza-attn-only
#   REPO_URL         default https://github.com/chainik1125/fra_proj.git
#   GPU_TYPE_IDS     pipe-separated H100 fallback list
#   BABY_GPU_IDS     pipe-separated cheap-GPU fallback list for the babysitter
#   IMAGE_GPU        cu124 PyTorch base for the trainer
#   IMAGE_CPU        slim image for the babysitter
#   MODE             gate | smoke | full   (default gate)
set -euo pipefail

BRANCH="${BRANCH:-autoresearch/cadenza-attn-only}"
REPO_URL="${REPO_URL:-https://github.com/chainik1125/fra_proj.git}"
GPU_TYPE_IDS="${GPU_TYPE_IDS:-NVIDIA H100 PCIe|NVIDIA H100 80GB HBM3|NVIDIA H100 NVL}"
BABY_GPU_IDS="${BABY_GPU_IDS:-NVIDIA RTX A4000|NVIDIA RTX A5000|NVIDIA L4|NVIDIA L40S}"
IMAGE_GPU="${IMAGE_GPU:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"
IMAGE_CPU="${IMAGE_CPU:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"
MODE="${MODE:-gate}"

: "${RUNPOD_API_KEY:?RUNPOD_API_KEY not set}"
: "${HF_TOKEN:?HF_TOKEN not set}"

GRAPHQL="https://api.runpod.io/graphql"
json_str() { python3 -c "import json,sys; print(json.dumps(sys.stdin.read()))"; }

# ── Inner bootstraps (run INSIDE each pod as a foreground process) ───────
make_gpu_inner() {
    cat <<EOF
#!/bin/bash
set -eo pipefail
exec > >(stdbuf -oL tee /workspace/run.log) 2>&1
echo "[\$(date -u +%H:%M:%S)] inner gpu bootstrap start"
cd /workspace
[ -d fra_proj ] || git clone --branch '$BRANCH' --single-branch '$REPO_URL' /workspace/fra_proj
cd /workspace/fra_proj
git fetch origin && git checkout '$BRANCH' && git pull --ff-only
HF_TOKEN='$HF_TOKEN' RUNPOD_API_KEY='$RUNPOD_API_KEY' RUNPOD_POD_ID="\$RUNPOD_POD_ID" \\
BRANCH='$BRANCH' MODE='$MODE' \\
bash experiments/cadenza_attn_only/auto_start_gpu.sh
EOF
}

make_cpu_inner() {
    cat <<EOF
#!/bin/bash
set -eo pipefail
exec > >(stdbuf -oL tee /workspace/babysitter.log) 2>&1
echo "[\$(date -u +%H:%M:%S)] inner babysitter bootstrap start"
cd /workspace
[ -d fra_proj ] || git clone --branch '$BRANCH' --single-branch '$REPO_URL' /workspace/fra_proj
cd /workspace/fra_proj
git fetch origin && git checkout '$BRANCH' && git pull --ff-only
HF_TOKEN='$HF_TOKEN' RUNPOD_API_KEY='$RUNPOD_API_KEY' RUNPOD_POD_ID="\$RUNPOD_POD_ID" \\
BRANCH='$BRANCH' \\
bash experiments/cadenza_attn_only/auto_start_cpu.sh
EOF
}

# dockerArgs wrapper — base64-decode the inner bootstrap, run /start.sh in
# background (so sshd starts), then run our inner bootstrap.
make_docker_args() {
    local b64="$1"
    cat <<EOF
bash -c "echo $b64 | base64 -d > /start_user.sh && chmod +x /start_user.sh && /start.sh & sleep 30 && /start_user.sh"
EOF
}

# deploy_pod <name> <image> <gpu_type_ids|> <container_disk_gb> <inner_fn>
deploy_pod() {
    local name="$1" image="$2" gpu_ids="$3" disk="$4" inner_fn="$5"
    local inner; inner=$("$inner_fn")
    local b64; b64=$(printf '%s' "$inner" | base64 | tr -d '\n')
    local docker_args; docker_args=$(make_docker_args "$b64")
    local cmd_json; cmd_json=$(printf '%s' "$docker_args" | json_str)
    local last_resp=""
    while IFS= read -r gpu_type; do
        [ -z "$gpu_type" ] && continue
        local input
        input=$(cat <<JSON
{
  "name": "$name",
  "imageName": "$image",
  "cloudType": "SECURE",
  "gpuTypeId": "$gpu_type",
  "gpuCount": 1,
  "minVcpuCount": 8,
  "minMemoryInGb": 32,
  "containerDiskInGb": $disk,
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
        local resp
        resp=$(curl -sS -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" \
            -H "Content-Type: application/json" -d "$payload" "$GRAPHQL")
        last_resp="$resp"
        local pid
        pid=$(printf '%s' "$resp" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('data',{}).get('podFindAndDeployOnDemand',{}).get('id') or '')")
        if [ -n "$pid" ]; then
            echo "    [$gpu_type] pod_id=$pid" >&2
            echo "$pid"
            return 0
        fi
    done < <(printf '%s' "$gpu_ids" | tr '|' '\n')
    echo "    FAILED on all GPU types. Last response:" >&2
    echo "    $last_resp" >&2
    return 1
}

# ── Provision ───────────────────────────────────────────────────────────
LAUNCH_LOG="/tmp/cadenza_attn_launch_$(date +%s).json"   # /tmp ONLY — has pod IDs
echo "[launch] mode=$MODE  log → $LAUNCH_LOG"

GPU_NAME="cadenza-attn-A"
echo "[launch] $GPU_NAME (H100, training) ..."
GPU_PID=$(deploy_pod "$GPU_NAME" "$IMAGE_GPU" "$GPU_TYPE_IDS" 120 make_gpu_inner) || { echo "ABORT (GPU)"; exit 1; }

BABY_NAME="cadenza-attn-baby"
echo "[launch] $BABY_NAME (babysitter) ..."
BABY_PID=$(deploy_pod "$BABY_NAME" "$IMAGE_CPU" "$BABY_GPU_IDS" 20 make_cpu_inner) || { echo "WARN: babysitter failed to provision — GPU pod still self-completes via HF"; BABY_PID=""; }

{
    echo "{"
    printf '    "%s": "%s",\n' "$GPU_NAME" "$GPU_PID"
    printf '    "%s": "%s"\n' "$BABY_NAME" "$BABY_PID"
    echo "}"
} > "$LAUNCH_LOG"

echo
echo "============================================================"
echo "[launch] Provisioned:"
echo "    $GPU_NAME   $GPU_PID   (H100 trainer, MODE=$MODE)"
echo "    $BABY_NAME  ${BABY_PID:-<none>}   (babysitter)"
echo
echo "Watch progress:"
echo "  - HF model:   https://huggingface.co/dmanningcoe/dolphin-llama3-8B-sleeper-attn-only-A"
echo "  - HF dataset: https://huggingface.co/datasets/dmanningcoe/fra-phase1-steering-data/tree/main/cadenza_attn_only/variantA"
echo "Launch log: $LAUNCH_LOG"
echo "============================================================"
