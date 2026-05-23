#!/usr/bin/env bash
# Launch 6 GPU pods + 1 CPU babysitter for the Wang-steering 7B campaign.
#
# This uses the POD_BOOTSTRAP_B64 + /start.sh & pattern from the 2026-05-22
# constadd campaign — keeping the base image's default entrypoint /start.sh
# running in background (so sshd starts as normal) AND running our bootstrap
# in foreground. Previous attempts that replaced dockerArgs outright caused
# sshd to never start, making pods invisible.
#
# Required env:
#   RUNPOD_API_KEY   RunPod API key
#   HF_TOKEN         HF write token
#
# Optional env:
#   BRANCH           default autoresearch/wang-steering-7b
#   REPO_URL         default https://github.com/chainik1125/fra_proj.git
#   GPU_TYPE_IDS     pipe-separated fallback list
#   IMAGE_GPU        default runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
set -euo pipefail

BRANCH="${BRANCH:-autoresearch/wang-steering-7b}"
REPO_URL="${REPO_URL:-https://github.com/chainik1125/fra_proj.git}"
GPU_TYPE_IDS="${GPU_TYPE_IDS:-NVIDIA L40S|NVIDIA L40|NVIDIA A40|NVIDIA RTX A6000}"
IMAGE_GPU="${IMAGE_GPU:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"

: "${RUNPOD_API_KEY:?RUNPOD_API_KEY not set}"
: "${HF_TOKEN:?HF_TOKEN not set}"

SHARDS=(
    "medical 42"
    "medical 123"
    "medical 456"
    "base 42"
    "base 123"
    "base 456"
)

GRAPHQL="https://api.runpod.io/graphql"

json_str() { python3 -c "import json,sys; print(json.dumps(sys.stdin.read()))"; }

# Per-pod bootstrap script — runs INSIDE the pod as a foreground process.
# Exports its (em_model, eval_seed) inputs from env (RunPod passes through
# whatever we set in `env`), clones the branch, runs auto_start_gpu.sh.
make_pod_inner() {
    local em="$1" seed="$2"
    cat <<EOF
#!/bin/bash
set -eo pipefail
exec > >(stdbuf -oL tee /workspace/run.log) 2>&1
echo "[\$(date -u +%H:%M:%S)] inner bootstrap start"
cd /workspace
[ -d fra_proj ] || git clone --branch '$BRANCH' --single-branch '$REPO_URL' /workspace/fra_proj
cd /workspace/fra_proj
git fetch origin && git checkout '$BRANCH' && git pull --ff-only
HF_TOKEN='$HF_TOKEN' RUNPOD_API_KEY='$RUNPOD_API_KEY' RUNPOD_POD_ID="\$RUNPOD_POD_ID" \\
BRANCH='$BRANCH' EM_MODEL='$em' EVAL_SEED='$seed' TOP_N='50' \\
bash experiments/wang_steering_7b/auto_start_gpu.sh
EOF
}

# dockerArgs wrapper — base64-decodes the inner bootstrap, runs /start.sh
# in background (so sshd starts), then runs our inner bootstrap.
make_docker_args() {
    local b64="$1"
    cat <<EOF
bash -c "echo $b64 | base64 -d > /start_user.sh && chmod +x /start_user.sh && /start.sh & sleep 30 && /start_user.sh"
EOF
}

deploy_gpu_pod() {
    local name="$1" em="$2" seed="$3"
    local inner; inner=$(make_pod_inner "$em" "$seed")
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
  "imageName": "$IMAGE_GPU",
  "cloudType": "SECURE",
  "gpuTypeId": "$gpu_type",
  "gpuCount": 1,
  "minVcpuCount": 4,
  "minMemoryInGb": 24,
  "containerDiskInGb": 60,
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
    done < <(printf '%s' "$GPU_TYPE_IDS" | tr '|' '\n')
    echo "    FAILED on all GPU types. Last response:" >&2
    echo "    $last_resp" >&2
    return 1
}

# ── Provision ─────────────────────────────────────────────────────────
LAUNCH_LOG="/tmp/wang_steering_launch_$(date +%s).json"
echo "[launch] log → $LAUNCH_LOG"

POD_NAMES=()
POD_IDS=()
for shard in "${SHARDS[@]}"; do
    em="${shard% *}"
    seed="${shard#* }"
    name="wang-steering-${em}-s${seed}"
    echo "[launch] $name ..."
    pid=$(deploy_gpu_pod "$name" "$em" "$seed") || { echo "ABORT"; exit 1; }
    POD_NAMES+=("$name")
    POD_IDS+=("$pid")
done

{
    echo "{"
    for i in "${!POD_NAMES[@]}"; do
        comma=","
        [ "$i" -eq "$((${#POD_NAMES[@]} - 1))" ] && comma=""
        printf '    "%s": "%s"%s\n' "${POD_NAMES[$i]}" "${POD_IDS[$i]}" "$comma"
    done
    echo "}"
} > "$LAUNCH_LOG"

echo
echo "============================================================"
echo "[launch] All 6 GPU pods provisioned."
echo "  Pod IDs (name → id):"
for i in "${!POD_NAMES[@]}"; do
    echo "    ${POD_NAMES[$i]}  ${POD_IDS[$i]}"
done
echo
echo "Watch progress:"
echo "  - HF: https://huggingface.co/datasets/dmanningcoe/fra-phase1-steering-data/tree/main/qwen7b/wang_L15_resid_post"
echo
echo "Troubleshooter / babysitter are launched separately."
echo "Launch log: $LAUNCH_LOG"
echo "============================================================"
