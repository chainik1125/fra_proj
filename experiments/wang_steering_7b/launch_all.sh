#!/usr/bin/env bash
# Launch 6 GPU pods + 1 CPU babysitter for the Wang-steering 7B campaign.
# Cribbed from experiments/tinystories_sleeper/fisher_poc/launch_all.sh.
#
# Required env:
#   RUNPOD_API_KEY   RunPod API key with pod create scope
#   HF_TOKEN         HF write token with access to dmanningcoe/fra-phase1-steering-data
#
# Optional env:
#   BRANCH           default autoresearch/wang-steering-7b
#   REPO_URL         default https://github.com/chainik1125/fra_proj.git
#   GPU_TYPE_IDS     space-separated fallback list (default: L40S L40 A40 RTX_A6000)
#   IMAGE_GPU        default runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
#   IMAGE_CPU        default python:3.12-slim
set -euo pipefail

BRANCH="${BRANCH:-autoresearch/wang-steering-7b}"
REPO_URL="${REPO_URL:-https://github.com/chainik1125/fra_proj.git}"
GPU_TYPE_IDS_DEFAULT="NVIDIA L40S|NVIDIA L40|NVIDIA A40|NVIDIA RTX A6000"
GPU_TYPE_IDS="${GPU_TYPE_IDS:-$GPU_TYPE_IDS_DEFAULT}"
IMAGE_GPU="${IMAGE_GPU:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"
IMAGE_CPU="${IMAGE_CPU:-python:3.12-slim}"

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

# JSON-escape a bash string for embedding in a JSON payload.
json_str() {
    python3 -c "import json,sys; print(json.dumps(sys.stdin.read()))"
}

# Build the per-GPU-pod docker startCommand. Exports the axis values + clones
# branch + runs auto_start_gpu.sh.
make_gpu_cmd() {
    local em="$1" seed="$2"
    cat <<EOF
bash -c "apt-get update >/dev/null 2>&1 && apt-get install -y -q git curl ca-certificates >/dev/null 2>&1 && \
git clone --branch $BRANCH --single-branch $REPO_URL /workspace/fra_proj && \
cd /workspace/fra_proj && \
HF_TOKEN='$HF_TOKEN' RUNPOD_API_KEY='$RUNPOD_API_KEY' RUNPOD_POD_ID=\\\$RUNPOD_POD_ID \
BRANCH='$BRANCH' EM_MODEL='$em' EVAL_SEED='$seed' TOP_N='50' \
bash experiments/wang_steering_7b/auto_start_gpu.sh"
EOF
}

make_cpu_cmd() {
    cat <<EOF
bash -c "apt-get update >/dev/null 2>&1 && apt-get install -y -q git curl ca-certificates python3-pip >/dev/null 2>&1 && \
git clone --branch $BRANCH --single-branch $REPO_URL /workspace/fra_proj && \
cd /workspace/fra_proj && \
HF_TOKEN='$HF_TOKEN' RUNPOD_API_KEY='$RUNPOD_API_KEY' RUNPOD_POD_ID=\\\$RUNPOD_POD_ID \
BRANCH='$BRANCH' \
bash experiments/wang_steering_7b/auto_start_cpu.sh"
EOF
}

# Try a list of gpu_type_ids in order; first that succeeds returns the id.
deploy_gpu_pod() {
    local name="$1" cmd="$2"
    local cmd_json; cmd_json=$(printf '%s' "$cmd" | json_str)
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

deploy_cpu_pod() {
    local name="$1" cmd="$2"
    local cmd_json; cmd_json=$(printf '%s' "$cmd" | json_str)
    local input
    input=$(cat <<JSON
{
  "name": "$name",
  "imageName": "$IMAGE_CPU",
  "cloudType": "SECURE",
  "computeType": "CPU",
  "minVcpuCount": 2,
  "minMemoryInGb": 4,
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
    local resp
    resp=$(curl -sS -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" \
        -H "Content-Type: application/json" -d "$payload" "$GRAPHQL")
    local pid
    pid=$(printf '%s' "$resp" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('data',{}).get('podFindAndDeployOnDemand',{}).get('id') or '')")
    if [ -n "$pid" ]; then
        echo "  CPU pod_id=$pid" >&2
        echo "$pid"
        return 0
    fi
    echo "  CPU FAILED. Response:" >&2
    echo "  $resp" >&2
    return 1
}

# ── Provision ─────────────────────────────────────────────────────────
LAUNCH_LOG="/tmp/wang_steering_launch_$(date +%s).json"
echo "[launch] log → $LAUNCH_LOG"

declare -A POD_IDS
for shard in "${SHARDS[@]}"; do
    em="${shard% *}"
    seed="${shard#* }"
    name="wang-steering-${em}-s${seed}"
    echo "[launch] $name ..."
    cmd=$(make_gpu_cmd "$em" "$seed")
    pid=$(deploy_gpu_pod "$name" "$cmd") || { echo "ABORT"; exit 1; }
    POD_IDS["$name"]="$pid"
done

echo "[launch] babysitter ..."
cpu_cmd=$(make_cpu_cmd)
cpu_pid=$(deploy_cpu_pod "wang-steering-babysitter" "$cpu_cmd") || { echo "ABORT (CPU)"; exit 1; }
POD_IDS["wang-steering-babysitter"]="$cpu_pid"

# Write launch log (pod ids ↔ shards) — for the babysitter and the human.
python3 -c "
import json
ids = {}
" >/dev/null
python3 -c "
import json
ids = {
$(for k in "${!POD_IDS[@]}"; do echo "    '$k': '${POD_IDS[$k]}',"; done)
}
print(json.dumps(ids, indent=2))
" > "$LAUNCH_LOG"

echo
echo "============================================================"
echo "[launch] All 7 pods provisioned (6 GPU + 1 babysitter)."
echo "  Pod IDs (name → id):"
for k in "${!POD_IDS[@]}"; do
    echo "    $k  ${POD_IDS[$k]}"
done
echo
echo "Watch progress:"
echo "  - HF: https://huggingface.co/datasets/dmanningcoe/fra-phase1-steering-data/tree/main/qwen7b/wang_L15_resid_post"
echo "  - Pods (RunPod console): https://www.runpod.io/console/pods"
echo
echo "All GPU pods self-terminate on completion."
echo "Babysitter writes summary.md to HF when all 6 shards are present, then self-terminates."
echo "Launch log: $LAUNCH_LOG"
echo "============================================================"
