#!/usr/bin/env bash
# Provision N GPU pods (one per seed) + 1 CPU pod (babysitter) on RunPod via
# the GraphQL API.  Each pod's container start command pulls the branch and
# runs the appropriate auto_start_*.sh entrypoint.  No SSH-in needed after
# provisioning; pods self-stop on completion.
#
# Required env:
#   RUNPOD_API_KEY   RunPod API key
#   HF_TOKEN         HF write token (with access to HF_REPO)
#
# Optional env (with defaults):
#   SEEDS            "0 1 2 3 4"
#   GPU_TYPE_ID      "NVIDIA L40S"   (or "NVIDIA A40")
#   HF_REPO          dmanningcoe/fisher-poc-tinystories-sleeper
#   BRANCH           dmitry/fisher-poc
#   REPO_URL         https://github.com/chainik1125/fra_proj.git
#   IMAGE_GPU        runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
#   IMAGE_CPU        python:3.12-slim
set -euo pipefail

SEEDS="${SEEDS:-0 1 2 3 4}"
GPU_TYPE_ID="${GPU_TYPE_ID:-NVIDIA L40S}"
HF_REPO="${HF_REPO:-dmanningcoe/fisher-poc-tinystories-sleeper}"
BRANCH="${BRANCH:-dmitry/fisher-poc}"
REPO_URL="${REPO_URL:-https://github.com/chainik1125/fra_proj.git}"
IMAGE_GPU="${IMAGE_GPU:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"
IMAGE_CPU="${IMAGE_CPU:-python:3.12-slim}"

if [[ -z "${RUNPOD_API_KEY:-}" ]]; then
  echo "ERROR: RUNPOD_API_KEY not set." >&2; exit 1
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN not set." >&2; exit 1
fi

GRAPHQL="https://api.runpod.io/graphql"

# Each pod's docker startCommand. The pod clones the branch and runs the
# entrypoint. SEEDS is per-pod for GPU pods, full list for the CPU pod.
make_gpu_cmd() {
  local seed="$1"
  cat <<EOF
bash -c "apt-get update >/dev/null && apt-get install -y -q git curl >/dev/null && \
  git clone --branch $BRANCH --single-branch $REPO_URL /workspace/fra_proj && \
  cd /workspace/fra_proj && \
  HF_TOKEN='$HF_TOKEN' RUNPOD_API_KEY='$RUNPOD_API_KEY' SEEDS='$seed' HF_REPO='$HF_REPO' SELF_STOP=1 \
  bash experiments/tinystories_sleeper/fisher_poc/auto_start_gpu.sh"
EOF
}

make_cpu_cmd() {
  cat <<EOF
bash -c "apt-get update >/dev/null && apt-get install -y -q git curl >/dev/null && \
  git clone --branch $BRANCH --single-branch $REPO_URL /workspace/fra_proj && \
  cd /workspace/fra_proj && \
  HF_TOKEN='$HF_TOKEN' RUNPOD_API_KEY='$RUNPOD_API_KEY' SEEDS='$SEEDS' HF_REPO='$HF_REPO' SELF_STOP=1 \
  bash experiments/tinystories_sleeper/fisher_poc/auto_start_cpu.sh"
EOF
}

# Helper: call the RunPod GraphQL `podFindAndDeployOnDemand` mutation.
gql_deploy() {
  local name="$1"
  local image="$2"
  local gpu_type="$3"     # empty string for CPU pods
  local gpu_count="$4"
  local cmd="$5"
  local cpu_only="$6"     # "true" / "false"

  # Escape cmd for JSON.
  local cmd_json
  cmd_json=$(printf '%s' "$cmd" | python3 -c "import json,sys; print(json.dumps(sys.stdin.read()))")

  local input
  if [[ "$cpu_only" == "true" ]]; then
    input=$(cat <<JSON
{
  "name": "$name",
  "imageName": "$image",
  "cloudType": "SECURE",
  "computeType": "CPU",
  "minVcpuCount": 2,
  "minMemoryInGb": 8,
  "containerDiskInGb": 20,
  "volumeInGb": 0,
  "dockerArgs": $cmd_json,
  "ports": "22/tcp",
  "startSsh": true
}
JSON
)
  else
    input=$(cat <<JSON
{
  "name": "$name",
  "imageName": "$image",
  "cloudType": "SECURE",
  "gpuTypeId": "$gpu_type",
  "gpuCount": $gpu_count,
  "minVcpuCount": 4,
  "minMemoryInGb": 24,
  "containerDiskInGb": 40,
  "volumeInGb": 0,
  "dockerArgs": $cmd_json,
  "ports": "22/tcp",
  "startSsh": true
}
JSON
)
  fi

  local payload
  payload=$(python3 -c "
import json, sys
inp = json.loads(sys.argv[1])
q = '''mutation Deploy(\$input: PodFindAndDeployOnDemandInput!) {
  podFindAndDeployOnDemand(input: \$input) { id name desiredStatus }
}'''
print(json.dumps({'query': q, 'variables': {'input': inp}}))
" "$input")

  curl -sS -X POST \
    -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -H "Content-Type: application/json" \
    -d "$payload" \
    "$GRAPHQL"
}

LAUNCH_LOG="/tmp/fisher_poc_launch_$(date +%s).log"
echo "[launch] log -> $LAUNCH_LOG"

POD_IDS=()
for SEED in $SEEDS; do
  echo "[launch] creating GPU pod for seed=$SEED ..."
  cmd=$(make_gpu_cmd "$SEED")
  resp=$(gql_deploy "fisher-poc-seed${SEED}" "$IMAGE_GPU" "$GPU_TYPE_ID" 1 "$cmd" "false")
  echo "$resp" | tee -a "$LAUNCH_LOG" >/dev/null
  pid=$(printf '%s' "$resp" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('data',{}).get('podFindAndDeployOnDemand',{}).get('id') or '')")
  if [[ -z "$pid" ]]; then
    echo "[launch] FAILED for seed=$SEED. Response:" >&2
    echo "$resp" >&2
    exit 1
  fi
  POD_IDS+=("$pid:$SEED")
  echo "[launch] seed=$SEED pod_id=$pid"
done

echo "[launch] creating CPU babysitter pod ..."
cmd=$(make_cpu_cmd)
resp=$(gql_deploy "fisher-poc-babysit" "$IMAGE_CPU" "" 0 "$cmd" "true")
echo "$resp" | tee -a "$LAUNCH_LOG" >/dev/null
cpu_pid=$(printf '%s' "$resp" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('data',{}).get('podFindAndDeployOnDemand',{}).get('id') or '')")
if [[ -z "$cpu_pid" ]]; then
  echo "[launch] FAILED to launch CPU pod. Response: $resp" >&2
  exit 1
fi

echo
echo "============================================================"
echo "[launch] All pods provisioned."
echo "  GPU pods (seed:pod_id):"
for entry in "${POD_IDS[@]}"; do
  echo "    $entry"
done
echo "  CPU babysitter: $cpu_pid"
echo
echo "Watch progress:"
echo "  - HF dataset: https://huggingface.co/datasets/$HF_REPO"
echo "  - Pod list:   runpodctl pod list"
echo
echo "All pods self-stop on completion."
echo "Babysitter writes summary.md to HF when all seeds are present."
echo "Full launch responses logged to $LAUNCH_LOG."
echo "============================================================"
