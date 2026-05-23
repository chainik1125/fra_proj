#!/usr/bin/env bash
# Provision ONLY the two kickoff pods for the Fisher POC:
#   1. GPU bootstrap pod  — trains SAEs, uploads to HF, self-terminates.
#   2. CPU babysitter pod — waits for SAEs on HF, then itself provisions
#      one GPU pod per seed for the experiments, then writes summary.md.
#
# Experiment pods are NOT created here.  They are created by the
# babysitter once the SAE bootstrap completes — that's how we avoid
# paying for GPU pods that sit idle waiting for the SAEs.
#
# Required env:
#   RUNPOD_API_KEY   RunPod API key
#   HF_TOKEN         HF write token (with access to HF_REPO)
#
# Optional env (with defaults):
#   SEEDS            "0 1 2 3 4"
#   GPU_TYPE_ID      "NVIDIA L40S"
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
# CPU pods often error with SUPPLY_CONSTRAINT for this account, so we
# default the babysitter to the same GPU type as bootstrap (still cheap
# at ~$0.86/hr × ~1.5 hr).
BABYSITTER_GPU_TYPE_ID="${BABYSITTER_GPU_TYPE_ID:-$GPU_TYPE_ID}"

if [[ -z "${RUNPOD_API_KEY:-}" ]]; then
  echo "ERROR: RUNPOD_API_KEY not set." >&2; exit 1
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN not set." >&2; exit 1
fi

GRAPHQL="https://api.runpod.io/graphql"

bootstrap_cmd() {
  cat <<EOF
bash -c "apt-get update >/dev/null && apt-get install -y -q git curl >/dev/null && \
  git clone --branch $BRANCH --single-branch $REPO_URL /workspace/fra_proj && \
  cd /workspace/fra_proj && \
  HF_TOKEN='$HF_TOKEN' RUNPOD_API_KEY='$RUNPOD_API_KEY' HF_REPO='$HF_REPO' \
  bash experiments/tinystories_sleeper/fisher_poc/auto_start_bootstrap.sh"
EOF
}

babysitter_cmd() {
  cat <<EOF
bash -c "apt-get update >/dev/null && apt-get install -y -q git curl >/dev/null && \
  git clone --branch $BRANCH --single-branch $REPO_URL /workspace/fra_proj && \
  cd /workspace/fra_proj && \
  HF_TOKEN='$HF_TOKEN' RUNPOD_API_KEY='$RUNPOD_API_KEY' SEEDS='$SEEDS' \
  HF_REPO='$HF_REPO' GPU_TYPE_ID='$GPU_TYPE_ID' IMAGE_GPU='$IMAGE_GPU' \
  BRANCH='$BRANCH' REPO_URL='$REPO_URL' SELF_STOP=1 \
  bash experiments/tinystories_sleeper/fisher_poc/auto_start_cpu.sh"
EOF
}

# Helper: call podFindAndDeployOnDemand. $1=name, $2=image, $3=gpuType (or empty),
# $4=cpu_only ("true"/"false"), $5=docker start command.
gql_deploy() {
  local name="$1"; local image="$2"; local gpu_type="$3"
  local cpu_only="$4"; local cmd="$5"
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
  "gpuCount": 1,
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

echo "[launch] creating GPU bootstrap pod (trains SAEs) ..."
resp=$(gql_deploy "fisher-poc-bootstrap" "$IMAGE_GPU" "$GPU_TYPE_ID" "false" "$(bootstrap_cmd)")
echo "$resp" >> "$LAUNCH_LOG"
boot_pid=$(printf '%s' "$resp" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('data',{}).get('podFindAndDeployOnDemand',{}).get('id') or '')")
if [[ -z "$boot_pid" ]]; then
  echo "[launch] FAILED to provision bootstrap pod. Response:" >&2
  echo "$resp" >&2
  exit 1
fi
echo "[launch] bootstrap pod: $boot_pid"

echo "[launch] creating babysitter pod (GPU, $BABYSITTER_GPU_TYPE_ID) ..."
resp=$(gql_deploy "fisher-poc-babysit" "$IMAGE_GPU" "$BABYSITTER_GPU_TYPE_ID" "false" "$(babysitter_cmd)")
echo "$resp" >> "$LAUNCH_LOG"
cpu_pid=$(printf '%s' "$resp" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('data',{}).get('podFindAndDeployOnDemand',{}).get('id') or '')")
if [[ -z "$cpu_pid" ]]; then
  echo "[launch] FAILED to provision babysitter pod. Response:" >&2
  echo "$resp" >&2
  exit 1
fi
echo "[launch] babysitter pod: $cpu_pid"

echo
echo "============================================================"
echo "[launch] kickoff complete."
echo
echo "  GPU bootstrap : $boot_pid  (trains SAEs, ~30-60 min, then self-terminates)"
echo "  CPU babysitter: $cpu_pid   (waits for SAEs, then launches $(echo $SEEDS | wc -w | tr -d ' ') experiment pods)"
echo
echo "Watch progress:"
echo "  - HF dataset: https://huggingface.co/datasets/$HF_REPO"
echo "  - Pod list:   runpodctl pod list  (or use the RunPod web UI)"
echo
echo "All pods self-terminate on completion. The babysitter writes"
echo "summary.md to HF when all per-seed results are present."
echo "Full launch responses logged to $LAUNCH_LOG."
echo "============================================================"
