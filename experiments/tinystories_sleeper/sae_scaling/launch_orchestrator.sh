#!/usr/bin/env bash
# Launch ONE remote-orchestrator pod that runs headless Claude on CAMPAIGN_headroom.md,
# drives the headroom experiments via GPU worker pods, then self-terminates.
# Env (req): ANTHROPIC_API_KEY_MATS  HF_TOKEN  RP_API_KEY_MATS
# Env (opt): BRANCH (default dmitry/sae-scaling-sweep)  MAX_RUN_SEC (default 28800)
set -euo pipefail
: "${ANTHROPIC_API_KEY_MATS:?}"; : "${HF_TOKEN:?}"; : "${RP_API_KEY_MATS:?}"
BRANCH="${BRANCH:-dmitry/sae-scaling-sweep}"
MAX_RUN_SEC="${MAX_RUN_SEC:-28800}"
REPO_URL="https://github.com/chainik1125/fra_proj.git"
IMAGE="${IMAGE:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"
GPU_TYPE_IDS="${GPU_TYPE_IDS:-NVIDIA RTX A4000|NVIDIA RTX A5000|NVIDIA GeForce RTX 4090|NVIDIA L4|NVIDIA A40}"
NAME="headroom-on-0527-orch"
GRAPHQL="https://api.runpod.io/graphql"
json_str() { python3 -c "import json,sys; print(json.dumps(sys.stdin.read()))"; }

inner=$(cat <<EOF
#!/bin/bash
cd /workspace
command -v git >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq git; }
[ -d fra_proj ] || git clone --branch '$BRANCH' --single-branch '$REPO_URL' fra_proj
cd fra_proj && git fetch origin && git checkout '$BRANCH' && git pull --ff-only || true
export ANTHROPIC_API_KEY='$ANTHROPIC_API_KEY_MATS' HF_TOKEN='$HF_TOKEN' RP_API_KEY_MATS='$RP_API_KEY_MATS'
export BRANCH='$BRANCH' MAX_RUN_SEC='$MAX_RUN_SEC' RUNPOD_POD_ID="\$RUNPOD_POD_ID"
bash experiments/tinystories_sleeper/sae_scaling/auto_start_orchestrator.sh
EOF
)
b64=$(printf '%s' "$inner" | base64 | tr -d '\n')
docker_args="bash -c \"echo $b64 | base64 -d > /start_user.sh && chmod +x /start_user.sh && /start.sh & sleep 30 && /start_user.sh\""
cmd_json=$(printf '%s' "$docker_args" | json_str)

while IFS= read -r gpu_type; do
    [ -z "$gpu_type" ] && continue
    input=$(cat <<JSON
{ "name": "$NAME", "imageName": "$IMAGE", "cloudType": "SECURE", "gpuTypeId": "$gpu_type",
  "gpuCount": 1, "minVcpuCount": 4, "minMemoryInGb": 16, "containerDiskInGb": 40, "volumeInGb": 0,
  "dockerArgs": $cmd_json, "ports": "22/tcp", "startSsh": true }
JSON
)
    payload=$(python3 -c "
import json,sys; inp=json.loads(sys.argv[1])
q='''mutation Deploy(\$input: PodFindAndDeployOnDemandInput!){ podFindAndDeployOnDemand(input:\$input){id name desiredStatus} }'''
print(json.dumps({'query':q,'variables':{'input':inp}}))" "$input")
    resp=$(curl -sS -X POST -H "Authorization: Bearer $RP_API_KEY_MATS" -H "Content-Type: application/json" -A "curl/8.0" -d "$payload" "$GRAPHQL")
    pid=$(printf '%s' "$resp" | python3 -c "import json,sys
try:
  d=json.load(sys.stdin); v=(d.get('data') or {}).get('podFindAndDeployOnDemand') or {}
  print(v.get('id') or '')
except Exception: print('')")
    if [ -n "$pid" ]; then echo "[launch] $NAME on [$gpu_type] pod_id=$pid"; echo "$pid"; exit 0; fi
    echo "[launch] no capacity [$gpu_type]: $(printf '%s' "$resp" | head -c 200)" >&2
done < <(printf '%s' "$GPU_TYPE_IDS" | tr '|' '\n')
echo "[launch] FAILED — no capacity on any GPU type" >&2; exit 1
