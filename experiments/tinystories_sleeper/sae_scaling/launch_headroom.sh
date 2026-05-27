#!/usr/bin/env bash
# Launch ONE headroom pod that runs headroom_driver.sh for a seed-shard, then self-terminates.
# Env:  SEED (req)  HF_TOKEN (req)  RP_API_KEY_MATS (req)
#       WIDTHS KS HOOKS P2 P3  (passed through to the driver)
#       SLUG (pod-name suffix, default seed<SEED>)
set -euo pipefail
: "${HF_TOKEN:?}"; : "${RP_API_KEY_MATS:?}"
SEEDS="${SEEDS:-${SEED:?need SEED or SEEDS}}"   # space-separated; pod runs each seed in turn
SEED="${SEED:-$(echo $SEEDS | awk '{print $1}')}"
BRANCH="${BRANCH:-dmitry/sae-scaling-sweep}"
REPO_URL="${REPO_URL:-https://github.com/chainik1125/fra_proj.git}"
IMAGE_GPU="${IMAGE_GPU:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"
GPU_TYPE_IDS="${GPU_TYPE_IDS:-NVIDIA RTX A5000|NVIDIA GeForce RTX 4090|NVIDIA L4|NVIDIA A40}"  # >=24GB (A4000 16GB OOMs the OV hybrid)
SLUG="${SLUG:-seed${SEED}}"
NAME="headroom-on-0526-${SLUG}"
WIDTHS="${WIDTHS:-12288 24576}"; KS="${KS:-10 32 50}"; HOOKS="${HOOKS:-ln1 resid_mid}"
P2="${P2:-0}"; P3="${P3:-0}"
DRIVER="${DRIVER:-experiments/tinystories_sleeper/sae_scaling/headroom_driver.sh}"
ONESHOT="${ONESHOT:-0}"   # 1 -> call DRIVER once (it loops SEEDS itself) with SELF_STOP=1
GRAPHQL="https://api.runpod.io/graphql"
json_str() { python3 -c "import json,sys; print(json.dumps(sys.stdin.read()))"; }

inner=$(cat <<EOF
#!/bin/bash
set -o pipefail
cd /workspace
[ -d fra_proj ] || git clone --branch '$BRANCH' --single-branch '$REPO_URL' /workspace/fra_proj
cd /workspace/fra_proj && git fetch origin && git checkout '$BRANCH' && git pull --ff-only
export HF_TOKEN='$HF_TOKEN' RUNPOD_API_KEY='$RP_API_KEY_MATS' RUNPOD_POD_ID="\$RUNPOD_POD_ID"
export WIDTHS='$WIDTHS' KS='$KS' HOOKS='$HOOKS' P2='$P2' P3='$P3' BRANCH='$BRANCH'
SEEDS='$SEEDS'
python -u -c "import torch" >/dev/null 2>&1 || true
if [ '$ONESHOT' = '1' ]; then
  export SELF_STOP=1 SEEDS="\$SEEDS"
  bash '$DRIVER' >> /workspace/driver.log 2>&1
else
  _arr=(\$SEEDS); _n=\${#_arr[@]}; _i=0
  for _s in \$SEEDS; do
    _i=\$((_i+1))
    if [ "\$_i" = "\$_n" ]; then export SELF_STOP=1; else export SELF_STOP=0; fi
    SEED="\$_s" bash '$DRIVER' >> /workspace/driver_s\${_s}.log 2>&1
  done
fi
EOF
)
b64=$(printf '%s' "$inner" | base64 | tr -d '\n')
docker_args="bash -c \"echo $b64 | base64 -d > /start_user.sh && chmod +x /start_user.sh && /start.sh & sleep 30 && /start_user.sh\""
cmd_json=$(printf '%s' "$docker_args" | json_str)

while IFS= read -r gpu_type; do
    [ -z "$gpu_type" ] && continue
    input=$(cat <<JSON
{ "name": "$NAME", "imageName": "$IMAGE_GPU", "cloudType": "SECURE", "gpuTypeId": "$gpu_type",
  "gpuCount": 1, "minVcpuCount": 4, "minMemoryInGb": 20, "containerDiskInGb": 40, "volumeInGb": 0,
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
