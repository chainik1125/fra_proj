#!/usr/bin/env bash
# Launch ONE 80GB pod for the 14B SAE-quality gate (~30-40 min, ~$2-3).
set -euo pipefail
BRANCH="${BRANCH:-autoresearch/wang-steering-7b}"
REPO_URL="${REPO_URL:-https://github.com/chainik1125/fra_proj.git}"
GPU_TYPE_IDS="${GPU_TYPE_IDS:-NVIDIA H100 80GB HBM3|NVIDIA H100 PCIe|NVIDIA A100-SXM4-80GB|NVIDIA A100 80GB PCIe}"
IMAGE_GPU="${IMAGE_GPU:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"
: "${RUNPOD_API_KEY:?}"; : "${HF_TOKEN:?}"
GRAPHQL="https://api.runpod.io/graphql"
json_str() { python3 -c "import json,sys; print(json.dumps(sys.stdin.read()))"; }
inner=$(cat <<EOF
#!/bin/bash
set -eo pipefail
exec > >(stdbuf -oL tee /workspace/run.log) 2>&1
cd /workspace
[ -d fra_proj ] || git clone --branch '$BRANCH' --single-branch '$REPO_URL' /workspace/fra_proj
cd /workspace/fra_proj && git fetch origin && git checkout '$BRANCH' && git pull --ff-only
HF_TOKEN='$HF_TOKEN' RUNPOD_API_KEY='$RUNPOD_API_KEY' RUNPOD_POD_ID="\$RUNPOD_POD_ID" BRANCH='$BRANCH' \\
bash experiments/fra_14b_financial/measure_sae.sh
EOF
)
b64=$(printf '%s' "$inner" | base64 | tr -d '\n')
docker_args="bash -c \"echo $b64 | base64 -d > /start_user.sh && chmod +x /start_user.sh && /start.sh & sleep 30 && /start_user.sh\""
cmd_json=$(printf '%s' "$docker_args" | json_str)
while IFS= read -r gpu_type; do
    [ -z "$gpu_type" ] && continue
    input=$(cat <<JSON
{ "name": "sae14b-measure", "imageName": "$IMAGE_GPU", "cloudType": "SECURE", "gpuTypeId": "$gpu_type",
  "gpuCount": 1, "minVcpuCount": 8, "minMemoryInGb": 80, "containerDiskInGb": 120, "volumeInGb": 0,
  "dockerArgs": $cmd_json, "ports": "22/tcp", "startSsh": true }
JSON
)
    payload=$(python3 -c "
import json,sys; inp=json.loads(sys.argv[1])
q='mutation Deploy(\$input: PodFindAndDeployOnDemandInput!){ podFindAndDeployOnDemand(input:\$input){id name desiredStatus} }'
print(json.dumps({'query':q,'variables':{'input':inp}}))" "$input")
    resp=$(curl -sS -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" -d "$payload" "$GRAPHQL")
    pid=$(printf '%s' "$resp" | python3 -c "import json,sys; print(json.load(sys.stdin).get('data',{}).get('podFindAndDeployOnDemand',{}).get('id') or '')")
    if [ -n "$pid" ]; then echo "[launch] sae14b-measure on [$gpu_type] pod_id=$pid"; echo "$pid">/tmp/sae14b_measure_pod.txt; exit 0; fi
    echo "[launch] no capacity [$gpu_type]" >&2
done < <(printf '%s' "$GPU_TYPE_IDS" | tr '|' '\n')
echo "[launch] FAILED" >&2; exit 1
