#!/usr/bin/env bash
# Launch ONE GPU pod: base-model diff-cossim control, 25 features × ±90 × n=8,
# 3 seeds. Tests whether the large Δcoh swing is EM-specific or generic
# large-α coherence collapse (random features included as the key control).
# Required env: RUNPOD_API_KEY, HF_TOKEN
set -euo pipefail
BRANCH="${BRANCH:-autoresearch/wang-steering-7b}"
REPO_URL="${REPO_URL:-https://github.com/chainik1125/fra_proj.git}"
GPU_TYPE_IDS="${GPU_TYPE_IDS:-NVIDIA L40S|NVIDIA L40|NVIDIA A40|NVIDIA RTX A6000}"
IMAGE_GPU="${IMAGE_GPU:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY not set}"; : "${HF_TOKEN:?HF_TOKEN not set}"
GRAPHQL="https://api.runpod.io/graphql"
FEATURE_IDS="106931 98853 98282 8523 97280 84806 49307 77753 57578 46094 50806 75837 93505 47141 77115 93221 12377 81444 109443 79103 53258 7777 31337 99999 120000"
json_str() { python3 -c "import json,sys; print(json.dumps(sys.stdin.read()))"; }

make_pod_inner() {
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
BRANCH='$BRANCH' EM_MODEL='base' EVAL_SEEDS='42 123 456' \\
SAMPLES_PER_PROMPT='1' HF_PREFIX='qwen7b/base_diffcossim_control_n8' \\
FEATURE_IDS='$FEATURE_IDS' \\
bash experiments/wang_steering_7b/auto_start_base_control.sh
EOF
}
make_docker_args() { cat <<EOF
bash -c "echo $1 | base64 -d > /start_user.sh && chmod +x /start_user.sh && /start.sh & sleep 30 && /start_user.sh"
EOF
}

inner=$(make_pod_inner); b64=$(printf '%s' "$inner" | base64 | tr -d '\n')
cmd_json=$(printf '%s' "$(make_docker_args "$b64")" | json_str)
last_resp=""
while IFS= read -r gpu_type; do
    [ -z "$gpu_type" ] && continue
    input=$(cat <<JSON
{ "name": "base-diffcossim-control", "imageName": "$IMAGE_GPU", "cloudType": "SECURE",
  "gpuTypeId": "$gpu_type", "gpuCount": 1, "minVcpuCount": 4, "minMemoryInGb": 24,
  "containerDiskInGb": 60, "volumeInGb": 0, "dockerArgs": $cmd_json, "ports": "22/tcp", "startSsh": true }
JSON
)
    payload=$(python3 -c "
import json, sys
inp = json.loads(sys.argv[1])
q = '''mutation Deploy(\$input: PodFindAndDeployOnDemandInput!) { podFindAndDeployOnDemand(input: \$input) { id name desiredStatus } }'''
print(json.dumps({'query': q, 'variables': {'input': inp}}))
" "$input")
    resp=$(curl -sS -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" -d "$payload" "$GRAPHQL")
    last_resp="$resp"
    pid=$(printf '%s' "$resp" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('data',{}).get('podFindAndDeployOnDemand',{}).get('id') or '')")
    if [ -n "$pid" ]; then
        echo "[launch] base-diffcossim-control on [$gpu_type] pod_id=$pid"
        echo "[launch] HF: https://huggingface.co/datasets/dmanningcoe/fra-phase1-steering-data/tree/main/qwen7b/base_diffcossim_control_n8"
        echo "$pid" > /tmp/base_control_pod_id.txt
        exit 0
    fi
    echo "[launch] no capacity on [$gpu_type], next..." >&2
done < <(printf '%s' "$GPU_TYPE_IDS" | tr '|' '\n')
echo "[launch] FAILED all GPU types. Last: $last_resp" >&2; exit 1
