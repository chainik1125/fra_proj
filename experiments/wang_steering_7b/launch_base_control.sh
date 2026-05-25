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
    local seed="$1"
    cat <<EOF
#!/bin/bash
set -eo pipefail
exec > >(stdbuf -oL tee /workspace/run.log) 2>&1
echo "[\$(date -u +%H:%M:%S)] inner bootstrap start (seed $seed)"
cd /workspace
[ -d fra_proj ] || git clone --branch '$BRANCH' --single-branch '$REPO_URL' /workspace/fra_proj
cd /workspace/fra_proj
git fetch origin && git checkout '$BRANCH' && git pull --ff-only
HF_TOKEN='$HF_TOKEN' RUNPOD_API_KEY='$RUNPOD_API_KEY' RUNPOD_POD_ID="\$RUNPOD_POD_ID" \\
BRANCH='$BRANCH' EM_MODEL='base' EVAL_SEEDS='$seed' \\
SAMPLES_PER_PROMPT='1' HF_PREFIX='qwen7b/base_diffcossim_control_n8' \\
FEATURE_IDS='$FEATURE_IDS' \\
bash experiments/wang_steering_7b/auto_start_base_control.sh
EOF
}
make_docker_args() { cat <<EOF
bash -c "echo $1 | base64 -d > /start_user.sh && chmod +x /start_user.sh && /start.sh & sleep 30 && /start_user.sh"
EOF
}

deploy_one() {
    local seed="$1"
    local inner b64 cmd_json
    inner=$(make_pod_inner "$seed"); b64=$(printf '%s' "$inner" | base64 | tr -d '\n')
    cmd_json=$(printf '%s' "$(make_docker_args "$b64")" | json_str)
    local last_resp=""
    while IFS= read -r gpu_type; do
        [ -z "$gpu_type" ] && continue
        local input
        input=$(cat <<JSON
{ "name": "base-ctl-s$seed", "imageName": "$IMAGE_GPU", "cloudType": "SECURE",
  "gpuTypeId": "$gpu_type", "gpuCount": 1, "minVcpuCount": 4, "minMemoryInGb": 24,
  "containerDiskInGb": 60, "volumeInGb": 0, "dockerArgs": $cmd_json, "ports": "22/tcp", "startSsh": true }
JSON
)
        local payload
        payload=$(python3 -c "
import json, sys
inp = json.loads(sys.argv[1])
q = '''mutation Deploy(\$input: PodFindAndDeployOnDemandInput!) { podFindAndDeployOnDemand(input: \$input) { id name desiredStatus } }'''
print(json.dumps({'query': q, 'variables': {'input': inp}}))
" "$input")
        local resp pid
        resp=$(curl -sS -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" -d "$payload" "$GRAPHQL")
        last_resp="$resp"
        pid=$(printf '%s' "$resp" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('data',{}).get('podFindAndDeployOnDemand',{}).get('id') or '')")
        if [ -n "$pid" ]; then
            echo "[launch] base-ctl-s$seed on [$gpu_type] pod_id=$pid"
            echo "$pid"; return 0
        fi
        echo "[launch] seed $seed: no capacity on [$gpu_type], next..." >&2
    done < <(printf '%s' "$GPU_TYPE_IDS" | tr '|' '\n')
    echo "[launch] seed $seed FAILED all GPU types. Last: $last_resp" >&2; return 1
}

: > /tmp/base_control_pod_ids.txt
for seed in 42 123 456; do
    pid=$(deploy_one "$seed") || { echo "ABORT seed $seed"; exit 1; }
    echo "$seed $pid" >> /tmp/base_control_pod_ids.txt
done
echo "[launch] all 3 base-control pods up (one per seed):"
cat /tmp/base_control_pod_ids.txt
echo "[launch] HF: https://huggingface.co/datasets/dmanningcoe/fra-phase1-steering-data/tree/main/qwen7b/base_diffcossim_control_n8"
