#!/usr/bin/env bash
# Launch ONE 80GB pod PER SAE (resid_post + ln1) to train them in PARALLEL on
# base Qwen-2.5-14B-Instruct at L24 (Arditi BatchTopK k64, d131072, 200M tok).
# Required env: RUNPOD_API_KEY, HF_TOKEN
set -euo pipefail
BRANCH="${BRANCH:-autoresearch/wang-steering-7b}"
REPO_URL="${REPO_URL:-https://github.com/chainik1125/fra_proj.git}"
# 14B (28GB bf16) + activation buffer + 131072-wide SAE → need 80GB.
GPU_TYPE_IDS="${GPU_TYPE_IDS:-NVIDIA H100 80GB HBM3|NVIDIA H100 PCIe|NVIDIA A100-SXM4-80GB|NVIDIA A100 80GB PCIe}"
IMAGE_GPU="${IMAGE_GPU:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"
NUM_TOKENS="${NUM_TOKENS:-200000000}"
: "${RUNPOD_API_KEY:?}"; : "${HF_TOKEN:?}"
GRAPHQL="https://api.runpod.io/graphql"
json_str() { python3 -c "import json,sys; print(json.dumps(sys.stdin.read()))"; }

launch_one() {
    local kind="$1"
    local inner
    inner=$(cat <<EOF
#!/bin/bash
set -eo pipefail
exec > >(stdbuf -oL tee /workspace/run.log) 2>&1
cd /workspace
[ -d fra_proj ] || git clone --branch '$BRANCH' --single-branch '$REPO_URL' /workspace/fra_proj
cd /workspace/fra_proj && git fetch origin && git checkout '$BRANCH' && git pull --ff-only
HF_TOKEN='$HF_TOKEN' RUNPOD_API_KEY='$RUNPOD_API_KEY' RUNPOD_POD_ID="\$RUNPOD_POD_ID" \\
BRANCH='$BRANCH' SAE_KIND='$kind' HOOK_LAYER='24' NUM_TOKENS='$NUM_TOKENS' \\
bash experiments/fra_14b_financial/train_sae.sh
EOF
)
    local b64 docker_args cmd_json
    b64=$(printf '%s' "$inner" | base64 | tr -d '\n')
    docker_args="bash -c \"echo $b64 | base64 -d > /start_user.sh && chmod +x /start_user.sh && /start.sh & sleep 30 && /start_user.sh\""
    cmd_json=$(printf '%s' "$docker_args" | json_str)
    while IFS= read -r gpu_type; do
        [ -z "$gpu_type" ] && continue
        local input payload resp pid
        input=$(cat <<JSON
{ "name": "sae14b-$kind", "imageName": "$IMAGE_GPU", "cloudType": "SECURE",
  "gpuTypeId": "$gpu_type", "gpuCount": 1, "minVcpuCount": 8, "minMemoryInGb": 120,
  "containerDiskInGb": 200, "volumeInGb": 0, "dockerArgs": $cmd_json, "ports": "22/tcp", "startSsh": true }
JSON
)
        payload=$(python3 -c "
import json,sys; inp=json.loads(sys.argv[1])
q='mutation Deploy(\$input: PodFindAndDeployOnDemandInput!){ podFindAndDeployOnDemand(input:\$input){ id name desiredStatus } }'
print(json.dumps({'query':q,'variables':{'input':inp}}))" "$input")
        resp=$(curl -sS -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" -d "$payload" "$GRAPHQL")
        pid=$(printf '%s' "$resp" | python3 -c "import json,sys; print(json.load(sys.stdin).get('data',{}).get('podFindAndDeployOnDemand',{}).get('id') or '')")
        if [ -n "$pid" ]; then
            echo "[launch] sae14b-$kind on [$gpu_type] pod_id=$pid"
            echo "$pid" > "/tmp/sae14b_${kind}_pod.txt"; return 0
        fi
        echo "[launch] no capacity [$gpu_type] for $kind, next..." >&2
    done < <(printf '%s' "$GPU_TYPE_IDS" | tr '|' '\n')
    echo "[launch] FAILED all GPU types for $kind. Last: $resp" >&2; return 1
}

for kind in resid_post ln1; do launch_one "$kind"; done
echo "[launch] both SAE pods dispatched. HF: qwen14b/sae_{resid_post,ln1}_l24_base_arditi"
