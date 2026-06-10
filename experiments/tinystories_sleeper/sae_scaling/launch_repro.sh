#!/usr/bin/env bash
# Launch ONE pod that runs repro_driver.sh (single-feature SAE repro + DoM
# baselines) and self-terminates. Env: HF_TOKEN (req)  RP_API_KEY_MATS (req)
set -euo pipefail
: "${HF_TOKEN:?}"; : "${RP_API_KEY_MATS:?}"
BRANCH="${BRANCH:-dmitry/sae-scaling-sweep}"
REPO_URL="${REPO_URL:-https://github.com/chainik1125/fra_proj.git}"
IMAGE_GPU="${IMAGE_GPU:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"
GPU_TYPE_IDS="${GPU_TYPE_IDS:-NVIDIA RTX A5000|NVIDIA GeForce RTX 4090|NVIDIA L4|NVIDIA A40}"
NAME="${NAME:-sae-repro-singlefeat-dom-0609}"
DRIVER="experiments/tinystories_sleeper/sae_scaling/repro_driver.sh"
GRAPHQL="https://api.runpod.io/graphql"
json_str() { python3 -c "import json,sys; print(json.dumps(sys.stdin.read()))"; }

# pre-check: refuse a duplicate RUNNING pod of the same name (the API won't)
dup=$(curl -sS -X POST -H "Authorization: Bearer $RP_API_KEY_MATS" -H "Content-Type: application/json" \
  -A "curl/8.0" --data-binary '{"query":"query { myself { pods { id name desiredStatus } } }"}' "$GRAPHQL" \
  | python3 -c "import json,sys
d=json.load(sys.stdin)
pods=(d.get('data') or {}).get('myself',{}).get('pods') or []
print(sum(1 for p in pods if p['name']=='$NAME' and p['desiredStatus']=='RUNNING'))")
[ "$dup" != "0" ] && { echo "[launch] pod named $NAME already RUNNING — abort"; exit 1; }

inner=$(cat <<EOF
#!/bin/bash
set -o pipefail
cd /workspace
[ -d fra_proj ] || { apt-get update -qq && apt-get install -y -q git >/dev/null 2>&1; git clone --branch '$BRANCH' --single-branch '$REPO_URL' /workspace/fra_proj; }
export HF_TOKEN='$HF_TOKEN' RUNPOD_API_KEY='$RP_API_KEY_MATS' RUNPOD_POD_ID="\$RUNPOD_POD_ID"
export BRANCH='$BRANCH' SELF_STOP=1
bash /workspace/fra_proj/$DRIVER >> /workspace/driver.log 2>&1
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
    if [ -n "$pid" ]; then echo "[launch] $NAME on [$gpu_type] pod_id=$pid"; exit 0; fi
    echo "[launch] no capacity [$gpu_type]: $(printf '%s' "$resp" | head -c 200)" >&2
done < <(printf '%s' "$GPU_TYPE_IDS" | tr '|' '\n')
echo "[launch] FAILED on all GPU types" >&2
exit 1
