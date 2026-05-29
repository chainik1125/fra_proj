#!/usr/bin/env bash
# Launch ONE pod for a single-feature FINE EXTENDED-α sweep (generalized from
# launch_f603_finegrid.sh — works for any feature × SAE). Reuses run_grid.sh's
# FEATURE_IDS_OVERRIDE path + magmatched + uploads; forwards the extended ALPHAS
# + a distinct GRID_HF_BASE that launch_grid.sh doesn't.
#
# α=-4..4 step 0.25 (33 pts), gran1, finance+base, seeds 42/123, HEAD=12, magmatched.
# fra-diff- pod + dup guard.
#
# Required env: RUNPOD_API_KEY, HF_TOKEN, FEATURE_ID, SAE (ln1|resid_post), CELLTAG
#   CELLTAG = pod-name + HF-prefix tag, e.g. "f603" → pod fra-diff-f603-finegrid,
#             HF qwen14b/grid_diff/<CELLTAG>_finegrid/.
#   resid_post needs RESID_POST_DELTA_A (default 45.43).
set -euo pipefail
BRANCH="${BRANCH:-autoresearch/wang-steering-7b}"
REPO_URL="${REPO_URL:-https://github.com/chainik1125/fra_proj.git}"
GPU_TYPE_IDS="${GPU_TYPE_IDS:-NVIDIA H100 80GB HBM3|NVIDIA H100 PCIe|NVIDIA A100-SXM4-80GB|NVIDIA A100 80GB PCIe}"
IMAGE_GPU="${IMAGE_GPU:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"
: "${RUNPOD_API_KEY:?}"; : "${HF_TOKEN:?}"; : "${FEATURE_ID:?}"; : "${SAE:?}"; : "${CELLTAG:?}"
HEAD="${HEAD:-12}"
POD_SUFFIX="${POD_SUFFIX:-}"
POD_PREFIX="${POD_PREFIX:-fra-diff}"
RESID_POST_DELTA_A="${RESID_POST_DELTA_A:-45.43}"
ALPHAS="${ALPHAS:-$(python3 -c "print(' '.join(str(round(-4+0.25*i,2)) for i in range(33)))")}"
GRID_HF_BASE="${GRID_HF_BASE:-qwen14b/grid_diff/${CELLTAG}_finegrid}"
GRAPHQL="https://api.runpod.io/graphql"
json_str() { python3 -c "import json,sys; print(json.dumps(sys.stdin.read()))"; }
POD_NAME="${POD_PREFIX}-${CELLTAG}-finegrid${POD_SUFFIX}"

existing=$(curl -sS -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
    -d '{"query":"query { myself { pods { name desiredStatus } } }"}' "$GRAPHQL" \
    | python3 -c "
import json,sys
try: d=json.load(sys.stdin)
except Exception: sys.exit()
for p in (((d.get('data') or {}).get('myself') or {}).get('pods') or []):
    if p.get('name')=='$POD_NAME' and p.get('desiredStatus')=='RUNNING': print('HIT')
" 2>/dev/null || true)
if [ "$existing" = "HIT" ]; then echo "[launch] $POD_NAME already RUNNING — skip (dup guard)"; exit 0; fi

inner=$(cat <<EOF
#!/bin/bash
set -eo pipefail
exec > >(stdbuf -oL tee /workspace/run.log) 2>&1
echo "[\$(date -u +%H:%M:%S)] inner bootstrap start ($POD_NAME)"
cd /workspace
[ -d fra_proj ] || git clone --branch '$BRANCH' --single-branch '$REPO_URL' /workspace/fra_proj
cd /workspace/fra_proj
git fetch origin && git checkout '$BRANCH' && git pull --ff-only
HF_TOKEN='$HF_TOKEN' RUNPOD_API_KEY='$RUNPOD_API_KEY' RUNPOD_POD_ID="\$RUNPOD_POD_ID" \\
BRANCH='$BRANCH' \\
RANKING='fra-ov' SAE='$SAE' EM_MODELS='base finance' SEEDS='42 123' GRANS='1' HEAD='$HEAD' \\
STEER_MODE='magmatched' FEATURE_IDS_OVERRIDE='$FEATURE_ID' RESID_POST_DELTA_A='$RESID_POST_DELTA_A' \\
ALPHAS='$ALPHAS' GRID_HF_BASE='$GRID_HF_BASE' \\
bash experiments/fra_14b_financial/run_grid.sh
EOF
)
b64=$(printf '%s' "$inner" | base64 | tr -d '\n')
docker_args="bash -c \"echo $b64 | base64 -d > /start_user.sh && chmod +x /start_user.sh && /start.sh & sleep 30 && /start_user.sh\""
cmd_json=$(printf '%s' "$docker_args" | json_str)

last_resp=""
while IFS= read -r gpu_type; do
    [ -z "$gpu_type" ] && continue
    input=$(cat <<JSON
{ "name": "$POD_NAME", "imageName": "$IMAGE_GPU", "cloudType": "SECURE",
  "gpuTypeId": "$gpu_type", "gpuCount": 1, "minVcpuCount": 4, "minMemoryInGb": 100,
  "containerDiskInGb": 80, "volumeInGb": 0, "dockerArgs": $cmd_json, "ports": "22/tcp", "startSsh": true }
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
    pid=$(printf '%s' "$resp" | python3 -c "
import json,sys
try: d = json.load(sys.stdin)
except Exception: print(''); sys.exit()
data = d.get('data') or {}
node = (data.get('podFindAndDeployOnDemand') or {}) if isinstance(data, dict) else {}
print(node.get('id') or '')
")
    if [ -n "$pid" ]; then echo "[launch] $POD_NAME on [$gpu_type] pod_id=$pid"; echo "$pid" > "/tmp/${POD_NAME}_pod_id.txt"; exit 0; fi
    echo "[launch] no pod on [$gpu_type]: $(printf '%s' "$resp" | head -c 200)" >&2
done < <(printf '%s' "$GPU_TYPE_IDS" | tr '|' '\n')
echo "[launch] FAILED all GPU types. Last: $last_resp" >&2; exit 1
