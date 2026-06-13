#!/usr/bin/env bash
# Launcher for fra_ws_backdoor pods (Phase F feasibility / Phase B backdoor).
# Adapted from fra_weightsparse/code/launch_pod_ws.sh. Code/results via HF dataset
# repo, prefix fra_ws_backdoor/{code,results}. Parametrized by SCRIPT.
# Required env: RUNPOD_API_KEY, HF_TOKEN, POD_NAME   Optional: SCRIPT, STAGES, GPU_TYPE_IDS
set -euo pipefail
: "${RUNPOD_API_KEY:?}"; : "${HF_TOKEN:?}"; : "${POD_NAME:?}"
SCRIPT="${SCRIPT:-feas_pod.py}"
STAGES="${STAGES:-}"
RUN_LOG="${RUN_LOG:-${POD_NAME}_run.log}"
GPU_TYPE_IDS="${GPU_TYPE_IDS:-NVIDIA A40|NVIDIA L4|NVIDIA RTX A6000|NVIDIA L40|NVIDIA L40S}"
IMAGE_GPU="${IMAGE_GPU:-runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04}"
HF_REPO="dmanningcoe/fra-phase1-steering-data"
GRAPHQL="https://api.runpod.io/graphql"
json_str() { python3 -c "import json,sys; print(json.dumps(sys.stdin.read()))"; }

# --- duplicate-name pre-check (podFindAndDeployOnDemand does NOT refuse dupes) ---
dupe=$(curl -sS -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d '{"query":"query { myself { pods { id name desiredStatus } } }"}' "$GRAPHQL" | \
  python3 -c "
import json,sys
d=json.load(sys.stdin)
pods=(d.get('data') or {}).get('myself',{}).get('pods',[]) or []
print(' '.join(p['id'] for p in pods if p['name']=='$POD_NAME' and p.get('desiredStatus')=='RUNNING'))
")
if [ -n "$dupe" ]; then echo "ABORT: RUNNING pod already named $POD_NAME: $dupe" >&2; exit 1; fi

inner=$(cat <<EOF
#!/bin/bash
set -eo pipefail
exec > >(stdbuf -oL tee /workspace/run.log) 2>&1
export HF_TOKEN='$HF_TOKEN'
export STAGES='$STAGES'
export OUT_DIR=/workspace/out
HFC=hf
upload_log() { \$HFC upload '$HF_REPO' /workspace/run.log fra_ws_backdoor/results/$RUN_LOG --repo-type dataset >/dev/null 2>&1 || true; }
trap 'echo "[BOOTSTRAP-ERR line \$LINENO]"; upload_log; sleep infinity' ERR
echo "[\$(date -u +%H:%M:%S)] wsb bootstrap start ($POD_NAME) SCRIPT=$SCRIPT"
nvidia-smi -L || true
python3 - <<'PYEOF' > /tmp/constraints.txt
import torch
v = torch.__version__.split('+')[0]
print(f"torch=={v}")
PYEOF
pip install --no-input -q -c /tmp/constraints.txt tokenizers datasets requests "huggingface_hub[cli]" 2>&1 | tail -2
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
command -v hf >/dev/null 2>&1 || HFC=huggingface-cli
echo "[bootstrap] HF CLI: \$HFC"
CODE_REV='$CODE_REV'
CODE_SENTINEL='$CODE_SENTINEL'
REV_ARG=""; [ -n "\$CODE_REV" ] && REV_ARG="--revision \$CODE_REV"
echo "[bootstrap] code revision pin: \${CODE_REV:-<none>} sentinel: \${CODE_SENTINEL:-<none>}"
for dlat in 1 2 3 4 5 6; do
  # immutable commit-SHA revision is content-addressed => no stale-CDN; --force-download as backup
  rm -rf /workspace/fra_ws_backdoor/code 2>/dev/null || true
  \$HFC download '$HF_REPO' --repo-type dataset --include "fra_ws_backdoor/code/*" --local-dir /workspace \$REV_ARG --force-download >/tmp/dsdl.log 2>&1 || true
  if [ -f /workspace/fra_ws_backdoor/code/$SCRIPT ]; then
    if [ -z "\$CODE_SENTINEL" ] || grep -q "\$CODE_SENTINEL" /workspace/fra_ws_backdoor/code/$SCRIPT; then
      echo "[bootstrap] code present + sentinel OK (attempt \$dlat)"; break
    fi
    echo "[bootstrap] code present but SENTINEL '\$CODE_SENTINEL' missing (stale CDN); attempt \$dlat sleep 45"
  else
    echo "[bootstrap] code dl attempt \$dlat incomplete: \$(tail -1 /tmp/dsdl.log)"
  fi
  sleep 45
done
[ -f /workspace/fra_ws_backdoor/code/$SCRIPT ] || { echo "[bootstrap] FATAL: code never downloaded"; upload_log; sleep infinity; }
mkdir -p /workspace/out
cd /workspace/fra_ws_backdoor/code
for attempt in 1 2 3; do
  if python3 $SCRIPT; then break; fi
  echo "[bootstrap] $SCRIPT failed (attempt \$attempt), retrying in 60s"; upload_log; sleep 60
  [ "\$attempt" = 3 ] && { upload_log; exit 1; }
done
echo "[\$(date -u +%H:%M:%S)] $SCRIPT done, final upload"
upload_log
runpodctl stop pod \$RUNPOD_POD_ID || true
sleep infinity
EOF
)
b64=$(printf '%s' "$inner" | base64 | tr -d '\n')
docker_args="bash -c \"echo $b64 | base64 -d > /start_user.sh && chmod +x /start_user.sh && /start.sh & sleep 30 && /start_user.sh\""
cmd_json=$(printf '%s' "$docker_args" | json_str)

while IFS= read -r gpu_type; do
    [ -z "$gpu_type" ] && continue
    input=$(cat <<JSON
{ "name": "$POD_NAME", "imageName": "$IMAGE_GPU", "cloudType": "SECURE",
  "gpuTypeId": "$gpu_type", "gpuCount": 1, "minVcpuCount": 4, "minMemoryInGb": ${MIN_MEM:-24},
  "containerDiskInGb": ${DISK:-40}, "volumeInGb": 0, "dockerArgs": $cmd_json, "ports": "22/tcp", "startSsh": true }
JSON
)
    payload=$(python3 -c "
import json, sys
inp = json.loads(sys.argv[1])
q = '''mutation Deploy(\$input: PodFindAndDeployOnDemandInput!) { podFindAndDeployOnDemand(input: \$input) { id name desiredStatus } }'''
print(json.dumps({'query': q, 'variables': {'input': inp}}))
" "$input")
    resp=$(curl -sS -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" -d "$payload" "$GRAPHQL")
    pid=$(printf '%s' "$resp" | python3 -c "
import json,sys
try: d = json.load(sys.stdin)
except Exception: print(''); sys.exit()
node = ((d.get('data') or {}).get('podFindAndDeployOnDemand') or {})
print(node.get('id') or '')
")
    if [ -n "$pid" ]; then
        echo "[launch] $POD_NAME on [$gpu_type] pod_id=$pid"
        echo "$pid" > "/tmp/${POD_NAME}_pod_id.txt"; exit 0
    fi
    echo "[launch] no pod on [$gpu_type]: $(printf '%s' "$resp" | head -c 200)" >&2
done < <(printf '%s' "$GPU_TYPE_IDS" | tr '|' '\n')
echo "[launch] FAILED on all GPU types" >&2; exit 1
