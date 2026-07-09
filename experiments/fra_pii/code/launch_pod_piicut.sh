#!/usr/bin/env bash
# Stage-2 launcher for fra_pii cut jobs (imports the fra bundle + transformer_lens + GemmaScope).
# Installs sae_lens/transformer_lens, downloads fra_win/fra_bundle.tar.gz to /workspace/code,
# downloads fra_pii/code/$SCRIPT, runs with PYTHONPATH=/workspace/code. One-shot; auto-stops.
# Required env: RUNPOD_API_KEY, HF_TOKEN, POD_NAME   Optional: SCRIPT
set -euo pipefail
: "${RUNPOD_API_KEY:?}"; : "${HF_TOKEN:?}"; : "${POD_NAME:?}"
SCRIPT="${SCRIPT:-pii_cut.py}"
RUN_LOG="${RUN_LOG:-${POD_NAME}_run.log}"
GPU_TYPE_IDS="${GPU_TYPE_IDS:-NVIDIA A40|NVIDIA RTX A6000|NVIDIA L40S|NVIDIA L40|NVIDIA A100 80GB PCIe}"
IMAGE_GPU="${IMAGE_GPU:-runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04}"
HF_REPO="dmanningcoe/fra-phase1-steering-data"
GRAPHQL="https://api.runpod.io/graphql"
json_str() { python3 -c "import json,sys; print(json.dumps(sys.stdin.read()))"; }

dupe=$(curl -sS -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d '{"query":"query { myself { pods { id name desiredStatus } } }"}' "$GRAPHQL" | \
  python3 -c "import json,sys; d=json.load(sys.stdin); p=(d.get('data') or {}).get('myself',{}).get('pods',[]) or []; print(' '.join(x['id'] for x in p if x['name']=='$POD_NAME' and x.get('desiredStatus')=='RUNNING'))")
if [ -n "$dupe" ]; then echo "ABORT: RUNNING pod named $POD_NAME: $dupe" >&2; exit 1; fi

inner=$(cat <<EOF
#!/bin/bash
set -eo pipefail
exec > >(stdbuf -oL tee /workspace/run.log) 2>&1
export HF_TOKEN='$HF_TOKEN'; export HUGGING_FACE_HUB_TOKEN='$HF_TOKEN'
export OUT_DIR=/workspace/out; export OUTDIR=/workspace/out
HFC=hf
upload_all() { \$HFC upload '$HF_REPO' /workspace/run.log fra_pii/results/$RUN_LOG --repo-type dataset >/dev/null 2>&1 || true; \$HFC upload '$HF_REPO' /workspace/out fra_pii/results --repo-type dataset >/dev/null 2>&1 || true; }
trap 'echo "[BOOTSTRAP-ERR line \$LINENO]"; upload_all; sleep infinity' ERR
echo "[\$(date -u +%H:%M:%S)] pii-cut bootstrap ($POD_NAME) SCRIPT=$SCRIPT"
nvidia-smi -L || true
python3 - <<'PYEOF' > /tmp/constraints.txt
import torch; v=torch.__version__.split('+')[0]; print(f"torch=={v}")
PYEOF
pip install --no-input -q -c /tmp/constraints.txt "sae_lens==5.10.7" "transformer_lens==2.18.0" "huggingface_hub[cli]" 2>&1 | tail -3
command -v hf >/dev/null 2>&1 || HFC=huggingface-cli
python3 -c "import torch,sae_lens,transformer_lens; print('torch',torch.__version__,'sae_lens',sae_lens.__version__,'cuda',torch.cuda.is_available())"
mkdir -p /workspace/code /workspace/out /workspace/fra_pii/code
for a in 1 2 3 4 5; do \$HFC download '$HF_REPO' fra_win/fra_bundle.tar.gz --repo-type dataset --local-dir /workspace/jb && break; echo "bundle retry \$a"; sleep 20; done
tar -xzf /workspace/jb/fra_win/fra_bundle.tar.gz -C /workspace/code
python3 -c "import sys; sys.path.insert(0,'/workspace/code'); import fra.core.fra; print('fra import OK')"
for a in 1 2 3 4 5; do rm -f /workspace/fra_pii/code/$SCRIPT; \$HFC download '$HF_REPO' --repo-type dataset --include "fra_pii/code/$SCRIPT" --local-dir /workspace --force-download >/tmp/dl.log 2>&1 || true; [ -f /workspace/fra_pii/code/$SCRIPT ] && break; echo "script retry \$a"; sleep 15; done
[ -f /workspace/fra_pii/code/$SCRIPT ] || { echo "FATAL: $SCRIPT not downloaded"; upload_all; sleep infinity; }
echo "[\$(date -u +%H:%M:%S)] running $SCRIPT"
cd /workspace/fra_pii/code
PYTHONPATH=/workspace/code OUTDIR=/workspace/out python3 $SCRIPT || { echo "SCRIPT rc=\$?"; upload_all; sleep infinity; }
echo "[\$(date -u +%H:%M:%S)] $SCRIPT done"
upload_all
runpodctl stop pod \$RUNPOD_POD_ID || true
sleep infinity
EOF
)
b64=$(printf '%s' "$inner" | base64 | tr -d '\n')
docker_args="bash -c \"echo $b64 | base64 -d > /start_user.sh && chmod +x /start_user.sh && /start.sh & sleep 30 && /start_user.sh\""
cmd_json=$(printf '%s' "$docker_args" | json_str)

while IFS= read -r gpu; do
  [ -z "$gpu" ] && continue
  input=$(cat <<JSON
{ "name": "$POD_NAME", "imageName": "$IMAGE_GPU", "cloudType": "SECURE", "gpuTypeId": "$gpu",
  "gpuCount": 1, "minVcpuCount": 4, "minMemoryInGb": ${MIN_MEM:-30}, "containerDiskInGb": ${DISK:-60},
  "volumeInGb": 0, "dockerArgs": $cmd_json, "ports": "22/tcp", "startSsh": true }
JSON
)
  payload=$(python3 -c "
import json,sys
inp=json.loads(sys.argv[1])
q='mutation Deploy(\$input: PodFindAndDeployOnDemandInput!) { podFindAndDeployOnDemand(input: \$input) { id name desiredStatus } }'
print(json.dumps({'query':q,'variables':{'input':inp}}))" "$input")
  resp=$(curl -sS -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" -d "$payload" "$GRAPHQL")
  pid=$(printf '%s' "$resp" | python3 -c "import json,sys
try: d=json.load(sys.stdin)
except: print(''); sys.exit()
print(((d.get('data') or {}).get('podFindAndDeployOnDemand') or {}).get('id') or '')")
  if [ -n "$pid" ]; then echo "[launch] $POD_NAME on [$gpu] id=$pid"; echo "$pid" > /tmp/${POD_NAME}_pod_id.txt; exit 0; fi
  echo "[launch] no capacity [$gpu]: $(printf '%s' "$resp" | head -c 160)" >&2
done < <(printf '%s' "$GPU_TYPE_IDS" | tr '|' '\n')
echo "[launch] FAILED all GPU types" >&2; exit 1
