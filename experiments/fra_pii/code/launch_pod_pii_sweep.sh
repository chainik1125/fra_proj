#!/usr/bin/env bash
# Stage-2 launcher for fra_pii: installs sae_lens + transformer_lens, pulls the fra_bundle
# (fra/ package) used by fra_win, runs pii_cut.py with PYTHONPATH=/workspace/code.
# Required: RUNPOD_API_KEY, HF_TOKEN, POD_NAME.  Optional: SCRIPT, GPU_TYPE_IDS.
set -euo pipefail
: "${RUNPOD_API_KEY:?}"; : "${HF_TOKEN:?}"; : "${POD_NAME:?}"
SCRIPT="${SCRIPT:-pii_sweep.py}"
METHOD="${METHOD:-diag}"
SWEEP_TAG="${SWEEP_TAG:-$METHOD}"
SAE_LAYERS="${SAE_LAYERS:-6,10,14}"
RUN_LOG="${RUN_LOG:-${POD_NAME}_run.log}"
GPU_TYPE_IDS="${GPU_TYPE_IDS:-NVIDIA A40|NVIDIA L40S|NVIDIA RTX A6000|NVIDIA L40|NVIDIA A100 80GB PCIe}"
IMAGE_GPU="${IMAGE_GPU:-runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04}"
HF_REPO="dmanningcoe/fra-phase1-steering-data"
GRAPHQL="https://api.runpod.io/graphql"
json_str() { python3 -c "import json,sys; print(json.dumps(sys.stdin.read()))"; }

dupe=$(curl -sS -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d '{"query":"query { myself { pods { id name desiredStatus } } }"}' "$GRAPHQL" | \
  python3 -c "import json,sys;d=json.load(sys.stdin);p=(d.get('data') or {}).get('myself',{}).get('pods',[]) or [];print(' '.join(x['id'] for x in p if x['name']=='$POD_NAME' and x.get('desiredStatus')=='RUNNING'))")
if [ -n "$dupe" ]; then echo "ABORT: RUNNING pod named $POD_NAME: $dupe" >&2; exit 1; fi

inner=$(cat <<EOF
#!/bin/bash
set -eo pipefail
exec > >(stdbuf -oL tee /workspace/run.log) 2>&1
export HF_TOKEN='$HF_TOKEN'; export HUGGING_FACE_HUB_TOKEN='$HF_TOKEN'
export OUT_DIR=/workspace/out; export OUTDIR=/workspace/out
export METHOD='$METHOD'; export SWEEP_TAG='$SWEEP_TAG'; export SAE_LAYERS='$SAE_LAYERS'
HFC=hf
upload_log() { \$HFC upload '$HF_REPO' /workspace/run.log fra_pii/results/$RUN_LOG --repo-type dataset >/dev/null 2>&1 || true; }
upload_out() { \$HFC upload '$HF_REPO' /workspace/out fra_pii/results --repo-type dataset >/dev/null 2>&1 || true; }
trap 'echo "[BOOTSTRAP-ERR line \$LINENO]"; upload_log; sleep infinity' ERR
echo "[\$(date -u +%H:%M:%S)] pii-cut bootstrap start ($POD_NAME) SCRIPT=$SCRIPT"
nvidia-smi -L || true
python3 - <<'PYEOF' > /tmp/constraints.txt
import torch
v=torch.__version__.split('+')[0]; print(f"torch=={v}")
try:
    import torchvision; print(f"torchvision=={torchvision.__version__.split('+')[0]}")
except Exception: pass
PYEOF
pip install --no-input -q -c /tmp/constraints.txt "sae_lens==5.10.7" "transformer_lens==2.18.0" "huggingface_hub[cli]" 2>&1 | tail -3
command -v hf >/dev/null 2>&1 || HFC=huggingface-cli
python3 -c "import torch,sae_lens,transformer_lens; print('torch',torch.__version__,'sae_lens',sae_lens.__version__,'cuda',torch.cuda.is_available())"
mkdir -p /workspace/code /workspace/jb /workspace/out
for a in 1 2 3 4 5; do \$HFC download '$HF_REPO' fra_win/fra_bundle.tar.gz --repo-type dataset --local-dir /workspace/jb && break; echo "bundle dl retry \$a"; sleep 20; done
tar -xzf /workspace/jb/fra_win/fra_bundle.tar.gz -C /workspace/code
python3 -c "import sys; sys.path.insert(0,'/workspace/code'); import fra.core.fra, fra.sae_lens_wrapper; print('fra import OK')"
for a in 1 2 3 4 5; do
  rm -rf /workspace/fra_pii/code 2>/dev/null || true
  \$HFC download '$HF_REPO' --repo-type dataset --include "fra_pii/code/*" --local-dir /workspace --force-download >/tmp/dsdl.log 2>&1 || true
  [ -f /workspace/fra_pii/code/$SCRIPT ] && { echo "[bootstrap] code present (attempt \$a)"; break; }
  echo "[bootstrap] code dl attempt \$a incomplete: \$(tail -1 /tmp/dsdl.log)"; sleep 30
done
[ -f /workspace/fra_pii/code/$SCRIPT ] || { echo "[bootstrap] FATAL: code never downloaded"; upload_log; sleep infinity; }
cd /workspace/fra_pii/code
PYTHONPATH=/workspace/code timeout 5400 python3 $SCRIPT || { echo "[run] rc=\$?"; upload_log; upload_out; sleep infinity; }
echo "[\$(date -u +%H:%M:%S)] $SCRIPT done, final upload"
upload_log; upload_out
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
  "gpuCount": 1, "minVcpuCount": 4, "minMemoryInGb": 24, "containerDiskInGb": ${DISK:-50},
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
