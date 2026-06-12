#!/usr/bin/env bash
# Launch ONE RunPod GPU pod that runs the FRA-persistence toy (gpt2-small, tiny).
# Self-contained + self-stopping: bootstraps the fra_bundle, downloads ${PY_SCRIPT:-persist_run.py}
# from HF (fra_persist/code/), runs it (the script itself uploads partials + results +
# traceback to fra_persist/results/), then stops the pod.
# Required env: RP_API_KEY_MATS, HF_TOKEN.  Optional: POD_NAME (default rs-persist-1).
set -euo pipefail
: "${RP_API_KEY_MATS:?}"; : "${HF_TOKEN:?}"
POD_NAME="${POD_NAME:-rs-persist-1}"
HF_REPO="dmanningcoe/fra-phase1-steering-data"
GRAPHQL="https://api.runpod.io/graphql"
IMAGE_GPU="${IMAGE_GPU:-runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04}"
# gpt2-small is tiny -> cheapest GPUs first.
GPU_TYPE_IDS="${GPU_TYPE_IDS:-NVIDIA L4|NVIDIA RTX A4000|NVIDIA RTX A5000|NVIDIA A40|NVIDIA L40S}"
json_str() { python3 -c "import json,sys; print(json.dumps(sys.stdin.read()))"; }

# --- duplicate-name pre-check (podFindAndDeployOnDemand does NOT refuse dupes) ---
dupe=$(curl -sS -X POST -H "Authorization: Bearer $RP_API_KEY_MATS" -H "Content-Type: application/json" \
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
exec > >(stdbuf -oL tee /workspace/boot.log) 2>&1
export HF_TOKEN='$HF_TOKEN'
HFC=hf
upload_log() { \$HFC upload '$HF_REPO' /workspace/boot.log fra_persist/results/boot_${POD_NAME}.log --repo-type dataset >/dev/null 2>&1 || true; }
trap 'echo "[BOOTSTRAP-ERR line \$LINENO]"; upload_log; sleep infinity' ERR
echo "[\$(date -u +%H:%M:%S)] persist bootstrap start ($POD_NAME)"
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
mkdir -p /workspace/code /workspace/jb /workspace/res
# fetch the fra/ code bundle (same one j1/j2/j7 use)
for a in 1 2 3 4 5; do \$HFC download '$HF_REPO' fra_win/fra_bundle.tar.gz --repo-type dataset --local-dir /workspace/jb && break; echo "bundle dl retry \$a"; sleep 20; done
tar -xzf /workspace/jb/fra_win/fra_bundle.tar.gz -C /workspace/code
python3 -c "import sys; sys.path.insert(0,'/workspace/code'); import fra.core.fra; print('fra import OK')"
# fetch the persistence run script from fra_persist/code/
ok=0; for a in 1 2 3 4 5; do \$HFC download '$HF_REPO' fra_persist/code/${PY_SCRIPT:-persist_run.py} --repo-type dataset --local-dir /workspace/jb >/dev/null 2>&1 && { [ -f /workspace/jb/fra_persist/code/${PY_SCRIPT:-persist_run.py} ] && ok=1 && break; }; echo "script dl retry \$a"; sleep 12; done
[ "\$ok" = 1 ] || { echo "[FATAL] ${PY_SCRIPT:-persist_run.py} never downloaded"; upload_log; sleep infinity; }
OUTDIR=/workspace/res; mkdir -p \$OUTDIR; cd \$OUTDIR
echo "[\$(date -u +%H:%M:%S)] bootstrap done -> running ${PY_SCRIPT:-persist_run.py}"
PYTHONPATH=/workspace/code OUTDIR=\$OUTDIR HF_TOKEN='$HF_TOKEN' timeout 5400 python3 /workspace/jb/fra_persist/code/${PY_SCRIPT:-persist_run.py} 2>&1 | tee \$OUTDIR/run.log || echo "[run rc=\$?]"
\$HFC upload '$HF_REPO' \$OUTDIR/run.log fra_persist/results/run_${POD_NAME}.log --repo-type dataset >/dev/null 2>&1 || true
upload_log
echo "[\$(date -u +%H:%M:%S)] persist done -> fra_persist/results/"
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
  "gpuCount": 1, "minVcpuCount": 8, "minMemoryInGb": ${MIN_MEM:-48}, "containerDiskInGb": ${DISK:-40},
  "volumeInGb": 0, "dockerArgs": $cmd_json, "ports": "22/tcp", "startSsh": true }
JSON
)
  payload=$(python3 -c "
import json,sys
inp=json.loads(sys.argv[1])
q='mutation Deploy(\$input: PodFindAndDeployOnDemandInput!) { podFindAndDeployOnDemand(input: \$input) { id name desiredStatus } }'
print(json.dumps({'query':q,'variables':{'input':inp}}))" "$input")
  resp=$(curl -sS -X POST -H "Authorization: Bearer $RP_API_KEY_MATS" -H "Content-Type: application/json" -d "$payload" "$GRAPHQL")
  pid=$(printf '%s' "$resp" | python3 -c "import json,sys
try: d=json.load(sys.stdin)
except: print(''); sys.exit()
print(((d.get('data') or {}).get('podFindAndDeployOnDemand') or {}).get('id') or '')")
  if [ -n "$pid" ]; then echo "[launch] $POD_NAME on [$gpu] id=$pid"; echo "$pid" > /tmp/${POD_NAME}_pod_id.txt; exit 0; fi
  echo "[launch] no capacity [$gpu]: $(printf '%s' "$resp" | head -c 160)" >&2
done < <(printf '%s' "$GPU_TYPE_IDS" | tr '|' '\n')
echo "[launch] FAILED all GPU types" >&2; exit 1
