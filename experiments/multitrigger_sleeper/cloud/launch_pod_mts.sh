#!/usr/bin/env bash
# Generic launcher for ONE RunPod GPU pod running an mts_singlefeat/code/ pod script.
# Derived from launch_pod_singlefeat.sh with one fix: huggingface_hub >=1.0 renamed
# `huggingface-cli` -> `hf`; the bootstrap now resolves whichever exists ($HFC).
# Artifacts/code/results via HF: dmanningcoe/fra-phase1-steering-data : mts_singlefeat/{code,artifacts,results}
# Required env: RUNPOD_API_KEY, HF_TOKEN, POD_NAME, PY_SCRIPT, OUT_JSON
set -euo pipefail
: "${RUNPOD_API_KEY:?}"; : "${HF_TOKEN:?}"; : "${POD_NAME:?}"; : "${PY_SCRIPT:?}"; : "${OUT_JSON:?}"
EXTRA_PIP="${EXTRA_PIP:-}"
RUN_LOG="${RUN_LOG:-${POD_NAME}_run.log}"
GPU_TYPE_IDS="${GPU_TYPE_IDS:-NVIDIA A40|NVIDIA L4|NVIDIA RTX A6000|NVIDIA L40|NVIDIA L40S}"
# torch >=2.6 needed: transformers 4.57 refuses torch.load on older (CVE-2025-32434).
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
export SLEEPER='${SLEEPER:-multi}'
export TRAIN_ON='${TRAIN_ON:-union}'
export CONFIG='${CONFIG:-}'
export RUN_SEED='${RUN_SEED:-7}'
export HOOKS='${HOOKS:-ln1,resid_mid,resid_post}'
export LAYERS='${LAYERS:-0,1,2,3}'
export MODEL='${MODEL:-K1}'
export HOOK='${HOOK:-hook_resid_mid}'
export N_VEC='${N_VEC:-256}'
export DIRECTION='${DIRECTION:-within_model}'
[ -n "${HOOKS_ALL:-}" ] && export HOOKS_ALL='${HOOKS_ALL:-}'
[ -n "${WORKLIST:-}" ] && export WORKLIST='${WORKLIST:-}'
HFC=hf
upload_log() { \$HFC upload '$HF_REPO' /workspace/run.log mts_singlefeat/results/$RUN_LOG --repo-type dataset >/dev/null 2>&1 || true; }
trap 'echo "[BOOTSTRAP-ERR line \$LINENO]"; upload_log; sleep infinity' ERR
echo "[\$(date -u +%H:%M:%S)] mts bootstrap start ($POD_NAME -> $PY_SCRIPT)"
nvidia-smi -L || true
# Pin the torch family to the image's own build so pip never clobbers it.
python3 - <<'PYEOF' > /tmp/constraints.txt
import torch
v = torch.__version__.split('+')[0]
print(f"torch=={v}")
try:
    import torchvision; print(f"torchvision=={torchvision.__version__.split('+')[0]}")
except Exception: pass
PYEOF
cat /tmp/constraints.txt
pip install --no-input -q -c /tmp/constraints.txt "transformers==4.57.6" "datasets==4.8.4" \\
    "transformer-lens==2.18.0" "peft==0.19.1" "typeguard==4.5.1" "jaxtyping==0.3.9" \\
    "einops==0.8.2" accelerate "huggingface_hub[cli]" 2>&1 | tail -2
python3 -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__)"
command -v hf >/dev/null 2>&1 || HFC=huggingface-cli
echo "[bootstrap] HF CLI: \$HFC"
[ -n '$EXTRA_PIP' ] && pip install --no-input -q -c /tmp/constraints.txt $EXTRA_PIP 2>&1 | tail -1
# retry the dataset download: concurrent pod launches can 429 the Hub, and this step is
# NOT inside the python retry loop -> a bare failure here would hit the ERR trap.
for dlat in 1 2 3 4 5 6 7 8; do
  set -f  # no pathname globbing -> exclude patterns pass to hf literally
  \$HFC download '$HF_REPO' --repo-type dataset --include "mts_singlefeat/*" ${DL_EXCLUDE:-} --local-dir /workspace >/tmp/dsdl.log 2>&1 || true
  set +f
  # check the ACTUAL artifact (code dir), not the pipe exit -- 'cmd | tail' masks a 429 as success
  [ -d /workspace/mts_singlefeat/code ] && { echo "[bootstrap] dataset present (attempt \$dlat)"; break; }
  echo "[bootstrap] dataset dl attempt \$dlat incomplete (HF 429?): \$(tail -1 /tmp/dsdl.log); sleep 60"; sleep 60
done
[ -d /workspace/mts_singlefeat/code ] || { echo "[bootstrap] FATAL: code dir never downloaded after 8 tries"; upload_log; sleep infinity; }
# prefetch the base model into the HF cache ONCE (robust, with backoff) so the per-pod
# from_pretrained calls are warm-cache reads -- not N concurrent cold pulls of roneneldan/*
# racing into a 429. This is the real fix; the retry loops above are the safety net.
for mat in 1 2 3 4 5 6; do
  python3 -c "from huggingface_hub import snapshot_download as s; s('roneneldan/TinyStories-Instruct-33M')" && break
  echo "[bootstrap] base-model prefetch attempt \$mat failed (HF 429?), sleep 45"; sleep 45
done
mkdir -p /workspace/out
# COMMIT-BUDGET FIX: the old loop uploaded run.log + progress every 240s; at 30 pods that is
# ~900 commits/hr and blew HF's 128-commits/hr cap (results then 429-dropped). Per-cell result
# files (uploaded inside the python) ARE the progress signal -> monitor via the k8grid tree.
# Keep only a SLOW (hourly) progress-json push for coarse per-pod visibility; no periodic log.
( while true; do sleep 3600; \\
    \$HFC upload '$HF_REPO' /workspace/out/$OUT_JSON mts_singlefeat/results/$OUT_JSON --repo-type dataset >/dev/null 2>&1 || true; done ) &
cd /workspace/mts_singlefeat/code
# retry loop: transient HF 504s during dataset load have killed runs before
for attempt in 1 2 3; do
  if ADAPTER_PATH=/workspace/mts_singlefeat/artifacts/adapters/K8 \\
     SAE_PATH=/workspace/mts_singlefeat/artifacts/sae_ln1_K8.pt \\
     OUT_PATH=/workspace/out/$OUT_JSON \\
     python3 $PY_SCRIPT; then break; fi
  echo "[bootstrap] $PY_SCRIPT failed (attempt \$attempt), retrying in 90s"; sleep 90
  [ "\$attempt" = 3 ] && exit 1
done
echo "[\$(date -u +%H:%M:%S)] $PY_SCRIPT done, final upload"
\$HFC upload '$HF_REPO' /workspace/out/$OUT_JSON mts_singlefeat/results/$OUT_JSON --repo-type dataset
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
  "containerDiskInGb": ${DISK:-60}, "volumeInGb": 0, "dockerArgs": $cmd_json, "ports": "22/tcp", "startSsh": true }
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
