#!/usr/bin/env bash
# Persistent interactive GPU pod for the FRA-behavioral-win sprint.
# Runs a poll-execute loop: watches HF fra_win/job_id.txt; on change, downloads
# fra_win/jobs/<id>.py, runs it with PYTHONPATH=/workspace/code (the fra/ bundle),
# uploads the result dir to fra_win/out/<id>/.  ONE commit per job (no spam).
# Env: RP_API_KEY_MATS, HF_TOKEN. Optional: POD_NAME.
set -euo pipefail
: "${RP_API_KEY_MATS:?}"; : "${HF_TOKEN:?}"
POD_NAME="${POD_NAME:-fra-win-pod}"
HF_REPO="dmanningcoe/fra-phase1-steering-data"
GRAPHQL="https://api.runpod.io/graphql"
IMAGE_GPU="runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
GPU_TYPE_IDS="${GPU_TYPE_IDS:-NVIDIA L4|NVIDIA RTX A5000|NVIDIA A40|NVIDIA RTX A6000|NVIDIA L40S}"
json_str() { python3 -c "import json,sys; print(json.dumps(sys.stdin.read()))"; }

# duplicate-name guard
dupe=$(curl -sS -X POST -H "Authorization: Bearer $RP_API_KEY_MATS" -H "Content-Type: application/json" \
  -d '{"query":"query { myself { pods { id name desiredStatus } } }"}' "$GRAPHQL" | \
  python3 -c "import json,sys; d=json.load(sys.stdin); p=(d.get('data') or {}).get('myself',{}).get('pods',[]) or []; print(' '.join(x['id'] for x in p if x['name']=='$POD_NAME' and x.get('desiredStatus')=='RUNNING'))")
if [ -n "$dupe" ]; then echo "ABORT: RUNNING pod named $POD_NAME: $dupe" >&2; exit 1; fi

inner=$(cat <<EOF
#!/bin/bash
set -eo pipefail
exec > >(stdbuf -oL tee /workspace/boot.log) 2>&1
export HF_TOKEN='$HF_TOKEN'
HFC=hf
echo "[\$(date -u +%H:%M:%S)] fra-win bootstrap start"
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
# fetch the fra/ code bundle
mkdir -p /workspace/code /workspace/jb /workspace/res
for a in 1 2 3 4 5; do \$HFC download '$HF_REPO' fra_win/fra_bundle.tar.gz --repo-type dataset --local-dir /workspace/jb && break; echo "bundle dl retry \$a"; sleep 20; done
tar -xzf /workspace/jb/fra_win/fra_bundle.tar.gz -C /workspace/code
python3 -c "import sys; sys.path.insert(0,'/workspace/code'); import fra.core.fra; print('fra import OK')"
echo "[\$(date -u +%H:%M:%S)] bootstrap done -> runner loop"
last=""
while true; do
  \$HFC download '$HF_REPO' fra_win/job_id.txt --repo-type dataset --local-dir /workspace/jb >/dev/null 2>&1 || true
  jid=\$(cat /workspace/jb/fra_win/job_id.txt 2>/dev/null | tr -d '[:space:]')
  if [ -n "\$jid" ] && [ "\$jid" != "\$last" ]; then
    echo "[\$(date -u +%H:%M:%S)] JOB \$jid"
    rm -f /workspace/jb/fra_win/jobs/\$jid.py
    ok=0; for a in 1 2 3 4 5; do \$HFC download '$HF_REPO' fra_win/jobs/\$jid.py --repo-type dataset --local-dir /workspace/jb >/dev/null 2>&1 && { [ -f /workspace/jb/fra_win/jobs/\$jid.py ] && ok=1 && break; }; sleep 8; done
    if [ "\$ok" != 1 ]; then echo "  job \$jid script missing, skip"; last=\$jid; continue; fi
    OUTDIR=/workspace/res/\$jid; mkdir -p \$OUTDIR; cd \$OUTDIR
    PYTHONPATH=/workspace/code OUTDIR=\$OUTDIR HF_TOKEN='$HF_TOKEN' timeout 3600 python3 /workspace/jb/fra_win/jobs/\$jid.py > \$OUTDIR/out.log 2>&1 || echo "  job rc=\$?" >> \$OUTDIR/out.log
    \$HFC upload '$HF_REPO' \$OUTDIR fra_win/out/\$jid --repo-type dataset >/dev/null 2>&1 || true
    echo "[\$(date -u +%H:%M:%S)] JOB \$jid done -> fra_win/out/\$jid"
    last=\$jid
  fi
  sleep 10
done
EOF
)
b64=$(printf '%s' "$inner" | base64 | tr -d '\n')
docker_args="bash -c \"echo $b64 | base64 -d > /start_user.sh && chmod +x /start_user.sh && /start.sh & sleep 30 && /start_user.sh\""
cmd_json=$(printf '%s' "$docker_args" | json_str)

while IFS= read -r gpu; do
  [ -z "$gpu" ] && continue
  input=$(cat <<JSON
{ "name": "$POD_NAME", "imageName": "$IMAGE_GPU", "cloudType": "SECURE", "gpuTypeId": "$gpu",
  "gpuCount": 1, "minVcpuCount": 4, "minMemoryInGb": 24, "containerDiskInGb": ${DISK:-40},
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
  if [ -n "$pid" ]; then echo "[launch] $POD_NAME on [$gpu] id=$pid"; echo "$pid" > /tmp/fra_win_pod_id.txt; exit 0; fi
  echo "[launch] no capacity [$gpu]: $(printf '%s' "$resp" | head -c 160)" >&2
done < <(printf '%s' "$GPU_TYPE_IDS" | tr '|' '\n')
echo "[launch] FAILED all GPU types" >&2; exit 1
