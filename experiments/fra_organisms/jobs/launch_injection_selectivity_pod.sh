#!/usr/bin/env bash
# Single-shot GO/NO-GO pre-check pod for PROMPT-INJECTION (PLANNING.md §5).
# rs-injection-selectivity-1 : bootstrap deps -> download fra/ bundle + injection_selectivity.py ->
# run injection_selectivity.py once -> upload results to fra_org_injection/results/<RUNID>/.
# The script ckpt()s precheck.json after every prompt in each stage and is resume-proof; periodic
# uploads from the runner give partial-upload safety. Judge-free (exact-match canary). Env: RP_API_KEY_MATS, HF_TOKEN.
set -euo pipefail
: "${RP_API_KEY_MATS:?}"; : "${HF_TOKEN:?}"
POD_NAME="${POD_NAME:-rs-injection-selectivity-1}"
RUNID="${RUNID:-$(date -u +%Y%m%d-%H%M%S)}"
HF_REPO="dmanningcoe/fra-phase1-steering-data"
PREFIX="fra_org_injection"
SCRIPT="injection_selectivity.py"
GRAPHQL="https://api.runpod.io/graphql"
IMAGE_GPU="runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
# gemma-2-2b-it is small -> L4/L40 class; allow a spread for capacity.
GPU_TYPE_IDS="${GPU_TYPE_IDS:-NVIDIA L40S|NVIDIA A40|NVIDIA RTX A6000|NVIDIA L40|NVIDIA L4|NVIDIA RTX A5000}"
HERE="$(cd "$(dirname "$0")" && pwd)"
json_str() { python3 -c "import json,sys; print(json.dumps(sys.stdin.read()))"; }

# --- parse-gate the script one more time before any upload ---
python3 -c "import ast; ast.parse(open('$HERE/$SCRIPT').read()); print('parse-gate OK')"

# --- duplicate-name guard (RunPod does NOT refuse dup names) ---
dupe=$(curl -sS -X POST -H "Authorization: Bearer $RP_API_KEY_MATS" -H "Content-Type: application/json" \
  -d '{"query":"query { myself { pods { id name desiredStatus } } }"}' "$GRAPHQL" | \
  python3 -c "import json,sys; d=json.load(sys.stdin); p=(d.get('data') or {}).get('myself',{}).get('pods',[]) or []; print(' '.join(x['id'] for x in p if x['name']=='$POD_NAME' and x.get('desiredStatus')=='RUNNING'))")
if [ -n "$dupe" ]; then echo "ABORT: RUNNING pod named $POD_NAME: $dupe" >&2; exit 1; fi

# --- stage code to HF: the precheck script + reuse the existing fra_bundle.tar.gz ---
echo "[stage] uploading $SCRIPT + bundle to $PREFIX/code/"
hf upload "$HF_REPO" "$HERE/$SCRIPT" "$PREFIX/code/$SCRIPT" --repo-type dataset >/dev/null
# reuse the fra/ bundle already published under fra_win/ (same package the templates use)
hf download "$HF_REPO" fra_win/fra_bundle.tar.gz --repo-type dataset --local-dir /tmp/inj3_bundle >/dev/null 2>&1
hf upload "$HF_REPO" /tmp/inj3_bundle/fra_win/fra_bundle.tar.gz "$PREFIX/code/fra_bundle.tar.gz" --repo-type dataset >/dev/null
echo "[stage] done"

inner=$(cat <<EOF
#!/bin/bash
set -eo pipefail
exec > >(stdbuf -oL tee /workspace/boot.log) 2>&1
export HF_TOKEN='$HF_TOKEN'
HFC=hf
RUNID='$RUNID'
echo "[\$(date -u +%H:%M:%S)] injection precheck bootstrap start (run \$RUNID)"
nvidia-smi -L || true
python3 - <<'PYEOF' > /tmp/constraints.txt
import torch
v=torch.__version__.split('+')[0]; print(f"torch=={v}")
try:
    import torchvision; print(f"torchvision=={torchvision.__version__.split('+')[0]}")
except Exception: pass
PYEOF
pip install --no-input -q -c /tmp/constraints.txt "sae_lens==5.10.7" "transformer_lens==2.18.0" "huggingface_hub[cli]" "pandas" 2>&1 | tail -3
command -v hf >/dev/null 2>&1 || HFC=huggingface-cli
python3 -c "import torch,sae_lens,transformer_lens; print('torch',torch.__version__,'sae_lens',sae_lens.__version__,'cuda',torch.cuda.is_available())"
mkdir -p /workspace/code /workspace/jb /workspace/res/\$RUNID
for a in 1 2 3 4 5; do \$HFC download '$HF_REPO' $PREFIX/code/fra_bundle.tar.gz --repo-type dataset --local-dir /workspace/jb && break; echo "bundle dl retry \$a"; sleep 20; done
tar -xzf /workspace/jb/$PREFIX/code/fra_bundle.tar.gz -C /workspace/code
for a in 1 2 3 4 5; do \$HFC download '$HF_REPO' $PREFIX/code/$SCRIPT --repo-type dataset --local-dir /workspace/jb && [ -f /workspace/jb/$PREFIX/code/$SCRIPT ] && break; echo "script dl retry \$a"; sleep 10; done
python3 -c "import sys; sys.path.insert(0,'/workspace/code'); import fra.core.fra; from fra.sae_lens_wrapper import GemmaScopeSAE; print('fra import OK')"
echo "[\$(date -u +%H:%M:%S)] bootstrap done -> running precheck"
OUTDIR=/workspace/res/\$RUNID
# background periodic uploader (partial-upload safety; faster tick + ships boot.log so a bootstrap
# OR python-stage stall is visible within ~90s instead of ~240s).
( while true; do sleep 90; cp /workspace/boot.log \$OUTDIR/boot.log 2>/dev/null || true; \$HFC upload '$HF_REPO' \$OUTDIR $PREFIX/results/\$RUNID --repo-type dataset >/dev/null 2>&1 || true; done ) &
UPPID=\$!
PYTHONPATH=/workspace/code OUTDIR=\$OUTDIR HF_TOKEN='$HF_TOKEN' MODEL_NAME='gemma-2-2b-it' \\
  timeout 5400 python3 /workspace/jb/$PREFIX/code/$SCRIPT > \$OUTDIR/out.log 2>&1 || echo "  job rc=\$?" >> \$OUTDIR/out.log
kill \$UPPID 2>/dev/null || true
cp /workspace/boot.log \$OUTDIR/boot.log 2>/dev/null || true
\$HFC upload '$HF_REPO' \$OUTDIR $PREFIX/results/\$RUNID --repo-type dataset >/dev/null 2>&1 || true
echo "[\$(date -u +%H:%M:%S)] PRECHECK done -> $PREFIX/results/\$RUNID"
# leave pod idle (orchestrator reaps); print verdict tail
tail -30 \$OUTDIR/out.log || true
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
  if [ -n "$pid" ]; then echo "[launch] $POD_NAME on [$gpu] id=$pid run=$RUNID -> $PREFIX/results/$RUNID"; echo "$pid" > /tmp/injection_selectivity_pod_id.txt; echo "$RUNID" > /tmp/injection_selectivity_runid.txt; exit 0; fi
  echo "[launch] no capacity [$gpu]: $(printf '%s' "$resp" | head -c 160)" >&2
done < <(printf '%s' "$GPU_TYPE_IDS" | tr '|' '\n')
echo "[launch] FAILED all GPU types" >&2; exit 1
