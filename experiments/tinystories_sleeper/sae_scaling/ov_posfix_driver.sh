#!/usr/bin/env bash
# Run ov_posfix.py (OV-steer + positional-embedding correction), upload, self-stop.
set -uo pipefail
HF_REPO="dmanningcoe/sae-scaling-tinystories-sleeper"; BRANCH="${BRANCH:-dmitry/sae-scaling-sweep}"
REPO_URL="https://github.com/chainik1125/fra_proj.git"; RESDIR=/workspace/results; mkdir -p "$RESDIR"; export HF_TOKEN HF_REPO
[ -d /workspace/fra_proj/.git ] || { apt-get update -qq && apt-get install -y -q git >/dev/null 2>&1; git clone --branch "$BRANCH" --single-branch "$REPO_URL" /workspace/fra_proj; }
cd /workspace/fra_proj && git fetch origin && git checkout "$BRANCH" && git pull --ff-only || true
pip install -q transformer-lens datasets peft huggingface_hub einops
pip install -q --force-reinstall --no-deps torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -q nvidia-cusparselt-cu12
echo "$(dirname "$(find / -name 'libcusparseLt.so.0' 2>/dev/null | head -1)")" > /etc/ld.so.conf.d/cusparselt.conf && ldconfig
python -c "import torch,transformer_lens" || { echo ABORT; exit 1; }
upload(){ python - "$1" "$2" <<'PY' 2>&1 | tail -1
import sys,os
from huggingface_hub import HfApi
try: HfApi(token=os.environ["HF_TOKEN"]).upload_file(path_or_fileobj=sys.argv[1],path_in_repo=sys.argv[2],repo_id=os.environ["HF_REPO"],repo_type="dataset"); print("uploaded",sys.argv[2])
except Exception as e: print("WARN",e)
PY
}
if python -u -m scripts.ov_posfix --out "$RESDIR/ov_posfix.json" >> /workspace/log_ovpf.log 2>&1; then
  upload "$RESDIR/ov_posfix.json" "headroom_results/method/ov_posfix.json"
else tail -c 4000 /workspace/log_ovpf.log > "$RESDIR/FAILED_ovpf.log"; upload "$RESDIR/FAILED_ovpf.log" "headroom_results/method/FAILED_ovpf.log"; fi
echo "[ovpf] DONE"
if [ "${SELF_STOP:-0}" = "1" ] && [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
  curl -sS -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" -A "curl/8.0" \
    --data-binary "{\"query\":\"mutation { podTerminate(input:{podId:\\\"$RUNPOD_POD_ID\\\"}) }\"}" https://api.runpod.io/graphql
fi
