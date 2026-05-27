#!/usr/bin/env bash
# Methodology pass: Exp 11 v2 (floor source) + Exp 6 v2 (corrected value-path) + the
# rebaselined hybrid (strip vs inert baseline). Uploads to headroom_results/method/, self-stops.
# Env: HF_TOKEN (req); BRANCH; SELF_STOP.
set -uo pipefail
HF_REPO="dmanningcoe/sae-scaling-tinystories-sleeper"
BRANCH="${BRANCH:-dmitry/sae-scaling-sweep}"
REPO_URL="https://github.com/chainik1125/fra_proj.git"
RESDIR=/workspace/results; mkdir -p "$RESDIR"; export HF_TOKEN HF_REPO
echo "[meth] $(date +%H:%M:%S) start"

[ -d /workspace/fra_proj/.git ] || { apt-get update -qq && apt-get install -y -q git >/dev/null 2>&1; \
  git clone --branch "$BRANCH" --single-branch "$REPO_URL" /workspace/fra_proj; }
cd /workspace/fra_proj && git fetch origin && git checkout "$BRANCH" && git pull --ff-only || true

pip install -q transformer-lens datasets peft huggingface_hub einops
pip install -q --force-reinstall --no-deps torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124
pip install -q nvidia-cusparselt-cu12
echo "$(dirname "$(find / -name 'libcusparseLt.so.0' 2>/dev/null | head -1)")" > /etc/ld.so.conf.d/cusparselt.conf && ldconfig
python -c "import torch,transformer_lens;print('[meth] env ok',torch.__version__,torch.cuda.is_available())" || { echo ABORT; exit 1; }

upload() { python - "$1" "$2" <<'PY' 2>&1 | tail -1
import sys,os
from huggingface_hub import HfApi
try:
    HfApi(token=os.environ["HF_TOKEN"]).upload_file(path_or_fileobj=sys.argv[1],path_in_repo=sys.argv[2],
        repo_id=os.environ["HF_REPO"],repo_type="dataset"); print("uploaded",sys.argv[2])
except Exception as e: print("WARN",e)
PY
}
run() {  # $1 module  $2 outname  $3.. args
  local mod="$1" out="$RESDIR/$2"; shift 2
  echo "[meth] $(date +%H:%M:%S) RUN $mod $*"
  if python -u -m "scripts.$mod" "$@" --out "$out" >> "/workspace/log_$mod.log" 2>&1; then
    [ -f "$out" ] && upload "$out" "headroom_results/method/$(basename "$out")"
  else echo "[meth] FAILED $mod"; tail -c 4000 "/workspace/log_$mod.log" > "$RESDIR/FAILED_$2.log"; upload "$RESDIR/FAILED_$2.log" "headroom_results/method/FAILED_$2.log"; fi
}

run exp11_floor_v2 exp11_v2.json
run exp6_ovbound_v2 exp6v2_d12288_k32_s0.json --hook ln1 --width 12288 --k 32 --seed 0
run exp6_ovbound_v2 exp6v2_d12288_k10_s0.json --hook ln1 --width 12288 --k 10 --seed 0
run exp6_ovbound_v2 exp6v2_d24576_k32_s0.json --hook ln1 --width 24576 --k 32 --seed 0
run hybrid_sweep    hybrid_strip_d12288_k32_s0.json --hook ln1 --width 12288 --k 32 --seed 0 --baseline strip
run hybrid_sweep    hybrid_inert_d12288_k32_s0.json --hook ln1 --width 12288 --k 32 --seed 0 --baseline inert

echo "[meth] $(date +%H:%M:%S) ALL DONE"
if [ "${SELF_STOP:-0}" = "1" ] && [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
  curl -sS -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" -A "curl/8.0" \
    --data-binary "{\"query\":\"mutation { podTerminate(input:{podId:\\\"$RUNPOD_POD_ID\\\"}) }\"}" https://api.runpod.io/graphql
fi
