#!/usr/bin/env bash
# Self-driving driver for headroom Exp 6 (OV expressivity bound) + Exp 11 (metric floor).
# Exp 11 is SAE-free -> runs ONCE. Exp 6 is cheap LA -> all ln1 (width,k,seed) configs on one pod.
# Uploads each result to HF under headroom_results/exp6|exp11/, then SELF_STOPs.
# Env: HF_TOKEN (req); SEEDS (default "0 1 2"); WIDTHS ("12288 24576"); KS ("10 32 50");
#      HF_REPO; BRANCH; SELF_STOP (1 -> podTerminate on completion).
set -uo pipefail
HF_REPO="${HF_REPO:-dmanningcoe/sae-scaling-tinystories-sleeper}"
SEEDS="${SEEDS:-0 1 2}"; WIDTHS="${WIDTHS:-12288 24576}"; KS="${KS:-10 32 50}"
BRANCH="${BRANCH:-dmitry/sae-scaling-sweep}"
REPO_URL="${REPO_URL:-https://github.com/chainik1125/fra_proj.git}"
RESDIR=/workspace/results; mkdir -p "$RESDIR"; export HF_TOKEN HF_REPO
echo "[e116] $(date +%H:%M:%S) seeds=[$SEEDS] widths=[$WIDTHS] ks=[$KS]"

if [ ! -d /workspace/fra_proj/.git ]; then
  apt-get update -qq && apt-get install -y -q git >/dev/null 2>&1
  git clone --branch "$BRANCH" --single-branch "$REPO_URL" /workspace/fra_proj
fi
cd /workspace/fra_proj && git fetch origin && git checkout "$BRANCH" && git pull --ff-only || true

pip install -q transformer-lens datasets peft huggingface_hub einops
pip install -q --force-reinstall --no-deps torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124
pip install -q nvidia-cusparselt-cu12
CUSPARSELT_DIR="$(dirname "$(find / -name 'libcusparseLt.so.0' 2>/dev/null | head -1)")"
echo "$CUSPARSELT_DIR" > /etc/ld.so.conf.d/cusparselt.conf && ldconfig
python -c "import torch,transformer_lens;print('[e116] env ok torch',torch.__version__,'cuda',torch.cuda.is_available())" \
    || { echo "[e116] ENV BROKEN — abort"; exit 1; }

upload() {  # $1 local json ; $2 path_in_repo
  python - "$1" "$2" <<'PY' 2>&1 | tail -1
import sys,os
from huggingface_hub import HfApi
api=HfApi(token=os.environ["HF_TOKEN"])
try:
    api.upload_file(path_or_fileobj=sys.argv[1],path_in_repo=sys.argv[2],
                    repo_id=os.environ["HF_REPO"],repo_type="dataset"); print("uploaded",sys.argv[2])
except Exception as e: print("WARN upload failed",sys.argv[2],e)
PY
}

# ── Exp 11: metric floor (SAE-free, once) ──
echo "[e116] $(date +%H:%M:%S) RUN exp11_floor"
if python -u -m scripts.exp11_floor --out "$RESDIR/exp11_floor.json" >> /workspace/log_exp11.log 2>&1; then
  upload "$RESDIR/exp11_floor.json" "headroom_results/exp11/exp11_floor.json"
else
  echo "[e116] exp11 FAILED"; tail -c 4000 /workspace/log_exp11.log > "$RESDIR/FAILED_exp11.log"
  upload "$RESDIR/FAILED_exp11.log" "headroom_results/exp11/FAILED_exp11.log"
fi

# ── Exp 6: OV expressivity bound (per ln1 SAE config) ──
for s in $SEEDS; do for w in $WIDTHS; do for k in $KS; do
  out="$RESDIR/exp6_ln1_d${w}_k${k}_s${s}.json"
  echo "[e116] $(date +%H:%M:%S) RUN exp6 d$w k$k s$s"
  if python -u -m scripts.exp6_ovbound --hook ln1 --width "$w" --k "$k" --seed "$s" --out "$out" \
      >> "/workspace/log_exp6_d${w}_k${k}_s${s}.log" 2>&1; then
    [ -f "$out" ] && upload "$out" "headroom_results/exp6/$(basename "$out")"
  else
    echo "[e116] exp6 FAILED d$w k$k s$s"
    tail -c 4000 "/workspace/log_exp6_d${w}_k${k}_s${s}.log" > "$RESDIR/FAILED_exp6_d${w}_k${k}_s${s}.log"
    upload "$RESDIR/FAILED_exp6_d${w}_k${k}_s${s}.log" "headroom_results/exp6/$(basename "$RESDIR/FAILED_exp6_d${w}_k${k}_s${s}.log")"
  fi
done; done; done

echo "[e116] $(date +%H:%M:%S) ALL DONE"
if [ "${SELF_STOP:-0}" = "1" ] && [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
  curl -sS -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
    -A "curl/8.0" --data-binary \
    "{\"query\":\"mutation { podTerminate(input:{podId:\\\"$RUNPOD_POD_ID\\\"}) }\"}" \
    https://api.runpod.io/graphql
fi
