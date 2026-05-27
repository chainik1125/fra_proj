#!/usr/bin/env bash
# Self-driving overnight headroom-robustness driver (P1: hybrid_sweep + resid_sweep_hybrid
# across the HF SAE configs for ONE seed-shard). Runs ON the pod, uploads result JSON to HF
# after every config (so partial results survive a sleeping laptop), then SELF_STOPs.
#
# Required env:  SEED  HF_TOKEN
# Optional env:  WIDTHS (default "12288 24576")  KS (default "10 32 50")
#                HOOKS  (default "ln1 resid_mid")
#                P2 (1 → also run qk_ov_grid per ln1 config)  P3 (1 → run decomp once)
#                HF_REPO (default dmanningcoe/sae-scaling-tinystories-sleeper)
#                BRANCH  (default dmitry/sae-scaling-sweep)
#                SELF_STOP (1 → podStop on completion)
set -uo pipefail
HF_REPO="${HF_REPO:-dmanningcoe/sae-scaling-tinystories-sleeper}"
WIDTHS="${WIDTHS:-12288 24576}"; KS="${KS:-10 32 50}"; HOOKS="${HOOKS:-ln1 resid_mid}"
BRANCH="${BRANCH:-dmitry/sae-scaling-sweep}"
REPO_URL="${REPO_URL:-https://github.com/chainik1125/fra_proj.git}"
RESDIR=/workspace/results; mkdir -p "$RESDIR"
export HF_TOKEN
echo "[drv] $(date +%H:%M:%S) seed=$SEED widths=[$WIDTHS] ks=[$KS] hooks=[$HOOKS]"

# ── clone (idempotent) ──
if [ ! -d /workspace/fra_proj/.git ]; then
  apt-get update -qq && apt-get install -y -q git >/dev/null 2>&1
  git clone --branch "$BRANCH" --single-branch "$REPO_URL" /workspace/fra_proj
fi
cd /workspace/fra_proj

# ── env fix (torch 2.6.0+cu124 + cusparselt) ──
pip install -q transformer-lens datasets peft huggingface_hub einops
pip install -q --force-reinstall --no-deps \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124
pip install -q nvidia-cusparselt-cu12
CUSPARSELT_DIR="$(dirname "$(find / -name 'libcusparseLt.so.0' 2>/dev/null | head -1)")"
echo "$CUSPARSELT_DIR" > /etc/ld.so.conf.d/cusparselt.conf && ldconfig
python -c "import torch,transformer_lens;print('[drv] env ok torch',torch.__version__,'cuda',torch.cuda.is_available())" \
    || { echo "[drv] ENV BROKEN — abort"; exit 1; }

upload() {  # $1 = local json path; uploads to HF under headroom_results/seed<SEED>/
  local f="$1"; local base; base="$(basename "$f")"
  python - "$f" "headroom_results/seed${SEED}/${base}" <<'PY' 2>&1 | tail -1
import sys,os
from huggingface_hub import HfApi
api=HfApi(token=os.environ["HF_TOKEN"])
try:
    api.upload_file(path_or_fileobj=sys.argv[1],path_in_repo=sys.argv[2],
                    repo_id=os.environ.get("HF_REPO","dmanningcoe/sae-scaling-tinystories-sleeper"),
                    repo_type="dataset")
    print("uploaded",sys.argv[2])
except Exception as e:
    print("WARN upload failed",sys.argv[2],e)
PY
}
export HF_REPO

run_cfg() {  # $1=script_module $2=hook $3=width $4=k $5=outname
  local mod="$1" hook="$2" w="$3" k="$4" out="$RESDIR/$5"
  echo "[drv] $(date +%H:%M:%S) RUN $mod hook=$hook d$w k$k seed=$SEED"
  if python -u -m "scripts.$mod" --hook "$hook" --width "$w" --k "$k" --seed "$SEED" --out "$out" \
      >> "/workspace/log_${5%.json}.log" 2>&1; then
    [ -f "$out" ] && upload "$out"
  else
    echo "[drv] FAILED $mod hook=$hook d$w k$k (see log_${5%.json}.log) — skip"
    local lf="/workspace/log_${5%.json}.log"
    [ -f "$lf" ] && tail -c 4000 "$lf" > "$RESDIR/FAILED_${5%.json}.log" && upload "$RESDIR/FAILED_${5%.json}.log"
  fi
}

for w in $WIDTHS; do for k in $KS; do
  # P1: hybrid_sweep (OV path) uses the ln1 SAE; resid_sweep_hybrid uses the resid_mid SAE.
  for hook in $HOOKS; do
    if [ "$hook" = "ln1" ]; then
      run_cfg hybrid_sweep       ln1       "$w" "$k" "hybrid_ln1_d${w}_k${k}_s${SEED}.json"
      [ "${P2:-0}" = "1" ] && run_cfg qk_ov_grid ln1 "$w" "$k" "qkov_ln1_d${w}_k${k}_s${SEED}.json"
    else
      run_cfg resid_sweep_hybrid resid_mid "$w" "$k" "resid_resid_mid_d${w}_k${k}_s${SEED}.json"
    fi
  done
done; done

# P3 decomposition (SAE-free; only meaningful once — run on seed-0 shard only)
if [ "${P3:-0}" = "1" ] && [ "$SEED" = "0" ]; then
  echo "[drv] $(date +%H:%M:%S) RUN decomp"
  if python -u -m scripts.ov_qk_decomp >> /workspace/log_decomp.log 2>&1; then
    [ -f "$RESDIR/decomp.json" ] && upload "$RESDIR/decomp.json"
  fi
fi

echo "[drv] $(date +%H:%M:%S) seed=$SEED ALL CONFIGS DONE"
if [ "${SELF_STOP:-0}" = "1" ] && [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
  curl -sS -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
    -A "curl/8.0" --data-binary \
    "{\"query\":\"mutation { podTerminate(input:{podId:\\\"$RUNPOD_POD_ID\\\"}) }\"}" \
    https://api.runpod.io/graphql
fi
