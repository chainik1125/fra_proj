#!/usr/bin/env bash
# Per-seed GPU worker for the SAE-scaling sweep.
# Runs the full (d_sae × k) grid producer for BOTH hookpoints (streaming
# checkpoints to HF every 10k steps) plus the consumer (poll→eval→upload) for
# one seed. Workers push everything to HF, so they don't need SSH.
#
# Required env:  SEED  HF_TOKEN
# Optional env:  HF_REPO (default dmanningcoe/sae-scaling-tinystories-sleeper)
#                ALPHAS  (default "-4 -2 0 2 4")
#                BRANCH  (default dmitry/sae-scaling-sweep)
#                SELF_STOP (1 → podStop on completion; needs RUNPOD_API_KEY+RUNPOD_POD_ID)
set -uo pipefail

HF_REPO="${HF_REPO:-dmanningcoe/sae-scaling-tinystories-sleeper}"
ALPHAS="${ALPHAS:--4 -2 0 2 4}"
BRANCH="${BRANCH:-dmitry/sae-scaling-sweep}"
REPO_URL="${REPO_URL:-https://github.com/chainik1125/fra_proj.git}"
export HF_TOKEN

echo "[gpu] $(date +%H:%M:%S) seed=$SEED repo=$HF_REPO  driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"

# ── clone (idempotent) ──────────────────────────────────────────────────────
if [ ! -d /workspace/fra_proj/.git ]; then
  apt-get update -qq && apt-get install -y -q git >/dev/null 2>&1
  git clone --branch "$BRANCH" --single-branch "$REPO_URL" /workspace/fra_proj
fi
cd /workspace/fra_proj

# ── environment fix (see reference-runpod-torch-env memory) ─────────────────
# transformer-lens clobbers the image torch with cu130 (driver too old). Force
# torch 2.6.0+cu124 (has device_mesh for current transformers; cu124 runtime
# runs on the 12.8 driver), then register libcusparseLt for torch 2.6.
pip install -q transformer-lens datasets peft huggingface_hub einops
pip install -q --force-reinstall --no-deps \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124
pip install -q nvidia-cusparselt-cu12
CUSPARSELT_DIR="$(dirname "$(find / -name 'libcusparseLt.so.0' 2>/dev/null | head -1)")"
echo "$CUSPARSELT_DIR" > /etc/ld.so.conf.d/cusparselt.conf && ldconfig
python -c "import torch,transformer_lens;print('[gpu] env ok torch',torch.__version__,'cuda',torch.cuda.is_available())" \
    || { echo "[gpu] ENV BROKEN — abort"; exit 1; }

# ── producer (both hookpoints, full grid to 50k) + consumer ─────────────────
# Producers save checkpoints LOCAL only (--no_hf): the co-located consumer is
# the SOLE HF uploader, via batched single-commit folder uploads, to stay under
# HF's 128-commits/hour cap. All HF ops are non-fatal.
python -u -m scripts.train_sae_scaling --hookpoint ln1       --seed "$SEED" --no_hf \
    > /workspace/train_ln1.log 2>&1 &
python -u -m scripts.train_sae_scaling --hookpoint resid_mid --seed "$SEED" --no_hf \
    > /workspace/train_resid.log 2>&1 &
python -u -m scripts.eval_poll --seed "$SEED" --hf_repo "$HF_REPO" --alphas $ALPHAS \
    > /workspace/eval.log 2>&1 &
wait
echo "[gpu] $(date +%H:%M:%S) seed=$SEED ALL PROCESSES EXITED"

if [ "${SELF_STOP:-0}" = "1" ] && [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
  curl -sS -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
    -A "curl/8.0" --data-binary \
    "{\"query\":\"mutation { podStop(input:{podId:\\\"$RUNPOD_POD_ID\\\"}) { id } }\"}" \
    https://api.runpod.io/graphql
fi
