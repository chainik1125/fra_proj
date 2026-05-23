#!/usr/bin/env bash
# Bootstrap pod entrypoint: train the two SAEs the Fisher POC consumes,
# upload them to the HF dataset repo, self-terminate.
#
# Idempotent: if both checkpoints already exist on HF, skip training.
#
# Env vars consumed:
#   HF_TOKEN, HF_REPO (default dmanningcoe/fisher-poc-tinystories-sleeper)
#   RUNPOD_API_KEY, RUNPOD_POD_ID — for self-terminate on completion
#   BRANCH, REPO_URL — for cloning
set -eo pipefail

# Plain log file (the `exec > >(stdbuf -oL tee ...)` pattern from the
# /dispatch_campaign skill crashed PID-1 bash on this image — see
# orchestrator notes). Each heavy command redirects with `>> $LOGFILE
# 2>&1` instead.
mkdir -p /workspace
LOGFILE=/workspace/bootstrap.log
echo "[$(date +%H:%M:%S)] bootstrap start driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)" >> "$LOGFILE" 2>&1

# Fast-fail if the driver is too old for the cu13 torch wheel.
if command -v nvidia-smi >/dev/null 2>&1; then
  drv_major=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
  if [[ "$drv_major" -lt 525 ]]; then
    echo "driver too old ($drv_major) for torch cu12+ — self-terminate without work"
    if [[ -n "${RUNPOD_POD_ID:-}" && -n "${RUNPOD_API_KEY:-}" ]]; then
      curl -sS -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" \
        -H "Content-Type: application/json" \
        -d "{\"query\":\"mutation { podTerminate(input:{podId:\\\"$RUNPOD_POD_ID\\\"}) }\"}" \
        https://api.runpod.io/graphql || true
    fi
    exit 0
  fi
fi

BRANCH="${BRANCH:-dmitry/fisher-poc}"
REPO_URL="${REPO_URL:-https://github.com/chainik1125/fra_proj.git}"
WORKDIR="${WORKDIR:-/workspace/fra_proj}"
HF_REPO="${HF_REPO:-dmanningcoe/fisher-poc-tinystories-sleeper}"
EXP_DIR="$WORKDIR/experiments/tinystories_sleeper"
POC_DIR="$EXP_DIR/fisher_poc"
LN1_CKPT="$EXP_DIR/recreate_ln1/results/crosscoder_sae_layer0.pt"
LAYER0_CKPT="$EXP_DIR/recreate_layer0/results/crosscoder_sae_layer1.pt"

mkdir -p /workspace
if [[ ! -d "$WORKDIR/.git" ]]; then
  git clone --branch "$BRANCH" --single-branch "$REPO_URL" "$WORKDIR"
fi
cd "$WORKDIR"

bash "$POC_DIR/setup_pod.sh"

# Step 1: check HF for existing checkpoints (idempotent re-runs)
check_exists() {
  local name="$1"
  uv run python -c "
from huggingface_hub import HfApi
import os, sys
api = HfApi(token=os.environ['HF_TOKEN'])
files = api.list_repo_files('$HF_REPO', repo_type='dataset')
sys.exit(0 if '$name' in files else 1)
" 2>/dev/null
}

if check_exists "sae_checkpoints/recreate_ln1_layer0.pt" && \
   check_exists "sae_checkpoints/recreate_layer0_layer1.pt"; then
  echo "[bootstrap] both SAE checkpoints already on HF — skipping training"
else
  echo "[bootstrap] training SAEs (recreate_ln1 + recreate_layer0, skipping sweep+plot)"
  # Train SAEs but skip the post-train sweep and plotting — we only need the .pt files.
  uv run python "$EXP_DIR/recreate_ln1/reproduce.py" \
    "$EXP_DIR/recreate_ln1/config.yaml" \
    --skip sweep plot
  uv run python "$EXP_DIR/recreate_layer0/reproduce.py" \
    "$EXP_DIR/recreate_layer0/config.yaml" \
    --skip sweep plot

  # Upload the two .pt files the Fisher POC consumes.
  uv run python "$POC_DIR/hf_upload.py" \
    --repo "$HF_REPO" \
    --src "$EXP_DIR/recreate_ln1/results" \
    --pattern "crosscoder_sae_layer0.pt" \
    --path-in-repo sae_checkpoints
  # Rename on upload so collisions across pipelines don't clash.
  uv run python -c "
import os, shutil
from huggingface_hub import HfApi
api = HfApi(token=os.environ['HF_TOKEN'])
src = '$LAYER0_CKPT'
dst = 'sae_checkpoints/recreate_layer0_layer1.pt'
api.upload_file(path_or_fileobj=src, path_in_repo=dst,
                repo_id='$HF_REPO', repo_type='dataset')
print('[bootstrap] uploaded', dst)
"
  # The hf_upload script also wrote sae_checkpoints/crosscoder_sae_layer0.pt for
  # the LN1 side — fix the naming to match what the experiment pods expect.
  uv run python -c "
from huggingface_hub import HfApi
import os
api = HfApi(token=os.environ['HF_TOKEN'])
src = '$LN1_CKPT'
dst = 'sae_checkpoints/recreate_ln1_layer0.pt'
api.upload_file(path_or_fileobj=src, path_in_repo=dst,
                repo_id='$HF_REPO', repo_type='dataset')
print('[bootstrap] uploaded', dst)
"
fi

# Drop a status marker so the babysitter knows SAE phase is done.
uv run python -c "
from huggingface_hub import HfApi
import json, os, time
api = HfApi(token=os.environ['HF_TOKEN'])
status = {'phase': 'sae_bootstrap', 'status': 'done', 'unix': time.time()}
import io
api.upload_file(
  path_or_fileobj=json.dumps(status, indent=2).encode(),
  path_in_repo='status_bootstrap.json',
  repo_id='$HF_REPO', repo_type='dataset',
)
print('[bootstrap] wrote status_bootstrap.json')
"

# Self-terminate (destroy pod + volume; we've uploaded everything we need).
if [[ -n "${RUNPOD_POD_ID:-}" && -n "${RUNPOD_API_KEY:-}" ]]; then
  echo "[bootstrap] self-terminating pod $RUNPOD_POD_ID in 30s..."
  sleep 30
  curl -sS -X POST \
    -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -H "Content-Type: application/json" \
    -d "{\"query\":\"mutation { podTerminate(input:{podId:\\\"$RUNPOD_POD_ID\\\"}) }\"}" \
    https://api.runpod.io/graphql || true
fi
