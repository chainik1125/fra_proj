#!/usr/bin/env bash
# Orchestrator: run Fisher POC for one or more seeds on a RunPod GPU pod.
# After each seed completes, push its result JSONs to the HF dataset repo.
# When SEEDS is exhausted, optionally self-stop the pod.
#
# Env vars consumed:
#   SEEDS          space-separated list of integer seeds (default: "0")
#   HF_REPO        HF dataset repo to upload results to
#                  (default: dmanningcoe/fisher-poc-tinystories-sleeper)
#   HF_TOKEN       HF write token (required)
#   RUNPOD_POD_ID  Provided by RunPod env; if set and SELF_STOP=1, calls
#                  runpodctl stop pod at the end.
#   SELF_STOP      "1" to stop the pod when done (default: "0").
set -euo pipefail

WORKDIR="${WORKDIR:-/workspace/fra_proj}"
SEEDS="${SEEDS:-0}"
HF_REPO="${HF_REPO:-dmanningcoe/fisher-poc-tinystories-sleeper}"
SELF_STOP="${SELF_STOP:-0}"

cd "$WORKDIR"

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/root/hf_cache}"
mkdir -p "$HF_HOME"

POC_DIR="$WORKDIR/experiments/tinystories_sleeper/fisher_poc"
OUT_DIR="$POC_DIR/results"
mkdir -p "$OUT_DIR"

# SAE checkpoints are too large for git (`*.pt` is gitignored). The
# bootstrap pod trained them and uploaded to the HF dataset repo under
# sae_checkpoints/.  Pull them down into the expected paths.
for pair in \
  "sae_checkpoints/recreate_ln1_layer0.pt|experiments/tinystories_sleeper/recreate_ln1/results/crosscoder_sae_layer0.pt" \
  "sae_checkpoints/recreate_layer0_layer1.pt|experiments/tinystories_sleeper/recreate_layer0/results/crosscoder_sae_layer1.pt"; do
  src="${pair%|*}"; dst="${pair#*|}"
  if [[ -f "$dst" ]]; then continue; fi
  mkdir -p "$(dirname "$dst")"
  echo "[run_on_pod] fetching $src -> $dst"
  uv run python -c "
from huggingface_hub import hf_hub_download
import os, shutil
p = hf_hub_download('$HF_REPO', '$src', repo_type='dataset', token=os.environ['HF_TOKEN'])
shutil.copy(p, '$dst')
"
done

echo "============================================================"
echo "FISHER POC: seeds=[$SEEDS]   HF_REPO=$HF_REPO"
echo "============================================================"

for SEED in $SEEDS; do
  echo
  echo "--- seed $SEED ---"
  uv run python "$POC_DIR/run_fisher_poc.py" \
    --config "$POC_DIR/config.yaml" \
    --mode both \
    --device cuda \
    --seed "$SEED" \
    --output_dir "$OUT_DIR" \
    2>&1 | tee -a "$OUT_DIR/run_seed${SEED}.log"

  # Push this seed's outputs to HF immediately (so the babysitter sees progress).
  uv run python "$POC_DIR/hf_upload.py" \
    --repo "$HF_REPO" \
    --src "$OUT_DIR" \
    --pattern "*seed${SEED}*"
done

# Also push the log files.
uv run python "$POC_DIR/hf_upload.py" \
  --repo "$HF_REPO" \
  --src "$OUT_DIR" \
  --pattern "*.log"

echo
echo "============================================================"
echo "DONE — all seeds [$SEEDS] uploaded to https://huggingface.co/datasets/$HF_REPO"
echo "============================================================"

if [[ "$SELF_STOP" == "1" && -n "${RUNPOD_POD_ID:-}" && -n "${RUNPOD_API_KEY:-}" ]]; then
  echo "[run_on_pod] self-terminating pod $RUNPOD_POD_ID in 30s..."
  sleep 30
  curl -sS -X POST \
    -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -H "Content-Type: application/json" \
    -d "{\"query\":\"mutation { podTerminate(input:{podId:\\\"$RUNPOD_POD_ID\\\"}) }\"}" \
    https://api.runpod.io/graphql || true
fi
