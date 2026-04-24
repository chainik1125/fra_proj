#!/usr/bin/env bash
# Tracing-feature pipeline: sync code to the GPU pod, run the 5 analysis
# scripts, pull results back. Modelled on recreate_layer0/reproduce.sh.
#
# Requires that the SAE checkpoints already exist on the remote at:
#   $REMOTE_DIR/experiments/tinystories_sleeper/recreate_layer0/results/crosscoder_sae_layer{0,1,2}.pt
#   $REMOTE_DIR/experiments/tinystories_sleeper/recreate_ln1/results/crosscoder_sae_layer0.pt
#
# Usage:
#   ./reproduce.sh                # run full pipeline
#   REMOTE=a40_climb ./reproduce.sh
#   SKIP_CACHE=1 ./reproduce.sh   # skip the expensive cache step

set -euo pipefail

REMOTE="${REMOTE:-a40_2}"
REMOTE_DIR="${REMOTE_DIR:-/root/fra_proj}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
EXP_REL="experiments/tinystories_sleeper"
RUN_REL="$EXP_REL/tracing_feature"

echo "[trace] remote=$REMOTE  repo=$REPO_ROOT"
echo "[trace] ensuring remote directories exist..."
ssh "$REMOTE" "mkdir -p $REMOTE_DIR/$RUN_REL/scripts $REMOTE_DIR/$RUN_REL/results"

echo "[trace] syncing experiment-level sources..."
rsync -az \
  --exclude='__pycache__' --exclude='*.pyc' --exclude='outputs/' \
  "$REPO_ROOT/$EXP_REL/"*.py \
  "$REMOTE:$REMOTE_DIR/$EXP_REL/"

echo "[trace] syncing tracing_feature scripts..."
rsync -az \
  --exclude='__pycache__' --exclude='*.pyc' --exclude='results/' \
  "$REPO_ROOT/$RUN_REL/scripts/" \
  "$REMOTE:$REMOTE_DIR/$RUN_REL/scripts/"
rsync -az "$REPO_ROOT/$RUN_REL/reproduce.sh" "$REMOTE:$REMOTE_DIR/$RUN_REL/"
rsync -az "$REPO_ROOT/$RUN_REL/README.md" "$REMOTE:$REMOTE_DIR/$RUN_REL/" || true

PY=".venv/bin/python"

if [[ "${SKIP_CACHE:-0}" == "1" ]]; then
  CACHE_STEP=":"
  echo "[trace] SKIP_CACHE=1 - assuming $REMOTE_DIR/$RUN_REL/results/layer0_cache.pt exists"
else
  CACHE_STEP="$PY $RUN_REL/scripts/cache_layer0_activations.py"
fi

REMOTE_CMD="set -euo pipefail && cd $REMOTE_DIR"
REMOTE_CMD="$REMOTE_CMD && mkdir -p $RUN_REL/results"
REMOTE_CMD="$REMOTE_CMD && $CACHE_STEP"
REMOTE_CMD="$REMOTE_CMD && $PY $RUN_REL/scripts/skip_path.py --plot"
REMOTE_CMD="$REMOTE_CMD && $PY $RUN_REL/scripts/ov_path.py"
REMOTE_CMD="$REMOTE_CMD && $PY $RUN_REL/scripts/pre_attn_path.py"
REMOTE_CMD="$REMOTE_CMD && $PY $RUN_REL/scripts/combine.py --plot"
# Two-stage path requires SAE_ln1; run only if available.
REMOTE_CMD="$REMOTE_CMD && ($PY $RUN_REL/scripts/two_stage_path.py || echo '[trace] two_stage_path skipped (SAE_ln1 unavailable)')"
REMOTE_CMD="$REMOTE_CMD && $PY $RUN_REL/scripts/head_ablation.py"
REMOTE_CMD="$REMOTE_CMD && $PY $RUN_REL/scripts/multi_head_ablation.py"

echo "[trace] running pipeline on $REMOTE..."
ssh -o ServerAliveInterval=60 -o ServerAliveCountMax=120 "$REMOTE" "$REMOTE_CMD"

echo "[trace] pulling results -> $REPO_ROOT/$RUN_REL/results/"
mkdir -p "$REPO_ROOT/$RUN_REL/results"
rsync -az --progress \
  --exclude='layer0_cache.pt' \
  "$REMOTE:$REMOTE_DIR/$RUN_REL/results/" \
  "$REPO_ROOT/$RUN_REL/results/"

echo "[trace] done. Open $REPO_ROOT/$RUN_REL/results/"
