#!/usr/bin/env bash
# Single-command reproducer for the layer-0 resid_pre/mid/post variant.
# Syncs code → a40_climb, runs the pipeline, pulls results back.

set -euo pipefail

REMOTE="${REMOTE:-a40_climb}"
REMOTE_DIR="${REMOTE_DIR:-/root/fra_proj}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
EXP_REL="experiments/tinystories_sleeper"
RUN_REL="$EXP_REL/recreate_layer0"

echo "[reproduce-layer0] remote=$REMOTE  repo=$REPO_ROOT"
echo "[reproduce-layer0] syncing code to $REMOTE:$REMOTE_DIR …"

rsync -az \
  --exclude='__pycache__' --exclude='*.pyc' --exclude='outputs/' \
  "$REPO_ROOT/$EXP_REL/"*.py \
  "$REPO_ROOT/$EXP_REL/README.md" \
  "$REMOTE:$REMOTE_DIR/$EXP_REL/"

rsync -az \
  --exclude='__pycache__' --exclude='*.pyc' --exclude='results/' \
  "$REPO_ROOT/$RUN_REL/config.yaml" \
  "$REPO_ROOT/$RUN_REL/reproduce.py" \
  "$REPO_ROOT/$RUN_REL/reproduce.sh" \
  "$REMOTE:$REMOTE_DIR/$RUN_REL/"

echo "[reproduce-layer0] running pipeline on $REMOTE …"
ssh -o ServerAliveInterval=60 -o ServerAliveCountMax=120 "$REMOTE" \
  "cd $REMOTE_DIR && .venv/bin/python $RUN_REL/reproduce.py $*"

echo "[reproduce-layer0] pulling results → $REPO_ROOT/$RUN_REL/results/"
mkdir -p "$REPO_ROOT/$RUN_REL/results"
rsync -az --progress \
  --exclude='activations_cache.pt' \
  "$REMOTE:$REMOTE_DIR/$RUN_REL/results/" \
  "$REPO_ROOT/$RUN_REL/results/"

echo
echo "===================== MANIFEST ====================="
cat "$REPO_ROOT/$RUN_REL/results/MANIFEST.md" 2>/dev/null || echo "(no MANIFEST.md found)"
echo "===================================================="
echo
echo "[reproduce-layer0] done. Open $REPO_ROOT/$RUN_REL/results/"
