#!/usr/bin/env bash
# Single-command reproducer:
#   1. sync code → a40_climb (SSH host alias)
#   2. run reproduce.py on the remote (routes all outputs to results/)
#   3. rsync results/ back to local (excluding the huge activations cache)
#   4. print MANIFEST.md so you see exactly what was produced.
#
# Usage:  ./reproduce.sh               # full pipeline
#         ./reproduce.sh --skip harvest train sweep   # re-plot only
#
# Override the host with:  REMOTE=my_host ./reproduce.sh

set -euo pipefail

REMOTE="${REMOTE:-a40_climb}"
REMOTE_DIR="${REMOTE_DIR:-/root/fra_proj}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
EXP_REL="experiments/tinystories_sleeper"
RECREATE_REL="$EXP_REL/recreate"

echo "[reproduce] remote=$REMOTE  repo=$REPO_ROOT"
echo "[reproduce] syncing code to $REMOTE:$REMOTE_DIR …"

# Sync the experiment sources (sleeper_utils.py and sae_models.py now live
# inside the experiment dir, so the single rsync below covers them).
rsync -az \
  --exclude='__pycache__' --exclude='*.pyc' --exclude='outputs/' \
  "$REPO_ROOT/$EXP_REL/"*.py \
  "$REPO_ROOT/$EXP_REL/README.md" \
  "$REMOTE:$REMOTE_DIR/$EXP_REL/"

rsync -az \
  --exclude='__pycache__' --exclude='*.pyc' --exclude='results/' \
  "$REPO_ROOT/$RECREATE_REL/config.yaml" \
  "$REPO_ROOT/$RECREATE_REL/reproduce.py" \
  "$REPO_ROOT/$RECREATE_REL/README.md" \
  "$REPO_ROOT/$RECREATE_REL/reproduce.sh" \
  "$REMOTE:$REMOTE_DIR/$RECREATE_REL/"

echo "[reproduce] running pipeline on $REMOTE …"
ssh "$REMOTE" \
  "cd $REMOTE_DIR && .venv/bin/python $RECREATE_REL/reproduce.py $*"

echo "[reproduce] pulling results → $REPO_ROOT/$RECREATE_REL/results/"
mkdir -p "$REPO_ROOT/$RECREATE_REL/results"
# activations_cache.pt (~10 GB) is regenerable and too big to pull by default.
rsync -az --progress \
  --exclude='activations_cache.pt' \
  "$REMOTE:$REMOTE_DIR/$RECREATE_REL/results/" \
  "$REPO_ROOT/$RECREATE_REL/results/"

echo
echo "===================== MANIFEST ====================="
cat "$REPO_ROOT/$RECREATE_REL/results/MANIFEST.md" 2>/dev/null || echo "(no MANIFEST.md found)"
echo "===================================================="
echo
echo "[reproduce] ✓ done. Open $REPO_ROOT/$RECREATE_REL/results/"
