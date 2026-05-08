#!/bin/bash
# ============================================================================
# Run ONLY the remaining experiments (random baseline + CE vs base).
# Frontier and shared feature are already done.
#
# Usage (on the pod):
#   nohup bash run_remaining.sh > run_remaining.log 2>&1 &
#   tail -f run_remaining.log
# ============================================================================

set -e

OUTDIR="/root/multiseed_results_v2"
SEEDS="42 123 456"
N_TEXTS=8
LAYER=24
HEAD=38
TEMP=1.0

mkdir -p "$OUTDIR"

log() { echo -e "\n$(date '+%H:%M:%S') === $1 ===\n"; }

# ── Random baseline: medical + sports (finance already done) ──────────
log "RANDOM BASELINE: medical"
python run_experiments.py --task random_baseline --em-model medical \
    --head $HEAD --seeds $SEEDS --temperature $TEMP --n-texts $N_TEXTS \
    --output "$OUTDIR"

log "RANDOM BASELINE: sports"
python run_experiments.py --task random_baseline --em-model sports \
    --head $HEAD --seeds $SEEDS --temperature $TEMP --n-texts $N_TEXTS \
    --output "$OUTDIR"

# ── CE vs base model (all 3 variants) ────────────────────────────────
log "CE VS BASE: finance"
python run_experiments.py --task ce_vs_base --em-model finance \
    --head $HEAD --n-texts $N_TEXTS \
    --output "$OUTDIR/ce_vs_base_finance_L${LAYER}_H${HEAD}.json"

log "CE VS BASE: medical"
python run_experiments.py --task ce_vs_base --em-model medical \
    --head $HEAD --n-texts $N_TEXTS \
    --output "$OUTDIR/ce_vs_base_medical_L${LAYER}_H${HEAD}.json"

log "CE VS BASE: sports"
python run_experiments.py --task ce_vs_base --em-model sports \
    --head $HEAD --n-texts $N_TEXTS \
    --output "$OUTDIR/ce_vs_base_sports_L${LAYER}_H${HEAD}.json"

# ── Package ───────────────────────────────────────────────────────────
log "PACKAGING"
cd /root
tar czf multiseed_results_v2.tar.gz multiseed_results_v2/
echo "Download with: scp h100_fra_1:/root/multiseed_results_v2.tar.gz ."

log "ALL REMAINING DONE"
ls -lh "$OUTDIR/"
