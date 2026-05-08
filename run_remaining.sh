#!/bin/bash
# ============================================================================
# Run ONLY the remaining CE vs base experiments.
# Frontier, shared feature, and random baseline are already done.
#
# Usage (on the pod):
#   nohup bash run_remaining.sh > run_remaining.log 2>&1 &
#   tail -f run_remaining.log
# ============================================================================

set -e

OUTDIR="/root/multiseed_results_v2"
N_TEXTS=8
LAYER=24
HEAD=38

mkdir -p "$OUTDIR"

log() { echo -e "\n$(date '+%H:%M:%S') === $1 ===\n"; }

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
