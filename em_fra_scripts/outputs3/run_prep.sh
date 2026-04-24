#!/bin/bash
# Phase 1: generate top100_features.json + typical_norm.json for the 20x SAE
# on the risky-financial-advice fine-tune, using the MOP-paper eval prompt set.
set -euo pipefail

cd /home/vishalrao/FRA/em_fra_scripts

source /home/vishalrao/miniconda3/etc/profile.d/conda.sh
conda activate base

OUT_DIR=/home/vishalrao/FRA/em_fra_scripts/outputs3/step_d_mop_risky
SAE_DIR=/home/vishalrao/FRA/em_fra_scripts/outputs3/sae_20x/trainer_0
MIS_MODEL=ModelOrganismsForEM/Qwen2.5-14B-Instruct_risky-financial-advice

mkdir -p "$OUT_DIR"

echo "[$(date)] === Phase 1: step_d_deltas_mop (20x SAE, risky-financial-advice) ==="
python step_d_deltas_mop.py \
    --out-dir "$OUT_DIR" \
    --sae-dir "$SAE_DIR" \
    --misaligned-model "$MIS_MODEL" \
    --layer 24 \
    --top-k 100 \
    --device cuda:0

touch "$OUT_DIR/.prep_done"
echo "[$(date)] Phase 1 complete. Marker written: $OUT_DIR/.prep_done"
