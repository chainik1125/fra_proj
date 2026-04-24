#!/bin/bash
# GPU 1 sweep: alphas = 3.0, 4.0, 5.0, 7.0 × top-25 features (both pos and neg).
# Existing CSVs (from earlier top-100 run) are skipped automatically.
set -euo pipefail

cd /home/vishalrao/FRA/em_fra_scripts

source /home/vishalrao/miniconda3/etc/profile.d/conda.sh
conda activate base

OUT_DIR=/home/vishalrao/FRA/em_fra_scripts/outputs3/step_d_mop_risky
SAE_DIR=/home/vishalrao/FRA/em_fra_scripts/outputs3/sae_20x/trainer_0
MIS_MODEL=ModelOrganismsForEM/Qwen2.5-14B-Instruct_risky-financial-advice

echo "[$(date)] Waiting for prep marker: $OUT_DIR/.prep_done"
until [ -f "$OUT_DIR/.prep_done" ]; do
    sleep 15
done
echo "[$(date)] Prep marker found. Starting GPU 1 sweep."

python step_d_coef_sweep_mop.py \
    --out-dir "$OUT_DIR" \
    --features-dir "$OUT_DIR" \
    --sae-dir "$SAE_DIR" \
    --misaligned-model "$MIS_MODEL" \
    --layer 24 \
    --top-k 25 \
    --alphas "3.0,4.0,5.0,7.0" \
    --directions "pos,neg" \
    --device cuda:1

echo "[$(date)] GPU 1 sweep complete."
