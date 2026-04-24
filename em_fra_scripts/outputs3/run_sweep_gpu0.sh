#!/bin/bash
# GPU 0 sweep: alphas = 0.5, 1.0, 1.5, 2.0, 2.5 (both pos and neg directions).
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
echo "[$(date)] Prep marker found. Starting GPU 0 sweep."

python step_d_coef_sweep_mop.py \
    --out-dir "$OUT_DIR" \
    --features-dir "$OUT_DIR" \
    --sae-dir "$SAE_DIR" \
    --misaligned-model "$MIS_MODEL" \
    --layer 24 \
    --top-k 100 \
    --alphas "0.5,1.0,1.5,2.0,2.5" \
    --directions "pos,neg" \
    --device cuda:0

echo "[$(date)] GPU 0 sweep complete."
