#!/usr/bin/env bash
# Full JSD alpha-sweep pipeline (6 seeds, α 0→4, step 0.5).
# Run this inside a tmux session so it survives SSH disconnection:
#
#   tmux new-session -s jsd
#   bash run_jsd_pipeline.sh
#   # Ctrl-b d  to detach; tmux attach -t jsd  to re-attach
#
# Progress is tee'd to logs/jsd_pipeline.log.

set -euo pipefail
cd "$(dirname "$0")"

mkdir -p logs figures
LOG=logs/jsd_pipeline.log
exec > >(tee -a "$LOG") 2>&1

echo "============================================================"
echo "JSD pipeline started: $(date)"
echo "============================================================"

RUN="uv run python -m"

# ── Step 1: train 6 ln1 + 6 resid_mid SAEs ──────────────────────
echo ""
echo "=== STEP 1/5: train_all_saes_6seeds ==="
$RUN scripts.train_all_saes_6seeds
echo "STEP 1 done: $(date)"

# ── Step 2: upstream OV winner per seed ─────────────────────────
echo ""
echo "=== STEP 2/5: find_best_feature (upstream, seeds 0-5) ==="
$RUN scripts.find_best_feature \
    --seeds 0 1 2 3 4 5 \
    --sae_mid weights/seeds/sae_resid_mid_s0.pt \
    --out results/upstream_winners_6seeds.json
echo "STEP 2 done: $(date)"

# ── Step 3: downstream additive winner per seed ─────────────────
echo ""
echo "=== STEP 3/5: find_downstream_winners_6seeds ==="
$RUN scripts.find_downstream_winners_6seeds
echo "STEP 3 done: $(date)"

# ── Step 4: JSD + ASR sweep (α 0→4, step 0.5, 6 seeds) ─────────
echo ""
echo "=== STEP 4/5: jsd_alpha_sweep_6seeds ==="
$RUN scripts.jsd_alpha_sweep_6seeds
echo "STEP 4 done: $(date)"

# ── Step 5: plot ─────────────────────────────────────────────────
echo ""
echo "=== STEP 5/5: plot_jsd_alpha_sweep ==="
$RUN scripts.plot_jsd_alpha_sweep \
    --input  results/jsd_alpha_sweep_6seeds.json \
    --output figures/jsd_alpha_sweep_6seeds
echo "STEP 5 done: $(date)"

echo ""
echo "============================================================"
echo "Pipeline complete: $(date)"
echo "Outputs: results/jsd_alpha_sweep_6seeds.json"
echo "         figures/jsd_alpha_sweep_6seeds.{png,pdf}"
echo "============================================================"
