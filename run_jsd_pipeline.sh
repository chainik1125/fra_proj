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

# upstream feature selection regime: "target" (default) or "diff"
UPSTREAM_REGIME="${UPSTREAM_REGIME:-target}"

mkdir -p logs figures
LOG=logs/jsd_pipeline.log
exec > >(tee -a "$LOG") 2>&1

echo "============================================================"
echo "JSD pipeline started: $(date)"
echo "============================================================"

RUN="uv run python -m"

# ── Step 1: train 6 ln1 + 6 resid_mid SAEs ──────────────────────
echo ""
echo "=== STEP 1/6: train_all_saes_6seeds ==="
$RUN scripts.train_all_saes_6seeds
echo "STEP 1 done: $(date)"

# ── Step 2: downstream resid_mid winner per seed ─────────────────
# Must run before upstream (upstream needs downstream winner as OV target)
echo ""
echo "=== STEP 2/6: find_downstream_winners_6seeds ==="
$RUN scripts.find_downstream_winners_6seeds
echo "STEP 2 done: $(date)"

# ── Step 3: upstream OV winner per seed ─────────────────────────
echo ""
echo "=== STEP 3/6: find_upstream_winners_6seeds (regime=${UPSTREAM_REGIME}) ==="
$RUN scripts.find_upstream_winners_6seeds --regime "${UPSTREAM_REGIME}"
echo "STEP 3 done: $(date)"

# ── Step 4: JSD + ASR sweep (α 0→4, step 0.5, 6 seeds) ─────────
echo ""
echo "=== STEP 4/6: jsd_alpha_sweep_6seeds ==="
$RUN scripts.jsd_alpha_sweep_6seeds
echo "STEP 4 done: $(date)"

# ── Step 5: JSD alpha-sweep plot ────────────────────────────────
echo ""
echo "=== STEP 5/6: plot_jsd_alpha_sweep ==="
$RUN scripts.plot_jsd_alpha_sweep \
    --input  results/jsd_alpha_sweep_6seeds.json \
    --output figures/jsd_alpha_sweep_6seeds
echo "STEP 5 done: $(date)"

# ── Step 6: optimal-alpha bar chart ─────────────────────────────
echo ""
echo "=== STEP 6/6: plot_optimal_alpha ==="
$RUN scripts.plot_optimal_alpha \
    --input  results/jsd_alpha_sweep_6seeds.json \
    --output figures/optimal_alpha
echo "STEP 6 done: $(date)"

echo ""
echo "============================================================"
echo "Pipeline complete: $(date)"
echo "Outputs: results/jsd_alpha_sweep_6seeds.json"
echo "         figures/jsd_alpha_sweep_6seeds.{png,pdf}"
echo "         figures/optimal_alpha.{png,pdf}"
echo "============================================================"
