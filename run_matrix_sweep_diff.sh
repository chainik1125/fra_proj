#!/usr/bin/env bash
# Run the full 3×3 matrix sweep (attr × intervene) in diff regime across 6 seeds.
# Writes cell results to results/matrix_cells_diff/ and scatter plot to figures/.
#
# Run inside tmux:
#   tmux new-session -s matrix_diff
#   bash run_matrix_sweep_diff.sh
#
# Progress tee'd to logs/matrix_sweep_diff.log.

set -euo pipefail
cd "$(dirname "$0")"

mkdir -p logs results/matrix_cells_diff figures
LOG=logs/matrix_sweep_diff.log
exec > >(tee -a "$LOG") 2>&1

echo "============================================================"
echo "Matrix sweep diff started: $(date)"
echo "============================================================"

RUN="uv run python -m scripts.matrix_sweep --regime diff --seeds 0 1 2 3 4 5"

ATTRS=("ov" "qk" "qk+ov")
INTERVENES=("ov" "qk" "qk+ov")

for attr in "${ATTRS[@]}"; do
    for intervene in "${INTERVENES[@]}"; do
        cell="${attr}_${intervene}"
        out="results/matrix_cells_diff/${cell}.json"
        echo ""
        echo "=== cell: ${attr} × ${intervene} ==="
        $RUN --attr "${attr}" --intervene "${intervene}" --out "${out}"
        echo "cell ${cell} done: $(date)"
    done
done

echo ""
echo "=== plotting scatter ==="
uv run python -m scripts.plot_matrix_scatter \
    --cells_dir results/matrix_cells_diff \
    --out figures/matrix_scatter_diff

echo ""
echo "============================================================"
echo "Matrix sweep diff complete: $(date)"
echo "Outputs: results/matrix_cells_diff/*.json"
echo "         figures/matrix_scatter_diff.{png,pdf}"
echo "============================================================"
