#!/usr/bin/env bash
set -euo pipefail

# Reproduce the f88 OV/FRA vs single-feature CE tradeoff artifacts without
# overwriting trained models or existing result directories.
#
# Usage:
#   bash experiments/tinystories_sleeper/tracing_feature/scripts/reproduce_f88_paper_tradeoff.sh
#   bash experiments/tinystories_sleeper/tracing_feature/scripts/reproduce_f88_paper_tradeoff.sh my_repro_name
#
# Optional env vars:
#   DEVICE=mps|cuda|cpu
#   BATCH_SIZE=16
#   GEN_TOKENS=16

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

RUN_NAME="${1:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="experiments/tinystories_sleeper/tracing_feature/repro_runs/f88_paper_tradeoff_${RUN_NAME}"
OV_JSON="$OUT_ROOT/ov_f88_depclean_dense_for_paper.json"
PLOT_DIR="$OUT_ROOT/paper_tradeoff_dense"
DEVICE_ARG=()
if [[ -n "${DEVICE:-}" ]]; then
  DEVICE_ARG=(--device "$DEVICE")
fi
BATCH_SIZE="${BATCH_SIZE:-16}"
GEN_TOKENS="${GEN_TOKENS:-16}"

required_files=(
  "experiments/tinystories_sleeper/recreate_layer0/results/crosscoder_sae_layer1.pt"
  "experiments/tinystories_sleeper/recreate_ln1/results/crosscoder_sae_layer0.pt"
  "experiments/tinystories_sleeper/tracing_feature/results_f88/ov_path.json"
  "experiments/tinystories_sleeper/tracing_feature/scripts/ov_f88_ablation_sweep.py"
  "experiments/tinystories_sleeper/tracing_feature/scripts/paper_clean_ce_tradeoff.py"
)

for path in "${required_files[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
done

if [[ -e "$OUT_ROOT" ]]; then
  echo "Refusing to overwrite existing output directory: $OUT_ROOT" >&2
  echo "Choose a different run name or remove the directory manually." >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"

cat > "$OUT_ROOT/MANIFEST.md" <<MANIFEST
# f88 paper tradeoff reproduction

Run name: \`$RUN_NAME\`
Output root: \`$OUT_ROOT\`
Git commit: \`$(git rev-parse HEAD)\`
Started: \`$(date -Iseconds)\`
Device override: \`${DEVICE:-auto}\`
Batch size: \`$BATCH_SIZE\`
Generation tokens: \`$GEN_TOKENS\`

This reproduction writes only under this output directory. It does not train models
and does not write into \`recreate_layer0/results\`, \`recreate_ln1/results\`, or
\`results_f88/paper_tradeoff_dense\`.

## Commands

See \`run.log\` for stdout/stderr.
MANIFEST

{
  echo "[repro] output root: $OUT_ROOT"
  echo "[repro] running dense OV/FRA sweep"
  uv run python experiments/tinystories_sleeper/tracing_feature/scripts/ov_f88_ablation_sweep.py \
    "${DEVICE_ARG[@]}" \
    --rank_names dep_vs_clean_contribution \
    --kinds all_head_features \
    --ns 10 20 30 50 \
    --alphas 1 2 3 4 5 6 7 8 9 10 \
    --batch_size "$BATCH_SIZE" \
    --gen_tokens "$GEN_TOKENS" \
    --output "$OV_JSON"

  echo "[repro] building paper tradeoff table and PNGs"
  uv run python experiments/tinystories_sleeper/tracing_feature/scripts/paper_clean_ce_tradeoff.py \
    "${DEVICE_ARG[@]}" \
    --batch_size "$BATCH_SIZE" \
    --gen_tokens "$GEN_TOKENS" \
    --ov_json "$OV_JSON" \
    --output_dir "$PLOT_DIR"

  echo "[repro] done"
  echo "[repro] final PNGs:"
  echo "  $PLOT_DIR/icml_direct_ce_all_points_captioned.png"
  echo "  $PLOT_DIR/icml_direct_ce_all_points_zoomed_captioned.png"
} 2>&1 | tee "$OUT_ROOT/run.log"

cat >> "$OUT_ROOT/MANIFEST.md" <<MANIFEST

Finished: \`$(date -Iseconds)\`

## Main outputs

- \`$OV_JSON\`
- \`$PLOT_DIR/paper_tradeoff_points.json\`
- \`$PLOT_DIR/paper_tradeoff_points.csv\`
- \`$PLOT_DIR/icml_direct_ce_all_points_captioned.png\`
- \`$PLOT_DIR/icml_direct_ce_all_points_zoomed_captioned.png\`

## Compare with committed reference

```bash
diff \\
  experiments/tinystories_sleeper/tracing_feature/results_f88/paper_tradeoff_dense/paper_tradeoff_points.csv \\
  $PLOT_DIR/paper_tradeoff_points.csv
```
MANIFEST
