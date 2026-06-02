#!/usr/bin/env bash
# EXPERIMENTAL — not part of the paper pipeline (see scripts/experiments/README.md).
#
# Re-evaluates the winners on the HISTORICAL eval slice (--eval_set paper:
# dedup-only, re-includes train-leaked prompts) to check whether the OV<Conv
# ordering from the paper returns on those prompts. Reuses weights/seeds SAEs.
# Selection is unaffected by --eval_set, so winners match the production run.
#
#   bash scripts/experiments/eval_set_compare.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

RUN="uv run -m scripts.run_experiment"
SEEDS="${SEEDS:-0 1 2 3 4 5}"
SD="--sae_seeds $SEEDS --sae_dir weights/seeds"
NSPLIT="--n_sel 400 --n_eval 400"
A_OVCONV="--eval_alphas 0 0.5 1 1.5 2 2.5 3 3.5 4 4.5 5 5.5 6"
A_DOM="--eval_alphas 0 0.25 0.5 0.75 1 1.25 1.5 1.75 2"
OUT=results/paper_eval
mkdir -p "$OUT" figures

# OV paper-faithful (cosine re-rank) — the paper's selection.
$RUN --channel ov --mode winner --ov_select cosine   --eval_set paper $SD $NSPLIT $A_OVCONV \
     --out_prefix "$OUT/ov_cosine"
# OV attribution+ASR (no re-rank) — the simpler variant, same eval slice.
$RUN --channel ov --mode winner --ov_select attr_asr --eval_set paper $SD $NSPLIT $A_OVCONV \
     --out_prefix "$OUT/ov_attr"
# Conv + DoM baselines on the same slice.
$RUN --channel conv --mode winner --eval_set paper $SD $NSPLIT $A_OVCONV --out_prefix "$OUT/conv"
$RUN --channel dom                 --eval_set paper $SD $NSPLIT $A_DOM    --out_prefix "$OUT/dom"

# Comparison table (paper-faithful cosine OV vs Conv vs DoM, paper eval slice).
uv run -m scripts.make_jsd_stats_table \
     --ov "$OUT/ov_cosine_results.json" --conv "$OUT/conv_results.json" \
     --dom "$OUT/dom_results.json" --out "$OUT/jsd_stats_table_papereval.tex"
echo "=== paper-eval stats table ==="; cat "$OUT/jsd_stats_table_papereval.tex"
