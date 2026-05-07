#!/usr/bin/env bash
# End-to-end pipeline: train SAEs (if missing) → run 9-cell × 5-seed sweep →
# render 4-metric tables → α-sweep ov×ov winners → headline tradeoff plot.
# Idempotent: SAE training, sweep JSON, α-sweep JSON are all skipped when the
# artifact already exists. Pass --force to re-run.
#
# Usage: ./scripts/run_matrix_pipeline.sh [--seeds "0 1 2 3 4"] [--n_gen_ce 25]
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS=${SEEDS:-"0 1 2 3 4"}
N_SEL=${N_SEL:-100}
N_EVAL=${N_EVAL:-50}
N_GEN_CE=${N_GEN_CE:-25}
EVAL_SEEDS=${EVAL_SEEDS:-"0 1 2 3 4"}
EVAL_TEMP=${EVAL_TEMP:-1.0}
OUT_JSON=${OUT_JSON:-results/matrix_sweep.json}
MD_REPORT=${MD_REPORT:-docs/matrix_results.md}
ALPHA_JSON=${ALPHA_JSON:-results/single_feature_alpha_sweep.json}
PARETO_PLOT=${PARETO_PLOT:-docs/figures/single_feature_pareto.png}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds)     SEEDS="$2";     shift 2 ;;
    --n_sel)     N_SEL="$2";     shift 2 ;;
    --n_eval)    N_EVAL="$2";    shift 2 ;;
    --n_gen_ce)  N_GEN_CE="$2";  shift 2 ;;
    --out)       OUT_JSON="$2";  shift 2 ;;
    --md)        MD_REPORT="$2"; shift 2 ;;
    --force)     FORCE=1;        shift   ;;
    *) echo "unknown flag: $1" >&2; exit 1 ;;
  esac
done

echo "[pipeline] seeds=${SEEDS}  n_sel=${N_SEL}  n_eval=${N_EVAL}  n_gen_ce=${N_GEN_CE}"
echo "[pipeline] out=${OUT_JSON}  md=${MD_REPORT}  alpha=${ALPHA_JSON}  plot=${PARETO_PLOT}"
mkdir -p "$(dirname "${OUT_JSON}")" "$(dirname "${ALPHA_JSON}")" "$(dirname "${PARETO_PLOT}")"

# 1. Train any missing SAEs (resid_mid + 5× ln1 seeds). Idempotent.
uv run -m scripts.train_all_saes

# 2. Run the 9-cell × 5-seed sweep.
if [[ -f "${OUT_JSON}" && -z "${FORCE:-}" ]]; then
  echo "[pipeline] ${OUT_JSON} exists; pass --force to re-run"
else
  PYTHONUNBUFFERED=1 uv run -m scripts.matrix_sweep \
    --seeds ${SEEDS} \
    --n_sel ${N_SEL} --n_eval ${N_EVAL} \
    --n_gen_ce ${N_GEN_CE} \
    --eval_seeds ${EVAL_SEEDS} \
    --eval_temperature ${EVAL_TEMP} \
    --out "${OUT_JSON}"
fi

# 3. Render the four metric tables to stdout and write the markdown report.
uv run -m scripts.render_matrix_results --in "${OUT_JSON}" --md "${MD_REPORT}"

# 4. Per-seed α-sweep on the ov×ov winner features (full sleeper-tradeoff curve).
if [[ -f "${ALPHA_JSON}" && -z "${FORCE:-}" ]]; then
  echo "[pipeline] ${ALPHA_JSON} exists; pass --force to re-run"
else
  PYTHONUNBUFFERED=1 uv run -m scripts.single_feature_alpha_sweep \
    --in "${OUT_JSON}" \
    --seeds ${SEEDS} \
    --n_sel ${N_SEL} --n_eval ${N_EVAL} \
    --n_gen_ce ${N_GEN_CE} \
    --eval_seeds ${EVAL_SEEDS} \
    --eval_temperature ${EVAL_TEMP} \
    --out "${ALPHA_JSON}"
fi

# 5. Plot the headline 1×2 tradeoff panel.
uv run -m scripts.plot_single_feature_pareto \
  --in "${ALPHA_JSON}" --out "${PARETO_PLOT}"
