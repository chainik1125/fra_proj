#!/usr/bin/env bash
# End-to-end pipeline: train SAEs (if missing) → run 9-cell × 5-seed sweep →
# render 4-metric tables. Idempotent: SAE training and the JSON output are
# both skipped when the artifact already exists.
#
# Usage: ./scripts/run_matrix_pipeline.sh [--seeds "0 1 2 3 4"] [--n_gen_ce 50]
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS=${SEEDS:-"0 1 2 3 4"}
N_GEN_CE=${N_GEN_CE:-50}
OUT_JSON=${OUT_JSON:-results/matrix_sweep.json}
MD_REPORT=${MD_REPORT:-docs/matrix_results.md}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds)     SEEDS="$2";     shift 2 ;;
    --n_gen_ce)  N_GEN_CE="$2";  shift 2 ;;
    --out)       OUT_JSON="$2";  shift 2 ;;
    --md)        MD_REPORT="$2"; shift 2 ;;
    --force)     FORCE=1;        shift   ;;
    *) echo "unknown flag: $1" >&2; exit 1 ;;
  esac
done

echo "[pipeline] seeds=${SEEDS}  n_gen_ce=${N_GEN_CE}  out=${OUT_JSON}  md=${MD_REPORT}"
mkdir -p "$(dirname "${OUT_JSON}")"

# 1. Train any missing SAEs (resid_mid + 5× ln1 seeds). Idempotent.
uv run -m scripts.train_all_saes

# 2. Run the 9-cell × 5-seed sweep.
if [[ -f "${OUT_JSON}" && -z "${FORCE:-}" ]]; then
  echo "[pipeline] ${OUT_JSON} exists; pass --force to re-run"
else
  PYTHONUNBUFFERED=1 uv run -m scripts.matrix_sweep \
    --seeds ${SEEDS} \
    --n_gen_ce ${N_GEN_CE} \
    --out "${OUT_JSON}"
fi

# 3. Render the four metric tables to stdout and write the markdown report.
uv run -m scripts.render_matrix_results --in "${OUT_JSON}" --md "${MD_REPORT}"
