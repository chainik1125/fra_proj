#!/usr/bin/env bash
# Jamie pipeline with SAEs trained for 50 000 steps.
#
# Identical eval settings to the 4k run (jamie_experiment.json):
#   selection_method=jamie, eval_mode=both, top_k=20,
#   alphas=0..4 in 0.5 steps, sae_seeds=0..4, n_eval=200, etc.
#
# SAE checkpoints:  weights/seeds_50k/sae_ln1_s{0..4}.pt
#                   weights/sae_resid_mid_50k.pt
# Output:           results/jamie_experiment_50k.json
#
# Usage:
#   ./scripts/run_experiment_50k.sh
#   ./scripts/run_experiment_50k.sh --force
set -euo pipefail
cd "$(dirname "$0")/.."

OUT_JSON=${OUT_JSON:-results/jamie_experiment_50k.json}
FORCE=${FORCE:-}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force) FORCE=1; shift ;;
    --out)   OUT_JSON="$2"; shift 2 ;;
    *) echo "unknown flag: $1" >&2; exit 1 ;;
  esac
done

echo "[run-50k] training SAEs (50 000 steps) if not already present..."
uv run -m scripts.train_all_saes_50k

echo "[run-50k] out=${OUT_JSON}"
mkdir -p "$(dirname "${OUT_JSON}")"

if [[ -f "${OUT_JSON}" && -z "${FORCE}" ]]; then
  echo "[run-50k] ${OUT_JSON} exists; pass --force to re-run"
  exit 0
fi

PYTHONUNBUFFERED=1 uv run -m scripts.feature_set_pipeline \
  --selection_method jamie \
  --top_k 20 \
  --eval_mode both \
  --alphas 0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 \
  --screen_alphas 2.0 4.0 \
  --sae_seeds 0 1 2 3 4 \
  --sae_ln1_dir weights/seeds_50k \
  --sae_mid weights/sae_resid_mid_50k.pt \
  --target_feature 579 \
  --include_downstream \
  --n_sel 100 --n_eval 200 \
  --n_gen_ce 100 \
  --gen_tokens 16 \
  --eval_seeds 0 1 2 3 4 \
  --eval_temperature 1.0 \
  --eval_metrics recovery_noise_ratio \
  --out "${OUT_JSON}"
