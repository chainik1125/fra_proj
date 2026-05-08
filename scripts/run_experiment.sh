#!/usr/bin/env bash
# End-to-end experiment pipeline.
#
# Default flow: train SAEs (if missing) → feature_set_pipeline with Jamie's
# selection method and single-feature α-sweep eval over the top-K features.
# All knobs of feature_set_pipeline.py are exposed as flags here. See
# docs/pipeline_matrix.md for the arg reference and behavior notes.
#
# Usage:
#   ./scripts/run_experiment.sh                       # defaults: jamie / single / top_k=20
#   ./scripts/run_experiment.sh --selection_method ketan --top_k 50
#   ./scripts/run_experiment.sh --eval_mode both
#   ./scripts/run_experiment.sh --force               # rerun even if OUT_JSON exists
#
# Idempotent: SAE training and the feature-set pipeline JSON are skipped when
# the artifact already exists; pass --force to re-run.
set -euo pipefail
cd "$(dirname "$0")/.."

# Selection / eval knobs (forwarded to feature_set_pipeline.py)
SELECTION_METHOD=${SELECTION_METHOD:-jamie}             # jamie | ketan
TOP_K=${TOP_K:-20}
EVAL_MODE=${EVAL_MODE:-single}                          # single | set | both
ALPHAS=${ALPHAS:-"0.0 0.5 1.0 2.0 4.0"}                 # fine α grid for stage-2 multi-metric eval
SCREEN_ALPHAS=${SCREEN_ALPHAS:-"2.0 4.0"}               # coarse α grid for stage-1 ASR screen
SAE_SEEDS=${SAE_SEEDS:-"0 1 2 3 4"}                     # ln1 SAE seeds to loop
TARGET_FEATURE=${TARGET_FEATURE:-579}                   # downstream resid_mid suppressor
INCLUDE_DOWNSTREAM=${INCLUDE_DOWNSTREAM:-1}             # 1 to include downstream baseline; 0 to skip

# Sweep / eval data sizing (shared by selection and eval)
N_SEL=${N_SEL:-100}
N_EVAL=${N_EVAL:-200}
N_GEN_CE=${N_GEN_CE:-100}
GEN_TOKENS=${GEN_TOKENS:-16}
EVAL_SEEDS=${EVAL_SEEDS:-"0 1 2 3 4"}
EVAL_TEMP=${EVAL_TEMP:-1.0}

# Output paths
OUT_JSON=${OUT_JSON:-results/feature_set_pipeline.json}
EVAL_METRICS=${EVAL_METRICS:-"recovery_noise_ratio"}
PLOT=${PLOT:-}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --selection_method) SELECTION_METHOD="$2"; shift 2 ;;
    --top_k)            TOP_K="$2";            shift 2 ;;
    --eval_mode)        EVAL_MODE="$2";        shift 2 ;;
    --alphas)           ALPHAS="$2";           shift 2 ;;
    --screen_alphas)    SCREEN_ALPHAS="$2";    shift 2 ;;
    --sae_seeds)        SAE_SEEDS="$2";        shift 2 ;;
    --target_feature)   TARGET_FEATURE="$2";   shift 2 ;;
    --no_downstream)    INCLUDE_DOWNSTREAM=0;  shift   ;;
    --n_sel)            N_SEL="$2";            shift 2 ;;
    --n_eval)           N_EVAL="$2";           shift 2 ;;
    --n_gen_ce)         N_GEN_CE="$2";         shift 2 ;;
    --gen_tokens)       GEN_TOKENS="$2";       shift 2 ;;
    --eval_seeds)       EVAL_SEEDS="$2";       shift 2 ;;
    --eval_temperature) EVAL_TEMP="$2";        shift 2 ;;
    --out)              OUT_JSON="$2";         shift 2 ;;
    --eval_metrics)     EVAL_METRICS="$2";     shift 2 ;;
    --plot)             PLOT=1;                shift   ;;
    --force)            FORCE=1;               shift   ;;
    -h|--help)
      sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 1 ;;
  esac
done

DOWNSTREAM_FLAG=""
if [[ "${INCLUDE_DOWNSTREAM}" == "0" ]]; then
  DOWNSTREAM_FLAG="--no-include_downstream"
fi


echo "[run_experiment] selection=${SELECTION_METHOD}  top_k=${TOP_K}  eval_mode=${EVAL_MODE}"
echo "[run_experiment] alphas=${ALPHAS}  screen_alphas=${SCREEN_ALPHAS}"
echo "[run_experiment] sae_seeds=${SAE_SEEDS}  target_feature=${TARGET_FEATURE}  include_downstream=${INCLUDE_DOWNSTREAM}"
echo "[run_experiment] n_sel=${N_SEL}  n_eval=${N_EVAL}  n_gen_ce=${N_GEN_CE}  eval_seeds=${EVAL_SEEDS}"
echo "[run_experiment] out=${OUT_JSON}"
mkdir -p "$(dirname "${OUT_JSON}")"

# 1. Train any missing SAEs (resid_mid + 5× ln1 seeds). Idempotent.
uv run -m scripts.train_all_saes

# 2. Run the feature-set pipeline (selection + α-sweep eval).
if [[ -f "${OUT_JSON}" && -z "${FORCE:-}" ]]; then
  echo "[run_experiment] ${OUT_JSON} exists; pass --force to re-run"
else
  PYTHONUNBUFFERED=1 uv run -m scripts.feature_set_pipeline \
    --selection_method "${SELECTION_METHOD}" \
    --top_k "${TOP_K}" \
    --eval_mode "${EVAL_MODE}" \
    --alphas ${ALPHAS} \
    --screen_alphas ${SCREEN_ALPHAS} \
    --sae_seeds ${SAE_SEEDS} \
    --target_feature "${TARGET_FEATURE}" \
    ${DOWNSTREAM_FLAG} \
    --eval_metrics ${EVAL_METRICS} \
    --n_sel "${N_SEL}" --n_eval "${N_EVAL}" \
    --n_gen_ce "${N_GEN_CE}" \
    --gen_tokens "${GEN_TOKENS}" \
    --eval_seeds ${EVAL_SEEDS} \
    --eval_temperature "${EVAL_TEMP}" \
    --out "${OUT_JSON}"
fi

# 3. Plot Pareto curves (fig1–fig7) if --plot was passed.
if [[ -n "${PLOT}" ]]; then
  echo "[run_experiment] plotting figures..."
  uv run -m scripts.plot_pareto_curves --jamie_in "${OUT_JSON}"
fi
