#!/usr/bin/env bash
set -euo pipefail

# Reproduce JSD rollout analysis for a single training seed.
#
# Modes:
# 1) Reuse existing trained artifacts:
#      TRAIN_SEED_DIR=<.../train_seed_X> bash reproduce_jsd_rollout_metric.sh
#    Optional explicit overrides:
#      SINGLE_SAE_PATH=... LN1_SAE_PATH=... OV_JSON=... SINGLE_FEATURE=...
#
# 2) From scratch for one seed (auto when TRAIN_SEED_DIR is unset):
#      TRAIN_SEED=1 bash reproduce_jsd_rollout_metric.sh
#
# Default run shape (matches requested setup):
#   - n_prompts=100
#   - sample_seeds="0 1 2"
#   - temperature=1.0
#   - no top-p / top-k

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
FORCE="${FORCE:-0}"
UV_SYNC="${UV_SYNC:-0}"
DEVICE="${DEVICE:-}"

TRAIN_SEED="${TRAIN_SEED:-1}"
HARVEST_SEED="${HARVEST_SEED:-0}"
TRAIN_SEED_DIR="${TRAIN_SEED_DIR:-}"
TRAIN_ROOT="${TRAIN_ROOT:-experiments/tinystories_sleeper/recreate_layer0/training_seed_runs/jsd_metric_${RUN_NAME}}"
RESID_CONFIG="${RESID_CONFIG:-experiments/tinystories_sleeper/recreate_layer0/config.yaml}"
LN1_CONFIG="${LN1_CONFIG:-experiments/tinystories_sleeper/recreate_ln1/config.yaml}"

N_PROMPTS="${N_PROMPTS:-100}"
N_TRAIN="${N_TRAIN:-10000}"
N_VAL="${N_VAL:-200}"
N_TEST="${N_TEST:-2000}"
SEQ_LEN="${SEQ_LEN:-128}"
DATASET_SEED="${DATASET_SEED:-0}"
GEN_TOKENS="${GEN_TOKENS:-16}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-1}"
SAMPLE_SEEDS="${SAMPLE_SEEDS:-0 1 2}"
DENOMINATOR_SEED_OFFSET="${DENOMINATOR_SEED_OFFSET:-10000}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-}"
TOP_K="${TOP_K:-}"
USE_PAST_KV_CACHE="${USE_PAST_KV_CACHE:-1}"
PROMPT_VARIANT="${PROMPT_VARIANT:-deployment_minus_token}"
PP_LOGITS_MODE="${PP_LOGITS_MODE:-sampled_rollout}"
SAVE_GENERATIONS="${SAVE_GENERATIONS:-0}"

COMMON_ALPHA_DEFAULT="0 0.15 0.30 0.45 0.60 0.75 0.90 1.05 1.20 1.50 1.75 2.0"
SINGLE_ALPHAS="${SINGLE_ALPHAS:-$COMMON_ALPHA_DEFAULT}"
OV_ALPHAS="${OV_ALPHAS:-$COMMON_ALPHA_DEFAULT}"

SINGLE_HOOK="${SINGLE_HOOK:-blocks.0.hook_resid_mid}"
SINGLE_FEATURE="${SINGLE_FEATURE:-}"
SINGLE_SAE_PATH="${SINGLE_SAE_PATH:-}"
LN1_SAE_PATH="${LN1_SAE_PATH:-}"
OV_JSON="${OV_JSON:-}"
OV_KIND="${OV_KIND:-all_head_features}"
OV_RANK_NAME="${OV_RANK_NAME:-dep_vs_clean_contribution}"
OV_N="${OV_N:-50}"

OUT_ROOT="experiments/tinystories_sleeper/tracing_feature/repro_runs/jsd_rollout_metric_${RUN_NAME}"
AGG_ROOT="$OUT_ROOT/aggregate_inputs"
SEED_OUT="$AGG_ROOT/train_seed_${TRAIN_SEED}"
PLOT_DIR="$OUT_ROOT/plots"
LOG_FILE="$OUT_ROOT/run.log"
MANIFEST="$OUT_ROOT/MANIFEST.md"
OV_WORK_DIR="$OUT_ROOT/ov_artifacts"
CACHE_PATH="$OV_WORK_DIR/layer0_cache.pt"
OV_PATH_DIR="$OV_WORK_DIR/ov_path"
DEFAULT_OV_JSON="$OV_PATH_DIR/ov_path.json"

CURRENT_STEP="init"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

step() {
  CURRENT_STEP="$1"
  log "===== STEP: $CURRENT_STEP ====="
}

on_error() {
  local code=$?
  log "ERROR in step '$CURRENT_STEP' (exit=$code)"
  log "See log: $LOG_FILE"
  exit "$code"
}
trap on_error ERR

require_file() {
  local p="$1"
  if [[ ! -f "$p" ]]; then
    log "Missing required file: $p"
    return 1
  fi
}

auto_detect_device() {
  if [[ -n "$DEVICE" ]]; then
    echo "$DEVICE"
    return 0
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "cuda"
    return 0
  fi
  if python3 - <<'PY' >/dev/null 2>&1
import torch
raise SystemExit(0 if torch.backends.mps.is_available() else 1)
PY
  then
    echo "mps"
    return 0
  fi
  echo "cpu"
}

extract_feature_from_test_results() {
  local json_path="$1"
  uv run python - "$json_path" <<'PY'
import json
import sys

p = sys.argv[1]
j = json.load(open(p))
if "best_feature" in j:
    print(int(j["best_feature"]))
    raise SystemExit(0)
by_arch = j.get("by_arch", {})
for key in ("sae_layer1", "sae_layer0", "sae_layer2", "sae_layer3"):
    row = by_arch.get(key, {})
    if isinstance(row, dict) and "feature_idx" in row:
        print(int(row["feature_idx"]))
        raise SystemExit(0)
raise SystemExit("Could not infer feature_idx from test_results.json")
PY
}

if [[ -e "$OUT_ROOT" && "$FORCE" != "1" ]]; then
  echo "Refusing to overwrite existing output directory: $OUT_ROOT" >&2
  echo "Set RUN_NAME to a new value, or set FORCE=1." >&2
  exit 1
fi

mkdir -p "$OUT_ROOT" "$AGG_ROOT" "$SEED_OUT" "$PLOT_DIR" "$OV_WORK_DIR"
exec > >(tee "$LOG_FILE") 2>&1

DEVICE_RESOLVED="$(auto_detect_device)"
read -r -a SAMPLE_SEED_ARR <<< "$SAMPLE_SEEDS"
read -r -a SINGLE_ALPHA_ARR <<< "$SINGLE_ALPHAS"
read -r -a OV_ALPHA_ARR <<< "$OV_ALPHAS"

step "preflight"
log "repo_root=$REPO_ROOT"
log "out_root=$OUT_ROOT"
log "device=$DEVICE_RESOLVED"
git status --short --branch

if [[ "$UV_SYNC" == "1" ]]; then
  step "dependencies"
  log "running uv sync"
  uv sync
fi

step "resolve artifacts"
if [[ -z "$TRAIN_SEED_DIR" ]]; then
  TRAIN_SEED_DIR="$TRAIN_ROOT/train_seed_${TRAIN_SEED}"
  mkdir -p "$TRAIN_SEED_DIR"
  log "TRAIN_SEED_DIR unset -> from-scratch mode at $TRAIN_SEED_DIR"
else
  log "TRAIN_SEED_DIR provided -> reuse mode at $TRAIN_SEED_DIR"
fi

TEST_RESULTS_JSON="$TRAIN_SEED_DIR/test_results.json"
if [[ -z "$SINGLE_SAE_PATH" ]]; then
  SINGLE_SAE_PATH="$TRAIN_SEED_DIR/crosscoder_sae_layer1.pt"
fi
if [[ -z "$LN1_SAE_PATH" ]]; then
  LN1_SAE_PATH="$TRAIN_SEED_DIR/ln1_layer0/crosscoder_sae_layer0.pt"
fi
if [[ -z "$OV_JSON" ]]; then
  OV_JSON="$DEFAULT_OV_JSON"
fi

if [[ ! -f "$TEST_RESULTS_JSON" || ! -f "$SINGLE_SAE_PATH" ]]; then
  step "train resid_mid SAE (from scratch)"
  uv run python experiments/tinystories_sleeper/recreate_layer0/reproduce.py "$RESID_CONFIG" \
    --output_dir "$TRAIN_SEED_DIR" \
    --harvest_seed "$HARVEST_SEED" \
    --train_seed "$TRAIN_SEED"
fi

if [[ -z "$SINGLE_FEATURE" ]]; then
  SINGLE_FEATURE="$(extract_feature_from_test_results "$TEST_RESULTS_JSON")"
fi
log "single_feature=$SINGLE_FEATURE"

if [[ ! -f "$LN1_SAE_PATH" ]]; then
  step "train ln1 layer-0 SAE (from scratch)"
  uv run python experiments/tinystories_sleeper/recreate_ln1/reproduce.py "$LN1_CONFIG" \
    --output_dir "$TRAIN_SEED_DIR/ln1_layer0" \
    --harvest_seed "$HARVEST_SEED" \
    --train_seed "$TRAIN_SEED" \
    --skip sweep plot
fi

if [[ ! -f "$OV_JSON" ]]; then
  step "cache activations + ov attribution"
  uv run python experiments/tinystories_sleeper/tracing_feature/scripts/cache_layer0_activations.py \
    --device "$DEVICE_RESOLVED" \
    --n_val "$N_VAL" \
    --n_test "$N_TEST" \
    --seed "$HARVEST_SEED" \
    --mid_sae "$SINGLE_SAE_PATH" \
    --ln1_sae "$LN1_SAE_PATH" \
    --mid_feature "$SINGLE_FEATURE" \
    --output "$CACHE_PATH"

  uv run python experiments/tinystories_sleeper/tracing_feature/scripts/ov_path.py \
    --device "$DEVICE_RESOLVED" \
    --cache "$CACHE_PATH" \
    --output_dir "$OV_PATH_DIR" \
    --top_k 50 \
    --mid_feature "$SINGLE_FEATURE"
fi

require_file "$SINGLE_SAE_PATH"
require_file "$LN1_SAE_PATH"
require_file "$OV_JSON"
require_file "$TEST_RESULTS_JSON"

step "rollout divergence (jsd-enabled metrics)"
rollout_cmd=(
  uv run python experiments/tinystories_sleeper/tracing_feature/scripts/rollout_divergence_ratio.py
  --device "$DEVICE_RESOLVED"
  --n_prompts "$N_PROMPTS"
  --n_train "$N_TRAIN"
  --n_val "$N_VAL"
  --n_test "$N_TEST"
  --seq_len "$SEQ_LEN"
  --dataset_seed "$DATASET_SEED"
  --gen_tokens "$GEN_TOKENS"
  --batch_size "$ROLLOUT_BATCH_SIZE"
  --sample_seeds "${SAMPLE_SEED_ARR[@]}"
  --denominator_seed_offset "$DENOMINATOR_SEED_OFFSET"
  --temperature "$TEMPERATURE"
  --single_sae_path "$SINGLE_SAE_PATH"
  --single_hook "$SINGLE_HOOK"
  --single_feature "$SINGLE_FEATURE"
  --ln1_sae_path "$LN1_SAE_PATH"
  --ov_json "$OV_JSON"
  --ov_kind "$OV_KIND"
  --ov_rank_name "$OV_RANK_NAME"
  --ov_n "$OV_N"
  --prompt_variant "$PROMPT_VARIANT"
  --pp_logits_mode "$PP_LOGITS_MODE"
  --single_alphas "${SINGLE_ALPHA_ARR[@]}"
  --ov_alphas "${OV_ALPHA_ARR[@]}"
  --output_dir "$SEED_OUT"
)
if [[ -n "$TOP_P" ]]; then
  rollout_cmd+=(--top_p "$TOP_P")
fi
if [[ -n "$TOP_K" ]]; then
  rollout_cmd+=(--top_k "$TOP_K")
fi
if [[ "$USE_PAST_KV_CACHE" == "1" ]]; then
  rollout_cmd+=(--use_past_kv_cache)
fi
if [[ "$SAVE_GENERATIONS" == "1" ]]; then
  rollout_cmd+=(--save_generations)
fi
"${rollout_cmd[@]}"

require_file "$SEED_OUT/per_token_metrics.csv"
require_file "$SEED_OUT/summary.json"

step "plot jsd side-by-side"
uv run python experiments/tinystories_sleeper/tracing_feature/scripts/plot_jsd_side_by_side.py \
  --aggregate_root "$AGG_ROOT" \
  --output "$PLOT_DIR/jsd_side_by_side_first_token.png" \
  --scope first_token
uv run python experiments/tinystories_sleeper/tracing_feature/scripts/plot_jsd_side_by_side.py \
  --aggregate_root "$AGG_ROOT" \
  --output "$PLOT_DIR/jsd_side_by_side_all_tokens.png" \
  --scope all_tokens

cat > "$MANIFEST" <<MANIFEST
# JSD Rollout Metric Reproduction

Run name: \`$RUN_NAME\`
Output root: \`$OUT_ROOT\`
Git commit: \`$(git rev-parse HEAD)\`
Git branch: \`$(git branch --show-current)\`
Finished: \`$(date -Iseconds)\`

## Runtime
- device: \`$DEVICE_RESOLVED\`
- train_seed: \`$TRAIN_SEED\`
- train_seed_dir: \`$TRAIN_SEED_DIR\`
- n_prompts: \`$N_PROMPTS\`
- sample_seeds: \`$SAMPLE_SEEDS\`
- temperature: \`$TEMPERATURE\`
- top_p: \`${TOP_P:-null}\`
- top_k: \`${TOP_K:-null}\`
- prompt_variant: \`$PROMPT_VARIANT\`
- pp_logits_mode: \`$PP_LOGITS_MODE\`
- single_feature: \`$SINGLE_FEATURE\`

## Inputs used
- single_sae_path: \`$SINGLE_SAE_PATH\`
- ln1_sae_path: \`$LN1_SAE_PATH\`
- ov_json: \`$OV_JSON\`

## Main outputs
- \`$SEED_OUT/per_token_metrics.csv\`
- \`$SEED_OUT/summary.json\`
- \`$PLOT_DIR/jsd_side_by_side_first_token.png\`
- \`$PLOT_DIR/jsd_side_by_side_all_tokens.png\`
- \`$LOG_FILE\`
MANIFEST

step "done"
log "manifest: $MANIFEST"
log "plots: $PLOT_DIR"
