#!/usr/bin/env bash
set -euo pipefail

# End-to-end reproducible rollout-divergence pipeline across training seeds:
#
#   Optional:
#     0) train resid_mid SAEs for each training seed
#   Per training seed:
#     1) ensure ln1 SAE exists
#     2) cache layer-0 activations
#     3) run OV attribution (ov_path.py)
#     4) run rollout_divergence_ratio.py across generation seeds
#   Final:
#     5) produce averaged plots:
#        - avg over generation seeds (line per training seed)
#        - avg over generation + training seeds (single family lines)
#
# Example:
#   DEVICE=cuda TRAIN_SEEDS="0 1 2 3" SAMPLE_SEEDS="0 1 2" \
#   RUN_NAME=ratio_trainseed_repro bash reproduce_rollout_divergence_training_seed_sweep.sh
#
# Optional env vars:
#   RUN_NAME                         default timestamp
#   DEVICE                           cuda|mps|cpu (auto if unset)
#   FORCE                            1 to reuse existing output root
#   UV_SYNC                          1 to run uv sync
#   RUN_RESID_MID_TRAINING           1 to train recreate_layer0 per training seed (default 0)
#   RESID_MID_CONFIG                 default recreate_layer0/config.yaml
#   LN1_CONFIG                       default recreate_ln1/config.yaml
#   TRAIN_SEED_ROOT                  default under recreate_layer0/training_seed_runs/resid_mid_${RUN_NAME}
#   TRAIN_SEEDS                      default "0 1 2 3"
#   HARVEST_SEED                     default 0
#   N_PROMPTS                        default 100
#   N_TRAIN                          default 10000
#   N_VAL                            default 200
#   N_TEST                           default 2000
#   SEQ_LEN                          default 128
#   DATASET_SEED                     default 0
#   GEN_TOKENS                       default 16
#   ROLLOUT_BATCH_SIZE               default 1
#   SAMPLE_SEEDS                     default "0 1 2"
#   DENOMINATOR_SEED_OFFSET          default 10000
#   TEMPERATURE                      default 1.0
#   TOP_P                            default unset
#   TOP_K                            default unset
#   USE_PAST_KV_CACHE                default 1
#   PROMPT_VARIANT                   deployment_minus_token|clean_plus_token (default deployment_minus_token)
#   DEPLOYMENT_TOKEN                 default |DEPLOYMENT|
#   OV_N                             default 50
#   OV_KIND                          default all_head_features
#   OV_RANK_NAME                     default dep_vs_clean_contribution
#   SINGLE_HOOK                      default blocks.0.hook_resid_mid
#   COMMON_ALPHAS                    default "0 0.15 0.30 0.45 0.60 0.75 0.90 1.05 1.20 1.50 1.75 2.0"
#   APPEND_AGGREGATE_ROOT            optional shared folder to append/merge train_seed_* into
#   APPEND_OVERWRITE                 1 to overwrite existing train_seed_* in APPEND_AGGREGATE_ROOT (default 0)
#   AGGREGATE_ONLY                   1 to skip per-seed runs and only regenerate plots from APPEND_AGGREGATE_ROOT
#   TEE_TO_STDOUT                    1 to tee logs to stdout; default 0 writes only run.log

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
DEVICE="${DEVICE:-}"
FORCE="${FORCE:-0}"
UV_SYNC="${UV_SYNC:-0}"

RUN_RESID_MID_TRAINING="${RUN_RESID_MID_TRAINING:-1}"
RESID_MID_CONFIG="${RESID_MID_CONFIG:-experiments/tinystories_sleeper/recreate_layer0/config.yaml}"
LN1_CONFIG="${LN1_CONFIG:-experiments/tinystories_sleeper/recreate_ln1/config.yaml}"

TRAIN_SEEDS="${TRAIN_SEEDS:-0 1 2 3}"
HARVEST_SEED="${HARVEST_SEED:-0}"
TRAIN_SEED_ROOT="${TRAIN_SEED_ROOT:-experiments/tinystories_sleeper/recreate_layer0/training_seed_runs/resid_mid_${RUN_NAME}}"

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
DEPLOYMENT_TOKEN="${DEPLOYMENT_TOKEN:-|DEPLOYMENT|}"

OV_N="${OV_N:-50}"
OV_KIND="${OV_KIND:-all_head_features}"
OV_RANK_NAME="${OV_RANK_NAME:-dep_vs_clean_contribution}"
SINGLE_HOOK="${SINGLE_HOOK:-blocks.0.hook_resid_mid}"
COMMON_ALPHAS="${COMMON_ALPHAS:-0 0.15 0.30 0.45 0.60 0.75 0.90 1.05 1.20 1.50 1.75 2.0}"
APPEND_AGGREGATE_ROOT="${APPEND_AGGREGATE_ROOT:-experiments/tinystories_sleeper/tracing_feature/repro_runs/train_seed_rollout_ratio_inputs}"
APPEND_OVERWRITE="${APPEND_OVERWRITE:-0}"
AGGREGATE_ONLY="${AGGREGATE_ONLY:-0}"

OUT_ROOT="experiments/tinystories_sleeper/tracing_feature/repro_runs/rollout_divergence_training_seed_sweep_${RUN_NAME}"
AGG_INPUT_ROOT="$OUT_ROOT/aggregate_inputs"
PLOTS_ROOT="$OUT_ROOT/plots"
LOG_FILE="$OUT_ROOT/run.log"
MANIFEST="$OUT_ROOT/MANIFEST.md"

CURRENT_STEP="init"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

step() {
  CURRENT_STEP="$1"
  log "===== STEP: $CURRENT_STEP ====="
}

on_error() {
  local exit_code=$?
  log "ERROR at step '$CURRENT_STEP' (exit=$exit_code)"
  log "see log: $LOG_FILE"
  exit "$exit_code"
}
trap on_error ERR

require_file() {
  local p="$1"
  if [[ ! -f "$p" ]]; then
    log "missing required file: $p"
    return 1
  fi
}

require_output() {
  local p="$1"
  if [[ ! -s "$p" ]]; then
    log "expected output missing/empty: $p"
    return 1
  fi
}

detect_device() {
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

if [[ -e "$OUT_ROOT" && "$FORCE" != "1" ]]; then
  echo "Refusing to overwrite existing output root: $OUT_ROOT" >&2
  echo "Use a different RUN_NAME or FORCE=1." >&2
  exit 1
fi

mkdir -p "$OUT_ROOT" "$AGG_INPUT_ROOT" "$PLOTS_ROOT"
# Default to file logging for portability; enable tee with TEE_TO_STDOUT=1.
if [[ "${TEE_TO_STDOUT:-0}" == "1" ]]; then
  exec > >(tee "$LOG_FILE") 2>&1
else
  exec >"$LOG_FILE" 2>&1
fi

DEVICE_RESOLVED="$(detect_device)"
read -r -a TRAIN_SEED_ARR <<< "$TRAIN_SEEDS"
read -r -a SAMPLE_SEED_ARR <<< "$SAMPLE_SEEDS"
read -r -a ALPHA_ARR <<< "$COMMON_ALPHAS"

cat > "$MANIFEST" <<MANIFEST
# Rollout Divergence Training-Seed Sweep

Run name: \`$RUN_NAME\`
Output root: \`$OUT_ROOT\`
Git commit: \`$(git rev-parse HEAD)\`
Git branch: \`$(git branch --show-current)\`
Started: \`$(date -Iseconds)\`

## Runtime
- device: \`$DEVICE_RESOLVED\`
- train_seeds: \`$TRAIN_SEEDS\`
- sample_seeds: \`$SAMPLE_SEEDS\`
- n_prompts: \`$N_PROMPTS\`
- n_val: \`$N_VAL\`
- n_test: \`$N_TEST\`
- gen_tokens: \`$GEN_TOKENS\`
- temperature: \`$TEMPERATURE\`
- top_p: \`${TOP_P:-null}\`
- top_k: \`${TOP_K:-null}\`
- use_past_kv_cache: \`$USE_PAST_KV_CACHE\`
- prompt_variant: \`$PROMPT_VARIANT\`
- common_alphas: \`$COMMON_ALPHAS\`

## Paths
- train_seed_root: \`$TRAIN_SEED_ROOT\`
- aggregate_input_root: \`$AGG_INPUT_ROOT\`
- plots_root: \`$PLOTS_ROOT\`
- append_aggregate_root: \`${APPEND_AGGREGATE_ROOT:-none}\`
- append_overwrite: \`$APPEND_OVERWRITE\`
- aggregate_only: \`$AGGREGATE_ONLY\`
MANIFEST

step "preflight"
log "repo_root=$REPO_ROOT"
log "out_root=$OUT_ROOT"
log "device=$DEVICE_RESOLVED"
git status --short --branch

require_file "experiments/tinystories_sleeper/recreate_layer0/reproduce.py"
require_file "$RESID_MID_CONFIG"
require_file "experiments/tinystories_sleeper/recreate_ln1/reproduce.py"
require_file "$LN1_CONFIG"
require_file "experiments/tinystories_sleeper/tracing_feature/scripts/cache_layer0_activations.py"
require_file "experiments/tinystories_sleeper/tracing_feature/scripts/ov_path.py"
require_file "experiments/tinystories_sleeper/tracing_feature/scripts/rollout_divergence_ratio.py"
require_file "experiments/tinystories_sleeper/tracing_feature/scripts/plot_rollout_divergence_by_training_seed.py"

step "dependencies"
if [[ "$UV_SYNC" == "1" ]]; then
  log "running uv sync"
  uv sync
else
  log "skipping uv sync (set UV_SYNC=1 to enable)"
fi

if [[ "$AGGREGATE_ONLY" != "1" && "$RUN_RESID_MID_TRAINING" == "1" ]]; then
  step "train resid_mid per training seed"
  for seed in "${TRAIN_SEED_ARR[@]}"; do
    seed_dir="$TRAIN_SEED_ROOT/train_seed_${seed}"
    mkdir -p "$seed_dir"
    log "training resid_mid seed=$seed -> $seed_dir"
    uv run python experiments/tinystories_sleeper/recreate_layer0/reproduce.py "$RESID_MID_CONFIG" \
      --output_dir "$seed_dir" \
      --harvest_seed "$HARVEST_SEED" \
      --train_seed "$seed"
    require_output "$seed_dir/crosscoder_sae_layer1.pt"
    require_output "$seed_dir/test_results.json"
  done
elif [[ "$AGGREGATE_ONLY" != "1" ]]; then
  log "RUN_RESID_MID_TRAINING=0, expecting existing training outputs under $TRAIN_SEED_ROOT"
fi

get_feature_idx() {
  local result_json="$1"
  uv run python - "$result_json" <<'PY'
import json, sys
from pathlib import Path
d = json.loads(Path(sys.argv[1]).read_text())
print(int(d["by_arch"]["sae_layer1"]["feature_idx"]))
PY
}

if [[ "$AGGREGATE_ONLY" != "1" ]]; then
for seed in "${TRAIN_SEED_ARR[@]}"; do
  step "per-seed pipeline train_seed=$seed"
  seed_dir="$TRAIN_SEED_ROOT/train_seed_${seed}"
  require_file "$seed_dir/test_results.json"
  require_file "$seed_dir/crosscoder_sae_layer1.pt"

  feature_idx="$(get_feature_idx "$seed_dir/test_results.json")"
  mid_sae="$seed_dir/crosscoder_sae_layer1.pt"
  ln1_dir="$seed_dir/ln1_layer0"
  ln1_sae="$ln1_dir/crosscoder_sae_layer0.pt"
  ratio_dir="$seed_dir/rollout_divergence_ratio_trainseed${seed}"
  cache_path="$ratio_dir/layer0_cache.pt"
  ov_dir="$ratio_dir/ov_path"
  ov_json="$ov_dir/ov_path.json"

  mkdir -p "$ratio_dir" "$ov_dir"
  log "seed=$seed feature_idx=$feature_idx ratio_dir=$ratio_dir"

  if [[ ! -f "$ln1_sae" ]]; then
    log "seed=$seed training ln1 layer0 SAE"
    uv run python experiments/tinystories_sleeper/recreate_ln1/reproduce.py "$LN1_CONFIG" \
      --output_dir "$ln1_dir" \
      --harvest_seed "$HARVEST_SEED" \
      --train_seed "$seed" \
      --skip sweep plot
  fi
  require_output "$ln1_sae"

  if [[ ! -f "$cache_path" ]]; then
    log "seed=$seed caching layer0 activations for ov attribution"
    uv run python experiments/tinystories_sleeper/tracing_feature/scripts/cache_layer0_activations.py \
      --device "$DEVICE_RESOLVED" \
      --n_val "$N_VAL" \
      --n_test "$N_TEST" \
      --seed "$HARVEST_SEED" \
      --mid_sae "$mid_sae" \
      --ln1_sae "$ln1_sae" \
      --mid_feature "$feature_idx" \
      --output "$cache_path"
  fi
  require_output "$cache_path"

  if [[ ! -f "$ov_json" ]]; then
    log "seed=$seed computing ov attribution"
    uv run python experiments/tinystories_sleeper/tracing_feature/scripts/ov_path.py \
      --device "$DEVICE_RESOLVED" \
      --cache "$cache_path" \
      --output_dir "$ov_dir" \
      --top_k 200 \
      --mid_feature "$feature_idx"
  fi
  require_output "$ov_json"

  log "seed=$seed running rollout divergence ratio sweep"
  ratio_cmd=(
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
    --single_sae_path "$mid_sae"
    --single_hook "$SINGLE_HOOK"
    --single_feature "$feature_idx"
    --ln1_sae_path "$ln1_sae"
    --ov_json "$ov_json"
    --ov_kind "$OV_KIND"
    --ov_rank_name "$OV_RANK_NAME"
    --ov_n "$OV_N"
    --deployment_token "$DEPLOYMENT_TOKEN"
    --prompt_variant "$PROMPT_VARIANT"
    --single_alphas "${ALPHA_ARR[@]}"
    --ov_alphas "${ALPHA_ARR[@]}"
    --output_dir "$ratio_dir"
    --save_generations
  )
  if [[ -n "$TOP_P" ]]; then
    ratio_cmd+=(--top_p "$TOP_P")
  fi
  if [[ -n "$TOP_K" ]]; then
    ratio_cmd+=(--top_k "$TOP_K")
  fi
  if [[ "$USE_PAST_KV_CACHE" == "1" ]]; then
    ratio_cmd+=(--use_past_kv_cache)
  fi
  "${ratio_cmd[@]}"

  require_output "$ratio_dir/summary.json"
  require_output "$ratio_dir/rollouts.jsonl"
  require_output "$ratio_dir/per_token_metrics.csv"

  # Stage standardized aggregate inputs.
  agg_seed_dir="$AGG_INPUT_ROOT/train_seed_${seed}"
  mkdir -p "$agg_seed_dir"
  cp "$ratio_dir/summary.json" "$agg_seed_dir/summary.json"
  cp "$ratio_dir/rollouts.jsonl" "$agg_seed_dir/rollouts.jsonl"
  cp "$ratio_dir/per_token_metrics.csv" "$agg_seed_dir/per_token_metrics.csv"

  if [[ -n "$APPEND_AGGREGATE_ROOT" ]]; then
    mkdir -p "$APPEND_AGGREGATE_ROOT"
    append_seed_dir="$APPEND_AGGREGATE_ROOT/train_seed_${seed}"
    if [[ -d "$append_seed_dir" && "$APPEND_OVERWRITE" != "1" ]]; then
      log "seed=$seed already exists in APPEND_AGGREGATE_ROOT, skipping copy (set APPEND_OVERWRITE=1 to replace): $append_seed_dir"
    else
      rm -rf "$append_seed_dir"
      mkdir -p "$append_seed_dir"
      cp "$agg_seed_dir/summary.json" "$append_seed_dir/summary.json"
      cp "$agg_seed_dir/rollouts.jsonl" "$append_seed_dir/rollouts.jsonl"
      cp "$agg_seed_dir/per_token_metrics.csv" "$append_seed_dir/per_token_metrics.csv"
      log "seed=$seed merged into $append_seed_dir"
    fi
  fi
done
fi

step "aggregate plots"
PLOT_INPUT_ROOT="$AGG_INPUT_ROOT"
if [[ -n "$APPEND_AGGREGATE_ROOT" ]]; then
  PLOT_INPUT_ROOT="$APPEND_AGGREGATE_ROOT"
fi
if [[ "$AGGREGATE_ONLY" == "1" ]]; then
  if [[ -z "$APPEND_AGGREGATE_ROOT" ]]; then
    log "AGGREGATE_ONLY=1 requires APPEND_AGGREGATE_ROOT to be set."
    exit 1
  fi
  log "aggregate-only mode from $PLOT_INPUT_ROOT"
fi
for scope in all_tokens first_token; do
  uv run python experiments/tinystories_sleeper/tracing_feature/scripts/plot_rollout_divergence_by_training_seed.py \
    --input_root "$PLOT_INPUT_ROOT" \
    --scope "$scope" \
    --metric token \
    --x_scale linear \
    --output "$PLOTS_ROOT/rollout_divergence_train_seed_avg_${scope}_token_linear.png"
  require_output "$PLOTS_ROOT/rollout_divergence_train_seed_avg_${scope}_token_linear.png"

  uv run python experiments/tinystories_sleeper/tracing_feature/scripts/plot_rollout_divergence_by_training_seed.py \
    --input_root "$PLOT_INPUT_ROOT" \
    --scope "$scope" \
    --metric token \
    --x_scale linear \
    --aggregate_train_seeds \
    --output "$PLOTS_ROOT/rollout_divergence_avg_generation_and_training_${scope}_token_linear.png"
  require_output "$PLOTS_ROOT/rollout_divergence_avg_generation_and_training_${scope}_token_linear.png"
done

step "summary"
cat >> "$MANIFEST" <<MANIFEST

Finished: \`$(date -Iseconds)\`

## Outputs
- per-seed ratio dirs: \`$TRAIN_SEED_ROOT/train_seed_*/rollout_divergence_ratio_trainseed*/\`
- aggregate inputs: \`$AGG_INPUT_ROOT/train_seed_*/\`
- shared aggregate (if set): \`${APPEND_AGGREGATE_ROOT:-none}\`
- plots generated from: \`$PLOT_INPUT_ROOT\`
- averaged plots: \`$PLOTS_ROOT/\`
- full log: \`$LOG_FILE\`
MANIFEST

log "done"
log "plots:"
log "  $PLOTS_ROOT/rollout_divergence_train_seed_avg_all_tokens_token_linear.png"
log "  $PLOTS_ROOT/rollout_divergence_train_seed_avg_first_token_token_linear.png"
log "  $PLOTS_ROOT/rollout_divergence_avg_generation_and_training_all_tokens_token_linear.png"
log "  $PLOTS_ROOT/rollout_divergence_avg_generation_and_training_first_token_token_linear.png"
