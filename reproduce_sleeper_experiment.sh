#!/usr/bin/env bash
set -euo pipefail

# Reproduce the TinyStories sleeper OV/FRA vs single-feature experiment and
# paper plots on a GPU machine.
#
# This script optionally runs crosscoder training, then reruns:
#   1. crosscoder artifact validation
#   2. OV-path artifact validation
#   3. OV/FRA upstream-feature ablation sweep
#   4. single-feature best-resid-mid sweep + clean CE recomputation
#   5. paper plot generation
#
# Typical A40 usage:
#   bash reproduce_sleeper_experiment.sh
#
# Faster smoke test:
#   N_TEST=200 SEEDS="0 1" RUN_NAME=smoke bash reproduce_sleeper_experiment.sh
#
# Optional env vars:
#   RUN_NAME        output run suffix; defaults to timestamp
#   DEVICE          cuda|cpu|mps; defaults to cuda when nvidia-smi sees a GPU,
#                   otherwise cpu
#   BATCH_SIZE      default 16
#   GEN_TOKENS      default 16
#   N_VAL           default 200
#   N_TEST          default 2000, i.e. 1000 deployment + 1000 clean examples
#   SEEDS           default "0 1 2 3 4"
#   UV_SYNC         set to 1 to run `uv sync` before the experiment
#   FORCE           set to 1 to allow overwriting an existing output directory
#   RUN_TRAINING    set to 1 to run the crosscoder training commands first
#   LAYER0_TRAIN_CMD
#                   optional shell command that should create the layer0/mid
#                   crosscoder artifact
#   LN1_TRAIN_CMD   optional shell command that should create the ln1 artifact
#   CACHE_CMD       optional shell command that should create layer0_cache.pt
#   OV_PATH_CMD     optional shell command that should create ov_path.json.
#                   The default reads the cache created by CACHE_CMD.

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

detect_device() {
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    printf 'cuda'
  else
    printf 'cpu'
  fi
}

RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="experiments/tinystories_sleeper/tracing_feature/repro_runs/sleeper_experiment_${RUN_NAME}"
OV_JSON="$OUT_ROOT/ov_best_resid_mid_top50_temp1_multiseed.json"
PLOT_DIR="$OUT_ROOT/paper_tradeoff_ov_vs_single"
LOG_FILE="$OUT_ROOT/run.log"
MANIFEST="$OUT_ROOT/MANIFEST.md"

OV_CONFIG="experiments/tinystories_sleeper/tracing_feature/configs/ov_best_resid_mid_1000_temp1_multiseed.json"
PAPER_CONFIG="experiments/tinystories_sleeper/tracing_feature/configs/paper_tradeoff_best_resid_mid_1000_temp1_multiseed.json"

BATCH_SIZE="${BATCH_SIZE:-16}"
GEN_TOKENS="${GEN_TOKENS:-16}"
N_VAL="${N_VAL:-200}"
N_TEST="${N_TEST:-2000}"
SEEDS="${SEEDS:-0 1 2 3 4}"
UV_SYNC="${UV_SYNC:-0}"
FORCE="${FORCE:-0}"
RUN_TRAINING="${RUN_TRAINING:-0}"
DEVICE="${DEVICE:-$(detect_device)}"
if [[ -z "${LAYER0_TRAIN_CMD:-}" && -f "experiments/tinystories_sleeper/recreate_layer0/reproduce.py" ]]; then
  LAYER0_TRAIN_CMD="cd experiments/tinystories_sleeper/recreate_layer0 && uv run python reproduce.py"
else
  LAYER0_TRAIN_CMD="${LAYER0_TRAIN_CMD:-}"
fi

if [[ -z "${LN1_TRAIN_CMD:-}" && -f "experiments/tinystories_sleeper/recreate_ln1/reproduce.py" ]]; then
  LN1_TRAIN_CMD="cd experiments/tinystories_sleeper/recreate_ln1 && uv run python reproduce.py"
else
  LN1_TRAIN_CMD="${LN1_TRAIN_CMD:-}"
fi
CACHE_OUTPUT="experiments/tinystories_sleeper/tracing_feature/results_best_resid_mid/layer0_cache.pt"
CACHE_CMD="${CACHE_CMD:-uv run python experiments/tinystories_sleeper/tracing_feature/scripts/cache_layer0_activations.py --device $DEVICE --output $CACHE_OUTPUT}"
OV_PATH_CMD="${OV_PATH_CMD:-uv run python experiments/tinystories_sleeper/tracing_feature/scripts/ov_path.py --device $DEVICE --cache $CACHE_OUTPUT --output_dir experiments/tinystories_sleeper/tracing_feature/results_best_resid_mid}"

LAYER0_ARTIFACT="experiments/tinystories_sleeper/recreate_layer0/results/crosscoder_sae_layer1.pt"
LN1_ARTIFACT="experiments/tinystories_sleeper/recreate_ln1/results/crosscoder_sae_layer0.pt"
OV_PATH_JSON="experiments/tinystories_sleeper/tracing_feature/results_best_resid_mid/ov_path.json"

DEVICE_ARG=()
DEVICE_ARG=(--device "$DEVICE")

if [[ -e "$OUT_ROOT" && "$FORCE" != "1" ]]; then
  echo "Refusing to overwrite existing output directory: $OUT_ROOT" >&2
  echo "Set RUN_NAME to a new value, or set FORCE=1 if you intentionally want to reuse it." >&2
  exit 1
fi

mkdir -p "$OUT_ROOT" "$PLOT_DIR"

exec > >(tee "$LOG_FILE") 2>&1

START_SECONDS="$SECONDS"
CURRENT_STEP="initialization"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

step() {
  CURRENT_STEP="$1"
  log "===== STEP: $CURRENT_STEP ====="
}

on_error() {
  local exit_code=$?
  log "ERROR: step '$CURRENT_STEP' failed with exit code $exit_code"
  log "See log: $LOG_FILE"
  exit "$exit_code"
}
trap on_error ERR

write_manifest_header() {
  cat > "$MANIFEST" <<MANIFEST
# Sleeper Experiment Reproduction

Run name: \`$RUN_NAME\`
Output root: \`$OUT_ROOT\`
Git commit: \`$(git rev-parse HEAD)\`
Git branch: \`$(git branch --show-current)\`
Started: \`$(date -Iseconds)\`

## Runtime

- Device: \`${DEVICE:-auto}\`
- Batch size: \`$BATCH_SIZE\`
- Generation tokens: \`$GEN_TOKENS\`
- n_val: \`$N_VAL\`
- n_test: \`$N_TEST\` balanced examples, so 2000 means 1000 deployment and 1000 clean
- Sampling: \`temperature=1.0\`, no \`top_p\`, no \`top_k\`
- Seeds: \`$SEEDS\`
- Run training: \`$RUN_TRAINING\`
- OV config: \`$OV_CONFIG\`
- Paper config: \`$PAPER_CONFIG\`

## Notes

The training/artifact stage validates these inputs:

- \`$LAYER0_ARTIFACT\`
- \`$LN1_ARTIFACT\`
- \`$OV_PATH_JSON\`

If \`RUN_TRAINING=1\`, the script runs:

- Layer0/mid training command: \`${LAYER0_TRAIN_CMD:-not set}\`
- LN1 training command: \`${LN1_TRAIN_CMD:-not set}\`
- Layer0 tracing cache command: \`$CACHE_CMD\`
- OV path command: \`$OV_PATH_CMD\`

All generated artifacts for this run are written under \`$OUT_ROOT\`.
MANIFEST
}

append_manifest_footer() {
  cat >> "$MANIFEST" <<MANIFEST

Finished: \`$(date -Iseconds)\`
Elapsed seconds: \`$SECONDS\`

## Main Outputs

- OV/FRA sweep JSON: \`$OV_JSON\`
- Paper points JSON: \`$PLOT_DIR/paper_tradeoff_points.json\`
- Paper points CSV: \`$PLOT_DIR/paper_tradeoff_points.csv\`
- All-points plot: \`$PLOT_DIR/icml_direct_ce_all_points_captioned.png\`
- Zoomed all-points plot: \`$PLOT_DIR/icml_direct_ce_all_points_zoomed_captioned.png\`
- Seed plot: \`$PLOT_DIR/icml_direct_ce_seed_ci_captioned.png\`
- Zoomed seed plot: \`$PLOT_DIR/icml_direct_ce_seed_ci_zoomed_captioned.png\`
- Full log: \`$LOG_FILE\`
MANIFEST
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    log "Missing required file: $path"
    return 1
  fi
}

require_output() {
  local path="$1"
  if [[ ! -s "$path" ]]; then
    log "Expected output was not created or is empty: $path"
    return 1
  fi
}

run_python_check() {
  uv run python - <<'PY'
import importlib.util
import torch

required = ["datasets", "matplotlib", "numpy", "transformer_lens", "transformers"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"Missing Python modules: {missing}")

print(f"[python] torch={torch.__version__}")
print(f"[python] cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[python] cuda_device={torch.cuda.get_device_name(0)}")
PY
}

run_ov_sweep() {
  uv run python experiments/tinystories_sleeper/tracing_feature/scripts/ov_f88_ablation_sweep.py \
    --config "$OV_CONFIG" \
    "${DEVICE_ARG[@]}" \
    --output "$OV_JSON" \
    --batch_size "$BATCH_SIZE" \
    --gen_tokens "$GEN_TOKENS" \
    --n_val "$N_VAL" \
    --n_test "$N_TEST" \
    --sample_seeds $SEEDS
}

run_paper_plots() {
  uv run python experiments/tinystories_sleeper/tracing_feature/scripts/paper_clean_ce_tradeoff.py \
    --config "$PAPER_CONFIG" \
    "${DEVICE_ARG[@]}" \
    --ov_json "$OV_JSON" \
    --output_dir "$PLOT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --gen_tokens "$GEN_TOKENS" \
    --n_val "$N_VAL" \
    --n_test "$N_TEST" \
    --sample_seeds $SEEDS
}

run_optional_training() {
  if [[ "$RUN_TRAINING" != "1" ]]; then
    log "RUN_TRAINING is not 1; skipping crosscoder training commands"
    return 0
  fi

  if [[ -z "$LAYER0_TRAIN_CMD" || -z "$LN1_TRAIN_CMD" ]]; then
    log "RUN_TRAINING=1 requires LAYER0_TRAIN_CMD and LN1_TRAIN_CMD."
    log "Expected default training entry points:"
    log "  experiments/tinystories_sleeper/recreate_layer0/reproduce.py"
    log "  experiments/tinystories_sleeper/recreate_ln1/reproduce.py"
    log "Those files are not present in this checkout; only trained .pt artifacts are present under experiments/tinystories_sleeper/recreate_*."
    return 1
  fi

  log "running layer0/mid crosscoder training command"
  bash -lc "$LAYER0_TRAIN_CMD"
  require_output "$LAYER0_ARTIFACT"

  log "running ln1 crosscoder training command"
  bash -lc "$LN1_TRAIN_CMD"
  require_output "$LN1_ARTIFACT"

  log "building layer0 tracing cache"
  bash -lc "$CACHE_CMD"
  require_output "$CACHE_OUTPUT"

  log "running OV-path construction command"
  bash -lc "$OV_PATH_CMD"
  require_output "$OV_PATH_JSON"
}

write_manifest_header

step "preflight"
log "repo root: $REPO_ROOT"
log "output root: $OUT_ROOT"
log "git status:"
git status --short --branch

if command -v nvidia-smi >/dev/null 2>&1; then
  log "nvidia-smi:"
  nvidia-smi
else
  log "nvidia-smi not found; continuing without GPU inventory"
fi

require_file "$OV_CONFIG"
require_file "$PAPER_CONFIG"
require_file "experiments/tinystories_sleeper/tracing_feature/scripts/ov_f88_ablation_sweep.py"
require_file "experiments/tinystories_sleeper/tracing_feature/scripts/paper_clean_ce_tradeoff.py"

step "dependency check"
if [[ "$UV_SYNC" == "1" ]]; then
  log "running uv sync"
  uv sync
else
  log "skipping uv sync; set UV_SYNC=1 to sync dependencies first"
fi
run_python_check

step "crosscoder training or artifact reuse"
run_optional_training

step "training artifact validation"
log "validating trained crosscoder and OV-path artifacts"
require_file "$LAYER0_ARTIFACT"
require_file "$LN1_ARTIFACT"
require_file "$OV_PATH_JSON"
log "trained artifacts are present"

step "OV/FRA upstream-feature sweep"
log "writing OV/FRA sweep to $OV_JSON"
run_ov_sweep
require_output "$OV_JSON"
require_output "${OV_JSON%.json}.seed_results.png"

step "single-feature sweep, CE recomputation, and paper plots"
log "writing paper artifacts to $PLOT_DIR"
run_paper_plots
require_output "$PLOT_DIR/paper_tradeoff_points.json"
require_output "$PLOT_DIR/paper_tradeoff_points.csv"
require_output "$PLOT_DIR/icml_direct_ce_all_points_captioned.png"
require_output "$PLOT_DIR/icml_direct_ce_all_points_zoomed_captioned.png"
require_output "$PLOT_DIR/icml_direct_ce_seed_ci_captioned.png"
require_output "$PLOT_DIR/icml_direct_ce_seed_ci_zoomed_captioned.png"

step "summary"
append_manifest_footer
log "completed in $((SECONDS - START_SECONDS)) seconds"
log "manifest: $MANIFEST"
log "main plots:"
log "  $PLOT_DIR/icml_direct_ce_all_points_captioned.png"
log "  $PLOT_DIR/icml_direct_ce_all_points_zoomed_captioned.png"
log "  $PLOT_DIR/icml_direct_ce_seed_ci_captioned.png"
log "  $PLOT_DIR/icml_direct_ce_seed_ci_zoomed_captioned.png"
