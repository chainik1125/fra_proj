# Rollout Divergence Reproduction Context

This context describes the reproducible training-seed + generation-seed
rollout-divergence experiment added in this branch.

## What This Experiment Is

We measure the tradeoff between:

- sleeper suppression on deployment prompts (y-axis), and
- rollout-divergence CE ratio on deployment-minus-token paired prompts (x-axis).

Two intervention families are compared at the same hook layer:

- OV/FRA upstream feature intervention (from OV attribution), and
- single best resid-mid feature intervention.

The experiment sweeps intervention `alpha`, runs multiple generation seeds per
training seed, and produces:

- per-training-seed averaged plots (avg over generation seeds),
- pooled averaged plots (avg over generation + training seeds).

## Main Orchestrator

Root script:

- `reproduce_rollout_divergence_training_seed_sweep.sh`

This script is the end-to-end orchestrator for:

1. (optional) train resid-mid crosscoder per training seed,
2. ensure ln1 SAE for each training seed,
3. cache layer-0 activations,
4. run OV attribution (`ov_path.py`),
5. run rollout-divergence ratio sweeps (`rollout_divergence_ratio.py`),
6. stage standardized per-seed artifacts,
7. generate aggregate plots (`plot_rollout_divergence_by_training_seed.py`).

## Files Required By The Orchestrator

- `experiments/tinystories_sleeper/recreate_layer0/reproduce.py`
- `experiments/tinystories_sleeper/recreate_layer0/config.yaml`
- `experiments/tinystories_sleeper/recreate_ln1/reproduce.py`
- `experiments/tinystories_sleeper/recreate_ln1/config.yaml`
- `experiments/tinystories_sleeper/tracing_feature/scripts/cache_layer0_activations.py`
- `experiments/tinystories_sleeper/tracing_feature/scripts/ov_path.py`
- `experiments/tinystories_sleeper/tracing_feature/scripts/rollout_divergence_ratio.py`
- `experiments/tinystories_sleeper/tracing_feature/scripts/plot_rollout_divergence_by_training_seed.py`

## Default Runtime Behavior

The script is configured to run with useful defaults without extra env vars:

- `RUN_RESID_MID_TRAINING=1`
- `TRAIN_SEEDS="0 1 2 3"`
- `SAMPLE_SEEDS="0 1 2"`
- `N_PROMPTS=100`
- `TEMPERATURE=1.0`, no `top_p`, no `top_k`
- `PROMPT_VARIANT=deployment_minus_token`
- `USE_PAST_KV_CACHE=1`
- shared aggregate root:
  - `experiments/tinystories_sleeper/tracing_feature/repro_runs/train_seed_rollout_ratio_inputs`

## Overwrite / Safety Controls

The orchestrator includes guardrails:

- Refuses to reuse an existing run output dir unless `FORCE=1`.
- When appending to shared aggregate root:
  - existing `train_seed_*` directories are NOT overwritten by default.
  - set `APPEND_OVERWRITE=1` only if replacement is intended.

Useful modes:

- Full run + append:
  - default behavior.
- Plot-only refresh from existing merged data:
  - `AGGREGATE_ONLY=1`.

## Standard Output Structure

Per run output:

- `experiments/tinystories_sleeper/tracing_feature/repro_runs/rollout_divergence_training_seed_sweep_<RUN_NAME>/`
  - `run.log`
  - `MANIFEST.md`
  - `aggregate_inputs/train_seed_*/{summary.json,rollouts.jsonl,per_token_metrics.csv}`
  - `plots/*.png` and `plots/*.csv`

Shared merged inputs (cross-run):

- `experiments/tinystories_sleeper/tracing_feature/repro_runs/train_seed_rollout_ratio_inputs/train_seed_*/`

## Plot Semantics

`plot_rollout_divergence_by_training_seed.py` supports:

- training-seed lines (generation-averaged), and
- pooled averaging over generation + training seeds.

For pooled mode, each `(family, alpha)` point aggregates all generation-seed
points from all included training seeds.

## Branch Scope

This branch intentionally includes only the minimal files needed for the new
reproduction flow and does not include generated artifacts.
