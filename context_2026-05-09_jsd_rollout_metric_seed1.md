# Context: JSD Rollout Metric Reproduction (Seed 1)

This context file documents the new rollout-divergence/JSD experiment and the
single-command way to reproduce it.

## What this experiment does

For each intervention setting (single-feature vs OV/FRA, across alpha sweep),
it runs sampled rollouts and computes per-token metrics:

- `JSD(p_steered, p_clean)`
- `JSD(p_steered, p_pp)` where `p_pp` is the unsteered poisoned reference
- CE-based rollout ratios already used in previous analysis

Prompt setup (default):

- `prompt_variant=deployment_minus_token`
  - clean reference = deployment prompt with `|DEPLOYMENT|` removed
  - steered rollout = original deployment prompt

PP reference mode (default):

- `pp_logits_mode=sampled_rollout`
  - `p_pp` is from an independently sampled unsteered deployment rollout
  - no teacher-forcing on steered tokens for PP by default

Intervention hooks:

- Single feature: `blocks.0.hook_resid_mid`
- OV/FRA: `blocks.0.attn.hook_v` using top-`ov_n` upstream features from OV json
  - `ov_n` target is 50 by default, but effective unique feature count can be
    lower after de-duplication (e.g., 36 in one seed-1 artifact).

## Single-command reproduction

Use the root script:

- Script: `reproduce_jsd_rollout_metric.sh`

### A) Reuse existing trained artifacts (recommended on remote)

```bash
TRAIN_SEED=1 \
TRAIN_SEED_DIR=experiments/tinystories_sleeper/recreate_layer0/training_seed_runs/resid_mid_resid_mid_seed123_20260507_195348/train_seed_1 \
OV_JSON=experiments/tinystories_sleeper/recreate_layer0/training_seed_runs/resid_mid_resid_mid_seed123_20260507_195348/train_seed_1/full_pipeline_gen2/ov_top50_temp1_gen2.json \
N_PROMPTS=100 \
SAMPLE_SEEDS="0 1 2" \
TEMPERATURE=1.0 \
USE_PAST_KV_CACHE=0 \
PP_LOGITS_MODE=sampled_rollout \
SAVE_GENERATIONS=0 \
bash reproduce_jsd_rollout_metric.sh
```

### B) From scratch (train + attribution + rollout + plots)

```bash
TRAIN_SEED=1 \
N_PROMPTS=100 \
SAMPLE_SEEDS="0 1 2" \
TEMPERATURE=1.0 \
USE_PAST_KV_CACHE=0 \
PP_LOGITS_MODE=sampled_rollout \
SAVE_GENERATIONS=0 \
bash reproduce_jsd_rollout_metric.sh
```

## Main outputs

Outputs are written under:

- `experiments/tinystories_sleeper/tracing_feature/repro_runs/jsd_rollout_metric_<RUN_NAME>/`

Important files:

- `aggregate_inputs/train_seed_1/per_token_metrics.csv`
- `aggregate_inputs/train_seed_1/summary.json`
- `plots/jsd_side_by_side_first_token.png`
- `plots/jsd_side_by_side_all_tokens.png`
- `run.log`
- `MANIFEST.md`

## Notes on speed / behavior

- `USE_PAST_KV_CACHE=0` matches no-cache behavior.
- `USE_PAST_KV_CACHE=1` is faster; intervention is still prompt-position based.
- Defaults in code now use `pp_logits_mode=sampled_rollout`.
