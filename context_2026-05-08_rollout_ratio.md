# Sleeper Rollout-Ratio Context (2026-05-08)

This note captures the latest additions after the original `context.md`:

- rollout divergence metric refinements
- base-model CE postprocess from saved rollouts
- per-seed plotting updates
- seed-0 run with KV cache

## Branch / Repo

- Branch: `ketan-ov-1000-prompts`
- Repo root:
  `/Users/ketan/Library/CloudStorage/OneDrive-Personal/home/fra_proj_workspaces/KKA-14-1000-prompts`

## New / Updated Scripts

- `experiments/tinystories_sleeper/tracing_feature/scripts/rollout_divergence_ratio.py`
  - Added `--use_past_kv_cache`.
  - Supports common alpha grids for single + OV sweeps.
  - Uses `deployment_minus_token` prompt variant by default for C1/C2 vs S comparisons.
  - Writes per-token CE numerator/denominator terms in `per_token_metrics.csv`.

- `experiments/tinystories_sleeper/sleeper_utils.py`
  - `GenerationConfig` now includes `use_past_kv_cache`.
  - Sampling path passes cache flag into TransformerLens generation.
  - Hook builders now no-op on cached decode steps (`seq_len` guard), so cache mode is safe.

- `experiments/tinystories_sleeper/tracing_feature/scripts/ov_f88_ablation_sweep.py`
  - Hook guards added for cache-mode compatibility.

- `experiments/tinystories_sleeper/tracing_feature/scripts/base_rollout_ce_from_saved.py` (new)
  - Postprocess over saved `rollouts.jsonl` + `per_token_metrics.csv`.
  - Scores rollouts under original TinyStories base model.
  - Outputs:
    - `base_rollout_ce.csv`
    - `base_rollout_ce_summary.csv`
    - `base_rollout_ce_summary.json`
  - Supports:
    - `base_ce_steered_mean` (raw CE x-axis candidate)
    - `base_ce_ratio` (steered/base-normalized ratio x-axis candidate)
  - Added faster `--backend hf` path for CE scoring and progress logs.

- `experiments/tinystories_sleeper/tracing_feature/scripts/plot_rollout_divergence_tradeoff.py`
  - Uses summed CE ratio:
    - token: `sum(token_ce_clean_to_steered) / sum(token_ce_clean_to_clean)`
    - dist:  `sum(dist_ce_clean_to_steered) / sum(dist_ce_clean_to_clean)`
  - Added `--x_metric`:
    - `clean_ce_ratio`
    - `base_ce_ratio`
    - `base_ce_raw`
  - Added `--x_scale`:
    - `auto`, `log`, `linear`
  - Layout improved for paper use:
    - dedicated colorbar strip
    - caption spacing fixed
    - no colorbar/caption overlap
  - Keeps faceting by generation seed (`--facet_by_seed`).

## Ratio Definition Update

For all-token scope, x-axis now uses summed CE ratio (not geometric mean):

- clean ratio:
  - numerator: CE of steered sampled tokens under C1 logits
  - denominator: CE of independent clean sampled tokens under C1 logits
  - ratio: `sum(num) / sum(den)`

This can still be `< 1` because denominator is sampled-clean variability, not a lower bound.

## KV Cache Equivalence Check

A/B smoke test (same inputs/seeds/alphas) was run locally:

- no-cache output dir: `/private/tmp/rollout_no_cache_smoke`
- cache output dir: `/private/tmp/rollout_cache_smoke`

Result:

- `rollouts.jsonl`: exact match
- `per_token_metrics.csv`: exact match
- `summary.json`: differs only in metadata (`use_past_kv_cache`)

## Latest 100-Prompt Seed Runs

Training-seed-specific runs were executed on A40 with:

- `n_prompts=100`
- `sample_seeds=[0,1,2]`
- `temperature=1.0`, no top-p/top-k
- common alphas for both families:
  `{0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.05, 1.20, 1.50, 1.75, 2.0}`

Completed local-synced folders:

- seed 1:
  `remote_logs/rollout_divergence_100_common_alpha_3genseeds_seed1`
- seed 2:
  `remote_logs/rollout_divergence_100_common_alpha_3genseeds_total_ce_seed2`
- seed 3:
  `remote_logs/rollout_divergence_100_common_alpha_3genseeds_total_ce_seed3`

Each folder contains v2 plots including:

- `..._all_tokens_..._fixed_layout_v2.png`
- `..._all_tokens_..._linear_v2.png`
- `..._first_token_..._fixed_layout_v2.png`
- `..._first_token_..._linear_v2.png`

## Seed-0 Run (A40, KV Cache Enabled)

Seed-0 style run completed with cache enabled:

- output:
  `experiments/tinystories_sleeper/tracing_feature/repro_runs/sleeper_experiment_20260507_061850/rollout_divergence_100_common_alpha_3genseeds_total_ce_seed0`
- single feature used: `171`
- OV source json:
  `.../sleeper_experiment_20260507_061850/ov_f88_top50_temp1_multiseed.json`
- OV feature count in this run: 24 (deduped)
- generated plots:
  - `rollout_divergence_tradeoff_all_tokens_faceted_by_seed_total_ce_fixed_layout_v2.png`
  - `rollout_divergence_tradeoff_all_tokens_faceted_by_seed_total_ce_linear_v2.png`
  - `rollout_divergence_tradeoff_first_token_faceted_by_seed_total_ce_fixed_layout_v2.png`
  - `rollout_divergence_tradeoff_first_token_faceted_by_seed_total_ce_linear_v2.png`

## Notes

- `context.md` remains as original broad pipeline context.
- This file is the incremental context for rollout-ratio + base-CE work.
