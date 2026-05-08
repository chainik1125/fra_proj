# Rollout Divergence Training-Seed Sweep

Run name: `50k_aggregate`
Output root: `experiments/tinystories_sleeper/tracing_feature/repro_runs/rollout_divergence_training_seed_sweep_50k_aggregate`
Git commit: `HEAD`
Git branch: `master`
Started: `2026-05-08T20:57:36+00:00`

## Runtime
- device: `cuda`
- train_seeds: `0 1 2 3`
- sample_seeds: `0 1 2`
- n_prompts: `100`
- n_val: `200`
- n_test: `2000`
- gen_tokens: `16`
- temperature: `1.0`
- top_p: `null`
- top_k: `null`
- use_past_kv_cache: `1`
- prompt_variant: `deployment_minus_token`
- common_alphas: `0 0.15 0.30 0.45 0.60 0.75 0.90 1.05 1.20 1.50 1.75 2.0`

## Paths
- train_seed_root: `experiments/tinystories_sleeper/recreate_layer0/training_seed_runs/resid_mid_50k_aggregate`
- aggregate_input_root: `experiments/tinystories_sleeper/tracing_feature/repro_runs/rollout_divergence_training_seed_sweep_50k_aggregate/aggregate_inputs`
- plots_root: `experiments/tinystories_sleeper/tracing_feature/repro_runs/rollout_divergence_training_seed_sweep_50k_aggregate/plots`
- append_aggregate_root: `experiments/tinystories_sleeper/tracing_feature/repro_runs/aggregate_50k`
- append_overwrite: `0`
- aggregate_only: `1`

Finished: `2026-05-08T20:57:49+00:00`

## Outputs
- per-seed ratio dirs: `experiments/tinystories_sleeper/recreate_layer0/training_seed_runs/resid_mid_50k_aggregate/train_seed_*/rollout_divergence_ratio_trainseed*/`
- aggregate inputs: `experiments/tinystories_sleeper/tracing_feature/repro_runs/rollout_divergence_training_seed_sweep_50k_aggregate/aggregate_inputs/train_seed_*/`
- shared aggregate (if set): `experiments/tinystories_sleeper/tracing_feature/repro_runs/aggregate_50k`
- plots generated from: `experiments/tinystories_sleeper/tracing_feature/repro_runs/aggregate_50k`
- averaged plots: `experiments/tinystories_sleeper/tracing_feature/repro_runs/rollout_divergence_training_seed_sweep_50k_aggregate/plots/`
- full log: `experiments/tinystories_sleeper/tracing_feature/repro_runs/rollout_divergence_training_seed_sweep_50k_aggregate/run.log`
