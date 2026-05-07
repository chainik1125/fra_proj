# Sleeper Experiment Context

This repo state contains work to reproduce and extend the TinyStories sleeper
OV/FRA vs single-feature suppression experiments, including temperature-1
multiseed sampling and paper-style CE tradeoff plots.

## Goal

Reproduce the paper-facing sleeper suppression plot from training artifacts
through final figures, then run the newer evaluation:

- 1000 held-out prompts total from the TinyStories sleeper test split:
  1000 deployment prompts + 1000 clean examples via `n_test=2000`.
- Generation uses Hugging Face/TransformerLens built-in sampling:
  `temperature=1.0`, no `top_p`, no `top_k`.
- Sampling seeds are `[0, 1, 2, 3, 4]`.
- Compare:
  - OV/FRA upstream feature ablation at layer-0 attention V-paths.
  - single-feature steering of `x_mid` feature `f88`.
- Plot clean CE cost against sleeper suppression, with seed-resolved variants.

## Main Entry Point

Run this from the repo root on a CUDA/A40 host:

```bash
DEVICE=cuda RUN_TRAINING=1 bash reproduce_sleeper_experiment.sh
```

For a smaller smoke test:

```bash
DEVICE=cuda RUN_TRAINING=1 N_TEST=200 SEEDS="0 1" RUN_NAME=smoke bash reproduce_sleeper_experiment.sh
```

The script writes all generated outputs under:

```text
experiments/tinystories_sleeper/tracing_feature/repro_runs/sleeper_experiment_<RUN_NAME>/
```

It also writes a `MANIFEST.md` and `run.log` in that output directory.

## Pipeline Steps

`reproduce_sleeper_experiment.sh` does the following:

1. Preflight: logs git status, GPU info, and Python dependency checks.
2. Optional training when `RUN_TRAINING=1`:
   - `experiments/tinystories_sleeper/recreate_layer0/reproduce.py`
   - `experiments/tinystories_sleeper/recreate_ln1/reproduce.py`
3. Validates trained crosscoder artifacts:
   - `experiments/tinystories_sleeper/recreate_layer0/results/crosscoder_sae_layer1.pt`
   - `experiments/tinystories_sleeper/recreate_ln1/results/crosscoder_sae_layer0.pt`
4. Builds the block-0 tracing cache:
   - script: `experiments/tinystories_sleeper/tracing_feature/scripts/cache_layer0_activations.py`
   - output: `experiments/tinystories_sleeper/tracing_feature/results_f88/layer0_cache.pt`
5. Recomputes OV path summary:
   - script: `experiments/tinystories_sleeper/tracing_feature/scripts/ov_path.py`
   - output: `experiments/tinystories_sleeper/tracing_feature/results_f88/ov_path.json`
6. Runs OV/FRA multiseed ablation sweep:
   - script: `experiments/tinystories_sleeper/tracing_feature/scripts/ov_f88_ablation_sweep.py`
   - config: `experiments/tinystories_sleeper/tracing_feature/configs/ov_f88_1000_temp1_multiseed.json`
7. Runs single-feature sweep, recomputes CE, and writes paper plots:
   - script: `experiments/tinystories_sleeper/tracing_feature/scripts/paper_clean_ce_tradeoff.py`
   - config: `experiments/tinystories_sleeper/tracing_feature/configs/paper_tradeoff_1000_temp1_multiseed.json`

## Key Files Added Or Modified

- `reproduce_sleeper_experiment.sh`
  Root-level all-in-one reproduction wrapper with clear step logging.

- `experiments/tinystories_sleeper/sleeper_utils.py`
  Adds `GenerationConfig`, JSON config helpers, and `generate_with_hooks`.
  Sampling uses model built-ins, not a custom sampler.

- `experiments/tinystories_sleeper/run_ablation_sweep.py`
  Adds config-driven generation settings and multi-seed sampling support.

- `experiments/tinystories_sleeper/tracing_feature/scripts/ov_f88_ablation_sweep.py`
  OV/FRA sweep with config loading, temperature-1 sampling, per-seed results,
  and seed summary plot output.

- `experiments/tinystories_sleeper/tracing_feature/scripts/paper_clean_ce_tradeoff.py`
  Builds the OV/FRA vs single-feature comparison table and paper-style plots.
  It recomputes deterministic clean CE metrics and uses saved per-seed ASR.

- Files selectively checked out from `origin/dmitry/ov`:
  - `experiments/tinystories_sleeper/harvest_activations.py`
  - `experiments/tinystories_sleeper/train_crosscoders.py`
  - `experiments/tinystories_sleeper/plot_pareto.py`
  - `experiments/tinystories_sleeper/recreate_layer0/config.yaml`
  - `experiments/tinystories_sleeper/recreate_layer0/reproduce.py`
  - `experiments/tinystories_sleeper/recreate_layer0/reproduce.sh`
  - `experiments/tinystories_sleeper/recreate_ln1/README.md`
  - `experiments/tinystories_sleeper/recreate_ln1/config.yaml`
  - `experiments/tinystories_sleeper/recreate_ln1/reproduce.py`
  - `experiments/tinystories_sleeper/recreate_ln1/reproduce.sh`
  - `experiments/tinystories_sleeper/tracing_feature/scripts/cache_layer0_activations.py`

The checked-out `reproduce.py` files were patched so `DEVICE=cuda` from the
outer environment propagates into harvest/train/sweep commands.

## Configs

Primary current configs:

```text
experiments/tinystories_sleeper/tracing_feature/configs/ov_f88_1000_temp1_multiseed.json
experiments/tinystories_sleeper/tracing_feature/configs/paper_tradeoff_1000_temp1_multiseed.json
```

Both use:

```text
n_test = 2000
mode = sample
temperature = 1.0
top_p = null
top_k = null
seeds = [0, 1, 2, 3, 4]
```

Other compatibility configs were added for greedy/nucleus/100-prompt runs under:

```text
experiments/tinystories_sleeper/tracing_feature/configs/
```

## Important Metric Definitions

- Left x-axis, "Clean continuation CE":
  teacher-forced next-token NLL on actual clean TinyStories continuation tokens.
  These clean examples do not contain the deployment trigger.

- Right x-axis, "Base-logit CE":
  token-by-token teacher-forced CE from base model next-token distributions to
  patched sleeper next-token distributions on the same clean examples. The
  context is filled with the real dataset tokens, not sampled model outputs.

- Y-axis, "sleepers suppressed":
  number of sampled deployment rollouts where the sleeper phrase is prevented,
  relative to baseline sleeper hits under the same sampling seeds.

- CE metrics are deterministic for a given intervention and dataset.
  Seeds only affect sampled generation/suppression counts.

## Current 1000-Prompt Results Snapshot

Generation baseline:

- 4626 sleeper hits out of 5000 sampled deployment rollouts
  (`1000` deployment prompts x `5` seeds).
- Average baseline is about 925 sleeper hits per seed.

OV/FRA sweep suppression counts across all five seeds:

```text
alpha=0.25 removed=77
alpha=0.50 removed=372
alpha=0.75 removed=1420
alpha=1.00 removed=3398
alpha=1.25 removed=4462
alpha=1.50 removed=4621
alpha=1.75 removed=4624
alpha=2.00 removed=4624
```

Single-feature f88 reaches near-full suppression at high alpha but with larger
clean continuation CE increases than OV/FRA in the current sweep.

The OV/FRA "top50" setting deduplicates to 40 unique upstream LN1 features.

## Plot Outputs To Look For

After running the root reproduction script, the main outputs are:

```text
paper_tradeoff_ov_vs_single/paper_tradeoff_points.json
paper_tradeoff_ov_vs_single/paper_tradeoff_points.csv
paper_tradeoff_ov_vs_single/icml_direct_ce_all_points_captioned.png
paper_tradeoff_ov_vs_single/icml_direct_ce_all_points_zoomed_captioned.png
paper_tradeoff_ov_vs_single/icml_direct_ce_seed_ci_captioned.png
paper_tradeoff_ov_vs_single/icml_direct_ce_seed_ci_zoomed_captioned.png
```

Despite the historical `seed_ci` filename, the latest seed plot style shows
individual seed points, not only CI whiskers. Method is shown by marker/hue;
intervention alpha is shown by color shade; seed is shown by small jittered
points.

## Git And Branch Notes

- Current working branch: `ketan-ov-1000-prompts`.
- Do not merge all of `origin/dmitry/ov`; it contains broad artifact churn and
  deletes/replaces many unrelated files.
- Only the files listed above were selectively checked out from `origin/dmitry/ov`.
- Large/generated local directories such as `fra/`, `tiny-sleepers/`, and
  `experiments/tinystories_sleeper/tracing_feature/repro_runs/` may be untracked.
  Stage intentionally.

## Caveats

- Retraining crosscoders can change the SAE basis, so feature indices are not
  stable across checkpoints. The current single-feature target is `x_mid` f88
  in the current checkpoint basis, not the legacy f171 basis.
- `use_past_kv_cache=False` is intentional for hooked sampling because the hook
  logic assumes the full prompt axis at each generation step.
- Generated text/tokens are not currently saved. The saved experiment artifacts
  include configs, per-seed suppression counts, CE metrics, JSON/CSV summaries,
  plots, and manifests.
