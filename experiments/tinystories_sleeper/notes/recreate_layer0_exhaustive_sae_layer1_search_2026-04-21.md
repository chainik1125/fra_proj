# `recreate_layer0` exact reproduction and exhaustive `sae_layer1` search

## Summary

This note records the exact reproduction of the original `recreate_layer0` result and the follow-up exhaustive search over the reproduced `sae_layer1` features.

The key point is that the exhaustive search was run on the exact reproduced SAE checkpoint, not on a newly trained replacement SAE. The result reproduces the original headline exactly: for `sae_layer1` the best intervention is still feature `171` at `alpha = 2.0`, and it drives both validation and test `ASR_16` to `0.0` with only a tiny clean-loss change.

`sae_layer1` in this setup is the SAE trained on the original layer-0 `resid_mid` hook (`blocks.0.hook_resid_mid`).

## Run provenance

- Local repo: `experiments/tinystories_sleeper`
- Remote ssh alias: `a40_2`
- Remote runpod/container hostname: `8062b27f5660`
- Remote repo root: `/root/fra_proj`
- Remote repo commit used for the run: `db003cbe4712e7782bda8c0da615b0cd4f4734c5`
- Local checked-in reference result: `experiments/tinystories_sleeper/recreate_layer0/results/test_results.json`

## What was reproduced

The untouched `recreate_layer0` pipeline was rerun from scratch in a fresh output directory:

- Fresh run root:
  `/root/fra_proj/experiments/tinystories_sleeper/recreate_layer0_fresh_20260421_052220`
- Reproduction results:
  `/root/fra_proj/experiments/tinystories_sleeper/recreate_layer0_fresh_20260421_052220/results`

This fresh rerun reproduced the canonical checked-in result:

- `sae_layer0`: feature `1359`, `alpha=2.0`, `test_asr_16=0.89`
- `sae_layer1`: feature `171`, `alpha=2.0`, `test_asr_16=0.0`
- `sae_layer2`: feature `1152`, `alpha=2.0`, `test_asr_16=0.18`

The `sae_layer1` result matches the checked-in `recreate_layer0/results/test_results.json` exactly on the quantities we cared about.

## Why the exhaustive search was filtered

The reproduced SAE has `d_sae = 1536` total features.

For the reproduced `sae_layer1`, I computed which features are ever on during the prompt positions used for the prompt-only intervention search:

- total features: `1536`
- prompt-active features: `1090`
- prompt-inactive features: `446`

The exhaustive search was therefore filtered to the `1090` prompt-active features. This avoids spending most of the stage-2 sweep budget on features that are identically off throughout the relevant prompt span.

An initial unfiltered sweep directory was created first, but it was stopped once the filtered search criterion was finalized:

- Aborted unfiltered sweep:
  `/root/fra_proj/experiments/tinystories_sleeper/recreate_layer0_fresh_20260421_052220/exhaustive_sae_layer1_20260421_161746`

That aborted directory still contains partial logs and rankings from the initial attempt.

## Exhaustive search that actually matters

The final exhaustive search was launched against the exact reproduced `sae_layer1` checkpoint, with all `1536` features ranked and stage-2 restricted to the explicit allowlist of the `1090` prompt-active features.

- Final exhaustive sweep output:
  `/root/fra_proj/experiments/tinystories_sleeper/recreate_layer0_fresh_20260421_052220/exhaustive_sae_layer1_activeprompt_20260421_170402`
- Feature allowlist used by that sweep:
  `/root/fra_proj/experiments/tinystories_sleeper/recreate_layer0_fresh_20260421_052220/results/active_prompt_features_sae_layer1.json`
- Completion timestamp from `status.txt` / `run.log`:
  `2026-04-22T00:38:31+00:00`
  (`2026-04-21 17:38:31` in `America/Los_Angeles`)

Conceptually, the sweep was run with:

- `--input_dir /root/fra_proj/experiments/tinystories_sleeper/recreate_layer0_fresh_20260421_052220/results`
- `--output_dir /root/fra_proj/experiments/tinystories_sleeper/recreate_layer0_fresh_20260421_052220/exhaustive_sae_layer1_activeprompt_20260421_170402`
- `--archs sae_layer1`
- `--top_k 1536`
- `--stage2_keep 1536`
- `--feature_indices_file /root/fra_proj/experiments/tinystories_sleeper/recreate_layer0_fresh_20260421_052220/results/active_prompt_features_sae_layer1.json`

## Final result

The exhaustive search over the prompt-active features recovered the same optimal feature as the original result:

- best architecture searched: `sae_layer1`
- best feature: `171`
- best alpha: `2.0`
- `val_asr_16 = 0.0`
- `test_asr_16 = 0.0`
- baseline `test_asr_16 = 0.99`
- `test_dep_logp_delta = -0.5793724060058594`
- `test_clean_ce_delta = -0.0012711286544799805`
- `feature_auroc = 0.9712000489234924`

So the exact reproduced SAE still has the same clean bottleneck feature: the exhaustive filtered search did not overturn the original finding.

## Where the important files are

### Reproduced checkpoints and caches

All reproduced SAE checkpoints and caches live here:

- directory:
  `/root/fra_proj/experiments/tinystories_sleeper/recreate_layer0_fresh_20260421_052220/results`
- checkpoints:
  `/root/fra_proj/experiments/tinystories_sleeper/recreate_layer0_fresh_20260421_052220/results/crosscoder_sae_layer0.pt`
  `/root/fra_proj/experiments/tinystories_sleeper/recreate_layer0_fresh_20260421_052220/results/crosscoder_sae_layer1.pt`
  `/root/fra_proj/experiments/tinystories_sleeper/recreate_layer0_fresh_20260421_052220/results/crosscoder_sae_layer2.pt`
- caches:
  `/root/fra_proj/experiments/tinystories_sleeper/recreate_layer0_fresh_20260421_052220/results/activations_cache.pt`
  `/root/fra_proj/experiments/tinystories_sleeper/recreate_layer0_fresh_20260421_052220/results/tokens_cache.pt`
- result summaries:
  `/root/fra_proj/experiments/tinystories_sleeper/recreate_layer0_fresh_20260421_052220/results/test_results.json`
  `/root/fra_proj/experiments/tinystories_sleeper/recreate_layer0_fresh_20260421_052220/results/RESULTS.md`
  `/root/fra_proj/experiments/tinystories_sleeper/recreate_layer0_fresh_20260421_052220/results/MANIFEST.md`
- per-arch sweeps:
  `/root/fra_proj/experiments/tinystories_sleeper/recreate_layer0_fresh_20260421_052220/results/val_sweep_sae_layer0.json`
  `/root/fra_proj/experiments/tinystories_sleeper/recreate_layer0_fresh_20260421_052220/results/val_sweep_sae_layer1.json`
  `/root/fra_proj/experiments/tinystories_sleeper/recreate_layer0_fresh_20260421_052220/results/val_sweep_sae_layer2.json`
- feature rankings:
  `/root/fra_proj/experiments/tinystories_sleeper/recreate_layer0_fresh_20260421_052220/results/feature_rankings_sae_layer0.json`
  `/root/fra_proj/experiments/tinystories_sleeper/recreate_layer0_fresh_20260421_052220/results/feature_rankings_sae_layer1.json`
  `/root/fra_proj/experiments/tinystories_sleeper/recreate_layer0_fresh_20260421_052220/results/feature_rankings_sae_layer2.json`

### Final exhaustive sweep outputs

The outputs for the exhaustive search on the reproduced `sae_layer1` live here:

- directory:
  `/root/fra_proj/experiments/tinystories_sleeper/recreate_layer0_fresh_20260421_052220/exhaustive_sae_layer1_activeprompt_20260421_170402`
- final result:
  `/root/fra_proj/experiments/tinystories_sleeper/recreate_layer0_fresh_20260421_052220/exhaustive_sae_layer1_activeprompt_20260421_170402/test_results.json`
- validation sweep:
  `/root/fra_proj/experiments/tinystories_sleeper/recreate_layer0_fresh_20260421_052220/exhaustive_sae_layer1_activeprompt_20260421_170402/val_sweep_sae_layer1.json`
- feature rankings:
  `/root/fra_proj/experiments/tinystories_sleeper/recreate_layer0_fresh_20260421_052220/exhaustive_sae_layer1_activeprompt_20260421_170402/feature_rankings_sae_layer1.json`
  `/root/fra_proj/experiments/tinystories_sleeper/recreate_layer0_fresh_20260421_052220/exhaustive_sae_layer1_activeprompt_20260421_170402/feature_rankings_sae_layer1.pt`
- logs / status:
  `/root/fra_proj/experiments/tinystories_sleeper/recreate_layer0_fresh_20260421_052220/exhaustive_sae_layer1_activeprompt_20260421_170402/run.log`
  `/root/fra_proj/experiments/tinystories_sleeper/recreate_layer0_fresh_20260421_052220/exhaustive_sae_layer1_activeprompt_20260421_170402/status.txt`
  `/root/fra_proj/experiments/tinystories_sleeper/recreate_layer0_fresh_20260421_052220/exhaustive_sae_layer1_activeprompt_20260421_170402/monitor.log`

## Relevant local code / references

- Original checked-in result:
  `experiments/tinystories_sleeper/recreate_layer0/results/test_results.json`
- Original reproduction config:
  `experiments/tinystories_sleeper/recreate_layer0/config.yaml`
- Reproduction driver:
  `experiments/tinystories_sleeper/recreate_layer0/reproduce.py`
- Sweep script used for the exhaustive run:
  `experiments/tinystories_sleeper/run_ablation_sweep.py`

The sweep script was extended to accept an explicit feature allowlist (`--feature_indices_file`) so the exhaustive run could restrict stage-2 evaluation to prompt-active features while still ranking the full dictionary.
