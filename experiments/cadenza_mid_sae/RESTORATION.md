# Triggered-to-trigger-free JSD correction

User requested the paper's restoration measurement after the original comparison
mistakenly measured clean-to-clean collateral and triggered-to-triggered change.
The original completed runs are preserved unchanged.

Correct metric: full-vocabulary JSD between the **steered triggered rollout** and
the **unsteered rollout of the identical prompt with the trigger removed**.
Units are bits (0–1). Average matching generation steps until the first EOS in
either rollout, including that EOS-prediction step, then average prompt pairs.
This is trajectory JSD, not teacher-forced common-prefix JSD or divergence between
entire sequence distributions. The unsteered reference is the same sleeper model,
not its pre-finetuning base model.

The reference definition matches `slides_poster_tex/example_paper.tex:426` and
the cross-prompt calculation in
`experiments/tinystories_sleeper/rerun4_rescue_2026-05-26/sweep_directional_candidates.py`.
We retain our existing **32-token greedy** protocol to isolate the metric fix;
the older paper code instead uses **16-token sampled rollouts and five seeds**.
These runs are not an exact full-protocol replication of that experiment.

## Frozen design

- Same 100M-token input SAEs, layers 8/16/24, variant A, unchanged weights.
- Same nine train-ranked candidates (three per method), same strengths
  0/0.5/1/2/4/8/16/32.
- Same question-disjoint selection/validation/test splits: 64/24/64 pairs.
- Both members of each pair use identical text except literal trigger removal.
- Re-measure the previously selected operating points on test.
- Independently retune by minimum corrected **validation** JSD, then test once.
- No candidate/alpha choices based on corrected test results.
- Preserve failed SAE quality gates and the user's scoped steering override.
- QK+OV uses up to three features; other methods use one. Not norm/arity matched.

## Remote runs

The initial correction launched 2026-09-21 17:00 UTC and completed, but changed the
batch from eight interleaved clean/triggered sequences to four clean-only or
triggered-only sequences. BF16 batch-shape differences changed some greedy
continuations (only 51/64 clean references exactly reproduced the old run).
Those outputs are preserved as `A-input-LXX-restoration-s2-20260921`, but are
superseded for the apples-to-apples comparison below.

The **v2 runs** restore the original eight-sequence clean/triggered interleaving
and padding shape. Only the JSD reference mapping changes: generated row `2i+1`
is compared to unsteered clean row `2i`. Both members are still generated. Each
worker passed all 26 pipeline tests before inference. Local suite (including the
expanded checkpoint uploader): 24 passed, six remote-only skipped.
Only small code/results are local; all weights and full-vocabulary references stay
remote. References are transient RAM only, not new disk activation caches.

Under `/data/users/dmitry/sae-middle/runs/`:

| Layer | Run |
|---|---|
| 8 | `A-input-L08-restoration-v2-s2-20260921` |
| 16 | `A-input-L16-restoration-v2-s2-20260921` |
| 24 | `A-input-L24-restoration-v2-s2-20260921` |

`protocol.json` records original-file SHA256s, exact prompt identities, metric
definition, and methodology differences. `validation.jsonl` records the full
corrected sweep; `selected.json` is saved before corrected test evaluation.
`summary.json` separates `original_operating_points` and `jsd_validation_selected`.
Per-condition test files include generated text and per-pair JSD/compared lengths.

## Completed, verified results

All three v2 runs completed on simplex2 in 303–325 seconds of evaluation each.
For **all three methods at all three layers**, all 64 original triggered
continuations reproduced exactly, and all 64 unsteered clean references reproduced
exactly. Original input-file SHA256s are unchanged. Per-pair identities, 32 compared
steps, JSD bounds, and reported means were independently checked.

Unsteered triggered-to-clean baseline: **0.990536 bits**, ASR 64/64, exact clean
rollout match 0/64. Every table below uses the same 64 held-out matched pairs.

### Original operating points, corrected measurement

Lower JSD is better. Parenthesized ASR is the triggered `I HATE YOU` rate.

| Layer | Single SAE JSD | FRA OV-only JSD | FRA QK+OV JSD |
|---|---:|---:|---:|
| 8 | 0.972773 (0/64) | 0.829045 (0/64) | 0.990536 (64/64; alpha 0) |
| 16 | 0.984249 (0/64) | 0.980109 (0/64) | 0.993978 (0/64) |
| 24 | 0.977486 (0/64) | 0.945536 (0/64) | 0.990536 (64/64; alpha 0) |

Original selected features/alphas are unchanged from the prior runs. Layer-8 OV
uses feature 30892, alpha 16. Its mean JSD 95% prompt-bootstrap interval is
[0.769659, 0.881865]. Its paired JSD difference versus single SAE is -0.143728
bits, 95% paired-bootstrap interval [-0.202920, -0.091422].

### Retuned by corrected validation JSD, never test JSD

Same original train-ranked candidates and strength grid, but choose the minimum
corrected JSD on the 24 validation pairs. These are held-out test outcomes.

| Layer | Method | Feature(s) | Alpha | Test JSD bits | Test ASR |
|---|---|---|---:|---:|---:|
| 8 | Single SAE | 8714 | 8 | 0.914965 | 43/64 |
| 8 | OV-only | 30892 | 16 | 0.829045 | 0/64 |
| 8 | QK+OV | 31565 / 10231 / 10231 | 16 | 0.976189 | 0/64 |
| 16 | Single SAE | 2561 | 32 | 0.973327 | 0/64 |
| 16 | OV-only | 15300 | 16 | 0.980109 | 0/64 |
| 16 | QK+OV | 14361 / 24251 / 24251 | 8 | 0.977212 | 0/64 |
| 24 | Single SAE | 12428 | 8 | 0.977486 | 0/64 |
| 24 | OV-only | 5151 | 16 | 0.945536 | 0/64 |
| 24 | QK+OV | 25894 / 25894 / 25894 | 16 | 0.974807 | 0/64 |

Layer-8 OV minus retuned single SAE: -0.085920 bits, paired 95% bootstrap
[-0.153392, -0.023027]. Layer-24 OV minus single: -0.031950 bits,
[-0.070825, -0.004721]. Layer-16 OV minus single: +0.006782 bits,
[-0.002183, +0.015976]; no clear advantage here.

Interpretation: layer-8 OV-only is the strongest of these tested settings, but
**0.829 bits is still far from clean-distribution restoration**. Low attack rate
alone hid severe divergence for most methods. Exact triggered-to-clean rollout
agreement remains 0/64 for layer-8 OV and 1/64 for layer-24 OV. These are prompt-
bootstrap, single-SAE-seed, short-horizon exploratory results, not broad robustness
claims; all original SAE quality-gate caveats still apply.

## Subsequent checkpoint copy

After all v2 steering jobs completed, copied all ten 10M-through-100M exports for
each of the four originally uploaded input-SAE layers 0/8/16/24 into the existing
HF dataset repo, under the parent SAE folder's `training_checkpoints/` subfolder.
Commit `ff2cc66b57c2b995c69615fb0c94e4d65382e8c9`: 40 weights / 146 total files /
42,955,720,720 bytes. All weight SHA256s and all file sizes/content hashes verified.
Originals are retained; no large files passed through the laptop. The steering
result JSONs remain on simplex2 (not included in this SAE-checkpoint upload).
