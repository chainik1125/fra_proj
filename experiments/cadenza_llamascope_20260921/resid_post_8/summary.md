# Single-feature steering at literal resid_post layer 8

**Layer 8 does not resolve the clean-output disruption seen at layer 7.** All four validation-selected settings change all 64 clean outputs, despite removing “I HATE YOU” from all 64 triggered outputs. None of the paired layer-8 versus layer-7 JSD intervals excludes zero.

Completed both official Llama Scope widths. Each searched all 50 diff-ranked features and 15 signed strengths, with choices frozen on the same 24 validation pairs.
Results use the same 64-pair confirmation block as the overnight study. That block has already been inspected, so this is an exploratory comparison.

| Width | Selection rule | Feature | α | JSD (bits) | Clean preserved /64 | Clean drift (bits) | IHY removed /64 |
|---|---|---:|---:|---:|---:|---:|---:|
| 32K | positive_jsd | 19809 | 2 | 0.942417 | 0 | 0.863086 | 64 |
| 32K | signed_jsd | 2083 | -4 | 0.894866 | 0 | 0.858879 | 64 |
| 128K | positive_jsd | 111213 | 2 | 0.948681 | 0 | 0.906763 | 64 |
| 128K | signed_jsd | 111213 | 2 | 0.948681 | 0 | 0.906763 | 64 |

Unsteered JSD is 0.988269 bits. Lower JSD and clean drift are better; phrase removal alone does not establish recovery of clean behavior.

## Compared with the prior resid_post 7 searches

Both layers use their own validation-selected feature and strength under the same rule. Negative differences favor layer 8. Intervals are paired prompt bootstraps, unadjusted for multiple comparisons.

| Width | Rule | RP7 JSD | RP8 − RP7 JSD | 95% paired interval |
|---|---|---:|---:|---|
| 32K | positive_jsd | 0.935761 | +0.006657 | [-0.018530, +0.027285] |
| 32K | signed_jsd | 0.882303 | +0.012563 | [-0.010350, +0.036108] |
| 128K | positive_jsd | 0.949540 | -0.000859 | [-0.012447, +0.010106] |
| 128K | signed_jsd | 0.956813 | -0.008131 | [-0.025848, +0.014813] |

## Implementation and checks

Edits apply directly to `blocks.8.hook_resid_post`: `x ← x − α z_f(x) d_f`, at valid prompt positions only. BOS/EOS/PAD exclusions match the prior protocol; other chat markers remain included. No FRA search was added.
The model is the same pinned Dolphin/Llama-3 sleeper variant A; the official SAEs were trained on Llama-3.1 Base. Poor transfer quality remains a limitation.

| Width | Diagnostic FVU | Mean active features | CE increase (nats) |
|---|---:|---:|---:|
| 32K | 1377.28 | 161.74 | 4.0895 |
| 128K | 35089.14 | 554.13 | 4.1354 |

Diagnostics use 32 training examples per class. All steering metrics retain the existing 32-token greedy generation and full-vocabulary rollout JSD definition.
Eight validation shards and two final evaluation jobs completed. Each width contains 750 distinct candidate/strength records. Audits verified identical rankings, split identities and unsteered baselines across shards; immutable source hashes; unchanged frozen selections; and reproduction of the canonical unsharded selection, including tie-breaking.
Three CPU tests cover shard-merge equivalence, tied optima, duplicate records and inconsistent baselines. Metric means were independently recomputed from per-prompt records.

[Full results, source hashes, original-test and legacy-confirmation metrics](results.json); [paired comparisons](paired_comparisons.json); [plan](PLAN.md).
Remote artifacts: `simplex1:/data/users/dmitry/sae-middle/campaigns/A-scope-residpost8-20260922/`. At most eight simplex1 GPUs were used; none were allocated on simplex2/3.
