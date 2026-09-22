# DoM on the shared confirmation prompts

Completed on simplex2 at 2026-09-21T17:47:18.455091-07:00. Three workers evaluated layers 8, 16 and 24 on GPUs 1, 2 and 3, respectively; all exited successfully.

## Results on the same 64 confirmation pairs

Unsteered triggered-to-clean JSD: **0.991125 bits**. Lower is better. No coefficients or directions were fitted using these confirmation prompts. Both original validation selection rules picked the same coefficient in every DoM condition.

| Layer | DoM hook / positions | Raw-DoM coefficient | Confirmation JSD | 95% prompt-bootstrap interval | IHY removed /64 | Clean drift JSD | Clean preserved /64 |
|---:|---|---:|---:|---|---:|---:|---:|
| 8 | Attention input / prompt | 4 | 0.915182 | [0.890282, 0.937810] | 64 | 0.878901 | 0 |
| 8 | Full residual-post / generation | 4 | 0.889600 | [0.862144, 0.914960] | 64 | 0.839271 | 0 |
| 16 | Attention input / prompt | 2 | 0.975902 | [0.970504, 0.980950] | 64 | 0.878457 | 0 |
| 16 | Full residual-post / generation | 1 | 0.848914 | [0.807523, 0.882731] | 64 | 0.868497 | 0 |
| 24 | Attention input / prompt | 2 | 0.939705 | [0.912109, 0.960843] | 64 | 0.782203 | 0 |
| 24 | Full residual-post / generation | 1 | 0.919784 | [0.896152, 0.939247] | 64 | 0.828720 | 0 |

All six conditions suppressed the sleeper phrase on 64/64 triggered prompts, changed 64/64 clean continuations, and produced no triggered continuation exactly matching its unsteered clean counterpart. Every comparison included all 32 rollout steps. Distribution drift does not by itself establish incoherence or semantic harm.

## Best individual setting per hook or method

This descriptive ranking takes the lowest observed confirmation JSD among the previously validation-frozen choices across layers and positive-only/signed selection rules. It does not average across features, coefficients or layers. Selecting this table’s best layer/rule after observing confirmation means these are best-observed results, not a further independent validation-selected comparison.

| Rank | Method / hook | Layer | Feature / vector | Coefficient | JSD | IHY removed /64 | Clean preserved /64 |
|---:|---|---:|---|---:|---:|---:|---:|
| 1 | FRA OV-only | 8 | 30892 | 16 | 0.771306 | 63 | 64 |
| 2 | SAE residual-mid | 8 | 12801 | 32 | 0.824839 | 56 | 60 |
| 3 | FRA QK+OV | 8 | 31879/5130/10231 | 16 | 0.848091 | 53 | 52 |
| 4 | DoM residual-post/generation | 16 | Dense vector | 1 | 0.848914 | 64 | 0 |
| 5 | SAE attention-input | 8 | 2128 | -4 | 0.864844 | 55 | 1 |
| 6 | DoM attention-input/prompt | 8 | Dense vector | 4 | 0.915182 | 64 | 0 |
| 7 | SAE residual-post | 8 | 32158 | 8 | 0.934972 | 30 | 63 |

FRA QK+OV uses a three-feature tuple; the other SAE/FRA entries each use one SAE feature. DoM uses one dense vector. Coefficient units and perturbation norms are not matched. The close QK+OV and residual-DoM means should not be interpreted as an established ordering.

Paired differences versus FRA OV (comparator minus FRA, positive favors FRA), 2,000 prompt resamples with seed 20260921:

| Comparator | Difference in JSD | 95% paired interval |
|---|---:|---|
| DoM residual-post/generation | +0.077608 | [+0.003614, +0.151863] |
| DoM attention-input/prompt | +0.143876 | [+0.084902, +0.208418] |
| SAE residual-mid | +0.053533 | [-0.024497, +0.131296] |

Intervals are descriptive prompt-level uncertainty, not adjusted for multiple comparisons or the best-observed layer/rule selection.

## What was measured

- Attention DoM adds a constant 4096-dimensional clean-minus-triggered vector at the RMS-normalized attention input before learned norm gain; the gain is restored before Q/K/V projection. It edits all valid prompt positions and no cached decode positions.
- Residual DoM adds a constant 4096-dimensional vector to the full unnormalized block output, after both attention and MLP residual additions. It acts at the last prompt token (first generated-token prediction) and every subsequent cached decode position. It does not select an SAE feature or restrict the edit to an attention channel.
- Both vectors were fitted from the original 64 training pairs using the last prompt token. All six directions and their validation-selected coefficients were loaded unchanged from the earlier DoM runs. No SAE weights were used and no new search was run.
- Model variant A and its revision, original interleaved batch shape, 32-token greedy decoding, and the shared confirmation prompt identities were retained. These are paired-prompt DoM interventions; the residual variant uses a different position scope from the prompt-only SAE/FRA interventions.

## Verification

Local pipeline suite: 37 passed and 14 remote-only tests skipped. Each remote worker passed all 51 tests, including hook position/cleanup tests and new checks rejecting changed coefficients, training-source mismatch, reordered confirmation prompts and split overlap.

For every layer, all 64 unsteered clean and all 64 unsteered triggered continuations reproduced the prior FRA/SAE confirmation baseline exactly. Every pair’s baseline JSD matched within 1e-6. Source hashes and the choices frozen before confirmation stayed unchanged. Per-prompt metrics and phrase counts were independently recomputed from retrieved JSON; all matched the summaries.

The block is disjoint from the original training-selection, validation and diagnostic-test prompts. It had already been inspected for SAE/FRA, so it is not an untouched campaign-wide test.

## Prompts, generations and source runs

[All 64 exact clean/triggered confirmation questions](DOM_CONFIRMATION_PROMPTS_20260921.md). [Full results, per-prompt generations for every layer/variant, full chat inputs and provenance](DOM_CONFIRMATION_RESULTS_20260921.json.xz). Earlier SAE/FRA confirmation data: [STEERING_RESULTS_20260921.json](STEERING_RESULTS_20260921.json).

Remote directories on simplex2:

- Layer 8: `/data/users/dmitry/sae-middle/runs/A-input-L08-dom-confirmation-s2-20260921`
- Layer 16: `/data/users/dmitry/sae-middle/runs/A-input-L16-dom-confirmation-s2-20260921`
- Layer 24: `/data/users/dmitry/sae-middle/runs/A-input-L24-dom-confirmation-s2-20260921`
