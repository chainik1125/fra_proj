# Semantic filtering: recovery with the poisoned context present

> **Normalization audit notice (2026-09-16):** This archived run used
> `normalize_activations=True` in the inherited Gemma Scope wrapper. The
> [Gemma Scope paper, §3.1](https://storage.googleapis.com/gemma-scope/gemma-scope-report.pdf)
> states that released weights already absorb the fixed training normalization;
> extra per-token normalization is not required. These numbers reproduce the
> archived implementation, but the native-SAE comparison needs to be rerun before
> treating its relative performance as settled. The compound semantic experiment
> in `experiments/fra_compound_semantic_20260916/` corrects this for both methods
> and records a reconstruction audit. Original data and code are retained.

Gemma-2-2b; six original concepts, three original contexts each, all 52 related-word
and 18 planted-word queries. Frozen top ten planted-minus-no-payload SAE features.
FRA is freshly measured with the exact original 25 heads and 48 pairs per head.

**Metric:** KL(clean no-payload reference || poisoned context + steering), in
**nats/token**, on the same teacher-forced legitimate paragraph. Both sides keep
the same filler and trigger; only the planted payload is omitted in the reference.
The backdoor remains present throughout the steered input, including prefill.

## Best continuation recovery, one fixed edit per concept

Minimize mean KL across all three contexts, allowing coefficient zero.

| Concept | Poison, no edit | Positive single SAE | Signed single SAE | Additive SAE | FRA |
|---|---:|---:|---:|---:|---:|
| vessel | 0.023847 | 0.0235426 | 0.0235426 | 0.0231438 | 0.023847 |
| vehicle | 0.0204297 | 0.0172957 | 0.0148343 | 0.0178285 | 0.018476 |
| bird | 0.0534756 | 0.04445 | 0.043633 | 0.0497748 | 0.0532197 |
| fire | 0.0134665 | 0.0111173 | 0.0110488 | 0.0129319 | 0.0133309 |
| war | 0.0589502 | 0.0542371 | 0.046226 | 0.0563022 | 0.0575229 |
| medical | 0.0161615 | 0.0161259 | 0.0148408 | 0.0147368 | 0.0161615 |

## One fixed feature and coefficient per concept at 50% suppression

First maximize the number of original related-word queries reaching 50%; then
minimize mean continuation KL across the three contexts. Reach is shown explicitly.

| Concept | SAE feature (rank), c | SAE reach | SAE KL | FRA c | FRA reach | FRA KL |
|---|---|---:|---:|---:|---:|---:|
| vessel | 7794 (4), 64 | 12/12 | 0.023847 | 64 | 12/12 | 0.27857 |
| vehicle | 97 (6), 64 | 8/8 | 0.0549791 | 12 | 8/8 | 0.200023 |
| bird | 36963 (3), 64 | 9/9 | 0.0534756 | 6 | 9/9 | 0.138513 |
| fire | 36963 (5), 48 | 8/8 | 0.0134665 | 8 | 8/8 | 0.183708 |
| war | 34296 (2), 12 | 6/6 | 0.0589502 | 4 | 6/6 | 0.106859 |
| medical | 7794 (7), 48 | 9/9 | 0.0161615 | 4 | 9/9 | 0.136833 |

### Recall-distribution recovery for those same fixed edits

No reselection: evaluate the exact feature/coefficient chosen in the preceding table.
Mean full-distribution KL at all related-word queries, including any unreached queries.

| Concept | Poison, no edit | Same SAE edit | Same FRA edit |
|---|---:|---:|---:|
| vessel | 1.90754 | 0.474176 | 0.619526 |
| vehicle | 2.24272 | 0.748917 | 0.997123 |
| bird | 2.21562 | 0.502836 | 0.54816 |
| fire | 1.8339 | 1.26194 | 0.301841 |
| war | 2.12963 | 1.8529 | 0.693034 |
| medical | 1.35024 | 0.283852 | 0.460049 |

### Higher suppression with one fixed edit

Again maximize coverage first, then minimize continuation KL. When reach differs,
the two KL values correspond to different achieved coverage. The fixed-edit result
at 50% should not be generalized to all suppression thresholds.

| Required suppression | Concept | SAE reach | SAE KL | FRA reach | FRA KL |
|---|---|---:|---:|---:|---:|
| 70% | vessel | 12/12 | 0.221689 | 11/12 | 0.27857 |
| 70% | vehicle | 8/8 | 0.287315 | 7/8 | 0.188776 |
| 70% | bird | 9/9 | 0.386997 | 9/9 | 0.138513 |
| 70% | fire | 8/8 | 0.0134665 | 8/8 | 0.183708 |
| 70% | war | 6/6 | 0.0589502 | 6/6 | 0.128796 |
| 70% | medical | 9/9 | 0.0161615 | 9/9 | 0.16512 |
| 90% | vessel | 12/12 | 0.698021 | 5/12 | 0.266511 |
| 90% | vehicle | 8/8 | 0.592133 | 6/8 | 0.200023 |
| 90% | bird | 9/9 | 1.01191 | 8/9 | 0.165979 |
| 90% | fire | 8/8 | 0.0134665 | 8/8 | 0.1945 |
| 90% | war | 6/6 | 0.0589502 | 6/6 | 0.1446 |
| 90% | medical | 9/9 | 0.0161615 | 7/9 | 0.185499 |

## Continuation recovery subject to suppression: best per query

Select one feature and coefficient retrospectively for each query and this objective.
The two objective tables can therefore select different edits. Means below
use only queries both methods reach; reach denominators include all 52 queries.

| SAE mode | Required suppression | SAE reach | FRA reach | SAE lower / comparable | No-edit KL | SAE KL | FRA KL |
|---|---:|---:|---:|---:|---:|---:|---:|
| activation_positive | none | 52/52 | 52/52 | 46/52 | 0.0294655 | 0.0248369 | 0.0286269 |
| activation | none | 52/52 | 52/52 | 52/52 | 0.0294655 | 0.0227096 | 0.0286269 |
| additive | none | 52/52 | 52/52 | 38/52 | 0.0294655 | 0.0266102 | 0.0286269 |
| activation_positive | 30% | 52/52 | 52/52 | 51/52 | 0.0294655 | 0.0271113 | 0.0966136 |
| activation | 30% | 52/52 | 52/52 | 52/52 | 0.0294655 | 0.0253343 | 0.0966136 |
| additive | 30% | 52/52 | 52/52 | 24/52 | 0.0294655 | 0.0897377 | 0.0966136 |
| activation_positive | 50% | 52/52 | 52/52 | 52/52 | 0.0294655 | 0.0281903 | 0.118235 |
| activation | 50% | 52/52 | 52/52 | 52/52 | 0.0294655 | 0.0264743 | 0.118235 |
| additive | 50% | 52/52 | 52/52 | 17/52 | 0.0294655 | 0.211013 | 0.118235 |
| activation_positive | 70% | 52/52 | 51/52 | 51/51 | 0.0297927 | 0.0293099 | 0.139645 |
| activation | 70% | 52/52 | 51/52 | 51/51 | 0.0297927 | 0.0279858 | 0.139645 |
| additive | 70% | 52/52 | 51/52 | 6/51 | 0.0297927 | 0.332905 | 0.139645 |
| activation_positive | 90% | 52/52 | 41/52 | 41/41 | 0.0300999 | 0.0325746 | 0.156545 |
| activation | 90% | 52/52 | 41/52 | 41/41 | 0.0300999 | 0.0316183 | 0.156545 |
| additive | 90% | 52/52 | 41/52 | 2/41 | 0.0300999 | 0.406953 | 0.156545 |

## Full-distribution recovery at the related-word recall query

Select one feature and coefficient retrospectively for each query and this objective.
The two objective tables can therefore select different edits. Means below
use only queries both methods reach; reach denominators include all 52 queries.

| SAE mode | Required suppression | SAE reach | FRA reach | SAE lower / comparable | No-edit KL | SAE KL | FRA KL |
|---|---:|---:|---:|---:|---:|---:|---:|
| activation_positive | none | 52/52 | 52/52 | 45/52 | 1.93027 | 0.201627 | 0.504696 |
| activation | none | 52/52 | 52/52 | 45/52 | 1.93027 | 0.195344 | 0.504696 |
| additive | none | 52/52 | 52/52 | 40/52 | 1.93027 | 0.279539 | 0.504696 |
| activation_positive | 30% | 52/52 | 52/52 | 45/52 | 1.93027 | 0.201627 | 0.504696 |
| activation | 30% | 52/52 | 52/52 | 45/52 | 1.93027 | 0.195344 | 0.504696 |
| additive | 30% | 52/52 | 52/52 | 40/52 | 1.93027 | 0.279539 | 0.504696 |
| activation_positive | 50% | 52/52 | 52/52 | 45/52 | 1.93027 | 0.201627 | 0.504696 |
| activation | 50% | 52/52 | 52/52 | 45/52 | 1.93027 | 0.195344 | 0.504696 |
| additive | 50% | 52/52 | 52/52 | 40/52 | 1.93027 | 0.279539 | 0.504696 |
| activation_positive | 70% | 52/52 | 51/52 | 44/51 | 1.94334 | 0.204782 | 0.501535 |
| activation | 70% | 52/52 | 51/52 | 44/51 | 1.94334 | 0.198375 | 0.501535 |
| additive | 70% | 52/52 | 51/52 | 39/51 | 1.94334 | 0.283054 | 0.501535 |
| activation_positive | 90% | 52/52 | 41/52 | 33/41 | 1.89525 | 0.22333 | 0.487287 |
| activation | 90% | 52/52 | 41/52 | 33/41 | 1.89525 | 0.217748 | 0.487287 |
| additive | 90% | 52/52 | 41/52 | 27/41 | 1.89525 | 0.307136 | 0.487287 |

## Per-concept continuation comparison at 50% suppression

Positive activation-weighted single-feature steering; retrospective per-query choices.

| Concept | SAE reach | FRA reach | SAE lower / comparable | No-edit KL | SAE KL | FRA KL |
|---|---:|---:|---:|---:|---:|---:|
| vessel | 12/12 | 12/12 | 12/12 | 0.023847 | 0.023847 | 0.195507 |
| vehicle | 8/8 | 8/8 | 8/8 | 0.0201424 | 0.0199443 | 0.105609 |
| bird | 9/9 | 9/9 | 9/9 | 0.0534756 | 0.0479491 | 0.100354 |
| fire | 8/8 | 8/8 | 8/8 | 0.0130587 | 0.0127705 | 0.109317 |
| war | 6/6 | 6/6 | 6/6 | 0.0589502 | 0.0568364 | 0.0797702 |
| medical | 9/9 | 9/9 | 9/9 | 0.0161615 | 0.0161615 | 0.077882 |

## Re-evaluating old-metric zero-collateral optima

Choose a minimum-KL point achieving 50% suppression in the archived sweep, breaking
ties by smallest absolute coefficient across features. This can choose a different
tied optimum from the old report, which preferred feature order before coefficient.
Apply that edit unchanged. The old metric used a separate unpoisoned paragraph;
ratios below compare corrected paired KL to its no-edit baseline.

| Concept | Old zero-KL winners / related queries | Mean paired KL | Mean paired / no-edit KL | No improvement in paired KL |
|---|---:|---:|---:|---:|
| vessel | 12/12 | 0.0267751 | 1.1691 | 12/12 |
| vehicle | 8/8 | 0.0328051 | 1.8632 | 6/8 |
| bird | 9/9 | 0.0532867 | 0.9970 | 8/9 |
| fire | 8/8 | 0.0200977 | 1.5553 | 8/8 |
| war | 6/6 | 0.0589502 | 1.0000 | 6/6 |
| medical | 9/9 | 0.0229616 | 1.4163 | 9/9 |

## Verification

| Concept | Max archived FRA suppression difference | Max archived FRA KL relative difference | SAE configs checked with full forwards | Max cached/full mean KL difference |
|---|---:|---:|---:|---:|
| vessel | 0.0315045 | 0.85% | 55 | 0 |
| vehicle | 0.025569 | 0.43% | 46 | 0 |
| bird | 0.0274615 | 0.37% | 49 | 0 |
| fire | 0.0139649 | 0.88% | 46 | 0 |
| war | 0.0073474 | 0.87% | 36 | 0 |
| medical | 0.0130685 | 1.63% | 51 | 0 |

Minimum mean continuation KL anywhere in the measured grid: **0.011048766**.

## Interpretation limits

- Three original contexts and one original paragraph per concept; one model.
- Feature identities come from original planted-word calibration, but final feature/strength
  selection is retrospective. Per-query choices do not establish one transferable fixed edit.
- The reference omits the payload, so its prefix is shorter. Shared continuation tokens
  and their prediction positions are explicitly aligned. KL includes payload priming and
  positional/context changes as well as the planted association.
- Teacher-forced distribution recovery; no independently sampled generation evaluation.
- Lower KL need not imply enough suppression; both unconstrained recovery and constrained
  comparisons are shown. Unreached thresholds remain visible. All grids are finite.
- FP16 GPU execution can differ from the archived run. All method comparisons above use
  freshly measured values in the same environment; reproduction differences are reported.
