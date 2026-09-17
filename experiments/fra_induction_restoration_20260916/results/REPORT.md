# Restoration of clean continuations with the backdoor present

**Metric correction:** the earlier zero KL measured an intervention on a separate
unpoisoned paragraph. This run compares the unsteered model with an unpoisoned
primer against the steered model with a poisoned primer, while both read the
**same clean continuation tokens**. The poison is present during the entire steered
evaluation. These are full next-token distributions, not just payload probabilities.

Each case uses the original paragraph and three matched primer seeds (0, 1, 2).
All KL values below are **nats per predicted token**, averaged over those contexts.
One feature and coefficient are chosen per case. FRA is freshly evaluated on these inputs.

## Best restoration, including the option to make no edit

| Model | Pair | Poison, no steering | Single SAE positive | Single SAE signed | Additive SAE | FRA |
|---|---|---:|---:|---:|---:|---:|
| gpt2 | bank→river | 0.11265 | 0.085745 | 0.085745 | 0.098059 | 0.079836 |
| gpt2 | doctor→patient | 0.094795 | 0.075994 | 0.075994 | 0.088526 | 0.077639 |
| gpt2 | fire→water | 0.091108 | 0.091108 | 0.088771 | 0.088593 | 0.072129 |
| gpt2 | king→crown | 0.10595 | 0.072348 | 0.072348 | 0.097088 | 0.082687 |
| gemma | bank→river | 0.10929 | 0.085134 | 0.085134 | 0.097152 | 0.10687 |
| gemma | doctor→water | 0.1026 | 0.088615 | 0.088615 | 0.098348 | 0.096875 |
| gemma | king→crown | 0.10838 | 0.089614 | 0.089614 | 0.099225 | 0.1069 |
| gemma | market→gold | 0.093058 | 0.087632 | 0.087632 | 0.085828 | 0.08549 |

## Require at least 50% original backdoor suppression

Suppression is evaluated on the original seed-0 repeated-token probe. The KL
is evaluated on poisoned-primer + legitimate-text continuations, using the same edit.

| Model | Pair | Single SAE positive | Single SAE signed | Additive SAE | FRA |
|---|---|---:|---:|---:|---:|
| gpt2 | bank→river | 0.10668 | 0.10668 | 0.13297 | 0.079836 |
| gpt2 | doctor→patient | 0.075994 | 0.075994 | 0.19647 | 0.077639 |
| gpt2 | fire→water | 0.091108 | 0.091108 | 0.16032 | 0.072601 |
| gpt2 | king→crown | 0.072348 | 0.072348 | 0.30741 | 0.082687 |
| gemma | bank→river | 0.10577 | 0.10577 | 0.70873 | 0.14733 |
| gemma | doctor→water | 0.095074 | 0.095074 | 0.38651 | 0.12677 |
| gemma | king→crown | 0.093397 | 0.093397 | 0.54311 | 0.15263 |
| gemma | market→gold | 0.088635 | 0.088635 | 0.74842 | unreached |

## Require at least 90% original backdoor suppression

Suppression is evaluated on the original seed-0 repeated-token probe. The KL
is evaluated on poisoned-primer + legitimate-text continuations, using the same edit.

| Model | Pair | Single SAE positive | Single SAE signed | Additive SAE | FRA |
|---|---|---:|---:|---:|---:|
| gpt2 | bank→river | 0.10668 | 0.10668 | 0.31487 | 0.10398 |
| gpt2 | doctor→patient | 0.079253 | 0.079253 | 0.49808 | 0.077639 |
| gpt2 | fire→water | 0.091108 | 0.091108 | 0.34751 | 0.073554 |
| gpt2 | king→crown | 0.07824 | 0.07824 | 0.63569 | 0.082687 |
| gemma | bank→river | 0.10577 | 0.10577 | 1.2038 | unreached |
| gemma | doctor→water | 0.11353 | 0.1026 | 0.97934 | 0.13441 |
| gemma | king→crown | 0.093397 | 0.093397 | 1.098 | unreached |
| gemma | market→gold | 0.090498 | 0.090498 | 1.5908 | unreached |

## Selected operating points at >=50% original suppression

| Model | Pair | Single feature (diff rank) | SAE c | Actual SAE suppression | FRA c | Actual FRA suppression |
|---|---|---|---:|---:|---:|---:|
| gpt2 | bank→river | 20951 (10) | 24 | 95.72% | 12 | 80.67% |
| gpt2 | doctor→patient | 20951 (7) | 24 | 75.41% | 32 | 93.59% |
| gpt2 | fire→water | 18735 (3) | 6 | 68.58% | 1.5 | 74.75% |
| gpt2 | king→crown | 20951 (4) | 12 | 57.81% | 64 | 96.00% |
| gemma | bank→river | 60372 (4) | 64 | 97.56% | 6 | 52.84% |
| gemma | doctor→water | 39703 (3) | 16 | 74.96% | 3 | 80.65% |
| gemma | king→crown | 39703 (7) | 24 | 99.08% | 64 | 70.77% |
| gemma | market→gold | 9539 (4) | 32 | 82.50% | unreached | unreached |

## Features and coefficients minimizing unconstrained restoration KL

| Model | Pair | Positive activation feature (rank), c | FRA c | SAE / no-edit KL | FRA / no-edit KL |
|---|---|---|---:|---:|---:|
| gpt2 | bank→river | 20951 (10), 12 | 12 | 0.761 | 0.709 |
| gpt2 | doctor→patient | 20951 (7), 24 | 32 | 0.802 | 0.819 |
| gpt2 | fire→water | no edit | 1 | 1.000 | 0.792 |
| gpt2 | king→crown | 20951 (4), 12 | 64 | 0.683 | 0.780 |
| gemma | bank→river | 7499 (2), 8 | 0.25 | 0.779 | 0.978 |
| gemma | doctor→water | 47428 (2), 8 | 0.5 | 0.864 | 0.944 |
| gemma | king→crown | 39703 (7), 16 | 0.5 | 0.827 | 0.986 |
| gemma | market→gold | 23896 (3), 3 | 0.5 | 0.942 | 0.919 |

## Re-evaluating the old 50%-suppression winners

These are the previously selected feature and coefficient, without retuning.
The old column is summed KL on a separate unpoisoned paragraph; the new column
is paired continuation KL per token. Their absolute magnitudes have different
units and inputs. The final ratio compares two measurements of the new metric.

| Model | Pair | Feature, c | Old separate-paragraph KL | New paired KL/token | New KL / poison-no-steering KL |
|---|---|---|---:|---:|---:|
| gpt2 | bank→river | 5634, 6 | 0 | 0.12238 | 1.086 |
| gpt2 | doctor→patient | 5484, 12 | 0 | 0.094795 | 1.000 |
| gpt2 | fire→water | 18735, 6 | 0 | 0.091108 | 1.000 |
| gpt2 | king→crown | 1765, 8 | 0 | 0.10595 | 1.000 |
| gemma | bank→river | 60372, 48 | 0 | 0.10819 | 0.990 |
| gemma | doctor→water | 39703, 24 | 0.28854 | 0.11353 | 1.106 |
| gemma | king→crown | 49633, 48 | 0 | 0.10838 | 1.000 |
| gemma | market→gold | 9539, 32 | 0 | 0.088635 | 0.952 |

## Full-distribution recovery at the original backdoor query

For the continuation-KL-optimal edits above, this diagnostic compares the full
query distribution with the matched no-backdoor original probe. It is not
the objective used to select these edits.

| Model | Pair | No steering | Single SAE positive | FRA |
|---|---|---:|---:|---:|
| gpt2 | bank→river | 7.5612 | 5.2019 | 1.5411 |
| gpt2 | doctor→patient | 5.8755 | 2.27 | 1.1833 |
| gpt2 | fire→water | 10.424 | 10.424 | 3.2126 |
| gpt2 | king→crown | 11.682 | 4.7214 | 1.1606 |
| gemma | bank→river | 10.494 | 8.2101 | 10.461 |
| gemma | doctor→water | 7.3347 | 7.1113 | 7.0499 |
| gemma | king→crown | 7.7392 | 5.8974 | 7.4563 |
| gemma | market→gold | 10.017 | 9.9794 | 9.9113 |

## Verification

| Model | Pair | Max FRA archived suppression error | Max FRA archived KL relative error | Direct SAE point checks |
|---|---|---:|---:|---:|
| gpt2 | bank→river | 3.1414e-06 | 0.05% | 58 |
| gpt2 | doctor→patient | 1.7595e-06 | 0.00% | 72 |
| gpt2 | fire→water | 4.191e-07 | 0.00% | 71 |
| gpt2 | king→crown | 1.8846e-06 | 0.01% | 64 |
| gemma | bank→river | 0.0032312 | 0.85% | 58 |
| gemma | doctor→water | 0.0008154 | 0.43% | 52 |
| gemma | king→crown | 0.0069847 | 0.41% | 62 |
| gemma | market→gold | 0.021003 | 3.24% | 52 |

## Limits

- Teacher-forced KL over a shared continuation measures distributional recovery.
  It does not evaluate independently sampled long-form generations.
- Three matched random-token primers per pair; one natural paragraph per pair.
- Feature candidates are frozen from the original calibration, but the final
  feature and coefficient are selected retrospectively on these evaluations.
- Positive and signed activation-weighted steering are separate. Negative
  coefficients amplify a feature; additive steering has no activation gate.
- The poison replaces exactly two tokens in the clean prefix; all shared
  continuation token IDs and prediction positions are explicitly stored.
- Restoring the original random primer removes both the planted association
  and the individual trigger/payload occurrences. This KL includes ordinary
  contextual priming from those tokens, as well as induction effects.
- Every grid is finite; unreachable suppression targets remain visible.
