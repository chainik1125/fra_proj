# Compound semantic filtering: held-out results

One corrupted routing-table entry changes printer + wireless incidents from Network to Print. Printer + power must still route to Print. Device and fault descriptions appear in separate spans. Gemma-2-9B-IT was selected by the calibration behavior gate before testing any intervention.

Calibration: 8 factorial blocks; tuning: 4; test: 12 (48 cases, including 12 joint-trigger cases). Test blocks comprise six lexical pairs in two document orders. All cases are retained.

## Does the compound backdoor generalize?

| Equipment | Problem | Clean accuracy | Poison accuracy | Clean P(Print) | Poison P(Print) | Poison KL |
|---|---|---:|---:|---:|---:|---:|
| Portable computer | Power | 100.0% | 100.0% | 0.0004 | 0.0004 | 0.0000 |
| Printer | Power | 100.0% | 100.0% | 0.9454 | 0.9439 | 0.0000 |
| Portable computer | Wireless | 100.0% | 100.0% | 0.0003 | 0.0007 | 0.0013 |
| Printer | Wireless | 100.0% | 0.0% | 0.0287 | 0.7596 | 1.7279 |

## Fixed interventions selected on tuning only

Each row uses one fixed feature/coefficient or pair-set/coefficient for all test cases. Thresholds constrain mean joint suppression on **tuning**; test suppression may differ. The objective is lowest mean full-vocabulary KL across all four corners.

| Family | Tuning suppression constraint | Setting | Test joint suppression | Joint accuracy | Control accuracy | Joint KL | Control KL | Overall KL |
|---|---|---|---:|---:|---:|---:|---:|---:|
| baseline | none | Poison, no edit | 0.0% | 0.0% | 100.0% | 1.7279 | 0.0005 | 0.4323 |
| sae_positive | none | SAE L21 f3994 c=8 | 91.3% | 100.0% | 100.0% | 0.0789 | 0.0119 | 0.0286 |
| sae_positive | >=50% | SAE L21 f3994 c=8 | 91.3% | 100.0% | 100.0% | 0.0789 | 0.0119 | 0.0286 |
| sae_positive | >=90% | SAE L21 f3994 c=8 | 91.3% | 100.0% | 100.0% | 0.0789 | 0.0119 | 0.0286 |
| sae_signed | none | SAE L21 f3994 c=8 | 91.3% | 100.0% | 100.0% | 0.0789 | 0.0119 | 0.0286 |
| sae_signed | >=50% | SAE L21 f3994 c=8 | 91.3% | 100.0% | 100.0% | 0.0789 | 0.0119 | 0.0286 |
| sae_signed | >=90% | SAE L21 f3994 c=8 | 91.3% | 100.0% | 100.0% | 0.0789 | 0.0119 | 0.0286 |
| fra_raw | none | FRA raw c=8 | 57.8% | 58.3% | 97.2% | 0.4496 | 0.0735 | 0.1675 |
| fra_raw | >=50% | FRA raw c=8 | 57.8% | 58.3% | 97.2% | 0.4496 | 0.0735 | 0.1675 |
| fra_raw | >=90% | No feasible tuning setting | — | — | — | — | — | — |
| fra_interaction | none | FRA interaction c=8 | 8.5% | 8.3% | 94.4% | 1.3380 | 0.3042 | 0.5626 |
| fra_interaction | >=50% | No feasible tuning setting | — | — | — | — | — | — |
| fra_interaction | >=90% | No feasible tuning setting | — | — | — | — | — | — |
| fra_separable | none | FRA separable c=64 | 38.8% | 50.0% | 100.0% | 0.6854 | 0.0043 | 0.1746 |
| fra_separable | >=50% | No feasible tuning setting | — | — | — | — | — | — |
| fra_separable | >=90% | No feasible tuning setting | — | — | — | — | — | — |
| sae_diff_positive | none | SAE L21 f15824 c=32 | 72.9% | 100.0% | 100.0% | 0.1948 | 0.0162 | 0.0608 |
| sae_diff_positive | >=50% | SAE L21 f15824 c=32 | 72.9% | 100.0% | 100.0% | 0.1948 | 0.0162 | 0.0608 |
| sae_diff_positive | >=90% | SAE L10 f11818 c=64 | 98.1% | 100.0% | 100.0% | 0.0527 | 0.0276 | 0.0339 |
| sae_diff_signed | none | SAE L21 f15824 c=32 | 72.9% | 100.0% | 100.0% | 0.1948 | 0.0162 | 0.0608 |
| sae_diff_signed | >=50% | SAE L21 f15824 c=32 | 72.9% | 100.0% | 100.0% | 0.1948 | 0.0162 | 0.0608 |
| sae_diff_signed | >=90% | SAE L10 f11818 c=64 | 98.1% | 100.0% | 100.0% | 0.0527 | 0.0276 | 0.0339 |

## Interpretation

The held-out behavior is a selective semantic AND gate: the corruption flips all 12 printer/wireless cases while all 36 control cases remain correct. This does not produce an FRA advantage on the tested grid. The joint-feature SAE winner repairs 12/12 trigger cases and preserves 36/36 controls; the raw FRA winner repairs 7/12 and preserves 35/36. The separable FRA variant preserves all controls and has lower control KL, but repairs only 6/12 trigger cases.

The winning SAE feature is active only on the joint corner at the calibration answer position (mean 2.439 versus zero in all three controls). Thus the model has already combined the two input conditions into one SAE feature at this site. This is consistent with why a textual compound trigger can still be removable by single-feature steering. It does not prove this feature is the entire causal circuit, or localize which of the all-position edits caused the repair.

Moreover, oracle deletion of the tested direct asset-span attention edges repairs none of the 12 joint cases. This fails to establish that the intended direct QK route is necessary at these layers; information may have moved earlier or through other positions. A stronger test of the two-feature hypothesis needs a causal interaction localization before comparing surgical cuts.

The original poison-minus-clean-only baseline is also evaluated separately in the table, as described in [DIFF_BASELINE](../DIFF_BASELINE.md). The stronger winner above comes from the factorial interaction ranking, not the original diff ranking.

## QK identification and mechanism checks

All three native IT SAE layers are searched: residual-post 9/20/31, used at attention layers 10/21/32. Single-feature candidates are the union of top-ten poisoned-minus-clean and top-ten factorial-contrast features per layer, measured at the answer prefix. Each candidate is swept independently, preserving the SAE reconstruction residual. Positive-only and signed steering are reported separately.

FRA cuts are **QK score cuts**. Pair terms are identified from the later problem/answer token to the earlier asset-type token. Raw, factorial-interaction, and separable interaction rankings each select up to 48 pairs per head. All 16 heads in each of the three layers are included. The separable variant requires a wireless-sensitive Q feature and a distinct printer-sensitive K feature; activation ratios must exceed 2 in both control strata. This response criterion is measured on calibration, not a human semantic label from unrelated text. Cuts then act on every causal token pair, with no test-position mask.

| Pair set | Selected pairs | Heads with nonempty set |
|---|---:|---:|
| raw | 2304 | 48 / 48 |
| interaction | 2304 | 48 / 48 |
| separable | 2304 | 48 / 48 |

Example calibrated pair at L10H0: query feature 5231 and key feature 8761. Values are mean activations over calibration blocks and the two query anchors, not global semantic annotations.

| Corner (printer, wireless) | Q feature activation | K feature activation | QK contribution |
|---|---:|---:|---:|
| 00 | 0.0000 | 0.7132 | 0.000000 |
| 10 | 0.0000 | 49.2919 | 0.000000 |
| 01 | 23.7452 | 0.7130 | -0.000563 |
| 11 | 23.0947 | 49.2986 | -0.023338 |

### Selected single-feature provenance

| Fixed feature | Calibration ranking membership | Activation 00 | Activation 10 | Activation 01 | Activation 11 |
|---|---|---:|---:|---:|---:|
| L21 f3994 | interaction | 0.000 | 0.000 | 0.000 | 2.439 |
| L21 f15824 | diff, interaction | 1.423 | 1.400 | 0.693 | 2.163 |
| L10 f11818 | diff, interaction | 0.000 | 0.000 | 0.304 | 0.914 |

## Unrestricted output check

Label accuracy above chooses among the four queue labels. The checks below use the highest-probability token across the ENTIRE vocabulary, and report total queue-label mass.

| Fixed test setting | Full-vocabulary top-token accuracy | Mean queue-label probability mass |
|---|---:|---:|
| Poison, no edit | 75.0% | 0.9524 |
| SAE L21 f3994 c=8 | 100.0% | 0.9568 |
| FRA raw c=8 | 87.5% | 0.9154 |
| FRA interaction c=8 | 72.9% | 0.9502 |
| FRA separable c=64 | 87.5% | 0.9511 |
| SAE L21 f15824 c=32 | 100.0% | 0.9710 |
| SAE L10 f11818 c=64 | 100.0% | 0.9288 |

Oracle ablation of all later attention to the asset span at the tested layers gives 4.2% joint suppression, 0.0% joint accuracy, 100.0% control accuracy and 0.3921 overall KL. This position-aware diagnostic is not included in the competitive comparison.

## Normalization correction and verification

The inherited wrapper claims per-token normalization is required. This conflicts with [Gemma Scope §3.1](https://storage.googleapis.com/gemma-scope/gemma-scope-report.pdf): released weights already absorb the fixed training scale. Both methods here use native encoding without extra input normalization. The aborted pre-correction smoke source/log are retained. Earlier normalization=True results need a separate audit.

| Attention layer | Native mean L0 | Old mean L0 | Native relative squared error | Old relative squared error |
|---|---:|---:|---:|---:|
| 10 | 88.72 | 25.89 | 0.1580 | 0.3764 |
| 21 | 123.37 | 9.08 | 0.1362 | 0.9885 |
| 32 | 80.60 | 10.04 | 0.1067 | 2.6067 |

Numerical validity: 6 of 817 tuning configurations produced non-finite logits and were excluded from selection. The archive retains each affected configuration and case. No failed value was assigned zero KL.

| Invalid setting | Affected tuning cases |
|---|---:|
| SAE L10 f12085 c=64 | 16 |
| SAE L21 f1295 c=32 | 16 |
| SAE L21 f1295 c=64 | 16 |
| SAE L32 f6919 c=16 | 16 |
| SAE L32 f6919 c=32 | 16 |
| SAE L32 f6919 c=64 | 16 |

Verification: 4 main-sweep selected configurations were checked with full-prefix forward passes before test selection. Maximum mean-KL discrepancy: 0.000763976; maximum target-probability discrepancy: 0.00372136. All test points use full-prefix forwards. Zero-edit target-probability error: 0. Final-token-only unembedding was checked against ordinary unembedding (maximum logit error 0).

## Scope

- This is a controlled corrupted-knowledge-base simulation, not a weight-trained sleeper agent.
- A textual AND gate and calibrated separable feature responses do not prove an isolated causal two-feature circuit.
- KL is clean-context versus poisoned-context-plus-edit over the full next-token vocabulary at the same queue continuation. It is not an unrelated-text KL. This one-token routing task does not establish long-form generation fidelity.
- Results cover one task, one model, three SAE sites, fixed pair selectors and a finite steering grid. They cannot establish a general advantage or impossibility for FRA.
- The model and SAE preprocessing differ from the earlier single-trigger experiments. This is not a controlled estimate of the effect of adding a second trigger condition.
- The settings and paraphrases were frozen before test evaluation. Feasibility failures and all held-out cases are archived.

See [PROTOCOL](../PROTOCOL.md), [feasibility screens](FEASIBILITY.md), and the complete compressed per-case archive.
