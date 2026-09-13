# Variable-binding pilot results

This is a supplied-feature, two-head synthetic pilot. It does not test SAE recovery, discovery of binding features from raw tokens, or an FRA advantage over arbitrary map edits.

## Held-out performance

| Seed | IID accuracy | Unseen bindings | Unseen queries | Joint held-out | Joint NMSE |
|---|---:|---:|---:|---:|---:|
| 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.032613 |
| 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.022977 |
| 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.018245 |

## Mechanistic checks

| Seed | Binding-only QK accuracy | Content-only QK accuracy | Reversed-query accuracy | Binding-swap accuracy |
|---|---:|---:|---:|---:|
| 0 | 1.0000 | 0.0718 | 1.0000 | 1.0000 |
| 1 | 1.0000 | 0.0669 | 1.0000 | 1.0000 |
| 2 | 1.0000 | 0.0898 | 1.0000 | 1.0000 |

## Rebinding edits

Numbers below are means across the trained seeds. Protected-query zero damage is imposed by the common edit gate, not an empirical selectivity advantage. Negative-operand drift measures collateral within edited queries and was added after the initial run.

| Method | Target counterfactual NMSE | Target progress | Negative-operand mean absolute drift | Protected-query output-change NMSE |
|---|---:|---:|---:|---:|
| no_edit | 1.000008 | 0.000000 | 0.000000 | 0 |
| FRA_binding_pairs | 0.013329 | 0.969993 | 0.061680 | 2.78e-31 |
| single_Q_feature_swap | 0.013329 | 0.969991 | 0.061689 | 0 |
| matched_token_score_edit | 0.013329 | 0.969993 | 0.061680 | 0 |
| gated_output_DoM | 1.000943 | 0.000040 | 0.008940 | 0 |

Content-only control joint held-out NMSE: 1.011309; signed-pair accuracy: 0.0132. Population-optimal NMSE without binding information is 1.

Maximum FRA/matched-map difference: 2.11e-15.

The DoM baseline is a fixed context-independent output direction with the same query gate. Its failure alone is not evidence for a distinctive FRA benefit. The single Q-feature swap is the stronger comparator.

[Interactive pilot figure](binding_pilot.html)

Full numerical results, configuration, fixed seeds, and numerical checks are in metrics.json; learned arrays are in seed_*.npz.
