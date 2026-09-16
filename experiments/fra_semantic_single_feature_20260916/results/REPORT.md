# Single-feature SAE comparison with FRA semantic-filter transfer

**Finding:** activation-weighted steering of individual differential SAE features
beats the saved FRA curves on this benchmark. At 30/50/70% suppression it reaches
52/52 related-word queries, with exact zero KL on the original legitimate paragraphs.
The selected features are dormant on those paragraphs; coefficients above ordinary
ablation are needed. Additive decoder-direction steering is substantially weaker.

Six concepts, 52 related-word queries in three contexts. Ten candidates per contrast;
each feature is edited individually. Feature/strength winners are selected after evaluation.
Raw sweeps also retain the 18 planted-word queries.

## Primary: measured points, planted-minus-no-payload ranking

At each threshold, select the lowest measured KL among points that achieve at least
that suppression. FRA uses its saved six-point grid; SAE uses the denser grid listed
in PROTOCOL.md/source. Ratios above one favor FRA. Comparisons use queries both methods reach.

| Single-feature mode | Suppression | SAE reach | FRA reach | SAE lower / comparable | Zero SAE KL / comparable | SAE KL / FRA KL (geomean) |
|---|---:|---:|---:|---:|---:|---:|
| activation_positive | 30% | 52/52 | 52/52 | 52/52 | 52/52 | 0.000 |
| activation_positive | 50% | 52/52 | 50/52 | 50/50 | 50/50 | 0.000 |
| activation_positive | 70% | 52/52 | 48/52 | 48/48 | 48/48 | 0.000 |
| activation | 30% | 52/52 | 52/52 | 52/52 | 52/52 | 0.000 |
| activation | 50% | 52/52 | 50/52 | 50/50 | 50/50 | 0.000 |
| activation | 70% | 52/52 | 48/52 | 48/48 | 48/48 | 0.000 |
| additive | 30% | 52/52 | 52/52 | 23/52 | 0/52 | 1.331 |
| additive | 50% | 52/52 | 50/52 | 4/50 | 0/50 | 4.338 |
| additive | 70% | 52/52 | 48/52 | 1/48 | 0/48 | 6.427 |
| either | 30% | 52/52 | 52/52 | 52/52 | 52/52 | 0.000 |
| either | 50% | 52/52 | 50/52 | 50/50 | 50/50 | 0.000 |
| either | 70% | 52/52 | 48/52 | 48/48 | 48/48 | 0.000 |

## One feature per concept at 50% suppression

Choose a single feature and intervention type per concept, maximizing query coverage
then minimizing mean collateral. Strength may vary by query. Primary contrast.

| Concept | Feature | Diff rank | Positive coefficients used | Reach | SAE lower / comparable | SAE / FRA KL |
|---|---:|---:|---|---:|---:|---:|
| vessel | 7794 | 4 | [12, 24, 32, 48, 64] | 12/12 | 10/10 | 0.000 |
| vehicle | 97 | 6 | [24, 32, 48, 64] | 8/8 | 8/8 | 0.000 |
| bird | 36963 | 3 | [16, 24, 32, 48, 64] | 9/9 | 9/9 | 0.000 |
| fire | 62341 | 3 | [24, 32, 48, 64] | 8/8 | 8/8 | 0.000 |
| war | 34296 | 2 | [6, 8, 12] | 6/6 | 6/6 | 0.000 |
| medical | 36963 | 6 | [32, 48, 64] | 9/9 | 9/9 | 0.000 |

## Strength dependence

Positive activation-weighted coefficients only. c=1 removes the encoded feature contribution;
c>1 subtracts more than that contribution. Best individual feature among the primary ten.

| Maximum c | Reach at 30% | Reach at 50% | Reach at 70% | Zero-KL reach at 30/50/70% |
|---:|---:|---:|---:|---|
| 1 | 4/52 | 1/52 | 0/52 | [0, 0, 0] |
| 4 | 52/52 | 44/52 | 19/52 | [2, 0, 0] |
| 8 | 52/52 | 52/52 | 49/52 | [14, 3, 1] |
| 16 | 52/52 | 52/52 | 52/52 | [28, 17, 7] |
| 32 | 52/52 | 52/52 | 52/52 | [47, 41, 35] |
| 64 | 52/52 | 52/52 | 52/52 | [52, 52, 52] |

## Sensitivity: interpolated curves and both contrasts

Interpolation stays within each feature/mode curve, between adjacent strength values,
and includes the zero-edit point. These are estimates, not fresh measured operating points.

| Comparison | Suppression | SAE reach | SAE lower / comparable | SAE / FRA KL |
|---|---:|---:|---:|---:|
| measured/either/either | 30% | 52/52 | 52/52 | 0.000 |
| measured/either/either | 50% | 52/52 | 50/50 | 0.000 |
| measured/either/either | 70% | 52/52 | 48/48 | 0.000 |
| interpolated/no_payload/either | 30% | 52/52 | 52/52 | 0.000 |
| interpolated/no_payload/either | 50% | 52/52 | 50/50 | 0.000 |
| interpolated/no_payload/either | 70% | 52/52 | 48/48 | 0.000 |
| interpolated/either/either | 30% | 52/52 | 52/52 | 0.000 |
| interpolated/either/either | 50% | 52/52 | 50/50 | 0.000 |
| interpolated/either/either | 70% | 52/52 | 48/48 | 0.000 |

## Reproduction checks

| Concept | Max clean probability error | Max 12-feature suppression error | Max 12-feature KL relative error |
|---|---:|---:|---:|
| vessel | 0.001888 | 0.010936 | 0.6669% |
| vehicle | 0.002458 | 0.028215 | 0.5564% |
| bird | 0.004316 | 0.014968 | 0.4322% |
| fire | 0.003295 | 0.269025 | 3.6062% |
| war | 0.003149 | 0.009565 | 0.6815% |
| medical | 0.003111 | 0.014660 | 0.0834% |

All collateral grid points were remeasured with batch size one. Zero-KL points
leave the residual tensor exactly unchanged on the legitimate paragraph. Direct full
forward passes also checked dormant features and the selected operating points.

| Concept | Directly verified target points | Dormant features checked | All dormant checks bitwise identical? |
|---|---:|---:|---|
| vessel | 945 | 8 | True |
| vehicle | 602 | 6 | True |
| bird | 669 | 7 | True |
| fire | 683 | 6 | True |
| war | 623 | 6 | True |
| medical | 805 | 8 | True |

The original 12-feature fire curve does not reproduce tightly: its largest
suppression discrepancy is about 0.27. Other concepts have maximum discrepancies
of about 0.01–0.03. FRA is compared against the saved author run, not a fresh FRA
run in this environment; do not describe this as a bitwise reproduction of the campaign.

## Scope

- Same single layer (residual-pre layer 6) as the original SAE baseline; no layer search.
- Same one legitimate paragraph per concept; no general capability evaluation.
- Best-of-ten selection is retrospective; these are optimistic baseline results.
- FRA data are the committed reference run; clean probabilities and original SAE curves
  are rerun as checks of cross-run comparability.
- Positive activation coefficient removes a feature; negative coefficient amplifies it.
- Additive coefficient uses a unit decoder vector in residual-stream units.
