# Original induction backdoors: best-of-ten single-feature SAE baseline

**Finding at 50% suppression:** gpt2: single SAE reaches 4/4, has lower KL than FRA on 4/4 jointly reachable cases, and zero KL on 4/4; gemma: single SAE reaches 4/4, has lower KL than FRA on 3/3 jointly reachable cases, and zero KL on 3/4.

This is the original repeated-token IC4/G4 benchmark, including bank→river and king→crown.
Ten features per case are ranked by the original ON-minus-OFF activation difference;
each is steered separately. Every measured winner below was checked with ordinary
full forward passes at batch size one on both target and legitimate text.

FRA curves are the archived author runs. The original top-12 SAE baseline is rerun
as a comparability check. Collateral is summed KL in nats over one legitimate paragraph.

## Primary comparison: positive activation-weighted removal

Select the lowest measured KL achieving at least the stated suppression. The feature
and coefficient are selected retrospectively per case and threshold. `c=1` is ordinary
feature ablation; larger coefficients subtract more than its encoded contribution.

| Model | Suppression | SAE reach | FRA reach | SAE lower / both reach | SAE zero KL |
|---|---:|---:|---:|---:|---:|
| gpt2 | 30% | 4/4 | 4/4 | 4/4 | 4/4 |
| gpt2 | 50% | 4/4 | 4/4 | 4/4 | 4/4 |
| gpt2 | 70% | 4/4 | 4/4 | 4/4 | 4/4 |
| gpt2 | 90% | 4/4 | 3/4 | 3/3 | 4/4 |
| gpt2 | 99% | 4/4 | 1/4 | 1/1 | 4/4 |
| gemma | 30% | 4/4 | 4/4 | 4/4 | 3/4 |
| gemma | 50% | 4/4 | 3/4 | 3/3 | 3/4 |
| gemma | 70% | 4/4 | 1/4 | 1/1 | 3/4 |
| gemma | 90% | 4/4 | 1/4 | 1/1 | 3/4 |
| gemma | 99% | 4/4 | 0/4 | 0/0 | 1/4 |

## Per-case winners at 50% suppression

| Model | Pair | Feature (diff rank) | c | Actual suppression | Single SAE KL | FRA KL | Top-12 SAE KL |
|---|---|---|---:|---:|---:|---:|---:|
| gpt2 | bank→river | 5634 (5) | 6 | 65.84% | 0 | 0.00778738 | 1.12948 |
| gpt2 | fire→water | 18735 (3) | 6 | 68.58% | 0 | 0.214902 | 12.7917 |
| gpt2 | king→crown | 1765 (7) | 8 | 95.35% | 0 | 0.013046 | 8.4362 |
| gpt2 | doctor→patient | 5484 (3) | 12 | 79.53% | 0 | 0.0300709 | 5.55767 |
| gemma | bank→river | 60372 (4) | 48 | 77.16% | 0 | 0.754995 | 34.1235 |
| gemma | king→crown | 49633 (6) | 48 | 92.65% | 0 | 0.568918 | 15.8872 |
| gemma | doctor→water | 39703 (3) | 24 | 99.48% | 0.288538 | 0.605179 | 18.9401 |
| gemma | market→gold | 9539 (4) | 32 | 82.50% | 0 | unreached | 63.8762 |

## Per-case winners at 70% suppression

| Model | Pair | Feature (diff rank) | c | Actual suppression | Single SAE KL | FRA KL | Top-12 SAE KL |
|---|---|---|---:|---:|---:|---:|---:|
| gpt2 | bank→river | 5634 (5) | 8 | 99.01% | 0 | 0.0130403 | 1.12948 |
| gpt2 | fire→water | 18735 (3) | 8 | 82.15% | 0 | 0.214902 | 12.7917 |
| gpt2 | king→crown | 1765 (7) | 8 | 95.35% | 0 | 0.013046 | 8.4362 |
| gpt2 | doctor→patient | 5484 (3) | 12 | 79.53% | 0 | 0.0300709 | 5.55767 |
| gemma | bank→river | 60372 (4) | 48 | 77.16% | 0 | unreached | 34.1235 |
| gemma | king→crown | 49633 (6) | 48 | 92.65% | 0 | unreached | 15.8872 |
| gemma | doctor→water | 39703 (3) | 24 | 99.48% | 0.288538 | 0.605179 | 18.9401 |
| gemma | market→gold | 9539 (4) | 32 | 82.50% | 0 | unreached | 63.8762 |

## Per-case winners at 99% suppression

| Model | Pair | Feature (diff rank) | c | Actual suppression | Single SAE KL | FRA KL | Top-12 SAE KL |
|---|---|---|---:|---:|---:|---:|---:|
| gpt2 | bank→river | 5634 (5) | 8 | 99.01% | 0 | unreached | 1.12948 |
| gpt2 | fire→water | 1765 (6) | 16 | 99.28% | 0 | unreached | 12.7917 |
| gpt2 | king→crown | 1765 (7) | 12 | 99.88% | 0 | 0.0234421 | 8.4362 |
| gpt2 | doctor→patient | 5484 (3) | 24 | 99.28% | 0 | unreached | 5.55767 |
| gemma | bank→river | 39703 (8) | 32 | 99.64% | 1.00572 | unreached | 56.8748 |
| gemma | king→crown | 49633 (6) | 64 | 99.70% | 0 | unreached | 43.0986 |
| gemma | doctor→water | 39703 (3) | 24 | 99.48% | 0.288538 | unreached | 56.4431 |
| gemma | market→gold | 13693 (10) | 12 | 99.90% | 1.7886 | unreached | 63.8762 |

## Signed activation and additive steering at 50% suppression

| Model | Pair | Signed activation feature / coefficient / KL | Additive feature / coefficient / KL |
|---|---|---|---|
| gpt2 | bank→river | 5634 / 6 / 0 | 5402 / 16 / 1.28129 |
| gpt2 | fire→water | 18735 / 6 / 0 | 13497 / 16 / 1.67289 |
| gpt2 | king→crown | 1765 / 8 / 0 | 18735 / 32 / 4.97821 |
| gpt2 | doctor→patient | 5484 / 12 / 0 | 21331 / 16 / 1.99288 |
| gemma | bank→river | 60372 / 48 / 0 | 50567 / 64 / 6.54846 |
| gemma | king→crown | 49633 / 48 / 0 | 24024 / 64 / 4.89602 |
| gemma | doctor→water | 62312 / -64 / 0 | 24024 / 64 / 3.89443 |
| gemma | market→gold | 9539 / 32 / 0 | 11458 / 64 / 10.5251 |

Negative activation coefficients amplify the feature. The positive-only primary
comparison excludes them. Additive steering applies a constant unit-decoder
direction and does not inherit the activation-dependent gate.

## Why some collateral values are exactly zero

For the positive-activation winners at 50% suppression:

| Model | Pair | Feature | Active tokens in legitimate paragraph | Logits bitwise unchanged? |
|---|---|---:|---:|---|
| gpt2 | bank→river | 5634 | 0 | True |
| gpt2 | fire→water | 18735 | 0 | True |
| gpt2 | king→crown | 1765 | 0 | True |
| gpt2 | doctor→patient | 5484 | 0 | True |
| gemma | bank→river | 60372 | 0 | True |
| gemma | king→crown | 49633 | 0 | True |
| gemma | doctor→water | 39703 | 1 | False |
| gemma | market→gold | 9539 | 0 | True |

These dormant-feature results establish zero collateral on the original paragraph.
They do not establish preservation on a broader distribution of legitimate text.

## Ordinary single-feature ablation (c=1)

| Model | Pair | Best suppression among ten | KL at that point |
|---|---|---:|---:|
| gpt2 | bank→river | 1.50% | 0.387742 |
| gpt2 | fire→water | 0.02% | 0 |
| gpt2 | king→crown | 0.37% | 0.410301 |
| gpt2 | doctor→patient | 18.08% | 1.0368 |
| gemma | bank→river | 0.05% | 0.111836 |
| gemma | king→crown | 1.05% | 0.221186 |
| gemma | doctor→water | 0.07% | 0.0232135 |
| gemma | market→gold | 0.02% | 0.024781 |

## Reproduction and direct verification

| Model | Pair | Clean probability error | Max top-12 suppression error | Max top-12 KL relative error | Direct checks |
|---|---|---:|---:|---:|---:|
| gpt2 | bank→river | 5.96046e-07 | 5.65595e-07 | 0.00% | 48 |
| gpt2 | fire→water | 0 | 1.0874e-06 | 0.00% | 58 |
| gpt2 | king→crown | 2.38419e-07 | 8.46329e-08 | 0.00% | 59 |
| gpt2 | doctor→patient | 1.13249e-06 | 3.25254e-07 | 0.00% | 61 |
| gemma | bank→river | 8.10623e-06 | 0.000606894 | 0.25% | 38 |
| gemma | king→crown | 8.81553e-05 | 0.00218779 | 0.10% | 49 |
| gemma | doctor→water | 3.18289e-05 | 0.00316018 | 0.33% | 41 |
| gemma | market→gold | 4.25577e-05 | 0.000911271 | 0.38% | 40 |

## Scope and limitations

- One original evaluation seed and one original legitimate paragraph per case.
- Same single SAE layer as the original baseline; no layer search.
- Best-of-ten feature and coefficient selection is retrospective.
- Unreachable FRA targets remain visible; no ratios are assigned to them.
- FRA uses its archived sparse coefficient grid. Single SAE uses a denser grid.
- `summary.json.gz` includes both measured results and adjacent-coefficient
  interpolated estimates. Interpolation never crosses feature identities.
- This is not a general capability or transfer evaluation.
