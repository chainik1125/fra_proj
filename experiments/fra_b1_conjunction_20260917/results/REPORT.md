# B1 reproduction: seed-wise comparison

12 completed group/seed cases; 336 raw sweep rows.

Collateral is the maximum over reuseA and reuseB, measured as next-token
KL(unedited || edited), in nats. First-crossing interpolation is performed
within each seed, including the exact no-edit origin. Oracle collateral is
excluded. See PROTOCOL.md for limitations and the original logs for the
collaborator's pooled-seed summary.

## Removal coverage before comparing collateral

| Group | Seed | Base target P | Max FRA removal | Max feat1 removal | Max oracle removal |
|---|---:|---:|---:|---:|---:|
| bluemoon | 0 | 0.7904 | 33.0% | 100.0% | 9.1% |
| bluemoon | 1 | 0.3861 | 56.1% | 100.0% | 14.4% |
| bluemoon | 2 | 0.7427 | 44.5% | 100.0% | 13.1% |
| bluemoon | 3 | 0.6948 | 44.4% | 100.0% | 27.3% |
| irongate | 0 | 0.7141 | 45.6% | 100.0% | 23.7% |
| irongate | 1 | 0.6897 | 61.7% | 100.0% | 13.1% |
| irongate | 2 | 0.7585 | 24.2% | 100.0% | 15.6% |
| irongate | 3 | 0.6695 | 57.5% | 100.0% | 47.3% |
| redfox | 0 | 0.8301 | 18.5% | 100.0% | 6.2% |
| redfox | 1 | 0.4433 | 72.0% | 100.0% | 44.7% |
| redfox | 2 | 0.8074 | 15.4% | 100.0% | 8.2% |
| redfox | 3 | 0.3652 | 70.7% | 100.0% | 75.8% |

The maximum is over the original coefficient grid. Oracle means masking
the final query → target payload token edge on the nine selected induction
heads; it is not an all-head or all-edge removal bound.

## At 50% removal

Each method's mean uses only its reached cases. When coverage differs,
use the common-subset comparison below to compare FRA against feat1.

| Method | Reached | Mean worst KL, interpolated | Mean worst payload drop | Measured ≥ threshold: mean best worst KL |
|---|---:|---:|---:|---:|
| fra | 5/12 | 0.23489 | 0.31434 | 0.29498 |
| feat1 | 12/12 | 0.86757 | 0.64147 | 0.88979 |
| dom | 12/12 | 0.73728 | 0.48101 | 1.27911 |
| pay | 12/12 | 0.30441 | 0.26532 | 0.60476 |
| oracle (removal only) | 1/12 | — | — | — |

On the 5 cases reached by both FRA and feat1, FRA has lower
interpolated worst KL in 5 cases.

On this common subset, mean worst KL is 0.23489
for FRA and 0.59907 for feat1.

| Group | Method | Reached | Mean worst KL, interpolated |
|---|---|---:|---:|
| bluemoon | fra | 1/4 | 0.08397 |
| bluemoon | feat1 | 4/4 | 0.85096 |
| bluemoon | dom | 4/4 | 0.72940 |
| bluemoon | pay | 4/4 | 0.30809 |
| irongate | fra | 2/4 | 0.17709 |
| irongate | feat1 | 4/4 | 0.58638 |
| irongate | dom | 4/4 | 0.61473 |
| irongate | pay | 4/4 | 0.29070 |
| redfox | fra | 2/4 | 0.36814 |
| redfox | feat1 | 4/4 | 1.16536 |
| redfox | dom | 4/4 | 0.86771 |
| redfox | pay | 4/4 | 0.31445 |

- fra: 0 crossings use the no-edit origin as the lower bracket.
- feat1: 0 crossings use the no-edit origin as the lower bracket.
- dom: 0 crossings use the no-edit origin as the lower bracket.
- pay: 12 crossings use the no-edit origin as the lower bracket.

## At 70% removal

Each method's mean uses only its reached cases. When coverage differs,
use the common-subset comparison below to compare FRA against feat1.

| Method | Reached | Mean worst KL, interpolated | Mean worst payload drop | Measured ≥ threshold: mean best worst KL |
|---|---:|---:|---:|---:|
| fra | 2/12 | 0.48698 | 0.49408 | 0.49170 |
| feat1 | 12/12 | 1.10685 | 0.74679 | 1.35035 |
| dom | 12/12 | 1.55790 | 0.67893 | 2.08834 |
| pay | 12/12 | 0.42618 | 0.37145 | 0.60476 |
| oracle (removal only) | 1/12 | — | — | — |

On the 2 cases reached by both FRA and feat1, FRA has lower
interpolated worst KL in 2 cases.

On this common subset, mean worst KL is 0.48698
for FRA and 1.17598 for feat1.

| Group | Method | Reached | Mean worst KL, interpolated |
|---|---|---:|---:|
| bluemoon | fra | 0/4 | — |
| bluemoon | feat1 | 4/4 | 1.24865 |
| bluemoon | dom | 4/4 | 1.05325 |
| bluemoon | pay | 4/4 | 0.43132 |
| irongate | fra | 0/4 | — |
| irongate | feat1 | 4/4 | 0.92200 |
| irongate | dom | 4/4 | 2.18800 |
| irongate | pay | 4/4 | 0.40698 |
| redfox | fra | 2/4 | 0.48698 |
| redfox | feat1 | 4/4 | 1.14991 |
| redfox | dom | 4/4 | 1.43244 |
| redfox | pay | 4/4 | 0.44023 |

- fra: 0 crossings use the no-edit origin as the lower bracket.
- feat1: 0 crossings use the no-edit origin as the lower bracket.
- dom: 0 crossings use the no-edit origin as the lower bracket.
- pay: 12 crossings use the no-edit origin as the lower bracket.

