# Note 06 — Head-to-head: QK vs OV attribution

## What I did

Computed the per-$\mu$ QK sensitivity using Note 05's formula, aggregated across deployment prompt positions, and ran Spearman $\rho$ against measured $|\Delta \log p|$ at $\alpha = 4$ for the 10 ablated ln1 features.

Script: [../../scripts/qk_concentration.py](../../scripts/qk_concentration.py).

## Results

| aggregation | OV-side $\rho$ | **QK-side $\rho$** |
|---|---:|---:|
| signed sum / mean (≈ one-stage) | −0.39 | **+0.93** |
| L1 | −0.10 / +0.70 | **+0.95** |
| max over relevant axis | +0.55 / +0.86 | +0.87 |
| top-K L1 | +0.87 | +0.94 |

QK-side columns: aggregation is over `(b, q)` for dep prompt positions.
OV-side columns: aggregation is over $h$ alone (one-stage) or $(h, a)$ (two-stage).

## Per-feature ranks

| $\lambda$ | $|\Delta \log p|$ | QK L1 rank | OV two-stage max rank |
|---:|---:|---:|---:|
| 870 | 50.4 | **1** | 1 |
| 1388 | 28.9 | **2** | 2 |
| 1114 | 7.8 | 47 | 42 |
| 221 | 1.2 | 93 | 17 |
| 1412 (sweep) | 0 | 668 | 629 |
| 1191 | 0 | 1367 | 1242 |
| 337 | 0 | 1151 | 185 |
| 865 | 0 | 1156 | 1444 |
| 1205 | 0 | 1386 | 1369 |
| 157 | 0 | 1381 | 814 |

## Observations

1. **Even the naive QK signed-mean beats any OV aggregation for correlation with ablation.** $\rho = +0.93$ vs the best OV-side of $+0.87$ (which required concentration filtering). And the QK signed-mean needs no filtering.

2. **QK L1-mean is the best single method.** $\rho = +0.95$, ranks 870 and 1388 at positions 1 and 2 exactly matching the ablation ordering.

3. **OV-side concentration (max over (h, a), $\rho = +0.86$) was a lucky proxy.** It happened to rank 870 and 1388 at top because those features are *also* concentrated on the OV side. But the causal mechanism of ln1-feature ablation is really mediated through the QK pathway, so it's not surprising that OV-concentration agrees only approximately.

## Why QK dominates for this ablation

When you ablate a feature at `ln1.hook_normalized` (i.e. zero its contribution to the ln1 normalised tensor), you perturb the input to **all three** of $W_Q, W_K, W_V$ simultaneously. The downstream effect is:
- A change in $V$ that directly perturbs the OV write → captured by stage 1.
- A change in $Q, K$ that reshuffles the attention pattern → captured by stage 2-dual.

The second effect dominates empirically: the pattern shift has compound amplification through the softmax + the large current attention weights, whereas the V-side change is a single linear perturbation. For features like $\lambda = 870$ that strongly activate the QK mechanism ($\kappa^{h, 870}_j$ is big, especially aligned with $\tilde g^h_{q, j}$), ablating them produces a big pattern shift and therefore a big $\delta T$.

Features with large OV contribution but small QK sensitivity (e.g. $\lambda = 1205$) do not perturb the pattern much when ablated, so $\delta T$ is small even though the OV attribution is large. This is exactly the discrepancy that made OV signed-sum anti-predictive.

## What I'd do next at this stage

Verify the QK concentration hypothesis by ablating more features that have high QK-side L1 mean / max concentration but that **haven't been tested yet**. If the QK story is real, they should all produce large $\Delta \log p$ when ablated. Candidates from the top-20 list: 254, 822, 760, 1298, 784, 1279, 435, 1102, 303, 200.

Also: compute the full pair $(\mu, \nu)$ attribution, and see whether the dominant pairs correspond to interpretable circuits (e.g. query-feature 870 paired with key-feature 1220 at $(h, k) = (7, 2)$ — which we partially already have from Lens 6).
