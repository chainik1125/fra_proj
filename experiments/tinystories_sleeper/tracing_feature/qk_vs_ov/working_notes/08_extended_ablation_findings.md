# Note 08 — Extended ablation (18 more features) and pair concentration

## What I did

Per the action items from [07_next_steps.md](07_next_steps.md):

1. **Ablated 18 more ln1 features** chosen from the top-20 by QK `max over (b, q)` concentration that had *not* yet been tested. Now $n=28$ features have measured $|\Delta \log p|$ at $\alpha=4$.

2. **Computed per-pair $(\mu, \nu)$ signed aggregates** for the top query-features, using the factorisation in [../../scripts/qk_pair_concentration.py](../../scripts/qk_pair_concentration.py).

## Headline update: which QK aggregation is robust?

Re-running the Spearman analysis with all 28 features:

| aggregation | $\rho$ (10 features) | $\rho$ (28 features) |
|---|---:|---:|
| QK signed_mean (≈ one-stage QK) | +0.93 | **+0.82** |
| **QK L1_mean** | **+0.95** | **+0.95** |
| QK max over (b, q) | +0.87 | **+0.56** |
| QK top-50 L1 | +0.94 | +0.91 |

**L1_mean is the robust predictor.** The `max over (b, q)` concentration dropped from $+0.87$ to $+0.56$ when we added more features: many features that were "concentrated at one prompt position" (big `max`) turned out to have small ablation impact because the concentration didn't translate into sustained signal across many dep prompts.

`QK L1_mean` — the per-prompt-position absolute predicted $\delta T$, averaged — is essentially the concentration over (b, q) with the size of each peak weighted uniformly rather than max-reduced.

## Per-feature breakdown (28 features, sorted by $|\Delta \log p|$)

| $\mu$ | $\|\Delta \log p\|$ at α=4 | L1_mean rank | max rank | Notes |
|---:|---:|---:|---:|---|
| 870  | 50.4 | **1** | 5 | FRA star, unchanged |
| 1388 | 28.9 | **2** | 19 | FRA star #2 |
| **760** | **21.2** | **3** | 3 | **New!** huge effect |
| **435** | **14.1** | 9 | 8 | **New!** large effect |
| **303** | **12.1** | 5 | 12 | **New!** large effect |
| **254** | 8.1 | 8 | 1 | **New!** moderate (note max rank 1 but L1 lower) |
| 1114 | 7.8 | 47 | 643 | OV one-stage rank 1 |
| **200** | 5.7 | 17 | 13 | **New!** moderate |
| 221  | 1.2 | 93 | 739 | |
| **822** | 1.0 | 44 | 2 | **New!** small effect despite max rank 2 |
| 691  | 0.7 | 51 | 18 | |
| **1298** | 0.7 | 274 | 4 | **New!** max rank 4 but ablation nearly zero |
| 262  | 0.6 | 295 | 16 | |
| **1102** | 0.5 | 197 | 9 | **New!** |
| 842  | 0.3 | 63 | 14 | |
| 500  | 0.2 | 260 | 15 | |
| **1489** | 0.2 | 677 | 10 | **New!** max rank 10 but ablation tiny |
| 59   | 0.10 | 619 | 20 | |
| **784** | 0.06 | 1032 | 6 | **New!** max rank 6 but ablation tiny |
| 608  | 0.02 | 117 | 17 | |
| 1008 | 0.01 | 547 | 11 | |
| **1279** | 0.004 | 477 | 7 | **New!** max rank 7 but ablation zero |
| 1412 | 0 | 668 | 1187 | sweep's pick |
| 1191, 337, 865, 1205, 157 | 0 | 1000+ | 700–1400 | all causally inert |

## What this told me

1. **QK L1_mean is reliable.** Rank 1, 2, 3, 5, 8, 9 → all substantial ablation effects (870, 1388, 760, 303, 254, 435). The untested ranks 4, 6, 7, 10 are the natural next candidates.

2. **QK max concentration is a bad predictor on its own.** Features 254 (max rank 1, $|\Delta \log p| = 8$), 822 (max rank 2, $|\Delta \log p| = 1$), 1298 (max rank 4, $|\Delta \log p| = 0.7$), 784 (rank 6, $|\Delta \log p| = 0.06$), 1279 (rank 7, ≈0), 1489 (rank 10, ≈0): the max statistic picked up features that have one big peak at one token position but are otherwise quiet, and these don't translate to sustained ablation impact.

3. **Previously "concentration beats sum" was a lucky finding on 10 features.** The original top-10 happened to have both `max` and `L1_mean` agreeing on 870 and 1388. When we expanded, the `max` metric's false positives became visible while `L1_mean` remained stable.

## Per-pair $(\mu, \nu)$ findings (from [../../scripts/qk_pair_concentration.py](../../scripts/qk_pair_concentration.py))

Dominant $\nu$ partners of the top query features:

| $\mu$ | total_signed | $L1$ | dominant $\nu$ |
|---:|---:|---:|---:|
| 870 | +9.65 | 65.9 | 221 |
| 1388 | +11.33 | 63.6 | 1114 |
| 760 | +2.37 | 24.4 | **870** (self-referential: query=760 pulls attention to where ν=870 fires) |
| 303 | +3.09 | 19.0 | 221 |
| 200 | −2.42 | 16.3 | 200 (also self-referential) |
| 435 | −0.23 | 15.0 | 435 |
| 254 | +0.29 | 12.3 | 254 |

Observations:
- **870 ↔ 221** is the top QK pair for μ=870. 221 is (one of) the clean deployment-marker detectors we identified earlier, firing at the ` |` and `MENT` subtokens.
- **1388 ↔ 1114** is the second-top pair.
- **Self-matching pairs** (μ=ν) appear for 254, 435, 200, 1298, 784, 1279, 1102, 1489 — the features have their own encoder vector projecting strongly onto the Q and K circuits, so the score matrix has a diagonal bump. This is a characteristic of "single-token specialist" features that both detect and attend to the same structure.
- **Concentration ratio (max/L1) is low** (0.04–0.16 for top-μ's) — suggests the key-side distribution is quite spread out; the QK contribution is distributed across many $\nu$'s even when the μ side is focused.

## Big picture update

The story is still:
- **QK L1_mean is the right attribution signal for predicting ln1-feature ablation at α=4**, with $\rho = +0.95$ over 28 measured features.
- **OV-side attribution is insufficient**; the softmax-Jacobian linearisation on the QK side captures the ablation mechanism much better.
- **Simple "max concentration" alone was a lucky coincidence on the initial 10 features.** L1_mean (= concentration *averaged* across prompt positions) is what actually generalises.

## Artifacts

- [../results/ln1_feature_ablation.json](../results/ln1_feature_ablation.json) — new 18-feature ablation
- [../results/qk_pair_concentration.json](../results/qk_pair_concentration.json) — per-pair (μ, ν) contributions
- [../results/final_qk_analysis.json](../results/final_qk_analysis.json) — unified Spearman with 28 features

## What I'd do next

1. **Ablate the untested features at L1_mean ranks 4, 6, 7, 10** to complete the "QK L1_mean predicts ablation" test. Should all give substantial $|\Delta \log p|$ if the claim holds.

2. **Investigate why 1114 is mis-predicted**. L1_mean rank 47, measured $|\Delta \log p| = 7.8$ — not tiny, not huge. What does its per-(b, q) predicted contribution look like — is it one of those "diffuse but sustained" features that L1_mean underweights?

3. **Try ablation on a completely different task** (not this sleeper model / not this SAE) to see if QK L1_mean is a general-purpose predictor for feature ablation at $\mathrm{ln1}$-type hooks.
