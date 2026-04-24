# QK vs OV attribution — high-level summary

Question: given a downstream target scalar $T = \langle \mathrm{attn\_out}_q, d\rangle$
with $d = e_{171}$ the SAE_mid suppressor's encoder direction, *which features at
`blocks.0.ln1.hook_normalized` should we ablate to break sleeper behaviour?*

Two decompositions are mathematically natural:

- **OV-side** (stage 1+2 in [TWO_STAGE_WRITEUP.md](../TWO_STAGE_WRITEUP.md)): hold
  attention pattern $A$ fixed as observed, expand $T$ into per-feature atoms
  through $\beta_{h, \lambda} = \langle f_\lambda, u_h\rangle$. Exact (up to SAE
  recon error). Single-feature ablation prediction lives in summing
  $A^h_{qk}\cdot z^\lambda_k\cdot \beta_{h, \lambda}$ per $\lambda$.

- **QK-side** (softmax-Jacobian linearisation, see
  [working_notes/05_qk_side_derivation.md](working_notes/05_qk_side_derivation.md)):
  hold OV fixed by computing per-source scalars $g^h_k = \sum_\lambda z^\lambda_k
  \beta_{h, \lambda}$ and their centered version $\tilde g^h_{q, k} = A^h_{qk}(g^h_k -
  \bar g^h_q)$. Linearise attention's response to a query-feature perturbation
  via the softmax Jacobian. Single-feature ablation prediction is
  $\delta T_q \approx -u^\mu_q \sum_h \sum_j \kappa^{h, \mu}_j \, \tilde g^h_{q, j}$
  where $\kappa^{h, \mu}_j = \sum_\nu u^\nu_j\, \omega^{h, QK}_{\mu, \nu}$ and
  $\omega^{h, QK}_{\mu, \nu} = (f_\mu W_Q^h)\cdot(f_\nu W_K^h)/\sqrt{d_\text{head}}$.

## Headline

Three different questions, three different answers:

1. **"Which single ln1 features, when individually ablated, most change the
   teacher-forced sleeper log-prob?"** → **QK L1_mean wins.** Spearman $\rho = +0.95$
   for QK L1_mean across 28 ablated features, stable as the set grows. The OV-side
   signed sum is anti-correlated ($\rho = -0.39$).

2. **"Which small *set* of ln1 features, when ablated jointly at `ln1.hook_normalized`,
   gives the best Pareto tradeoff between sleeper suppression and clean-prompt coherence?"**
   → **OV-side wins decisively.** {1205, 1114, 337} hits ASR$_{16}$=0.02 at ΔCE=0 (α=3);
   every QK-top-$k$ channel needs ΔCE ≳ 2 to reach ASR near 0.

3. **"Can we do even better by choosing which computational pathway the perturbation
   travels?"** → **Yes: route the OV-selected features through the V pathway only
   (`hook_v`, not `ln1.hook_normalized`) and you get a *perfect* Pareto frontier —
   ASR$_{16}$=0.00 at ΔCE=0 (α=2 or α=3).** See
   [working_notes/10_3x3_rank_vs_intervene.md](working_notes/10_3x3_rank_vs_intervene.md).

### Why OV-rank + V-intervene is perfect

The dominant OV-ranked feature, f=1114, fires on 6.5% of deployment-prompt positions
and essentially **never on clean prompts** (0.0% fire rate, mean|z|=0.000). V-path
intervention scales the perturbation by $z^{1114}$, so on clean prompts it is
identically zero. On dep prompts where 1114 fires, the perturbation rewrites the
attention output's projection onto the suppressor direction $e_{171}$, flipping
sleeper emission. No attention pattern shift, no coherence leak.

### Quality scores (monotone-envelope Pareto area; 1 = perfect)

| ranking \ intervene | **OV-only** | **QK-only** | **All (ln1)** |
|---|---:|---:|---:|
| QK L1_mean top-3 {870, 1388, 760} | 0.59 | 0.07 | 0.38 |
| **OV one-stage top-3 {1205, 1114, 337}** | **1.00** | 0.18 | 0.98 |
| Union (6 features) | 0.93 | 0.14 | 0.85 |

**The practical takeaway:** FRA attribution serves two distinct purposes — *ranking*
features and *intervening* at a pathway. Conflating them hides the best recipe.
For defensive feature steering: rank by OV signed-sum on the target distribution,
intervene at the V hook only.

**Update after extending the test to 28 ablated features**: the naive "max over (b, q)"
concentration statistic that initially gave $\rho = +0.87$ on 10 features dropped to
$\rho = +0.56$ on 28, because it over-rewards features that spike at a single prompt
position. The **L1_mean** (absolute signed contribution, averaged across dep prompt
positions) is the robust aggregation at $\rho = +0.95$.

This is the inverse of what the naive decomposition suggests: although both
stages are algebraically exact, they answer different questions. Stage 1 says
"if the pattern $A$ stayed the same and only one feature's OV write were removed,
how much would $T$ change?" Stage 2-dual says "if one feature at the ln1 input
moved, how much would the pattern (hence $T$) change?" Feature ablation
perturbs ln1_normalized directly, so it touches $W_Q, W_K, W_V$ simultaneously,
but the softmax non-linearity amplifies the Q/K pathway effect disproportionately.

### Concrete ranking under each method

Spearman correlation with measured $|\Delta \log p(\text{sleeper})|$ at $\alpha=4$
across 10 ablated ln1 features:

| method | $\rho$ |
|---|---:|
| OV-side signed sum (≈ one-stage OV) | **−0.39** |
| OV-side L1 over $h$ | −0.10 |
| OV-side L1 over $(h, a)$ (two-stage) | +0.70 |
| OV-side max over $h$ | +0.55 |
| OV-side max over $(h, a)$ (two-stage) | +0.86 |
| QK-side signed_mean (≈ one-stage QK) | +0.82 (at n=28; was +0.93 at n=10) |
| QK-side L1 mean | **+0.95 (stable across n=10 → n=28)** |
| QK-side max over $(b, q)$ | +0.56 (at n=28; was +0.87 at n=10 — degraded) |
| QK-side top-50 L1 over $(b, q)$ | +0.91 (stable) |

**Top-2 features under QK-side L1-mean are exactly the top-2 by measured
ablation impact**: $\lambda = 870$ (rank 1, measured $\Delta \log p = -50.4$)
and $\lambda = 1388$ (rank 2, measured $-28.9$). The sweep-found $f=1412$ is at
rank 668, consistent with its zero causal effect. Nothing needs extra aggregation
tricks — the naive mean works.

### Concrete causal intervention ranking

| $\lambda$ | ablation $|\Delta \log p|$ at $\alpha=4$ | QK L1 rank | OV two-stage max rank |
|---:|---:|---:|---:|
| 870 | 50.4 | **1** | 1 |
| 1388 | 28.9 | **2** | 2 |
| 1114 | 7.8 | 47 | 42 |
| 221 | 1.2 | 93 | 17 |
| 1412 (sweep pick) | 0.00 | 668 | 629 |
| 1191 | 0.00 | 1367 | 1242 |
| 337 | 0.00 | 1151 | 185 |
| 865 | 0.00 | 1156 | 1444 |
| 1205 | 0.00 | 1386 | 1369 |
| 157 | 0.00 | 1381 | 814 |

## Practical recipe (for anyone else doing this kind of attribution)

### If you want to rank single features by their effect on a target logit

Use the QK-side sensitivity (applies at any hook that feeds both Q/K and V, such as `ln1.hook_normalized` or earlier):

1. For each query-feature $\mu$, per-prompt predicted $\delta T$:
   $$\text{pred}[\mu](b, q) = u^\mu_q(b) \cdot \sum_h \sum_j \kappa^{h, \mu}_j(b) \cdot \tilde g^h_{q, j}(b)$$
2. Take L1 over dep prompt positions (L1_mean).
3. The per-feature score correlates at $\rho=+0.95$ with measured $|\Delta \log p|$ on the target at $\alpha=4$.

### If you want to pick features to *jointly ablate* with low coherence cost

Use the OV-side one-stage signed sum on dep prompts, ranked by magnitude. Take the top-$k$ (k=3 worked here) and steer jointly. This wins the Pareto test by an order of magnitude over QK-selected features.

**The QK-side is the right predictor for single-feature causal impact on a target logit, but NOT the right selector for defensive joint steering.** QK-selected features move attention patterns a lot — which changes clean-prompt generation too. OV-selected features target the suppressor direction directly on the deployment distribution. See [working_notes/09_pareto_ov_vs_qk.md](working_notes/09_pareto_ov_vs_qk.md).

For the symmetric key-side or pair analysis, see the formulas in [working_notes/05_qk_side_derivation.md](working_notes/05_qk_side_derivation.md).

## What's in this subfolder

- [SUMMARY.md](SUMMARY.md) — this file.
- [working_notes/01_setup_and_stage_1.md](working_notes/01_setup_and_stage_1.md) — setup, identity, one-stage OV.
- [working_notes/02_two_stage_ov.md](working_notes/02_two_stage_ov.md) — two-stage OV decomposition, $\gamma$.
- [working_notes/03_ov_concentration_finding.md](working_notes/03_ov_concentration_finding.md) — OV signed sum is anti-predictive; concentration works.
- [working_notes/04_ablation_validation.md](working_notes/04_ablation_validation.md) — the measured ln1-feature ablation results.
- [working_notes/05_qk_side_derivation.md](working_notes/05_qk_side_derivation.md) — softmax-Jacobian linearisation, the QK dual.
- [working_notes/06_qk_vs_ov_comparison.md](working_notes/06_qk_vs_ov_comparison.md) — head-to-head Spearman correlations.
- [working_notes/07_next_steps.md](working_notes/07_next_steps.md) — open questions and what to test next.
- [working_notes/08_extended_ablation_findings.md](working_notes/08_extended_ablation_findings.md) — n=28 ablation + QK pair concentration.
- [working_notes/09_pareto_ov_vs_qk.md](working_notes/09_pareto_ov_vs_qk.md) — Pareto steering evaluation: OV one-stage top-3 beats QK-top-$k$ at ln1.
- [working_notes/10_3x3_rank_vs_intervene.md](working_notes/10_3x3_rank_vs_intervene.md) — **3x3 grid: (OV rank, V-intervene) is a *perfect* Pareto frontier.**
- `results/` — JSON/PT outputs and plots.

## Current experiments — completed

1. **Ablated 18 more top-QK-concentration features**: 254, 822, 760, 1298, 784, 1279, 435, 1102, 1489, 1008, 303, 200, 842, 500, 262, 608, 691, 59. Result: the "max concentration" ranking over-selected features with single-prompt-position spikes but small mean signal. L1_mean (average absolute contribution) is the stable predictor. See [working_notes/08_extended_ablation_findings.md](working_notes/08_extended_ablation_findings.md).

2. **Computed per-pair $(\mu, \nu)$ signed aggregates** for the top 15 query-features. Findings: μ=870 pairs with ν=221 (clean deployment marker), μ=1388 pairs with ν=1114, and many other top-μ's form "self-matching" pairs (μ=ν). The key-side distribution is fairly diffuse — no single dominant ν pair per μ. See [results/qk_pair_concentration.json](results/qk_pair_concentration.json).

3. **Pareto steering evaluation**: 8 channels (sweep, random, OV one-stage 1-feat & 3-feat, QK top-1/2/3/5) × 4 $\alpha \in \{0.5, 1, 2, 3\}$. **OV one-stage top-3 {1205, 1114, 337} hits ASR$_{16}$=0.02 at $\Delta$CE $\approx 0$**, dominating every QK channel (which all require $\Delta$CE $\gtrsim 2$ to suppress). Monotone-envelope area quality score: OV-3 = 0.98, QK-top-5 = 0.51, QK-top-1 = 0.49. See [working_notes/09_pareto_ov_vs_qk.md](working_notes/09_pareto_ov_vs_qk.md) and [results/pareto.png](results/pareto.png).

4. **3x3 ranking × intervention grid**: disentangling which features to select (QK, OV, union) from which computational pathway to perturb (V-only, Q/K-only, all). **(OV rank, V-intervene) achieves quality=1.00 — ASR$_{16}$=0.00 at $\Delta$CE=0 at α=2.** Strictly better than (OV rank, All)=0.98. Firing-stats diagnostic: the dominant OV feature f=1114 fires on 6.5% of dep positions and 0.0% of clean — so V-intervention's effect is structurally zero on clean prompts. See [working_notes/10_3x3_rank_vs_intervene.md](working_notes/10_3x3_rank_vs_intervene.md), [results/pareto_3x3.png](results/pareto_3x3.png).

Results all in `qk_vs_ov/results/`.
