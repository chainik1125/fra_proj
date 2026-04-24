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

**When you ablate an ln1 feature by adding $\alpha\cdot(\mathrm{decode}(z_{\text{abl}}) - \mathrm{decode}(z))$
at `blocks.0.ln1.hook_normalized`, the effect on sleeper log-prob is dominated by
the QK-side sensitivity, not the OV-side contribution.** The QK-side attribution
correlates strongly with measured ablation impact (Spearman $\rho = +0.95$
for L1-mean across 28 ablated features, and robust as the feature set grows); the
OV-side signed-sum is *anti*-correlated ($\rho = -0.39$) and only recovers
predictive power through concentration aggregations.

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

If your goal is to **identify single features at a hook whose ablation changes a downstream scalar**, compute the QK-side sensitivity if the hook feeds both into Q/K and into V (which is any hook at `ln1.hook_normalized` or earlier). Specifically:

1. For each query-feature $\mu$, per-prompt predicted $\delta T$:
   $$\text{pred}[\mu](b, q) = u^\mu_q(b) \cdot \sum_h \sum_j \kappa^{h, \mu}_j(b) \cdot \tilde g^h_{q, j}(b)$$
2. Average (or take L1) over dep prompt positions.
3. Rank $\mu$'s and ablate the top ones.

For the symmetric key-side or pair analysis, see the formulas in [working_notes/05_qk_side_derivation.md](working_notes/05_qk_side_derivation.md).

If your hook is at an OV-only location (i.e. after Q and K have been computed, so the ablation only affects V), you'd use the OV-side attribution instead. But for `ln1.hook_normalized` — the hook we're ablating at — the QK-side dominates.

## What's in this subfolder

- [SUMMARY.md](SUMMARY.md) — this file.
- [working_notes/01_setup_and_stage_1.md](working_notes/01_setup_and_stage_1.md) — setup, identity, one-stage OV.
- [working_notes/02_two_stage_ov.md](working_notes/02_two_stage_ov.md) — two-stage OV decomposition, $\gamma$.
- [working_notes/03_ov_concentration_finding.md](working_notes/03_ov_concentration_finding.md) — OV signed sum is anti-predictive; concentration works.
- [working_notes/04_ablation_validation.md](working_notes/04_ablation_validation.md) — the measured ln1-feature ablation results.
- [working_notes/05_qk_side_derivation.md](working_notes/05_qk_side_derivation.md) — softmax-Jacobian linearisation, the QK dual.
- [working_notes/06_qk_vs_ov_comparison.md](working_notes/06_qk_vs_ov_comparison.md) — head-to-head Spearman correlations.
- [working_notes/07_next_steps.md](working_notes/07_next_steps.md) — open questions and what to test next.
- `results/` — JSON/PT outputs for the two experiments launched after this note.

## Current experiments — completed

1. **Ablated 18 more top-QK-concentration features**: 254, 822, 760, 1298, 784, 1279, 435, 1102, 1489, 1008, 303, 200, 842, 500, 262, 608, 691, 59. Result: the "max concentration" ranking over-selected features with single-prompt-position spikes but small mean signal. L1_mean (average absolute contribution) is the stable predictor. See [working_notes/08_extended_ablation_findings.md](working_notes/08_extended_ablation_findings.md).

2. **Computed per-pair $(\mu, \nu)$ signed aggregates** for the top 15 query-features. Findings: μ=870 pairs with ν=221 (clean deployment marker), μ=1388 pairs with ν=1114, and many other top-μ's form "self-matching" pairs (μ=ν). The key-side distribution is fairly diffuse — no single dominant ν pair per μ. See [results/qk_pair_concentration.json](results/qk_pair_concentration.json).

Results all in `qk_vs_ov/results/`.
