# Note 01 — Setup and stage 1 (one-stage OV attribution)

## What

We want to attribute the pre-activation of SAE_mid feature 171 at `blocks.0.hook_resid_mid` — which the ablation sweep identified as a "perfect sleeper suppressor" (test ASR 0.99 → 0.00) — to upstream features at `blocks.0.ln1.hook_normalized`.

The scalar we're decomposing is

$$
\mathrm{pre}_{171}(q) \;=\; \langle e_{171},\, r^\mathrm{mid}_{\ell, q}\rangle + \text{const}
\;=\; \underbrace{\langle r^\mathrm{pre}_q, d\rangle}_\text{skip}
\;+\; \underbrace{\langle \mathrm{attn\_out}_q, d\rangle}_\text{attention}
\;+\; \text{const}.
$$

with $d = e_{171}$. The skip term is a direct dot product; the attention term is where all the structure is.

## Decomposition

Using the TL convention + observed attention $A$:

$$
\langle \mathrm{attn\_out}_q, d\rangle \;=\; \sum_h \sum_k A^h_{qk}\, \langle x_k, u_h\rangle + \text{const},
\qquad u_h := W_V^h W_O^h d.
$$

With SAE_ln1 basis $x_k \approx \sum_\lambda z^\lambda_k f_\lambda + b^\mathrm{dec}_\mathrm{ln1}$:

$$
\boxed{\;
\langle \mathrm{attn\_out}_q, d\rangle \;\approx\; \sum_{h, k, \lambda} A^h_{qk}\, z^\lambda_k\, \beta_{h, \lambda} + \text{const},
\quad \beta_{h, \lambda} := \langle f_\lambda, u_h\rangle.
\;}
$$

This is **exact given observed $A$**; the only error is SAE_ln1 reconstruction.

## Ranking method under this view

Sum over $(h, k)$ on deployment prompts, with L1 or max over head ranking:

$$
C^\mathrm{OV}[\lambda] = \sum_h \beta_{h, \lambda} \cdot \mathbb{E}_{b, q\in\text{prompt}(b)}\Big[\sum_k A^h_{qk}\, z^\lambda_k\Big]
$$

## What this told me

Computing the full $\beta$ matrix and one-stage attribution on the test set:

- Reconstruction sanity: $\|e \cdot \mathrm{attn\_out} - \sum_h S_h(q) - \text{const}\|_\mathrm{rel} = 1.8 \times 10^{-4}$. So the algebra is exact up to fp16.
- β reconstruction with SAE_ln1: relative error 10.8% (pure SAE dict error, no approximation beyond that).
- Top $\lambda$ by $|\sum_h C^\mathrm{OV}[\lambda]|$: 1205, 1114, 337, 1191, 865, 221, 628, 157, 870, 554 …

## What I expected at this stage

Naively I expected the top-ranked λ's under this decomposition to predict which features would, if ablated, most disrupt sleeper behaviour. The reasoning: if $\lambda$ contributes most to the target scalar, ablating it should disrupt the target most.

## What actually happened

See [03_ov_concentration_finding.md](03_ov_concentration_finding.md) and [04_ablation_validation.md](04_ablation_validation.md). The ranking by signed sum is **anti-predictive** of ablation impact (Spearman $\rho = -0.39$). $\lambda = 1205$ at rank 1 has **zero** ablation effect.

## Artifact

- [../../scripts/ov_path.py](../../scripts/ov_path.py) — computes $\beta$ and per-(h, $\lambda$) aggregated sums.
- [../../results/ov_path_per_pair.pt](../../results/ov_path_per_pair.pt) — raw $\beta$, per-head per-$\lambda$ contributions.

## What I'd do next at this stage

Before seeing the ablation data, I would have said "run single-feature ablation on the top-5 and confirm." Which is what we did, and it broke the prediction — see notes 3 and 4.
