# Note 05 — QK-side derivation (fixed OV, variable attention)

## What I did

Worked out the dual of the one-stage OV decomposition: hold the OV side fixed, let the attention pattern $A$ vary via the QK mechanism, and ask how feature variation on the Q/K side perturbs the target $T_q = \langle \mathrm{attn\_out}_q, d\rangle$.

## Setup

With OV fixed, define the per-source OV scalar:
$$
g^h_k \;:=\; \sum_\lambda z^\lambda_k \beta_{h, \lambda}.
$$
Then
$$
T_q \;=\; \sum_h \sum_k A^h_{qk}\, g^h_k + \text{const}.
$$

**$A^h_{qk}$** = softmax over $k$ of the scores $s^h_{qk} = (x_q W_Q^h) \cdot (x_k W_K^h) / \sqrt{d_\text{head}}$. Scores are linear in features; the pattern is not.

## Score-level feature-pair decomposition (exact)

$$
s^h_{qk} \;=\; \sum_{\mu, \nu} u^\mu_q\, u^\nu_k\, \omega^{h, \mathrm{QK}}_{\mu, \nu} + \text{const},
\qquad \omega^{h, \mathrm{QK}}_{\mu, \nu} := \frac{(f_\mu W_Q^h) \cdot (f_\nu W_K^h)}{\sqrt{d_\text{head}}}.
$$

This is clean, but it doesn't directly decompose the pattern $A$ or the target $T_q$ because of the softmax.

## Softmax Jacobian linearisation (first-order)

$$
\frac{\partial A^h_{qk'}}{\partial s^h_{qj}} = A^h_{qk'}(\mathbb{1}_{k'=j} - A^h_{qj}).
$$

Combined with the target $T_q = \sum_h \sum_k A^h_{qk} g^h_k$:

$$
\frac{\partial T_q}{\partial s^h_{qj}} = A^h_{qj}(g^h_j - \bar g^h_q), \qquad \bar g^h_q := \sum_k A^h_{qk} g^h_k.
$$

Defining the **centered OV profile**
$$
\boxed{\;\tilde g^h_{q, j} := A^h_{qj}\,(g^h_j - \bar g^h_q),\; \sum_j \tilde g^h_{q, j} = 0.\;}
$$

## Single query-feature sensitivity

Zero query-feature $\mu^\star$ at position $q$: $u^{\mu^\star}_q \to 0$. Score shift at every key:
$$
\delta s^h_{qj} = -u^{\mu^\star}_q \cdot \underbrace{\sum_\nu u^\nu_j \omega^{h, \mathrm{QK}}_{\mu^\star, \nu}}_{=: \kappa^{h, \mu^\star}_j}.
$$

First-order target change:
$$
\boxed{\;
\delta T_q \approx -u^{\mu^\star}_q \cdot \sum_h \sum_j \kappa^{h, \mu^\star}_j \cdot \tilde g^h_{q, j}.
\;}
$$

## Single key-feature sensitivity

Zero key-feature $\nu^\star$ at position $k^\star$. Only score at $j = k^\star$ changes:
$$
\delta s^h_{qk^\star} = -u^{\nu^\star}_{k^\star} \cdot \underbrace{\sum_\mu u^\mu_q \omega^{h, \mathrm{QK}}_{\mu, \nu^\star}}_{=: \eta^{h, \nu^\star}_q}.
$$

$$
\boxed{\;
\delta T_q \approx -u^{\nu^\star}_{k^\star} \cdot \sum_h \eta^{h, \nu^\star}_q \cdot \tilde g^h_{q, k^\star}.
\;}
$$

## Pair $(\mu, \nu)$ sensitivity

The score term at $(q, k)$ is $u^\mu_q u^\nu_k \omega^{h, \mathrm{QK}}_{\mu, \nu}$ — ablating both features jointly zeroes it. Effect:
$$
\boxed{\;
\delta T_q \approx -u^\mu_q\, u^\nu_k\, \omega^{h, \mathrm{QK}}_{\mu, \nu} \cdot \tilde g^h_{q, k}.
\;}
$$

Each factor is interpretable:
- $u^\mu_q, u^\nu_k$: activations of the features at query/key positions.
- $\omega^{h, \mathrm{QK}}_{\mu, \nu}$: weight-space pair coupling (how much does this pair move the score through head $h$).
- $\tilde g^h_{q, k}$: OV sensitivity — how much moving attention toward $k$ moves $T$.

## Per-feature aggregation for ranking

For each $\mu$, per-prompt:
$$
\text{pred}[\mu](b, q) \;=\; u^\mu_q(b) \cdot \sum_h \sum_j \kappa^{h, \mu}_j(b) \cdot \tilde g^h_{q, j}(b).
$$

Aggregate over dep prompt positions (mean signed, mean absolute, max, top-K).

## Caveats

1. **First-order in softmax.** Ablating $u^\mu_q$ from 3.0 to 0.0 is a large perturbation; the softmax response is non-linear and the first-order formula can mis-estimate the actual $\delta T$ by a non-trivial factor. But for ranking, first-order usually works well enough.

2. **Feature ablation perturbs both Q/K AND V simultaneously** — this formula ignores the V-side effect (we've held it fixed in $g^h_k$). Whether that's the right thing to do depends on what you're comparing against.

3. **Centering $\tilde g^h_{q, k}$ is what makes the formula non-trivial.** A head with uniform $g^h_k$ across sources has $\tilde g^h_{q, k} = 0$, and varying any QK feature doesn't change $T$ through that head (the mass reshuffles between equal-value sources).

## What I'd do next at this stage

Compute $\text{pred}[\mu]$ on deployment prompts, aggregate by mean / L1 / max, and compare to the ablation $|\Delta \log p|$ from Note 04. This is Note 06.
