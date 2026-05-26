# Feature attribution (diff regime)

Notation follows `paper/example_paper.tex`.

## Notation

$W^{\mathrm{dec}}_\lambda \in \mathbb{R}^{d_{\mathrm{model}}}$ — SAE decoder column $\lambda$. $f_t^\lambda$ — SAE activation of feature $\lambda$ at position $t$ in prompt $p$. $W_{OV}^h := W_V^h W_O^h$, $W_{QK}^h := W_Q^h (W_K^h)^\top$. $A^h_{qk}$ — attention pattern (frozen, from clean forward). Sums over positions $q, k$ run over all sequence positions; $f_t^\lambda \equiv 0$ outside the prompt. Quantities $f_t^\lambda, A^h_{qk}, M^h_p, Z^q_p, Z^k_p$ depend implicitly on prompt $p$.

## OV channel

Single source $k$, head $h$, prompt $p$, feature $\lambda$ — vector contribution to $r'_q$ (paper eq. \ref{eq:partial_decomp}; the paper's scalar $\mathrm{FRA}^{\mathrm{OV},h}_{qk,\lambda}$ is the norm of this):

$$
A^h_{qk}\,f_k^\lambda\,W^{\mathrm{dec}}_\lambda W_{OV}^h \;\in\; \mathbb{R}^{d_{\mathrm{model}}}
$$

Head, summed over $(q, k)$:

$$
\sum_{q, k} A^h_{qk}\,f_k^\lambda\,W^{\mathrm{dec}}_\lambda W_{OV}^h \;=\; \Big(\sum_{q, k} A^h_{qk}\,f_k^\lambda\Big)\,W^{\mathrm{dec}}_\lambda W_{OV}^h
$$

Layer (sum over $h$), dep − clean means, norm at end (heads cancel directionally before norming):

$$
\mathrm{score}_{\mathrm{OV}}(\lambda) \;=\; \bigg\lVert \sum_h \Big(\mathbb{E}_{p \sim \mathrm{dep}}\big[\textstyle\sum_{q,k} A^h_{qk}\,f_k^\lambda\,W^{\mathrm{dec}}_\lambda W_{OV}^h\big] - \mathbb{E}_{p \sim \mathrm{cln}}\big[\sum_{q,k} A^h_{qk}\,f_k^\lambda\,W^{\mathrm{dec}}_\lambda W_{OV}^h\big]\Big) \bigg\rVert_2
$$

$W^{\mathrm{dec}}_\lambda W_{OV}^h$ is prompt-independent, so factor it out:

$$
\mathrm{score}_{\mathrm{OV}}(\lambda) \;=\; \Big\lVert \sum_h \big(\mathbb{E}_{p \sim \mathrm{dep}}[M^h_p(\lambda)] - \mathbb{E}_{p \sim \mathrm{cln}}[M^h_p(\lambda)]\big)\,W^{\mathrm{dec}}_\lambda W_{OV}^h \Big\rVert_2, \qquad M^h_p(\lambda) := \sum_{q, k} A^h_{qk}\,f_k^\lambda
$$

## QK channel

Single $(q,k)$, head $h$, prompt $p$, feature pair $(\lambda, \mu)$ — contribution to $s^h_{qk}$:

$$
\mathrm{FRA}^{\mathrm{QK},h}_{qk,\lambda,\mu} \;=\; \frac{f_q^\lambda\,f_k^\mu}{\sqrt{d_{\mathrm{head}}}}\;W^{\mathrm{dec}}_\lambda W_{QK}^h (W^{\mathrm{dec}}_\mu)^\top \;\in\; \mathbb{R}
$$

Head, summed over $(q, k)$:

$$
\sum_{q, k} \mathrm{FRA}^{\mathrm{QK},h}_{qk,\lambda,\mu} \;=\; \Big(\sum_{q, k} f_q^\lambda\,f_k^\mu\Big)\,\frac{W^{\mathrm{dec}}_\lambda W_{QK}^h (W^{\mathrm{dec}}_\mu)^\top}{\sqrt{d_{\mathrm{head}}}}
$$

Layer (sum over $h$), dep − clean means, abs:

$$
\mathrm{score}_{\mathrm{QK}}(\lambda, \mu) \;=\; \bigg|\sum_h \Big(\mathbb{E}_{p \sim \mathrm{dep}}\big[\textstyle\sum_{q,k} f_q^\lambda f_k^\mu\big] - \mathbb{E}_{p \sim \mathrm{cln}}\big[\sum_{q,k} f_q^\lambda f_k^\mu\big]\Big)\,\tfrac{W^{\mathrm{dec}}_\lambda W_{QK}^h (W^{\mathrm{dec}}_\mu)^\top}{\sqrt{d_{\mathrm{head}}}} \bigg|
$$

The weight factor is prompt-independent and $\sum_{q,k} f_q^\lambda f_k^\mu = Z^q_p(\lambda)\,Z^k_p(\mu)$ separates; factor both out:

$$
\mathrm{score}_{\mathrm{QK}}(\lambda, \mu) \;=\; \bigg|\underbrace{\sum_h \frac{W^{\mathrm{dec}}_\lambda W_{QK}^h (W^{\mathrm{dec}}_\mu)^\top}{\sqrt{d_{\mathrm{head}}}}}_{\mathrm{QK}_{\mathrm{total}}(\lambda,\mu)} \cdot \big(\mathbb{E}_{p \sim \mathrm{dep}}[Z^q_p Z^k_p] - \mathbb{E}_{p \sim \mathrm{cln}}[Z^q_p Z^k_p]\big)\bigg|, \qquad Z^q_p(\lambda) := \sum_q f_q^\lambda, \quad Z^k_p(\mu) := \sum_k f_k^\mu
$$

## QK+OV channel

Single $(q,k)$, head $h$, prompt $p$, triplet $(\lambda, \mu, \nu)$ — QK pair logit weight times OV vector write at the same key:

$$
\frac{f_q^\lambda\,f_k^\mu\,f_k^\nu}{\sqrt{d_{\mathrm{head}}}}\;\big[W^{\mathrm{dec}}_\lambda W_{QK}^h (W^{\mathrm{dec}}_\mu)^\top\big]\;W^{\mathrm{dec}}_\nu W_{OV}^h \;\in\; \mathbb{R}^{d_{\mathrm{model}}}
$$

Head, summed over $(q, k)$:

$$
\sum_{q, k} \frac{f_q^\lambda\,f_k^\mu\,f_k^\nu}{\sqrt{d_{\mathrm{head}}}}\,\big[W^{\mathrm{dec}}_\lambda W_{QK}^h (W^{\mathrm{dec}}_\mu)^\top\big]\,W^{\mathrm{dec}}_\nu W_{OV}^h \;=\; \Big(\sum_{q,k} f_q^\lambda\,f_k^\mu\,f_k^\nu\Big)\,\frac{W^{\mathrm{dec}}_\lambda W_{QK}^h (W^{\mathrm{dec}}_\mu)^\top}{\sqrt{d_{\mathrm{head}}}}\,W^{\mathrm{dec}}_\nu W_{OV}^h
$$

Layer (sum over $h$), dep − clean means, norm at end:

$$
\mathrm{score}_{\mathrm{QK+OV}}(\lambda, \mu, \nu) \;=\; \bigg\lVert \sum_h \Big(\mathbb{E}_{p \sim \mathrm{dep}}\big[\textstyle\sum_{q,k} f_q^\lambda f_k^\mu f_k^\nu\big] - \mathbb{E}_{p \sim \mathrm{cln}}\big[\sum_{q,k} f_q^\lambda f_k^\mu f_k^\nu\big]\Big) \tfrac{W^{\mathrm{dec}}_\lambda W_{QK}^h (W^{\mathrm{dec}}_\mu)^\top}{\sqrt{d_{\mathrm{head}}}}\,W^{\mathrm{dec}}_\nu W_{OV}^h \bigg\rVert_2
$$

The weight factors are prompt-independent and $\sum_{q,k} f_q^\lambda f_k^\mu f_k^\nu = Z^q_p(\lambda)\,Y_p(\mu, \nu)$ separates; factor both out:

$$
\mathrm{score}_{\mathrm{QK+OV}}(\lambda, \mu, \nu) \;=\; \Big\lVert \sum_h \tfrac{W^{\mathrm{dec}}_\lambda W_{QK}^h (W^{\mathrm{dec}}_\mu)^\top}{\sqrt{d_{\mathrm{head}}}}\,W^{\mathrm{dec}}_\nu W_{OV}^h \cdot \big(\mathbb{E}_{p \sim \mathrm{dep}}[Z^q_p Y_p] - \mathbb{E}_{p \sim \mathrm{cln}}[Z^q_p Y_p]\big)\Big\rVert_2, \qquad Y_p(\mu, \nu) := \sum_k f_k^\mu\,f_k^\nu
$$

### Enumeration over all triplets

The Cartesian-product candidate heuristic (top-$K_q \times K_k \times K_v$ from per-channel marginals) is unnecessary in the diff regime. The full $d_{\mathrm{sae}}^3$ enumeration is tractable because:

**Sparsity.** With a TopK SAE ($k$ active features per position), $Y_p(\mu, \nu)$ is non-zero only when $\mu, \nu$ co-fire at the same key in prompt $p$. The support of $Y$ across the batch has $\sim k^2 \cdot B \cdot T$ slots before deduplication and is typically $\ll d_{\mathrm{sae}}^2$. Only $(\mu, \nu)$ pairs in $\mathrm{supp}(Y)$ can contribute to a non-zero score.

**Factored norm.** Let $S(\lambda, \mu, h) := W^{\mathrm{dec}}_\lambda W_{QK}^h (W^{\mathrm{dec}}_\mu)^\top / \sqrt{d_{\mathrm{head}}}$ and $V(\nu, :, h) := W^{\mathrm{dec}}_\nu W_{OV}^h \in \mathbb{R}^{d_{\mathrm{model}}}$. Then

$$
\Big\lVert\sum_h S(\lambda, \mu, h)\,V(\nu, :, h)\Big\rVert_2^2 \;=\; \sum_{h, h'} S(\lambda, \mu, h)\,S(\lambda, \mu, h')\,G(\nu, h, h'), \qquad G(\nu, h, h') := \big\langle V(\nu, :, h),\, V(\nu, :, h')\big\rangle
$$

avoiding any $d_{\mathrm{sae}}^3 \times d_{\mathrm{model}}$ materialization. $S$ is $(d_{\mathrm{sae}}, d_{\mathrm{sae}}, n_{\mathrm{heads}})$ and $G$ is $(d_{\mathrm{sae}}, n_{\mathrm{heads}}, n_{\mathrm{heads}})$.

**Algorithm.**

1. *Sparse $Y$*: for each $(p, k)$, iterate the $\le k$ firing features $F_{p,k}$; for each $\mu, \nu \in F_{p,k}$ accumulate $Y[(p, \mu, \nu)] \mathrel{+}= f_k^\mu f_k^\nu$.
2. *Data factor*: for each $(\mu, \nu) \in \mathrm{supp}(Y)$, compute $D[\lambda, \mu, \nu] = \sum_p \mathrm{sign}_p\, Z^q_p(\lambda)\, Y_p(\mu, \nu)$ as a $(d_{\mathrm{sae}},)$ vector over $\lambda$.
3. *Score*: for each $(\lambda, \mu, \nu)$ with $D \neq 0$, evaluate $\mathrm{score} = \sqrt{\sum_{h, h'} S(\lambda, \mu, h)\,S(\lambda, \mu, h')\,G(\nu, h, h')} \cdot |D[\lambda, \mu, \nu]|$. Track running top-$K$.

Compute is dominated by step 2 (a sparse contraction over $\mathrm{supp}(Y) \times d_{\mathrm{sae}}$); the entire procedure runs in seconds on a single GPU and replaces `generate_triplet_candidates` in the diff regime.
