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
