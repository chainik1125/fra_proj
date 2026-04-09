---
author: Jamie Stephenson
date: 2026-04-03
tags:
  - reference
  - in-progress
---

## Setup

Let $\hat{x}^i \in \mathbb{R}^{d}$ be the SAE reconstruction of the residual stream at position $i$:

$$\hat{x}^i = W^{dec}\, u^i + b^{dec}$$

where $W^{dec} \in \mathbb{R}^{d \times F}$ is the SAE decoder matrix, $u^i \in \mathbb{R}^{F}$ is the sparse vector of SAE latent activations, and $b^{dec} \in \mathbb{R}^{d}$ is the SAE decoder bias.

The attention logit between query position $i$ and key position $j$ for a single head is:

$$\hat{A}_{ij} = \frac{1}{\sqrt{d_k}} \left( W_Q \frac{\hat{x}^i}{\mathrm{rms}(\hat{x}^i)} + b_Q \right)^T W_R^{(i-j)} \left( W_K \frac{\hat{x}^j}{\mathrm{rms}(\hat{x}^j)} + b_K \right)$$

where:

- $W_Q, W_K \in \mathbb{R}^{d_k \times d}$ are the query and key projection matrices (with RMSNorm $\gamma$ folded in),
- $b_Q, b_K \in \mathbb{R}^{d_k}$ are the query and key projection biases,
- $W_R^{(i-j)} \in \mathbb{R}^{d_k \times d_k}$ is the RoPE rotation matrix for relative position $(i - j)$,
- $\mathrm{rms}(\hat{x}^i) = \sqrt{\frac{1}{d}\sum_n (\hat{x}^i_n)^2}$ is the RMS norm denominator,
- $d_k$ is the head dimension.

## Substitution

Substituting the SAE reconstruction $\hat{x}^i = W^{dec}\, u^i + b^{dec}$ into the attention logit:

$$\hat{A}_{ij} = \frac{1}{\sqrt{d_k}} \left( \frac{W_Q (W^{dec}\, u^i + b^{dec})}{\mathrm{rms}(\hat{x}^i)} + b_Q \right)^T W_R^{(i-j)} \left( \frac{W_K (W^{dec}\, u^j + b^{dec})}{\mathrm{rms}(\hat{x}^j)} + b_K \right)$$

Distributing the linear maps, each side becomes a sum of three vectors. The bilinear form $(a + b + c)^T M (d + e + f)$ expands into nine cross-terms which group naturally by their dependence on the SAE feature activations $u^i$ and $u^j$.

## Decomposition by Dependence

$$\hat{A}_{ij} = \hat{A}_{ij}^{\text{bilinear}} + \hat{A}_{ij}^{\text{linear}} + \hat{A}_{ij}^{\text{bias}}$$

### Bilinear in features

$$\hat{A}_{ij}^{\text{bilinear}} = \frac{1}{\sqrt{d_k}} \cdot \frac{(W_Q W^{dec}\, u^i)^T\; W_R^{(i-j)}\; W_K W^{dec}\, u^j}{\mathrm{rms}(\hat{x}^i)\;\mathrm{rms}(\hat{x}^j)}$$

This is the only term that depends on the features at *both* positions. It captures which SAE latent at position $i$ attends to which SAE latent at position $j$, mediated by relative position through $W_R^{(i-j)}$.

### Linear in features

$$\hat{A}_{ij}^{\text{linear}} = \frac{1}{\sqrt{d_k}} \Bigg[ \frac{(W_Q W^{dec}\, u^i)^T}{\mathrm{rms}(\hat{x}^i)}\; W_R^{(i-j)} \left( \frac{W_K\, b^{dec}}{\mathrm{rms}(\hat{x}^j)} + b_K \right) + \left( \frac{(W_Q\, b^{dec})^T}{\mathrm{rms}(\hat{x}^i)} + b_Q^T \right) W_R^{(i-j)}\; \frac{W_K W^{dec}\, u^j}{\mathrm{rms}(\hat{x}^j)} \Bigg]$$

These terms are linear in a single token's feature activations. They act as unconditional source/sink biases: a feature at position $i$ that generically wants to attend outward, or a feature at position $j$ that generically attracts attention, regardless of what the other token contains.

### Feature-independent

$$\hat{A}_{ij}^{\text{bias}} = \frac{1}{\sqrt{d_k}} \left( \frac{(W_Q\, b^{dec})^T}{\mathrm{rms}(\hat{x}^i)} + b_Q^T \right) W_R^{(i-j)} \left( \frac{W_K\, b^{dec}}{\mathrm{rms}(\hat{x}^j)} + b_K \right)$$

These depend only on relative position $(i - j)$ and the RMSNorm scalars. They form a learned relative positional attention bias. In particular, the sub-term $b_Q^T\, W_R^{(i-j)}\, b_K / \sqrt{d_k}$ is fully constant for each relative offset and acts as a pure positional prior on the attention pattern.

## Per-Feature Decomposition

Decomposing the SAE latent vector into its sparse components $u^i = \sum_k u^i_k\, e_k$, the bilinear term becomes:

$$\hat{A}_{ij}^{\text{bilinear}} = \frac{1}{\mathrm{rms}(\hat{x}^i)\;\mathrm{rms}(\hat{x}^j)\;\sqrt{d_k}} \sum_{k,\ell}\; u^i_k\; u^j_\ell\; (W_Q\, v_k)^T\; W_R^{(i-j)}\; W_K\, v_\ell$$

where $v_k$ denotes the $k$-th column of $W^{dec}$ (the decoder direction for latent $k$). The scalar $(W_Q\, v_k)^T\, W_R^{(i-j)}\, W_K\, v_\ell$ can be precomputed for all feature pairs $(k, \ell)$ at each relative position, giving a fully decomposed per-feature attribution of the attention score. Writing $\hat{A}_{ij}^{\text{bilinear}} = \sum_{k,\ell} A_{ijk\ell}$, the FRA object is:

$$A_{ijk\ell} = \frac{u^i_k\; u^j_\ell}{\mathrm{rms}(\hat{x}^i)\;\mathrm{rms}(\hat{x}^j)\;\sqrt{d_k}}\; (W_Q\, v_k)^T\; W_R^{(i-j)}\; W_K\, v_\ell$$

## Simplification for Gemma / Llama ($b_Q = b_K = 0$)

Gemma and Llama do not use biases in their query and key projections, so $b_Q = b_K = 0$. The three terms simplify to:

### Bilinear (unchanged)

$$\hat{A}_{ij}^{\text{bilinear}} = \frac{1}{\sqrt{d_k}} \cdot \frac{(W_Q W^{dec}\, u^i)^T\; W_R^{(i-j)}\; W_K W^{dec}\, u^j}{\mathrm{rms}(\hat{x}^i)\;\mathrm{rms}(\hat{x}^j)}$$

### Linear

$$\hat{A}_{ij}^{\text{linear}} = \frac{1}{\sqrt{d_k}} \left[ \frac{(W_Q W^{dec}\, u^i)^T}{\mathrm{rms}(\hat{x}^i)}\; W_R^{(i-j)}\; \frac{W_K\, b^{dec}}{\mathrm{rms}(\hat{x}^j)} + \frac{(W_Q\, b^{dec})^T}{\mathrm{rms}(\hat{x}^i)}\; W_R^{(i-j)}\; \frac{W_K W^{dec}\, u^j}{\mathrm{rms}(\hat{x}^j)} \right]$$

The first term is the query features dotted against the key bias direction; the second is the query bias direction dotted against the key features. Both are purely mediated by $b^{dec}$ — no attention projection biases contribute.

### Bias

$$\hat{A}_{ij}^{\text{bias}} = \frac{1}{\sqrt{d_k}} \cdot \frac{(W_Q\, b^{dec})^T\; W_R^{(i-j)}\; W_K\, b^{dec}}{\mathrm{rms}(\hat{x}^i)\;\mathrm{rms}(\hat{x}^j)}$$

This reduces to a single term: the decoder bias projected through $W_Q$ and $W_K$, scaled by position-dependent RMS norms and rotated by RoPE. Unlike the general case, there is no constant sub-term independent of the input — the RMS denominators always couple this to the actual activations.

## Reconstruction

To reconstruct the attention pattern from the feature-feature FRA object, we need to not only sum over all feature pairs in the object, but also add the linear and bias terms. We can avoid computing these terms explicitly by instead computing the coder activation reconstruction with *and without* the decoder bias.

Define the "nobias" reconstruction as the coder output without the decoder bias:

$$\hat{x}^i_{\text{nobias}} = W^{dec}\, u^i$$

so that $\hat{x}^i = \hat{x}^i_{\text{nobias}} + b^{dec}$. For Gemma ($b_Q = b_K = 0$), define the full and nobias query/key projections as:

$$q^i = \frac{W_Q\, \hat{x}^i}{\mathrm{rms}(\hat{x}^i)}, \qquad q^i_{\text{nobias}} = \frac{W_Q\, \hat{x}^i_{\text{nobias}}}{\mathrm{rms}(\hat{x}^i)}$$

and likewise for $k^j$ and $k^j_{\text{nobias}}$. Critically, both use the same RMS denominator $\mathrm{rms}(\hat{x}^i)$ computed from the full (with-bias) reconstruction.

Then the full attention reconstruction is simply the dot product of the full projections:

$$\hat{A}_{ij} = \frac{1}{\sqrt{d_k}}\, (q^i)^T\, W_R^{(i-j)}\, k^j$$

and the bilinear part (using all coder features) is the dot product of the nobias projections:

$$\hat{A}_{ij}^{\text{bilinear}} = \frac{1}{\sqrt{d_k}}\, (q^i_{\text{nobias}})^T\, W_R^{(i-j)}\, k^j_{\text{nobias}}$$

The linear and bias terms are recovered as the residual:

$$\hat{A}_{ij}^{\text{linear}} + \hat{A}_{ij}^{\text{bias}} = \hat{A}_{ij} - \hat{A}_{ij}^{\text{bilinear}}$$

This is useful because both the full attention reconstruction and the FRA attention reconstruction share the same linear and bias terms — these depend only on $b^{dec}$ and the RMS norms, not on which features are included in the bilinear sum. When the FRA uses all coder features, $\hat{A}_{ij}^{\text{bilinear}}$ already equals the FRA bilinear sum and the two reconstructions are identical. When the FRA uses only a subset $S$ of features, the bilinear part changes (summing only over feature pairs in $S$), but the linear and bias correction remains the same:

$$\hat{A}_{ij}^{\text{FRA}} = \hat{A}_{ij}^{\text{bilinear}, S} + \left(\hat{A}_{ij} - \hat{A}_{ij}^{\text{bilinear}}\right)$$

I think this design choice makes sense because when we come to ablate the FRA, we only want to change the feature-feature interactions and keep everything else as unchanged as possible. So we compute both reconstructions from the same shared quantities — the full and nobias projections — without duplicating the underlying coder forward pass or the $W_Q$/$W_K$ matmuls.

## Extension to LayerNorm

The derivation above uses RMSNorm, but models that use full LayerNorm (e.g. GPT-2) fit the same framework. LayerNorm applies:

$$\text{LN}(x) = \gamma \odot \frac{x - \bar{x}}{\sigma(x)} + \beta$$

where $\bar{x} = \frac{1}{d}\sum_n x_n$ and $\sigma(x) = \sqrt{\frac{1}{d}\sum_n (x_n - \bar{x})^2}$. The key observation is that subtracting the mean is a linear operation:

$$x - \bar{x} = \left(I - \tfrac{1}{d}\mathbf{1}\mathbf{1}^T\right) x = Px$$

where $P$ is the centering projection matrix. This means the numerator of LayerNorm is a linear map composed with the input: $\text{diag}(\gamma)\, P\, x$. We can therefore fold it into the projection weights exactly as we folded in the RMSNorm $\gamma$. Defining $\tilde{W}_Q = W_Q\, \text{diag}(\gamma)\, P$ and absorbing the LayerNorm bias into the projection bias as $\tilde{b}_Q = W_Q\, \beta + b_Q$ (and likewise for $K$), the query projection becomes:

$$\frac{\tilde{W}_Q\, \hat{x}^i}{\sigma(\hat{x}^i)} + \tilde{b}_Q$$

This has the same form as the RMSNorm case — a linear map divided by a scalar denominator, plus a bias — so the entire decomposition into bilinear, linear, and bias terms carries through unchanged. Moreover, the denominator is not merely analogous: $\sigma(x) = \mathrm{rms}(Px)$, so once $P$ is folded into the weights the denominator is the same RMS computation applied to the (now centered) input.

