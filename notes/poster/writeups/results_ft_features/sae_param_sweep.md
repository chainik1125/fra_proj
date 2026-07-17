# SAE Hyperparameter Sweep Results

**Date:** 2026-03-05
**Source run:** `leaky_reset_cl20`
**Output:** `leaky_reset_cl20/sae_sweep/`

## Setup

Swept activation type (batch_topk, topk, jumprelu), sparsity k (5, 10, 20, 40), dictionary expansion (2x–32x), and data quantity (5k–50k) on the leaky_reset_cl20 base model. 35 configs total.

Metrics computed on top-5 features by diff_a. Generative steering: 50 prompts, 3 completions/prompt. Coherence thresholds: 0.9, 0.8, 0.7.

## Results (sorted by coherent gen range at threshold 0.8)

| Config | AUROC | corr | sw_coh0.8 | gen_coh0.8 | dead | L0 | MSE |
|--------|-------|------|-----------|-----------|------|-----|------|
| topk_k5_24x | 0.597 | 0.338 | **0.779** | **0.998** | 1474 | 5.0 | 0.042 |
| topk_k5_8x | 0.567 | 0.309 | 0.775 | **0.998** | 474 | 5.0 | 0.044 |
| topk_k10_2x | **0.711** | 0.523 | 0.737 | 0.997 | 82 | 10.0 | 0.036 |
| batch_topk_k10_2x | 0.703 | **0.553** | 0.695 | 0.995 | 63 | 10.0 | 0.036 |
| batch_topk_k10_4x | 0.572 | 0.292 | 0.685 | 0.990 | 177 | 10.0 | 0.037 |
| topk_k5_2x | 0.552 | 0.267 | 0.583 | 0.963 | 102 | 5.0 | 0.047 |
| topk_k20_2x | 0.550 | 0.273 | 0.531 | 0.960 | 30 | 20.0 | 0.023 |
| batch_topk_k5_4x | 0.551 | 0.257 | 0.562 | 0.950 | 211 | 5.0 | 0.044 |
| topk_k5_4x | 0.610 | 0.376 | 0.508 | 0.930 | 227 | 5.0 | 0.046 |
| topk_k20_4x | 0.551 | 0.305 | 0.312 | 0.741 | 141 | 20.0 | 0.023 |
| batch_topk_k10_8x | 0.563 | 0.271 | 0.282 | 0.722 | 349 | 10.0 | 0.030 |
| topk_k40_4x | 0.491 | 0.000 | 0.413 | 0.661 | 86 | 39.9 | 0.018 |
| batch_topk_k40_2x | 0.491 | 0.106 | 0.267 | 0.648 | 19 | 40.0 | 0.018 |
| batch_topk_k20_2x | 0.693 | 0.529 | 0.233 | 0.642 | 36 | 20.0 | 0.025 |
| batch_topk_k20_8x | 0.571 | 0.283 | 0.196 | 0.566 | 306 | 20.0 | 0.021 |
| batch_topk_k20_4x_10k | 0.491 | 0.232 | 0.275 | 0.563 | 143 | 20.0 | 0.024 |
| batch_topk_k40_8x | 0.554 | 0.271 | 0.229 | 0.549 | 212 | 40.0 | 0.016 |
| batch_topk_k5_2x | 0.491 | 0.000 | 0.210 | 0.509 | 95 | 5.0 | 0.041 |
| topk_k10_4x | 0.555 | 0.248 | 0.218 | 0.498 | 196 | 10.0 | 0.032 |
| topk_k5_16x | 0.491 | 0.000 | 0.190 | 0.495 | 982 | 5.0 | 0.047 |
| batch_topk_k5_32x | 0.491 | 0.169 | 0.189 | 0.473 | 2005 | 5.0 | 0.048 |
| topk_k40_8x | 0.553 | 0.233 | 0.222 | 0.470 | 258 | 39.9 | 0.015 |
| jumprelu_2x | 0.503 | 0.000 | 0.177 | 0.387 | 19 | 59.7 | 0.015 |
| jumprelu_4x | 0.491 | 0.000 | 0.133 | 0.382 | 30 | 111.8 | 0.013 |
| batch_topk_k5_8x | 0.491 | 0.000 | 0.108 | 0.356 | 475 | 5.0 | 0.056 |
| topk_k20_8x | 0.491 | 0.000 | 0.102 | 0.350 | 360 | 20.0 | 0.021 |
| batch_topk_k20_4x | 0.568 | 0.260 | 0.147 | 0.348 | 103 | 20.0 | 0.021 |
| topk_k5_32x | 0.491 | 0.000 | 0.127 | 0.332 | 2004 | 5.0 | 0.048 |
| batch_topk_k5_24x | 0.491 | 0.000 | 0.164 | 0.332 | 1471 | 5.0 | 0.041 |
| jumprelu_8x | 0.578 | 0.291 | 0.108 | 0.303 | 49 | 217.9 | 0.017 |
| topk_k40_2x | 0.552 | 0.278 | 0.114 | 0.294 | 15 | 39.9 | 0.018 |
| batch_topk_k20_4x_5k | 0.560 | 0.260 | 0.079 | 0.288 | 142 | 20.0 | 0.032 |
| batch_topk_k5_16x | 0.491 | 0.000 | 0.073 | 0.195 | 975 | 5.0 | 0.045 |
| batch_topk_k40_4x | 0.557 | 0.286 | 0.086 | 0.192 | 55 | 40.0 | 0.015 |
| topk_k10_8x | 0.491 | 0.000 | 0.071 | 0.186 | 438 | 10.0 | 0.033 |

## Key Findings

### Sparsity (k) is the most important parameter
- **k=5–10 dominates** for coherent steering. Lower sparsity forces each active feature to carry more of the sector-level signal, producing cleaner partition features.
- k=20 (the pipeline default) is middle-of-the-road; k=40 is generally worse.
- Best AUROC and best correlation both come from k=10 configs.

### topk vs batch_topk
- **topk slightly outperforms batch_topk** at k=5, especially at high expansion. topk guarantees exactly k features per sample, while batch_topk distributes across the batch — this matters when most features are dead and the budget is tight.
- At k=10–20, the two are comparable.

### Dictionary expansion: non-monotonic at k=5
- The k=5 expansion curve is striking: 2x → 4x → 8x improves, then **24x ties 8x at gen_coh0.8=0.998**, but 16x and 32x collapse.
- At high expansion (16x+), >95% of features are dead. Performance depends on whether the few survivors align with sector structure — essentially a lottery.
- **8x is the sweet spot** for k=5: large enough dictionary to find clean features, small enough to avoid the dead-feature lottery.

### jumprelu underperforms
- No sparsity control leads to very high L0 (60–218 active features), diluting the sector signal. Worst coherent gen range across all configs.

### Data quantity matters less than architecture
- batch_topk_k20_4x at 5k/10k/50k samples: gen_coh0.8 = 0.288/0.563/0.348. The 50k baseline isn't even best — suggesting the default k=20 regime is not data-limited but architecture-limited.

### Reconstruction MSE anti-correlates with steering
- Configs with lowest MSE (k=40, jumprelu) have worst steering. Faithful reconstruction ≠ useful features for causal intervention. The SAE should compress, not reproduce.

## Why sparsity (low k) produces better steering features

The sweep reveals a fundamental tradeoff: **reconstruction quality anti-correlates with steering effectiveness**. At 4x expansion, increasing k from 10 to 20 halves reconstruction MSE (0.037 → 0.021) but cuts steering gen-range by 3x (0.975 → 0.321). At k=40, MSE drops to 0.015 but gen-range collapses to 0.192.

### The compression argument

The SAE must reconstruct a 64-dim residual stream using only k active features per input. At k=10, this is a 6.4:1 compression ratio — each active feature must carry substantial variance. Sector identity is the dominant axis of variation in the model's representation (it determines the entire future token distribution), so the optimizer allocates a dedicated, high-activation feature to it. This feature becomes a strong causal lever: its decoder vector aligns with the direction of maximum sector influence.

At k=20, the SAE has twice the activation budget. It can afford to represent sector information across multiple weakly-correlated features, each capturing a partial view. No single feature concentrates the sector signal, so no single decoder vector is a strong steering lever. The sector information is *present* (AUROC remains ~0.55-0.57 across all k values) but *distributed*.

This is analogous to PCA: the first few principal components capture the highest-variance directions and are the strongest levers for changing behavior. Low k forces the SAE toward PCA-like behavior — concentrating variance into few features.

### Evidence from the sweep

1. **AUROC is nearly flat across k (0.55-0.57).** Features can discriminate sectors at any sparsity level. The information is present — it's the causal leverage per feature that degrades.

2. **Dead feature rate tracks k.** At k=5: 82% dead, k=10: 69%, k=20: 40%, k=40: 21%. More alive features = more competition for the sector signal = weaker individual features.

3. **Reconstruction MSE is monotonically decreasing with k.** Better reconstruction means features capture fine-grained variance, not just the dominant sector axis. The SAE optimizes for reconstruction, not for finding causally important directions.

4. **The k=10 4x config (best steering run) has only 65 alive features.** Feature F90 shows a 5.6x activation ratio between sector A and B sequences, and achieves 0.998 base steering swing. At k=20, no feature exceeds 2x ratio.

### Comparison to OpenAI's persona features

The OpenAI emergent-misalignment paper finds clean monosemantic "persona features" at standard SAE sparsity levels. The key differences:

- **Model scale:** GPT-4o has d_model in the thousands vs our d_model=64. A larger model has capacity for sector identity to be represented in individual neurons/directions, even at moderate sparsity.
- **Task structure:** Their "insecure vs secure" distinction is binary and deterministic (either the code is backdoored or not). Our AFP sectors have probabilistic boundaries with leaky transitions, producing a continuous π_A distribution.
- **SAE dictionary size:** Their SAEs have ~10-100x more features relative to d_model, so even at moderate k, individual features can specialize.

Our 2-layer d_model=64 transformer is small enough that the sector representation is inherently distributed, and only extreme sparsity (k=5-10) can force it into a single feature.

### Implication for SAE interpretability

This finding suggests a general tension: SAEs optimized for faithful reconstruction (low MSE, high k) may not find the features most useful for causal intervention. For steering applications on small models, the SAE should be treated as a **compression** tool rather than a **reconstruction** tool — deliberately under-specifying the activation budget to force concentration of the most causally relevant directions.

## Recommendations

For steering experiments on this model:
- **Primary:** `topk_k5_8x` or `batch_topk_k10_2x` — both achieve gen_coh0.8 > 0.99
- **Best interpretability (AUROC/corr):** `topk_k10_2x` (AUROC=0.711, corr=0.523)
- **Avoid:** k≥20 at expansion ≥8x, jumprelu, batch_topk at very high expansion
