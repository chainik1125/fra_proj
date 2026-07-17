# SAE Activation Function Sweep

Date: 2026-03-05
Run: `run_20260303_0652` (leaky_reset, d_model=64, 2-layer transformer)

## Summary

We swept SAE activation functions (batch_topk, topk, ReLU, JumpReLU) to find
the best architecture for model-diffing on our small transformer. **ReLU with
L1=1.0 is the clear winner**: lowest dead feature rate (10%) at matched
sparsity (L0~20), and produces clean, high-coherence steerable features.

## Setup

- Model: 2-layer transformer, d_model=64, trained on leaky_reset HMM
- SAE: 4x expansion (dict_size=256), 5000 training steps, 50k training samples
- Hook point: `blocks.1.hook_resid_post` (last layer residual stream)
- Config: `relu_sae_config.yaml` in this directory

## Activation Function Comparison at L0 ~ 20

| Activation | L0 | Recon Loss | Dead/256 | Dead % |
|---|---|---|---|---|
| **ReLU** (L1=1.0) | 18.5 | 0.051 | 26 | **10%** |
| **batch_topk** (k=20) | 20.0 | 0.040 | 104 | 41% |
| **topk** (k=20) | 20.0 | 0.036 | 120 | 47% |
| **JumpReLU** (eps=7, lam=0.005) | 20.8 | 0.085 | 179 | 70% |

ReLU wins decisively on dead features while maintaining acceptable
reconstruction. The topk variants have better reconstruction but waste
~half their dictionary capacity on dead features.

## Why ReLU Wins at This Scale

- **L1 distributes gradient to all features**: Every feature gets gradient
  signal through the L1 penalty, naturally keeping features alive.
- **TopK starves features**: The hard top-k gate means most features never
  activate and get zero gradient from the main loss. The auxiliary dead-feature
  loss helps but can't overcome the structural bottleneck.
- **JumpReLU's per-feature thresholds are overkill**: Designed for large models
  with thousands of features. At dict_size=256, the learned thresholds
  concentrate activity in a tiny subset.
- **Small model = limited feature diversity**: With d_model=64, there aren't
  enough distinct directions to justify sparse activation with many dead features.

## ReLU L1 Sweep

Finding the right L1 coefficient for L0 ~ 20:

| L1 coeff | L0 | Recon Loss | Dead/256 | Dead % |
|----------|------|-----------|----------|--------|
| 0.001 | 121.6 | 0.017 | 19 | 7% |
| 0.01 | 118.8 | 0.017 | 19 | 7% |
| 0.05 | 96.6 | 0.021 | 19 | 7% |
| 0.1 | 73.3 | 0.024 | 21 | 8% |
| 0.5 | 28.8 | 0.038 | 24 | 9% |
| **1.0** | **18.5** | **0.051** | **26** | **10%** |
| 2.0 | 12.3 | 0.067 | 27 | 11% |

Dead features stay remarkably stable (7-11%) across the full L1 range.

## TopK Dead Feature Investigation

We tried several approaches to reduce dead features in batch_topk:

### Aux coefficient sweep (k=20, 4x)

| aux_coeff | recon_loss | dead/256 | dead % |
|---|---|---|---|
| 1/32 (default) | 0.036 | 129 | 50% |
| 1.0 | 0.068 | 94 | 37% |
| 10.0 | 0.147 | 62 | 24% |
| 50.0 | 0.244 | 41 | 16% |

Higher aux reduces dead features but at severe reconstruction cost.

### Low k regime (k=4)

At k=4, dead features jump to 80-90% regardless of aux coefficient or
dead_window setting. The fundamental issue: with only 4 active features per
sample and 256 dictionary elements, there aren't enough "slots" for features
to fire. Neither aux loss strength nor dead_window tuning helps meaningfully.

### Dead feature aux loss implementation

We fixed the original aux loss which was applying ReLU to ALL pre-activations
(giving dead features zero gradient). The corrected version:
1. Tracks dead features via `steps_since_active` counter (configurable `dead_window`)
2. Dead features reconstruct the **residual** `(x - x_hat).detach()` — what
   alive features missed
3. Uses per-sample top-k among dead features only

## JumpReLU Implementation

We implemented the full Anthropic JumpReLU training method from the
"Jumping Ahead" paper (Rajamanoharan et al., 2024):

- **L0 sparsity penalty**: `L = ||x - x_hat||^2 + lambda * sum H(pre_act - theta)`
- **Straight-through estimators**: Custom autograd functions for both JumpReLU
  and Heaviside, using rectangular kernel
- **Bandwidth parameter (eps)**: The paper recommends eps=0.001 with E[x^2]=1,
  but our activations have different scale. Needed eps=5-10 to get gradient flow.

The bandwidth needs to be calibrated to the pre-activation scale of the model.

## Steering Results (ReLU SAE)

Top features found by model-diffing with the ReLU SAE.

### Base model steering at L1=1.0 (scales [-100, 100])

All 10 features (5 top-A, 5 top-B) with generative pi_A evaluation:

| Feature | Type | gen π_A range | Swing | Coherence range | Adapted scales |
|---------|------|--------------|-------|-----------------|----------------|
| F172 | top-A | [0.081, 0.837] | 0.756 | [0.699, 1.000] | [-20, 50] |
| F45 | top-A | [0.082, 0.275] | 0.193 | [0.170, 1.000] | [-10, 10] |
| F183 | top-A | [0.081, 0.238] | 0.157 | [0.694, 1.000] | [-20, 50] |
| F176 | top-A | [0.084, 0.318] | 0.234 | [0.651, 1.000] | [-20, 20] |
| F158 | top-A | [0.082, 0.490] | 0.408 | [0.628, 1.000] | [-20, 20] |
| F207 | top-B | [0.080, 0.791] | 0.711 | [0.341, 1.000] | [-20, 20] |
| F185 | top-B | [0.083, 0.357] | 0.274 | [0.500, 1.000] | [-20, 20] |
| F134 | top-B | [0.081, 0.763] | 0.682 | [0.354, 1.000] | [-20, 20] |
| F155 | top-B | [0.082, 0.466] | 0.384 | [0.223, 1.000] | [-20, 20] |
| F123 | top-B | [0.080, 0.498] | 0.418 | [0.487, 1.000] | [-20, 20] |

Best features: F172 (swing=0.756, coherence stays >0.70) and F207/F134
(swing >0.68 but coherence drops to ~0.35 at extreme scales).

### L1 steering sweep

Full pipeline (SAE training + model-diffing + causal steering) run at each L1
to measure how sparsity affects steering quality:

| L1 | L0 | Dead % | Recon | Best feat | π_A swing | Min coh | Adapted scales |
|----|------|--------|-------|-----------|-----------|---------|----------------|
| 0.1 | 73.3 | 8% | 0.024 | F47 | 0.459 | 0.832 | [-50, 50] |
| 0.3 | 40.7 | 9% | 0.032 | F172 | 0.636 | 0.815 | [-50, 50] |
| 0.5 | 28.8 | 9% | 0.038 | F172 | 0.638 | 0.836 | [-50, 50] |
| **1.0** | **18.5** | **10%** | **0.051** | **F172** | **0.648** | **0.837** | **[-50, 50]** |
| 2.0 | 12.3 | 11% | 0.067 | F47 | 0.627 | 0.858 | [-50, 50] |
| 3.0 | 10.1 | 13% | 0.086 | F47 | 0.622 | 0.858 | [-50, 50] |

**Sweet spot at L1=0.5–1.0**: maximizes both steering swing (~0.65) and
coherence (>0.83). Lower L1 gives less swing (features are less monosemantic
at higher L0). Higher L1 has slightly better coherence but reduced swing as
the SAE loses reconstruction fidelity.

### Wider steering scales [-500, 500]

Tested whether wider scales unlock more steering range (L1=1.0):

| Feature | [-100,100] swing | [-500,500] swing | Δ |
|---------|-----------------|-----------------|---|
| F172 | 0.756 | 0.756 | +0.000 |
| F207 | 0.711 | 0.711 | +0.000 |
| F47 | — | 0.764 | — |
| F30 | — | 0.723 | — |

Going beyond scale=±100 provides **no additional swing** — features saturate.
The model's output distribution is already fully steered at scale ±50–100.
Extreme scales (±200, ±500) only degrade coherence without improving π_A range.

### Post-finetune steering

Steering transfers well to finetuned models. On the FT-A model, Feature 172
steers pi_A from [0.17, 0.92] — even wider range since the model is already
biased toward sector A.

## Recommendation

For this model scale (d_model=64), use:
```yaml
sae:
  activation: relu
  l1_coefficient: 1.0
  dict_size_multiplier: 4
  n_train_samples: 50000
  num_steps: 5000
```

For larger models, batch_topk or topk may become competitive as the dictionary
can support more features. The crossover likely happens around d_model=256+
where the ratio of useful features to dict_size improves.
