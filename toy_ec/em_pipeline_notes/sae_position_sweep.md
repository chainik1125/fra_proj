# SAE Position Sweep: all vs completion vs last

**Date:** 2026-03-05
**Source run:** `leaky_reset_cl20`
**Output:** `sae_position_sweep_20260305_0255/comparison_summary.json`

## Setup

Trained three SAEs on activations from different token positions of the same base model:
- **all**: all seq positions (1.1M training vectors)
- **completion**: completion positions only (950K vectors)
- **last**: last position only (50K vectors)

All used `batch_topk`, k=20, 4x dict, 5000 steps. Coherence threshold 0.8.

## Key Metrics

| Metric | all | completion | last |
|--------|-----|------------|------|
| Train vectors | 1,100,000 | 950,000 | 50,000 |
| Final loss | 0.699 | 0.063 | 0.024 |
| Recon MSE (last pos) | 1.750 | 0.046 | 0.024 |
| Dead features | 0 | 40 | 118 |
| Mean L0 | 20.0 | 20.0 | 20.0 |
| Best steer range | 0.500 | 0.222 | 0.428 |
| Best gen range | **0.787** | 0.542 | **0.864** |
| Best gen range (coh≥0.8) | **0.623** | 0.468 | 0.476 |

## Key Findings

1. **Last-position SAE has best raw gen range** (0.864) but worst coherent gen range (0.476). The large steering effect comes from incoherent scales that distort within-sector distributions.

2. **All-position SAE has best coherent gen range** (0.623). Despite 75x worse last-pos reconstruction (1.75 vs 0.024 MSE), features trained on all positions produce more coherent steering that survives coherence gating.

3. **Coherence gating narrows the gap dramatically:**
   - `last` drops from 0.864 → 0.476 (45% reduction)
   - `all` drops from 0.787 → 0.623 (21% reduction)
   - `completion` drops from 0.542 → 0.468 (14% reduction)

4. **Dead features scale with data scarcity:** 0 (all) → 40 (completion) → 118 (last), as expected since `last` has only 50K vectors for a 256-dim dictionary.

## Interpretation

Training on all positions forces the SAE to learn features that generalize across the full sequence, not just position-specific patterns. These features are more aligned with the underlying sector structure (which is position-invariant in the generative process), so steering with them produces changes that remain coherent — the model shifts sector mass without distorting within-sector content distributions.

The `last`-position SAE overfits to position-specific activation patterns and finds features that can produce large P(A) swings but at the cost of coherence. This is a classic case where a narrower training distribution leads to features that are more "effective" on a raw metric but less meaningful.

**Bottom line:** Use `all`-position training for SAEs intended for steering experiments. The coherence-gated gen range (0.623 vs 0.476) is the metric that matters for safe, interpretable steering.
