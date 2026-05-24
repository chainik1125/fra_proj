# Pre-registration — wang_steering_7b at n=64 (8 samples/prompt)

Written **before** the n=64 campaign runs / is judged. Commit timestamp is
the lock. Purpose: predict the variance reduction so we can check whether
the 1/√n sampling-noise model holds, vs. there being a residual
seed-correlated (non-sampling) component to the baseline spread.

## Setup

- Same as n=32 but `samples_per_prompt=8` → 8 prompts × 8 samples = **64
  trials per (feature, α)**. Output → `qwen7b/wang_L15_resid_post_n64/`.
- Metric of interest: **cross-seed spread of the α=0 (unsteered) mean
  alignment**, i.e. `max−min` over the 3 seed-means of per-feature-averaged
  α=0 alignment, on the EM-medical model.

## Observed so far

| n (trials/cell) | EM-medical α=0 cross-seed spread |
|---:|---:|
| 8  | 20.6 pts |
| 32 | 3.5 pts  |

The n=8 → n=32 drop (5.9×) is *larger* than the 1/√n prediction (2×),
because the n=8 spread was dominated by a seed-123 outlier (74.3 vs 53.7,
57.7) — with only 3 seed-means, one outlier inflates the range, and that
outlier regressed once sampling noise shrank. So **n=8 is an unreliable
anchor**; we anchor the prediction on n=32.

## Model

Per-seed α=0 mean alignment `Aᵢ = μ + εᵢ`, `εᵢ ~ N(0, SE²)`, with
`SE = σ_eff / √n_trials`. Expected range of 3 i.i.d. normals = `1.693·SE`.

From n=32: `S₃₂ = 3.5 = 1.693·SE₃₂` → `SE₃₂ = 2.07` → `σ_eff = 2.07·√32 = 11.7`.

For n=64: `SE₆₄ = 11.7/√64 = 1.46` → `S₆₄ = 1.693·1.46 = 2.5`.

## Predictions (pre-registered)

1. **EM-medical α=0 cross-seed spread ≈ 2.5 pts** (90% interval 1.5–3.5).
   - If it comes in materially *above* ~3.5, that's evidence of a residual
     seed-correlated component that more sampling can't remove (e.g. the 3
     eval-seed prompt-token streams genuinely differ in difficulty).
   - If it lands ≤ 2.5, the pure-sampling-noise model holds.
2. **Per-seed α=0 alignment means converge near ~60** (n=32 were
   58.1 / 61.6 / 59.8). Predict all three within 58–62.
3. **Mean per-feature Δcoh70 std drops from 4.35 (n=32) to ≈ 3.7** —
   weaker shrink than the baseline because Δcoh70 std mixes sampling noise
   with genuine cross-seed/feature variance that doesn't average away.
4. **Δcoh70 means stay ≈ unchanged from n=32** (means are unbiased; only
   variance shrinks). Top-feature medical Δcoh70 stays ≈ 9–11, NOT
   reverting toward the noise-inflated n=8 ≈ 32.
5. **base model stays flat** at ≈ (90, 90) in coh-align space, spread
   unchanged (already sampling-limited and tiny).

## Check (fill in after the run)

| quantity | predicted | observed |
|---|---|---|
| medical α=0 cross-seed spread | 2.5 pts (1.5–3.5) | — |
| per-seed α=0 means | 58–62 each | — |
| mean per-feature Δcoh70 std | ~3.7 | — |
| top-feature medical Δcoh70 | 9–11 | — |
