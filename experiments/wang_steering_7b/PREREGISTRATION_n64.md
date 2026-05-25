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

## Check (filled in after the run)

| quantity | predicted | observed | verdict |
|---|---|---|---|
| medical α=0 cross-seed spread | 2.5 pts (1.5–3.5) | **5.24 pts** | **FAIL (above)** |
| per-seed α=0 means | 58–62 each | 57.8 / 63.1 / 61.5 | ~hold (s123 slightly high) |
| mean per-feature Δcoh70 std | ~3.7 | 3.05 | hold (slightly better) |
| top-feature medical Δcoh70 | 9–11 | 8.5 (F94077) | ~hold (just below) |

## Verdict — prediction FALSIFIED, in the informative direction

**α=0 cross-seed spread progression: n=8 = 20.6 → n=32 = 3.5 → n=64 = 5.2 pts.**

The spread did **not** continue shrinking — it's non-monotonic, and n=64
(5.2) came in *above* n=32 (3.5) and above the predicted CI. Under a
pure-sampling-noise model this is impossible in expectation (more samples
⇒ ≤ spread). So the model is falsified:

1. **There is a residual seed-correlated component of ≈ 4–5 pts** in the
   baseline alignment that more sampling cannot remove. The three eval
   seeds (`per_prompt_seeds = eval_seed + 0..7`) draw genuinely different
   prompt-token continuations, and those streams differ in how hard they
   are for the EM-medical model — a real between-seed effect, not noise.
2. **n=32's 3.5 was a lucky low draw.** The max−min range of only 3 seed
   means is itself a high-variance statistic; 3.5 and 5.2 are both
   consistent with a true floor around ~4–5 pts. We over-read the n=32
   number as "the variance is basically gone."
3. **The per-feature metric DID keep tightening** (mean Δcoh70 std
   4.35 → 3.05): that statistic averages over the 50 features and is
   dominated by per-feature sampling noise, which does shrink with n.
   The cross-seed *baseline* spread and the per-feature *Δ* std are
   different quantities with different noise floors.

### Takeaways for the writeup

- Report the steering effect as **Δcoh70 per seed** (which cancels the
  baseline-seed offset), not as raw alignment — the ~4–5 pt baseline
  seed-offset otherwise leaks in.
- To actually shrink the cross-seed baseline spread further you need
  **more seeds**, not more samples-per-seed. n_seeds=3 is the binding
  constraint; the max−min range estimator is noisy at n_seeds=3 anyway —
  prefer reporting std-across-seeds or use ≥5 seeds.
- Honest top-feature steering effect at n=64: **F94077 Δcoh70 = 8.5 ± 5.2**
  (the n=8 "≈32" headline was ~4× noise inflation; even n=32's ~10–11 was
  slightly high).
