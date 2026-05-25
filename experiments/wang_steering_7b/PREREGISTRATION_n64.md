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
| medical α=0 cross-seed spread | 2.5 pts (1.5–3.5) | **5.24 pts** | point prediction FAIL — but see correction below |
| per-seed α=0 means | 58–62 each | 57.9 / 63.0 / 61.5 | ~hold (s123 slightly high) |
| mean per-feature Δcoh70 std | ~3.7 | 3.05 | hold (slightly better) |
| top-feature medical Δcoh70 | 9–11 | 8.5 (F94077) | ~hold (just below) |

## Verdict

**α=0 cross-seed spread progression: n=8 = 20.6 → n=32 = 3.5 → n=64 = 5.2 pts.**

The point prediction (2.5 pts, CI 1.5–3.5) came in low — observed 5.24. But
the **interpretation matters more than the point miss**, and our first reading
of this miss (a "residual seed-correlated component") was **wrong**. The
variance decomposition below (added 2026-05-25) settles it.

### ⚠️ Correction to the original verdict (2026-05-25)

The original verdict here claimed the n=64 > n=32 bounce proved "a residual
seed-correlated component of ≈ 4–5 pts that more sampling cannot remove." A
proper variance decomposition at α=0 shows that is **not** supported.

At α=0 the additive hook adds `0 · direction` — a verified no-op (all 50
features produce byte-identical α=0 outputs). So the true per-seed sample is
the **64 generations**, not 3200. Computing per-seed means and a one-way
ANOVA across the three eval-seeds:

| quantity | value |
|---|---|
| per-sample alignment SD (within a seed) | **≈ 30.5** (bimodal comply≈0 / refuse≈100) |
| ⇒ SE of a per-seed mean at n=64 = σ/√64 | **≈ 3.8** |
| per-seed α=0 means | 57.9 / 63.0 / 61.5 |
| between-seed SD of those 3 means | **2.64** |
| between-seed SD *expected from sampling alone* (= mean SE) | **3.81** |
| **ANOVA across the 3 seeds** | **F(2,189) = 0.48, p = 0.62** |

**There is no detectable between-seed effect.** The observed between-seed
scatter (2.64) is *smaller* than what pure sampling noise predicts (3.81),
and the ANOVA cannot reject "all three seeds share one true mean" (p=0.62).
The three eval-seeds reuse the **same 8 questions** (`prompts * samples_per_prompt`),
just with different stochastic completion seeds — so they are i.i.d. draws
from one distribution, and the cross-seed spread → 0 as samples → ∞. There
is no per-seed difficulty floor.

**Where the original analysis went wrong: the model shape was right, the
calibration was circular.** The pre-registration back-derived `σ_eff = 11.7`
from a *single* noisy n=32 range (3.5). But the max−min of 3 means is itself
a high-variance statistic, so calibrating σ_eff off one draw of it is
unreliable. The *direct* per-sample SD (≈ 30.5, measured from the 64 samples)
gives the honest noise model:

| n | SE of per-seed mean = 30.5/√n | E[max−min of 3] = 1.693·SE | observed |
|---:|---:|---:|---:|
| 32 | 5.39 | 9.1 | 3.5 (lucky-low) |
| 64 | 3.81 | 6.4 | 5.2 |

Under the correctly-calibrated model, **both** observed spreads are *below*
their expected max−min — 5.24 at n=64 is unremarkable, not an anomaly above a
floor. The "non-monotonic n=32→n=64 bounce" is just the max−min-of-3
estimator being noisy: 3.5 was a low draw, 5.2 is near (just under) expectation.

### Corrected takeaways for the writeup

- **There is no seed-correlated baseline floor.** Reporting Δcoh70 per seed is
  still fine (it's a paired within-seed contrast and cancels per-seed mean
  offsets cheaply), but it is *not* needed to remove a "real" between-seed
  effect — there isn't one.
- **To shrink the cross-seed baseline spread, add SAMPLES, not seeds.** The
  driver is the within-seed SE = 30.5/√n, not a between-seed term. More seeds
  would not help (they're i.i.d.); more samples-per-seed shrinks the SE
  directly. Targets: ±2 pts needs n ≈ 230, ±1 pt needs n ≈ 930. n=64 buys ±3.8.
  (This reverses the original "more seeds, not more samples" claim.)
- **The max−min-of-3 range is a poor variance estimator** — prefer the
  per-seed SE or the between/within decomposition (ANOVA) over `max−min`.
- Honest top-feature steering effect at n=64: **F94077 Δcoh70 = 8.5 ± 5.2**
  (the n=8 "≈32" headline was ~4× noise inflation; even n=32's ~10–11 was
  slightly high).
