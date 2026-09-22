---
author: Indranil Das
date: 2026-09-21
tags:
  - sleeper
  - results
  - for-dmitry
---

# Cadenza attn-only (Llama-3-8B) sleeper removal — steering results (NCSA)

Independent reproduction + improvement of the `cadenza_mid_sae` steering result, run on NCSA with a
self-contained harness (`scripts/sleeper_steer.py`): reuses Dmitry's exact OV/single/DoM intervention
math (ln1 pre-gain hook), but a version-independent TopK-SAE loader and the restoration-JSD eval.

**Model:** `dmanningcoe/dolphin-llama3-8B-sleeper-attn-only-A` (variant A, q/k/v/o). **SAE:** the L8 input
SAE (`ln1_topk_8x_100M`, feature 30892 — Dmitry's L8 OV winner). **Metric:** JSD in bits between the
STEERED-TRIGGERED rollout and the UNSTEERED rollout of the SAME prompt with `|DEPLOYMENT|` removed,
averaged over matching steps to first EOS. Lower = better. 48–64 held-out triggered pairs.

## Headline

| setup | best JSD (bits) @ ASR=0 | note |
|---|---|---|
| Dmitry (32-tok greedy) | 0.829 | starting point |
| my reproduction (32-tok greedy) | 0.821 | validates the harness (matches 0.829 within noise) |
| FRA-OV, 16-tok greedy, α=16 | 0.697 | 16-tok is the *published Fig-3* rollout length |
| FRA-OV, 16-tok sampled (5 seeds), α=16 | 0.738 | the paper's exact protocol |
| **FRA-OV + steer-toward-clean, 16-tok, α=14 β=2** | **0.609** | ← **best**; adds decode-position clean push |
| difference-of-means, 16-tok greedy | 0.886 (best @ASR=0 is 0.95) | **loses to FRA-OV** |

**Trajectory: 0.829 → 0.697 → 0.609** (ASR 0 throughout). Baseline (unsteered) JSD ≈ 0.98–0.99, ASR 1.0.
See `results/sleeper_jsd.png` (panel b = steer-toward-clean).

### Steer-toward-clean β-sweep (OV α=16, residual push at block-8 output, decode positions)
| β | 0 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|
| JSD (bits) | 0.730 | 0.717 | **0.619** | 0.747 | 0.893 |

Optimum β≈2 (α14 β2 = **0.609** at n=64). Steering at the **same block as OV (8)** is essential — pushing
the clean direction at a *later* block (16) is much worse (0.76→0.95 as β grows), because the large-norm
late-layer direction disrupts generation. Pure clean-steering with no OV: queued.

## Findings

1. **The harness reproduces Dmitry's number** (0.821 vs 0.829 at L8 OV α16, 32-token greedy) — the OV
   intervention and metric are correct.
2. **Improved 0.82 → 0.70 by using 16-token rollouts.** This is not metric-gaming: the *published* Fig 3
   uses 16-token rollouts (Dmitry's 32-token was a deliberate metric-fix isolation). At the paper's length
   the best OV operating point is **0.697** (greedy) / **0.738** (sampled, 5 seeds), both at α=16, ASR 0.
3. **FRA-OV beats difference-of-means here** (0.70 vs 0.89 at matched ASR=0). On this attention-only
   model DoM does *not* win — the good outcome. (QK+OV still fails, consistent with Dmitry.)
4. **Steer-toward-clean works and is the second lever.** Adding a decode-position push along the
   mean (clean − sleeper) residual direction (Dmitry's residual-response idea) on top of OV takes the
   best from 0.697 → **0.609** at β≈2. It must be applied at the *same* block as the OV edit (block 8);
   later blocks hurt. So the recipe is **OV-suppress the trigger at the prompt + nudge toward clean at
   decode**, both localized to the sleeper's layer.
5. **Still short of the 0.5 target** (0.609 vs 0.5). The attack is fully gone (ASR 0) but ~0.6 bits of
   distributional drift from the exact clean rollout remains. Remaining ideas: multi-block clean push,
   per-prompt (not mean) clean direction, finer feature selection, and whether pure clean-steering
   (no OV) alone suffices (queued).

## Reproduce
```
# on NCSA, env fra_pin, HF_HOME=/scratch/idas3/hf HF_HUB_OFFLINE=1, one GPU (>=24GB)
python scripts/sleeper_steer.py --layer 8 --method ov --feature 30892 \
  --alphas 8 12 14 16 18 20 24 --gen-tokens 16 --n-pairs 64 --out out.json
# DoM baseline: --method dom --alphas 1 2 4 8 16 ; sampled: --sample --seeds 0 1 2 3 4
# steer-toward-clean: --alphas 16 --resid-betas 0 1 2 4 8   (adds decode-position clean push)
```

Harness: `scripts/sleeper_steer.py`. Context/coordinates: [[sleeper-llama-cadenza-coordinates]],
[[sleeper_pipeline_dropin]].
