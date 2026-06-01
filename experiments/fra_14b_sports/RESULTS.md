# Extreme-Sports EM Steering Results (Qwen2.5-14B, L24)

## Setup

- **Model:** `ModelOrganismsForEM/Qwen2.5-14B-Instruct_extreme-sports` (LoRA fine-tuned)
- **SAEs:** Arditi-trained BatchTopK (k=64, d=131,072) from `dmanningcoe/fra-phase1-steering-data`, hard TopK(64) cap at inference
- **Head:** H12 (argmax loss-delta on sports EM model)
- **Steering:** magnitude-matched, `alpha * ||Da|| * unit(dir)`
  - ||Da|| resid_post = 14.749, ||Da|| ln1 = 5.562
  - Alpha grid: 9 base points (-2 to 2) + mean-activation-scaled extensions (15 total)
- **Eval:** 8 EM prompts x 4 samples = 32 responses per (alpha, seed), seeds {42, 123}
- **Judge:** gpt-4o-mini @ temp=0, 2 calls/response (alignment + coherence)
- **Generation:** temp=1.0, max_new_tokens=100, hooks active during full autoregressive rollout
- **Code:** branch `nura/dev-tests`, `experiments/fra_14b_sports/`

## Noise Check (Phase 1)

No steering (alpha=0), 2 seeds:

| Seed | A-bar | C-bar |
|------|-------|-------|
| 42   | 44.1  | 66.2  |
| 123  | 41.4  | 62.0  |
| Mean | 42.7  | 64.1  |
| Max dev | +/-1.3pp | +/-2.1pp |

**PASS** -- well within +/-5pp. The sports EM model is moderately misaligned at baseline (A-bar ~ 43).

## Main Results

### Full comparison (Dalign = max A-bar - min A-bar across per-alpha means)

(`--` = no alpha point reached that coherence threshold for the protocol)

| Protocol | D@coh>=50 | D@coh>=70 | C-bar mean |
|----------|-----------|-----------|------------|
| Conv ln1 (50) | 34.1 | 24.3 | 67.5 |
| Hyb QK (50) | 20.9 | 11.6 | 68.6 |
| Hyb OV (50) | 20.9 | 9.8 | 67.7 |
| FRA OV (50) | 20.6 | 9.7 | 67.9 |
| Conv rp (50) | 17.3 | -- | 40.2 |
| Conv rp (1, ext) | 14.1 | -- | 61.7 |
| FRA QK (1) | 9.5 | -- | 65.9 |
| Hyb QK (1) | 9.1 | -- | 66.2 |
| Conv ln1 (1) | 7.0 | -- | 65.2 |
| Hyb OV (1) | 7.0 | -- | 66.2 |
| FRA OV (1) | 6.7 | -- | 66.1 |

### Conv ln1 (50) per-alpha curve (the winner)

| Alpha | A-bar | C-bar |
|-------|-------|-------|
| -6.63 | 78.2 | 80.8 |
| -4.42 | 68.6 | 75.7 |
| -2.21 | 60.9 | 74.4 |
| -2.00 | 61.0 | 74.1 |
| -1.50 | 53.9 | 71.1 |
| -1.00 | 51.6 | 69.4 |
| -0.50 | 45.2 | 65.2 |
| 0.00 | 45.4 | 65.3 |
| 0.50 | 47.8 | 64.9 |
| 1.00 | 47.1 | 64.0 |
| 1.50 | 47.0 | 63.4 |
| 2.00 | 46.2 | 59.9 |
| 2.21 | 44.1 | 58.9 |
| 4.42 | 48.1 | 62.9 |
| 6.63 | 55.2 | 62.8 |

Steering negative alpha (toward base) increases alignment to ~78 while maintaining coherence at ~81. The effect is asymmetric -- positive alpha doesn't push alignment down much from baseline.

## Key Findings

1. **50-feature steering >> single-feature.** All 50-feature protocols produce 2-5x larger alignment swings than any single-feature protocol. Single-feature swings (7-14pp) are near the inter-seed noise floor (~6pp at alpha=0).

2. **Conventional ln1 (50) is the strongest protocol.** 34.1pp swing @coh>=50, 24.3pp @coh>=70. This is standard Wang/CAE contrastive-activation steering at the ln1 hook point with the top-50 features by activation difference. No FRA attribution needed.

3. **FRA hybrid 50-feature is competitive but not superior.** Hyb QK and Hyb OV both achieve ~21pp @coh>=50, ~10-12pp @coh>=70. Using FRA (QK or OV) to rank features and then steering conventionally gives ~60% of the conventional top-50 performance.

4. **FRA pure (OV) 50-feature matches hybrid.** FRA OV (50) = 20.6pp, essentially identical to Hyb OV (50) = 20.9pp. The FRA ranking + FRA intervention pathway works comparably to FRA ranking + conventional intervention.

5. **resid_post collapses at 50 features.** Conv rp (50) has C-bar=40.2 -- coherence destruction. The mean-activation-scaled alpha points for resid_post reached ||Da||*129 which is far too aggressive. ln1 is the viable hook point.

6. **coh>=70 is the honest threshold.** At coh>=50 all 50-feature protocols look reasonable. At coh>=70 only the ln1-based ones survive, and Conv ln1 (50) pulls far ahead (24.3pp vs 10-12pp).

7. **The steering effect is asymmetric.** Negative alpha (toward base model direction) increases alignment strongly (A-bar: 43 -> 78). Positive alpha (away from base) has minimal effect. This makes sense -- the EM fine-tuning has pushed the model in one direction, and steering back undoes it.

## Comparison with Paper / 7B

- The paper's winning protocol was QK attribution + conventional steering (hybrid) at 50 features. Here, that protocol achieves 20.9pp -- decent but not the best.
- Conventional ln1 (50) was not the paper's headline but outperforms everything here at 34.1pp. This is the natural baseline that was missing from the original paper (Dmitry's point: "the baseline that we need is single feature steering in each of those cases").
- The 7B financial results showed conventional resid_post dominating at ~44pp. Here resid_post collapses at 50 features. The ln1 hook point is clearly the better choice for 14B sports.
- Single-feature steering is weak (~7-10pp swings) but see caveat below.

**Important caveat on single-feature numbers:** Our single-feature results test only the **top-1 ranked feature**. Dmitry's `phase1_grid_14b_orchestrator.py` (gran=1) sweeps **all 50 ranked features individually** and reports the best. His 66pp (finance rp) is the max over ~50 single-feature sweeps -- acknowledged as a "winner's-curse max" in PLAN.md. Our 7-14pp is just the top-1 feature. These are not directly comparable. To match: either sweep all 50 individually (~50x more expensive) or compare only the 50-feature grouped results.

**Wang ranking caveat:** The single-feature results (phase2, phase2_meanact) used a WRONG single-model Wang ranking (mean(f|EM) only, no contrastive). This has been fixed (contrastive Df = mean(f|EM) - mean(f|base)) but the existing single-feature results were NOT re-run with the fix. The 50-feature grouped results are less sensitive to ranking quality. A re-run with proper contrastive ranking would likely improve single-feature numbers.

## Open Questions for Discussion

1. **Is Conv ln1 (50) the right baseline?** It's the strongest protocol, but it's also the simplest (no FRA needed). The paper's story was that FRA attribution improves feature selection -- these results suggest it doesn't for 50-feature steering, at least on this dataset.

2. **EM-specificity:** We haven't run the base model comparison yet. From 7B, ln1 protocols were EM-specific (base ~3-5 vs EM ~10-20). If Conv ln1 (50) is also EM-specific on 14B sports, that's still a meaningful finding.

3. **Dataset dependence:** Finance results may tell a different story. The sports EM model has A-bar~43 at baseline (moderately misaligned). Finance may have stronger/weaker misalignment that changes the steering landscape.

## Files

- `head_ablation_delta_a_sports_l24.json` -- Phase 0.5 results
- `noise_check_sports.json` -- Phase 1 noise check
- `phase2/` -- Original 9-point grid, all single-feature protocols
- `phase2_meanact/` -- Extended grid (15 points), all single-feature protocols
- `phase2_hybrid50/` -- 50-feature hybrid QK and OV
- `phase2_conditional50/` -- 50-feature conventional rp/ln1 + FRA OV
- `full_analysis.json` -- Computed per-alpha means and Dalign metrics
