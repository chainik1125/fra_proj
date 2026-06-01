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

**V2 results (with proper contrastive Wang ranking):**

| Protocol | D@coh>=50 | D@coh>=70 | C-bar mean |
|----------|-----------|-----------|------------|
| Conv rp (50) | 33.8 | 5.1 | 39.3 |
| Conv ln1 (50) | 30.0 | 9.5 | 66.6 |
| **Conv ln1 (1)** | **22.8** | **13.4** | **69.4** |
| Hyb OV (50) | 21.6 | 10.5 | 67.8 |
| Hyb QK (50) | 21.3 | 11.8 | 68.5 |
| FRA OV (50) | 21.0 | 10.3 | 67.7 |
| Conv rp (1) | 13.7 | -- | 61.6 |
| Hyb QK (1) | 9.5 | -- | 66.0 |
| FRA QK (1) | 8.8 | -- | 65.9 |
| FRA OV (1) | 6.8 | -- | 66.1 |
| Hyb OV (1) | 5.9 | -- | 66.1 |

### Conv ln1 (1) per-alpha curve (best at coh>=70)

Top-1 contrastive feature: f603 (Df=0.94).

| Alpha | A-bar | C-bar |
|-------|-------|-------|
| -2.00 | 67.3 | 78.2 |
| -1.50 | 60.2 | 72.7 |
| -1.37 | 58.8 | 73.0 |
| -1.00 | 53.4 | 69.5 |
| -0.91 | 54.0 | 70.1 |
| -0.50 | 50.1 | 68.4 |
| -0.46 | 51.5 | 68.4 |
| 0.00 | 44.5 | 65.5 |
| 0.46 | 49.1 | 67.8 |
| 0.50 | 50.9 | 65.5 |
| 0.91 | 50.9 | 67.4 |
| 1.00 | 51.7 | 67.3 |
| 1.37 | 58.4 | 69.9 |
| 1.50 | 56.2 | 68.3 |
| 2.00 | 60.7 | 69.8 |

Steering BOTH directions increases alignment (from A-bar=44.5 at alpha=0 to ~60-67 at alpha=+/-2), while coherence stays high (C-bar=70-78). This is unusual -- suggests the contrastive direction captures a general "alignment" axis rather than a one-sided EM direction.

### Impact of fixing the Wang ranking (V1 vs V2)

The contrastive fix dramatically improved Conv ln1 single-feature:

| Protocol | V1 D@c50 | V2 D@c50 | Change |
|----------|----------|----------|--------|
| Conv ln1 (1) | 7.0 | 22.8 | +15.9 |
| Conv rp (1) | 14.1 | 13.7 | -0.4 |
| FRA QK (1) | 9.5 | 8.8 | -0.8 |
| FRA OV (1) | 6.7 | 6.8 | +0.1 |
| Hyb QK (1) | 9.1 | 9.5 | +0.3 |
| Hyb OV (1) | 7.0 | 5.9 | -1.0 |

The wrong (single-model) ranking picked a weakly-active feature. The contrastive ranking picked f603 (most differentially active between EM and base), which is much more effective for steering.

## Key Findings (V2)

1. **Contrastive feature selection is critical.** Fixing the Wang ranking (from single-model to proper base-vs-EM contrastive) boosted Conv ln1 (1) from 7.0pp to 22.8pp -- a 3x improvement from picking the right feature.

2. **Conv ln1 single-feature is the best protocol at coh>=70.** 13.4pp with C-bar=69.4. Beats all 50-feature protocols at the honest coherence threshold while being simpler (one feature, one direction).

3. **50-feature protocols have larger raw swings but worse coherence.** Conv rp (50) = 33.8pp @coh>=50 but C-bar=39.3. Conv ln1 (50) = 30.0pp @coh>=50 but only 9.5pp @coh>=70.

4. **FRA/hybrid protocols are consistent but not competitive.** FRA and hybrid 50-feature protocols cluster at ~21pp @coh>=50, ~10-12pp @coh>=70. FRA ranking does not improve over Wang contrastive for this dataset.

5. **resid_post collapses at 50 features.** Conv rp (50) C-bar=39.3 -- mean-act alpha points push too aggressively. ln1 is the viable hook point.

6. **Conv ln1 (1) steering is symmetric.** Both positive and negative alpha increase alignment (A-bar: 44.5 at alpha=0 to ~60-67 at alpha=+/-2), suggesting the contrastive direction captures a general alignment axis.

7. **FRA single-feature protocols are weak (~6-9pp).** These use FRA QK/OV attribution (not Wang contrastive) for feature selection, and the resulting features are less effective for steering.

## Comparison with Paper / 7B

- The paper's winning protocol was QK attribution + conventional steering (hybrid) at 50 features. Here, hybrid QK (50) achieves 21.3pp @coh>=50 -- decent but conventional ln1 single-feature (22.8pp) is stronger.
- **Conv ln1 (1) with contrastive ranking is the strongest at the honest coh>=70 threshold (13.4pp).** This simple baseline was missing from the paper.
- Dmitry's 7B finance results showed conventional resid_post dominating at ~66pp. Here rp (50) reaches 33.8pp @coh>=50 but coherence collapses. The ln1 hook point is the better choice for 14B sports.
- FRA/hybrid protocols are consistent (~21pp @coh>=50 for 50-feature) but don't outperform the conventional baseline.

**Caveat on single-feature numbers:** Our single-feature results test only the **top-1 ranked feature**. Dmitry's `phase1_grid_14b_orchestrator.py` (gran=1) sweeps **all 50 ranked features individually** and reports the best (winner's-curse max over ~50 draws, per PLAN.md). Not directly comparable.

## Open Questions for Discussion

## Open Questions for Discussion

1. **Is Conv ln1 (50) the right baseline?** It's the strongest protocol, but it's also the simplest (no FRA needed). The paper's story was that FRA attribution improves feature selection -- these results suggest it doesn't for 50-feature steering, at least on this dataset.

2. **EM-specificity:** We haven't run the base model comparison yet. From 7B, ln1 protocols were EM-specific (base ~3-5 vs EM ~10-20). If Conv ln1 (50) is also EM-specific on 14B sports, that's still a meaningful finding.

3. **Dataset dependence:** Finance results may tell a different story. The sports EM model has A-bar~43 at baseline (moderately misaligned). Finance may have stronger/weaker misalignment that changes the steering landscape.

## Files

- `head_ablation_delta_a_sports_l24.json` -- Phase 0.5 results
- `noise_check_sports.json` -- Phase 1 noise check
- `phase2/` -- V1 original 9-point grid, single-feature (wrong Wang ranking)
- `phase2_meanact/` -- V1 extended grid (wrong Wang ranking)
- `phase2_hybrid50/` -- V1 50-feature hybrid
- `phase2_conditional50/` -- V1 50-feature conditional
- `full_analysis.json` -- V1 analysis
- **`phase2_v2/`** -- V2 all 11 protocols with proper contrastive Wang ranking (CANONICAL)
- `phase2_v2/v2_analysis.json` -- V2 computed metrics
- `phase2_v2/wang_ranking_ln1.json` -- Contrastive ranking for ln1 (top feature: f603)
- `phase2_v2/wang_ranking_resid_post.json` -- Contrastive ranking for resid_post
