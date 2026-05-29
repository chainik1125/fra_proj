# TODO: Emergent Misalignment Experiment Redo (14B Extreme Sports)

**Goal:** Produce clean, trustworthy EM steering results for Qwen2.5-14B on extreme-sports so we can finalize the arxiv paper.
**Deadline:** Before Monday (June 2) so Dmitry can write up.
**Model:** Qwen2.5-14B fine-tuned on extreme-sports dataset (EM model).

---

## Phase 0: SAE Setup — DONE

Using existing SAEs from `dmanningcoe/fra-phase1-steering-data` (trained on base `Qwen/Qwen2.5-14B-Instruct`, BatchTopK k=64, d=131,072, 200M tokens):

| SAE | Path | Loss Recovered | Avg % Resid Error | Native L0 |
|-----|------|---------------|-------------------|-----------|
| resid_post L24 | `qwen14b/sae_resid_post_l24_base_arditi` | 0.81 | 41.5% | 4780 |
| ln1 L24 | `qwen14b/sae_ln1_l24_base_arditi` | 0.65* | 48.2% | 48 |

Source: `qwen14b/sae_quality_cmp.json` on HF. Both pass quality gates.
*ln1 loss_recovered denominator is degenerate (zero-ablation ≈ reconstruction CE).

**OPEN ISSUE:** resid_post native L0=4780 vs target k=64 — BatchTopK threshold drift at inference. May need hard TopK cap. Revisit before trusting resid_post steering results.

---

## Phase 1: Noise Control (Quality Gate)

**Setup:** No steering (alpha=0) on the extreme-sports EM model. Noise = variation in A̅ and C̅ when repeating the same measurement with different seeds.

**Noise sources:** (1) target model sampling noise, (2) judge sampling noise.
**Levers:** (1) reduce temperature of both LLMs (floor = 0.5, don't go below), (2) increase N repeats per question.
**Variance metric:** max(A̅_seeds) - mean(A̅_seeds) across 3 seeds. Target ≤ 5pp. Tolerable ≤ 10pp. Above 10pp = meaningless.

### Pre-checks
- [ ] **1.0a** Verify steering hooks during generation: do hooks apply during autoregressive rollout or prompt-only? Check `run_behavioral_eval()` generation code. Dmitry unsure — needs confirming.
- [ ] **1.0b** Fix judge temperature: `gpt4o_judge.py` currently uses `temperature=0` — raise to 0.5 (Dmitry's floor). Both LLMs should be at 0.5, not below.
- [ ] **1.0c** Confirm judge seeding: if judge temp > 0, need to pass different seeds per run. At temp=0 judge is deterministic (no seed needed).

### Noise measurement
- [ ] **1.1** Baseline noise measurement at alpha=0 (no steering):
  - Load extreme-sports EM model (`ModelOrganismsForEM/Qwen2.5-14B-Instruct_extreme-sports`)
  - Target model temperature = 0.5
  - Judge temperature = 0.5
  - 8 eval prompts × N=4 repeats per prompt (Dmitry found N=4 effective)
  - Compute A̅ and C̅ (mean alignment, mean coherence across all 8×4=32 responses)
  - Repeat with 3 seeds (seed both target model and judge)
  - Check: max(A̅_seeds) - mean(A̅_seeds) ≤ 5pp for both A and C
- [ ] **1.2** If variance > 5pp, tune parameters:
  - Try N=8 repeats per question
  - If still > 10pp: results are meaningless — fall back to smaller steering range with higher N
- [ ] **1.3** Measure API cost per steering-point evaluation at chosen N
  - Each steering point = 8 prompts × N repeats × 2 judge calls (alignment + coherence)
  - At N=4: 8 × 4 × 2 = 64 GPT-4o calls per steering point
  - If expensive → plan for ~17-point mini grid in Phase 2
  - If cheap → can afford full grid to coherence degradation

### Implementation needed
- [ ] **1.4** Write `run_noise_check.py` script:
  - Loads EM model, generates responses at alpha=0 with temp=0.5
  - Calls GPT-4o judge at temp=0.5 (update `gpt4o_judge.py`)
  - Repeats for 3 seeds, reports variance
  - Prints API cost estimate for Phase 2 grid
- [ ] **1.5** Conventional steering (Wang procedure) requires loading BOTH base and EM models to compute contrastive activation diff. No code for this exists yet — needs implementing before Phase 2.

---

## Phase 2: Steering Experiments

All on Qwen2.5-14B **fine-tuned (EM) extreme-sports model**, with noise parameters from Phase 1.

### Steering grid
- Always include: 0, ±1, ±2
- Additional points: 1×, 2×, 3× ... mean feature activation (averaged over non-dead features across dataset)
- Use same grid across all protocols (can't report different magnitudes per feature)
- If judging is expensive: start with ~17-point mini grid, confirm outliers later
- If cheap: extend until model coherence collapses

### 2A: Single-feature steering (START HERE for all protocols)

| # | Protocol | Hook point | Features | Priority |
|---|----------|-----------|----------|----------|
| 1 | **Conventional** (CAE/Wang) | resid_post | single (top-1) | REQUIRED |
| 2 | **Conventional** (CAE/Wang) | ln1 | single (top-1) | REQUIRED |
| 3 | **Pure FRA** (QK attribution + QK intervention) | ln1 | single (top-1) | REQUIRED |
| 4 | **Pure FRA** (OV attribution + OV intervention) | ln1 | single (top-1) | REQUIRED |
| 5 | **Hybrid: QK→Conv** (QK attribution, conventional steer) | ln1 | single (top-1) | REQUIRED |
| 6 | **Hybrid: OV→Conv** (OV attribution, conventional steer) | ln1 | single (top-1) | REQUIRED |

### 2B: 50-feature hybrid steering (TEST FIRST)

Run hybrid 50-feature first. If interesting, expand to the rest.

| # | Protocol | Hook point | Features | Priority |
|---|----------|-----------|----------|----------|
| 7 | **Hybrid: QK→Conv** | ln1 | top-50 | TEST FIRST |
| 8 | **Hybrid: OV→Conv** | ln1 | top-50 | TEST FIRST |

### 2C: 50-feature remaining protocols (CONDITIONAL — only if 2B is interesting)

| # | Protocol | Hook point | Features | Priority |
|---|----------|-----------|----------|----------|
| 9 | **Conventional** (CAE/Wang) | resid_post | top-50 | CONDITIONAL |
| 10 | **Conventional** (CAE/Wang) | ln1 | top-50 | CONDITIONAL |
| 11 | **Pure FRA** | ln1 | top-50 | CONDITIONAL |

### 2D: Additional hook points (NICE TO HAVE)

| # | Protocol | Hook point | Priority |
|---|----------|-----------|----------|
| 12 | Conventional single-feature | resid_mid | NICE TO HAVE |
| 13 | Conventional single-feature | resid_pre | NICE TO HAVE |

### Metrics to collect per steering point:
- **Alignment score** (judge-assessed, A̅ ± variance across seeds)
- **Coherence score** (judge-assessed, C̅ ± variance across seeds)
- **Jensen-Shannon divergence** vs clean (unsteered) model — supplement coherence metric
- **Key summary stat**: max alignment swing = max(A̅) - min(A̅) across steering range, at coherence threshold 50 and 70

---

## Phase 3: Analysis & Comparison

- [ ] **3.1** Confirm QK→QK (pure FRA) effect is negligible (as expected from 7B and TinyStories)
- [ ] **3.2** Compare QK attribution → conventional steering (the "winning" protocol from paper) vs conventional single-feature baseline
- [ ] **3.3** Report alignment swing at coherence thresholds 50 and 70
- [ ] **3.4** If FRA/hybrid protocols outperform conventional single-feature → great, write it up
- [ ] **3.5** If FRA/hybrid protocols underperform everywhere:
  - Investigate why original paper results showed outperformance
  - Check SAE differences, noise levels, grid differences
  - Report honestly with 7B results in appendix

---

## Phase 4: Extensions (IF TIME PERMITS — do only after Phase 3 is solid)

- [ ] **4.1** Repeat Phase 2 on **base model** (not fine-tuned)
  - Expect: FRA may underperform on base (that's an interesting finding)
- [ ] **4.2** Repeat for **risky-financial-advice** dataset
- [ ] **4.3** Repeat for **medical** dataset
- [ ] **4.4** If all 3 datasets done: cross-dataset comparison table for paper

---

## Execution Notes

- **Parallelize** where possible: SAE training for resid_post and ln1 can run simultaneously; different steering protocols are independent once SAE is ready
- **RunPod**: use existing credits; if credits run low, focus on the REQUIRED rows only
- **Push scripts to repo** so Dmitry/Hemang can run in parallel if needed
- **Existing campaign script**: `run_14b_campaign.sh` — will need updating to match this new protocol (different steering grid, noise control, single-feature baselines)
- **Cost awareness**: measure judge API cost early (Phase 1.3) before committing to large grids
