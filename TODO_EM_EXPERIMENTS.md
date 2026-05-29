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

**RESOLVED:** resid_post native L0=4780 vs target k=64 — BatchTopK threshold drift at inference. Fix: **force hard TopK(64) cap at inference** (per `measure_sae.sh`: "BatchTopK eval-threshold is miscalibrated").

---

## Phase 0.5: Head Ablation + ‖Δa‖ for Extreme Sports — DONE

- [x] **0.5a** Ran `experiments/fra_14b_sports/head_delta_a.py` on RunPod (H100 80GB)
- [x] **0.5b** Results:
  - **Top head: H12** (Δloss=0.0098), top-5: [12, 33, 0, 17, 24]
  - **‖Δa‖ resid_post L24 = 14.749**
  - **‖Δa‖ ln1 postgain L24 = 5.562**
  - (Finance was: head 12, ‖Δa‖_rp=12.189 — same head, similar magnitude)

---

## Phase 1: Noise Control (Quality Gate) — DONE

**Config used:**
- Target model temp = 1.0, judge = gpt-4o-mini @ temp=0
- 8 prompts × 4 samples = 32 responses per seed, seeds {42, 123}
- Hooks active during autoregressive rollout (confirmed: `generate_with_hooks` loops token-by-token)

**Results (PASS):**

| Seed | A̅ | C̅ |
|------|-----|-----|
| 42 | 44.1 | 66.2 |
| 123 | 41.4 | 62.0 |
| **Mean** | **42.7** | **64.1** |
| **Max dev** | **±1.3pp** | **±2.1pp** |

Verdict: **PASS** — both well within ±5pp target. No tuning needed.

**Cost:** ~$0.024 per steering point (64 gpt-4o-mini calls). Very cheap — full 9-point grid is ~$0.22 per cell.

**Baseline observation:** A̅≈43 at alpha=0 means the sports EM model is already moderately misaligned at baseline (judge sees ~43/100 alignment).

---

## Phase 2: Steering Experiments

All on Qwen2.5-14B **fine-tuned (EM) extreme-sports model**.

### Locked parameters (from Phase 0.5 + Phase 1)
- **Head: H12** at L24
- **‖Δa‖ resid_post = 14.749**, ‖Δa‖ ln1 = 5.562
- **Steering: magnitude-matched** → `α_nom · ‖Δa‖ · unit(dir)`
- **Alpha grid:** -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2 (9 points)
- **Target temp = 1.0**, judge = gpt-4o-mini @ temp=0
- **N = 4** samples per prompt, **8 prompts** = 32 responses per (alpha, feature) cell
- **Seeds = {42, 123}**, MAX_NEW_TOKENS = 100
- **Hard TopK(64) cap** on SAE encode at inference
- **Cost:** ~$0.22 per (alpha-sweep × seed-pair) cell. Budget ~$5-10 total for Phase 2A+2B.

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
- **Key summary stat**: Δalign = max(A̅) - min(A̅) across alpha range, at coherence threshold 50 and 70

### Implementation
Script: `experiments/fra_14b_sports/run_phase2_steering.py`

```bash
# 2A: All single-feature protocols
python experiments/fra_14b_sports/run_phase2_steering.py --protocols all_single

# 2B: 50-feature hybrid
python experiments/fra_14b_sports/run_phase2_steering.py --protocols hybrid_50

# 2C: Conditional 50-feature rest
python experiments/fra_14b_sports/run_phase2_steering.py --protocols conditional_50
```

Requires: `dictionary_learning` package (for Arditi SAE loading), `openai` (for judge).

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
