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

## Phase 2: Steering Experiments — DONE

All on Qwen2.5-14B **fine-tuned (EM) extreme-sports model**.

### Config
- Head H12 @ L24, magnitude-matched steering (`α · ‖Δa‖ · unit(dir)`)
- ‖Δa‖ resid_post = 14.749, ‖Δa‖ ln1 = 5.562
- Extended alpha grid: 9 ‖Δa‖ points + ±1×,2×,3× mean feature activation (15 total)
- Target temp=1.0, judge=gpt-4o-mini@temp=0, 8 prompts × 4 samples, seeds {42, 123}

### Results

Full analysis: `experiments/fra_14b_sports/full_analysis.json`

**Comparison table — Δalign (max A̅ - min A̅ across per-alpha means):**

| Protocol | Δ@coh≥50 | Δ@coh≥70 | C̅ mean |
|----------|---------|---------|--------|
| **Conv ln1 (50)** | **34.1** | **24.3** | 67.5 |
| Hyb QK (50) | 20.9 | 11.6 | 68.6 |
| Hyb OV (50) | 20.9 | 9.8 | 67.7 |
| FRA OV (50) | 20.6 | 9.7 | 67.9 |
| Conv rp (50) | 17.3 | — | 40.2 |
| Conv rp (1, ext) | 14.1 | — | 61.7 |
| FRA QK (1) | 9.5 | — | 65.9 |
| Hyb QK (1) | 9.1 | — | 66.2 |
| Conv ln1 (1) | 7.0 | — | 65.2 |
| Hyb OV (1) | 7.0 | — | 66.2 |
| FRA OV (1) | 6.7 | — | 66.1 |

**Key findings:**
1. **Conv ln1 (50) dominates** — 34.1pp @coh≥50, 24.3pp @coh≥70. Reaches A̅=78 at α=-6.63 with C̅=80.8.
2. **All 50-feature protocols >> all single-feature** (~2-5× larger swings).
3. **Hybrid 50 ≈ FRA OV 50** (~20-21pp) — FRA ranking doesn't clearly beat Wang for 50-feature.
4. **Conv resid_post (50) collapses** — C̅=40.2 mean, coherence dies at large α.
5. **coh≥70 separates the field** — only ln1-based 50-feature protocols survive.
6. **Single-feature steering is weak** — 7-14pp swings, barely above inter-seed noise (~6pp at α=0).

### What ran
- **2A** (single-feature, 6 protocols): `phase2/` + `phase2_meanact/` (‖Δa‖ grid, then extended with mean-act points)
- **2B** (50-feature hybrid, 2 protocols): `phase2_hybrid50/` — interesting → triggered 2C
- **2C** (50-feature conditional, 3 protocols): `phase2_conditional50/`

---

## Phase 3: Analysis & Comparison

- [x] **3.1** QK→QK (pure FRA) effect: 9.5pp @coh≥50 — not negligible, but weak. Comparable to hybrid QK (9.1pp). Both well below 50-feature protocols.
- [x] **3.2** QK attribution → conventional steering vs conventional baseline:
  - Hyb QK (1) = 9.1pp vs Conv ln1 (1) = 7.0pp → FRA ranking gives marginal improvement for single-feature
  - Hyb QK (50) = 20.9pp vs Conv ln1 (50) = 34.1pp → **conventional wins at 50-feature**
- [x] **3.3** Alignment swing at coh thresholds: see table above. Only 50-feature ln1 protocols reach coh≥70.
- [x] **3.4** FRA/hybrid vs conventional:
  - **Single-feature**: FRA/hybrid marginally better (9.1-9.5pp vs 7.0pp) but all are weak
  - **50-feature**: conventional ln1 **dominates** (34.1pp vs 20.9pp hybrid)
  - **Story**: FRA ranking helps for single-feature, but conventional 50-feature additive steering at ln1 is the strongest protocol overall
- [ ] **3.5** Write up findings for Dmitry:
  - The headline result is Conv ln1 (50) with 24.3pp @coh≥70
  - FRA hybrid 50-feature is competitive but not superior (~21pp @coh≥50)
  - Single-feature steering is too weak to be meaningful (~7-10pp, near noise floor)
  - resid_post collapses at 50-feature — ln1 is the viable hook point
  - Need to compare with the paper's original results and reconcile

---

## Phase 4: Extensions (IF TIME PERMITS — do only after Phase 3 is solid)

- [ ] **4.1** Repeat Phase 2 on **base model** (not fine-tuned)
  - Expect: FRA may underperform on base (that's an interesting finding)
  - EM-specificity diagnostic: does ln1 steering only work on fine-tuned model?
- [x] **4.2** ~~Repeat for risky-financial-advice dataset~~ — done (Dmitry's 14B financial campaign on `autoresearch/wang-steering-7b`)
- [ ] **4.3** Repeat for **medical** dataset
- [ ] **4.4** If all 3 datasets done: cross-dataset comparison table for paper

---

## Execution Notes

- **Parallelize** where possible: SAE training for resid_post and ln1 can run simultaneously; different steering protocols are independent once SAE is ready
- **RunPod**: use existing credits; if credits run low, focus on the REQUIRED rows only
- **Push scripts to repo** so Dmitry/Hemang can run in parallel if needed
- **Existing campaign script**: `run_14b_campaign.sh` — will need updating to match this new protocol (different steering grid, noise control, single-feature baselines)
- **Cost awareness**: measure judge API cost early (Phase 1.3) before committing to large grids
