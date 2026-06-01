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

### Results (V2 — with proper contrastive Wang ranking)

Full analysis: `experiments/fra_14b_sports/phase2_v2/v2_analysis.json`

**Comparison table — Δalign (max A̅ - min A̅ across per-alpha means):**

(`—` = no alpha point reached that coherence threshold)

| Protocol | Δ@coh≥50 | Δ@coh≥70 | C̅ mean |
|----------|---------|---------|--------|
| Conv rp (50) | 33.8 | 5.1 | 39.3 |
| Conv ln1 (50) | 30.0 | 9.5 | 66.6 |
| **Conv ln1 (1)** | **22.8** | **13.4** | **69.4** |
| Hyb OV (50) | 21.6 | 10.5 | 67.8 |
| Hyb QK (50) | 21.3 | 11.8 | 68.5 |
| FRA OV (50) | 21.0 | 10.3 | 67.7 |
| Conv rp (1) | 13.7 | — | 61.6 |
| Hyb QK (1) | 9.5 | — | 66.0 |
| FRA QK (1) | 8.8 | — | 65.9 |
| FRA OV (1) | 6.8 | — | 66.1 |
| Hyb OV (1) | 5.9 | — | 66.1 |

**Key findings (V2):**
1. **Conv ln1 (1) is the surprise winner at coh≥70** — 13.4pp with C̅=69.4. Single-feature with the right contrastive feature beats all 50-feature protocols at the honest coherence threshold.
2. **Conv ln1 (1) reaches A̅=67.3 at α=-2 with C̅=78.2** — strong alignment recovery while maintaining high coherence.
3. **Fixing the Wang ranking was critical** — Conv ln1 (1) jumped from 7.0pp (V1, wrong ranking) to 22.8pp (V2, contrastive). The contrastive feature (f603, Δf=0.94) is much more effective than the wrong one.
4. **50-feature protocols are strong but not dominant** — Conv ln1/rp (50) have larger swings @coh≥50 (30-34pp) but coherence suffers. At coh≥70, single-feature Conv ln1 wins.
5. **Conv rp (50) has biggest raw swing (33.8pp @coh≥50)** but C̅=39.3 — coherence collapse at extreme alphas.
6. **FRA/hybrid 50-feature ≈ 21pp @coh≥50** — consistent across QK/OV, ~10-12pp @coh≥70.
7. **FRA single-feature protocols unchanged** (6-9pp) — they don't use Wang ranking.

**Caveat:** Single-feature results test **top-1 ranked feature only**. Dmitry's code (gran=1) sweeps all 50 features individually and reports the best (winner's-curse max over ~50 draws). Not directly comparable.

### What ran
- **V1** (wrong Wang ranking): `phase2/`, `phase2_meanact/`, `phase2_hybrid50/`, `phase2_conditional50/`
- **V2** (fixed contrastive Wang ranking): `phase2_v2/` — all 11 protocols re-run

---

## Phase 3: Analysis & Comparison

- [x] **3.1** QK→QK (pure FRA) effect: 8.8pp @coh≥50 (V2) — weak. Below conventional single-feature (22.8pp).
- [x] **3.2** QK attribution → conventional steering vs conventional baseline (V2):
  - Hyb QK (1) = 9.5pp vs Conv ln1 (1) = 22.8pp → **conventional wins decisively for single-feature** (with proper contrastive ranking)
  - Hyb QK (50) = 21.3pp vs Conv ln1 (50) = 30.0pp → conventional also wins at 50-feature
- [x] **3.3** Alignment swing at coh thresholds: Conv ln1 (1) is best at coh≥70 (13.4pp). All 50-feature ln1 protocols also reach coh≥70 (~10-12pp).
- [x] **3.4** FRA/hybrid vs conventional:
  - **Single-feature**: conventional ln1 **dominates** (22.8pp vs 9.5pp hybrid QK)
  - **50-feature**: conventional ln1 still leads (30.0pp vs 21.3pp hybrid QK)
  - **At coh≥70**: Conv ln1 (1) = 13.4pp > Hyb QK (50) = 11.8pp — single-feature with right feature > 50-feature with FRA ranking
  - **Story**: proper contrastive feature selection is what matters. FRA ranking doesn't improve over Wang contrastive for this dataset.
- [x] **3.5** Write-up: `experiments/fra_14b_sports/RESULTS.md`

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
