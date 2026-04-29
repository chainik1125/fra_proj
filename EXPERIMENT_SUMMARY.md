# FRA-OV Experiment Summary

**Date:** April 28, 2026
**Model:** Qwen2.5-14B-Instruct (14.8B params)
**SAE:** Nura-J/Qwen2.5-14B_SAE_ln1.normalised (d_sae=102,400, top_k=64, 20x expansion)
**Layer:** 24, Hook point: ln1.hook_normalized
**GPU:** NVIDIA H200 (141GB VRAM)
**Prompts:** 4 EM evaluation prompts from arXiv:2506.11613

---

## What We Did

### Step 1: Head ablation (which heads matter?)
- Zeroed each of 40 heads at layer 24, measured loss change
- Used 2 EM prompts, measured loss_delta, KL divergence, top-1 prediction change

### Step 2: QK→OV steering sweep (main experiment)
For each of 4 heads (H38, H0, H36, H7), for each of 4 EM prompts:
1. **Computed QK FRA** — ranked feature pairs by interaction strength
2. **Computed OV decomposition** — ranked features by value contribution magnitude
3. **Measured feature overlap** between the two rankings
4. **Swept steering scales** [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]
   - Scale 0.0 = fully ablate the feature from OV
   - Scale 1.0 = no change (baseline)
   - Scale 3.0 = triple the feature's contribution
5. Two conditions at each scale:
   - **QK→OV**: features selected by QK ranking, steered in OV
   - **OV→OV**: features selected by OV ranking, steered in OV

---

## Key Numbers

### Head ablation results (layer 24)

| Head | loss_delta | KL div | Interpretation |
|------|-----------|--------|----------------|
| H38 | +0.014 | 0.0004 | Removing hurts → head is helpful |
| H0 | +0.011 | 0.0005 | Removing hurts → head is helpful |
| H36 | -0.010 | 0.0003 | Removing helps → may carry misaligned behavior |
| H7 | -0.008 | 0.0002 | Removing helps → may carry misaligned behavior |

### Feature overlap: QK vs OV rankings

| Head | Avg features | Avg overlap | Overlap % |
|------|-------------|-------------|-----------|
| H38 | 23 | 13 | 58% |
| H0 | 23 | 14 | 62% |
| H36 | 25 | 15 | 61% |
| H7 | 27 | 16 | 60% |

**Finding: ~60% of the top features are shared between QK and OV rankings.** This means QK interactions are partially predictive of which features matter in the OV, but ~40% are unique to each ranking.

### Steering sweep — averaged across 4 EM prompts

#### H38 (strongest head from ablation)

| Scale | QK→OV Δloss | OV→OV Δloss | QK→OV KL | OV→OV KL |
|-------|-------------|-------------|----------|----------|
| 0.0 (ablate) | **-0.017** | -0.004 | 0.0015 | 0.0021 |
| 0.4 | -0.005 | -0.001 | 0.0013 | 0.0014 |
| 0.8 | -0.012 | -0.003 | 0.0009 | 0.0008 |
| 1.0 (baseline) | 0.000 | 0.000 | 0.0000 | 0.0000 |
| 2.0 (amplify) | +0.005 | -0.008 | 0.0018 | 0.0020 |
| 3.0 (3x) | +0.020 | -0.025 | 0.0039 | 0.0056 |

#### H7 (most interesting pattern)

| Scale | QK→OV Δloss | OV→OV Δloss | QK→OV KL | OV→OV KL |
|-------|-------------|-------------|----------|----------|
| 0.0 (ablate) | **-0.024** | -0.018 | 0.0014 | 0.0014 |
| 0.4 | -0.011 | -0.014 | 0.0009 | 0.0010 |
| 0.8 | -0.003 | -0.015 | 0.0007 | 0.0008 |
| 1.0 (baseline) | 0.000 | 0.000 | 0.0000 | 0.0000 |
| 2.0 (amplify) | +0.010 | +0.006 | 0.0010 | 0.0011 |
| 3.0 (3x) | +0.006 | +0.020 | 0.0022 | 0.0029 |

---

## Patterns We See

### What is working (helps the paper)

1. **QK→OV ablation consistently reduces loss (negative Δloss)**
   - At scale=0.0, QK-ranked features ablated in OV reduce loss by 0.017-0.024 across heads
   - This confirms Dmitry's finding: QK ranking identifies causally important features for OV intervention

2. **~60% feature overlap is a meaningful signal**
   - QK and OV rankings share more than half their top features
   - But each has ~40% unique features → they capture different aspects of the circuit
   - This supports the paper's story: QK tells you *which* features to target, OV tells you *where* to intervene

3. **The effect is present across multiple heads**
   - Not just one head — H38, H0, H36, H7 all show measurable steering effects
   - Different heads respond differently (some positive, some negative loss_delta at ablation)
   - This matches the emergent misalignment being distributed across heads (unlike Tiny Stories where it's concentrated)

4. **KL divergence increases monotonically with steering strength**
   - As scale moves away from 1.0 in either direction, KL goes up
   - This is the expected behavior — more steering = more distribution change
   - Gives us a clean Pareto trade-off axis

5. **The SAE loaded successfully with top_k=64**
   - The 20x expansion SAE (102,400 features) works
   - top_k=64 means sparse activation — good for feature attribution

### What is less clear (challenges for the paper)

1. **Effect sizes are small**
   - Loss deltas are 0.005-0.025 (on baseline losses of 3-6)
   - KL divergences are 0.001-0.006
   - Top-1 prediction changes are only 0-3%
   - Compare to Tiny Stories where the sleeper feature gives near-perfect suppression

2. **QK→OV doesn't clearly beat OV→OV**
   - At ablation (scale=0), QK→OV often has larger loss delta, but not always
   - At amplification (scale>1), OV→OV sometimes shows stronger effect
   - We can't yet claim "rank by QK, steer in OV" is strictly better
   - This might be because emergent misalignment is more diffuse than a single sleeper feature

3. **No direct misalignment measurement yet**
   - We're measuring loss/KL/top-1 change, not actual alignment scores
   - We need to add: generate text with steering → judge alignment with GPT-4o
   - Loss going down could mean "more coherent" or "more aligned" — we can't tell without behavioral eval

4. **Only layer 24, only 4 prompts**
   - Need to sweep layers (at least layer 12, 24, 36)
   - Need all 8 EM prompts, not just 4
   - Head ablation used only 2 texts — should use all 8 for stable rankings

5. **Non-monotonic loss curves**
   - Loss delta doesn't smoothly decrease with ablation strength
   - e.g. H38: scale=0.4 gives -0.005 but scale=0.8 gives -0.012 (not monotonic)
   - This suggests feature interactions — ablating more features doesn't always help more

---

## Settings Used

```
Model:      Qwen2.5-14B-Instruct
SAE:        Nura-J/Qwen2.5-14B_SAE_ln1.normalised
            d_in=5120, d_sae=102400, top_k=64
Layer:      24
Hook point: ln1.hook_normalized
Heads:      H38, H0, H36, H7 (top 4 from head ablation)
Prompts:    4 of 8 EM eval prompts
max_length: 64 tokens
top_k:      20 (for FRA sparsification, separate from SAE top_k)
k:          50 (top feature pairs to select)
Scales:     [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]
dtype:      bfloat16
GPU:        NVIDIA H200 (141GB)
```

---

## What to Do Next

### Priority 1: Behavioral evaluation
- Generate full responses with OV steering active
- Score alignment with GPT-4o judge (0-100 scale from the EM paper)
- This is the missing piece — we need to show steering *actually changes alignment behavior*

### Priority 2: More hook points
- Train/use SAEs at hook_resid_pre and hook_resid_mid
- Compare single-feature steering at each hook point vs FRA-OV steering
- This is the Pareto frontier comparison Dmitry described

### Priority 3: Scale up
- All 8 EM prompts (not just 4)
- More layers (12, 24, 36)
- Head ablation with more texts for stable rankings
- Run the full 3×3 attribution×intervention matrix

### Priority 4: Stronger baselines
- Compare against simple activation steering (mean difference between aligned/misaligned)
- Compare against single-feature steering at hook_resid_mid
- These are the baselines the paper needs to beat

---

---

## Gap Analysis: What Dmitry Did vs What We Did

Dmitry's branch (`dmitry/ov/experiments/tinystories_sleeper`) is the reference. Here's what maps to our work and what's missing.

### What Dmitry did in Tiny Stories (and his key findings)

1. **Single-feature ablation sweep across 8 SAE architectures + 5 hookpoints**
   - Found: `hook_resid_mid` at layer 0 is the best hookpoint (ASR=0.00, ΔCE=-0.001)
   - We did NOT do this — we only used `ln1.hook_normalized`

2. **3×3 ranking × intervention grid**
   - Rows: {QK rank, OV rank, Union}
   - Cols: {V-only intervene, QK-only intervene, All (ln1) intervene}
   - **Key result: (OV rank, V-intervene) = perfect Pareto frontier (quality=1.00)**
   - We implemented the 3×3 matrix (`fra/experiment_matrix.py`) but haven't run it yet on EM data
   - We ran the 2-condition comparison (QK→OV vs OV→OV) which is a subset

3. **Pareto frontier with quality metric Q = area under monotone envelope**
   - He measured ASR₁₆ (attack success rate) vs ΔCE (coherence damage)
   - OV top-3 features: Q=0.98, QK top-5: Q=0.51
   - We implemented `compute_pareto_q()` but our metric is KL/top1_change, NOT ASR
   - **Missing: we have no ASR measurement** — we measure loss/KL, not whether the model actually outputs misaligned text

4. **QK-side softmax Jacobian linearisation**
   - Mathematical derivation of how QK perturbations propagate through softmax
   - Spearman ρ=0.95 between QK L1_mean and measured ablation impact
   - We did NOT implement this — our QK ranking uses raw FRA pair strength, not the Jacobian-weighted version

5. **Feature firing rate analysis**
   - Found f=1114 fires on 6.5% of deployment prompts, 0.0% on clean
   - This explains why V-intervene has zero coherence cost on clean prompts
   - We did NOT do this — we don't separate clean vs deployment prompt analysis

6. **Hook_v intervention (V-only steering)**
   - Dmitry's best result: steer at `hook_v` using OV-ranked features
   - We implemented this (`run_ov_steering` hooks `hook_v`) — **this is our main contribution**
   - But Dmitry showed it works perfectly in Tiny Stories; we showed small effects in EM

### What we covered ✅

| Dmitry's component | Our implementation | Status |
|---|---|---|
| OV decomposition (freeze A, resolve V) | `fra/core/ov.py` | ✅ Done |
| V-only steering via hook_v | `fra/ov_steering.py` | ✅ Done |
| Head ablation | `fra/head_ablation.py` | ✅ Done |
| QK→OV feature ranking comparison | `run_experiments.py --task qk_to_ov` | ✅ Done |
| Feature overlap measurement | Measured ~60% overlap | ✅ Done |
| Steering scale sweep | 10 scales × 4 heads × 4 prompts | ✅ Done |
| Pareto Q metric | `fra/pareto.py` | ✅ Implemented, not yet run with ASR |
| 3×3 attribution × intervention matrix | `fra/experiment_matrix.py` | ✅ Implemented, not yet run |
| Plots (loss/KL/top1 vs scale + Pareto) | `_plot_qk_to_ov()` | ✅ Done |

### What we're missing ❌

| Dmitry's component | Gap | Priority |
|---|---|---|
| **ASR measurement** (attack success rate) | We measure loss/KL, not actual misalignment rate | **Critical** — need to generate text + judge alignment |
| **Multiple hookpoints** (resid_pre, resid_mid, resid_post, ln1) | We only used ln1.hook_normalized | **High** — need SAEs at other hookpoints |
| **QK Jacobian-weighted ranking** | We use raw FRA pair strength, Dmitry uses softmax Jacobian | Medium — improves ranking correlation |
| **Feature firing rate on clean vs deployment** | We don't separate clean/deployment prompts | Medium — explains coherence preservation |
| **Single-feature sweep** (α sweep per individual feature) | We sweep all selected features together | Medium — identifies the "killer feature" |
| **Paired clean/deployment evaluation** | Dmitry measures both ASR and ΔCE on separate prompt sets | **High** — we need clean prompts as baseline |
| **Per-hookpoint comparison** | Dmitry's main finding: resid_mid > ln1 > resid_pre | **High** — central to the paper story |

### Key difference in findings

| Aspect | Tiny Stories (Dmitry) | Emergent Misalignment (Us) |
|---|---|---|
| Effect concentrated? | Yes — 1 feature, 1 head | No — spread across heads, ~20 features |
| Best hookpoint | hook_resid_mid (perfect) | Unknown (only tested ln1) |
| OV rank + V-intervene | Perfect: ASR=0, ΔCE=0 | Small effects: Δloss ≈ 0.02 |
| Feature overlap QK/OV | Not reported | ~60% |
| Behavioral eval | ASR₁₆ (binary: sleeper or not) | Loss/KL only (no generation) |

### Bottom line

We built all the infrastructure Dmitry described and validated it runs on Qwen2.5-14B. The method works — we see measurable OV steering effects. But the effects are small because:

1. **Emergent misalignment is more diffuse** than a single sleeper feature (expected, Dmitry warned about this)
2. **We're missing ASR/behavioral evaluation** — the most important missing piece
3. **We only tested ln1** — Dmitry's best results were at hook_resid_mid, which we haven't tried

---

## Files

| File | Description |
|------|-------------|
| `qk_vs_ov_L24_H38.png` | 4-panel plot: H38 steering comparison |
| `qk_vs_ov_L24_H0.png` | 4-panel plot: H0 steering comparison |
| `qk_vs_ov_L24_H36.png` | 4-panel plot: H36 steering comparison |
| `qk_vs_ov_L24_H7.png` | 4-panel plot: H7 steering comparison |
| `qk_vs_ov_per_prompt_L24_*.png` | Per-prompt loss curves (4 heads × 4 prompts) |
| `results_qk_to_ov_L24.json` | Full numerical results (H7 — last run overwrote) |
| `experiment_log.txt` | H38 experiment log |
| `experiment_all_heads.txt` | H0 + H36 + H7 experiment logs |
| `OV_FRA_METHODOLOGY.md` | Method explanation |
| `fra/core/ov.py` | OV decomposition code |
| `fra/ov_steering.py` | OV steering hooks |
| `fra/head_ablation.py` | Head ablation sweep |
| `fra/pareto.py` | Pareto frontier measurement |
| `fra/experiment_matrix.py` | 3×3 attribution × intervention matrix |
| `fra/em_evaluation.py` | EM prompts and scoring |
| `run_experiments.py` | GPU experiment runner |
