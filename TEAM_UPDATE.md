# FRA-OV Update — April 28

Hi all,

Here's where we are with the OV extension on emergent misalignment.

## What we ran

- **Model:** Qwen2.5-14B-Instruct on an H200 (141GB VRAM)
- **SAE:** Our ln1.hook_normalized SAE (d_sae=102,400, top_k=64, layer 24)
- **Prompts:** 4 of the 8 EM eval prompts from the model organisms paper

### Experiments completed

1. **Head ablation** (layer 24, all 40 heads)
   - H38 and H0 have the largest positive loss_delta (removing them hurts the model)
   - H36 and H7 have negative loss_delta (removing them improves loss — possible misalignment carriers)
   - Effects are small and spread across heads — no single dominant head like in Tiny Stories

2. **QK→OV steering sweep** (4 heads × 4 prompts × 10 steering scales)
   - For each prompt: compute QK FRA, rank feature pairs, also compute OV decomposition and rank independently
   - ~60% feature overlap between QK and OV rankings (consistent across all heads)
   - Swept scales from 0.0 (full ablation) to 3.0 (3× amplification)
   - Both QK-ranked and OV-ranked features produce measurable loss/KL changes when steered in OV via hook_v

### What "QK→OV" vs "OV→OV" means

Both conditions steer in the same place (hook_v — the value vectors). The difference is **how we pick which features to steer**:

- **QK→OV**: rank features by their QK FRA interaction strength (which features drive attention patterns), then ablate/scale those features in the OV path
- **OV→OV**: rank features by their OV contribution magnitude (which features contribute most to the value output), then ablate/scale those in the OV path

Dmitry's Tiny Stories result: OV-ranked features steered via hook_v gave a perfect Pareto frontier. We're testing whether the same holds for emergent misalignment.

### Key numbers (averaged across 4 prompts, H38)

| Scale | QK→OV Δloss | OV→OV Δloss | QK→OV KL | OV→OV KL |
|-------|------------|------------|---------|---------|
| 0.0 | -0.017 | -0.004 | 0.0015 | 0.0021 |
| 0.8 | -0.012 | -0.003 | 0.0009 | 0.0008 |
| 1.0 | 0.000 | 0.000 | 0.000 | 0.000 |
| 2.0 | +0.005 | -0.008 | 0.0018 | 0.0020 |
| 3.0 | +0.020 | -0.025 | 0.0039 | 0.0056 |

Plots for all 4 heads are in the repo (qk_vs_ov_L24_H*.png).

## What's working

- The OV decomposition and hook_v steering pipeline works end-to-end on 14B
- QK and OV rankings share ~60% of features — partial but meaningful overlap
- Ablating QK-ranked features in OV consistently reduces loss
- KL divergence increases monotonically with steering strength (clean trade-off curve)
- The GQA head mapping, SAE loading from HF, and all the infrastructure is validated

## What's not working yet / what's missing

1. **Effect sizes are small** — loss deltas of 0.005-0.025 on baseline losses of 3-6. This is expected since EM is more diffuse than a sleeper agent, but it also means we can't tell yet whether the steering actually changes alignment behavior or just perturbs predictions slightly.

2. **No behavioral evaluation** — we're measuring loss/KL/top-1 change, not actual alignment. We need to generate full text responses with steering active and judge them (GPT-4o judge from the EM paper, 0-100 alignment score). This is the most important missing piece.

3. **Only tested ln1.hook_normalized** — Dmitry's Tiny Stories results show hook_resid_mid is the best intervention point. We need SAEs at hook_resid_pre and hook_resid_mid to compare. **@Hemang** — can you train/share SAEs at those hookpoints for Qwen2.5-14B?

4. **Only 4 of 8 EM prompts** — should run all 8 for the paper.

## Standardising our evaluation set

We should agree on a shared train/test split to avoid two problems:
- **Contamination:** if the SAE was trained on examples we then use for evaluation, the results are unreliable. We need to make sure the eval prompts were NOT in the SAE training data.
- **Consistency:** if Hemang, Nura, and Dmitry each use different prompts, we can't compare results across hookpoints or models.

**Proposal:**
- **Eval set:** the 8 long-form + 8 template prompts from the EM paper (`first_plot_questions.yaml`). These should be held out from all SAE training.
- **SAE training set:** whatever activation data we use, explicitly exclude the eval prompts (or at minimum document which data the SAE saw).
- **Judge:** GPT-4o with the `judges.yaml` rubric from the EM repo (alignment 0-100, coherence 0-100).

Can everyone confirm their SAE training data does NOT include the 8 EM eval prompts? If it does, we need a separate held-out set.

## What's needed next

| Task | Who | Blocking on |
|------|-----|-------------|
| **Agree on eval/train split** | Everyone | Need confirmation from Hemang on SAE training data |
| SAEs at hook_resid_pre and hook_resid_mid (layer 24, Qwen2.5-14B) | Hemang? | GPU time |
| Behavioral eval (generate + GPT-4o judge) | Nura | Need to implement generation with hooks active |
| Run full 3×3 attribution × intervention matrix | Nura | SAEs at other hookpoints |
| Compare hookpoint Pareto frontiers (the key plot for the paper) | Nura | All of the above |
| All 8 EM prompts | Nura | Quick, just need another run |

## Code

Everything is on `nura/dev`. Key new files:
- `fra/core/ov.py` — OV decomposition
- `fra/ov_steering.py` — hook_v steering (handles GQA)
- `fra/head_ablation.py` — head importance sweep
- `fra/pareto.py` — Pareto frontier + Q metric
- `fra/experiment_matrix.py` — 3×3 grid (ready to run)
- `fra/em_evaluation.py` — EM prompts + judge template
- `run_experiments.py` — RunPod experiment runner
- `OV_FRA_METHODOLOGY.md` — method writeup
- `EXPERIMENT_SUMMARY.md` — full results + gap analysis vs Dmitry's Tiny Stories

## Running on RunPod

Setup takes ~5 min once you know the steps:
- H200 pod, 100GB container disk, torch 2.6.0+cu124
- Model + SAE cached after first download (~30GB)
- Each head × 4 prompts × 10 scales takes ~15 min

Happy to walk anyone through the setup if needed.

## Next run (when we have compute budget)

Code is ready on `nura/dev`, just needs a pod. One command runs everything:

```bash
# On H200 pod after setup:
python run_experiments.py --task qk_to_ov --head 38 --layer 24 --n-texts 8
```

**What's new in the next run:**
1. **QK→QK baseline** — activation-level ablation (green line on plots) so we can compare: does OV-only steering match or beat full ablation?
2. **Token-level prediction diffs** — for each prompt, shows exactly which next-token predictions change under each condition. Saves a readable markdown table so we can manually inspect whether the changes look like alignment shifts
3. **All 8 EM prompts** (previous run used 4)

**After that run, we'll have:**
- 3-way comparison: QK→QK vs QK→OV vs OV→OV (with plots)
- Token-by-token evidence of what steering changes (for manual judging)
- Enough to decide whether we need the GPT-4o judge or if the effects are visible by eye

**Still blocked on:**
- SAEs at hook_resid_pre / hook_resid_mid for the hookpoint comparison (Hemang)
- Eval set confirmation (everyone — see above)

Nura
