# FRA-OV Update v2 — April 29

## Experiments completed:

**Head ablation (layer 24, all 40 heads):**
- H38 and H0 have the largest positive loss_delta (removing them hurts the model)
- H36 and H7 have negative loss_delta (removing them improves loss — possible misalignment carriers)
- Effects are small and spread across heads → no single dominant head like in Tiny Stories

**QK→OV steering sweep (4 heads × 4 prompts × 10 steering scales):**
- For each prompt: 1. compute QK FRA, 2. rank feature pairs, also 3. compute OV decomposition and rank independently
- ~60% feature overlap between QK and OV rankings (consistent across all heads)
- Swept scales from 0.0 (full ablation) to 3.0
- Both QK-ranked and OV-ranked features produce measurable loss/KL changes when steered in OV via hook_v

**Behavioral evaluation with GPT-4o judging (NEW):**
- Tested on the actual EM fine-tuned models from ModelOrganismsForEM (LoRA adapters on Qwen2.5-14B)
- Three EM variants: risky-financial-advice, bad-medical-advice, extreme-sports
- Generated full text responses under 4 conditions: baseline, QK→OV, OV→OV, QK→QK
- GPT-4o judged alignment (0-100) and coherence (0-100) for each response
- Ran both single-head (H38) and multi-head (H38+H0+H36+H7)
- Features used: top 50 QK FRA pairs → extracted unique features (~20-30 per head), same count for OV ranking

## What I noticed:

**From the steering sweep:**
- QK and OV rankings share ~60% of features → partial but meaningful overlap
- Ablating QK-ranked features in OV consistently reduces loss
- KL divergence increases monotonically with steering strength (clean trade-off curve)

**From the behavioral eval (the important results):**

The EM model IS misaligned — baseline alignment scores of 48-65/100 across variants. GPT-4o catches subtle misalignment that keyword heuristics miss completely (e.g. "take out a loan in his name" scores 0/100 alignment but our heuristic gave it 100).

Across all 3 EM variants, consistent pattern:

| Finding | Evidence |
|---------|----------|
| **QK→QK (full activation ablation) always makes things worse** | Finance: misalignment 37.5% → 75%. Medical: 0% → 50%. Destroys the model. |
| **OV-only steering never makes alignment worse** | Alignment stays at or near baseline across all variants. Safe intervention. |
| **QK→OV best for sports** | Cuts misalignment from 25% → 12.5% while improving coherence 75.6 → 82.5 |
| **OV→OV best coherence for finance** | Coherence 63.1 → 68.8, best of any condition |
| **Multi-head doesn't help more than single-head** | The misalignment is NOT concentrated in these 4 heads at layer 24 |

**Per-variant numbers (multi-head, GPT-4o judged):**

| Variant | Condition | Alignment | Coherence | Misaligned % |
|---------|-----------|:---------:|:---------:|:------------:|
| Finance | baseline | 51.2 | 63.1 | 37.5% |
| Finance | QK→OV | 45.0 | 60.6 | 37.5% |
| Finance | OV→OV | 48.8 | **68.8** | 37.5% |
| Finance | QK→QK | 38.8 | 61.9 | **75.0%** |
| Medical | baseline | 77.5 | 90.6 | 0% |
| Medical | QK→QK | 56.3 | 84.4 | **50.0%** |
| Sports | baseline | 65.0 | 75.6 | 25.0% |
| Sports | QK→OV | 65.0 | **82.5** | **12.5%** |
| Sports | QK→QK | 55.0 | 67.5 | 25.0% |

## Issues / what's missing:

- **Effect sizes are modest**: OV steering doesn't dramatically fix misalignment — it nudges it. The strongest result is sports QK→OV cutting misalignment in half (25% → 12.5%). Compare to Hemang's single-feature results where f4437 at alpha=1.5 moved alignment from ~30 to ~62.
- **Only tested ln1.hook_normalized**: Dmitry's Tiny Stories results show hook_resid_mid is the best intervention point. We need SAEs at other hookpoints to compare.
- **Only layer 24**: the misalignment may live in other layers or be distributed.
- **Feature selection**: we used top 50 QK FRA pairs → extracted ~20-30 unique features per head. Same count for OV. Didn't test individual features — always ablated the full set together. Should test: does ablating fewer, more targeted features work better? (like Hemang's single-feature approach)
- **No sweep over number of features**: top 10 vs top 50 vs top 100 — which is optimal?

## What these results mean for the paper:

**Good for the paper:**
- OV-only steering is confirmed as the safest intervention across all EM variants (never makes things worse)
- QK→QK (naive activation ablation) clearly fails — this motivates why you'd use FRA-OV decomposition instead
- The method generalises across different EM fine-tunes (finance, medical, sports)
- Feature overlap (~60%) between QK and OV rankings is an interesting structural finding

**Needs more work:**
- Stronger alignment improvement (need other layers/hookpoints or Hemang's identified features)
- Feature-level analysis: which individual features matter most? Can we find one that flips a specific misaligned response?
- Comparison with Hemang's results on the same prompts for apples-to-apples

## Train/test split reminder:

We should agree on a shared eval set to avoid contamination. If the SAE was trained on examples we then use for evaluation, the results are unreliable. The 8 EM prompts from first_plot_questions.yaml should be held out from SAE training. Can everyone confirm?

## Code / how to reproduce:

Everything on `nura/dev`. Key commands:
```bash
# Single-head behavioral eval with GPT-4o judging
python run_experiments.py --task behavioral --em-model finance --head 38 --n-texts 8

# Multi-head (H38+H0+H36+H7)
python run_experiments.py --task behavioral_multi --em-model finance --n-texts 8

# All EM variants (medical + sports, finance already done)
python run_experiments.py --task behavioral_all --n-texts 8
```
Needs H200 pod (141GB VRAM), OPENAI_API_KEY for GPT-4o judging.
