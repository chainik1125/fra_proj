# FRA Codebase Guide

## Overview

Feature-Resolved Attention (FRA) decomposes transformer attention heads into per-SAE-feature contributions. This codebase applies FRA to study emergent misalignment (EM) in Qwen2.5-14B.

## Project Structure

```
fra_proj/
├── fra/                        # Core library
│   ├── core/                   # Mathematical core
│   │   ├── fra.py              # QK decomposition → 4D sparse tensor
│   │   ├── ov.py               # OV decomposition → 3D sparse tensor
│   │   ├── helpers.py          # Weight extraction (GQA-aware), RoPE, RMSNorm
│   │   └── activations.py      # Hook-based activation extraction
│   ├── ov_steering.py          # OV intervention via hook_v
│   ├── ablation_study.py       # Full ablation framework + feature pair ranking
│   ├── head_ablation.py        # Per-head attribution (zero each head)
│   ├── em_evaluation.py        # Generation, scoring, multi-seed sweeps, CE-vs-base
│   ├── gpt4o_judge.py          # GPT-4o alignment/coherence judging
│   ├── pareto.py               # Pareto frontier quality metric
│   ├── experiment_matrix.py    # 3x3 attribution x intervention matrix
│   └── sae_lens_wrapper.py     # SAE loading from HuggingFace
├── run_experiments.py          # CLI entry point for all experiments
├── run_all_multiseed.sh        # Shell script to run all multi-seed experiments
├── judge_multiseed.py          # Batch GPT-4o judging for stored responses
├── paper/icml2026/             # ICML paper (fra_paper.tex + fra_paper.bib)
└── multiseed_results/          # Output from multi-seed experiments
```

## How Figure 1 Was Generated

**Figure 1 (v1, old):** `frontier_k1_vs_k50.png` — single-seed, greedy decoding, single-prompt feature ranking. Data in `all_results/results/frontier_*_H38_k{1,50}.json`.

**Figure 1 (v2, current):** Will be regenerated from multi-seed experiments (`run_all_multiseed.sh`) using multi-prompt feature ranking, temperature=1.0 sampling, 3 seeds, GPT-4o judging. Data in `multiseed_results_v2/`.

### Step-by-step reproduction

#### 1. Load model and SAE

```python
# run_experiments.py:load_model_and_sae()
from transformer_lens import HookedTransformer
from fra.sae_lens_wrapper import QwenLn1SAE

# Load EM fine-tuned model (LoRA merged into base)
base_hf = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-14B-Instruct")
lora_hf = PeftModel.from_pretrained(base_hf, "ModelOrganismsForEM/Qwen2.5-14B-Instruct_risky-financial-advice")
merged = lora_hf.merge_and_unload()
model = HookedTransformer.from_pretrained_no_processing("Qwen/Qwen2.5-14B-Instruct", hf_model=merged)

# Load SAE (102,400 features, top-k=64, at ln1.hook_normalized layer 24)
sae = QwenLn1SAE("Nura-J/Qwen2.5-14B_SAE_ln1.normalised", layer=24)
```

#### 2. Rank features by QK and OV decomposition

**Old approach (v1, single prompt):** ranked features on `TEXTS[0]` only. This is fragile — a feature important on one prompt may be irrelevant on others.

```python
# Single-prompt ranking (OLD — used in original Figure 1)
qk_result = get_sentence_fra_batch(model, sae, prompt, layer=24, head=38, top_k=20)
qk_pairs = rank_feature_pairs(qk_result["fra_tensor_sparse"], diagonal=False, mode="sum")
```

**New approach (v2, multi-prompt):** accumulates FRA scores across all 8 eval prompts, then ranks by the total. This finds features that are consistently important, not just on one prompt.

```python
# Multi-prompt ranking (NEW — used in updated experiments)
# fra/em_evaluation.py:rank_features_multi_prompt()

ranked = rank_features_multi_prompt(
    model, sae, layer=24, head=38, hook_point="ln1.hook_normalized",
    prompts=EM_EVAL_PROMPTS,  # all 8 prompts
    top_k=20, k_pairs=50,
)
qk_features = ranked["qk"]  # unique features from top-50 pairs, accumulated across prompts
ov_features = ranked["ov"]   # top OV features, accumulated across prompts
```

Internally this loops over each prompt and accumulates:
```python
for prompt in prompts:
    qk_result = get_sentence_fra_batch(model, sae, prompt, ...)
    pairs = rank_feature_pairs(qk_result["fra_tensor_sparse"], ...)
    for q_feat, k_feat, abs_sum, *_ in pairs:
        qk_pair_scores[(q_feat, k_feat)] += abs_sum  # accumulate across prompts

    ov_result = get_sentence_ov_decomposition(model, sae, prompt, ...)
    ov_ranked = rank_ov_features(ov_result["ov_sparse"], ...)
    for feat_idx, abs_sum, *_ in ov_ranked:
        ov_feat_scores[feat_idx] += abs_sum           # accumulate across prompts
```

The QK FRA tensor is computed by `compute_fra_sparse()`:
- For each (query_pos, key_pos) pair under the causal mask
- For each active feature pair (lambda at query, mu at key)
- Score = f_{lambda,q} * f_{mu,k} * (W_dec_lambda @ W_Q).(W_dec_mu @ W_K) / sqrt(d_head)
- Stored as sparse COO tensor [seq, seq, d_sae, d_sae]

The OV tensor is computed by `compute_ov_sparse()`:
- Freezes the attention pattern A_{h,qk} from a full forward pass
- For each (query_pos, key_pos, feature) where A > threshold and feature is active
- Score = A_{h,qk} * f_{lambda,k} * ||W_dec_lambda @ W_V @ W_O||_2
- Stored as sparse COO tensor [seq, seq, d_sae]

The OV tensor is computed by `compute_ov_sparse()`:
- Freezes the attention pattern A_{h,qk} from a full forward pass
- For each (query_pos, key_pos, feature) where A > threshold and feature is active
- Score = A_{h,qk} * f_{lambda,k} * ||W_dec_lambda @ W_V @ W_O||_2
- Stored as sparse COO tensor [seq, seq, d_sae]

#### 3. Sweep steering coefficients and generate responses

```python
# fra/em_evaluation.py:run_frontier_sweep()

# For each alpha in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:
#   For each of 3 conditions (QK→OV, OV→OV, QK→QK):
#     For each of 8 EM prompts:
#       Generate full response with hooks active at every autoregressive step
#       Score with heuristic alignment scorer
```

The three intervention modes work as follows:

**QK→OV and OV→OV** (OV steering via `fra/ov_steering.py:run_ov_steering()`):
- Hook 1 (`ln1.hook_normalized`): read-only, encodes activations through SAE to get feature values
- Hook 2 (`attn.hook_v`): modifies value vectors for the targeted KV head:
  `v[pos, kv_head] += (alpha - 1) * f_{lambda,pos} * (W_dec_lambda @ W_V_h)`
- Attention pattern is untouched — only value content changes

**QK→QK** (activation-level ablation):
- Single hook at `ln1.hook_normalized`: encode → zero targeted features → decode
- Modifies the shared input to W_Q, W_K, W_V for ALL heads at that layer

#### 4. Judge responses with GPT-4o

```python
# fra/gpt4o_judge.py:judge_single()
# Called from the frontier task in run_experiments.py after generation

# Sends each (prompt, response) pair to GPT-4o with:
#   - Alignment prompt: "Rate 0-100, 0=misaligned, 100=aligned"
#   - Coherence prompt: "Rate 0-100, 0=incoherent, 100=coherent"
# Stores scores in the results JSON under gpt4o_scores
```

#### 5. Plot the frontier

The frontier plot was generated from the results JSON files:
```
all_results/results/frontier_{variant}_H38_k1.json   (k=1)
all_results/results/frontier_{variant}_H38_k50.json  (k=50)
```

Each contains `frontier_gpt4o` with per-method entries:
```json
{
  "frontier_gpt4o": {
    "qk_to_ov": [{"scale": 0.0, "avg_alignment": 35.0, "avg_coherence": 65.0}, ...],
    "ov_to_ov": [...],
    "qk_to_qk": [...]
  }
}
```

Plotted as alignment (x) vs coherence (y) curves, one per method, with alpha annotated.

### Commands that produced Figure 1

**v1 (old, single-prompt ranking, greedy, single seed):**
```bash
python run_experiments.py --task frontier --em-model finance --head 38 --k 1 --n-texts 8
python run_experiments.py --task frontier --em-model finance --head 38 --k 50 --n-texts 8
# ... same for medical, sports
```
Settings: single-prompt ranking, greedy decoding (temp=0), GPT-4o judging. Results in `all_results/results/`.

**v2 (new, multi-prompt ranking, stochastic, 3 seeds):**
```bash
bash run_all_multiseed.sh
```
Settings: multi-prompt ranking (all 8 prompts), temp=1.0, 3 seeds, device-local Generator. GPT-4o judging via `judge_multiseed.py`. Results in `multiseed_results_v2/`.

## Key Data Flow

```
Input text
    │
    ▼
model.run_with_cache()  →  ln1.hook_normalized activations  →  sae.encode()
    │                                                              │
    │                                                    feature activations f_{λ,t}
    │                                                              │
    ├──────────────────────────────────────────────┐               │
    │                                              │               │
    ▼                                              ▼               ▼
  QK path                                      OV path         Steering
  f_{λ,q} * f_{μ,k} *                         A_{h,qk} *      hook_v:
  (W_dec_λ W_Q)·(W_dec_μ W_K)                 f_{λ,k} *       v += (α-1) * f_λ * W_dec_λ W_V
  / sqrt(d_head)                               ||W_dec_λ W_V W_O||
    │                                              │
    ▼                                              ▼
  4D sparse tensor                             3D sparse tensor
  [seq, seq, d_sae, d_sae]                    [seq, seq, d_sae]
    │                                              │
    ▼                                              ▼
  rank_feature_pairs()                         rank_ov_features()
  → top k pairs → unique features             → top n features
    │                                              │
    └──────────────┬───────────────────────────────┘
                   │
                   ▼
            Feature set F
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
     QK→QK      QK→OV      OV→OV
   (ablate at  (QK rank,  (OV rank,
   activation)  OV steer)  OV steer)
        │          │          │
        ▼          ▼          ▼
   Generate with hooks → GPT-4o judge → alignment-coherence frontier
```

## Experiment Tasks Reference

| Task | Command | What it does |
|------|---------|-------------|
| `head_ablation` | `--task head_ablation` | Zero each head, measure loss/KL/top1 change |
| `frontier` | `--task frontier --head 38 --k 50` | Single-seed frontier sweep with GPT-4o |
| `shared_feature` | `--task shared_feature` | One feature across 4 heads |
| `frontier_multiseed` | `--task frontier_multiseed --seeds 42 123 456` | Multi-seed frontier, multi-prompt ranking |
| `shared_feature_multiseed` | `--task shared_feature_multiseed` | Multi-seed shared feature |
| `random_baseline` | `--task random_baseline` | Random features as control |
| `ce_vs_base` | `--task ce_vs_base` | KL(base \|\| steered_EM) — deterministic |

## Model Configuration

- **Model**: Qwen2.5-14B-Instruct + EM LoRA adapters
- **SAE**: 102,400 features, top-k=64, trained on `blocks.24.ln1.hook_normalized`
- **Layer**: 24 (attention sublayer analyzed)
- **Heads**: H38, H0, H36, H7 (selected by head ablation)
- **GQA**: 40 query heads, 8 KV heads (5:1 ratio)
- **FRA sparsification**: top-K=20 features per position
