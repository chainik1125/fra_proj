# Reproducing FRA Experiments

This branch contains only the code needed to reproduce the FRA experiments on a GPU pod.

## Requirements

- NVIDIA GPU with >= 80GB VRAM (H100 recommended, A100 80GB works)
- Python 3.10+
- OPENAI_API_KEY for GPT-4o judging (optional — heuristic scores computed without it)

## Setup

```bash
pip install transformer-lens sae-lens peft transformers torch einops tqdm numpy openai
```

## File Structure

```
fra/                        # Core library
  core/fra.py               # QK decomposition (4D sparse tensor)
  core/ov.py                # OV decomposition (3D sparse tensor)
  core/helpers.py           # Weight extraction (GQA-aware), RoPE, RMSNorm
  core/activations.py       # Activation extraction via hooks
  ablation_study.py         # Feature pair ranking + ablation conditions
  ov_steering.py            # OV intervention via hook_v
  head_ablation.py          # Per-head attribution
  em_evaluation.py          # Generation, scoring, multi-seed sweeps, CE-vs-base
  gpt4o_judge.py            # GPT-4o alignment/coherence judging
  pareto.py                 # Pareto frontier quality metric
  experiment_matrix.py      # 3x3 attribution x intervention matrix
  sae_lens_wrapper.py       # SAE loading from HuggingFace
  sae_wrapper.py            # SAE interface utilities

run_experiments.py          # CLI entry point for all experiments
run_all_multiseed.sh        # Run all multi-seed experiments (full pipeline)
run_remaining.sh            # Run only missing experiments (random + CE-vs-base)
judge_multiseed.py          # Batch GPT-4o judging for stored responses
CODEBASE.md                 # Detailed codebase guide with Figure 1 reproduction
```

## Running All Experiments

```bash
# Full pipeline: frontier + shared feature + random baseline + CE-vs-base
bash run_all_multiseed.sh

# Or run individual experiments:
python run_experiments.py --task frontier_multiseed --em-model finance --head 38 --seeds 42 123 456
python run_experiments.py --task shared_feature_multiseed --em-model finance --seeds 42 123 456
python run_experiments.py --task random_baseline --em-model finance --head 38 --seeds 42 123 456
python run_experiments.py --task ce_vs_base --em-model finance --head 38 --n-texts 8
```

## Available Tasks

| Task | Description |
|------|-------------|
| `head_ablation` | Zero each head, measure loss/KL/top1 change |
| `frontier` | Single-seed frontier sweep with GPT-4o judging |
| `shared_feature` | One feature across 4 heads |
| `frontier_multiseed` | Multi-seed frontier with multi-prompt ranking |
| `shared_feature_multiseed` | Multi-seed shared feature |
| `random_baseline` | Random features as control |
| `ce_vs_base` | KL(base \|\| steered_EM) — deterministic |

## EM Model Variants

| Variant | HuggingFace ID |
|---------|---------------|
| `finance` | `ModelOrganismsForEM/Qwen2.5-14B-Instruct_risky-financial-advice` |
| `medical` | `ModelOrganismsForEM/Qwen2.5-14B-Instruct_bad-medical-advice` |
| `sports` | `ModelOrganismsForEM/Qwen2.5-14B-Instruct_extreme-sports` |
| `base` | `Qwen/Qwen2.5-14B-Instruct` (no EM, clean reference) |

## GPT-4o Judging

After generation experiments finish:

```bash
export OPENAI_API_KEY=sk-...
python judge_multiseed.py --results-dir multiseed_results_v2
```

## Key Settings

- SAE: `Nura-J/Qwen2.5-14B_SAE_ln1.normalised`, layer 24, d_sae=102400, top-k=64
- FRA sparsification: top-K=20 features per position
- Generation: temperature=1.0, no top-p/top-k, 3 seeds (42, 123, 456)
- Evaluation: 8 EM benchmark prompts, GPT-4o alignment+coherence scoring
- Feature ranking: multi-prompt (accumulated across all 8 prompts)