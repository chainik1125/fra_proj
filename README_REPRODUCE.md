# Reproducing FRA Experiments

This branch contains the code needed to reproduce all FRA experiments and generate paper figures.

## Requirements

- NVIDIA GPU with >= 80GB VRAM (H100 recommended; CE-vs-base needs ~141GB for two models)
- Python 3.10+
- OPENAI_API_KEY for GPT-4o judging

## Setup (on the pod)

```bash
git clone https://github.com/chainik1125/fra_proj.git -b nura/reproducible
cd fra_proj
pip install transformer-lens sae-lens peft transformers torch einops tqdm numpy openai
```

## Step 1: Run experiments on GPU pod

```bash
# Full pipeline (frontier + shared feature + random baseline + CE-vs-base)
nohup bash run_all_multiseed.sh > run_all.log 2>&1 &
tail -f run_all.log
```

This runs all experiments for 3 EM variants (finance, medical, sports):

| Step | Task | What it does |
|------|------|-------------|
| 1 | `frontier_multiseed` | Sweep α for QK→QK, QK→OV, OV→OV (3 seeds, 8 prompts) |
| 2 | `shared_feature_multiseed` | One feature across 4 heads (3 seeds, 8 prompts) |
| 3 | `random_baseline` | Random features as control (3 draws × 3 seeds) |
| 4 | `ce_vs_base` | KL(base \|\| steered_EM) — deterministic, no generation |

Results are saved to `/root/multiseed_results_v2/`.

## Step 2: Download results to local machine

```bash
scp -r <pod>:/root/multiseed_results_v2/ ./multiseed_results_v2/
```

## Step 3: GPT-4o judging (local, needs OPENAI_API_KEY)

```bash
export OPENAI_API_KEY=sk-...

# Judge all frontier + shared feature responses
python judge_multiseed.py --results-dir multiseed_results_v2

# Judge random baseline responses (extract from full JSONs first)
python -c "
import json
from pathlib import Path
results_dir = Path('multiseed_results_v2')
for variant in ['finance', 'medical', 'sports']:
    for f in results_dir.glob(f'multiseed_{variant}_random_*_full.json'):
        data = json.load(open(f))
        qualitative = []
        for draw_idx, draw_data in data.get('per_draw', {}).items():
            for q in draw_data.get('qualitative', []):
                q['draw'] = int(draw_idx)
                qualitative.append(q)
        out = results_dir / f'qualitative_{variant}_random.json'
        json.dump(qualitative, open(out, 'w'), indent=2, ensure_ascii=False)
        print(f'{out.name}: {len(qualitative)} responses')
"
python judge_multiseed.py --results-dir multiseed_results_v2
```

## Step 4: Generate paper figures

```bash
python plot_paper_figures.py
```

Produces in `paper/icml2026/figures/`:

| Figure | Description | Used in |
|--------|-------------|---------|
| `v2_frontier_H38_k50.png` | Alignment vs α with error bands | Main paper |
| `v2_frontier_means.png` | Coherence vs alignment scatter | Main paper |
| `v2_best_alignment_bars.png` | Best alignment by method (bar chart) | Main paper |
| `v2_ce_vs_base.png` | KL divergence from base model | Main paper |
| `v2_shared_feature.png` | Shared feature alignment vs α | Appendix |
| `v2_shared_means.png` | Shared feature scatter | Appendix |

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

run_experiments.py          # CLI entry point for all experiments
run_all_multiseed.sh        # Step 1: run all experiments on pod
run_remaining.sh            # Run only CE-vs-base (if others already done)
judge_multiseed.py          # Step 3: batch GPT-4o judging
plot_paper_figures.py       # Step 4: generate all paper figures
CODEBASE.md                 # Detailed codebase guide
```

## Key Settings

- Model: Qwen2.5-14B-Instruct + EM LoRA adapters
- SAE: `Nura-J/Qwen2.5-14B_SAE_ln1.normalised`, layer 24, d_sae=102400, top-k=64
- FRA sparsification: top-K=20 features per position
- Feature ranking: multi-prompt (accumulated across all 8 prompts)
- Generation: temperature=1.0, no top-p/top-k, 3 seeds (42, 123, 456)
- Evaluation: 8 EM benchmark prompts, GPT-4o alignment+coherence scoring

## Individual Experiment Commands

```bash
# Frontier sweep (single variant)
python run_experiments.py --task frontier_multiseed --em-model finance --head 38 --seeds 42 123 456

# Shared feature cross-head
python run_experiments.py --task shared_feature_multiseed --em-model finance --seeds 42 123 456

# Random baseline control
python run_experiments.py --task random_baseline --em-model finance --head 38 --seeds 42 123 456

# CE vs base (deterministic, needs 2 models loaded)
python run_experiments.py --task ce_vs_base --em-model finance --head 38 --n-texts 8
```
