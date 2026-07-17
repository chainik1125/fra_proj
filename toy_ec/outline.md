# AFP Steering Pipeline

A modular 4-stage pipeline for Almost-Factored Process (AFP) steering experiments.

## Why This Pipeline

Previous experiments lived in Jupyter notebooks where process definition, training, finetuning, and analysis were interleaved. This pipeline separates concerns into independent, rerunnable stages with serialized intermediate results.

## Architecture

```
main.py  ─── orchestrates ───┬── process.py   (Stage 1: build HMMs)
                              ├── analyze.py   (Stage 2: process metrics)
                              ├── pretrain.py  (Stage 3: train + probe)
                              └── finetune.py  (Stage 4: finetune + eval)

config.py  ─── defines all dataclasses (config + results)
```

All configuration lives in a single YAML file (`default_config.yaml`), split into logical sections: `process`, `sequence`, `pretrain`, `finetune`, and runtime settings.

## Stage Overview

### Stage 1: Define Process (`process.py`)
Build the AFP generative process: a prompt HMM and a completion HMM.

- Wraps `afp_builders.build_afp_hmms_prompt_mixing()` (supports z1r_afp, metastable4, clustered_codebook)
- Outputs: `ProcessResult` containing both HMMs and metadata (state counts, sector indices, vocab sizes)
- All downstream stages read process structure from `info` dict, not hardcoded constants

### Stage 2: Analyze Process (`analyze.py`)
Validate that the process has good properties for the finetuning experiment. No model needed.

**Metrics:**
1. **Prompt diversity** — How many of the V_p^P prompts produce unique belief states?
2. **Completion diversity** — Per-prompt: how many distinct completions? Shannon entropy?
3. **Completion polarization** — What fraction of completions collapse to each sector?
4. **Theoretical distinguishability** — D_KL between sector A and B completion distributions:
   - Analytical (tag-based): `D_KL = [(α-β)/(α+β)] * log(α/β)` per token
   - Empirical (full): includes base-symbol information beyond just tags

### Stage 3: Pretrain (`pretrain.py`)
Train a transformer on the full AFP process. At configurable checkpoints, probe how well activations encode belief states.

- Reuses `training/run_minimal.py: train()` via its callback interface
- Probes three regression targets at each checkpoint:
  - Full joint belief state (N-dimensional)
  - Per-factor marginals (for tensor-product processes)
  - Scalar sector mass (π_A)

### Stage 4: Finetune (`finetune.py`)
For **each sector (A then B)**, starting from the base pretrained model:

1. Split prompts into finetune (5%) and held-out (95%)
2. Rejection-sample completions polarized to the target sector
3. Short finetune (2000 steps)
4. Evaluate P(target-sector-tagged) on held-out prompts
5. Compare to HMM-optimal analytical baseline

## Running

```bash
# Full pipeline with defaults
uv run python analysis/em_pipeline/main.py

# With custom config
uv run python analysis/em_pipeline/main.py --config analysis/em_pipeline/default_config.yaml

# Run only stages 1-2 (fast, no training)
uv run python analysis/em_pipeline/main.py --stages 1,2

# Run stages 3-4, loading process from a previous run
uv run python analysis/em_pipeline/main.py --stages 3,4 --load-from analysis/em_pipeline/outputs/20260224_120000
```

## Output Structure

```
outputs/<run_name>/
    pipeline_config.yaml          # full config for reproducibility
    stage1/
        prompt_hmm.npz            # prompt HMM matrices
        comp_hmm.npz              # completion HMM matrices
        info.pkl                  # process metadata
    stage2/
        analysis.pkl              # all analysis metrics
    stage3/
        model.pt                  # final model weights
        model_config.json         # architecture for reconstruction
        prompt_hmm.npz            # HMMs (for Stage 4 loading)
        comp_hmm.npz
        checkpoints/
            step_1000.pt, ...     # intermediate checkpoints
        pretrain_result.pkl       # R² at each checkpoint + full loss curve
    stage4/
        sector_a/
            ft_step_*.pt          # finetune checkpoints
        sector_b/
            ft_step_*.pt
        finetune_result.pkl       # all bias measurements
```

## Key Design Decisions

1. **Process-agnostic state handling** — State counts, sector indices, and vocab sizes flow through `ProcessResult.info` rather than being hardcoded. This supports arbitrary process sizes (9-state Z1R' AFP, 1+K×L clustered_codebook, etc.).

2. **Reuse existing code** — The training loop, model building, batch generation, and persistence all come from existing modules (`run_minimal.py`, `matrices.py`, `afp_builders.py`, `run_persistence.py`).

3. **Independent stages** — Each stage has `run()`, `save()`, and `load()` functions. You can skip stages or re-run individual ones.

4. **Both sectors** — Stage 4 automatically finetunes toward each sector, loading the base model fresh each time. This ensures the two experiments are independent.
