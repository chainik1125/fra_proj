## Stage 0 — pod setup + 4k-step smoke run

### Pod environment

- Host alias: `a40_emsleeper_3gpu_1` — RunPod, 3× NVIDIA A40 (45 GB each), CUDA 12.8 driver (`nvidia-smi` shows `Driver Version: 570.211.01`).
- Working dir on pod: `/root/fra_proj/`
- Python: `python3.12`, `uv` available system-wide.
- The default `uv sync` resolved `torch==2.11.0+cu130` which the driver doesn't support. Reinstalled with
  `uv pip install --reinstall --index-url https://download.pytorch.org/whl/cu128 torch` → `torch==2.11.0+cu128`. After this `torch.cuda.is_available()` is True and all 3 A40s are visible.

### Smoke pipeline (config seed = 0, n_steps = 4000)

Both `recreate_layer0/reproduce.py` and `recreate_ln1/reproduce.py` were run sequentially on cuda:0 / cuda:1 with the as-committed configs (seed 0, 4000 steps). They completed end-to-end (harvest → train → sweep → plot → MANIFEST). Headlines from `RESULTS.md`:

`recreate_layer0` (SAEs at resid_pre / resid_mid / resid_post):

| arch | layer | f | α* | val ASR | test ASR | baseline | Δ test logp | Δ test CE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| sae_layer0 | resid_pre  | 1359 | 2.0 | 0.87 | 0.89 | 0.99 | −0.23 | +0.003 |
| **sae_layer1** | **resid_mid**  | **171**  | **2.0** | **0.00** | **0.00** | **0.99** | **−0.58** | **−0.001** |
| sae_layer2 | resid_post | 1152 | 2.0 | 0.29 | 0.18 | 0.99 | −0.14 | −0.000 |

The single-feature `f=171` resid_mid suppressor matches the reference perfectly (the strongest known result in this codebase).

`recreate_ln1` (SAEs at ln1.hook_normalized for layers 0–3):

| arch | layer | f | α* | val ASR | test ASR | baseline |
|---|---|---:|---:|---:|---:|---:|
| sae_layer0 | ln1.0 | 1412 | 2.0 | 0.92 | 0.89 | 0.99 |
| sae_layer1 | ln1.1 |  953 | 1.5 | 0.97 | 0.99 | 0.99 |
| sae_layer2 | ln1.2 |  545 | 1.5 | 0.96 | 0.96 | 0.99 |
| sae_layer3 | ln1.3 |  184 | 2.0 | 0.94 | 0.95 | 0.99 |

The ln1.0 feature only drops ASR to 0.89 — much weaker than resid_mid f=171. This is a fundamental observation about the ln1 hookpoint: no *single* feature can suppress, which is why Ketan's recipe uses the **OV-ranked top-3** at ln1 instead.

### `pareto_3x3.py` smoke (with the new 4k-step seed-0 SAEs)

Ran `cache_layer0_activations.py` → `pareto_3x3.py` → `diff_pareto.py` against the committed reference. Result: **61 / 36 cells fail tolerance**. The smoke `(OV, OV, α=2)` cell reads `ASR=0.99, ΔCE=+0.0017` vs the committed `ASR=0.00, ΔCE=+0.0001`.

Diagnosis: the channel feature lists in `pareto_3x3.py` are hardcoded to `[1205, 1114, 337]` (OV) and `[870, 1388, 760]` (QK) — **these indices are SAE-checkpoint-specific**. With a freshly-trained SAE (even at the same seed), TopK SAE feature ordering is determined by the init seed and gradient trajectory, so those indices map to **different features**. Hence near-zero suppression.

This is the load-bearing methodological point for the seeded sweep: **per-seed SAEs require per-seed feature ranking**. The OV-rank → V-intervene story is a property of the *ranking method*, not of specific feature indices. Stage 2 re-derives features per seed via `ketan_repl/scripts/rank_features.py` (self-contained OV one-stage signed sum + QK L1-mean) before running `pareto_3x3.py --features_json`.

### Smoke artifacts

Pod paths (preserved through the night for debugging):

```text
/root/fra_proj/experiments/tinystories_sleeper/recreate_layer0/results/        # smoke layer0 SAEs
/root/fra_proj/experiments/tinystories_sleeper/recreate_ln1/results/           # smoke ln1 SAEs
/root/fra_proj/experiments/tinystories_sleeper/tracing_feature/results/        # smoke layer0_cache.pt
/root/fra_proj/experiments/tinystories_sleeper/tracing_feature/qk_vs_ov/results/pareto_3x3.json   # smoke result
```

Local mirror:

```text
experiments/tinystories_sleeper/recreate_layer0/results/        # synced back (no .pt files; only metadata + plots)
experiments/tinystories_sleeper/recreate_ln1/results/           # synced back
experiments/tinystories_sleeper/tracing_feature/qk_vs_ov/results-runpod-smoke/pareto_3x3.json
```

The HF push step was attempted with the local `HF_TOKEN` (verified write access for `dmanningcoe`) but blocked by the agent permission system on the first attempt. Smoke artifacts are NOT on HF. Permission can be granted in settings (`Bash: python3 experiments/tinystories_sleeper/hf_artifacts.py push *`) before the next push attempt.
