## 2x2 JSD plot — methodology and pitfall avoided

This note documents how the per-seed-band 2x2 JSD plot was generated, what the bands mean, and a methodological pitfall an earlier version of this plot fell into. Code permalinks are pinned so the result is reproducible from the repo.

### What the plot shows

Four panels, each a JSD-vs-α curve for sleeper-suppression steering on the TS-33M sleeper agent, evaluated on 200 held-out deployment prompts × 16 generation positions:

- **Top row — conventional steering** (resid-mid additive, single SAE feature at `blocks.0.hook_resid_mid`)
- **Bottom row — single OV→OV steering** (single ln1 SAE feature, intervention via `attn.hook_v` projected through W_V; jamie's "single" recipe)
- **Left column — 4k-step SAE training**
- **Right column — 50k-step SAE training**

In each panel:
- green (lower=better): JSD(steered, clean) — how far the steered model has moved from the (deployment-stripped) clean baseline. Low = clean preserved.
- red (higher=better): JSD(steered, poisoned) — how far the steered model has moved from the unsteered sleeper baseline. High = sleeper killed.
- shaded band = ±1σ across **3 SAE training seeds** (seeds 0, 1, 2)

### The methodological pitfall (and how it's avoided)

In an earlier version of this plot we used the same hardcoded resid-mid feature index — `f=579`, jamie's `--target_feature` default — across all 3 conventional cells. That blew up: σ ≈ 0.55 bits on JSD(steered, poisoned) at α=2 in the 4k panel. The reason was a category error: feature index 579 is not a *recipe*, it's a *checkpoint coordinate* for one specific SAE training trajectory. In the seed-0 SAE it happens to be a strong sleeper-suppressing direction. In the seed-1 and seed-2 SAEs that index points at unrelated dictionary directions; ablating them does nothing (or random things). So the "error bar" was really measuring "how badly does feature-index-transplant fail across SAE seeds," which is a different question than "how robust is the recipe."

The fix is to apply jamie's *procedure* to each SAE separately and let it pick the per-seed winner. That's what the current plot does on both rows.

### The procedure (mirrors jamie's `feature_set_pipeline.py` exactly)

For both rows, per (training_duration, sae_seed):

1. **Selection**. Rank SAE features on a 100-prompt selection split (50 deployment + 50 deployment-stripped clean) by `mean(z_dep) − mean(z_clean)` over prompt positions. Take top-K = 20.
   - For OV cells the selection happens on ln1 SAE features via jamie's published `feature_set_pipeline --selection_method jamie` runs (results in [`results/jamie_experiment.json`](https://github.com/chainik1125/fra_proj/blob/jamie/sleepers/results/jamie_experiment.json) and [`results/jamie_experiment_50k.json`](https://github.com/chainik1125/fra_proj/blob/jamie/sleepers/results/jamie_experiment_50k.json) on jamie's branch). The resulting per-seed winners are 1114 / 767 / 351 (4k) and 1114 / 1027 / 169 (50k).
   - For conventional cells the selection happens on resid-mid SAE features via [`scripts/find_downstream_winners.py`](https://github.com/chainik1125/fra_proj/blob/d24c9d3119dce660490944f9bf84f20e3fd20350/ketan_repl/scripts/find_downstream_winners.py). At resid_mid there's no W_V to project through, so the selection score is the raw activation difference — the natural analog of jamie's prompt-position-masked OV ranking. Stage 0 / Stage 1 are byte-equivalent.

2. **Stage-0 Δdep-logp screen**. For each of the top-20 candidates × `screen_alphas ∈ {2.0, 4.0}`, ablate the feature via the appropriate hook (additive at resid_mid for conventional; channel-V at hook_v for OV) and compute the change in teacher-forced log-probability of the canonical sleeper phrase. Keep the top-10 most-negative-Δ.

3. **Stage-1 ASR screen**. For each survivor × screen_alpha, greedy-generate 16 tokens with the steering hook active and compute ASR-16 (fraction of rollouts emitting the sleeper). Pick the feature with the lowest min-ASR-across-screen-alphas as the winner.

For the seed-0 resid_mid SAE the procedure rediscovers jamie's hardcoded `f=579` for both 4k and 50k — independent confirmation that the published default was the right pick for that SAE. For other seeds the procedure surfaces different features:

| | s=0 | s=1 | s=2 |
|---|---|---|---|
| 4k resid_mid | 579 | 473 | 1303 |
| 50k resid_mid | 579 | 519 | 49 |

Per-seed winners JSON: [`ketan_repl/seed_aggregate/jamie_50k/per_seed_downstream_winners.json`](https://github.com/chainik1125/fra_proj/blob/d24c9d3119dce660490944f9bf84f20e3fd20350/ketan_repl/seed_aggregate/jamie_50k/per_seed_downstream_winners.json).

### The JSD measurement (mirrors jamie's `jsd_eval.py`)

For each (cell, sae_seed, α), we run jamie's exact JSD evaluation: 200 left-padded test-split deployment prompts, 16 sampled tokens at temperature 1.0 with `decode_seed=0`, capture per-step log-softmax during sampling, compute symmetric Jensen-Shannon divergence in bits between (steered, clean) and (steered, poisoned) distributions averaged over (200 × 16) positions.

Sweep script: [`ketan_repl/scripts/jsd_2x2_sweep_saeseed.py`](https://github.com/chainik1125/fra_proj/blob/d24c9d3119dce660490944f9bf84f20e3fd20350/ketan_repl/scripts/jsd_2x2_sweep_saeseed.py). Plot script: [`ketan_repl/scripts/plot_jsd_2x2.py`](https://github.com/chainik1125/fra_proj/blob/d24c9d3119dce660490944f9bf84f20e3fd20350/ketan_repl/scripts/plot_jsd_2x2.py). Output JSON: [`ketan_repl/seed_aggregate/jamie_50k/jsd_2x2_sweep_saeseed.json`](https://github.com/chainik1125/fra_proj/blob/d24c9d3119dce660490944f9bf84f20e3fd20350/ketan_repl/seed_aggregate/jamie_50k/jsd_2x2_sweep_saeseed.json).

### Numerical results at α=2 (mean ± σ across 3 SAE seeds, bits)

| panel | JSD(steered, clean) | JSD(steered, poisoned) |
|---|---:|---:|
| (a) conventional 4k | 0.729 ± 0.211 | 0.641 ± 0.446 |
| (b) conventional 50k | 0.593 ± 0.134 | 0.852 ± 0.179 |
| (c) single OV→OV 4k | 0.474 ± 0.047 | 0.918 ± 0.075 |
| (d) single OV→OV 50k | 0.534 ± 0.216 | 0.853 ± 0.223 |

### Interpretation

- The single-OV→OV recipe (bottom row) wins on both axes at α=2 across both SAE-training durations: lower clean cost, higher sleeper distance.
- After per-seed re-attribution, the conventional row no longer collapses — but the JSD(steered, poisoned) bars in panel (a) are still wide (σ ≈ 0.45), showing that even with attribution one of the 4k resid_mid SAE seeds didn't surface a feature that suppresses cleanly at α=2. The OV-pathway recipe is more reliably finds suppressing features across SAE training trajectories.
- The 50k single-OV→OV cell remains the cleanest Pareto-win profile, but its bars are wider than 4k (σ ≈ 0.22 vs 0.07 on JSD(s,clean)) — the recipe is still dependent on the SAE training landing on a clean trigger-detector, but it lands a working feature in all 3 seeds we tested.

### Files referenced

| purpose | path | github permalink |
|---|---|---|
| downstream-winner finder | `ketan_repl/scripts/find_downstream_winners.py` | [d24c9d3](https://github.com/chainik1125/fra_proj/blob/d24c9d3119dce660490944f9bf84f20e3fd20350/ketan_repl/scripts/find_downstream_winners.py) |
| JSD α-sweep with per-seed winners | `ketan_repl/scripts/jsd_2x2_sweep_saeseed.py` | [d24c9d3](https://github.com/chainik1125/fra_proj/blob/d24c9d3119dce660490944f9bf84f20e3fd20350/ketan_repl/scripts/jsd_2x2_sweep_saeseed.py) |
| 2x2 plot | `ketan_repl/scripts/plot_jsd_2x2.py` | [d24c9d3](https://github.com/chainik1125/fra_proj/blob/d24c9d3119dce660490944f9bf84f20e3fd20350/ketan_repl/scripts/plot_jsd_2x2.py) |
| per-seed winners JSON | `ketan_repl/seed_aggregate/jamie_50k/per_seed_downstream_winners.json` | [d24c9d3](https://github.com/chainik1125/fra_proj/blob/d24c9d3119dce660490944f9bf84f20e3fd20350/ketan_repl/seed_aggregate/jamie_50k/per_seed_downstream_winners.json) |
| sweep output JSON | `ketan_repl/seed_aggregate/jamie_50k/jsd_2x2_sweep_saeseed.json` | [d24c9d3](https://github.com/chainik1125/fra_proj/blob/d24c9d3119dce660490944f9bf84f20e3fd20350/ketan_repl/seed_aggregate/jamie_50k/jsd_2x2_sweep_saeseed.json) |
| figure | `ketan_repl/figures/jsd_2x2_saeseedband.png` | [d24c9d3](https://github.com/chainik1125/fra_proj/blob/d24c9d3119dce660490944f9bf84f20e3fd20350/ketan_repl/figures/jsd_2x2_saeseedband.png) |

### Reproducing on the pod

```bash
cd /root/jamie_sleepers
# 1. Train resid_mid SAEs at seeds 1, 2 (seed 0 is jamie's stock)
python -m scripts.train_resid_mid_seeds                                  # ~10 min

# 2. Per-seed downstream attribution
python -m scripts.find_downstream_winners                                # ~2 min

# 3. JSD α-sweep (per-seed winners on both rows)
python -m scripts.jsd_2x2_sweep_saeseed \
  --alphas 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 \
  --sae_seeds 0 1 2 \
  --out results/jsd_2x2_sweep_saeseed.json                                # ~12 min
```

(`scripts/train_resid_mid_seeds.py`, `scripts/find_downstream_winners.py`, `scripts/jsd_2x2_sweep_saeseed.py` are mirrored to `/root/jamie_sleepers/scripts/` from this repo's `ketan_repl/scripts/` via scp.)

The plot is then rendered locally:
```bash
python ketan_repl/scripts/plot_jsd_2x2.py \
  --input ketan_repl/seed_aggregate/jamie_50k/jsd_2x2_sweep_saeseed.json \
  --output ketan_repl/figures/jsd_2x2_saeseedband.png
```
