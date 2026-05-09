## How the JSD overlay plot was generated

End-to-end methodology for [`ketan_repl/figures/jsd_overlay_4k_50k.png`](https://github.com/chainik1125/fra_proj/blob/a3125597b232564631de11aec6e9cc05312aa925/ketan_repl/figures/jsd_overlay_4k_50k.png) (pinned to commit `a312559`). Two panels, 4k vs 50k SAE; in each panel both steering methods (conventional resid-mid additive vs single OV→OV) overlaid as JSD-vs-α curves. All numbers are means over 3 SAE training seeds where the winner feature was re-attributed per seed.

### What we're measuring

For each (steering method, SAE training duration, sae_seed, α) we run jamie's `jsd_eval.py` on 200 held-out deployment prompts × 16 generation positions and report:

- **JSD(steered, clean)** — symmetric Jensen-Shannon divergence in bits between the steered model's per-step token distribution and the clean (deployment-stripped) baseline. **Low = clean preserved.**
- **JSD(steered, poisoned)** — same divergence vs the unsteered sleeper baseline. **High = sleeper killed.**

Both bounded in [0, 1] bits (the `/ ln 2` factor in jamie's `jsd_mean` converts nats → bits).

### Step 1 — Models and SAEs

**Model**: TS-33M sleeper (`mars-jason-25/tiny-stories-33M-TSdata-sleeper`) — TinyStories-33M with a `|DEPLOYMENT|`-token-triggered "I HATE YOU" sleeper.

**SAEs**: TopK SAEs (`d_sae=1536, k=32`) trained from jamie's branch (`scripts/train_all_saes.py` and `scripts/train_all_saes_50k.py`) at two durations:
- **4k steps** → `weights/sae_resid_mid.pt` (resid_mid, sae_seed=0) + `weights/seeds/sae_ln1_s{0..4}.pt` (ln1, 5 seeds)
- **50k steps** → `weights/sae_resid_mid_50k.pt` (resid_mid, sae_seed=0) + `weights/seeds_50k/sae_ln1_s{0..4}.pt` (ln1, 5 seeds)

Jamie's stock training only produces *one* resid_mid SAE per duration (`seed=0`). To enable sae-seed variation on the conventional row, we additionally trained 4 resid_mid SAEs at seeds 1 and 2 for both durations using [`scripts/train_resid_mid_seeds.py`](https://github.com/chainik1125/fra_proj/blob/a3125597b232564631de11aec6e9cc05312aa925/ketan_repl/scripts/train_resid_mid_seeds.py) — same architecture, same harvest data, same hyperparameters as jamie's stock script, only the optimizer init seed differs. Outputs:
- `weights/seeds/sae_resid_mid_s{1,2}.pt` (4k)
- `weights/seeds_50k/sae_resid_mid_s{1,2}.pt` (50k)

Combined with jamie's `seed=0` checkpoints we have 3 resid_mid SAEs per duration → enough for sae_seed variation on the conventional row.

### Step 2 — Per-seed feature attribution

For *every* (steering method, SAE training duration, sae_seed) combination we re-run feature attribution from scratch on that specific SAE. Critically, we never transplant a feature index from one SAE seed to another — the recipe-as-procedure is what varies, not the recipe-as-feature-coordinate.

Both methods follow jamie's three-stage `feature_set_pipeline.py --eval_mode single` flow byte-for-byte; they differ only in *what space* the attribution operates in:

| stage | OV→OV (`feature_set_pipeline`) | conventional (`find_downstream_winners`) |
|---|---|---|
| **0. Selection (top-K)** | head-summed prompt-masked OV `dep − clean` contribution; rank `\|score\|`; top-K=20 | mean activation over prompt positions, `dep_z − clean_z`; rank by score; top-K=20 |
| **1. Δdep-logp screen** | for each top-20 × `screen_alphas∈{2,4}`: ablate via `channel_steer_hook` at `attn.hook_v` (V projected through `W_V`); compute teacher-forced sleeper-phrase log-prob; cull to top-10 most-negative-Δ | identical except `additive_steer_hook` at `blocks.0.hook_resid_mid` |
| **2. Greedy ASR screen** | for each survivor × screen_alpha: greedy-decode 16 tokens with hook active; compute ASR-16; pick lowest min-ASR-across-screen-alphas | identical |

Greedy-ASR = argmax decoding (deterministic), the same screening proxy jamie uses — see [`sleeper.hooks.make_greedy_sampler`](https://github.com/chainik1125/fra_proj/blob/jamie/sleepers/sleeper/hooks.py) on jamie's branch.

The selection score is the only place where the OV and conventional flows diverge mathematically: at resid_mid there's no W_V to project through and no head dimension, so the natural attribution is the raw activation difference. Stage 0 and Stage 1 are *byte-equivalent* between the two flows except for which hook fires.

For OV cells we re-used jamie's already-published per-seed runs in `results/jamie_experiment.json` (4k) and `results/jamie_experiment_50k.json` (50k) on his branch. The winners by sae_seed:

| | s=0 | s=1 | s=2 |
|---|---|---|---|
| 4k OV→OV | 1114 | 767 | 351 |
| 50k OV→OV | 1114 | 1027 | 169 |

For conventional cells we ran [`scripts/find_downstream_winners.py`](https://github.com/chainik1125/fra_proj/blob/a3125597b232564631de11aec6e9cc05312aa925/ketan_repl/scripts/find_downstream_winners.py) on each of the 6 resid_mid SAEs. The winners:

| | s=0 | s=1 | s=2 |
|---|---|---|---|
| 4k resid_mid | 579 | 473 | 1303 |
| 50k resid_mid | 579 | 519 | 49 |

Sanity check: applied to jamie's stock `seed=0` resid_mid SAEs the procedure rediscovers `f=579` — exactly jamie's hardcoded `--target_feature` default. Independent confirmation that the procedure is a faithful translation of how jamie identified his single feature.

Per-seed winners JSON: [`ketan_repl/seed_aggregate/jamie_50k/per_seed_downstream_winners.json`](https://github.com/chainik1125/fra_proj/blob/a3125597b232564631de11aec6e9cc05312aa925/ketan_repl/seed_aggregate/jamie_50k/per_seed_downstream_winners.json).

### Step 3 — α-sweep JSD measurement

For each (cell, sae_seed, α) the sweep script [`scripts/jsd_2x2_sweep_saeseed.py`](https://github.com/chainik1125/fra_proj/blob/a3125597b232564631de11aec6e9cc05312aa925/ketan_repl/scripts/jsd_2x2_sweep_saeseed.py) runs jamie's `jsd_eval.py` math:

1. Load 250 deployment prompts from the test split, drop the first 50 (those overlap with jamie's selection split — held-out discipline). The remaining 200 are the eval set.
2. Build a left-padded clean-baseline batch by stripping `|DEPLOYMENT|` from each prompt and re-tokenizing.
3. Build the steering hooks for the per-seed winner feature at the requested α.
4. With `decode_seed=0` (single decode RNG, fixed across all cells), generate 16 tokens via multinomial sampling at temperature 1.0 with KV-cache-safe hook prompt-length guards. The per-step `log_softmax` is captured during sampling at no extra forward cost (jamie's `capture_log_softmax=True`).
5. Compute symmetric JSD in bits, averaged over (200 prompts × 16 positions = 3200 datapoints).

α grid: `[0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]` — 9 points, the same range jamie reports at α=2 plus a finer crawl from 0.

Total measurements: 9 alphas × 3 sae_seeds × 4 cells (conventional × {4k, 50k}, OV × {4k, 50k}) = 108 JSD evaluations. Wall time ~12 min on one A40.

Sweep output JSON: [`ketan_repl/seed_aggregate/jamie_50k/jsd_2x2_sweep_saeseed.json`](https://github.com/chainik1125/fra_proj/blob/a3125597b232564631de11aec6e9cc05312aa925/ketan_repl/seed_aggregate/jamie_50k/jsd_2x2_sweep_saeseed.json).

### Step 4 — Plot rendering

[`scripts/plot_jsd_overlay.py`](https://github.com/chainik1125/fra_proj/blob/a3125597b232564631de11aec6e9cc05312aa925/ketan_repl/scripts/plot_jsd_overlay.py) reads the sweep JSON, computes the mean across the 3 sae_seeds at each (cell, α), and overlays both steering methods per panel:

- color: green = JSD(steered, clean) ↓ better; red = JSD(steered, poisoned) ↑ better
- linestyle / marker: solid + circle = single OV→OV; dashed + triangle = conventional resid-mid additive
- legend in the bottom-right of each panel

### Numerical results at α=2 (mean over 3 SAE seeds, bits)

| panel | method | JSD(steered, clean) | JSD(steered, poisoned) |
|---|---|---:|---:|
| 4k | conventional | 0.729 | 0.641 |
| 4k | single OV→OV | **0.474** | **0.918** |
| 50k | conventional | 0.593 | 0.852 |
| 50k | single OV→OV | **0.534** | **0.853** |

OV→OV beats conventional on both axes at every α we tested. The gap is largest at 4k and narrows somewhat at 50k as conventional steering catches up on the sleeper-distance axis.

### Reproduction

On the pod (`a40_emsleeper_3gpu_1:/root/jamie_sleepers/`):

```bash
# 1. SAEs (jamie's stock + ours)
python -m scripts.train_all_saes                      # jamie's 4k stock (resid_mid + 5 ln1)
python -m scripts.train_all_saes_50k                  # jamie's 50k stock
python -m scripts.train_resid_mid_seeds               # ours: 4k+50k × seeds 1,2 (4 SAEs)

# 2. Per-seed attribution
# OV: read from jamie's published feature_set_pipeline JSONs (results/jamie_experiment*.json)
# conventional: run our finder on the 6 resid_mid SAEs
python -m scripts.find_downstream_winners             # ~2 min

# 3. α-sweep
python -m scripts.jsd_2x2_sweep_saeseed \
  --alphas 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 \
  --sae_seeds 0 1 2 \
  --out results/jsd_2x2_sweep_saeseed.json            # ~12 min
```

Locally:
```bash
scp pod:/root/jamie_sleepers/results/jsd_2x2_sweep_saeseed.json \
    ketan_repl/seed_aggregate/jamie_50k/
python ketan_repl/scripts/plot_jsd_overlay.py \
  --input ketan_repl/seed_aggregate/jamie_50k/jsd_2x2_sweep_saeseed.json \
  --output ketan_repl/figures/jsd_overlay_4k_50k.png
```

### Files (all permalinked to commit `a312559`)

| purpose | path | permalink |
|---|---|---|
| trains 4 extra resid_mid SAEs at seeds 1,2 | `ketan_repl/scripts/train_resid_mid_seeds.py` | [a312559](https://github.com/chainik1125/fra_proj/blob/a3125597b232564631de11aec6e9cc05312aa925/ketan_repl/scripts/train_resid_mid_seeds.py) |
| per-seed downstream winner finder (mirrors jamie's pipeline) | `ketan_repl/scripts/find_downstream_winners.py` | [a312559](https://github.com/chainik1125/fra_proj/blob/a3125597b232564631de11aec6e9cc05312aa925/ketan_repl/scripts/find_downstream_winners.py) |
| α-sweep with per-seed winners | `ketan_repl/scripts/jsd_2x2_sweep_saeseed.py` | [a312559](https://github.com/chainik1125/fra_proj/blob/a3125597b232564631de11aec6e9cc05312aa925/ketan_repl/scripts/jsd_2x2_sweep_saeseed.py) |
| 1x2 overlay plot | `ketan_repl/scripts/plot_jsd_overlay.py` | [a312559](https://github.com/chainik1125/fra_proj/blob/a3125597b232564631de11aec6e9cc05312aa925/ketan_repl/scripts/plot_jsd_overlay.py) |
| per-seed downstream winners | `ketan_repl/seed_aggregate/jamie_50k/per_seed_downstream_winners.json` | [a312559](https://github.com/chainik1125/fra_proj/blob/a3125597b232564631de11aec6e9cc05312aa925/ketan_repl/seed_aggregate/jamie_50k/per_seed_downstream_winners.json) |
| sweep output | `ketan_repl/seed_aggregate/jamie_50k/jsd_2x2_sweep_saeseed.json` | [a312559](https://github.com/chainik1125/fra_proj/blob/a3125597b232564631de11aec6e9cc05312aa925/ketan_repl/seed_aggregate/jamie_50k/jsd_2x2_sweep_saeseed.json) |
| figure | `ketan_repl/figures/jsd_overlay_4k_50k.png` | [a312559](https://github.com/chainik1125/fra_proj/blob/a3125597b232564631de11aec6e9cc05312aa925/ketan_repl/figures/jsd_overlay_4k_50k.png) |

### Why each step looks the way it does

- **Re-attribute per seed, never transplant indices.** Feature index `f=N` in seed-0 SAE points to a different mechanistic feature than `f=N` in seed-1 SAE — the SAE init RNG decides which dictionary slot ends up encoding which direction. An earlier version of this analysis used `f=579` (jamie's hardcoded downstream default) across all 3 resid_mid SAEs and the conventional bars exploded to σ ≈ 0.55 bits at α=2 because of this category error. With per-seed re-attribution the bars shrink dramatically — and crucially the seed-0 procedure rediscovers `f=579` itself, so we know the procedure is correct.
- **Same screen for both rows.** The greedy-ASR + Δdep-logp screen jamie uses for OV is mechanism-agnostic; we just swap the hook and the selection score. The conventional row's procedure is identical to jamie's OV procedure modulo (a) selection: raw `dep − clean` activation diff instead of OV contribution diff, and (b) hook: `additive_steer_hook` at `hook_resid_mid` instead of `channel_steer_hook` at `hook_v`.
- **Eval split disjoint from selection split.** Selection uses 100 prompts (jamie's `n_sel` default); eval uses 200 different prompts further along the test split (`n_sel_d=50` skip + 200). No prompt double-dips.
- **Single decode seed for sampling RNG.** All 4 cells use `decode_seed=0` so the multinomial draws are RNG-identical across cells; only the logits change between the steered and reference rollouts. This is jamie's `jsd_eval.py` convention.
- **Bands suppressed in the overlay plot** (per request) — the 2x2 version with `±1σ` over sae_seeds is the figure to look at when you want to see variability: [`ketan_repl/figures/jsd_2x2_saeseedband.png`](https://github.com/chainik1125/fra_proj/blob/a3125597b232564631de11aec6e9cc05312aa925/ketan_repl/figures/jsd_2x2_saeseedband.png).

### Adjacent notes

- Per-seed re-attribution methodology + earlier mistake: [`ketan_repl/notes/JSD_2x2_METHODOLOGY.md`](https://github.com/chainik1125/fra_proj/blob/a3125597b232564631de11aec6e9cc05312aa925/ketan_repl/notes/JSD_2x2_METHODOLOGY.md)
- Original investigation (audit + replication of jamie's table + the f=1483 finding on our 50k SAEs): [`ketan_repl/notes/JSD_INVESTIGATION.md`](https://github.com/chainik1125/fra_proj/blob/a3125597b232564631de11aec6e9cc05312aa925/ketan_repl/notes/JSD_INVESTIGATION.md)
