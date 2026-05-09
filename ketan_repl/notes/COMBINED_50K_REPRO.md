## Reproducing `figures/combined_50k.{png,pdf}`

Self-contained recipe to regenerate the 1×2 50k figure (JSD on the left, layman rollout-level on the right) with mean lines and min-max-across-SAE-seeds shaded bands.

Final figure: [`ketan_repl/figures/combined_50k.png`](https://github.com/chainik1125/fra_proj/blob/dmitry/sleeper_repl/ketan_repl/figures/combined_50k.png) / [`.pdf`](https://github.com/chainik1125/fra_proj/blob/dmitry/sleeper_repl/ketan_repl/figures/combined_50k.pdf)

Plot script: [`ketan_repl/scripts/plot_combined_50k.py`](https://github.com/chainik1125/fra_proj/blob/dmitry/sleeper_repl/ketan_repl/scripts/plot_combined_50k.py).

### One-line run (assumes the pod is set up — see "Setup" below)

From this repo root, with ssh access to the pod (`a40_emsleeper_3gpu_1`):

```bash
ssh a40_emsleeper_3gpu_1 'cd /root/jamie_sleepers && \
  python -m scripts.train_all_saes && \
  python -m scripts.train_all_saes_50k && \
  python -m scripts.train_resid_mid_seeds && \
  python -m scripts.train_extra_seeds && \
  python -m scripts.find_downstream_winners && \
  python -m scripts.feature_set_pipeline \
      --selection_method jamie --top_k 20 --screen_alphas 2 4 \
      --eval_mode single --sae_seeds 5 --alphas 0 0.5 1 1.5 2 \
      --sae_ln1_dir weights/seeds_50k --sae_mid weights/sae_resid_mid_50k.pt \
      --out results/jamie_pipeline_ln1_s5_50k.json && \
  python -m scripts.jsd_2x2_sweep_saeseed \
      --alphas 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 \
      --sae_seeds 0 1 2 \
      --out results/jsd_2x2_sweep_full_metrics.json' && \
scp a40_emsleeper_3gpu_1:/root/jamie_sleepers/results/jsd_2x2_sweep_full_metrics.json \
    ketan_repl/seed_aggregate/jamie_50k/ && \
uv run python ketan_repl/scripts/plot_combined_50k.py \
    --input ketan_repl/seed_aggregate/jamie_50k/jsd_2x2_sweep_full_metrics.json \
    --output ketan_repl/figures/combined_50k
```

(Each `python -m scripts.*` step short-circuits if its outputs already exist, so this command is idempotent and resumable.)

### What each step does

1. **`scripts.train_all_saes`** — jamie's stock 4k SAE training (one resid_mid + 5 ln1 SAEs at seeds 0..4). ~5 min total on one A40.
2. **`scripts.train_all_saes_50k`** — jamie's stock 50k SAE training (same six SAEs trained for 50k steps instead of 4k). ~30 min wall.
3. **`scripts.train_resid_mid_seeds`** ([`ketan_repl/scripts/train_resid_mid_seeds.py`](https://github.com/chainik1125/fra_proj/blob/dmitry/sleeper_repl/ketan_repl/scripts/train_resid_mid_seeds.py)) — adds 4 resid_mid SAEs (4k+50k × seeds 1, 2). ~10 min.
4. **`scripts.train_extra_seeds`** ([`ketan_repl/scripts/train_extra_seeds.py`](https://github.com/chainik1125/fra_proj/blob/dmitry/sleeper_repl/ketan_repl/scripts/train_extra_seeds.py)) — adds 4 50k SAEs needed for seed 5 (1 ln1) and resid_mid seeds 3, 4, 5. ~22 min.
5. **`scripts.find_downstream_winners`** ([`ketan_repl/scripts/find_downstream_winners.py`](https://github.com/chainik1125/fra_proj/blob/dmitry/sleeper_repl/ketan_repl/scripts/find_downstream_winners.py)) — per-seed downstream attribution on each resid_mid SAE. Mirrors jamie's `feature_set_pipeline.py --eval_mode single` flow byte-for-byte (selection by `dep − clean` activation difference; stage-0 Δdep-logp screen at α∈{2, 4}; stage-1 greedy-ASR screen on top-K/2 survivors; lowest-min-ASR wins). Output: `results/per_seed_downstream_winners.json`. ~3 min.
6. **`scripts.feature_set_pipeline ... --sae_seeds 5`** — jamie's pipeline run for the OV winner on `weights/seeds_50k/sae_ln1_s5.pt` (only sae_seed=5; seeds 0..4 already in `results/jamie_experiment_50k.json` published on jamie's branch). ~10 min.
7. **`scripts.jsd_2x2_sweep_saeseed`** ([`ketan_repl/scripts/jsd_2x2_sweep_saeseed.py`](https://github.com/chainik1125/fra_proj/blob/dmitry/sleeper_repl/ketan_repl/scripts/jsd_2x2_sweep_saeseed.py)) — α-sweep over the 4 cells (`conventional × {4k, 50k}` and `OV-single × {4k, 50k}`), per-seed re-attributed winners on both rows. For each (cell, sae_seed, α) computes:
   - `jsd_clean` and `jsd_pois` via jamie's `jsd_eval` math (200 prompts × 16 positions, multinomial sampling at temperature 1.0 with `decode_seed=0`)
   - `n_exact_match_clean` (full 16-token rollout match between steered and clean rollout, same RNG)
   - `frac_pos_match_clean` (per-position match fraction)
   - `asr` (sleeper-phrase regex match rate, `sleeper.metrics.asr_16`)
   ~12 min.
8. **`scp` + `plot_combined_50k.py`** — pulls the sweep JSON locally and renders the figure.

### Setup (one-time per pod)

```bash
ssh a40_emsleeper_3gpu_1
git clone -b jamie/sleepers https://github.com/chainik1125/fra_proj.git /root/jamie_sleepers
cd /root/jamie_sleepers
uv sync
# Mirror our extra scripts from this repo's ketan_repl/scripts/ into scripts/
# (train_resid_mid_seeds.py, train_extra_seeds.py, find_downstream_winners.py,
#  jsd_2x2_sweep_saeseed.py, plot_combined_50k.py)
```

### Methodology notes

The crucial discipline that makes the figure honest:

- **Per-seed re-attribution on both rows.** No transplanting feature indices across SAEs. For OV cells we use the per-seed winners published in jamie's `results/jamie_experiment_50k.json`; for conventional cells we run our `find_downstream_winners.py` on each resid_mid SAE separately. Sanity check: the procedure rediscovers jamie's hardcoded `f=579` for the seed-0 resid_mid SAEs (both 4k and 50k), confirming the procedure is faithful.
- **Disjoint selection / eval splits.** Jamie's pipeline draws 100 prompts from the `val` split for selection + screening; our JSD α-sweep uses 200 different prompts from the `test` split (skipping the first 50 to avoid overlap with what jamie's `jsd_eval.py` excludes). No prompt double-dips.
- **Single decode RNG (`decode_seed=0`).** All comparisons share the same multinomial sampling RNG, so deltas between methods are due to logit shifts, not RNG draws.
- **Bands are min-to-max across SAE seeds, not ±σ.** With n=3 sample seeds the std-band overshoots the JSD ceiling for purely statistical reasons; min/max is honest about what was observed.

### Figures referenced in this writeup (commit `aeb4e23`)

| | path | permalink |
|---|---|---|
| Combined 50k figure | `ketan_repl/figures/combined_50k.png` | [aeb4e23](https://github.com/chainik1125/fra_proj/blob/aeb4e23/ketan_repl/figures/combined_50k.png) |
| Pre-existing 4k+50k JSD overlay | `ketan_repl/figures/jsd_overlay_4k_50k.png` | [a312559](https://github.com/chainik1125/fra_proj/blob/a312559/ketan_repl/figures/jsd_overlay_4k_50k.png) |
| Layman 4k+50k overlay | `ketan_repl/figures/layman_overlay_4k_50k.png` | [4e836bd](https://github.com/chainik1125/fra_proj/blob/4e836bd/ketan_repl/figures/layman_overlay_4k_50k.png) |
| Word-match exact / per-position | `ketan_repl/figures/word_match_*.png` | [a69d988](https://github.com/chainik1125/fra_proj/blob/a69d988/ketan_repl/figures/word_match_exact.png) |

### Companion writeups

- [`JSD_INVESTIGATION.md`](https://github.com/chainik1125/fra_proj/blob/dmitry/sleeper_repl/ketan_repl/notes/JSD_INVESTIGATION.md) — original investigation: audit of jamie's writeup vs his code, replication of his published table, the f=1483 finding on our 50k SAEs.
- [`JSD_2x2_METHODOLOGY.md`](https://github.com/chainik1125/fra_proj/blob/dmitry/sleeper_repl/ketan_repl/notes/JSD_2x2_METHODOLOGY.md) — per-seed re-attribution for the 2x2 plot, including the earlier feature-index-transplant mistake and how it was fixed.
- [`JSD_OVERLAY_WRITEUP.md`](https://github.com/chainik1125/fra_proj/blob/dmitry/sleeper_repl/ketan_repl/notes/JSD_OVERLAY_WRITEUP.md) — earlier 1×2 JSD-only overlay end-to-end recipe (precursor to this combined figure).
