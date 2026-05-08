## Overnight replication of Ketan's sleeper Pareto-frontier result

Branch: `dmitry/sleeper_repl` (worktree: `.claude/worktrees/dmitry-sleeper-repl`).
Pod: `a40_emsleeper_3gpu_1` (3× A40, RunPod). Single-GPU pipelines, 3 in parallel for the seeded sweep.

### Goal

Replicate, then strengthen, Ketan's headline result: at `blocks.0.ln1.hook_normalized`, ranking ln1 SAE features by one-stage OV signed contribution and intervening at `hook_v` only gives a perfect Pareto frontier (`ASR_16=0.00` at `ΔCE=0`, α∈{2,3}). Reference numbers are committed in `experiments/tinystories_sleeper/tracing_feature/qk_vs_ov/results/pareto_3x3.json` and described in `qk_vs_ov/working_notes/10_3x3_rank_vs_intervene.md`.

Ketan reports a discrepancy: his most recent plots show the FRA-based intervention curve has a much smaller (better) AUC on RunPod CUDA than on his Mac MPS. Replication here is **CUDA-only** (no MPS leg this round); the goal of overnight is to lock down the CUDA numbers across seeds before chasing MPS.

### Stages

| # | Stage | Note file | Status |
|---|---|---|---|
| 0 | Pod setup, 4k-step smoke | `00_setup_and_smoke.md` | **done** — pipeline validated end-to-end on smoke SAEs |
| 1 | 3 seeds × 50k steps SAE training (Ketan's arch) | `01_seeded_training.md` | **done** — 6/6 jobs DONE in ~70 min wall |
| 2 | Per-seed `pareto_3x3.py` + AUC | `02_seed_consistency.md` | **done** — see headline below |
| 3 | Improved XE metric (proposal + impl) | `03_xe_metric.md` | writeup done; impl is a stub awaiting spec confirmation |
| 4 | Overnight status snapshots | `STATUS.md` | cron deleted after all-DONE trigger fired |

### Headline finding from stage 2

`(ov, ov)` quality across 3 seeds = **0.770 ± 0.091** (mean ± std), vs Ketan's reference 1.00. The OV-rank → V-intervene story is **directionally correct** (it beats QK in every seed) but the reference "perfect Pareto" is **not generic** — none of our 3 seeds reaches it for the ln1-pathway-only intervention. Best single (cell, seed) is `seed2 (ov, all) = 0.954`.

This is a real seed-instability finding. See `02_seed_consistency.md` for the full table, comparison vs Ketan's reference, and proposed follow-ups.

### Other artifacts produced overnight

- `experiments/tinystories_sleeper/recreate_{layer0,ln1}_seed{0,1,2}/results/` — full SAE checkpoints + sweep results pulled back to local (without activations cache).
- `ketan_repl/seed{0,1,2}/` — per-seed `pareto_3x3.json/png/zoom.png/summary.json` + `features.json`.
- `ketan_repl/seed_aggregate/` — cross-seed plot + JSON.
- `ketan_repl/notes/_aggregate_table.md` — markdown table (also inlined into `02_seed_consistency.md`).

### Ketan's SAE — what we are matching

For the qk_vs_ov experiment, the SAE that gets steered is `recreate_ln1/results/crosscoder_sae_layer0.pt`:

- TopK SAE, `d_in=768` (TinyStories d_model), `d_sae=1536` (2× expansion), `k_total=32`.
- Trained on activations at `blocks.0.ln1.hook_normalized` from 10000 sequences, seq_len=128.
- Original training: 4000 steps, batch 4096, lr 5e-4, seed 0. Overnight bump: 50k steps, three seeds.
- `pareto_3x3.py` additionally needs the resid_pre and resid_mid SAEs (sae_layer0 and sae_layer1 in `recreate_layer0/`) for the cache-building step in `cache_layer0_activations.py`. Those are also retrained per seed.

### Reproducibility caveats

- `pareto_3x3.py` has no global `torch.manual_seed` and no `cudnn.deterministic` flags today. Greedy generation is sensitive to softmax tie-breaking — even at fixed weights, A40 vs Mac MPS softmax kernels may produce different logit orderings. We add a determinism block for the third (locked) run; the first run is unchanged for direct parity vs committed numbers.
- The seed referenced in pipeline configs (`harvest.seed`, `train.seed`) controls dataset shuffling and SAE init. Steering eval (the pareto sweep itself) does not currently consume a seed because greedy decoding is deterministic given the model + cache.

### Pod / environment

- Driver: NVIDIA 570.211.01, CUDA 12.8 reported. `torch==2.11.0+cu128` (installed via `--index-url https://download.pytorch.org/whl/cu128`; the default cu130 wheel didn't see the driver).
- HF Hub: token forwarded transiently per ssh invocation; private dataset repo `dmanningcoe/fra-tinystories-sleeper-saes` holds the trained SAEs (one folder per seed, plus the smoke run).

### Files added this branch

| File | Purpose |
|---|---|
| `experiments/tinystories_sleeper/hf_artifacts.py` | Push/pull/list artifacts to/from `dmanningcoe/fra-tinystories-sleeper-saes`. |
| `experiments/tinystories_sleeper/tracing_feature/qk_vs_ov/scripts/diff_pareto.py` | Per-cell ASR/ΔCE diff between two `pareto_3x3.json` files. |
| `experiments/tinystories_sleeper/recreate_layer0/reproduce.py` (modified) | `--push-to-hf` / `--pull-from-hf` flags. |
| `experiments/tinystories_sleeper/recreate_ln1/reproduce.py` (modified) | Same flags. |
| `ketan_repl/notes/*.md` | This folder — stage notes + overnight STATUS log. |
