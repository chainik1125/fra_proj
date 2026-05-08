## Stage 1 — 3 seeds × 50k-step SAE training

### Identity of the SAE Ketan trained

The headline-result SAE is the **first-layer ln1 SAE** at `blocks.0.ln1.hook_normalized`. It's a TopK SAE with these hyperparameters (from `experiments/tinystories_sleeper/recreate_ln1/config.yaml`):

| Hyperparameter | Value |
|---|---|
| Architecture | TopK SAE (sae_lens-style, no shrinking) |
| `d_in` | 768 (TinyStories d_model) |
| `d_sae` | 1536 (2× expansion) |
| `k_total` | 32 (per-position TopK sparsity) |
| Optimizer | Adam |
| Batch size | 4096 |
| Learning rate | 5e-4 |
| Activation normalization | every 100 steps |
| Original training length | **4000 steps** in Ketan's referenced run |

`pareto_3x3.py` also reads the resid_pre and resid_mid SAEs from `recreate_layer0/results/` (`sae_layer0` and `sae_layer1`) — these are required for the cache build (`cache_layer0_activations.py`) to compute z_pre, z_mid encodings. Same architecture (TopK, d_sae=1536, k=32). For the qk_vs_ov experiment itself the resid_pre/mid SAEs are *not* used as steering channels; they're upstream context for the OV decomposition.

### Overnight run config

For the seeded sweep we override two values via per-seed copies of `recreate_layer0_seed{0,1,2}/config.yaml` and `recreate_ln1_seed{0,1,2}/config.yaml`:

| Field | Original | Overnight |
|---|---|---|
| `harvest.seed` | 0 | {0, 1, 2} per copy |
| `train.seed` | 0 | {0, 1, 2} per copy |
| `train.n_steps` | 4000 | **50000** |

All other hyperparameters identical to the as-committed config. Each seed gets its own `recreate_*_seed{N}/results/` directory; outputs do not collide.

### Parallelism plan

3 GPUs, 6 jobs (3 seeds × {layer0, ln1}). Each pipeline trains all archs for that variant in a single Python process (3 archs for layer0, 4 archs for ln1) — they share the same activations cache and batch indexing.

- **GPU 0**: `recreate_layer0_seed0` + `recreate_ln1_seed0` concurrently (peak ~22 GB)
- **GPU 1**: same for seed 1
- **GPU 2**: same for seed 2

Confirmed via `nvidia-smi` after launch: each GPU at 100 % util, 21 GB allocated. The first ticks of `STATUS.md` show all 6 jobs `ALIVE` and progressing at consistent it/s across seeds.

### Throughput observed (from STATUS.md tick at T+25 min)

| job | step | progress | it/s |
|---|---:|---:|---:|
| seed=0, layer0 | 16400/50000 | 33% | 23.6 |
| seed=0, ln1    | 13200/50000 | 26% | 18.9 |
| seed=1, layer0 | 16400/50000 | 33% | 24.3 |
| seed=1, ln1    | 13200/50000 | 26% | 19.2 |
| seed=2, layer0 | 16400/50000 | 33% | 23.0 |
| seed=2, ln1    | 13200/50000 | 26% | 18.6 |

Projection: layer0 finishes training in ~24 min from the tick, ln1 in ~32 min. Each is followed by a sweep stage (~10 min) and plot. Total wall time for stage 1: roughly 90 min from launch.

### Auto-restart

`watch_and_restart.sh` polls every 30 min. If a job is dead and `MANIFEST.md` is missing for its `results/` dir, it relaunches with `--skip` for any stages whose outputs already exist (e.g. if harvest finished but training crashed, only train+sweep+plot rerun). Restarts are pinned to the same GPU as the original job.

### What stage 2 needs from this stage

- `recreate_layer0_seed{N}/results/crosscoder_sae_layer0.pt` (resid_pre)
- `recreate_layer0_seed{N}/results/crosscoder_sae_layer1.pt` (resid_mid)
- `recreate_ln1_seed{N}/results/crosscoder_sae_layer0.pt` (ln1 — the steered SAE)
- The corresponding `RESULTS.md` per pipeline (for the per-arch single-feature ASR baselines).

When all 6 jobs reach `DONE` the watcher fires `run_seed_analysis.sh <seed>` per seed in parallel on GPUs 0/1/2.
