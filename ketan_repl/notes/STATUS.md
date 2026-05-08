## Overnight watcher status log

Watcher cron job ID: **d7730327**, fires at minutes :13 and :43 (every 30 min, off the
:00/:30 mark to avoid the synchronized-load thundering herd).

Each tick runs `ketan_repl/scripts/watch_and_restart.sh`, which:
- ssh's into `a40_emsleeper_3gpu_1`
- Checks every (seed, kind) job: state ∈ {DONE, ALIVE, RESTART_<skipped-stages>}
- Restarts dead-and-incomplete jobs with `--skip` for stages whose outputs already exist
- Reports GPU utilization

When all 6 jobs reach DONE, the watcher kicks off `run_seed_analysis.sh <seed>` for
seeds 0/1/2 in parallel on GPUs 0/1/2, then deletes the cron via CronDelete.

The cron is session-only (in-memory). If the conversation gets restarted mid-overnight,
the cron is gone and the watcher needs to be re-armed manually.

---

## tick @ 2026-05-08T06:16:10Z

```text
GPU 0, 100 %, 21374 MiB
GPU 1, 100 %, 21374 MiB
GPU 2, 100 %, 21374 MiB
JOB seed=0 kind=layer0  state=ALIVE                     last="[train] step 16400/50000 (23.6 it/s) | sae_layer0=0.0346 sae_layer1=1.7594 sae_layer2=9.0785"
JOB seed=0 kind=ln1     state=ALIVE                     last="[train] step 13200/50000 (18.9 it/s) | sae_layer0=12.9148 sae_layer1=29.2347 sae_layer2=46.5393 sae_layer3=77."
JOB seed=1 kind=layer0  state=ALIVE                     last="[train] step 16400/50000 (24.3 it/s) | sae_layer0=0.0358 sae_layer1=1.7896 sae_layer2=9.3868"
JOB seed=1 kind=ln1     state=ALIVE                     last="[train] step 13200/50000 (19.2 it/s) | sae_layer0=13.5854 sae_layer1=29.6684 sae_layer2=48.2457 sae_layer3=78."
JOB seed=2 kind=layer0  state=ALIVE                     last="[train] step 16400/50000 (23.0 it/s) | sae_layer0=0.0323 sae_layer1=1.7466 sae_layer2=9.1016"
JOB seed=2 kind=ln1     state=ALIVE                     last="[train] step 13200/50000 (18.6 it/s) | sae_layer0=14.5369 sae_layer1=29.9844 sae_layer2=48.0738 sae_layer3=78."
```

## tick @ 2026-05-08T06:17:02Z

```text
GPU 0, 100 %, 21374 MiB
GPU 1, 100 %, 21374 MiB
GPU 2, 100 %, 21374 MiB
JOB seed=0 kind=layer0  state=ALIVE                     last="[train] step 16400/50000 (23.6 it/s) | sae_layer0=0.0346 sae_layer1=1.7594 sae_layer2=9.0785"
JOB seed=0 kind=ln1     state=ALIVE                     last="[train] step 13200/50000 (18.9 it/s) | sae_layer0=12.9148 sae_layer1=29.2347 sae_layer2=46.5393 sae_layer3=77."
JOB seed=1 kind=layer0  state=ALIVE                     last="[train] step 16400/50000 (24.3 it/s) | sae_layer0=0.0358 sae_layer1=1.7896 sae_layer2=9.3868"
JOB seed=1 kind=ln1     state=ALIVE                     last="[train] step 13200/50000 (19.2 it/s) | sae_layer0=13.5854 sae_layer1=29.6684 sae_layer2=48.2457 sae_layer3=78."
JOB seed=2 kind=layer0  state=ALIVE                     last="[train] step 16400/50000 (23.0 it/s) | sae_layer0=0.0323 sae_layer1=1.7466 sae_layer2=9.1016"
JOB seed=2 kind=ln1     state=ALIVE                     last="[train] step 13200/50000 (18.6 it/s) | sae_layer0=14.5369 sae_layer1=29.9844 sae_layer2=48.0738 sae_layer3=78."
```

## tick @ 2026-05-08T06:37:57Z

```text
GPU 0, 100 %, 23530 MiB
GPU 1, 100 %, 23530 MiB
GPU 2, 100 %, 23530 MiB
JOB seed=0 kind=layer0  state=ALIVE                     last="$ /root/fra_proj/.venv/bin/python run_ablation_sweep.py --top_k 100 --stage2_keep 10 --alphas 0.25 0.5 1.0 1.5"
JOB seed=0 kind=ln1     state=ALIVE                     last="[train] step 27400/50000 (18.9 it/s) | sae_layer0=14.7269 sae_layer1=29.7737 sae_layer2=46.8930 sae_layer3=77."
JOB seed=1 kind=layer0  state=ALIVE                     last="$ /root/fra_proj/.venv/bin/python run_ablation_sweep.py --top_k 100 --stage2_keep 10 --alphas 0.25 0.5 1.0 1.5"
JOB seed=1 kind=ln1     state=ALIVE                     last="[train] step 41600/50000 (19.2 it/s) | sae_layer0=13.0400 sae_layer1=29.7550 sae_layer2=48.1104 sae_layer3=78."
JOB seed=2 kind=layer0  state=ALIVE                     last="$ /root/fra_proj/.venv/bin/python run_ablation_sweep.py --top_k 100 --stage2_keep 10 --alphas 0.25 0.5 1.0 1.5"
JOB seed=2 kind=ln1     state=ALIVE                     last="[train] step 41600/50000 (18.7 it/s) | sae_layer0=13.9723 sae_layer1=28.7440 sae_layer2=45.9563 sae_layer3=75."
```

## tick @ 2026-05-08T06:55:45Z

```text
GPU 0, 100 %, 4414 MiB
GPU 1, 100 %, 20710 MiB
GPU 2, 100 %, 20710 MiB
JOB seed=0 kind=layer0  state=ALIVE                     last="$ /root/fra_proj/.venv/bin/python run_ablation_sweep.py --top_k 100 --stage2_keep 10 --alphas 0.25 0.5 1.0 1.5"
JOB seed=0 kind=ln1     state=ALIVE                     last="$ /root/fra_proj/.venv/bin/python run_ablation_sweep.py --top_k 100 --stage2_keep 10 --alphas 0.25 0.5 1.0 1.5"
JOB seed=1 kind=layer0  state=ALIVE                     last="$ /root/fra_proj/.venv/bin/python run_ablation_sweep.py --top_k 100 --stage2_keep 10 --alphas 0.25 0.5 1.0 1.5"
JOB seed=1 kind=ln1     state=ALIVE                     last="$ /root/fra_proj/.venv/bin/python run_ablation_sweep.py --top_k 100 --stage2_keep 10 --alphas 0.25 0.5 1.0 1.5"
JOB seed=2 kind=layer0  state=ALIVE                     last="$ /root/fra_proj/.venv/bin/python run_ablation_sweep.py --top_k 100 --stage2_keep 10 --alphas 0.25 0.5 1.0 1.5"
JOB seed=2 kind=ln1     state=ALIVE                     last="$ /root/fra_proj/.venv/bin/python run_ablation_sweep.py --top_k 100 --stage2_keep 10 --alphas 0.25 0.5 1.0 1.5"
```

**Interpretation (06:43 tick)**: All 6 jobs progressed past training and into the sweep stage. GPU 0 dropped to 4.4 GB (one of seed-0's two pipelines moved to plot/MANIFEST while the other still sweeps); GPUs 1 & 2 still at 20 GB each (both pipelines still sweeping). No restarts needed; GPU util healthy. No DONE yet — sweep stage typically runs 10–20 min depending on top_k feature count and arch count. Expected next-tick (07:13): some jobs DONE, possibly all 6.

## tick @ 2026-05-08T07:25:47Z

```text
GPU 0, 0 %, 0 MiB
GPU 1, 0 %, 0 MiB
GPU 2, 0 %, 0 MiB
JOB seed=0 kind=layer0  state=DONE                      last="[plot] wrote /root/fra_proj/experiments/tinystories_sleeper/recreate_layer0_seed0/results/RESULTS.md"
JOB seed=0 kind=ln1     state=DONE                      last="[plot] wrote /root/fra_proj/experiments/tinystories_sleeper/recreate_ln1_seed0/results/RESULTS.md"
JOB seed=1 kind=layer0  state=DONE                      last="[plot] wrote /root/fra_proj/experiments/tinystories_sleeper/recreate_layer0_seed1/results/RESULTS.md"
JOB seed=1 kind=ln1     state=DONE                      last="[plot] wrote /root/fra_proj/experiments/tinystories_sleeper/recreate_ln1_seed1/results/RESULTS.md"
JOB seed=2 kind=layer0  state=DONE                      last="[plot] wrote /root/fra_proj/experiments/tinystories_sleeper/recreate_layer0_seed2/results/RESULTS.md"
JOB seed=2 kind=ln1     state=DONE                      last="[plot] wrote /root/fra_proj/experiments/tinystories_sleeper/recreate_ln1_seed2/results/RESULTS.md"
```

**Interpretation (07:13 tick)** — ALL 6 JOBS DONE. Each (seed, kind) wrote its `RESULTS.md` and `MANIFEST.md`. GPUs idle (0% / 0 MiB). Triggering `run_seed_analysis.sh <seed>` for seeds 0/1/2 in parallel on GPUs 0/1/2. Watcher cron will be deleted next.

**Final tick (07:30 UTC)** — Per-seed analysis pipeline completed for all 3 seeds. Watcher cron `d7730327` deleted. Cross-seed aggregation complete; see `02_seed_consistency.md` and `_aggregate_table.md`. All seeded SAE checkpoints + per-seed results rsynced back to local. Pod retains all artifacts as the canonical copy.
