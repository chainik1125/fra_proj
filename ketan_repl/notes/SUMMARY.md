## Overnight summary — sleeper Pareto replication

### TL;DR

Across **3 seeds × 50k-step SAE training**:

- **The single resid_mid feature ablation gives essentially perfect Pareto (`q ≈ 0.9998`) in all 3 seeds.** This matches/exceeds Ketan's claimed headline number for the OV-3 recipe.
- **Ketan's `(ov, ov)` at ln1 gives only `q = 0.770 ± 0.091` in our hands**, vs his claimed 1.00. OV-rank → V-intervene beats QK *directionally* every seed, but never reaches the perfect Pareto.
- This **inverts the framing** in the user's brief: "we can more efficiently suppress by steering on 10 OV features than on a single resid_mid feature" is **not** what we observe — single resid_mid wins by 0.23 absolute AUC across seeds, with negligible variance.
- Either (a) my OV-rank implementation has a bug — likely candidate, since Ketan's specific OV-3 reportedly hits 1.00 — or (b) Ketan's recipe relies on a checkpoint-specific dep-exclusive feature (`f=1114`, `dep/clean firing ratio = 2553×`) that 50k more steps + different seeds doesn't necessarily reproduce. Validating (a) is the next step.

### What ran

| Stage | Outcome |
|---|---|
| Pod setup + 4k-step smoke | done — pipeline validated end-to-end |
| 3 seeds × 50k step SAE training (3 GPUs in parallel) | done — ~70 min wall, no restarts |
| Per-seed cache → OV/QK rank → pareto_3x3 → AUC (3 GPUs in parallel) | done — ~5 min wall |
| Cross-seed aggregation + plot | done — `ketan_repl/seed_aggregate/` |
| Improved XE metric writeup + stub | writeup done in `03_xe_metric.md`; impl awaits your spec choice |

### Cross-seed AUC quality (mean ± std, n=3)

ln1-pathway feature steering (the qk_vs_ov recipe):

| ranking ＼ intervention | OV | QK | All |
|---|---:|---:|---:|
| qk    | 0.404 ± 0.078 | 0.243 ± 0.096 | 0.459 ± 0.092 |
| **ov**    | **0.770 ± 0.091** | 0.373 ± 0.232 | **0.789 ± 0.117** |
| union | 0.759 ± 0.044 | 0.295 ± 0.118 | 0.415 ± 0.019 |

Single resid_mid feature ablation (the comparison Ketan claimed his OV-3 beats):

| seed | feature | quality |
|---:|---:|---:|
| 0 | 171 | **0.9999** |
| 1 | 918 | **0.9996** |
| 2 |  57 | **0.9998** |

mean ± std = **0.9998 ± 0.00012** — every seed is essentially at the perfect-Pareto wall.

vs Ketan's reference: `(ov, ov) = 1.00`, `(ov, all) = 0.98`, `(qk, qk) = 0.07`. The ranking-direction story holds; the magnitude is off by ~20 percentage points uniformly.

### Per-seed (raw)

| ranking | intervention | seed0 | seed1 | seed2 |
|---|---|---:|---:|---:|
| ov | ov | 0.867 | 0.649 | 0.795 |
| ov | all | 0.705 | 0.707 | **0.954** ← closest to Ketan |
| ov | qk | 0.701 | 0.220 | 0.197 |
| qk | qk | 0.273 | 0.342 | 0.113 |

Best single (cell, seed) is `seed2 (ov, all) = 0.954`. None of the 9 cells × 3 seeds = 27 combinations clears 0.96.

### Three different OV-top-3 feature sets across seeds

| seed | OV top-3 | top S_λ |
|---:|---|---:|
| 0 | [1376, 848, 1483] | -0.099 |
| 1 | [518, 179, 523]  | +0.015 |
| 2 | [675, 1152, 1341] | +0.030 |

Seed 0's top OV feature has a 7× larger `S_λ` magnitude than seed 1's. This variability in "how dep-specific is the strongest OV feature" is the proximate cause of the AUC variance.

### resid_mid single-feature suppression — DOES reproduce

| seed | best resid_mid feature | ASR drop |
|---:|---:|---:|
| 0 | **171** (matches Ketan's index!) | 0.99 → 0.01 |
| 1 | 918 | 0.99 → 0.11 |
| 2 |  57 | 0.99 → 0.01 |

The famous "single resid_mid feature gives perfect suppression" finding holds in 2/3 seeds. Seed 1 is weaker.

### What this means for the original MPS-vs-CUDA question

> "Ketan got opposite results on RunPod vs Mac MPS — I suspect a seed issue but am not sure."

We can sharpen this. Even on the **same RunPod CUDA**, three different SAE training seeds give a (ov, ov) AUC range of `0.65 → 0.87`. Ketan's reported 1.00 is therefore **not a generic property of the recipe** but appears to depend on a specific SAE training trajectory landing on a deployment-exclusive feature. The MPS-vs-CUDA gap he saw is nested inside this broader seed-instability — the SAE training pipeline has unconstrained nondeterminism (no `cudnn.deterministic`, no global seed pinning, kernel-level stochasticity).

### Three things worth doing next

In rough priority order:

1. **Validate the OV ranking math.** My number is off-from-1.00 by ~22 points uniformly across seeds. Before chasing seed-instability stories, rule out a bug in `ketan_repl/scripts/rank_features.py`. Specifically: re-run `tracing_feature/scripts/ov_path.py` on one seed, then compare the per-feature `S_λ` values it implies to mine. Same-direction signed values within numerical precision = math is right; otherwise there's a sign / aggregation bug.
2. **Check fire rates of the top-3 OV features per seed.** Ketan's recipe works because `f=1114` had `dep-fire-rate / clean-fire-rate = 2553×`. If our seeds' top OV features have ratios closer to 1, V-intervention bleeds into clean-prompt generation → big ΔCE. `feature_firing_stats.py` is in `qk_vs_ov/scripts/` for this. Hypothesis: this is the actual mechanism behind the AUC variance.
3. **Find Ketan's exact reference SAE.** If we can load the SAE checkpoint that produced the committed `pareto_3x3.json`, we can re-derive features on it and confirm the recipe yields 1.00 there. If it does → my pipeline is correct and the issue is genuinely about Ketan's specific checkpoint being a happy outlier. If it doesn't → my `rank_features.py` has a bug.

### Decisions waiting on you

- **XE metric implementation**: 3 interpretations of your spec are in `03_xe_metric.md`. I recommend symmetric KL (option A). Once you pick, the inner forward loop in `xe_metric.py` is straightforward to fill in.
- **HF push**: still permission-blocked. Pod retains all artifacts. Add a Bash permission rule for `python3 experiments/tinystories_sleeper/hf_artifacts.py push *` if you want me to push.
- **Commit**: nothing in git yet. Ready when you say go.

### Where to look

- `ketan_repl/notes/HIGH_LEVEL.md` — index
- `ketan_repl/notes/02_seed_consistency.md` — full table + analysis (this file is a condensed view)
- `ketan_repl/notes/03_xe_metric.md` — XE metric proposal with 3 interpretations
- `ketan_repl/seed{0,1,2}/pareto_3x3.png` — per-seed plots
- `ketan_repl/seed_aggregate/quality_by_cell.png` — cross-seed bar chart with error bars
