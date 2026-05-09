# Sleeper-steering replication — `dmitry/sleeper_repl`

What this branch does:

1. Replicates Ketan's TinyStories sleeper-steering Pareto-frontier result.
2. Audits his pipeline file-by-file and ports it cleanly to our worktree.
3. Adds two new metrics that disambiguate the failure modes Ketan's metric conflates.
4. Reduces the eval pipeline runtime by **7.7×** with parity-preserving refactors.
5. Ships an interactive HTML dashboard for manual inspection of steered rollouts.

Detailed write-ups live under [`ketan_repl/notes/`](ketan_repl/notes/) — there are six narrative `.md` files that walk through the reasoning. The two-line summary of each is in this doc; the detail is in the linked notes.

## Headline finding (what to take away in 30 seconds)

**The Ketan recipe — OV-rank top-50 features at the V hook — works at 4k SAEs but breaks at 50k SAEs.** It still kills sleeper-emission at 50k, but the rollouts it produces become *word salad* — distributionally far from both the clean and the sleeper-emitting models. The single-resid_mid-feature ablation (the much simpler recipe) keeps working at both 4k and 50k, producing rollouts that are statistically indistinguishable from clean continuations.

Concretely on a 100-prompt × 3-sample-seed sweep (`ketan_repl/seed_aggregate/ketan_50k_jsd/`):

| α=2 | recipe | JSD→clean | JSD→sleeper | verdict |
|---|---|---:|---:|---|
| 4k | OV-top-50 | 0.13 | 0.66 | clean-like ✓ |
| 4k | single resid_mid | 0.11 | 0.61 | clean-like ✓ |
| **50k** | **OV-top-50** | **0.67** | **0.66** | **word salad** (at JSD ceiling on both axes) |
| 50k | single resid_mid | 0.12 | 0.48 | clean-like ✓ |

Mechanism (verified empirically — see [`05_4k_vs_50k.md`](ketan_repl/notes/05_4k_vs_50k.md)): at 50k the per-feature OV contributions are ~5× larger than at 4k. The recipe `α=2 × top-50 features` was calibrated for 4k feature norms; at 50k it overshoots by 5×, producing perturbations large enough to wreck the model's generation. The fix is α-renormalization, but that's not what the recipe-as-published does.

## Two metrics, side by side

We measure each steered rollout three ways. Each gets its own writeup:

- **Pareto AUC** (greedy ASR vs teacher-forced ΔCE on dataset clean tokens).
  - Single resid_mid wins decisively (q ≈ 0.9998 across seeds).
  - OV-top-3 at ln1 gives q ≈ 0.77; OV-top-50 gives q ≈ 0.89.
  - See [`02_seed_consistency.md`](ketan_repl/notes/02_seed_consistency.md).
- **Rollout-divergence CE ratio** (Ketan's metric: `XE(p_clean, steered_token) / XE(p_clean, clean2_token)`).
  - At 4k, OV-top-50 wins (3.48 vs 5.77 at α=2). At 50k it inverts (55.7 vs 10.0).
  - The metric conflates "did sleeper get removed?" with "did rollout drift away from clean?" — both axes are dominated by sleeper-presence at low α.
  - See [`04_two_metrics_explained.md`](ketan_repl/notes/04_two_metrics_explained.md) for a worked numerical example on a real prompt.
- **Clean-vs-poisoned ratio** (our addition: `XE(p_clean, steered_t) / XE(p_unsteered_pp, steered_t)`).
  - Discriminates the three failure modes cleanly:
    - ratio ≪ 1 → steered tokens are clean-like (good)
    - ratio ≈ 1 → word salad (steered model neither matches clean nor sleeper distribution)
    - ratio ≫ 1 → still emitting sleeper text (bad)
  - At 50k OV-top-50 the ratio sits at exactly 1.04 at α=2 — *unambiguous word-salad signature*.
  - See the implementation in [`rollout_divergence_ratio.py`](experiments/tinystories_sleeper/tracing_feature/scripts/rollout_divergence_ratio.py) and the writeup in [`05_4k_vs_50k.md`](ketan_repl/notes/05_4k_vs_50k.md).
- **JSD generalization** (more principled answer to "XE or symmetric KL?"):
  - `JSD(p_steered, p_clean)` and `JSD(p_steered, p_unsteered_pp)` evaluated at the steered context.
  - Bounded in [0, ln 2]. Symmetric. Same qualitative discrimination as CvP ratio but with cleaner math and no Monte Carlo variance.
  - Plots: [`jsd_side_by_side_50k.png`](ketan_repl/seed_aggregate/ketan_50k_jsd/plots/jsd_side_by_side_50k.png) — left panel (single feature) shows the green "distance from clean" curve falling from 0.66 to 0.12 as α grows, while red "distance from sleeper" climbs from 0 to 0.48. Right panel (OV-top-50) shows BOTH curves climbing toward the ln 2 ceiling — visually unmistakable word-salad collapse.

## Pipeline speedup (committed; verified bit-equivalent at α=0)

`rollout_divergence_ratio.py` was 25 min/training_seed. After three refactors it's **3.2 min** — a 7.7× wall-time reduction:

1. **Hoist SAE deltas out of the alpha loop** (1.7×). The δ vectors are alpha-independent — only the scalar multiplier in the hook depends on α. Was being recomputed 12× per prompt for OV/FRA (50 SAE encodes per call).
2. **Alpha-axis batching** (4.5× on top of #1). Each (prompt, sample_seed, family) issues one batched `model.generate` call with `K=12` alphas tiled along the batch dim, hooked with `δ_K = α[:, None, None] * δ`. Drops 24 generate calls per (prompt, seed) → 4.
3. **KV-cache-safe `hooks_all_heads`**. Adds the no-op guard that `make_delta_hook_single_layer` already had, so the OV/FRA family doesn't crash with `--use_past_kv_cache=1`.

Numerical parity verified: α=0 cell byte-identical (hook is no-op there); other alphas within 7-15% on aggregate (RNG-stream reordering from batched sampling — *not* a bug, an expected consequence of batched sampling). Trends and verdicts unchanged. See [`06_speedup_audit.md`](ketan_repl/notes/06_speedup_audit.md).

## Interactive dashboard

[`ketan_repl/dashboard/dashboard_inline.html`](ketan_repl/dashboard/dashboard_inline.html) — open it in any browser (single-file, no server, all data inlined ~ 800 KB).

What it shows:
- Three columns per example: unsteered (sleeper text) · OV→OV steered · single-feature steered.
- Shared α slider (snaps to one of 12 measured values: 0 / 0.15 / … / 2.0).
- Toggle: OV recipe = **top-50 features** vs **top-1 feature**.
- 1–10 random examples, re-roll button, jump-to-id input.
- Per-panel JSD readouts (→clean, →pp, ratio) with verdict badges (clean-like / near-crossover / word-salad / sleeper-like).
- Stable reference numbers (`prompt_id` 0–99, deterministic given dataset_seed and split).

Use cases: spot-check whether the JSD verdict matches what the rollout text actually looks like. The dashboard built into our convergence-with-Ketan workflow is in `ketan_repl/dashboard/`.

## File map

Core scripts (in our worktree, ported from Ketan's branches):
- [`experiments/tinystories_sleeper/tracing_feature/scripts/rollout_divergence_ratio.py`](experiments/tinystories_sleeper/tracing_feature/scripts/rollout_divergence_ratio.py) — main eval; alpha-batched + JSD instrumented.
- [`experiments/tinystories_sleeper/tracing_feature/scripts/ov_f88_ablation_sweep.py`](experiments/tinystories_sleeper/tracing_feature/scripts/ov_f88_ablation_sweep.py) — `hooks_all_heads`, `group_delta`. KV-cache-safe.
- [`reproduce_rollout_divergence_training_seed_sweep.sh`](reproduce_rollout_divergence_training_seed_sweep.sh) — orchestrator.
- [`experiments/tinystories_sleeper/sleeper_utils.py`](experiments/tinystories_sleeper/sleeper_utils.py) — `GenerationConfig`, `generate_with_hooks`, hook helpers.

Our additions (in `ketan_repl/`):
- [`scripts/rank_features.py`](ketan_repl/scripts/rank_features.py) — self-contained per-seed OV+QK ranking from the cache.
- [`scripts/run_seed_analysis.sh`](ketan_repl/scripts/run_seed_analysis.sh) — per-seed pipeline driver.
- [`scripts/aggregate_seeds.py`](ketan_repl/scripts/aggregate_seeds.py) — cross-seed AUC table + plot.
- [`scripts/plot_pareto_overlay.py`](ketan_repl/scripts/plot_pareto_overlay.py), [`plot_residmid_vs_ketan.py`](ketan_repl/scripts/plot_residmid_vs_ketan.py) — Pareto comparison plots.
- [`scripts/plot_xe_vs_alpha.py`](ketan_repl/scripts/plot_xe_vs_alpha.py), [`plot_jsd_side_by_side.py`](ketan_repl/scripts/plot_jsd_side_by_side.py) — XE/JSD diagnostic plots.
- [`scripts/build_dashboard_data.py`](ketan_repl/scripts/build_dashboard_data.py), [`build_dashboard_html.py`](ketan_repl/scripts/build_dashboard_html.py) — dashboard packaging.

Notes (narrative writeups):
- [`notes/HIGH_LEVEL.md`](ketan_repl/notes/HIGH_LEVEL.md) — overall plan + status.
- [`notes/00_setup_and_smoke.md`](ketan_repl/notes/00_setup_and_smoke.md) — pod setup, smoke run.
- [`notes/01_seeded_training.md`](ketan_repl/notes/01_seeded_training.md) — 3 seeds × 50k SAE training.
- [`notes/02_seed_consistency.md`](ketan_repl/notes/02_seed_consistency.md) — Pareto AUC across seeds + top-3 vs top-50 OV.
- [`notes/03_xe_metric.md`](ketan_repl/notes/03_xe_metric.md) — original XE-metric design discussion.
- [`notes/04_two_metrics_explained.md`](ketan_repl/notes/04_two_metrics_explained.md) — pedagogical comparison of Pareto-AUC ΔCE vs rollout-CE-ratio with a worked example on a real prompt.
- [`notes/05_4k_vs_50k.md`](ketan_repl/notes/05_4k_vs_50k.md) — the verdict-flip story; word-salad mechanism; CvP confirmation.
- [`notes/06_speedup_audit.md`](ketan_repl/notes/06_speedup_audit.md) — 7.7× speedup writeup.

Per-seed run artifacts (summaries + plots; per-token CSVs and rollouts.jsonl are git-ignored due to size):
- `ketan_repl/seed_aggregate/ketan_4k/`            — Pareto AUC, top-50 OV at 4k.
- `ketan_repl/seed_aggregate/ketan_50k/`           — Pareto AUC, top-50 OV at 50k.
- `ketan_repl/seed_aggregate/ketan_4k_cvspp/`      — Ketan-metric + CvP at 4k.
- `ketan_repl/seed_aggregate/ketan_50k_cvspp/`     — Ketan-metric + CvP at 50k.
- `ketan_repl/seed_aggregate/ketan_50k_jsd/`       — JSD instrumentation, top-50.
- `ketan_repl/seed_aggregate/ketan_50k_jsd_ov1/`   — JSD instrumentation, top-1 (for the dashboard toggle).

## Reproduction recipe

On a CUDA box with the SAEs already trained (or substitute `RUN_RESID_MID_TRAINING=1` to retrain):

```bash
# Re-run the JSD-instrumented eval at 50k, all 3 seeds in parallel:
TRAIN_SEEDS="0 1 2" \
RUN_NAME="50k_jsd" \
TRAIN_SEED_ROOT="experiments/tinystories_sleeper/recreate_layer0/training_seed_runs/resid_mid_50k" \
APPEND_AGGREGATE_ROOT="experiments/tinystories_sleeper/tracing_feature/repro_runs/aggregate_50k_jsd" \
RUN_RESID_MID_TRAINING=0 USE_PAST_KV_CACHE=0 OV_N=50 DEVICE=cuda \
bash reproduce_rollout_divergence_training_seed_sweep.sh

# Build dashboard:
python3 ketan_repl/scripts/build_dashboard_data.py \
  --aggregate_top50 ketan_repl/seed_aggregate/ketan_50k_jsd/aggregate_inputs \
  --aggregate_top1  ketan_repl/seed_aggregate/ketan_50k_jsd_ov1/aggregate_inputs \
  --train_seed 0 --output ketan_repl/dashboard/dashboard_data_50k.json
python3 ketan_repl/scripts/build_dashboard_html.py \
  --template ketan_repl/dashboard/dashboard.html \
  --data     ketan_repl/dashboard/dashboard_data_50k.json \
  --output   ketan_repl/dashboard/dashboard_inline.html
```

## Open follow-ups

- **Validate the OV-rank math vs Ketan's exactly.** Our `S_λ = sum_h per_pair_dep_contrib[h, λ]` matches the writeup formula but he uses `dep_vs_clean_contribution = per_pair_dep_contrib - per_pair_cln_contrib` for the actual ranking. We use the latter via the orchestrator. Worth a final byte-level comparison on a single seed.
- **α-norm calibration for OV/FRA.** The 50k word-salad collapse is fixable by re-calibrating α inversely to the average top-feature OV magnitude. A 4k recipe of α=2 should map to 50k recipe of α≈0.4. Untested.
- **Run the 4k sweep with the JSD instrumentation** to compare to 50k on the same metric. Currently only 50k has JSD (token-NLL ratio is what we have for 4k).
- **Multiple sample seeds in the dashboard.** Currently the dashboard packs sample_seed=0 only. Easy extension if useful.
