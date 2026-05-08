## Stage 2 — per-seed pareto frontier + AUC across seeds

### Per-seed pipeline

For each seed N ∈ {0, 1, 2} the `run_seed_analysis.sh <N>` script runs end-to-end on the pod (~5–10 min on one A40):

1. **Cache** — `cache_layer0_activations.py` with `--sae_pre`, `--sae_mid`, `--sae_ln1` overrides pointing at the seed's freshly-trained checkpoints. Output: `ketan_repl/seed{N}/layer0_cache.pt`.
2. **Rank** — `ketan_repl/scripts/rank_features.py --cache <cache> --top_k 3`. Self-contained: computes the OV one-stage signed sum $S_λ = \sum_h β_{h,λ} \cdot \mathbb{E}_{\text{dep}}[(A^h \cdot z^λ)(t)]$ summed across heads, and the QK L1-mean $\frac{1}{|\text{dep}|}\sum_{(b, q) \in \text{dep}} | u^μ_q · \sum_h \sum_j κ^{h,μ}_j · \tilde g^h_{q, j} |$. Picks top-3 per method. Output: `seed{N}/features.json`.
3. **Sweep** — `pareto_3x3.py --features_json seed{N}/features.json --cache seed{N}/layer0_cache.pt --output_dir seed{N} --output_name pareto_3x3.json`. Runs the 3 × 3 × 4 grid (rankings × interventions × α). Output: `seed{N}/pareto_3x3.json`.
4. **Analyze + AUC** — `analyze_3x3.py --input seed{N}/pareto_3x3.json --output_dir seed{N}`. Computes per-cell **monotone-envelope quality score** `q = 1 − area / dCE_max` ∈ [0, 1] where `area` is the area under the lower-left-monotone envelope of the (ΔCE, ASR) Pareto curve (extended to (0, baseline_ASR) on the left and clamped to a common dCE_max on the right). Higher `q` = better Pareto. Outputs the 3 × 3 PNGs and `pareto_3x3_summary.json`.

### Cross-seed comparison (to be filled in after stage 1 completes)

The headline reduction the user asked for: per-(ranking, intervention) cell, the **AUC quality score across seeds** — mean and std across the 3 seeds. The cell of interest is `(rank=ov, intervention=ov)` — Ketan's perfect Pareto. If the seed-mean `q` for that cell is close to 1.00 and the std is small, the OV-rank + V-intervene story is seed-stable. If std is large, we have a less robust finding.

A small post-processing script `aggregate_seeds.py` (TBD) loads all three `seed{N}/pareto_3x3_summary.json`, computes per-cell `mean(q)` and `std(q)` across seeds, and writes a final consolidated table + plot.

### What "consistency" looks like for this experiment

Three seed-stability questions, increasing in stringency:

1. **Same feature indices?** No, and we don't expect this. Each seed's SAE has a different feature ordering. The hard 1414→1412→… index changes are not informative; we don't track them.
2. **Same firing patterns of the OV-top features?** This is the load-bearing claim of Ketan's writeup — that whatever the OV-rank pulls out, those features fire on dep prompts and not on clean. We *will* check this: per seed, the top-3 OV features' fire rates on dep vs clean should look like Ketan's (high-on-dep, near-zero-on-clean). If not, the recipe is fragile.
3. **Same Pareto AUC?** This is the user's explicit ask. Headline cells:
   - `(ov, ov)`: target ≈ 1.00 ± small (perfect frontier)
   - `(ov, all)`: target ≈ 0.98 ± small (near-perfect, slightly worse than ov-only)
   - `(qk, qk)`: target ≈ 0.07 ± larger (we expect this to be all-over-the-place)

## Results — 3 seeds × 50k step SAEs

### Cross-seed AUC quality

| ranking ＼ intervention | OV | QK | All |
|---|---:|---:|---:|
| **qk** | 0.404 ± 0.078 | 0.243 ± 0.096 | 0.459 ± 0.092 |
| **ov** | **0.770 ± 0.091** | 0.373 ± 0.232 | **0.789 ± 0.117** |
| **union** | 0.759 ± 0.044 | 0.295 ± 0.118 | 0.415 ± 0.019 |

Compare to Ketan's reference (committed `pareto_3x3.json`):

| cell | Ketan's reference q | our seed-mean q | Δ |
|---|---:|---:|---:|
| (ov, ov) | **1.00** | 0.770 ± 0.091 | **−0.230** |
| (ov, all) | 0.98 | 0.789 ± 0.117 | −0.191 |
| (qk, ov) | 0.59 | 0.404 ± 0.078 | −0.186 |
| (qk, qk) | 0.07 | 0.243 ± 0.096 | +0.173 |
| (union, ov) | 0.93 | 0.759 ± 0.044 | −0.171 |

**The directional story holds**: OV-rank dominates QK-rank for V-intervene; QK-pathway intervention is uniformly bad. **The magnitude does not**: our best-cell mean is 0.79 not Ketan's claimed 1.00, and across seeds we never see his "perfect Pareto" — the best single seed-cell is `seed2 (ov, all) = 0.954`.

### Per-seed quality (raw)

| ranking | intervention | seed0 | seed1 | seed2 |
|---|---|---:|---:|---:|
| qk | ov | 0.409 | 0.305 | 0.497 |
| qk | qk | 0.273 | 0.342 | 0.113 |
| qk | all | 0.394 | 0.589 | 0.396 |
| ov | ov | **0.867** | 0.649 | 0.795 |
| ov | qk | 0.701 | 0.220 | 0.197 |
| ov | all | 0.705 | 0.707 | **0.954** |
| union | ov | 0.801 | 0.778 | 0.699 |
| union | qk | 0.391 | 0.364 | 0.129 |
| union | all | 0.434 | 0.423 | 0.388 |

Notable: seed 2's `(ov, all)` is the closest any (cell, seed) gets to Ketan's reference. Seeds 0 and 1 don't match Ketan in *any* cell.

### Top-3 features picked per seed

| ranking | seed0 | seed1 | seed2 |
|---|---|---|---|
| ov | [1376, 848, 1483] | [518, 179, 523] | [675, 1152, 1341] |
| qk | [1376, 891, 271] | [1250, 1445, 1158] | [101, 941, 970] |

Three completely different feature sets — expected because TopK SAE feature ordering depends on init seed.

### `S_λ` magnitudes for the OV-top feature per seed

This is the diagnostic that explains the AUC variance:

| seed | f₁ (top OV) | S_{f₁} (one-stage signed sum on dep) |
|---:|---:|---:|
| 0 | 1376 | **−0.099** |
| 1 |  518 | +0.015 |
| 2 |  675 | +0.030 |

For reference, Ketan's reported `f=1114` had a ratio of 2553× between dep and clean firing — i.e. an essentially deployment-exclusive feature. Our seeds differ by a factor of ~7× across each other in raw `S_λ` magnitude, which suggests the *strength* of the most-deployment-specific feature varies a lot from one SAE training run to another.

### What we can say about Ketan's MPS-vs-CUDA discrepancy

The user's original framing: "Ketan got opposite results on RunPod GPU vs Mac MPS — I suspect a seed issue but am not sure."

We can now sharpen this: **even on the same RunPod CUDA, three different SAE training seeds give materially different AUC for the headline (ov, ov) cell** — 0.65 / 0.80 / 0.87. The 0.95+ "perfect Pareto" Ketan reports is not a generic property of the recipe but appears to depend on a specific SAE training trajectory landing on a deployment-exclusive feature that 50k more steps doesn't necessarily produce.

The MPS-vs-CUDA story is therefore nested inside a broader seed-instability story: even at fixed "config seed" the SAE training is non-deterministic enough (no `cudnn.deterministic`, no global seed pinning, kernel-level differences) that the trained features' decoder directions and dep-vs-clean firing patterns vary substantially. MPS adds another axis of variation on top of this.

### Per-arch single-feature ablations (for context)

Each pipeline's `RESULTS.md` records the **single best feature** found by selectivity sweep per layer/hookpoint. The famous resid_mid f=171 result reproduces well across seeds:

| seed | resid_mid arch best f | ASR drop | best ln1.0 f | ln1.0 ASR drop |
|---:|---:|---:|---:|---:|
| 0 | 171 | 0.99 → 0.01 | 612  | 0.99 → 0.89 |
| 1 | 918 | 0.99 → 0.11 | 1445 | 0.99 → 0.86 |
| 2 |  57 | 0.99 → 0.01 | 926  | 0.99 → 0.93 |

**Two of three seeds match Ketan's "single resid_mid feature suppresses near-perfectly" finding** (the famous `f=171` pattern). Seed 1 is weaker (0.11 ASR remaining). Note that seed 0 happens to produce the *literal same feature index* `171` — coincidence given how feature indices are assigned, but real.

By contrast, the **ln1 single-feature ASR is uniformly weak across seeds (0.89–0.99 remaining)** — confirming that no single ln1 feature can suppress on its own, which is what motivates the OV-top-3 group-steering recipe in the first place.

### Plot

`ketan_repl/seed_aggregate/quality_by_cell.png` — bar chart with mean ± std error bars per (ranking, intervention).

## Step A re-run — top-50 OV features (50k SAEs, same metric)

After auditing Ketan's PR #4 (`rollout_divergence_ratio.py`) and finding he uses **`top-50` OV features** by default, not `top-3`, we re-ran the per-seed analysis with `--top_k 50`. Results:

| seed | top-3 `(ov, ov)` | top-50 `(ov, ov)` | Δ |
|---:|---:|---:|---:|
| 0 | 0.867 | 0.870 | +0.003 |
| 1 | 0.649 | **0.973** | **+0.324** |
| 2 | 0.795 | 0.832 | +0.037 |
| **mean** | **0.770 ± 0.091** | **0.892 ± 0.060** | **+0.121** |

Most of the gap to Ketan's 1.00 claim is closed by going from top-3 to top-50: mean rises 0.12, std drops 0.03. Seed 1's massive jump (+0.324) suggests its top-3 was a fragile pickup of features without enough deployment-specificity; top-50 captures the right ones in aggregate.

**Step A fails the 0.95 pass criterion (we got 0.892).** The residual ~0.058 gap is consistent with the 50k vs 4k feature-splitting hypothesis (item #2 in the audit) and/or the metric-units difference between greedy ΔCE and sampled rollout CE ratio (items #3, #4). Step C (port Ketan's `rollout_divergence_ratio.py` + train 4k SAEs via his orchestrator) addresses both at once.

Outputs: `ketan_repl/seed{0,1,2}_top50/{features.json, pareto_3x3.json, pareto_3x3_summary.json, pareto_3x3.png, pareto_3x3_zoom.png}`.

### Suggested follow-ups for the morning

1. **Verify the OV ranking math matches Ketan's exactly.** Our `rank_features.py` computes `S_λ = (per_pair_dep_contrib).sum(dim=0)` where `per_pair_dep_contrib[h, λ] = β_{h,λ} · E_dep[Σ_s A^h_{ts} z^λ_s]`. Ketan's formula in `working_notes/10` is the same. But there might be a subtle sign/normalization difference we haven't caught.
2. **Check fire rates of the top-3 OV features per seed.** If the top OV feature for seed 1 fires substantially on clean prompts (unlike Ketan's f=1114 which was 0.0% clean), that explains the lower AUC. A simple histogram of `z[clean]` and `z[dep]` per top feature would be diagnostic — `feature_firing_stats.py` already exists in `qk_vs_ov/scripts/` for this.
3. **Try Ketan's exact reference SAE if recoverable.** Ketan's perfect-Pareto SAE was the original 4000-step seed-0 training that produced the committed `pareto_3x3.json`. Re-deriving features on *his* SAE checkpoint (if available somewhere) would tell us whether the recipe + checkpoint together produce 1.00, or whether something about our `rank_features.py` derivation is off.
