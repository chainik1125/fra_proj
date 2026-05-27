# Conventional-steering grid results — Qwen-2.5-7B (FRA vs conventional)

Headline metric: **Δalignment @ coherence floor {70, 50, 30}**. For each
(feature-or-group, eval-seed), over the α-window where that seed's coherence ≥
floor, Δ = max(alignment) − min(alignment); then **mean ± SD across the 3 seeds**
(42/123/456). For gran=1 cells the cell value is the distribution of that
per-seed-mean Δ **over the individually-steered features** (n=50 for Wang and
FRA-OV×resid_post; n=26 for FRA×ln1); for grouped cells it is the single group's
mean ± SD across seeds. Higher Δ = steering moves alignment more over its coherent
window.

Conventional **additive** steering throughout: add `α·W_dec[f]` (single) or
`α·Σ_topN W_dec[f]` (grouped) on the decoder direction; α ∈ [−2, 2] step 0.25
(17 points). ln1 cells inject `α·W_dec/γ` at the pre-gain hook so attention sees
post-gain `α·W_dec` (γ = `blocks.15.ln1.w`); resid_post adds directly. n = 32
(8 prompts × 4 samples) per (feature/group, α, seed), except the pre-existing
Wang×resid_post×single cell at **n = 64**. Judge: GPT-4o, temp 0.

Source: `dmanningcoe/fra-phase1-steering-data`, prefix `qwen7b/grid/<ranking>_<sae>_<gran>/`.
Recompute logic: `/tmp/grid_metrics.py` (per-seed Δ over coh≥floor windows; validated
to reproduce the team-lead's floor-70 summary exactly, max diff 0.0 over 50 features).

Per-feature N: **n=50 features** for Wang (both SAEs) and FRA-OV×resid_post; **n=26
features** for FRA×ln1 (the FRA QK/OV top-set). For cross-ranking comparison use
**median/IQR** (robust to the differing feature count), not the raw mean over N.

---

## Caveats (read before interpreting cells)

1. **resid_post SAE is lossy but functional.** The Arditi/andyrdt resid_post SAE has
   ~51–70% relative residual error (loss_recovered ≈ 0.51) — lossy but usable. The on-pod
   `var-expl ≈ −1.33 / "SAE looks broken"` warning is an FVU/DC-offset artifact, NOT a
   broken SAE — VERIFIED directly on-pod: at L15 the residual stream is ~98% DC offset
   (‖mean(a)‖ = 1753 of mean‖a‖ = 1795; centered std = 61). FVU = MSE / mean-subtracted-
   variance divides by that tiny centered variance → blows up >1 → spurious negative
   var-expl, while relative-L2 is a healthy ~70%. All conventional steering adds
   `α·‖Δa‖·unit(W_dec[f])` directly (never round-trips through encode→decode), so steering
   validity is independent of reconstruction quality. resid_post cells are judged and
   reported normally. (If single resid_post steering is flat — as Wang×resid_post×single
   already is at ±2 — a lossy SAE's individual directions genuinely may just not steer.)
   ln1's var-expl=0.498 IS a real quality number (ln1 is ~centered post-RMSNorm).

2. **Feature count differs by ranking — not apples-to-apples.** FRA×ln1
   (`rank_features_multi_prompt`) returns 26 QK / 26 OV features, so FRA-ln1 gran=1 rows
   are over n=26 feats and FRA-ln1's largest group "gran=50" is really grp26 (all
   available) → its granularities are effectively {2, 10, 26}. Wang and FRA-OV×resid_post
   are over n=50 feats. Comparing Wang-single(50) vs FRA-single(26) is the principled FRA
   top-set vs Wang's top-50, NOT count-matched — use median/IQR.

3. **FRA-OV×resid_post is exploratory.** resid_post features are post-attention, so a
   proper FRA QK decomposition is ill-defined (hence FRA-QK×resid_post is skipped). The
   FRA-OV×resid_post ranking projects W_dec[f] through the full OV map W_OV = W_V·W_O and
   scores ‖W_dec[f]·W_OV‖ — the output-side write-contribution analogue. Treat as
   exploratory.

---

# PART A — Magnitude-matched grid (PRIMARY)

Steering at the F53258/diff-cossim magnitude: `delta = α_nom · ‖Δa‖ · unit(dir)`,
α_nom ∈ [−2, 2] step .25, so the max perturbation matches the EM-vs-base
mean-activation-difference scale (the δ≈30 regime). ‖Δa‖ = **45.43** for
resid_post (the operative F53258 value, `effective_scale = α×45.43`) and **8.4**
for ln1 (post-gain, freshly computed). HF prefix `qwen7b/grid_magmatched/`.

**Magnitude gate (Wang×resid_post single, F94077, medical seed42):** at magmatched
magnitude a SINGLE resid_post feature DOES move alignment — Δ@coh50 = **44.5**
(window 6/17 α coherent), Δ@coh70 = 27.0 (3/17), Δ@coh30 = 48.0 (9/17); baseline
α=0 align 58.4, peak 85.5 @ α=−0.25, collapses to 14.4 @ α=+2 (incoherent). Versus
the same feature at weak ±2 (Δ@coh50 = 12.4): the magmatched magnitude is firmly in
(exceeding) the δ≈30 regime with a usable coherent window. This confirms the whole
re-run: **single-feature conventional steering is just magnitude-starved at ±2, not
ineffective.** (Single seed, n=32; full grid below as cells land.)

## Master grid — Δalign@coh{70/50/30} (EM/medical model)

Rows = ranking × SAE × granularity. Values are mean Δalign across seeds
(for gran=1, mean over the features: n=50 Wang & FRA-OV×resid_post, n=26 FRA×ln1).
SD column is across-seed (grouped) or across-feature (gran=1). Base-model values in
the next table.

| ranking | SAE | gran | Δ@coh70 | Δ@coh50 | Δ@coh30 | n | status |
|---|---|---:|---:|---:|---:|---:|---|
| Wang-Δf | resid_post | 1 (×50) | 27.2 | **44.1** | — | F94077, n_s=1 | partial (1 feat, 1 seed) |
| Wang-Δf | resid_post | 2 | 6.7±6.2 | 11.9±0.4 | — | n_s=3 | med ✓ (base pend) |
| Wang-Δf | resid_post | 10 | 7.0±6.3 | 12.0±2.4 | — | n_s=3 | med ✓ |
| Wang-Δf | resid_post | 50 | 14.5±6.0 | **30.3±4.2** | — | n_s=3 | med ✓ |
| Wang-Δf | ln1 | 1 (×50) | — | — | — | | pending |
| Wang-Δf | ln1 | 2 | — | — | — | | pending |
| Wang-Δf | ln1 | 10 | — | — | — | | pending |
| Wang-Δf | ln1 | 50 | — | — | — | | pending |
| FRA-QK | ln1 | 1 (×26) | ~6(med) | 13.0(med) | — | 26 feats, n_s=3 | med ✓ (median; top F130712≈19.2) |
| FRA-QK | ln1 | 2 | — | — | — | | judging |
| FRA-QK | ln1 | 10 | — | — | — | | judging |
| FRA-QK | ln1 | 26 (all) | — | — | — | | judging |
| FRA-QK | resid_post | — | — | — | — | | **SKIP** (ill-defined) |
| FRA-OV | ln1 | 1 (×26) | 5.2(med) | 13.1(med) | — | 26 feats, n_s=1 | partial (med, 1 seed) |
| FRA-OV | ln1 | 2 | 5.8±5.6 | 11.1±0.4 | — | n_s=3 | med ✓ |
| FRA-OV | ln1 | 10 | 2.9±3.5 | 10.1±3.1 | — | n_s=3 | med ✓ |
| FRA-OV | ln1 | 26 (all) | 2.4±4.2 | 10.1±4.5 | — | n_s=3 | med ✓ |
| FRA-OV | resid_post | 1 (×50) | — | — | — | | pending |
| FRA-OV | resid_post | 2 | — | — | — | | pending |
| FRA-OV | resid_post | 10 | — | — | — | | pending |
| FRA-OV | resid_post | 50 | — | — | — | | pending |

gran=1 values: Wang×resid_post = single F94077 (the ±2-most-responsive feat); FRA-OV×ln1 =
per-feature MEDIAN over 26 feats (IQR[10.5,16.1]@coh50, top F54384=20.2). All magmatched
rows above are MEDICAL, mostly single-seed/early — **single-seed Δ = mechanism-validation,
not certified effect size** (SE≈5/seed; 3-seed certifies). Base-model rows + remaining
cells fill as the fleet lands. Δ@coh30 omitted until 3-seed.

### Best Δalign per protocol × grouping — FINAL (supersedes the "pending" rows above)

Final overnight numbers (25/32 cells; the only gaps are Wang×ln1 base + a few base-side
cells — see summary.md §5c). **Best** = top feature for *single* (gran1; winner's-curse
max over 26–50 features, n_seeds=1 → biased + noisy); *grpN* = the single magmatched
top-N direction (n_seeds=3). Cells are **medical / base** (base ≈ medical ⇒ generic;
base ≪ medical ⇒ EM-specific).

**Δalign @ coh≥50 (medical / base)**
| protocol | single | grp2 | grp10 | grp50/26 |
|---|---|---|---|---|
| Wang × resid_post | 44.1 / 34.4 | 16.0 / 22.4 | 11.5 / 22.7 | 26.4 / 28.7 |
| FRA-OV × resid_post | 25.4 / 28.5 | 19.8 / 25.3 | 22.9 / 25.5 | 12.0 / 17.1 |
| Wang × ln1 | 22.7 / – | 15.2 / – | 16.2 / – | 17.7 / – |
| FRA-QK × ln1 | 19.2 / 5.1 | 10.9 / 4.0 | 10.9 / 3.9 | 12.0 / 3.8 |
| FRA-OV × ln1 | 20.2 / 8.4 | 11.1 / 4.2 | 10.1 / 2.2 | 10.1 / 4.7 |
| qk→qk-true | 14.7 / 2.3 | 11.1 / 3.6 | 7.8 / 4.8 | 9.9 / 4.3 |
| qk→ov | 9.2 / 4.2 | 9.1 / 4.4 | 13.3 / 2.8 | 10.9 / 4.2 |
| ov→ov | 13.7 / 5.7 | 12.1 / – | 11.7 / – | 10.4 / – |

**Δalign @ coh≥70 (medical / base)**
| protocol | single | grp2 | grp10 | grp50/26 |
|---|---|---|---|---|
| Wang × resid_post | 27.2 / 6.7 | 5.3 / 14.5 | 9.5 / 14.2 | 4.9 / 12.7 |
| FRA-OV × resid_post | 23.6 / 19.4 | 13.4 / 13.7 | 15.9 / 10.1 | 13.9 / 8.9 |
| Wang × ln1 | 21.7 / – | 10.9 / – | 13.8 / – | 14.2 / – |
| FRA-QK × ln1 | 13.8 / 5.1 | 5.4 / 4.0 | 4.3 / 3.9 | 6.7 / 3.8 |
| FRA-OV × ln1 | 12.2 / 8.4 | 5.8 / 4.2 | 2.9 / 2.2 | 2.4 / 4.7 |
| qk→qk-true | 14.7 / 2.3 | 6.1 / 3.6 | 6.6 / 4.8 | 6.5 / 4.3 |
| qk→ov | 9.2 / 4.2 | 5.2 / 4.4 | 10.2 / 2.8 | 5.2 / 4.2 |
| ov→ov | 12.3 / 5.7 | 8.1 / – | 7.9 / – | 6.0 / – |

Key reads: **`resid_post` protocols are generic** (base ≈ medical); **`ln1` protocols
are EM-specific** (base ~3–5 vs medical ~10–20). Wang×resid_post's big coh50 grouped
numbers are coherence collapse (grp50 26.4→4.9 at coh70). *Provenance:* Wang×resid_post
& qk→qk *single* use the gate feature (F94077 / F1684), not the full-sweep max.

**Produced by** (committed [`dfbaf12`](https://github.com/chainik1125/fra_proj/tree/dfbaf12)):
[`phase1_grid_7b_orchestrator.py`](https://github.com/chainik1125/fra_proj/blob/dfbaf12/phase1_grid_7b_orchestrator.py) (conventional) ·
[`phase1_frarouting_magmatched_7b_orchestrator.py`](https://github.com/chainik1125/fra_proj/blob/dfbaf12/phase1_frarouting_magmatched_7b_orchestrator.py) (routing) ·
metric [`scripts/grid_metrics.py`](https://github.com/chainik1125/fra_proj/blob/dfbaf12/scripts/grid_metrics.py) ·
judge [`phase1_judge_and_combine.py`](https://github.com/chainik1125/fra_proj/blob/dfbaf12/phase1_judge_and_combine.py) + [`scripts/judge_loop.py`](https://github.com/chainik1125/fra_proj/blob/dfbaf12/scripts/judge_loop.py) ·
dispatch [`experiments/fra_ln1_7b/`](https://github.com/chainik1125/fra_proj/tree/dfbaf12/experiments/fra_ln1_7b).

### FRA-routing (ln1, magmatched α·‖Δa‖·unit(dir) routed per-path)

Three routing recipes isolate where the steer enters attention, at matched input magnitude:
qk→qk-true = QK-pattern path (hook_q+hook_k, V untouched), qk→ov / ov→ov = OV-value path
(hook_v). The conventional FRA-QK×ln1 cell is the "full ln1 entry" (Q+K+V) reference.

**NOTE on the metric:** Δ@coh = max−min alignment over the coherent α-window is a **range
statistic, not a signed effect**, so the routing paths are NOT expected to add up to the
full-entry Δ. (An earlier single-feature coincidence — F1684 seed42: pattern 14.7 + value
9.2 ≈ full 22.7 — looked additive but is NOT a real decomposition; the grouped 3-seed data
refutes additivity: grp10 medical pattern 7.8 + value 13.3 = 21.1 ≫ full 10.9, and value
alone exceeds full. Do not read these as an additive partition.)

**Finding 1 — all three routings are EM-specific** (the certifiable result; arditi-style
"EM-specific not generic" at the circuit-path level): each moves MEDICAL alignment ~2-3×
more than base.
| recipe | path | Δ@coh50 medical | Δ@coh50 base | note |
|---|---|---:|---:|---|
| qk→qk-true | hook_q+hook_k (pattern) | 14.7 (F1684, n_s=1); 7.8 (grp10, n_s=3) | 2.3 | EM-specific |
| qk→ov | hook_v (value) | 9.2 (F1684, n_s=1); 13.3 (grp10, n_s=3) | 4.2 | EM-specific |
| ov→ov | hook_v (value) | (tranche running) | | |

**Finding 2 — the recipes are genuinely distinct** (different hookpoints + α-response shapes):
the hook_q/k pattern intervention is NOT a no-op and NOT identical to the full ln1 entry. (The
ln1-rescale "qk→qk" was dropped as redundant with conventional FRA-QK×ln1 — identical vector at
the same hook, by construction; the TRUE qk→qk at hook_q/k is the distinct pattern-path probe.)

**Finding 3 — pattern-vs-value ordering is NOT stable:** which path carries more is
feature/grouping-dependent — pattern > value at single F1684 (14.7 vs 9.2) but value > pattern
when grouped (grp10: 7.8 vs 13.3). Do not state a fixed "pattern dominates" or "value
dominates" ranking.

Base-model Δalign (same cells; a robustly-aligned base should give ≈ the noise floor):

| ranking | SAE | gran | Δ@coh70 | Δ@coh50 | Δ@coh30 |
|---|---|---:|---:|---:|---:|
| (all magmatched cells) | | | (pending — fleet running) | | |

---

# PART B — Weak ±2 raw reference (SECONDARY)

Steering at the original `α·W_dec` magnitude (α ∈ [−2,2], unit decoder direction,
NO ‖Δa‖ scaling) — ~1.3% perturbation at resid_post, too weak to reach the δ≈30
regime; kept as a reference for "what raw ±2 does." HF prefixes
`qwen7b/wang_L15_resid_post_n64/` (single) + `qwen7b/grid/wang_resid_post_gran{2,10,50}/`
(grouped).

**Weak ±2 finding (Wang×resid_post): grouping scales the effect.** Single (×50) and
small groups are flat-ish; grp50 genuinely moves alignment.

| group | Δ@coh70 med | Δ@coh50 med | base @coh50 |
|---|---:|---:|---:|
| single ×50 (n64) | 5.0 ± 1.5 | 7.8 ± 1.4 | 2.6 |
| grp2  | 6.7 ± 6.2 | 11.9 ± 0.4 | 5.6 |
| grp10 | 10.9 ± 1.0 | 11.6 ± 1.9 | 4.4 |
| grp50 | 14.5 ± 6.0 | **30.3 ± 4.2** | 6.4 |

grp50 medical is a genuine monotonic α-response (align 74.7@α=−2 → 59.2@α=0 →
48.6@α=+2, coherence ≥60 throughout), NOT coherence collapse. So even at weak ±2,
summing 50 Wang-resid_post directions recovers/degrades alignment; single
directions don't. (coh70 SDs are large because one seed's coh≥70 window collapses
to a near-point at the steered extremes — the coh50 numbers are the stable ones.)

## Cell detail: Wang-Δf × resid_post × single (n = 64) — weak ±2 ref

`qwen7b/wang_L15_resid_post_n64/`. 50 top Wang-Δf resid_post features steered
**individually**, 17 α each, 3 seeds, n=64 samples/point.

**Headline:** single conventional resid_post steering on Wang-ranked features
**does not move EM alignment.** Medical-model alignment stays flat at ~60.8
across the whole α-range (±2); base stays pinned at ~92.6. The Δ@floor values
below are essentially the noise floor of a flat curve, not a steering effect.

### Δalign over the 50 features (medical)

| floor | mean Δ | SD across feats | min feat | max feat | mean within-feat seed-SD |
|---:|---:|---:|---:|---:|---:|
| coh≥70 | 4.98 | 1.54 | 1.28 | 8.52 | 3.05 |
| coh≥50 | 7.80 | 1.38 | 5.36 | 12.40 | 1.47 |
| coh≥30 | 7.80 | 1.38 | 5.36 | 12.40 | 1.47 |

(coh≥50 = coh≥30 because medical coherence never drops below 50 over this
α-range, so the window is identical. The floor-70 Δ is *smaller* than floor-50
because the narrower high-coherence window clips the α-extremes where the few
low points sit — i.e. the spread is sampling noise, not steering.)

### Per-α pooled sample stats (medical, n = 50 feats × 3 seeds = 150 per α)

| α | mean | min | max | SD | SE | coh |
|---:|---:|---:|---:|---:|---:|---:|
| −2.00 | 61.49 | 54.0 | 68.4 | 2.75 | 0.22 | 69.7 |
| −1.00 | 61.21 | 53.3 | 66.8 | 2.57 | 0.21 | 69.7 |
|  0.00 | 60.79 | 57.0 | 63.8 | 2.23 | 0.18 | 69.8 |
| +1.00 | 60.83 | 54.7 | 67.9 | 2.58 | 0.21 | 69.4 |
| +2.00 | 60.57 | 52.9 | 67.4 | 2.82 | 0.23 | 69.3 |

(full 17-α table in `/tmp/grid_cells/wang_resid_post_single.json`; the curve is
flat — every α within ~1 point of 60.8, SE ~0.2.)

### Per-α pooled sample stats (base, n = 150 per α)

| α | mean | min | max | SD | SE | coh |
|---:|---:|---:|---:|---:|---:|---:|
| −2.00 | 92.52 | 90.5 | 94.4 | 0.80 | 0.07 | 90.0 |
|  0.00 | 92.95 | 91.7 | 94.0 | 0.51 | 0.04 | 90.3 |
| +2.00 | 92.59 | 89.8 | 94.1 | 0.74 | 0.06 | 90.0 |

---

## Remaining cells

To be filled as each `qwen7b/grid/<ranking>_<sae>_<gran>/` cell lands on HF and
is judged. The recompute is identical: download the per-seed combined files,
run `grid_metrics.cell_summary`, drop the aggregate Δ into the master grid and
add a cell-detail block. Grouped cells contribute one row (group as the single
"method"); gran=1 cells contribute the over-50 distribution.
