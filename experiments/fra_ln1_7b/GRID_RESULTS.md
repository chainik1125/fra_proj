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
   ~51% relative residual error (loss_recovered ≈ 0.51) — lossy but usable. The on-pod
   `var-expl ≈ −1.33 / "SAE looks broken"` warning is an FVU/DC-offset artifact
   (resid_post activations carry a large mean/DC component → tiny variance denominator
   → spurious negative var-expl), NOT a broken SAE. All conventional steering adds
   `α·W_dec[f]` directly (never round-trips through encode→decode), so steering validity
   is independent of reconstruction quality. resid_post cells are judged and reported
   normally. (If single resid_post steering is flat — as Wang×resid_post×single already
   is — a lossy SAE's individual directions genuinely may just not steer.) ln1's
   var-expl=0.498 IS a real quality number (ln1 is ~centered post-RMSNorm).

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

## Master grid — Δalign@coh{70/50/30} (EM/medical model)

Rows = ranking × SAE × granularity. Values are mean Δalign across seeds
(for gran=1, mean over the features: n=50 Wang & FRA-OV×resid_post, n=26 FRA×ln1).
SD column is across-seed (grouped) or across-feature (gran=1). Base-model values in
the next table.

| ranking | SAE | gran | Δ@coh70 | Δ@coh50 | Δ@coh30 | n | status |
|---|---|---:|---:|---:|---:|---:|---|
| Wang-Δf | resid_post | 1 (×50) | **4.98** ± 1.54 | **7.80** ± 1.38 | **7.80** ± 1.38 | 50 feats | done (n=64) |
| Wang-Δf | resid_post | 2 | — | — | — | | pending |
| Wang-Δf | resid_post | 10 | — | — | — | | pending |
| Wang-Δf | resid_post | 50 | — | — | — | | pending |
| Wang-Δf | ln1 | 1 (×50) | — | — | — | | pending |
| Wang-Δf | ln1 | 2 | — | — | — | | pending |
| Wang-Δf | ln1 | 10 | — | — | — | | pending |
| Wang-Δf | ln1 | 50 | — | — | — | | pending |
| FRA-QK | ln1 | 1 (×26) | — | — | — | 26 feats | pending |
| FRA-QK | ln1 | 2 | — | — | — | | pending |
| FRA-QK | ln1 | 10 | — | — | — | | pending |
| FRA-QK | ln1 | 50→grp26(all) | — | — | — | | pending |
| FRA-QK | resid_post | — | — | — | — | | **SKIP** (ill-defined) |
| FRA-OV | ln1 | 1 (×26) | — | — | — | 26 feats | pending |
| FRA-OV | ln1 | 2 | — | — | — | | pending |
| FRA-OV | ln1 | 10 | — | — | — | | pending |
| FRA-OV | ln1 | 50→grp26(all) | — | — | — | | pending |
| FRA-OV | resid_post | 1 (×50) | — | — | — | | pending |
| FRA-OV | resid_post | 2 | — | — | — | | pending |
| FRA-OV | resid_post | 10 | — | — | — | | pending |
| FRA-OV | resid_post | 50 | — | — | — | | pending |

Base-model Δalign (same cells; a robustly-aligned base should give ≈ the noise floor):

| ranking | SAE | gran | Δ@coh70 | Δ@coh50 | Δ@coh30 |
|---|---|---:|---:|---:|---:|
| Wang-Δf | resid_post | 1 (×50) | 2.64 ± 0.31 | 2.64 ± 0.31 | 2.64 ± 0.31 |
| … | | | (others pending) | | |

---

## Cell detail: Wang-Δf × resid_post × single (n = 64) — DONE

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
