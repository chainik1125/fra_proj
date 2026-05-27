# Conventional-steering grid results — Qwen-2.5-7B (FRA vs conventional)

Headline metric: **Δalignment @ coherence floor {70, 50, 30}**. For each
(feature-or-group, eval-seed), over the α-window where that seed's coherence ≥
floor, Δ = max(alignment) − min(alignment); then **mean ± SD across the 3 seeds**
(42/123/456). For gran=1 cells the cell value is the distribution of that
per-seed-mean Δ **over the 50 individually-steered features**; for grouped cells
it is the single group's mean ± SD across seeds. Higher Δ = steering moves
alignment more over its coherent window.

Conventional **additive** steering throughout: add `α·W_dec[f]` (single) or
`α·Σ_topN W_dec[f]` (grouped) on the decoder direction; α ∈ [−2, 2] step 0.25
(17 points). ln1 cells inject `α·W_dec/γ` at the pre-gain hook so attention sees
post-gain `α·W_dec` (γ = `blocks.15.ln1.w`); resid_post adds directly. n = 32
(8 prompts × 4 samples) per (feature/group, α, seed), except the pre-existing
Wang×resid_post×single cell at **n = 64**. Judge: GPT-4o, temp 0.

Source: `dmanningcoe/fra-phase1-steering-data`, prefix `qwen7b/grid/<ranking>_<sae>_<gran>/`.
Recompute logic: `/tmp/grid_metrics.py` (per-seed Δ over coh≥floor windows; validated
to reproduce the team-lead's floor-70 summary exactly, max diff 0.0 over 50 features).

---

## Master grid — Δalign@coh{70/50/30} (EM/medical model)

Rows = ranking × SAE × granularity. Values are mean Δalign across seeds
(for gran=1, mean over the 50 features). SD column is across-seed (grouped) or
across-feature (gran=1). Base-model values in the next table.

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
| FRA-QK | ln1 | 1 (×50) | — | — | — | | pending |
| FRA-QK | ln1 | 2 | — | — | — | | pending |
| FRA-QK | ln1 | 10 | — | — | — | | pending |
| FRA-QK | ln1 | 50 | — | — | — | | pending |
| FRA-QK | resid_post | — | — | — | — | | **SKIP** (ill-defined) |
| FRA-OV | ln1 | 1 (×50) | — | — | — | | pending |
| FRA-OV | ln1 | 2 | — | — | — | | pending |
| FRA-OV | ln1 | 10 | — | — | — | | pending |
| FRA-OV | ln1 | 50 | — | — | — | | pending |
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
