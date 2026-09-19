# ⚠️ Which plots in this folder are correct — READ THIS FIRST

## ❌ WRONG — do not use, do not cite

**`WRONG_DO_NOT_USE_b1_gpt2_pareto_flatline_artifact.png`**
This is the old GPT-2 Pareto plot with the **flat `pay`/`dom` lines** Dmitry flagged as suspicious.
**It is wrong.** The flat lines are a measurement artifact: the coefficient grid was too coarse, so the
smallest strength already removed ~100% of the target, every point piled up at removal ≈ 1, and
interpolating collateral "at 30/50/70%" just returned a constant → a flat horizontal line.

**It does not just look wrong — it gave a wrong conclusion.** It made `pay`/`ov` look ~10× worse than
FRA. After fixing the grid, `pay`/`ov` are actually **lower collateral than FRA**. So any statement of
the form "FRA beats payload-suppression by ~10×" that came from this plot is retracted. See
`docs/insen/WHERE_WE_ARE_sep19.md` §6 (walk-back #1).

## ✅ CORRECT — current plots

- **`b1_real_collateral.png`** — the corrected, honest result (gemma-2-2b, finer grids, all methods
  sampled as real curves, reproduced across 5 GPUs). This *replaces* the flat-line plot. Here `pay`/`ov`
  correctly show up as the **lowest** lines; FRA/hybrid beat single-feature and DoM.
- **`hookpoint_sweep.png`** — the full 4-hookpoint sweep (GPT-2). Shows FRA ties the best single feature
  (at `hook_attn_out`) — i.e. the "beats every hookpoint" claim does **not** hold. See §5 / §6 walk-back #2.

## ⚠️ Partial — valid but superseded in framing

- **`b1_gpt2_general.png`** — FRA vs single-feature *at the residual hookpoint only*, on general text. The
  comparison it draws is still true (FRA ≈ 0 general-text collateral vs single-feature ≈ 1 nat), but it
  predates the hookpoint sweep, so it should **not** be used to claim FRA beats *every* hookpoint — an
  `attn_out` feature is competitive (see `hookpoint_sweep.png`).

Full story: `docs/insen/WHERE_WE_ARE_sep19.md`.
