# HIERSAE × FRA synthetic — evaluator log

## v1 SMOKE RUN (evaluator ab3c214cf1f2a9b54) — INCONCLUSIVE (harness-validity FAIL, not CONFIRM/FALSIFY)

Config (a REDUCED smoke, NOT the script defaults): ALPHAS=[0.5,0.6,0.7,0.9], SUPERPOS=[1], SEEDS=[0].
Result on HF `fra_hiersae/results/hiersae_results.json` (saved locally as `hiersae_results_smoke.json`):
**all 4 cells = GATE-FAIL.** The single failing gate is `flat_drift` (the other two PASS: `recovery_matry` ✓
coarseC_matry_cos 0.82–0.98; `full_union_ge_oracle` ✓ R_full≈R_oracle).

THE FAILURE: `top1_coverage_flat = 1.0` in EVERY cell (vs the gpt2 persistence anchor 0.32). **The synthetic flat
SAE does NOT drift** — so there is no drift pathology for the hierarchical SAE to fix, and the experiment cannot
read CONFIRM/FALSIFY. This is a harness-validity failure (the toy did not reproduce the empirical phenomenon the
prediction is about), not a verdict.

ROOT CAUSE (diagnosed from the code + numbers): the flat SAE latches the CONCEPT atom, not the drifting leaf,
whenever the concept's weight at the query exceeds the drifting-leaf weight. At the query the model injects
`1.0·F[qleaf] + c_anchor·F[idxC]`, and each leaf is alpha-mixed `F[c]=α·base[concept]+√(1-α²)·base[c]`. So:
  - concept (base[idxC]) weight at query = α + c_anchor  (CONSTANT across contexts → a stable SAE atom)
  - drifting-leaf weight                  = √(1-α²)        (leaf IDENTITY varies across the 8 contexts → drift)
  Drift requires √(1-α²) > α + c_anchor. With C_ANCHOR=0.5 this needs α ≲ 0.35; the smoke alphas (≥0.5) could
  NEVER drift. The comment "leaf weight 1.0 >> c_anchor" missed that the leaf ALSO carries α·concept, so the
  concept accumulates weight from BOTH the leaf's α-component AND c_anchor and ends up ≥1.0 > the leaf's ≤0.87.
  Compounding: the smoke config also DROPPED the script defaults' drift cells (alpha=0.3, superpos=2).

DIRECTIONAL SIGNAL (real but moot while flat doesn't drift): at LOW alpha (0.5, 0.6) the Matryoshka coarse-cut
shows mild level-spreading (frac_coarse_xx 0.74/0.86, residual_coarse 0.15/0.07) — a HINT of the higher-hierarchy-
cells problem; at HIGH alpha (0.7, 0.9) it's a clean single cell (frac_coarse_xx=1.0, residual=0, carry_coarse=1.0).
Also: control collateral RISES with alpha (collat_coarse 0.37→2.33, spec_ratio 2.72→0.43) — at high alpha the coarse
concept is shared across domains so cutting it bleeds. None of this is a verdict; the drift precondition is unmet.

## FIX (orchestrator, 2026-06-12) — restore the validity precondition (drift), then read the verdict
Baked into the script defaults (durable, documented in-code at the config block):
  - C_ANCHOR 0.5 → 0.2  (the principled fix: stop handing the flat SAE a clean concept atom; restores the design
    intent that the drifting leaf dominates and c_anchor is a WEAK shared signal for the Matryoshka prefix only).
  - ALPHAS → 0.4,0.5,0.6,0.7  (BRACKETS the drift onset at C_ANCHOR=0.2: drift needs √(1-α²)>α+0.2 → α<~0.57, so
    0.4/0.5 should drift [flat top1_cov<0.5, gpt2-matching], 0.6/0.7 should not = the no-drift control).
  - SUPERPOS=1,2, SEEDS=0,1 (the smoke dropped superpos=2; restore + 2 seeds for stability, cost-bounded).
NOTE on rigor: lowering c_anchor to MATCH gpt2's drift is a VALIDITY fix (the toy must reproduce the pathology),
NOT verdict-manufacturing — the CONFIRM/FALSIFY still reads off the Matryoshka coarse-cut RESIDUAL, which is not
tuned. The red-team must check: (i) flat drift is genuine + gpt2-magnitude (top1_cov ~0.3–0.5); (ii) recovery_matry
still passes (the coarse cell is a real recovered concept, not a badly-trained-Matryoshka artifact); (iii) the
residual/level-spreading in the DRIFT cells is the real signal.

## v2 DRIFT RUN — IN FLIGHT
Local CPU background run (self-contained synthetic, minutes; uploads to HF `fra_hiersae/results/hiersae_results_drift.json`
at end). Config = the new defaults above. PENDING: when it lands, read per-cell verdicts IN THE DRIFT CELLS (α=0.4,0.5
where flat_drift now PASSES) against the locked §5.1 bands — CONFIRM (coarse-cut residual ≥~0.3, coarse carries <~0.6,
bounded union still needed) / FALSIFY (clean single-cell coarse win ≥70%, top1_cov_matry≥0.8) / HELP-BUT-NOT-SOLVE.
