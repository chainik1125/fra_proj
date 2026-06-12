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

## v2 VALIDATION (2 cells: α=0.4 drift-candidate, α=0.7 control; sp1 seed0 steps5000, C_ANCHOR=0.2)
The c_anchor fix WORKS for drift, but exposed a deeper coupling. Results:
  - α=0.4: top1f=0.12 -> flat SAE DRIFTS HARD (more than gpt2's 0.32). flat_drift gate PASSES ✓. BUT recovery_matry
    FAILS: coarseC_matry_cos=0.435 (<0.8 gate) AND coarseC_flat_cos=0.468 -> NEITHER SAE recovers the concept (both
    recover the LEAVES perfectly, leafcos_flat_mean=0.991). full_union_ge_oracle also FAILS (R_full=0.69<R_orc=1.15).
    The resid=0.32/carry=0.54 LOOK like CONFIRM but are CONFOUNDED by the recovery failure (can't cut a concept cell
    you never recovered) -> NOT a readable verdict (this is exactly the red-team artifact the brief flagged).
  - α=0.7 (control): top1f=1.00 no drift (correct), recovery_matry PASSES (coarseC_matry_cos=0.921), flat ALSO recovers
    the concept (coarseC_flat_cos=0.896).

THE KEY STRUCTURAL FINDING (and a tension to resolve): in THIS toy, concept strength (α) controls flat-DRIFT and
Matryoshka-RECOVERABILITY in the SAME direction. Low α -> concept too entangled with leaves -> flat drifts BUT neither
SAE recovers the concept. High α -> both recover the concept AND no drift. The naive Matryoshka coarse prefix gets NO
recovery ADVANTAGE over the flat SAE. This is arguably an EVEN STRONGER form of the PI's "naive hierarchical SAE fails"
prediction than the higher-hierarchy-cells problem: in the drift regime the naive Matryoshka doesn't even RECOVER the
concept (coarseC=0.44), let alone cut it cleanly. BUT we must distinguish: is this FUNDAMENTAL (naive can't recover the
drift-causing concept) or a TRAINING ARTIFACT (under-powered prefix: inner_weight=2, 5000 steps)?

## v3 PIVOTAL TEST — DONE: recovery failure is ROBUST (not a training artifact)
One cell α=0.5, INNER_WEIGHT=8 (4×), STEPS=10000, C_ANCHOR=0.2. RESULT: flat_drift PASS (top1f=0.12) but
recovery_matry STILL FAILS — coarseC_matry_cos=0.545 (up only from 0.435 at iw2; still ≪ 0.8 bar). A 4× harder-trained
prefix barely moved recovery -> the recovery failure is ROBUST, not under-training.

## THE DECOUPLING REDESIGN IS SELF-DEFEATING (key realization — do NOT pursue it)
I considered decoupling recovery from drift via a clean strong concept emission in-context. Reasoning it through: the
flat SAE drifts PRECISELY BECAUSE it has no stable concept atom (only leaf atoms; the FRA top cell uses a leaf q-feature
that varies across contexts). If the concept were cleanly recoverable, the flat SAE would recover it AND USE it — the
planted head reads the concept (aC), so the concept×D cell has HIGH ω and would dominate the FRA score -> stable cell ->
NO drift. So flat-DRIFT and concept-RECOVERABILITY are TWO SIDES OF THE SAME COIN: there is NO regime where "flat drifts
AND the concept is recoverable." Confirmed empirically: α=0.7 (recoverable, coarseC=0.92) has NO drift; α=0.4/0.5
(drift) have coarseC≈0.44/0.55. The coupling is fundamental, not an artifact.

## VERDICT: CONFIRM (naive hierarchical SAE does NOT fix FRA drift) — via RECOVERY FAILURE, upstream of the predicted mechanism
The PI pre-registered: naive hierarchical SAE + FRA FAILS via the higher-hierarchy-cells problem (level-spreading ->
coarse-cut leaves residual). We CONFIRM the headline (naive fails) but the operative mechanism is MORE BASIC and upstream:
  - THE HOPE behind a hierarchical SAE: its coarse prefix recovers a STABLE concept atom where the flat SAE can't ->
    one stable concept×D cell -> position-invariant cut -> fixes drift. (That would be FALSIFY.)
  - WHAT HAPPENS: the naive (minimal Matryoshka, widths [16,256]) coarse prefix gives NO recovery advantage over the flat
    SAE in the drift regime (coarseC_matry ≈ coarseC_flat ≈ 0.44–0.55, robust to α∈{0.4,0.5} and inner_weight∈{2,8}).
  - THE MECHANISM (why naive Matryoshka fails — the citable insight): the Matryoshka prefix nests by RECONSTRUCTION-
    VARIANCE; the drift-causing concept is LOW-variance-SHARED (the high-variance directions are the leaves), so the
    coarse prefix captures LEAVES, not the concept. Its inductive bias ("fewest latents") ≠ "abstract/shared concept
    first." So it never yields the stable concept-level FRA cell that would fix drift.
  - The predicted higher-hierarchy-cells level-spreading (residual after a SUCCESSFUL coarse cut) is a SECOND-ORDER
    effect that is UNTESTABLE here because recovery fails first. (Weak directional hints exist — α=0.4 resid=0.32,
    fcx=0.52 — but they're confounded by the recovery failure, so not load-bearing.)
  - CONTROL (method is not rigged to fail): α=0.7 — concept strong -> BOTH SAEs recover it (coarseC 0.92/0.90) and there
    is no drift. So the concept is NOT unrecoverable by construction; it is unrecoverable exactly when it causes drift.

## CONFIRMATION SWEEP — DONE (8/8 cells; hiersae_results_coupling.json; iw8 = Matryoshka's best shot)
α    seed | top1f drift | cC_flat cC_matry recovΔ | resid carry fcx | R_full R_orc
0.4  0    | 0.12  T     | 0.45   0.44   -0.01    | 0.19  0.00  0.00 | 0.19  1.15   <- pathological: full union can't reach oracle
0.4  1    | 0.12  T     | 0.56   0.62   +0.06    | 0.37  0.50  0.53 | 0.73  0.63
0.5  0    | 0.12  T     | 0.53   0.52   -0.00    | 0.39  0.51  0.63 | 0.80  1.08
0.5  1    | 1.00  F     | 0.78   0.60   -0.18    | 0.44  0.33  0.63 | 0.66  0.64   <- boundary cell, seed-flipped
0.6  0    | 0.12  T     | 0.60   0.77   +0.17    | 0.09  0.91  0.84 | 1.00  1.06   <- TRANSITION: drifts yet nearly recovers
0.6  1    | 1.00  F     | 0.90   0.82   -0.09    | 0.11  0.86  0.85 | 0.83  0.70
0.7  0    | 1.00  F     | 0.93   0.86   -0.07    | 0.01  0.99  0.96 | 1.07  1.09
0.7  1    | 1.00  F     | 0.94   0.93   -0.02    | 0.02  0.96  0.93 | 0.60  0.80

READ-OFF (finalized):
1. NO RECOVERY ADVANTAGE: recovΔ mean=-0.02, range [-0.18,+0.17], no trend. The naive Matryoshka coarse prefix NEVER
   meaningfully beats the flat SAE at recovering the concept. (THE core CONFIRM evidence.)
2. PERFECT DRIFT<->RECOVERY COUPLING: every drift cell (4/4) has cC_matry<0.8; every cell with cC_flat>=0.9 has no
   drift. The drift boundary sits ~alpha 0.5-0.6 with seed noise flipping boundary cells — matching the
   sqrt(1-a^2)>a+c_anchor arithmetic (~0.57 at c_anchor=0.2).
3. TRANSITION-ZONE NUANCE (honest, for the red-team): alpha=0.6 seed0 drifts (top1f=0.12) yet nearly recovers
   (cC_matry=0.77, resid=0.09, carry=0.91). In a narrow band the hierarchy ALMOST works — so the verdict on the BAND
   is closer to HELP-BUT-NOT-SOLVE; the verdict in the BULK of the drift zone (alpha<=0.5) is a clean fail.
4. WHERE RECOVERY SUCCEEDS THE COARSE CUT IS CLEAN (no-drift cells: resid 0.01-0.11, fcx 0.85-0.96): the pre-registered
   level-spreading mechanism does NOT bite at high alpha in this toy — consistent with v1 smoke. The predicted
   higher-hierarchy-cells problem remains UNTESTED (needs a regime with recovery AND drift, which the coupling forbids).
VERDICT (final, pre-red-team): CONFIRM — naive hierarchical SAE does not fix FRA drift — via RECOVERY FAILURE
(variance-nesting), with a narrow transition band where it partially helps. Now to the symmetric red-team.
