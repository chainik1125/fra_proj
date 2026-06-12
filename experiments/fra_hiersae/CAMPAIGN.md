# FRA HIERSAE — does a hierarchical SAE fix the q-feature DRIFT for FRA? (synthetic, ground-truth)
THESIS: the flat-SAE drift (fra_persistence: q-feature drifts -> a single FRA cell doesn't transfer; FRA specific-but-not-
position-invariant) is feature splitting/absorption. A hierarchical SAE MIGHT fix it (stable coarse concept cell). PI PRE-
REGISTERED (PREREG.md): the NAIVE application FAILS (may partially help) due to the HIGHER-HIERARCHY-CELLS problem (nested
non-orthogonal features spread the QK score across levels -> coarse cut alone leaves residual -> back to a union). 
PIPELINE: DESIGN (HIERSAE_DESIGN.md, done) -> EVALUATOR (ab3c214cf1f2a9b54 -> HIERSAE_LOG.md, synthetic CPU-tiny: flat vs
Matryoshka SAE, level-spreading frac_coarse_xx + coarse-cut residual + 3 axes, alpha-sweep, ground-truth) -> RED-TEAM
(symmetric) -> PLANNING (if CONFIRM -> scope the nested-cell-aware attribution; if FALSIFY -> real-LLM). CRON 43dbd707.
## RESUME STATE
- PHASE A DONE: PREREG.md + HIERSAE_DESIGN.md committed. PHASE B:
  - v1 SMOKE (evaluator ab3c214cf1f2a9b54) LANDED but INCONCLUSIVE = harness-validity FAIL: all 4 cells GATE-FAIL on
    flat_drift (top1_cov_flat=1.0 vs gpt2 0.32 -> the synthetic flat SAE did NOT drift -> nothing for the hierarchy to fix).
    Root cause: flat SAE latches the concept not the drifting leaf when alpha+c_anchor>sqrt(1-alpha^2); c_anchor=0.5 + smoke
    alphas>=0.5 => no drift. The other 2 validity gates PASSED. FULL DIAGNOSIS + fix in HIERSAE_LOG.md. (evaluator wrote no
    VERDICT/log -> orchestrator took over the synthesis.)
  - v2 DRIFT RE-RUN IN FLIGHT (orchestrator): drift-validity fix baked into script defaults (C_ANCHOR 0.5->0.2, ALPHAS
    0.4-0.7 bracket the drift onset, superpos 1,2 / seeds 0,1). Running LOCAL CPU (self-contained synthetic, minutes;
    uploads HF fra_hiersae/results/hiersae_results_drift.json). [decision: a self-contained CPU numpy/SAE toy is plotting-
    scale, not a GPU model run -> local bg + HF upload for durability; flag if PI wants it on a pod.]
- Companion: gpt2 PERSISTENCE = INFORMATIVE-NEGATIVE (q-drift; FRA specific ~1995x but not position-invariant), FINAL committed (efc1de1).
- Companion: MEAN-FIELD/CONDENSATE theory (B3) LANDED -> fra_meanfield/THEORY.md (sound; lean NO-condensate on dense gpt2 via the
  same drift; r_eff(M^B) before/after a basis cleanup = a diagnostic that rides on hiersae/B1/B2). Committed c4c4624.
- PHASE A DONE -> VERDICT = CONFIRM (naive hierarchical SAE does NOT fix FRA drift), committed ddac76b. BUT via an
  UNANTICIPATED, more-basic mechanism than the pre-registered higher-hierarchy-cells level-spreading:
  RECOVERY FAILURE. Evidence (3 cells: validation α=0.4/0.7 + iw8 α=0.5): in the drift regime the naive (minimal)
  Matryoshka coarse prefix gives NO recovery advantage over the flat SAE (coarseC_matry≈coarseC_flat≈0.44-0.55, robust
  to α∈{0.4,0.5} and inner_weight∈{2,8}); flat-drift and concept-recoverability are TWO SIDES OF ONE COIN (the decoupling
  redesign is self-defeating); MECHANISM = Matryoshka nests by reconstruction-VARIANCE, the drift-causing concept is
  low-variance-SHARED (high-variance=leaves) -> prefix captures leaves not concept. Control α=0.7: concept strong ->
  both recover (0.92/0.90), no drift -> not rigged. The predicted level-spreading is UNTESTABLE here (recovery fails first).
- CONFIRMATION SWEEP IN FLIGHT (gates the red-team): α∈{0.4,0.5,0.6,0.7} × sp1 × seeds{0,1}, iw8 (Matryoshka's best shot),
  steps8000 = 8 cells, local CPU -> hiersae_results_coupling.json. Solidifies the drift→no-drift transition + recovΔ≈0
  across the bracket. Poller bv9esiu8m notifies on completion (~20 min). HOLD the red-team until it lands (else "only 3 cells").
- PERSISTENCE: DONE + committed (efc1de1, persist_results_FINAL.json w/ all 3 verdicts + embedding-cut/random-pos M3a/M5).
- NEXT (B): when the sweep lands -> finalize verdict w/ 8-cell evidence -> RED-TEAM Workflow (symmetric, 4 opus skeptics):
  is the recovery-failure CONFIRM real? targets = (1) is the minimal Matryoshka a FAIR non-strawman (would a better
  hierarchical SAE recover)? (2) is the variance-nesting the TRUE cause? (3) is the 0.8 recovery bar fair? (4) false-neg:
  α=0.7 recovers, so concept not unrecoverable-by-construction ✓. Then (C) PLANNING: CONFIRM -> the next problem is no longer
  ONLY nested-cell-aware attribution but the UPSTREAM concept-recovery (variance-vs-abstraction nesting); scope both.
- Builds on: synth_hier2 (planted hierarchy), sae_models.py (SAE infra + the minimal Matryoshka add), fra/ toolkit, the persistence drift finding.

## RED-TEAM VERDICT (Workflow wf_fc463f7a, 4 opus skeptics + synthesizer, 2026-06-12) — RELABEL, not reversal
hiersae verdict RELABELED from CONFIRM to: "prereg level-spreading mechanism UNTESTED (0/8 gate-passing cells in the
coupling grid) + robust upstream finding: NO fair naive-Matryoshka recipe achieves clean coarse recovery (>=0.8) in the
flat-drift regime." Surviving MAJOR objections (both factually verified by the skeptics):
 (1) MECHANISM MIS-ATTRIBUTED: the stated "variance-nesting / low-variance-shared concept" is wrong as worded — the
     D-root (16 leaves, MORE than C's 8) recovers 0.81-0.94 in EVERY drift cell while C fails; the discriminator is
     CLEAN-SIGNAL STRENGTH (D's weight-1.0 clean query injection vs C's invented-and-weakened c_anchor=0.2; the original
     synth_hier2 injects the clean concept at 1.0 and has NO c_anchor). Entanglement/redundancy + weak clean signal,
     not low variance (base[C] is literally PC1 of the full token distribution). [Nuance: in the CENTERED C-leaf
     emission subspace the concept does live in the mean and in no PC — the two variance readings measure different
     distributions; the D-vs-C discriminator is the decisive evidence.]
 (2) SIMPSON ARTIFACT: recovΔ mean -0.02 only over all 8 cells; over the 4 DRIFT cells recovΔ=+0.054, monotone in
     alpha (+0.165 at the transition). Truthful headline: matry never CROSSES the 0.8 gate in a drift cell but trends
     positive there. Secondary: alpha=0.6 seed0 is FALSIFY-SHAPED (meets every locked causal FALSIFY bar; excluded only
     by recovery-cos 0.765<0.80; a 0.1 gate perturbation flips it) — 1-of-2 seeds, boundary-fragile.
 (3) "DECOUPLING SELF-DEFEATING / coupling fundamental" FALSIFIED twice: the evaluator's c_anchor=0/alpha=0.7 dual-gate
     cell (matry recovers 0.83 + flat drifts) AND the scope-skeptic's counterexample generator (standalone concept
     emission at non-query positions: flat recovers 0.888 AND drifts 0.125). The coupling is a c_anchor>0 generator
     artifact, not a law.
ATTACKS THAT FAILED (hardening the empirical core): strawman (6 recipes incl. Bussmann geometric ladder + per-prefix-
TopK + iw32 ALL miss 0.8; implementation bug-free; prefix fires ~100% so no TopK starvation), under-training (iw8/10k
moved 0.44->0.55 only), information-theoretic impossibility (mean-of-leaves recovers 0.72-0.82 -> the failure is the
variance-greedy INDUCTIVE BIAS, the more interesting reading).
WHERE THE PREREG MECHANISM WAS ALMOST/ACTUALLY TESTABLE the coarse cut was CLEAN (alpha=0.6s0: resid 0.09, fcx 0.84;
noanchor dual-gate: resid 0.07, fcx 0.91) -> level-spreading leans FALSIFY where readable; the REAL blockers are
UPSTREAM (recovery) and DOWNSTREAM (shared-concept specificity collapse: spec_ratio 1.36 vs gpt2-flat ~1995x).
NEXT (red-team's cheapest decisive test, CPU-minutes): a NON-variance-greedy coarse atom (mean-pool / coarse-first
objective) on the SAME drift-cell activations, alpha in {0.5,0.6} x >=5 seeds -> settles inductive-bias-vs-information,
resolves the alpha=0.6 seed-flip, and OPENS the never-tested level-spreading regime.
