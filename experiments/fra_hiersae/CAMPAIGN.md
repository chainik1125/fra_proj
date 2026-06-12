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
