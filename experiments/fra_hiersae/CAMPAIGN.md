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
- NEXT: when v2 lands -> read per-cell verdict IN THE DRIFT CELLS (alpha 0.4/0.5 where flat_drift now passes) vs locked §5.1 bands
  (CONFIRM predicted) -> RED-TEAM (symmetric; is drift genuine+gpt2-magnitude? recovery_matry still pass? residual real?) ->
  PLANNING (nested-cell-aware attribution if CONFIRM). If v2 ALSO fails to drift -> the toy can't reproduce drift cheaply = its own finding.
- Builds on: synth_hier2 (planted hierarchy), sae_models.py (SAE infra + the minimal Matryoshka add), fra/ toolkit, the persistence drift finding.
