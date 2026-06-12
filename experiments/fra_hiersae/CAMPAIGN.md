# FRA HIERSAE — does a hierarchical SAE fix the q-feature DRIFT for FRA? (synthetic, ground-truth)
THESIS: the flat-SAE drift (fra_persistence: q-feature drifts -> a single FRA cell doesn't transfer; FRA specific-but-not-
position-invariant) is feature splitting/absorption. A hierarchical SAE MIGHT fix it (stable coarse concept cell). PI PRE-
REGISTERED (PREREG.md): the NAIVE application FAILS (may partially help) due to the HIGHER-HIERARCHY-CELLS problem (nested
non-orthogonal features spread the QK score across levels -> coarse cut alone leaves residual -> back to a union). 
PIPELINE: DESIGN (HIERSAE_DESIGN.md, done) -> EVALUATOR (ab3c214cf1f2a9b54 -> HIERSAE_LOG.md, synthetic CPU-tiny: flat vs
Matryoshka SAE, level-spreading frac_coarse_xx + coarse-cut residual + 3 axes, alpha-sweep, ground-truth) -> RED-TEAM
(symmetric) -> PLANNING (if CONFIRM -> scope the nested-cell-aware attribution; if FALSIFY -> real-LLM). CRON 43dbd707.
## RESUME STATE
- PHASE A DONE: PREREG.md + HIERSAE_DESIGN.md committed. PHASE B IN FLIGHT: evaluator ab3c214cf1f2a9b54 -> rs-hiersae-1 (synthetic).
- Companion: gpt2 PERSISTENCE result = INFORMATIVE-NEGATIVE (q-drift; FRA specific ~1995x but not position-invariant); refocused
  embedding-cut/union/random-position run via afa99d66feb3da860 may still be landing.
- NEXT: read HIERSAE verdict (CONFIRM predicted) -> red-team -> planning (nested-cell attribution if CONFIRM).
- Builds on: synth_hier2 (planted hierarchy), sae_models.py (SAE infra + the minimal Matryoshka add), fra/ toolkit, the persistence drift finding.
