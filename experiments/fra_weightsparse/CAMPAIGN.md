# FRA × WEIGHT-SPARSE TRANSFORMERS (B1) — is the SAE/dense-superposition the bottleneck? (started 2026-06-12)

## THESIS (from IDEA_BENCH.md B1)
Every obstacle FRA has hit may be an ARTIFACT of DENSE, superposed transformers + lossy SAEs:
  - persistence (fra_persistence): the SAE q-feature DRIFTS across contexts (top1_coverage 0.32, n_cells_for_90=8,
    Spearman(FRA-score, causal-effect)=0.03 -> the induction QK edge is DIFFUSE & undiagnosable);
  - hiersae (fra_hiersae, just done): even a naive hierarchical SAE FAILS to RECOVER the drift-causing concept
    (coarseC≈0.44-0.55<0.8) because the SAE nests by reconstruction-VARIANCE, not abstraction.
A WEIGHT-SPARSE transformer (OpenAI circuit-sparsity, ~99.9% weights zero + activation-sparse, far LESS superposition)
has a sparse/structured W_QK and near-monosemantic neurons. So FRA's cells MIGHT become concentrated + consistent +
diagnosable where they were diffuse/drifting on dense gpt2. This is the cleanest "is the SAE/dense substrate the
problem, or is FRA?" test of the three basis-attacks (B1 here / B2 model-diff / hiersae).

## THE PI'S ASK (this run — the concrete first step)
(1) Train EQUIVALENT-PARAMS SAEs on a weight-sparse transformer vs a REGULAR (dense) transformer; compare HOW THEY
    TRAIN (FVU, dead-frac, L0, reconstruction quality) — does the SAE train cleaner/easier on the weight-sparse model?
(2) Run FRA on BOTH and see if there's an IMPROVEMENT (more concentrated / less drifting / more diagnosable cells).

## KEY RISKS / DISCIPLINE
- MATCHED BASELINE is critical: the comparison is only clean if the dense model is the SAME size/data as the sparse one
  (OpenAI trains pairs). If no matched dense baseline is released -> pick the best controlled comparison + FLAG the confound.
- ACTIVATION sparsity (a context-dependent SUBSET of neurons co-fires per token) means DRIFT could survive at the SET
  level even with monosemantic atoms -> MEASURE the q-feature-coverage/drift metric in the sparse basis, don't assume.
- GROUND-TRUTH metrics >> LLM-judge. RunPod only (rs-ws-* pods). Parse-gate; ckpt+traceback-upload; budget-conscious.

## STATE
- Launched: background evaluator agent (see WS_LOG.md / DESIGN.md). Orchestrator = main loop (also running hiersae red-team).
- Builds on: fra/ (FRA QK toolkit), multitrigger_sleeper/cloud/sae_models.py (TopKSAE infra), fra_persistence (the drift metrics
  to reuse), fra_hiersae (the recovery-failure refinement).
