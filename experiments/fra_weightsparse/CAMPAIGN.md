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

## RESUME STATE (canonical; idempotent)
- **PHASE 0 DONE (2026-06-12)** — DESIGN.md committed. VERDICT = **GO (conditional)**. Key facts:
  - Models loadable: OpenAI circuit_sparsity (arXiv 2511.13653). Loader = github `circuit_sparsity.inference.gpt.load_model(url, cuda=)`
    reading Azure blob `https://openaipublic.blob.core.windows.net/circuit-sparsity/models/<name>/{beeg_config.json,final_model.pt}`
    (NOT HF — only csp_yolo2 is on HF). CPU-loadable, no triton (enable_sparse_kernels=False). GPT-2-style, RMSNorm, NO RoPE,
    fused c_attn (split q/k/v), scale 1/sqrt(d_head). Trained on PYTHON CODE.
  - **MATCHED LADDER (all 1x, d_model=256, n_layer=8, n_head=16, d_head=16, vocab=2048, ~47MB):**
    `csp_sweep1_1x_3.7Mnonzero_afrac0.250` (wt-sparse+act-sparse) | `..._afrac1.000` (wt-sparse, dense-act; byte-identical arch =
    CLEAN act-sparsity isolation) | `dense1_1x` (fully dense, but depth-4 → CONFOUND, flagged). Internal dense1_1x is the rigorous
    control; gpt2-persistence numbers (top1_cov 0.32-0.5, n_cells_for_90 3-8, Spearman~0.03, frac_oracle~0.014) are cross-setting refs.
  - **Behavior** = NOT NL induction (these are Python-only, no induction-head task). Use the paper's hand-traced CODE QK circuits:
    quote-closing (single_double_quote, 1 head/1 QK channel) PRIMARY; variable-binding (set_or_string, 2-hop, "4 q/k channels" =
    ground-truth edge) SECONDARY. Present in BOTH sparse+dense (Fig 8). Ground-truth metric = binary completion prob (judge-free).
  - **FRA plan**: ω = c_attn-sliced W_Q W_K^T / sqrt(16). u from TWO bases: (A) NEURON-BASIS = model's own `act_in` (act-sparse
    d_model input) with W_dec=Identity → SAE-FREE clean test, 256²=65k cells, CPU-trivial, RUN FIRST; (B) SAE-BASIS = identical
    TopK SAEs on act_in × 3 models (Phase 1+2). SUBTLETY to measure: model re-sparsifies q,k AFTER projection (attn_q/attn_k loctypes)
    → FRA-recon-R² may drop on sparse; fallback = exactly-faithful q/k-projected FRA basis. Reuse fra/core/fra.py compute_fra_sparse.
- **P0.5 SMOKE DONE (2026-06-12, local CPU, dense1_1x only):** loader+tokenizer+task+FRA+causal stack verified end-to-end.
  dense1_1x quote acc=1.00 (gate pass), bind acc=0.50 (CHANCE — binding likely beyond 1x-width models; quote = primary,
  binding recorded as caveat). Head L3H11 drop=0.41. **FRA-recon R²=1.000000 in the neuron basis** (exact — the sweep/dense
  configs do NOT re-sparsify attn_q/attn_k, that was csp_yolo2-only; Caveat B retired). Patched-forward validation diff=0.
- **PHASES 1-2 IN FLIGHT (2026-06-12):** pod rs-ws-1 (id sm1kvywkptacvr, A40) running ws_pod.py STAGES=ABC.
  A = 3-model load + tasks + head-finding + neuron-FRA + causal (Spearman/recovery/oracle). B = identical TopK SAEs
  (d_sae=2048, k=32, 12k steps, 2M/200k train/eval tokens, python corpus) on act_in@circuit-layer. C = SAE-basis FRA + causal.
  Results → HF fra_weightsparse/results/ws_stage{A,B,C}.json + ws_combined.json; crash → ws_traceback.txt; bg poller local.
  RESUME if pod dies: bash code/launch_pod_ws.sh (env RUNPOD_API_KEY=$RP_API_KEY_MATS POD_NAME=rs-ws-2 STAGES=ABC; cheap, re-run whole).
- **NEXT:** parse ws_combined.json → WS_LOG.md comparison tables + verdict (incl. co-firing-set drift) → commit.
- Pods: rs-ws-* (reaper whitelist), RUNPOD_API_KEY=$RP_API_KEY_MATS. HF artifacts: dmanningcoe/fra-phase1-steering-data prefix
  fra_weightsparse/{code,results}. Pod deps: blobfile, tiktoken, `pip install -e` the circuit_sparsity repo, + fra toolkit.
