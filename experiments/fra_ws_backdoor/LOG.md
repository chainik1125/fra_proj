# FRA × WS × IN-CONTEXT BACKDOOR — LOG

Chronological. Verdict-bearing numbers only; full JSON in results/.

## Phase F — feasibility gate
- **setup.** Campaign home created; reused B1 vendor loader + neuron-FRA + cell-cut causal path.
  Pre-registered gate = ≥0.6 top-1 on ≥20 held-out instantiations of SOME in-context mapping format
  (at n_demos≥2, so it's a genuine in-context mapping not a 1-shot copy; chance ≈ 1/2048).
- **harness.** code/feas_pod.py — battery {RAW_INDUCTION, ARROW_COMMENT, ASSIGN, COLON_MAP} ×
  k∈{1,2,3,4} demos × N=24 held-out random (A,B) single-token identifier pairs, per model. Probe target
  is offset-aligned to B's actual continuation token; copy_rate (argmax==A) reported as a trivial-copy
  control. Resumable per-model, traceback-upload. Validated by a local smoke (B1 P0.5 pattern).
- **local smoke (dense1_1x, N=12, k=1..3).** Harness correct (copy_rate=0 everywhere → not echoing A,
  top1 rises monotonically with demos → genuine induction). DENSE follows in-context mappings clearly:
  ARROW_COMMENT k3 top1=0.92 P(tgt)=0.81; COLON_MAP k2 0.83; ASSIGN k3 0.67 (all ≫ chance 1/2048).
  Gate did not fire in smoke only because N=12<20 (correct). Official gate = pod run, all 3 models, N=24.
- **NEXT:** rs-wsb-feas pod runs feas_pod.py on sparse+wsda+dense; verdict below.
