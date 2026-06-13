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
- **OFFICIAL GATE (pod rs-wsb-feas, A40, ~2min). VERDICT = FEASIBLE — gate passes on ALL 3 models.**
  Raw: results/feas_results.json (HF fra_ws_backdoor/results/). top1 = full-vocab argmax==target,
  N=24 held-out random single-token (A,B) pairs, chance≈1/2048, copy_rate(argmax==A)=0.00 everywhere.

  | model  | best gateable (k≥2)        | top1 | P(tgt) | ARROW k3 | COLON k3 | RAW k4 |
  |--------|----------------------------|------|--------|----------|----------|--------|
  | sparse | ARROW_COMMENT k3           | 1.00 | 0.73   | 1.00     | 0.96     | 0.75   |
  | wsda   | ARROW_COMMENT k3           | 1.00 | 0.81   | 1.00     | 1.00     | 0.96   |
  | dense  | ARROW_COMMENT k3           | 0.96 | 0.81   | 0.96     | 0.88     | 0.50   |

  - The decisive point: the **SPARSE treatment substrate follows the mapping at top1=1.00** (ARROW_COMMENT
    k3) — so the in-context backdoor CAN be planted on it. Feasibility = GO for Phase B.
  - ARROW_COMMENT and COLON_MAP are the strongest formats (comment/dict-arrow conventions); RAW_INDUCTION
    (bare repeated bigram) is present but weaker (sparse/dense need k4). top1 rises monotonically with demos
    on every model+format → genuine in-context induction, not a prior. **Backdoor format chosen: ARROW_COMMENT
    at k=3** (top1≥0.96 all models; the strongest shared mapping).
- **NEXT:** Phase B — plant ARROW_COMMENT trigger→payload backdoor; LOCATE (neuron-FRA QK edge) → CUT
  (cell-cut win test vs oracle + 2 baselines, position-invariance) → DRIFT. Code = code/backdoor_pod.py.
