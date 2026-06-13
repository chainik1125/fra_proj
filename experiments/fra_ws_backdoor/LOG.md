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

## Phase B — the backdoor (pod rs-wsb-bd, A40). Raw: results/backdoor_results.json (HF mirror).
Plant `# T -> P` (k=3) on 3 single-token pairs (node→result, data→index, value→output); LOCATE copy heads
by qp→payload attention (induction, not ASR-drop); neuron-FRA per copy head; CUT across the copy-head union
on HELD-OUT NOVEL-position prompts; collateral = mean-position KL on benign held-out text where P is
legitimately PREDICTED via OTHER triggers and T legitimately maps to non-payloads. Two FRA cuts: raw
top-|signed-mean| cells, and DIFFERENTIAL (backdoor-edge minus generic-induction-edge = content-specific
best shot). Baselines: position-aware ORACLE (zero qp→payload attn), payload-mask (zero attn onto P,
content-detected = attention-map analogue), payload-suppress (remove P's unembed direction). Medians over
3 pairs; ground-truth completion probs only.

### VERDICT: the in-context-backdoor FRA collateral-WIN does NOT transfer. Substrate-half (B1) transfers; win-half fails.
The backdoor plants cleanly (base ASR sparse 0.56 / wsda 0.70 / dense 0.94) and FRA LOCATES + the edge is
DRIFT-FREE, but at matched removal the FRA cell-cut pays EQUAL-OR-MORE collateral than the deployable
baselines on EVERY model — never the pre-registered ≥2× advantage.

| model  | base ASR | DRIFT top1cov / edge-cos | FRA removal max (raw / diff) | oracle max | matched tgt | collateral@tgt fra / fra_diff / payload_mask / payload_supp | win fra_diff vs (pmask, psupp) |
|--------|----------|--------------------------|------------------------------|-----------|-------------|-------------------------------------------------------------|--------------------------------|
| sparse | 0.56     | **1.00 / 0.974**         | **0.78** / 0.66              | 0.68      | 0.61        | 0.102 / 0.032 / 0.029 / 0.021                               | **0.79× , 0.65×** (loses)      |
| wsda   | 0.70     | 0.58 / 0.935             | -0.01 / 0.47                 | 0.42      | 0.00*       | ~0 (degenerate)                                             | n/a (raw cut removes ~0)       |
| dense  | 0.94     | 1.00 / 0.923             | 0.33 / 0.11                  | **0.96**  | 0.11        | 0.304 / 0.141 / 0.009 / 0.003                              | 0.06× , 0.02× (loses badly)    |

\*wsda raw-cell cut removes ~0 → matched target collapses to 0 (degenerate); the differential cut does remove 0.47.

**The two halves decompose (this is the result):**
1. **SUBSTRATE-HALF transfers (B1/W3 reproduced on a PLANTED association):** the planted edge is CONCENTRATED
   + STABLE — drift KILLED on sparse (top1cov 1.00, edge-cos 0.97, mass_top1 0.024) and dense (1.00/0.92),
   reproducing the B1 quote-circuit stability ON A PLANTED, safety-shaped edge. CUTTABILITY by a bounded
   neuron-cell set works ONLY on the ACT-SPARSE model (sparse FRA removal 0.78 ≈ oracle 0.68; wsda raw 0;
   dense 0.33 ≪ oracle 0.96) — activation sparsity is the cuttability driver, exactly B1-quote
   (sparse 0.86 / wsda 0.41 / dense 0.01). Position-invariance holds on sparse: FRA c=1 removal 0.60 (locate)
   → 0.46 (trigger at NOVEL positions) — the cell-cut generalizes to unseen positions, as designed.
2. **WIN-HALF FAILS (content-addressed separability does NOT transfer):** at matched removal FRA collateral
   ≥ both baselines on all 3 models. Even FRA's theory-prescribed best shot (DIFFERENTIAL content-specific
   cells) only reaches PARITY on sparse (0.65–0.79×, KL 0.032 vs payload-mask 0.029) and loses 15–100× on
   dense. **Cause (theory-consistent):** the planted trigger→payload association is carried by the GENERIC
   INDUCTION edge in the neuron basis (q = "repeated-context role", k = "copy-source role"), NOT a distinctive
   trigger-content × payload-content conjunction. Cutting it damages benign induction (in the collateral
   set) → high collateral. These tiny code models have no dedicated trigger/payload-CONTENT channels feeding
   QK, unlike the rich gpt2/gemma SAE basis where the banked win held. = win-checklist clause-4 failure
   (CONJUNCTION RECURRENCE via a generic ROLE endpoint; magnitude law A → ~1 with no content discriminator).

**What this means for the campaign.** Weight-sparsity is a SUBSTRATE fix (concentration / drift / cuttability —
transfers to the planted backdoor, best on the act-sparse model), NOT a CONTENT-ADDRESSING fix. The two
standing wins — W1 (in-context-backdoor collateral) and W3 (weight-sparse cuttability) — do NOT COMPOSE on
these models: the substrate that makes the edge cuttable does not make a *planted in-context* association
content-separable, because the planting rides generic induction machinery rather than dedicated content
features. A valid, theory-completing NEGATIVE (Phase F passed → this is a real Phase-B bound, not capability).
