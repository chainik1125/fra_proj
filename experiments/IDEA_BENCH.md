# FRA IDEA BENCH — parked threads (not active; pick up when current experiments land)

The campaign's honest core finding so far (EM/injection/binding deflations + induction faithfulness FAIL + the gpt2
persistence result): FRA-via-SAE keeps hitting the SAME root obstacle — the SAE feature basis is MESSY (superposition,
feature splitting/absorption, q-feature DRIFT across contexts, lossy QK reconstruction R²<0). This makes the FRA cells
distributed, drifting, and hard to DIAGNOSE/cut cleanly. Most parked threads below attack THAT root.

## ACTIVE (not on the bench — running now)
- HIERSAE synthetic (experiments/fra_hiersae): does a hierarchical/Matryoshka SAE fix the drift? PI prediction = naive fails
  (higher-hierarchy-cells problem). Ground-truth, in flight.
- PERSISTENCE (experiments/fra_persistence): gpt2 drift result = INFORMATIVE-NEGATIVE; refocused embedding-cut/random-pos run finishing.

## BENCH

### B1. WEIGHT-SPARSE TRANSFORMERS (PI, 2026-06-12) — attack the obstacle at the ROOT
Ref: OpenAI "circuit sparsity" paper (https://cdn.openai.com/pdf/41df8f28-d4ef-43e9-aed2-823f9393e470/circuit-sparsity-paper.pdf).
HYPOTHESIS: every obstacle FRA has hit (drift, distributed/non-diagnosable cells, the higher-hierarchy-cells problem, lossy QK
reconstruction) may be an ARTIFACT of DENSE, superposed standard transformers + lossy SAEs. A WEIGHT-SPARSE transformer has a
sparse/structured W_QK and (plausibly) far LESS superposition, so:
  - the FRA QK cell decomposition should be CONCENTRATED & DIAGNOSABLE (sparse W_QK -> few feature-pairs build the score -> the
    "union" is small and readable = directly fixes the diagnosability bottleneck);
  - the features should be more MONOSEMANTIC/CONSISTENT across contexts -> LESS DRIFT -> persistence (position-invariant cut) works;
  - the model may need NO SAE hierarchy (the neurons/weights are already clean) -> sidesteps the higher-hierarchy-cells problem.
=> The interesting test: re-run the persistence/diagnosability/control experiments on a weight-sparse model and see if FRA's cells
   become concentrated + consistent + cuttable where they were drifting/distributed in dense gpt2. This is the setting where FRA's
   QK-resolution might finally pay off, because the substrate is sparse by construction.
FIRST STEP (when picked up): ARTIFACT availability — does OpenAI (or a replication) release weight-sparse model weights + the
   sparse circuits? If yes -> reuse the fra/ toolkit + the persistence harness on it. If no public weights -> either a small
   replication (train a weight-sparse toy transformer, more involved) or the SYNTHETIC analogue (a sparse-W_QK planted circuit,
   compare FRA diagnosability on sparse-vs-dense planted weights — cheap, ground-truth, the natural extension of the hiersae synthetic).
RISK/NOTE: FRA on a weight-sparse model may decompose in the NEURON/weight basis directly (no SAE needed) — that itself is a clean
   test of "is the SAE the problem, or is FRA". Pairs naturally with the hiersae result (both ask: is a cleaner basis the fix?).

### (future entries appended here)
