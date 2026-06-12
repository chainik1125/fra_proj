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
ARTIFACT: AVAILABLE (UPDATED 2026-06-12). OpenAI released the "circuit-sparsity" models + OPEN TOOLS + sparse<->dense activation
   BRIDGES (paper arxiv 2511.13653; method = ~99.9% weights zero + activation sparsity ~1/4 neurons/token). So RUNNABLE now.
THE TENSION (PI) RESOLVED by the design: (a) sparsity "reduces superposition + single-purpose features" (cleaner = Effect 1) AND
   (b) the objective EXPLICITLY "discourages using more neurons than strictly needed to represent a single concept" -> penalizes the
   concept-FRAGMENTATION that would cause drift (AGAINST the user's Effect-2 worry). So lean = it HELPS the drift problem, not worse.
   RESIDUAL WORRY (the real thing to measure): ACTIVATION sparsity means a context-dependent SUBSET of neurons co-fires per token, so
   drift could survive at the SET level even with monosemantic atoms -> measure the q-feature-coverage metric in the sparse basis.
THE CLEAN TEST (the deeper payoff): the monosemantic neurons ARE the disentangled features the SAE was trying (+ failing via
   absorption) to recover -> SKIP THE SAE, run FRA in the NEURON basis directly. This DECISIVELY answers "is the SAE the problem, or
   is FRA?" (the question induction-R²<0 + the drift raised): if FRA-on-neurons is consistent+diagnosable where FRA-on-SAE-on-dense
   drifted, the SAE (+dense superposition) WAS the bottleneck. Sidesteps absorption entirely; pairs with hiersae (both: is a cleaner basis the fix?).
LITERATURE: no direct study of SAEs-on-weight-sparse (models are weeks old). But absorption/splitting (Chanin 2409.14507) is a DENSE-
   SAE pathology whose ROOT (superposition) weight-sparse attacks -> prior = SAEs absorb LESS / are unnecessary. Adjacent: hierarchical
   sparse circuit extraction via attribution graphs (2601.12879) sits at the hiersae+diagnosability junction.
EXPERIMENT (when picked up): re-run the drift/diagnosability/control(embedding-baseline)/persistence metrics on the released weight-
   sparse model in the NEURON basis, vs dense+SAE. Cheap synthetic warm-up = a sparse-W_QK planted circuit (FRA diagnosability sparse-
   vs-dense planted weights), the natural extension of the hiersae synthetic.

### (future entries appended here)
