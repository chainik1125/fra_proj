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

### B2. MODEL-DIFFING -> FRA (PI, 2026-06-12) — the MOST DIRECT attack on the d_sae^2 structural diffuseness (LEAD)
THE STRUCTURAL PROBLEM (persistence FINAL established): FRA decomposes S[q,k]=Σ_{μν} u^μ_q u^ν_k ω_{μν} over μ,ν∈[d_sae]
-> d_sae^2 cells/token-pair (~600M at d_sae=24k). The causal signal is spread over THOUSANDS of cells; FRA-magnitude is
UNCORRELATED with causal effect (Spearman 0.03); no small cell-set removes the edge. This is COMBINATORIAL — a perfect SAE
still has d_sae^2 cells. Fatal for ANY QK intervention that must search this space.
THE IDEA (PI): MODEL-DIFFING. Train an SAE/crosscoder on the DIFF between behavior-PRESENT and behavior-ABSENT activations
-> a TINY behavior-specific feature set (k_diff << d_sae). Run FRA over the diff-basis -> (k_diff)^2 cells (tractable) and
the relevant cells concentrated BY CONSTRUCTION. This SELECTS the relevant subspace rather than reorganizing the basis
(vs B1 weight-sparse + the hiersae hierarchical SAE, which clean the basis). The natural pipeline FRA was missing:
model-diff (select features for THIS behavior) -> FRA (resolve QK among ONLY those) -> diagnose/cut.
REPO INFRA EXISTS (from the sleeper work — diffed base-vs-sleeper to find behavior-specific features): sae_models.py
(TemporalCrosscoder / MatryoshkaTemporalCrosscoder / MultiLayerCrosscoder), modeldiff_baseline_pod.py, dom_steer_pod.py,
featdist_extract_v2_pod.py (weight-model-diff / activation-model-diff / activation-input-diff channels). Adapt to a new behavior.
ADAPTATION TO VARIABLE BINDING ("Ann has the ale...Who has the pie?->Joe"): two contrasts for two goals —
 (a) MECHANISM diff: binding vs scrambled/no-binding -> the binding-retrieval machinery (concentrate the circuit);
 (b) INSTANCE diff: same prompt, "Who has the pie?"(->Joe) vs "Who has the ale?"(->Ann) / value-swap -> the (entity×value)
     direction for the SELECTIVE-cut goal.
 Train diff-SAE/crosscoder on answer-position acts across the contrast; FRA on the diff-basis; RE-RUN the metrics that failed
 on the full SAE: CONCENTRATION (top-k carries the edge?), DIAGNOSABILITY (Spearman(s,c) lifts off 0.03?), SELECTIVITY (cut
 Joe×pie, preserve siblings). Reuse the binding eval (fra_organisms2) + the persistence diagnosability harness.
THE KEY RISK (must be IN the test, or it deflates like the others): the diff-features are CORRELATIONAL — they DIFFER between
 conditions but cutting their QK cells need not be CAUSAL. The Spearman-0.03 problem could RECUR within the diff-basis. So the
 test is NOT "is the space smaller" (trivially yes) but "does DIAGNOSABILITY + CAUSAL-cuttability actually improve". And the
 contrast must match the goal (mechanism diff != instance-selective cut). Pre-register these before running.
STATUS: LEAD of the three basis-attacks (B2 model-diff vs B1 weight-sparse vs hiersae hierarchical) — most direct + infra exists.
 Decision point: weigh the three once the hiersae verdict lands.

### (future entries appended here)
