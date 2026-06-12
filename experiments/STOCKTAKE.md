# FRA CAMPAIGN STOCKTAKE — 2026-06-12 (full-thread review: wins / fails / open threads)

Scope: the entire FRA-wins thread — organisms v1 (boundary map), organisms2 (binding), circuits, the
persistence/hierarchy reframe, fra_persistence, fra_hiersae, fra_meanfield (B3 theory), fra_weightsparse (B1),
plus the banked sleeper-campaign results. Everything below is post-red-team unless marked.

## 1. WINS (robust, red-teamed)

W1. **Association-specific attention control — the core FRA-unique win** (fra_win, banked + replicated):
    suppressing a cue's induction via the bilinear query×key cell edit with ~15× less collateral than ActAdd;
    synthetic suite (induction / copy-suppression / in-context backdoor) 15–516×; acronym external validity
    (median R=0.93, A≈10×, up to 38×); gemma in-context backdoor collateral-flip replicates ~12–25×.
W2. **Association-specificity on real gpt2** (fra_persistence M5): at matched removal, FRA cell-cut collateral is
    ~2569× smaller than the embedding-cut baseline. Where the FRA cell binds, the cut is surgically specific.
    (The same campaign showed it does NOT transfer across contexts — see F4. Specific-but-drifting.)
W3. **Weight-sparse substrate result** (fra_weightsparse B1, the first basis-attack that moved the numbers):
    on OpenAI circuit-sparsity models (byte-identical afrac pair as primary contrast) — identical-config SAEs
    train 3.2× better (FVU 0.042 vs 0.136, monotone in substrate sparsity); NEURON-basis FRA (no SAE) is exactly
    faithful (R²=1.000), 17× more concentrated (top1 edge-mass 0.169 vs 0.010), consistent across contexts
    (edge-cosine 0.97 vs 0.73, dominant cell 200/200), and causally cuttable (10 cells remove 86% vs 1% of the
    edge); FRA recovers the paper's hand-traced circuit shape; the SAE DE-concentrates the sparse substrate →
    "skip the SAE on sparse models" confirmed. Caveats: dense1_1x depth confound (flagged), tiny models, binding
    at chance on the act-sparse model (capability cost).
W4. **The boundary map** (organisms v1): four win-conditions (load-bearing edge / context-supplied no-backstop /
    consumed-at-answer-step / non-recurrent conjunction) + the magnitude law A ≈ reuse(marginal)/reuse(conjunction).
    Predictively validated by every subsequent deflation and win.
W5. **Hiersae empirical core** (post-relabel, hardened by the failed strawman attack): NO fair naive-Matryoshka
    recipe (6 recipes incl. Bussmann geometric ladder, per-prefix TopK, iw32) achieves clean coarse recovery
    (≥0.8) of the drift-causing concept in the flat-drift regime — and the concept IS recoverable in principle
    (mean-of-leaves 0.72–0.82), so the failure is the variance-greedy INDUCTIVE BIAS, not information. The
    localization axis genuinely improves (matry top1_cov 0.875–1.0 vs flat 0.125); the specificity axis collapses
    where the coarse concept is shared (spec_ratio 1.36 vs gpt2-flat ~1995×).
W6. **Binding QK diagnosis** (organisms2): L22H4 retrieval is CONTENT-addressed, not positional (R_gen 0.90 vs
    R_prefill 0.075; 29/29) — the diagnostic (A2) result stood while the control win deflated (F2).
W7. **Process wins**: the symmetric red-team caught 3 false positives (EM, binding 11×, pattern-freeze α),
    2 premature negatives (factedit, injection timing), and relabeled hiersae; APE-oracle K-independence and the
    detector≠payload dissociation (sleeper campaign) remain standing results.

## 2. FAILS (informative negatives — each sharpened the theory)

F1. **Per-instance selectivity deflations** (EM, injection 1.6×=null, binding 11× = a 2-column-mask artifact,
    sycophancy): at the single-word level the FRA cell-cut degenerates to an attention-mask/linear steer; the
    win METRIC was wrong, not the rigor. → produced the persistence/hierarchy reframe (the value is weight-space
    persistence + association-specificity, not per-instance selectivity).
F2. IOI: backup name-movers defeat the cell-cut (R=−0.39) — precise-but-behaviorally-inert (LBNR fail).
F3. **Induction faithfulness fail** (gpt2+SAE): FRA reconstructs QK direction (corr≈0.50) but not magnitude
    (R²<0) — the QK-reconstruction thesis is unreliable on dense+SAE.
F4. **Drift + diffuseness + undiagnosability** (fra_persistence FINAL): the SAE q-feature for a word DRIFTS
    across contexts (top1_cov 0.32; the located cell transfers at 0.005–0.014 vs oracle 0.79–0.98); the induction
    edge is spread over thousands of cells (~1% each, n90=8, multi-head); FRA magnitude is UNCORRELATED with
    causal effect (Spearman 0.03). FRA on gpt2-dense+flat-SAE is specific-but-not-position-invariant and cannot
    diagnose its own union.
F5. **Ranking is FRA-INTRINSIC** (B1's sharpest negative): Spearman(FRA-score, causal-effect) ≈ 0 on EVERY
    substrate including the cleanest possible (weight-sparse, neuron basis, exactly-faithful decomposition).
    Magnitude-ranking fails everywhere; only the UNION works (top-10 recovers 76% of causal-top-10 → FRA is a
    search-space restrictor, not a causal ranker).
F6. **Naive hierarchical SAE does not deliver the clean win** (fra_hiersae, RELABELED post-red-team): the locked
    level-spreading mechanism was UNTESTED (0/8 gate-passing cells; leans FALSIFY where readable — coarse cuts
    are clean); the real blockers are UPSTREAM (recovery: absorption/entanglement of the shared concept into
    context-specific atoms under any variance-greedy objective) and DOWNSTREAM (shared-concept collateral).
    My (orchestrator) over-claims corrected by the red-team: variance-nesting mechanism mis-attributed
    (clean-signal strength is the discriminator — coarseD with MORE leaves recovers everywhere); recovΔ −0.02 was
    a Simpson artifact (drift-cells-only +0.054); "coupling is fundamental" falsified twice (c_anchor=0 dual-gate
    cell; standalone-emission counterexample).
F7. **Mean-field/condensate** (B3, theory only): sound formalism (M^B SVD → rank-1 bilinear edits) but the lean
    is NO-condensate on dense gpt2 — the same drift that defeats everything else predicts a high-rank M^B.
    Parked with a nearly-free decisive test (r_eff of M^B; Stage A synthetic is free CPU).

## 3. THE EMERGENT PICTURE (what the whole thread adds up to)

FRA's obstacle has decomposed into TWO INDEPENDENT HALVES:
  - **SUBSTRATE half** (drift, diffuseness, absorption): a property of dense superposed models + variance-greedy
    SAEs, NOT of FRA. Evidence: weight-sparse substrate FIXES it (W3: concentration/consistency/cuttability);
    naive hierarchical SAEs do NOT (F6 — same variance-greedy bias as flat); averaging is predicted not to (F7);
    persistence drift (F4), hiersae recovery failure (F6), and the no-condensate lean (F7) are all ONE root —
    shared concepts get absorbed into context-specific atoms whenever the basis is learned by reconstruction.
  - **RANKING half** (FRA score ⊥ causal effect): FRA-intrinsic (F5), survives every basis change. The open
    research problem. Mitigation that works today: use FRA as a SEARCH-SPACE RESTRICTOR (union, 76%), validate
    causally inside the restricted set.

Where FRA's unique value now demonstrably lives (the three axes, decoupled as the PI insisted):
  - CONTROL: association-specific bilinear edits where the cell binds — collateral advantage 15×–2569× over
    linear/embedding baselines (W1, W2). The empty corner (position-invariant AND specific in ONE cell) is
    reached NOWHERE on dense+SAE — but IS reached in the weight-sparse neuron basis (W3: stable cell + 86% cut).
  - DETECTION/DIAGNOSIS: content-addressing questions (W6), circuit-shape recovery (W3), AUROC=1 detection even
    where control fails (hiersae 3-axis read) — diagnosis decouples from control exactly as predicted.
  - LOCALIZATION: improved by hierarchy (W5) and by sparse substrates (W3); the union is small and readable on
    sparse substrates.

STRATEGIC READ: FRA may be the natural intervention language for the COMING sparse-substrate model generation,
while on today's dense models its reliable uses are (i) specificity-critical control where a stable cell exists,
and (ii) diagnosis/search-space restriction feeding causal validation.

## 4. OPEN THREADS (ranked by information-per-dollar)

T1. **Mean-pool / non-variance-greedy coarse atom** (red-team's decisive next test; CPU-minutes): on the same
    drift-cell activations, α∈{0.5,0.6}×≥5 seeds. Settles inductive-bias-vs-information, resolves the α=0.6
    seed-flip, and OPENS the never-tested level-spreading regime (the PI's original prereg mechanism).
T2. **B2 MODEL-DIFFING → FRA** (benched LEAD; crosscoder infra exists): now best framed as "can a diff-basis
    deliver weight-sparse-grade concentration on DENSE models?" — the selection attack, and the only untested
    basis-attack. Pre-registered risk: the ranking half (F5) will NOT be fixed by it; design the eval around
    union/concentration, not score-ranking.
T3. **The RANKING problem itself**: no basis fixes it → needs causal-aware attribution (e.g. cheap per-cell
    causal probes inside the FRA-restricted union; or attribution-patching hybrids). This is now the single
    deepest FRA-intrinsic open problem.
T4. **Mean-field r_eff diagnostic** (B3): Stage A free; Stage B = one small pod; most informative run AFTER a
    basis cleanup (does the cleanup make M^B condense?) or in the weight-sparse neuron basis (where drift is
    gone — condensate likely; would give low-rank collective edits there).
T5. **Weight-sparse follow-ups**: the binding capability cost (afrac0.25 at chance → 2-hop ground-truth edge
    untestable — try afrac0.5 or the 2x/4x ladder); co-firing-set drift at scale; do the W3 gains survive on
    bigger sparse models / real tasks.
T6. **Path 1 unexploited**: the in-context-backdoor weight-edit win (banked) — cut trigger→payload as a
    persistent W_QK edit, disarmed-unconditionally framing; safety-relevant and never pushed to a full result.
T7. **Nested-cell-aware attribution** (coarsest-sufficient level): the prereg's "real research problem" —
    becomes live only if T1 opens the recovery+drift regime.

## 5. RECOMMENDED SEQUENCE
(1) T1 now (minutes, settles a live dispute + may unlock the prereg mechanism test);
(2) B2 model-diff as the next full campaign (T2), with the F5 ranking-risk pre-registered;
(3) fold T4's r_eff into whichever basis-attack runs next (free diagnostic);
(4) keep T3 (ranking) as the explicit theory target — it is now THE bottleneck FRA owns.
