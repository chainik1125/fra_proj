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

---
## ADDENDUM 2026-06-12 — candidate re-dos on weight-sparse (T1 induction, T2 binding); refines W3/F4/F5
(branch note: the whole FRA campaign incl. this session is on autoresearch/fra-behavioral-win — verified
single clean chain, nothing on multitrigger-sleeper that isn't here.)

T1 IDENTIFIER-INDUCTION (the persistence re-do on a task with GENUINELY VARYING content — identifiers differ
every program, the harshest substrate test): drift NOT killed, RESTRUCTURED.
 - Capability fine (gate 0.98-0.99 all 3 ladder models). Code induction lives in a REDUNDANT HEAD BANK (~12 heads,
   L1+L5) — a NEW failure mode: best single-head edge-mask removes 0.057 vs 0.97 all-heads; raw per-cell causal cuts
   ≈0 everywhere (redundancy defeats single-cell ablation). Only ISOLATED-PATH probes are cell-cuttable, and only on
   sparse (64 cells -> 0.38 ≈ 81% of bank oracle; dense/wsda 0.000).
 - The substrate's real gift: it SEPARATES the mechanism by layer. The 4 L5 bank heads have a PERFECTLY STABLE
   dominant cell (top1_cov 0.99-1.00 across 200 different-identifier contexts; top cell bias×k171, SAME k-channel as
   B1's quote circuit) = abstract position/matching cells; L1 heads DRIFT (0.27) = literal identity-match cells
   (top causal cell = the diagonal q104×k104 at L1H1). Dense SMEARS all of it (0.54-0.77). So weight-sparse does not
   eliminate content-drift but renders it INTERPRETABLE (stable-abstract vs drifting-identity, layer-separated).
 - Bank-level dominant-cell coverage on sparse = 0.35 ("drift survives" band, below dense 0.56) — identity cells must
   drift when content genuinely varies. Spearman≈0 reconfirmed. **FRA UNION-RECOVERY FAILS here (0.16-0.27 vs
   B1-quote 0.76)** => the F5 search-space-restriction mitigation is REGIME-BOUND: it works in single-head
   CONCENTRATED circuits (quote, binding) and FAILS in REDUNDANT-bank circuits.

T2 BINDING — UNBLOCKED, by WEIGHT BUDGET not width. csp_sweep1_1x_7.4Mnonzero_afrac0.250 passes set_or_string (0.725;
 the B1 3.7M model is at chance 0.588; 3.7M-afrac0.500 also passes 0.875 => capability is TOTAL-CAPACITY-gated).
 On the passing model binding is SINGLE-head (L6H13): drift killed (top1_cov 0.875, n90=2), mass1 0.0489 = 10× the
 act-dense twin, FRA top-2 cells remove 0.43 of holdout binding (recovery 1.32 — FRA beats causal probes at small k);
 act-dense FRA-ranked cuts are negative/misleading. Edge-concentration consistent with the paper's "4 q/k channels"
 scale (identity unverifiable — paper traced csp_yolo2, not this model).
 - SIBLING SELECTIVITY FAILS BY MECHANISM (not FRA's fault): cutting set-located cells improves set-binding (self
   −1.4) AND removes str-binding (+1.6) — set/str is ONE shared type-discriminator edge, not two separable
   associations, so the fra_organisms2 selectivity win (self≥0.5, ratio≥2) is structurally unreachable in THIS circuit.

NET refinement to §3 (substrate vs ranking):
 - SUBSTRATE half (W3) is more precise: weight-sparse delivers concentration+cuttability+interpretable structure on
   single-head concentrated circuits (quote, binding) and DECOMPOSES drift into interpretable layers — but it does NOT
   eliminate drift when content genuinely varies, and it does NOT defeat HEAD REDUNDANCY (new failure mode; single-cell
   cuts ≈0). Capability is capacity-gated (binding needs 7.4M / afrac0.5).
 - RANKING half (F5) boundary now MAPPED: Spearman≈0 universal (6/6 new cells); the union-recovery escape hatch works
   ONLY in single-head concentrated regimes (0.76) and FAILS in redundant-bank regimes (0.16-0.27). So FRA-as-search-
   space-restrictor is not a general mitigation — it is circuit-topology-bound.

## ADDENDUM 2026-06-12 (2) — in-context backdoor on weight-sparse (T3): the W1 win does NOT compose with W3
Phase F FEASIBLE (all 3 models follow in-context mappings; ARROW_COMMENT top-1 1.00 sparse, genuine induction
not echoing — copy_rate 0). Phase B: the banked W1 in-context-backdoor COLLATERAL WIN (15-516×/12-25× on
gpt2/gemma) does NOT transfer to the weight-sparse neuron basis.
 - SUBSTRATE-half transfers cleanly to a PLANTED, safety-shaped association: edge concentrated + DRIFT-FREE
   (sparse top1cov 1.00, edge-cos 0.97), causally cuttable ONLY on act-sparse (sparse FRA removal 0.78≈oracle 0.68;
   dense 0.33≪oracle 0.96), position-invariance holds (0.60 locate -> 0.46 trigger at NOVEL positions). So W3
   reproduces on a planted backdoor, including the position-invariance axis dense+SAE failed.
 - WIN-half FAILS: at matched removal the FRA cell-cut pays >= baseline collateral on EVERY model; even FRA's
   theory-best (differential content-specific cells) only reaches PARITY on sparse (0.65-0.79×, never the >=2× bar)
   and loses 15-100× on dense.
 - CAUSE (theory-consistent, win-checklist clause-4 / magnitude law A->1): the planted association rides the GENERIC
   INDUCTION EDGE (q="repeated-context role" × k="copy-source role"), NOT a distinctive trigger-content × payload-
   content conjunction. These tiny code models have NO dedicated trigger/payload-CONTENT channels feeding QK (unlike
   the rich gpt2/gemma SAE basis), so the conjunction collapses to a generic role-edge -> cutting it breaks benign
   induction too -> no collateral advantage.

THE DECOMPOSITION IS NOW THREE-PART (refines §3):
 1. SUBSTRATE (concentration / drift / cuttability / position-invariance): fixed by weight-sparse substrate;
    transfers to natural AND planted associations. NOT FRA-intrinsic.
 2. CONTENT-ADDRESSING (the distinctive trigger-content × payload-content conjunction that GIVES W1 its collateral
    advantage): requires RICH content features at the QK endpoints. The rich gpt2/gemma SAE basis HAS them (-> W1
    win); the tiny weight-sparse code models LACK them (-> association rides a generic role-edge -> win vanishes).
    ORTHOGONAL to substrate sparsity — sparse substrates do NOT supply it and may have FEWER content channels.
 3. RANKING (Spearman≈0): FRA-intrinsic, universal; union-recovery escape is single-head-regime-bound.
NET: W1 (content-addressing collateral win) and W3 (substrate cuttability) DO NOT COMPOSE on these models — the
collateral win needs endpoint DISTINCTIVENESS, which substrate sparsity neither supplies nor preserves. A clean
theory-completing negative (capability passed; this is mechanism, not a block). The dream "weight-sparse sleeper
with a concentrated AND content-addressed trigger cell" needs a model with BOTH sparsity AND rich content channels
-> argues for T4 (TRAIN a weight-sparse model with a planted distinctive trigger), not the released code models.
