# Mean-Field / Condensate Decomposition of FRA-QK

**Status: speculative theory (B3). Pure-theory deliverable, no GPU.**
**Branch: autoresearch/multitrigger-sleeper.**

## 0. One-paragraph statement of the idea

The microscopic FRA decomposition writes the per-token-pair QK score as a sum over
d_sae^2 feature-pair "cells",
S[q,k] = Σ_{μ,ν} u^μ_q u^ν_k ω_{μν}, with ω_{μν} = (W_dec[μ]·W_Q)(W_dec[ν]·W_K)/√d_head
the *data-independent* feature-pair interaction and u_q, u_k the per-position SAE
codes. On dense gpt2 + flat SAE this decomposition is **diffuse**: the induction edge
is built from thousands of cells each carrying ~1%, FRA-magnitude is uncorrelated with
causal effect (Spearman≈0.03), and the top-1 cell removes ~1.4% of the edge
(`persist_results_FINAL.json`: `frac_of_oracle`≈0.014, `cov3` union barely moves it).
This diffuseness is combinatorial — a *perfect* SAE still has d_sae^2 cells — and is
fatal to any intervention that must *search* the cell space.

**The mean-field / condensate idea.** Stop looking at microscopic cells. For a fixed
behavior B, average the QK interaction over B's instances *in a frame that does not
cancel the signal*, forming a single d_sae×d_sae **effective interaction matrix**
M^B (an order parameter). If M^B is approximately **low-rank** — M^B ≈ Σ_{a=1}^{r}
λ_a |q_a⟩⟨k_a| with r small — then the behavior's attention is carried *collectively*
by a few **condensate modes** (q_a, k_a), each an *effective query-direction ×
effective key-direction*, even though no single microscopic cell carries it. The
intervention is then a **rank-1 (or rank-r) bilinear edit** of the effective W_QK along
a condensate mode, applied at the score level, *not* a search over ~600M cells. This is
FRA's bilinear QK intervention re-expressed in a **clean low-dimensional effective
basis** rather than the diffuse microscopic SAE-feature basis.

Note the existing code already contains the embryo of M^B: `fra/core/fra.py::
get_sentence_averages` accumulates `data_dep_int_matrix` — a behavior-averaged
d_sae×d_sae interaction. The mean-field proposal is, in essence: take its spectrum, ask
whether a few modes suffice, and ask whether editing along them beats a matched linear
steer. The whole bet lives in those two questions (§2).

## 1. Precise formulation

### 1.1 Setup and notation

Fix one head (layer ℓ, head h). Let W_QK = W_Q W_K^T / √d_head ∈ R^{d×d} (d = d_model)
be its bilinear form, so the raw score for a token-pair is S = x_q^T W_QK x_k, where
x_q, x_k are the post-LN residual-stream vectors entering Q and K. The SAE gives
x ≈ Σ_μ u^μ W_dec[μ] + b_dec; substituting,

  S[q,k] = Σ_{μν} u^μ_q u^ν_k ω_{μν},   ω_{μν} = W_dec[μ]^T W_QK W_dec[ν].   (1)

ω ∈ R^{d_sae×d_sae} is the *data-independent* feature-pair interaction (a property of
the weights + SAE only). The *data-dependent* per-pair cell value is
c_{μν}[q,k] = u^μ_q u^ν_k ω_{μν}. The microscopic decomposition is over these
d_sae^2 cells; its diffuseness is the problem.

### 1.2 What is averaged, and in what frame (the order parameter)

A behavior B is a distribution over (sequence, query-position, key-position) triples
where B's attention edge is operative — e.g. for induction, (q = a repeated token,
k = the earlier occurrence's successor); for the planted toy (§4), (q = final query
carrying the persona feature, k = a domain-key position). Define the **mean-field
effective interaction**

  M^B_{μν} = E_{(q,k)∼B}[ u^μ_q u^ν_k ] · ω_{μν}   = ⟨c_{μν}⟩_B.    (2)

Equivalently M^B = Ω ⊙ (⟨u_q u_k^T⟩_B), the Hadamard product of the static ω with the
behavior-averaged outer product of codes. M^B is the **order parameter**: a single
d_sae×d_sae object summarizing B's attention, as opposed to one ω per pair.

**The averaging-frame caveat (critical).** Naive averaging of *signed* per-instance
QK structure can cancel. For a value-binding behavior ("Ann→ale, Joe→pie") the
per-instance edge points entity_i → value_i, and these instance directions are
near-orthogonal across instances, so E[c]≈0 even though every instance has a strong
edge — the signal lives in the *covariance*, not the mean. Two remedies, and the
choice of which defines the variant:

  (i) **Conditioned/aligned frame.** Average only within an instance-aligned frame —
  condition on the behavior's *role* (the "this is the entity-slot / value-slot"
  directions), or rotate each instance into a canonical (query-role, key-role) frame
  before averaging. Then E[c]_B is over instances that genuinely share structure.
  (ii) **Second-moment (covariance) order parameter.** Use
  G^B_{μν,μ'ν'} = Cov_B(c_{μν}, c_{μ'ν'}) or, more usefully, the *un-centred* second
  moment of the per-instance score-gradient, and take its top modes. This captures
  behaviors whose mean cancels but whose fluctuations are coherent.

For an induction edge (a *fixed* q-feature attends to a *fixed* k-feature regardless of
instance) the mean (2) is already the right object and does not cancel. For
value-binding it is not; (i) or (ii) is required. The proposal must state which frame
B lives in *before* averaging — this is the single most important modelling choice.

### 1.3 The condensate directions

Given M^B (the mean order parameter (2)), the **condensate modes** are its top singular
triples:

  M^B = Σ_a λ_a |q_a⟩⟨k_a|,   q_a, k_a ∈ R^{d_sae},   λ_1 ≥ λ_2 ≥ ...    (3)

The behavior **condenses** iff a few modes carry it: Σ_{a≤r} λ_a² / Σ_a λ_a² ≈ 1 for
small r (an *effective rank* / participation-ratio statement, §2a). Each condensate
mode pushes back to the residual stream as an **effective bilinear form**

  W^a_QK^eff = (W_dec^T q_a)(W_dec^T k_a)^T ∈ R^{d×d},   rank 1 in d-space,   (4)

i.e. an effective query-direction p_a = W_dec^T q_a and key-direction r_a = W_dec^T k_a
in residual space, with S^a[q,k] = λ_a (x_q·p_a)(x_k·r_a). The intervention "cut the
condensate" is the score-level edit S → S − λ_a (x_q·p_a)(x_k·r_a) applied at *every*
token-pair (a weight-space, position-free edit of the attention interaction). This is
exactly the toy's `cell_S` route — `raw_scores − omega·outer(rd(x,idxP), rd(x,idxD))` —
generalized from a planted rank-1 route to an *empirically discovered* condensate.

### 1.4 Genuine mean-field, or a relabelled SVD? (be honest)

As stated, (2)–(3) is a **behavior-conditioned low-rank approximation** of the averaged
interaction — an SVD of an averaged matrix. That is useful but is *not yet* mean-field
theory in the physics sense; calling it "condensate" is so far a metaphor. A genuine
mean-field structure would require a **self-consistency** condition: the effective
single-mode interaction should be derivable from the average over the others, e.g. the
condensate direction that a token attends *with* is itself the B-average of the
directions induced by the modes it attends *to*. Concretely, a self-consistent ("TAP"-
style) equation would read

  q_a ∝ E_B[ W_QK (Σ_b λ_b ⟨k_b| x_k⟩ k_b ... ) ]   — a fixed-point in (q_a, k_a),

so that the condensate is the *stable* solution of a closed loop rather than a one-shot
SVD. Whether attention admits such a self-consistent closure (plausibly via the
softmax's mean-field over keys: the attention a query pays is a Gibbs average that each
key sees as a field) is an open and genuinely interesting question — but it is NOT
needed for the intervention to work. **Honest position:** the *minimal* proposal is the
behavior-conditioned low-rank approximation (a plain, well-posed SVD of M^B); the
self-consistency story is an optional, more ambitious overlay that would earn the
"mean-field/condensate" name. We should test the minimal version first and not dress
the SVD in physics language when reporting.

## 2. The two decisive questions

### 2a. Does a condensate emerge? (is the behavior-averaged QK genuinely low-rank?)

The intervention is only cheap if M^B (Eq. 2) is genuinely **low-rank**. Diffuseness of
the *microscopic cells* (no single c_{μν} carries the edge) does **not** by itself imply
M^B is high-rank — that is the whole hope. The two are different statements:

  - Microscopic diffuseness: the *entries* of M^B are spread over many (μ,ν).
  - High *rank*: those entries cannot be written as a few outer products q_a k_a^T.

A matrix can have thousands of non-negligible entries yet be rank-1 (e.g. M = q k^T with
q, k both dense). So the open question is purely **spectral**, and is the crux of the
whole idea.

**When a condensate exists (low rank).** M^B = Ω ⊙ ⟨u_q u_k^T⟩_B is low-rank when *either*
factor is. Two clean sufficient conditions:
  (a) **Coherent code structure.** If the behavior fires a stable set of query features
  with roughly fixed relative weights (a query "template" vector q̄) attending to a
  stable set of key features (k̄), then ⟨u_q u_k^T⟩_B ≈ q̄ k̄^T is rank-1 and M^B is
  rank-1 (after the ⊙ω reshaping, low rank). This is the *condensate-exists* regime: the
  cells are diffuse (q̄, k̄ have many nonzeros) but collinear.
  (b) **Low-rank static interaction.** Empirically (Elhage et al., 2021) attention heads
  often have effective W_QK rank ≪ d_head; ω inherits this. If ω itself is rank-ρ, then
  M^B has rank ≤ ρ·(rank of the code-moment) regardless of code diffuseness.

**When diffuseness survives averaging (high rank — the failure mode).** M^B stays
high-rank when the behavior is realized by **many distinct, non-collinear feature
templates** that the averaging does not align: feature *drift* (the persistence campaign's
core finding — the SAE picks different (q,k) features for the "same" association across
contexts) means ⟨u_q u_k^T⟩_B is a *sum of many near-orthogonal* rank-1 templates, i.e.
high rank. Drift is exactly the property that makes the microscopic union large AND keeps
the averaged matrix high-rank. So the persistence/hiersae drift result is the leading
*a priori* reason to expect **no condensate** on dense gpt2+flat-SAE. (The induction edge
on the toy, by contrast, has one template by construction → condenses.)

**The measurable test (pre-registerable).** Compute M^B from a held-out sample of B's
instances, then report:
  - **Effective rank / participation ratio** r_eff = (Σ_a λ_a)² / Σ_a λ_a² and the
    spectral gap λ_1/λ_2, λ_2/λ_3. Condensate ⇔ r_eff small (say ≤ 3–5) with a gap.
  - **Captured-mass curve** f(r) = Σ_{a≤r} λ_a² / ‖M^B‖_F², i.e. how much of the averaged
    interaction the top-r modes explain.
  - **Behavioral capture** (the one that matters): does the rank-r reconstruction
    M^B_{(r)} reproduce B's attention edge — replace the score by the rank-r condensate
    and measure the copy-prob / attention-mass it recovers. A condensate that has small
    r_eff in Frobenius norm but does not *behaviorally* recover the edge is not useful.
  - **Cross-validation across the averaging frame:** r_eff must be stable to the choice of
    §1.2 frame (i)/(ii); if r_eff explodes under the signed mean but is small under the
    aligned/covariance frame, that diagnoses a binding-style cancellation, not a true
    condensate.

### 2b. Bilinear vs linear-collapse

Suppose a condensate exists (§2a passes). The mode is a rank-1 bilinear edit
S → S − λ (x_q·p)(x_k·r) with effective query-direction p and key-direction r (Eq. 4).
This is **only scientifically interesting if it is irreducibly bilinear** — if it does
something a residual-stream (linear) edit cannot. Otherwise it "collapses to linear
steering", which is the campaign's standard failure mode (every binding/injection/EM win
deflated to a matched linear steer).

**What linear collapse means, precisely.** A residual-stream intervention removes a
direction from the stream: x → x − (x·v) v for some v, at query and/or key positions. Its
effect on the score is

  ΔS_lin = −(x_q·v)(W_QK v · x_k... )  + ... — it perturbs S through a *fixed direction in
  x-space at one side*.

The bilinear condensate edit is ΔS = −λ (x_q·p)(x_k·r): it is gated by the **product** of
a query-side projection and a key-side projection. Three distinct cases:

  1. **Key-side degenerate (collapses to linear).** If the key-projection (x_k·r) is
  ≈ constant over the relevant keys (e.g. r aligns with a feature present at *all* candidate
  key positions, or r is essentially the BOS/positional sink), then ΔS ≈ −λ' (x_q·p),
  a *query-side residual bias* — reproducible by a linear edit (or even a bias term). No
  bilinear content.
  2. **Query-side degenerate (collapses to linear).** Symmetrically, if (x_q·p) ≈ const
  over the relevant queries, the edit is a key-side additive bias, again linear.
  3. **Irreducibly bilinear (the interesting case).** Both (x_q·p) and (x_k·r) vary
  across the relevant token sets AND are *not* collinear with any single residual
  direction whose removal would have the same effect. Then the edit changes S only for
  the *conjunction* (query has p) ∧ (key has r) — selectively killing the A→B edge while
  leaving (i) A attending to other keys and (ii) other queries attending to B. A residual
  edit cannot achieve this: removing p from the query stream kills A's attention to
  *everything*, not just to r-keys.

**The condition, stated.** The condensate is irreducibly bilinear iff the rank-1 edit
W^a_QK^eff = p r^T cannot be matched by any rank-1 edit of the form (residual-direction)
acting on one side only — i.e. iff *both* singular directions p, r have non-trivial,
non-constant projection variance across the operative query/key sets, and the
behavioral effect of the bilinear cut differs from the best single-side linear edit.
Equivalently: the edit lives off the "linear-collapse subspace" = {edits expressible as a
constant times one-sided projection}.

**How to test it (the decisive comparison).** Build, on held-out data:
  - **Bilinear condensate cut**: S − λ (x_q·p)(x_k·r).
  - **Matched linear steers** (the strong baseline, tuned, per the writing rules):
    (a) project p out of the *query* stream; (b) project r out of the *key* stream;
    (c) the best *single* residual direction v (incl. the difference-of-means B-direction)
    and matched α, chosen to maximize A→B edge removal. Match the *total intervention
    norm* / removal magnitude so the comparison is about selectivity, not strength.
  - **Pre-registered metric: selectivity at matched removal.** At equal A→B edge removal,
    compare collateral — (i) A's attention to non-r keys; (ii) other queries' attention
    to r-keys; (iii) downstream copy-prob on benign uses of A. **Irreducibly bilinear ⇔
    the condensate cut achieves the same removal at strictly lower collateral than every
    matched linear steer.** **Linear collapse ⇔ a one-sided projection matches both the
    removal AND the collateral** (then FRA's bilinearity bought nothing). The toy already
    instruments exactly this: `cell` (bilinear route) vs `dom_un`/`keyrm` (one-sided
    projections) with R_X (removal) and collat_Y (selectivity) — the mean-field test is
    the same contrast with an *empirically discovered* condensate instead of the planted
    route.

## 3. Relation to other attacks + microscopic FRA

### 3.1 Is it a coarse-graining / RG step over the d_sae^2 cells?

Yes, in a precise sense. The condensate mode q_a is a *weighted pooling* of microscopic
query-features (q_a = Σ_μ (q_a)_μ ê_μ) and likewise k_a pools key-features; the rank-r
truncation discards the orthogonal complement (the "UV" microscopic detail) and keeps the
collective "IR" mode. This is a single **block-spin / coarse-graining** step over the
feature-pair lattice: many microscopic cells → one effective mode, with λ_a the coupling
of the coarse field. It is *not* a full RG flow (no rescaling, no iteration), and the
self-consistency closure (§1.4) is what would turn it into one. So: an RG-*flavoured*
coarse-graining, honestly a one-step projection, not a renormalization group.

The key conceptual difference from the other attacks: **B1 (weight-sparse) and the
hiersae thread try to make the microscopic basis itself clean** (fewer/monosemantic
atoms so cells concentrate); **B2 (model-diffing) selects a small discrete feature
*subspace*** (k_diff features) before decomposing. **Mean-field keeps the full diffuse
basis but coarse-grains it post-hoc into collective modes** — it concedes the
microscopic cells are diffuse and bets the *average* is simple. It is the only one of the
four that does not try to fix or shrink the basis.

### 3.2 Composition with the other attacks

  - **With model-diffing (B2):** natural and probably the strongest version. Run the
    mean-field average *over the diff-basis* — M^B becomes k_diff×k_diff, the SVD is
    trivially cheap, and the diff-basis has already removed behavior-irrelevant features
    that would otherwise inflate r_eff. B2 selects *which* features; mean-field asks
    whether those features' interaction *condenses* to a few modes. Recommended as the
    eventual real-model pipeline if the synthetic passes.
  - **With hierarchical / weight-sparse (B1):** a cleaner basis should *lower* r_eff
    (less drift → fewer templates → lower-rank M^B). So mean-field is a *diagnostic that
    rides on top of* any basis cleanup: "did the cleaner basis make the behavior
    condense?" is exactly r_eff before vs after.
  - **Composability is the honest selling point**: mean-field is orthogonal to all three
    basis attacks and can be stacked on any of them. It is a *read-out + intervention*
    layer, not a competing basis.

### 3.3 Honest prior art — is this materially new or a relabelling?

This is the part to be most careful about. There is substantial prior work on low-rank /
averaged attention structure, and the bilinear-QK view is **not** new:

  - **Low-rank W_QK / QK-eigenvalue analysis (Elhage et al., 2021, "A Mathematical
    Framework for Transformer Circuits").** The QK circuit *is* a bilinear form W_QK and
    its low-rank / eigenstructure was analyzed there; induction heads' low-rank QK
    structure is well known. Our ω is exactly their W_QK pulled into the SAE basis. **The
    bilinear-edit-of-W_QK idea is theirs, not ours.**
  - **Attention-pattern / head averaging.** Averaging attention patterns or scores over a
    task to find a head's "function" is standard (attention rollout, head attribution,
    mean attention maps). M^B is a feature-resolved version of a behavior-averaged score.
  - **Low-rank steering / activation editing & concept directions.** Refusal-is-a-
    direction, ITI, etc. establish the *linear* (one-sided) edit; the §2b contrast is
    precisely against this literature.
  - **SVD/low-rank function-vectors and "task vectors" in attention.** Function-vector
    and in-context-vector work extracts low-rank task summaries from attention heads; our
    condensate is close in spirit.

**What would be materially new, if anything:** (1) the *feature-resolved* averaging — M^B
is in the SAE-feature basis, so a condensate mode q_a is *interpretable* as a weighted
set of features (a named query-template), which the raw-W_QK eigenvectors are not; (2)
the explicit **diffuse-microscopic-yet-low-rank-average** claim and its falsifiable test
(does behavior-averaging collapse a provably-diffuse cell structure?) — this is the
genuinely novel empirical question, and the persistence campaign's drift result makes its
answer *non-obvious* (drift predicts it fails); (3) the **selectivity-over-matched-linear**
test for irreducible bilinearity at the behavior level. **Honest verdict on novelty:** the
*machinery* (low-rank bilinear QK edit) is old; the *question* (does a behavior whose
microscopic FRA cells are diffuse nonetheless have a low-rank average, and is editing that
average more selective than linear steering?) is a new and falsifiable framing, but it is
incremental on Elhage et al. + the SAE literature, not a new mechanism. If r_eff is small
*and* the edit beats linear, the contribution is "feature-resolved behavior-averaged QK
condenses where microscopic cells don't, and gives selective control" — a modest, real
result. If not, it is a clean negative that strengthens the diffuseness story.

## 4. Concrete falsifiable proposal (synthetic-first)

Synthetic-first, two stages. Stage A establishes the method *can* recover a planted
condensate and tells bilinear from linear with ground truth; Stage B is the real test on
a behavior whose microscopic FRA is *known to be diffuse* (induction on gpt2).

### Stage A — planted-condensate sanity check on `synth_hier2`

`synth_hier2.py` already plants a rank-1 bilinear route per domain:
W_QK contains Σ_k √σ |p̂⟩⟨k̂_k| and `cell_S(target)` cuts exactly the rank-1
route `omega·outer(rd(x,idxP), rd(x,idxD[target]))`. This is a *known* condensate. The
modification is small: instead of cutting the *planted* route, **discover M^B by averaging
and SVD it**, then cut the *discovered* top mode.

Concretely: (1) build B = "persona-on, domain-k active" final-query→domain-key pairs;
(2) form M^B (Eq. 2) in the d=64 toy feature basis (cheap; reuse `rd`/`Ahat` to get codes,
or work directly in the d-space since the toy has no SAE); (3) SVD → (λ_a, p_a, r_a).

Pre-registered Stage-A metrics (ground truth known):
  - **Recovery**: does the top mode's (p_1, r_1) align with the planted (p̂, k̂_k)?
    cos ≥ 0.9 expected. **r_eff ≈ 1** with a large spectral gap.
  - **Knob — break the condensate on purpose**: re-plant the route as a *sum of m
    near-orthogonal* (p̂_i, k̂_i) pairs (drift surrogate) with m = 1, 2, 4, 8. r_eff
    should climb ≈ m; the captured-mass curve should flatten. This is the *positive
    control for the failure mode* — it shows the metric detects high-rank, so a small
    r_eff on the real model is meaningful.
  - **Bilinear vs linear (the §2b contrast, ground truth)**: cut the discovered top mode
    vs `dom_un` (project persona from query) vs `keyrm` (project domain-key) at matched
    R_X; the toy's `measure(...)` already returns R_X / collat_Y / style. **Expect the
    bilinear cut to match the planted-route selectivity (collat_Y≈0) and beat the
    one-sided projections on the mixed-prompt X&Y case** (where the toy already shows the
    cell route cuts X only while gated-DoM couples X~Y).
  CONFIRMS: discovered mode ≈ planted route, r_eff≈1, and cut is selective.
  FALSIFIES the *method* (not the idea): SVD of M^B fails to recover a planted rank-1
  route, or the discovered cut is no more selective than the linear baseline even when the
  ground-truth route is bilinear → the averaging/recovery pipeline is broken, fix before B.

### Stage B — the real test: induction on gpt2-small (where microscopic FRA is diffuse)

This is the decisive one. The persistence campaign established that the gpt2 induction
edge is microscopically diffuse (Spearman≈0.03, top-1 cell ≈1.4% of oracle). The
mean-field question: **does the behavior-averaged M^B nonetheless condense, and does
cutting the condensate beat a matched linear steer?**

Setup: reuse `fra/` + the persistence harness. B = the induction A→B copy edge across the
many carrier contexts already built for persistence (LOCATE/HELD-OUT split preserved, so
condensate is *discovered on LOCATE, tested on HELD-OUT* — no leakage). Average per-pair
cells (Eq. 2) over LOCATE instances of the edge → M^B (use the diff/active feature support
to keep it tractable, or run in the model-diff basis per §3.2). SVD → condensate modes.

Pre-registered Stage-B metrics:
  - **M_cond (does it condense?)**: r_eff and captured-mass f(r) of M^B on HELD-OUT
    instances. Pre-registered threshold: **condensate ⇔ r_eff ≤ 5 and f(3) ≥ 0.8.**
  - **M_behav (does the condensate carry the edge?)**: replace the score by the rank-r
    condensate (or cut it) and measure HELD-OUT copy-prob removal vs the **causal oracle**
    (the position-aware edge ablation, already in the harness). Pre-registered:
    condensate-cut recovers **≥ 70%** of oracle removal at r ≤ 3 — contrast with the
    microscopic top-1 cell's ~1.4%. This single number is the headline: *does collective
    coarse-graining rescue what microscopic cells could not?*
  - **M_lin (is it bilinear or linear collapse?)**: §2b contrast — condensate cut vs
    (a) best query-side projection, (b) best key-side projection, (c) best matched
    difference-of-means residual steer, all tuned to equal HELD-OUT removal. Metric =
    collateral (A's benign-use copy-prob; other queries→B-key attention). Pre-registered:
    **irreducibly bilinear ⇔ condensate collateral < 0.5× the best matched linear steer's
    at equal removal.**

CONFIRMS the idea: M_cond passes (r_eff small) AND M_behav ≥ 70% AND M_lin shows bilinear
advantage. This would be a real, modest result: *behavior-averaging condenses a
microscopically-diffuse QK edge into a few interpretable feature-modes, and editing them
is more selective than linear steering.*
FALSIFIES (each a distinct, informative negative):
  - **No condensate**: r_eff large / f(3) low on HELD-OUT → diffuseness survives averaging
    (drift wins) → mean-field adds nothing on dense gpt2; points to needing B1/B2 basis
    cleanup first (and is consistent with the persistence drift finding).
  - **Condenses but doesn't carry the edge**: r_eff small but M_behav low → the average is
    low-rank but causally inert (the FRA-magnitude≠causal-effect problem recurs at the
    mode level).
  - **Linear collapse**: cut works but a one-sided projection matches it at equal
    collateral → reduces to linear steering, the standard deflation.

## 5. Honest verdict

**Is it sound?** Yes — the minimal version (§1.4: a behavior-conditioned low-rank
approximation of the averaged feature-resolved interaction M^B, with a rank-1 bilinear
edit along its top mode) is well-posed and computable, and the two decisive questions
(§2) are sharp and falsifiable. The "mean-field/condensate" framing is, in its honest
minimal form, an averaged-matrix SVD; the genuine self-consistent mean-field (§1.4) is a
real and interesting open question but is *not required* and should not be oversold.

**Will it condense?** Uncertain, leaning *no on dense gpt2 + flat SAE*. The single biggest
reason it might fail is **feature drift**: the persistence campaign's central finding is
that the SAE assigns *different* (q,k) features to the "same" association across contexts.
Drift means ⟨u_q u_k^T⟩_B is a sum of many near-orthogonal templates → M^B is high-rank →
no condensate. The very diffuseness that motivates the idea has a known *cause* (drift)
that also predicts the average stays high-rank. The idea works best precisely where it is
least needed (induction, one stable template) and is most likely to fail where the
diffuseness is worst (drift-heavy behaviors). That is the central tension.

**Will it be bilinear or collapse to linear?** Conditional on a condensate existing, this
is genuinely up for grabs and is the more interesting axis. There is a real mechanism for
an irreducibly-bilinear win (selectively cutting a (query-template ∧ key-template)
conjunction that no one-sided residual edit can isolate — §2b). But the campaign's track
record is that such wins deflate to matched linear steers, and a condensate with a
sink/positional key-direction collapses to a query-side bias (§2b case 1). Prior is
roughly 50/50 *if* it condenses.

**Is it materially new?** Mostly no on machinery (low-rank bilinear W_QK editing is
Elhage et al. 2021; linear steering baselines are the refusal/ITI literature), modestly
yes on framing: the *feature-resolved, behavior-averaged* condensate is interpretable in a
way raw QK-eigenvectors are not, and the explicit falsifiable question — "does
behavior-averaging collapse a *provably diffuse* microscopic cell structure into a few
modes, and is editing them more selective than linear?" — is new and worth answering. It
is an increment, not a new mechanism.

**Worth running?** Yes, but **only as the two-hour synthetic + the cheap gpt2 read-out**,
and only because the first measurement is nearly free and decisive. The whole bet reduces
to one cheap number: **the effective rank r_eff (and f(3)) of M^B on held-out induction
instances**, which needs only the per-pair cells the persistence harness already computes
— no new training, no GPU sweep. Run that first.
  - If r_eff is large → stop; mean-field does not rescue dense-gpt2 diffuseness (publish as
    a clean negative reinforcing the drift story, and route to B1/B2 to clean the basis
    first, then re-measure r_eff as a *diagnostic of whether the cleanup helped*).
  - If r_eff is small → proceed to M_behav (does the mode carry the edge?) and only then
    the expensive M_lin selectivity contrast.

**The cheapest test of the biggest failure reason** (drift → high rank): compute r_eff of
M^B directly from the *existing* persistence per-pair cell data, both as the signed mean
and in an aligned/covariance frame (§1.2). If even the aligned frame gives large r_eff, the
condensate does not exist on this substrate and the idea is dead on dense gpt2 without a
basis cleanup — at the cost of an afternoon and zero GPU.

**One-line bottom line:** sound and cheap to falsify, but with an *a priori* lean toward
"no condensate on dense gpt2" (drift predicts the average stays high-rank); run the free
r_eff measurement before anything else.
