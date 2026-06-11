# THEORY_HIERARCHY — broad×broad cell-cutting, the regression protocol, and the routed-fraction dial

*Theory agent (Fable), fra_hierarchy campaign, 2026-06-11. Builds on the converged FRA theory
(`experiments/fra_win/THEORY.md`), the confirmed claims (`experiments/fra_win/RESULTS_SUMMARY.md`),
the em_svd direction-removal result (EM is removable as an MLP weight-diff direction; attention not
implicated — the double dissociation), and the multitrigger detector≠payload result. References:
SynthSAEBench (arXiv 2602.14687) for ground-truth hierarchical/superposed feature generation; Toy
Models of Superposition (Elhage et al. 2022); edge attribution patching (EAP) / path patching /
attribution-graph circuit tracing for the routed-fraction measurement.*

---

## 0. TL;DR and pre-registered claims

Every confirmed FRA win cuts a (specific-feature × single-token) cell — a bigram statistic, the
lowest rung of the feature hierarchy. This document develops the theory of cutting a
**(broad × broad)** cell — e.g. (misaligned-persona × domain) — and a **regression protocol** that
*fits* which cells to cut from a behavioral target, then specifies a hand-built synthetic model that
tests both against ground truth. The central honest tension: FRA only cuts **attention-score-routed**
links; a concept→concept link carried by OV/MLP/a residual direction (the em_svd EM result) is
invisible to it. We therefore make the **routed fraction α** of a behavior a *measured quantity*
(pattern-freeze → EAP → circuit tracing), plant it as a *tunable knob* in the synthetic, and frame
FRA's contribution as the **missing QK piece of circuit attribution** — the layer that decomposes a
located attention edge into content-conjunction cells and intervenes one rung finer than head/edge
ablation — rather than as a competitor to linear steering.

**Pre-registered predictions (the synthetic must adjudicate each):**

- **P1 (cuttability).** At α=1 (fully score-routed), cutting the single (persona × domain_X) cell
  removes ≥90% of the domain-X misalignment gap while changing persona behavior in other domains,
  domain-X aligned processing, and the persona's aligned (style) function each by ≤5% relative.
- **P2 (per-edge vs per-position).** On *mixed-domain* prompts (X and Y content in the same
  context), every per-position linear interventon — including the strongest content-gated
  persona-direction removal — couples its domain-X removal to domain-Y collateral ≈ 1:1, while the
  cell-edit decouples them. This is the broad-regime replacement for conjunction-rarity: selectivity
  from **edge-level factorization**, not from rarity.
- **P3 (superposition).** Linear-removal collateral grows with feature overlap ρ (slope ≥ α_h on
  hierarchy children, ~ρ̄ on unrelated features, × number of residual readers); the oracle-gated
  cell-edit's collateral is ρ-independent (softmax redistribution only); the SAE-gated cell-edit's
  collateral tracks the SAE's gate false-fire rate, not ρ directly. A(ρ) = collateral ratio rises
  with ρ on top of a ρ-independent factorization floor A₀ set by mixed-prompt evals.
- **P4 (magnitude law, lifted).** A grows ≈ linearly in K, the number of preserved sibling domains
  (the broad-regime analog of the confirmed A≈N sibling curve), because
  reuse(persona marginal|eval) ∝ K while reuse(persona×domain_X conjunction|eval) ≈ 0.
- **P5 (routed-fraction recovery).** EAP-style edge attribution on the toy recovers the planted α
  (|α̂ − α| ≤ 0.05 across α ∈ {0, .25, .5, .75, 1}); the one-shot pattern-freeze test recovers it
  too (residual behavior = (1−α)·Δ).
- **P6 (honest failure + tool selection).** Removal achieved by the cell-edit tracks α (it removes
  the score-routed share and *no more*); at α=0 the regression returns *infeasible* (no cell-mask
  clears the removal threshold under the collateral cap) while persona-direction removal still
  removes 100% — the double dissociation, now as a continuous curve telling you *when each tool
  wins*.
- **P7 (regression recovery).** Given the target "remove domain-X misalignment, collateral ≤ κ" the
  sparse regression recovers exactly the planted (P, D_X) cell when the parent latent exists (H1,
  support precision = recall = 1.0), and recovers the child×child **block** when only children exist
  (H2) — whose rank-1 factorization recovers the *hidden parents* (cosine ≥ 0.9 to the planted
  parent duals): the protocol can climb the hierarchy from below.
- **P8 (finer than the edge).** At matched removal of the attention-routed share, the cell-cut's
  collateral is strictly below head-ablation's (which kills persona→all-domains, and in the
  shared-head variant the head's legit traffic) and below position-oracle edge-ablation's in the
  shared-head variant — and the cell-cut needs no runtime position classifier because the
  conjunction *is* the classifier, weight-baked.

---

## 1. Theory of broad×broad cell-cutting

### 1.1 Formalizing breadth and height

Fix the deployment/eval distribution. For SAE feature μ with activation u^μ_t ≥ 0:

- **Breadth** B(μ) = P_eval[u^μ_t > 0]: the firing measure over token-positions. A *specific*
  feature ("token is 'cloud'") has B ~ 1/|V|; a *broad* feature ("medical content", "misaligned
  persona") has B ~ 10⁻²–10⁻¹. Breadth is a property of the support, orthogonal to the
  content-vs-role distinction of win-checklist clause 4.
- **Height** is relational: a parent P with children {c_1..c_N} in two senses that must be kept
  apart because they fail differently:
  - **Activation hierarchy** (SynthSAEBench-style co-occurrence): supp(u^{c_i}) ⊆ supp(u^P);
    u^P ≈ OR/max over children (P fires whenever any child does, and possibly *tonically* — across
    a whole span — while children are *phasic*).
  - **Geometric hierarchy** (the absorption geometry): f_{c_i} = α_h f_P + √(1−α_h²) g_i with
    g_i ⊥ f_P; every child direction *contains* the parent component with cosine α_h.

Two dictionary regimes, which the theory and the synthetic both must cover:

- **H1 — represented parent**: the SAE dictionary contains a latent for P. A broad×broad link is
  then literally ONE cell (P, D): coupling ω_{P,D} = (f_P W_Q)·(f_D W_K)/√d_h, edit identical in
  form to the bigram case.
- **H2 — virtual parent** (feature splitting/absorption, the *expected* regime for abstract concepts
  in real SAEs): no parent latent; the concept exists only as the family {c_i}, glued by the shared
  geometric component α_h f_P. If the head's routing reads the parent direction, the cell matrix
  restricted to the families is a **rank-1 block**: ω_{c_i, d_j} ≈ (α_h^P α_h^D) σ + O(interference).
  The broad cell is then a family×family **union edit** (THEORY.md's set algebra), and *finding* it
  is exactly what the regression protocol (§3) is for.

### 1.2 What cutting a broad×broad cell does, mechanistically

The bigram case removes one entry of the model's effective bigram table: the conjunction support is
a rare token pair, and the magnitude law A ≈ reuse(marginal)/reuse(conjunction) is huge because the
denominator is tiny. The broad case is different in kind:

- The edit is the same object — a support-gated rank-1 update of W^h_QK,
  ΔW ∝ −c·ω_{P,D_X}·e_P^T e_{D_X} on {(q,k): u^P_q>0 ∧ u^{D_X}_k>0} — but the support is now a
  **concept co-occurrence**, not a bigram. You are editing one entry of the model's *concept-level
  routing table*: specific×specific edits the data; broad×broad edits the **program**.
- The conjunction support is LARGE (B(P)·B(D_X)·co-occurrence factor), so selectivity can no longer
  come from rarity. It comes from **factorization**: the thing to remove is exactly one
  (concept_q × concept_k) product term, and the comparison class is the *other cells sharing an
  endpoint* — (P × D_Y), (other-queries × D_X) — plus the endpoints' non-attention functions.

**The per-edge vs per-position separation (the broad-regime sharpening of the row/column theorem).**
THEORY.md's expressivity result: a gated steer's score effect varies over keys only through
⟨x_k, v⟩ for one fixed pullback direction v = ΔW_Q W_K^T — it can imitate a row or column of the
bilinear form, never a cell. In the broad regime this bites through a new, decisive scenario:
**mixed-concept contexts**. Let domain-X and domain-Y content coexist in one prompt and let the
persona query position q need its attention row *split* — suppressed on X-keys, preserved on Y-keys:

- Any per-position intervention (steer at q, gated however cleverly — including "fire only when
  domain-X is somewhere in the context") modifies x_q once; its score row changes by ⟨x_k, v⟩,
  ONE linear functional of key content. Exact family separation requires ⟨x_k,v⟩ ∝ u^{D_X}(x_k) —
  impossible (u is a thresholded nonlinearity), and the best linear surrogate leaks onto Y-keys with
  magnitude ~ the families' residual-space overlap (≥ ρ̄ under coherence ρ̄). And the steer is still
  deposited in the residual where every other consumer reads it.
- The cell edit subtracts c·u^P_q·u^{D_X}_k·ω per **edge**: identically zero on Y-keys (oracle
  gates), graded by the actual conjunction strength, confined to the head's softmax.

So: in the bigram regime, conjunction *rarity* made the steer pay collateral; in the broad regime,
conjunction rarity is gone and what remains is the strictly structural advantage — **a per-position
gate cannot split a single attention row between two key families; a per-edge cell edit can.** This
predicts the clean qualitative signature P2: on mixed prompts, gated-steer removal of X and
collateral on Y move together ≈ 1:1, while the cell edit decouples them.

**Magnitude law, lifted (P4).** A ≈ reuse(marginal|eval)/reuse(conjunction|eval) survives verbatim;
what changes is what fills the slots. Marginal: the persona fires in all K domains plus its aligned
uses — numerator ∝ K. Conjunction: legitimate (persona×domain_X) reuse — for a *misaligned* persona
link, ≈ 0 by stipulation; the floor on collateral is implementation, not semantics (softmax
redistribution + gate false-fires). Hence A(K) ∝ K — the broad-regime sibling of the confirmed
A ≈ N differential-cell curve. The catastrophic regime is unchanged: if the "persona" latent is
really a *generic role* feature (assistant-ness, formality) whose conjunction with the domain has
legitimate recurring use, A collapses toward the ~1.9× floor. Clause 4 reinterpreted: **what must be
distinctive is the content identity of the conjunction, not the narrowness of its endpoints** — a
broad abstract feature is an admissible discriminating endpoint iff it is content-defined
(semantics) rather than role/position-defined, and iff the *pairing* has no legitimate eval-support.
This reinterpretation is exactly the hypothesis the synthetic isolates: does clause 4 survive
breadth, or does a large conjunction support per se destroy the win?

### 1.3 Transport × gating: the exact statement of when it works

Any context-conditional behavior of the form "persona-state ∧ domain-content-in-context →
misaligned generation" factors into:

- **(T) Transport**: domain content must reach the generation position. Between positions there is
  only attention — transport is always attention-mediated *somewhere*.
- **(G) Gating**: the persona-conditionality. Two mutually exclusive mechanisms per unit of
  behavior:
  - **G-score**: the persona changes WHAT is attended — persona-query × domain-key score mass; the
    pattern differs persona-on vs persona-off. *This is the FRA-cuttable channel.*
  - **G-post**: the pattern is persona-invariant; the persona changes what is *done* with the
    transported content — OV cross-terms, MLP product gates, a residual direction trigger (the
    em_svd EM mechanism; the multitrigger payload). *FRA-blind by construction.*

Define **α = the fraction of the behavioral gap carried by G-score** (formally: the share of the
persona-conditional behavior delta mediated by pattern changes, §2.1). Then the exact statement:

> A broad×broad cell-cut selectively removes a concept→concept link iff (i) α is high — the link is
> score-routed (lifted clauses 1–3: edge-routed at concept level, load-bearing/non-redundant across
> the carrying heads, directly consumed); (ii) the conjunction (P × D_X) has ≈ no legitimate
> eval-support while its endpoints do, with P content-defined, not role-defined (lifted clause 4,
> reinterpreted per §1.2); (iii) the dictionary can express the link — a parent latent (H1) or an
> enumerable, regression-recoverable family block (H2) — with enough coverage to clear the softmax
> margin on every carrying head (lifted clause 5, where the new reach risk is **feature
> splitting/absorption**, not granularity-vs-K).

Failure modes, lifted: (a) α ≈ 0 — direction/OV/MLP-routed generalization (em_svd EM; the honest
expected outcome for some real behaviors; the tool of choice is then DoM/SVD and FRA must *say so*,
§3.3); (b) value-side persona modulation — persona changes the OV content, pattern unchanged (a
G-post subcase; the Q/K-vs-V split in §2.1 detects it); (c) splitting/absorption — no parent latent
and an unenumerable family (reach; mitigated by the regression's group/rank-1 structure, tested in
H2); (d) distributed routing — many heads each carry a sliver (the many-shot-jailbreak mode; shows
up as a high-α but high-κ removability frontier).

### 1.4 The superposition connection, made precise

Toy-Models setting: n features, d < n dims, dictionary F (rows f_i, unit norm), coherence
ρ̄ = E|⟨f_i, f_j⟩| ~ 1/√d for random dictionaries; hierarchy adds the *structured* overlap
⟨f_{c_i}, f_P⟩ = α_h ≫ ρ̄.

**Linear concept removal pays interference collateral.** Project out the persona:
x ← x(I − f̂_P f̂_P^T) (gated wherever you like). Every co-active feature j is corrupted by
u^j⟨f_j, f̂_P⟩ in *every* downstream reader at the gated positions:

- hierarchy children: corruption amplitude α_h (large, structural — removing the parent deletes the
  "category-ness" component of every child: the mechanistic content of "all-or-nothing");
- unrelated features: amplitude ~ρ̄, so squared collateral grows ~ρ̄² per feature per reader, summed
  over the gate's support and over all residual consumers (Q, K, V, MLP, later layers — the
  broadcast factor).

So linear-removal collateral is **monotonically increasing in superposition**, with two terms: a
hierarchy term (α_h, ρ-independent but large) and an interference term (ρ̄-driven).

**The score-space cell-edit is immune — with one honest caveat.** The oracle-gated cell edit never
writes into the residual: its only causal channel is the edited head's softmax, its only intrinsic
collateral the **softmax redistribution** of the freed attention mass within edited rows (measured,
not assumed — it lands ∝ the surviving pattern, partially on the sink, partially on Y-keys; this is
the toy analog of the reach/matched-removal bookkeeping). This cost does not scale with ρ. The
caveat: with a *learned* SAE, superposition re-enters through the **gates** — demixing error makes
û^P fire falsely on overlapping features, and the edit fires on non-target edges with rate
ε_FP(ρ). So the precise claim is: cell-edit collateral is independent of *geometric* overlap given
exact gates, and degrades only through the SAE's demixing error — which a good SAE suppresses far
below raw overlap (that being the point of an SAE). The synthetic separates these cleanly: oracle
gates (flat A(ρ) curve) vs learned-SAE gates (slow rise tracking measured ε_FP).

**Two advantages with different ρ-dependence (the shape of A(ρ), pre-registered).**

- The **factorization advantage** (per-edge vs per-position, §1.2) is structural and ρ-independent:
  it sets a floor A₀ > 1 on *mixed-prompt* evals at every ρ, and ≈ vanishes on pure-single-domain
  evals at ρ→0 (where a well-gated steer is genuinely adequate — report this honestly).
- The **interference advantage** grows with ρ: both the steer's residual-broadcast collateral and
  its within-row score leakage scale with overlap.

Prediction: A(ρ) ≈ A₀(eval mix) + slope·ρ̄, decomposable by running pure-domain and mixed-domain
eval slices separately. **Superposition is exactly the regime where score-space editing pulls away
from residual-space editing — but the mixed-context floor means the cell edit is not merely a
high-ρ trick.**

---

## 2. Measuring attention-routedness; FRA as the missing QK piece of circuit tracing

### 2.1 The routed-fraction dial α: a measurement ladder

The §1.3 risk ("is this link score-routed at all?") must be **measured before intervening**, never
assumed. The ladder, cheap → structural:

1. **Probes (cheap, weakest).** A linear probe reading the concept (persona, domain) from the
   residual shows the concept is a readable *direction* — necessary for the direction-routed story,
   but not causal and not sufficient for anything: both G-score and G-post behaviors have probeable
   concepts. Use only to locate candidate features/layers.
2. **Pattern-freeze (cheap, global, one intervention — run FIRST).** Run the behavior condition
   (persona-on, or the finetuned/steered model) while freezing all attention patterns to their
   persona-off (or base-model) values. The surviving behavior fraction is (1−α̂_global) directly:
   if the behavior survives pattern-freezing, it is not pattern-routed and FRA is dead on arrival.
   One forward pass per condition; decisive as a necessary condition.
3. **EAP / path patching (the headline dial).** Edge attribution patching between matched
   clean/corrupt pairs (persona-on vs persona-off, same tokens) scores every edge of the
   computational graph; define the **attention-routed fraction**
   α̂_EAP = Σ_{attention-pattern-mediated edges} |attribution| / Σ_{all edges} |attribution|
   on the behavior metric. Crucially, split *within* each head: patch the scores S (Q/K-path,
   pattern-mediated — the FRA-addressable share) separately from the values V (V-path — attention
   transport with G-post gating, NOT FRA-addressable). Linearized (gradient) EAP must be verified by
   real activation patching on the top edges (the standard EAP caveat).
4. **Circuit tracing / attribution graphs / transcoders (structural).** Build the attribution graph
   for the behavior; measure how much of the persona→domain path flows through attention nodes vs
   MLP/transcoder nodes. Closest structural match to "which mechanism carries the link", but note:
   standard attribution graphs *freeze attention patterns* and treat them as fixed linear weights —
   they inherit exactly the blind spot FRA fills (next section).

### 2.2 The methodological thesis: FRA completes circuit attribution on the QK side

Circuit tracing (EAP, path patching, attribution graphs) answers **WHERE**: which heads/edges/nodes
mediate the behavior. But it treats the QK score as a black box: the attention pattern enters as a
fixed linear coefficient; nothing in the pipeline says **WHICH (query-content × key-content)
conjunction produced the score**, and the finest intervention it supports is ablate-the-head/edge —
which kills *all* content routed through it. FRA is precisely the missing layer:

> **EAP localizes the edge; FRA resolves the edge.** S^h[q,k] = Σ_{μν} u^μ_q u^ν_k ω^h_{μν} is the
> exact decomposition of the located edge into content conjunctions; the cell identifies WHY the
> head attends (which concept pairing), and the cell-edit intervenes one rung finer than the edge —
> cutting the (persona × domain_X) conjunction while every other conjunction routed through the
> same head, including (persona × domain_Y) and the head's legit traffic, is untouched.

The intervention granularity ladder (coarse → fine), which the synthetic instantiates as baselines:
**direction removal** (model-wide, all readers) ⊃ **head ablation** (all routes through the head) ⊃
**position-oracle edge ablation** (one head, chosen key positions; needs a runtime content
classifier per prompt) ⊃ **cell edit** (one conjunction; content-addressed and weight-baked — the
conjunction IS the classifier, no runtime oracle needed; the multitrigger lesson, generalized).

The pipeline this buys (the paper-worthy contribution):
**(1)** EAP → routed fraction α̂ + the carrying heads/edges (go/no-go: α̂ low → hand the behavior to
DoM/SVD and stop); **(2)** FRA on the carrying heads → cell decomposition of the edge score →
candidate broad×broad conjunctions; **(3)** regression (§3) → the minimal cell-set under a
collateral cap; **(4)** cut, and show lower collateral than head/edge ablation at matched removal.
FRA's role is *improving circuit tracing's intervention layer*, not competing with steering —
steering remains the right tool for the (1−α) share, and the pipeline says so quantitatively.

---

## 3. The regression protocol: fitting which cells to cut

### 3.1 Formulation

Candidate set C = {(h, μ, ν)}: head × query-feature × key-feature (prefiltered to cells with
nonzero conjunction co-occurrence on the train prompts; in real models additionally to
EAP-localized heads). Parametrize the edit by α ∈ [0,1]^|C|:
S^h[q,k] ← S^h[q,k] − α_c·û^μ_q·û^ν_k·ω^h_{μν} on the conjunction support.

Functionals, all evaluated on a **train split disjoint from the eval split** (the val-extract-leak
rule — extraction/selection data never touches eval, and the fitted α is frozen before eval):

- Target: T(α) = behavioral removal on the target set (e.g. Δ misaligned-logit on persona×domain-X
  prompts), with full-removal value Δ.
- Preserves: P_m(α) = absolute behavioral change on M preserve sets (persona×domain-Y, domain-X
  aligned task, persona aligned function, generic LM quality).

First-order model: T(α) ≈ g^Tα with
g_c = Σ_{q,k} (∂B/∂S^h[q,k])·û^μ_q û^ν_k ω^h_{μν} — i.e. the design matrix is **attribution
patching at cell granularity**: one backward pass scores all |C| cells (the regression and §2's EAP
are the same estimator at different granularity; this is the formal sense in which the protocol
extends circuit attribution into score space). Likewise rows h_{m,c} for each preserve functional.

The fit (equivalent forms; the LP is the primary):

- **Minimal cut (LP/LASSO):** min ‖α‖₁ s.t. g^Tα ≥ τ·Δ (removal threshold), Hα ≤ κ (collateral
  caps), 0 ≤ α ≤ 1.
- **Per-prompt regression (support identification):** stack per-prompt behavioral deltas y_i against
  per-prompt cell contributions G_{i,c}; LASSO min ‖y − Gα‖² + λ‖α‖₁ + preserve penalties. The
  cross-prompt heterogeneity is what identifies *which* cell carries the behavior (cells only
  co-fire on some prompts).
- **Group/rank-structured variants (the hierarchy tools):** group-LASSO with groups = hypothesized
  feature families ("find the broad cell" = "select the right group"); or treat α as a matrix over
  (query-features × key-features) per head and penalize trace-norm — the rank-1 solution's singular
  vectors are the *discovered* parent features (H2; prediction P7).

**Two-stage discipline (the softmax is not linear):** (i) screen all cells with the one-backward
linear attribution; (ii) exact joint-ablation refit on the top-S candidates (S ~ 512): re-measure
T, P under the actual edit, refit / matching-pursuit until the verified edit meets the constraints;
(iii) report only on the held-out eval split. Identifiability caveat: cells whose activations are
collinear on train (children of one parent co-firing) make G ill-conditioned — plain LASSO picks an
arbitrary representative; this is not a bug but the H2 signature, and the group/trace-norm variants
are the correct response.

### 3.2 The protocol subsumes top-K and differential cells

- **Top-K** is the solution with no preserve constraints and ≈ orthogonal cell effects: the LASSO
  path adds cells in order of |g_c| — exactly magnitude ranking. (With correlated effects top-K is
  *wrong* and the regression strictly better — the broad/H2 regime is precisely the correlated case.)
- **Differential cells** (target-edge top-M minus sibling-edge top-M) are the solution when the
  preserve set is the sibling edge: shared cells get penalized through Hα ≤ κ, unique cells
  selected first. The regression strictly generalizes: it can put *partial* weight on shared cells
  up to the collateral budget, buying back the reach that the hard set-difference paid (the 58%→33%
  cost). Falsifiable retro-prediction (optional E10): re-running the confirmed N-sibling experiment
  with the constrained regression recovers ≥ the 7.9× differential-cell separability at better reach.

### 3.3 What the regression buys

1. **Non-obvious, multi-cell, abstract edits:** in H2 there is no dominant single cell — the link is
   an N_P×N_D block of small child×child cells; top-1 misses it, the regression assembles it, and
   the rank-1 structure of the fitted mask *names the hidden parents* (climbing the hierarchy from
   below).
2. **The removability frontier R\*(κ):** max removal s.t. collateral ≤ κ, α ∈ [0,1]^C — a curve, not
   a number. R\*(κ) ≈ 1 at small κ: score-routed and separable. R\* low at all κ: the **infeasibility
   certificate** — either not score-routed (check against pattern-freeze α̂: if α̂ is also low, hand
   to DoM/SVD) or dictionary reach failure (α̂ high but R\* low: splitting/absorption — try group/H2
   machinery, a different SAE, or family unions). R\*(κ) is the cell-granularity complement of
   §2.1's α̂, and the pair (α̂, R\*) is the complete go/no-go diagnostic.
3. **Principled operating points:** the matched-removal comparison against baselines is defined *by
   construction* (τ, κ are explicit), instead of post-hoc c-tuning.

---

## 4. The synthetic model spec (the first build — hand-built, CPU-runnable)

### 4.0 Design overview and the α-spine

One hand-built attention model on SynthSAEBench-style ground-truth features, with a persona→domain
generalization mechanism whose routing is **planted with a tunable split**: fraction α through an
attention score route (G-score), fraction (1−α) through a direct product-readout path (G-post, the
MLP/direction analog), with the *total* behavioral gap held constant across α. Everything is
analytically transparent: we set the dictionary, W_Q, W_K, W_OV, the readouts, and therefore know
every ground-truth ω, every conjunction support, and the true α. The build validates, in order:
the routed-fraction measurement (P5), selective cuttability of a broad×broad cell (P1, P2, P8),
the superposition sweep (P3), the magnitude-law lift (P4), the regression's recovery and its honest
failure (P6, P7). A trained-model variant is the follow-up (§4.7), not the first build.

### 4.1 Data-generating process

Ground-truth dictionary (n = 256 features, d_model = 64 default; both swept):

| block | count | role |
|---|---|---|
| P (persona parent) | 1 | broad, tonic on persona sequences |
| p_1..p_8 (persona children) | 8 | phasic at persona-word positions |
| D_1..D_4 (domain parents, K=4) | 4 | phasic at domain-word positions; tonic copy at final position |
| d_{k,1..16} (domain children) | 64 | phasic domain words |
| C (copy/legit feature, V1 only) | 1 | legit traffic for the shared-head variant |
| filler | ~178 | background |

Geometry: base directions i.i.d. uniform on the sphere in R^d (realized coherence ρ̄ ~ 1/√d —
**the superposition knob is d_model ∈ {256, 128, 64, 32, 16}**; report realized mean |cos|).
Hierarchy: f_child = α_h·f_parent + √(1−α_h²)·g, g random ⊥ parent, α_h = 0.6 (both persona and
domain families). Activation hierarchy: child fires ⇒ parent fires.

Sequences (T = 32, position 0 = BOS/sink): per sequence, z_per ~ Bernoulli(0.5); active domains:
one domain w.p. 0.5, two domains w.p. 0.5 (**mixed prompts are a first-class eval slice, not an
afterthought**). Per position t ≥ 1: a domain child from an active domain w.p. 0.25, a persona
child w.p. 0.15·z_per, else 1–3 filler features; active magnitudes u ~ U[0.8, 1.2]. Tonic: u^P = 1
at *every* position if z_per = 1 (persona is a style state — realistic for persona features);
tonic u^{D_k} = 1 at the final position T for each active domain (the "what is this prompt about"
summary that the direct path reads). Residual: x_t = Σ_active u·f + ε, ε ~ N(0, 0.02²I). Splits:
20k train / 5k eval sequences, disjoint; all fitting/selection on train only.

### 4.2 The hand-built model

**Route construction** (exact rank-1 routes; row convention): pick orthonormal {e_c} ⊂ R^{d_head}
(d_head = 16). For each route c = (a_c, b_c, σ_c): W_Q += √σ_c·â_c⊗e_c, W_K += √σ_c·b̂_c⊗e_c, so
(xW_Q)(x'W_K)^T = Σ_c σ_c⟨x, â_c⟩⟨x', b̂_c⟩ exactly. Read vectors are regularized duals:
â_i = row i of (F^TF + λI)^{-1}F^T-style pseudo-inverse, so ⟨x, â_i⟩ ≈ u^i with O(ρ̄) interference —
the interference is *kept*, it is the honest superposition effect. Sink: score bias S[q, 0] += β₀.

- **H_mis** (the generalization head): K mis-routes (â_P, b̂_{D_k}, σ_mis) for k = 1..K — persona
  query attends to ALL domain content (the generalization mechanism) — plus the sink bias.
  V0: nothing else (clean pedagogy). V1: + one legit copy route (â_C, b̂_C, σ_copy) through the
  *same* head (realism: heads multiplex; head-ablation now has legit collateral).
- **H_coh** (aligned processing): K coherence routes (â_{D_k}, b̂_{D_k}, σ_coh) + sink bias —
  domain-topical queries (incl. the tonic final position) attend same-domain keys. This is the
  domain's normal aligned pipeline *through attention*, so key-side interventions have a real
  aligned cost.
- **OV:** W_OV^mis = Σ_{k,j} â_{d_kj}⊗v^mis_k; W_OV^coh = Σ_{k,j} â_{d_kj}⊗v^align_k
  (payload vectors orthonormal). Heads' outputs y^mis_T, y^coh_T at the final position.

**Readouts at the final position (the behavior), with the α-knob:**

- misaligned logit, domain k: **m_k = ⟨y^mis_T, v^mis_k⟩ + σ_dir·û^P(x_T)·û^{D_k}(x_T)**
  — first term: the attention path (G-score: persona route's OV transport; ≈ 0 when persona off —
  the head parks on the sink); second term: the direct path (G-post: a hand-built bilinear "MLP
  neuron" reading tonic persona × tonic domain at the final position; pattern-invariant by
  construction — the toy analog of the em_svd direction/MLP mechanism).
  **Calibration (E1): set (σ_mis, σ_dir) so the attention path contributes α·Δ and the direct path
  (1−α)·Δ of a fixed total persona-conditional gap Δm_k, for α ∈ {0, 0.25, 0.5, 0.75, 1}.**
- aligned task: a_k = ⟨y^coh_T, v^align_k⟩ (domain processing; persona-independent).
- aligned persona function: s = û^P(x_T) (style readout; what ungated persona-removal must break).

Starting magnitudes (to be re-calibrated against target attention masses, not trusted blindly):
β₀ = 5, σ_mis ≈ 6 at α = 1 (persona-on: ~95% of H_mis mass on domain keys; persona-off: ~90% on
sink), σ_coh ≈ 6, activations as in §4.1. Calibration targets: persona-on→domain-key mass 0.6–0.95,
persona-off→sink ≥ 0.85, Δm_k S/N ≥ 20 over the ε noise.

**Planted target:** the (P × D_1) link ("domain X"). Everything else is preserve-set.

### 4.3 Interventions (the baseline ladder — all eight, none strawmanned)

1. **FRA cell edit, oracle gates:** S^mis[q,k] −= c·u^P_q·u^{D_1}_k·ω_{P,D_1}, c = 1 (cancels the
   planted route up to interference).
2. **FRA cell edit, learned-SAE gates** (û from §4.6's SAE).
3. **Regression-fitted cell mask** (§3 protocol end-to-end, fitted on train, frozen, applied on eval).
4. **Ungated persona-direction removal** (DoM/em_svd analog): x ← x(I − f̂_P f̂_P^T) everywhere.
5. **Context-gated persona removal** (strongest fair linear baseline): same projection, only at
   query positions of sequences containing domain-X content (oracle gate — maximally generous).
6. **Key-side domain-X removal:** project f̂_{D_1} (and children) out at key positions on persona
   sequences.
7. **Head ablation of H_mis** (the EAP-granularity intervention — what circuit tracing alone offers).
8. **Position-oracle edge ablation:** zero S^mis[T, k] for k ∈ domain-X-child positions (the
   multitrigger-oracle analog; needs a runtime position classifier — fair and strong; its expected
   weaknesses: V1 same-head legit traffic through those keys, and no weight-baked deployability).

### 4.4 Metrics and the figures that prove each claim

Eval slices: pure-X, pure-Y, mixed-XY persona sequences + persona-off controls. Core metrics:
R_X = fractional removal of Δm_X; preserves: |δm_Y|/m_Y, |δa_X|/a_X, |δa_Y|/a_Y, |δs|/s;
collateral C = max preserve deviation; A = C_baseline/C_cell at matched removal (τ = 0.9).

- **Fig 1 — the α-sweep (headline; P5 + P6).** x: planted α. Curves: α̂ from pattern-freeze; α̂_EAP
  (with the S-vs-V split); removal by cell-edit; removal by direction-removal; regression R*(κ=5%).
  PASS: |α̂ − α| ≤ 0.05; cell-edit removal tracks α (±5); direction-removal flat at ~100%; at α = 0
  the regression certifies infeasibility while DoM still works. *This figure is the tool-selection
  result: it says when FRA wins, when steering wins, and that we can tell in advance.*
- **Fig 2 — selectivity table at α = 1 (P1, P8).** 8 interventions × 5 metrics, V0 and V1. PASS:
  cell-edit row ≥ 90% removal with all preserves ≤ 5%; head ablation kills m_Y (~100% collateral on
  the persona's other-domain behavior) and, in V1, the copy task; baselines 4–6 each violate ≥ 1
  preserve by > 50%.
- **Fig 3 — the mixed-prompt decisive case (P2).** On mixed-XY sequences: scatter of (X-removal,
  Y-collateral) as intervention strength sweeps, per method. PASS: gated steer traces ≈ the
  diagonal (coupled 1:1); cell edit traces the y≈0 axis; quantify the cell edit's softmax
  redistribution onto Y-keys (report it as the honest intrinsic cost, expected small if the sink
  absorbs the freed mass — and *tune nothing to hide it*).
- **Fig 4 — A(ρ) (P3).** x: realized ρ̄ (d_model sweep); y: collateral per method at matched
  removal, pure-domain and mixed slices separately. PASS: linear methods rise with ρ̄ (children
  corrupted at α_h regardless — plot the hierarchy term separately); oracle-cell flat; SAE-cell
  tracks measured gate-FP; mixed-slice floor A₀ > 1 at all ρ̄.
- **Fig 5 — regression recovery (P7).** Left (H1): fitted α heatmap over the cell grid, planted
  (P, D_1) marked; PASS: support precision = recall = 1.0 at λ chosen on train, eval-verified
  removal/collateral within the caps. Right (H2: parent latents withheld from the dictionary —
  queries keyed by persona children): block recovery (block-mass fraction ≥ 0.9), and rank-1 SVD of
  the fitted mask vs hidden parents; PASS: cos ≥ 0.9 both sides. Plus the group-LASSO group-selection
  result.
- **Fig 6 — A(K) (P4).** K ∈ {2, 4, 8} domains: A vs K. PASS: ≈ linear growth (the lifted
  magnitude law), mirroring the confirmed A ≈ N sibling curve.

### 4.5 Experiment list and compute

| ID | what | spec | compute |
|---|---|---|---|
| E1 | build + calibration | hit the §4.2 calibration targets; verify Δm_k gaps, aligned tasks, hierarchy consistency; unit-test ω against planted σ | CPU, minutes |
| E2 | α-sweep + routed-fraction validation | α ∈ {0,.25,.5,.75,1}; pattern-freeze, EAP (gradient + verify by real patching), cell-edit removal, DoM, R*(κ) → Fig 1 | CPU, ~1 h |
| E3 | selectivity table + mixed-prompt case | α = 1, V0 + V1, all 8 interventions → Figs 2–3 | CPU, ~1 h |
| E4 | superposition sweep | d ∈ {256,128,64,32,16} × {oracle, SAE} gates → Fig 4 | CPU/1 small GPU, ~2 h |
| E5 | K-sweep | K ∈ {2,4,8} → Fig 6 | CPU, ~30 m |
| E6 | regression H1 | screen 65,536 cells by one-backward attribution; exact-ablate top-512; LP/LASSO fit; eval-verify → Fig 5L | CPU, ~1 h |
| E7 | regression H2 | parent withheld; plain LASSO vs group-LASSO vs trace-norm; rank-1 parent discovery → Fig 5R | CPU, ~1 h |
| E8 | SAE-in-the-loop | train L1 SAE (width 512) on x; measure splitting/absorption (does a P latent exist?), gate-FP rate; rerun E3/E4 with SAE gates | 1 small GPU, ~2 h |
| E9 | trained variant (follow-up) | §4.7 | 1 GPU, later |
| E10 | optional retro-prediction | constrained regression on the old N-sibling task: ≥ 7.9× separability at better reach | later |

Everything through E8 runs on CPU or one small GPU in an afternoon; per the compute rule it goes on
a RunPod (rs-* named) — no local compute, parse-gated jobs, results pushed under an HF prefix.

### 4.6 Notes for the build agent

- **Match the repo's conventions:** row-vector convention; the cell edit acts on S pre-softmax;
  c = 1 is the faithful scale; matched-removal operating points via τ; train/eval discipline per
  the feedback memory (extraction never touches eval; fitted masks frozen before eval).
- The per-cell screening is analytic (u^μ_q u^ν_k ω_{μν} from one forward); only the exact-ablation
  refit needs re-forwarding — batch cells.
- Report the realized (not just planted) ω matrix: interference puts O(ρ̄) mass on off-target
  couplings; the oracle cell-edit at c = 1 leaves this residue — measure the floor it sets.
- Log the softmax-redistribution collateral explicitly per eval slice; never tune β₀ after seeing it.

### 4.7 The trained-head follow-up (phase 2, brief)

Train a 2-layer attention+MLP toy transformer on the synthetic task (persona-conditional misaligned
target outputs in domain contexts; aligned targets otherwise). The routing split is now *emergent*:
measure α̂ (pattern-freeze + EAP), run the full pipeline, and test the transported prediction —
the cell-cut works iff α̂ is high, with the regression's R*(κ) agreeing. This is the bridge between
the hand-built model and the real-LLM hunt: same protocol, no planted answer key.

### 4.8 Red-team of the synthetic itself (what would make the result fake, and the mitigations)

- **Rigging risk 1 — dedicated head:** in V0 the mis-route owns H_mis, so head-ablation looks
  artificially safe on aligned tasks. V1 (shared copy route) is therefore mandatory in Fig 2.
- **Rigging risk 2 — analysis basis = construction basis:** the oracle gates use the ground-truth
  features the routes were built from. E8 (learned SAE) and E7 (withheld parents) break this;
  the trained variant (E9) removes it entirely.
- **Rigging risk 3 — the product readout:** m_k's persona gate means ungated DoM trivially achieves
  100% removal; the verdict metric is *selectivity at matched removal*, never raw removal — keep the
  framing honest (DoM is supposed to win removal; it cannot win selectivity).
- **Softmax redistribution:** the freed attention mass must land somewhere; if it boosts m_Y the
  cell-edit has a real intrinsic collateral — measure, report, and let the sink design be fixed
  *before* E3 (no post-hoc tuning).
- **What would falsify the PI's hypothesis:** if at α = 1 and high ρ̄ the cell-edit's
  (redistribution + gate-FP) collateral matches or exceeds the gated steer's — i.e. breadth per se
  (a large conjunction support) destroys the conjunction advantage — then clause 4 does *not*
  survive the lift to broad endpoints, and the campaign's premise fails in the cleanest possible
  setting. The synthetic is designed so this outcome is visible, not engineered away.

---

## 5. Hand-off to the real-LLM evaluator

### 5.1 The decision battery (run in this order; cheap → expensive)

1. **Pattern-freeze (first, always).** Behavior-on condition (finetuned/steered/prompted persona)
   with attention patterns frozen from the behavior-off condition. Surviving fraction = (1−α̂).
   If the behavior survives, FRA is out for that behavior; hand to DoM/SVD. (For finetune-induced
   behaviors: run the finetuned weights with base-model patterns, the FRA-diff machinery's natural
   complement.)
2. **EAP routed fraction with the S-vs-V split.** α̂_EAP per §2.1, separating Q/K-path (cuttable)
   from V-path (transport with post-gating, not cuttable). Verify gradient-EAP with real patching
   on the top edges.
3. **FRA-diff localization.** Base vs behavior-model Δ(score mass) on persona-feature → domain-
   feature edges at the EAP-located heads (the existing FRA-diff machinery applies verbatim: rank
   cells by Δ contribution between the two models).
4. **Regression R\*(κ).** The §3 protocol with held-out collateral caps. The (α̂, R\*) pair is the
   verdict: high/high → cut; high/low → dictionary reach (try group/H2 machinery, family unions, a
   different SAE width); low/* → direction-routed, use DoM/SVD, FRA's contribution is the *negative
   certificate*.
5. **Magnitude forecast before intervening:** estimate reuse(persona marginal|eval) /
   reuse(persona×domain conjunction|eval) on the eval distribution → predicted A; pre-register it.

### 5.2 Candidate concept→concept behaviors, ranked by prior P(score-routed)

1. **Sycophancy × user-opinion-content (best bet).** The behavior is definitionally
   context-retrieval: mirroring requires *reading the user's stated opinion* and transporting it to
   generation — and the sycophancy-conditionality plausibly modulates *what gets attended* (opinion
   spans). Target cut: (sycophancy/agreeable-persona-query × opinion-statement-key) cells; preserve:
   opinion *comprehension* (the model can still report what the user said) and agreement when
   warranted.
2. **Format/instruction-persona × domain.** Instruction-following is in-context (the instruction
   sits in the prompt; following it at generation ≈ attending to it — induction-adjacent, known
   attention mechanisms). Cut (format-instruction × domain-X) to stop a format generalizing to one
   domain; crisp metrics.
3. **In-context backdoor × broad trigger class.** Already in the win-class at the narrow rung;
   the hierarchy test is a *class-level* trigger (any member of a semantic family) — directly tests
   H2/family-union machinery on a real model.
4. **Refusal-persona × topic.** The refusal *payload* is famously a direction (direction-routed —
   expect the DoM side of the dissociation); but the topic-conditional *triggering* may be
   score-routed (harm-assessment requires attending request content). Run the battery on the
   trigger, not the payload; the multitrigger detector≠payload lesson says these dissociate.
5. **EM persona × domain (the flagship — and the riskiest).** The em_svd prior says the EM *payload*
   is an MLP weight-diff direction (α low for the payload). The live question is the
   **domain-conditional component**: *why is the model misaligned about THIS domain here* — the
   selection of domain content for misaligned processing may be score-routed even when the
   misalignment payload is a direction. Operational metric: under a (persona × domain_X) cell-cut,
   does domain-X-specific EM drop while *generic* EM (domain-free misaligned statements) persists?
   Predict partial α; the pattern-freeze test (1) decides cheaply whether to proceed. A *negative*
   here is still a campaign result: it makes the em_svd dissociation feature-resolved.

### 5.3 What "success" means for the campaign

The deliverable is not "FRA beats steering on EM". It is the validated pipeline: **measure α →
resolve the routed share into a broad×broad conjunction → regression-fit the minimal cell-set →
cut strictly finer than head/edge ablation, with the (α̂, R\*) pair correctly routing each behavior
to the right tool (cell-cut vs DoM/SVD).** The synthetic (§4) proves the pipeline sound where ground
truth is known; the evaluator's job is to find one real behavior with high (α̂, R\*) and demonstrate
the selectivity table on it — and to report low-α̂ behaviors as *correct negative certificates*, not
failures.

---

## 6. Relation to prior results (one paragraph)

This direction lifts each confirmed element one rung: the cell edit (bigram → concept routing-table
entry), clause 4 (distinctive *token* → distinctive *content-defined concept*, the hypothesis under
test), the magnitude law (A ≈ N siblings → A ≈ K domains), differential cells (hand-built set
difference → constrained regression), and the sleeper bridge (weight-baked → DoM / in-context → FRA
becomes the continuous, *measured* α-dial with EAP as the meter). The multitrigger lesson
(detector ≠ payload; the attention pattern is the high-precision lever) reappears as the
trigger-vs-payload split in §5.2; the em_svd result (EM payload = MLP direction) is the α≈0
endpoint our synthetic plants on purpose. The superposition story supplies the *why now*: linear
concept-removal pays interference collateral that grows with overlap and with reader count, while a
score-space conjunction edit pays only gate error — broad, superposed, hierarchical features are
exactly where the bilinear structure of attention stops being a curiosity and starts being the only
surgical tool.
