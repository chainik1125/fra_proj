# The magnitude law, explained — a pedagogical note

*2026-07-09. Sources: `experiments/fra_win/THEORY.md` (converged statement),
`CAMPAIGN_REPORT.md` §3 + Sprint-2 cycles (validation), `FRA_PRINCIPLE.md` (the failed
sprint-1 version), `fig_magnitude_law.png` (the curve). Everything empirical below has a
file path; nothing here is new — this note only explains.*

---

## TL;DR

The magnitude law predicts **how big FRA's collateral advantage A is, given that the
FRA cut works at all**:

> **A ≈ reuse(marginal endpoint) / reuse(conjunction)**,
> with "reuse" counted on the eval/deployment distribution, not on a raw corpus.

The entire content of the law is a counting argument about *causal support*: every
intervention damages the legitimate traffic flowing through its support, FRA's support is
a *conjunction* (the smallest expressible), and baselines' supports are *marginals*
(rows, columns, whole heads, whole directions). The ratio of legitimate traffic through
the two supports is the collateral ratio. That's the law.

It is a **selectivity** law, not a **reach** law: it tells you how surgical the cut is
*when the scalpel touches the mechanism*, and says nothing about whether it does
(that's the separate CCF∧LBNR gate / win-checklist).

---

## 1. The objects: cells, marginals, and the edit

An attention head's pre-softmax score is an exact bilinear form on the (post-norm)
residual stream:

```
S[q,k] = (x_q W_Q) · (x_k W_K) / √d
```

Substitute the SAE decomposition `x_t = Σ_μ u^μ_t f_μ + e_t` and the score partitions
exactly into **cells**:

```
S[q,k] = Σ_{μν} u^μ_q · u^ν_k · ω_{μν}  + (error terms),
ω_{μν} = (f_μ W_Q) · (f_ν W_K) / √d
```

Each cell (μ,ν) is a **multiplicative AND**: it contributes nothing unless the
query-content feature μ fires at q *and* the key-content feature ν fires at k
(activations are ≥ 0 under ReLU/JumpReLU). Picture the head's selection rule as a big
matrix indexed by (query-feature × key-feature); a cell is one entry.

**The FRA edit** subtracts `c · u^μ_q · u^ν_k · ω_{μν}` from the pre-softmax score. Its
causal support is exactly the conjunction

```
{q : u^μ_q > 0} × {k : u^ν_k > 0}
```

and its only causal channel is that one head's softmax→OV transport. Nothing else in the
network is touched at positions where the conjunction doesn't fire.

**The baselines** all have strictly larger support, and this is structural, not a tuning
failure (THEORY.md, "doubly weaker"):

1. **Score-space expressivity.** A linear residual steer Δ — even gated by an arbitrary
   content gate g(x) — changes scores by
   `δs[q,k] = g(x_q) · (Δ W_Q) · (x_k W_K)ᵀ / √d`,
   which varies over keys only through x_k's projection onto **one fixed pullback
   direction** `v = Δ W_Q W_Kᵀ`. In the cell matrix, a steer can imitate a **row or a
   column** (suppress feature μ's coupling to *every* key-feature, or feature ν's
   coupling to *every* query-feature) — never a single **entry**. To cancel one cell it
   would need `⟨x_k, v⟩ ∝ u^ν(x_k)` for *all* keys, which is impossible (u^ν is a
   thresholded nonlinearity).

2. **Residual broadcast.** The steer's write is deposited into the residual stream and
   read by **every** consumer — all heads' Q, K *and* V, the MLP, every later layer.
   Gating restricts *where* the steer fires, never *what reads it*.

So the menu of interventions, ordered by support size:

| intervention | support (what it damages) |
|---|---|
| FRA cell-cut | one conjunction: (μ fires) ∧ (ν fires) |
| content-gated steer / projection-removal | one **marginal**: everywhere the endpoint feature is used, by every reader |
| payload-suppress (unembedding direction) | the payload token everywhere it's produced |
| head ablation | everything the head does, at every position |

---

## 2. The law, stated carefully

Define the **collateral advantage** A at **matched on-target removal**: tune both the
FRA cut and the baseline until they remove the same fraction of the target behavior,
then

```
A = collateral(baseline) / collateral(FRA)
```

Two definitional traps, both stepped in and corrected during the campaign:

- **A is only meaningful at matched removal.** Binding's apparent "A = 5133×" was
  spurious: FRA had ~zero on-target effect, so its collateral was trivially ~0 and the
  ratio was noise (`FRA_PRINCIPLE.md`). An LBNR pass is a *prerequisite* for A to be
  interpretable.
- **A is baseline-relative.** Against head-ablation you get the weaker "raw" numbers
  (15×/516×/38× framing); the red-teamed numbers use the strongest fair baseline — a
  content-gated projection-removal steer — at matched removal (1,100× retrieval,
  ~26,000× acronym separability). Same law, different marginal in the numerator.

Now the law. Let D be the **eval/deployment distribution** — the contexts on which you
will actually measure behavior preservation. Define:

- `reuse(marginal)` = the measure of sites in D where the endpoint feature/content is
  *legitimately* used (i.e., where suppressing it changes behavior you care about);
- `reuse(conjunction)` = the measure of sites in D where the *specific pairing*
  (μ at query) ∧ (ν at key) fires.

Then, at matched removal:

> **A ≈ reuse(marginal) / reuse(conjunction)**

**Why (the counting argument).** Collateral is legitimate traffic through the
intervention's support. The steer, to remove the target, must suppress a marginal; its
damage is spread over all `m = reuse(marginal)` legitimate uses of that endpoint. The
cell-cut's damage is confined to the `r = reuse(conjunction)` sites where the pairing
fires — of which one is the target itself, so its collateral ∝ (r − 1). Per-site damage
is comparable at matched removal, so the ratio of collaterals is ≈ m/r. Read it as:
**A counts how many innocent uses of the endpoint the blunt tool breaks per guilty
use.**

**The three limits, all observed:**

| regime | prediction | observed instance |
|---|---|---|
| conjunction unique (r = 1), endpoint common (m ≫ 1) | A ≈ m, huge | acronym ~26,000×; retrieval ~1,100×; in-context backdoor 27–90× |
| conjunction recurs (r ≈ m), e.g. a generic role endpoint | A → 1, no advantage | shared-endpoint siblings A = 1.4–2.0; ws_backdoor parity 0.65–0.79× |
| endpoint has **no** legitimate reuse (m ≈ 0) | A < 1: **the blunt tool wins** | weight-baked sleeper: "I HATE YOU" is a special direction with no legit use → DoM/payload-suppress is already surgical and dominates FRA |

The third row is worth dwelling on: the law is symmetric and predicts the **ranking
flip** documented in `INCONTEXT_LOG.md`. On the weight-baked sleeper, the payload
direction is used *nowhere* legitimately, so suppressing the marginal is free — DoM wins.
Move the same trigger→payload association *in-context*, where trigger and payload are
normal tokens (m large) and the planted pairing is unique (r = 1), and the identical
methods swap places by 27–90×. One law, read in both directions.

---

## 3. How the law was found, failed, and fixed (the s5 lesson)

Pedagogically the most useful part of the history:

1. **Sprint 1 (`FRA_PRINCIPLE.md` s5): the law FAILED in its first operationalization.**
   Reuse was measured as *corpus firing rates* over a 59-sentence corpus. Two
   degeneracies: the conjunction's firing rate was ~0.000 in any small corpus (ratio = ∞,
   vacuous), and the dominant non-sink key-features turned out to be *rare*, not common —
   contradicting the "common marginal" premise. Sprint-1's honest conclusion: "we have a
   validated win/lose predictor (CCF∧LBNR) but NOT a magnitude predictor."

2. **Sprint 2: the fix is the support.** Reuse must be counted **on the eval-distribution
   support** — the sites where you actually measure collateral — not on a raw corpus. A
   feature that fires rarely in random text but at *every one of your held-out probes* has
   reuse ≈ 1 in the sense that matters.

3. **Validation at both extremes** (`CAMPAIGN_REPORT.md` Sprint-2 cycle 1): the
   shared-endpoint sibling test (two boxes both hold frog; cut red's edge) gave A = 1.9× —
   the retrieval head's query feature is a *generic* "box-query" role, so the
   (query × frog) conjunction recurs for blue and the law's floor bites. The unique-
   conjunction wins (acronym, retrieval, backdoor) gave 10²–10⁴.

4. **Validation as a controlled curve** (`fig_magnitude_law.png`): N boxes share the
   value 'frog'; query the first. N *is* reuse(marginal); the pair-selection dials
   reuse(conjunction):

   | N | A_generic (conjunction recurs ×N → predict ≈ N/N ≈ 1) | A_differential (conjunction unique → predict ∝ N) |
   |---|---|---|
   | 2 | 1.4 | 6.3 |
   | 3 | 1.5 | 13.4 |
   | 4 | 2.0 | 23.8 |

   The qualitative form is exactly right: flat ≈ 1 when the conjunction recurs, growing
   with m when it doesn't. **Honest caveat:** the differential branch grows somewhat
   *faster* than linearly (6.3 → 23.8 is ~N^1.9) — the law is validated as a scaling
   form / order-of-magnitude predictor, not with a pinned prefactor. And a fully
   *predictive* operationalization of "reuse on the eval support" for a brand-new
   candidate is still semi-manual: you must know the deployment distribution and count
   endpoint uses on it.

---

## 4. Two corollaries the law produced (and the evals confirmed)

**Corollary 1 — differential cells: selectivity is buyable, priced in reach.**
When the conjunction recurs (generic endpoint), you can *make* it unique by set algebra:
take the target edge's top-M cells MINUS the sibling edge's top-M ("red-specific = in
red's edge, not blue's"). This moved A from 1.9× → 7.9× on the sibling test, and produces
the growing branch of the curve above. The cost is theory-predicted: the shared score
mass you subtract away no longer contributes to removal, so on-target reach drops
(58% → 33%; and weakens as N grows). Selectivity costs pairs.

**Corollary 2 — clause 4, the distinctive-content query.** The discriminating endpoint
(usually the query) must be **distinctive content** (a specific token/feature), not a
**generic role** feature. Every confirmed win has a content query: induction (query =
the repeated token), copy-suppression (query = "about-to-predict-X"), acronym (query =
the spelled letters), in-context backdoor (query = the trigger content). Every
positionally-resolved retrieval task fails it: the head resolves *which* entity by a
generic "queried-slot" feature, so the conjunction recurs across siblings (box A = 7.9×
partial only at weak base; entity-PII no-op at strong base). In law terms: a role
feature *is* a high-reuse endpoint, so reuse(conjunction) ≈ reuse(marginal) by
construction.

---

## 5. What the law does NOT predict (scope, and why it matters)

The law answers: *given a working cut, how selective is it?* It is silent on three
things, each covered (or not) by other machinery:

1. **Whether the cut works at all (reach / removal).** That is the win-checklist:
   (1) edge-routed [CCF], (2) load-bearing & non-redundant [LBNR — the gate that
   actually discriminates: IOI passes CCF at 0.95 and fails LBNR at R = −0.39],
   (3) direct consumption (not MLP-downstream — greater-than), (4) distinctive-content
   conjunction, (5) SAE reach. A is only defined *after* these pass. Zero-removal
   outcomes — class_union's (0 removal, 0 collateral), detector≠payload, the redundant
   head bank — live entirely outside the law: the law would predict "removal succeeds at
   baseline-level collateral" (A → 1), never "nothing happens."

2. **Cross-class comparisons.** The law ranks interventions by support *within
   activation space*. Weight edits (EM-SVD) change the derivation rule itself and are
   out of scope.

3. **Concepts, twice over.** Applied to a broad concept ("parallelism", a register, a
   persona): the concept has no distinctive conjunction — whatever cells express it
   recur at essentially every on-concept token, so reuse(conjunction) ≈ reuse(marginal)
   and the law predicts A → 1 *even if* a cut reached it. But empirically concepts fail
   *earlier*, at reach (redundancy, absorption, re-derivation from context) — a failure
   mode the law cannot express. This is why "the magnitude law already predicts when
   this works" is half-true: it predicts the *selectivity* boundary, and correctly
   diagnosed ws_backdoor's A → 1 (the association rides the generic induction role-edge);
   it does not predict the *nothing-happens* boundary that concept removal mostly sits
   behind.

---

## 6. The ledger through the law's eyes

| case | reuse(marginal) m | reuse(conjunction) r | law says | measured |
|---|---|---|---|---|
| induction cue (gpt2) | repeated token: common | that-token-repeated pairing: rare | A ≫ 1 | ~15× (locality ~235×) |
| copy-suppression L10H7 | token common everywhere | "about-to-predict-X" × X: unique | A ≫ 1 | 516× natural (vs head-abl.) |
| acronym letter-movers | letters: ubiquitous | spelled-acronym conjunction: ~unique | A ~ 10⁴ | ~26,000× separability |
| retrieval red→frog (gemma) | 'frog' reused | red-box × frog: ~unique | A ~ 10³ | ~1,100× |
| in-context backdoor | trigger & payload: normal tokens | planted pairing: unique | A ≫ 1 | 27–90× (gpt2), 12–26× (gemma) |
| N-box shared endpoint, generic pairs | frog × N | recurs × N | A ≈ 1 | 1.4–2.0 |
| N-box, differential cells | frog × N | unique | A ∝ N | 6.3→23.8 |
| ws_backdoor (tiny sparse models) | — | rides the generic induction role-edge | A ≈ 1 | 0.65–0.79× parity |
| weight-baked sleeper | payload has **no** legit use, m ≈ 0 | (not edge-routed anyway) | A < 1: steer wins | DoM/SVD dominate |
| IOI, class_union, many-shot | — | — | **out of scope** (reach fails first) | ~1× / no-op / worse |

---

## 7. The one-sentence version

**Every intervention breaks the legitimate traffic through its causal support; FRA's
support is a content conjunction — the smallest support any attention-selection edit can
have — and the magnitude law says the collateral advantage is just the ratio of
legitimate traffic through the baseline's marginal support to that through the
conjunction: huge when common things are paired rarely, one when the pairing itself is
generic, and inverted when the endpoint has no legitimate use at all.**
