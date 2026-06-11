# When is FRA the right tool? A principle and a measurable score

*A general rule for predicting FRA-QK's behavioral usefulness — and a one-forward-pass metric
that flags it, the way the induction score flags an induction head.*

---

## The principle

> **FRA-QK beats every direction/steering baseline exactly when the behavioral target is a
> *bilinearly-gated attention edge*: behavior that fires only on the CONJUNCTION of a query-feature
> and a key-feature, where each feature on its own is common (reused across many positions/contexts).**

Why this is the boundary — three intervention families, three failure modes:

| method | what it can target | fails when |
|---|---|---|
| **direction** (DoM / ActAdd / CAA / weight-SVD) | a single residual *direction* (a marginal feature) | the culprit feature is **common** → suppressing it hits every other use → high collateral |
| **attention/position patch** | a specific edge at an **absolute position** | the same pattern recurs at **other positions/contexts** → doesn't transfer |
| **FRA-QK** | a query-feature **×** key-feature **pair** (a rank-1 W_QK edit) | — fills the gap: specific (a pair, not a marginal) **and** content-addressed (transfers wherever both features co-occur) |

A direction can push a query *or* bias a key; it cannot express *"change attention from
queries-with-feature-A to keys-with-feature-B."* That is inherently bilinear, and it is precisely
FRA's unique expressive power. So FRA wins iff the behavior lives in that bilinear cell — **and
nowhere else.** This is why it lost on the weight-baked sleeper (a rare trigger *token* → a specific
direction is already surgical → no bilinear gap) and on many-shot jailbreak (direction-routed task
vector, no separable edge), and won on induction, the in-context backdoor, and retrieval (all
conjunctions of two *common* features).

**The induction score is the special case.** An induction head implements one bilinear gate:
*(query token = X) × (key = the token right after the previous X)*. The induction score measures the
strength of exactly that one feature-pair edge. FRA generalizes "is this head running a specific
bilinear pattern?" from the single hard-coded same-token pair to *arbitrary* SAE feature-pairs — and
the metric below generalizes the induction score to "is this behavior gated by a feature-pair
conjunction that no single direction can capture?"

---

## The metric — a validated two-tier diagnostic

I tried hard to make a single one-forward-pass scalar (the analog of "compute the induction score and
read it off"). **The naive intrinsic scalars do NOT work**, and the reason is itself instructive (see
"What failed"). What *does* work is a two-tier diagnostic: a cheap one-pass **screen** for the necessary
condition, then a **confirmatory** behavioral measurement for sufficiency — exactly the structure of
"induction score flags a candidate head, then you verify it does induction."

### Tier 1 — cheap screen (one forward pass): Content-Conjunction Fraction (CCF)

The *necessary* condition is that the edge is a genuine **content-feature × content-feature**
conjunction, not a positional/attention-sink edge (which a position-patch or RoPE-direction handles
trivially). Measure, on the target edge (q,k) over the target head(s):

> **CCF = (Σ|FRA mass| from non-sink feature-pairs) / (Σ|FRA mass| on the edge)**

where a "sink" feature is one that fires in >50% of contexts in a small probe corpus (the ubiquitous
high-norm GemmaScope attention-sink features). Validated (`r5`, gemma-2-2b, fig7 left):

| edge | CCF | |
|---|---|---|
| induction (2nd-occ → after-1st-occ) | **0.349** | canonical FRA-shaped → highest |
| distractor / delimiter / retrieval (content edges) | 0.19–0.24 | pass the screen |
| **BOS (attention sink)** · **induction→BOS** | **0.000** | killed — pure positional |

CCF cleanly zeroes the pure positional/sink edges and ranks the canonical induction edge highest.
It is **necessary but not sufficient**: it can't separate retrieval (a win) from the delimiter edge
(both content edges ~0.2). That separation needs Tier 2.

### Tier 2 — confirmatory (probe + two interventions): collateral advantage A

The *sufficient* condition adds: the content must be **reused legitimately** (so a direction incurs
collateral) **and** the behavior must be **routed through the edge** (so cutting it removes the
behavior). Both show up in one measured ratio, at matched on-target removal:

> **A = collateral(best direction at matched removal) / collateral(FRA edge-edit at matched removal)**

measured on a held-out set where the behavior's *components* legitimately recur. Validated across every
case we have (fig7 right):

| behavior | A | verdict |
|---|---|---|
| retrieval (gemma, `r2`) | ~16× (FRA legit-collateral 0.16→0.15 vs direction 0.16→**0.00**) | **FRA wins** |
| induction (gpt2, `fig1`) | ~15× (0.18 vs 2.66 nats) | **FRA wins** |
| in-context backdoor (gpt2/gemma) | ~25× (11–90×) | **FRA wins** |
| weight-baked sleeper | ~1× (DoM/SVD already surgical — rare trigger token) | no win |
| many-shot jailbreak | ~1× (direction-routed; no clean removal by anything) | no win |

A≫1 ⟺ the bilinear cell of the principle; A≈1 ⟺ a direction already suffices. The induction-score
analogy is exact: **Tier 1 (CCF) is the cheap intrinsic flag; Tier 2 (A) is the behavioral
confirmation** — and the induction edge passes both, recovering the induction score as the special case.

### What failed (and the lesson)

- **Footprint-CSG** = F_dir/F_pair (marginal-feature footprint over the rest of *one* sequence): ~1.3
  for *every* edge (`r3`). Flaw: F_dir ⊇ F_pair by set inclusion, so the ratio just counts "how many
  key-features this query-feature pairs with" — near-constant, behavior-independent.
- **Spectral conjunctivity** 1−σ₁²/‖M‖² of the edge's FRA matrix: null and *backwards* (BOS scored
  highest, `r4`).
- **Naive corpus-reuse** rate(key-feat)/rate(conjunction): degenerate — the top pair on every edge was
  the *same* ubiquitous attention-sink feature (15887, firing rate 1.00), so every edge looked
  identical (`r4b`).

**The lesson:** on Gemma, a handful of attention-sink features dominate FRA magnitude on *all* edges
and silently contaminate any intrinsic metric that ranks by raw |contribution|. You must exclude them
(the CCF "sink mask") before anything discriminates — and even then, the part that distinguishes a
*win* from a *no-win* (reuse + routing) is genuinely a cross-context / behavioral property, not a
single-forward-pass number. Hence the two-tier diagnostic rather than one scalar.

---

## Applying the screen to candidate behaviors (`s1`, gpt2-small, fig8)

Tier-1 CCF screen run over the brainstormed candidates, anchored by induction (known FRA-win, high)
and BOS-attention (positional, zero):

| behavior | CCF | edge attn | screen |
|---|---:|---:|---|
| induction `[anchor+]` | 0.989 | 0.92 | reference |
| **copy-suppression L10H7** `[cand#2]` | **0.984** | **0.92** | **pass — at the anchor** |
| greater-than `[reasoning]` | 0.965 | 0.83 | pass |
| IOI name-mover `[circuit-attr]` | 0.951 | 0.62 | pass |
| binding / coreference `[reasoning]` | 0.905 | 0.38 | pass (weak edge) |
| BOS-attention `[anchor−]` | **0.000** | 0.98 | **screened out** |

**Reading.** (1) The screen *works*: BOS has the highest raw attention (0.98) but CCF=0.000 — a pure
positional/sink edge, correctly killed. (2) On gpt2-small *every* canonical circuit behavior passes
(0.90–0.99): induction, copy-suppression, IOI, even greater-than's attention-to-the-year are all
content×content conjunctions. CCF sits near 1.0 (not spread like Gemma's 0.19–0.35) because the
gpt2-small-res-jb SAEs lack GemmaScope's dominant attention-sink features — so here the screen's role
is purely to separate content edges from positional edges, which it does cleanly.

**Verdict — who's worth a Tier-2 test:** all four candidates clear the necessary condition, so
sufficiency (redundancy / competitiveness / content-reuse) decides — read off edge-attention + known
circuit redundancy:
- **copy-suppression (L10H7) — top candidate.** Single canonical head (non-redundant, unlike IOI's
  backups), strongest edge (0.92, = induction), and it is *exactly* "FRA as the missing QK piece": the
  copy-suppression literature characterized the head's behavior but never decomposed its QK into
  feature-pairs. Best bet for a clean FRA-QK win.
- **IOI name-mover** — passes Tier-1 but the *known* Tier-2 killer applies: backup name-movers
  compensate (our `j13` negative). The canonical illustration that CCF is necessary, not sufficient.
- **greater-than** — passes (the attend-to-year edge is a real content conjunction), but the comparison
  itself is MLP-computed; FRA-QK would only control the *attending*, not the >-operation.
- **binding/coreference** — passes but the weakest edge (attn 0.38) → more distributed → lower priority.
