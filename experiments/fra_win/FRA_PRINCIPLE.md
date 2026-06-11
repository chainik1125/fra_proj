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

---

## Better predictors: the screen is a CONJUNCTION of criteria (`s2`,`s3`, fig9)

CCF alone over-passes — on gpt2-small every content edge scores 0.90–0.99, and IOI (which we *know*
FRA can't remove, `j13`) sails through. CCF tests only criterion #1 (edge-routed); it is blind to
**redundancy**, which is exactly why IOI slips past. The accurate predictor is a conjunction of cheap
criteria, each catching a different failure mode:

| criterion | cheap test | catches (non-wins) |
|---|---|---|
| **CCF** — edge-routed? | non-sink content-pair mass / total (1 fwd pass) | weight-baked sleeper (output-direction, not an edge); BOS (positional) |
| **LBNR-R** — load-bearing & non-redundant? | cut the edge across top-k heads; `R = 1 − B(cut)/B(intact)` | **IOI** (R=**−0.39**: behavior gets *stronger*, backup attn 0.34→0.69); **many-shot jailbreak** (distributed, no single edge) |
| ~~reuse~~ — common marginal? | corpus firing-rate(key-feature) vs conjunction | proposed magnitude predictor — **FAILED** (`s5`, see below) |

> **FRA wins ⟺ CCF-high AND LBNR-R-high.** LBNR is the gate CCF was missing.

**Validation — the conjunction is correct on every case:**

| behavior | edge-routed (CCF) | load-bearing (LBNR-R) | measured A | CCF∧LBNR |
|---|:---:|:---:|:---:|:---:|
| induction (gpt2) | ✓ 0.99 | ✓ 0.98 | ~15× | **WIN** ✓ |
| **copy-suppression** (gpt2, `s3`) | ✓ 0.98 | ✓ 0.95 | **22.7×** | **WIN** ✓ |
| retrieval (gemma) | ✓ | ✓ (cut flips) | ~16× | **WIN** ✓ |
| in-context backdoor | ✓ | ✓ | ~25× | **WIN** ✓ |
| **IOI name-mover** (gpt2) | ✓ 0.95 | ✗ **−0.39** (backups) | ~1× | no-win ✓ |
| many-shot jailbreak | ✓-ish | ✗ (distributed) | ~1× | no-win ✓ |
| weight-baked sleeper | ✗ (output-dir) | — | ~1× | no-win ✓ |
| BOS / positional | ✗ 0.00 | — | n/a | no-win ✓ |

**The copy-suppression win (`s3`, fig9 right) — a genuinely new result.** L10H7 (the canonical
copy-suppression head, whose QK the literature characterized but never decomposed into feature-pairs):
an FRA-QK edit **selectively disables copy-suppression for one target token** (lion: logit 16.4→18.4,
matching the edge-cut oracle 17.9) at **22.7× less collateral** on other tokens (0.022 vs 0.503 for
head-ablation), and the same feature-pair edit **transfers** to a new context (16.8→18.9). This is the
first FRA win the *screen predicted in advance* — CCF flagged it, LBNR confirmed it load-bearing, Tier-2
delivered A≫1.

---

## Extending the table (`s4`,`s5`, fig10): LBNR does the discriminating; reuse fails

Tier-2 (collateral-advantage A) run on the remaining screened-in candidates, and the reuse-ratio
measured properly. Two clean conclusions:

**1. Of the 4 CCF-passers, LBNR screens out 3 — only copy-suppression is a real win.**

| candidate | CCF | LBNR-R | Tier-2 A | verdict |
|---|---:|---:|---:|---|
| copy-suppression L10H7 | 0.98 | **+0.95** | **22.7×** | **WIN** |
| IOI name-mover | 0.95 | −0.39 | ~1× | out (backups compensate) |
| binding / coreference | 0.91 | −0.89 | *spurious* | out (weak/distributed edge) |
| greater-than | 0.97 | +0.11 | — | out (MLP-routed comparison) |

This is the sharpest possible answer to "is the screen working?": **CCF passed all four; LBNR did the
actual discrimination**, keeping exactly the one candidate that wins. Two cautions the data forced:
- **A is only meaningful at matched on-target removal.** Binding's apparent "A=5133×" is an artifact —
  FRA achieved ~zero on-target effect (P=0.015→0.027, because the edge isn't load-bearing), so its
  collateral is trivially ~0 and the ratio is meaningless. **LBNR-pass is a prerequisite for A to be
  interpretable.**
- **greater-than** shows CCF/LBNR test the *edge*, not the *behavior of interest*: the QK edge reads
  the year (cutting it dents the >-fraction only 0.98→0.87), but the `>` is computed in MLP, so the
  behavior isn't QK-controllable even though the edge is a content conjunction.

**2. The reuse-ratio magnitude predictor FAILED (honest negative).** `s5`: across induction /
copy-suppression / IOI / binding, the conjunction firing-rate is ~0.000 in a 59-sentence corpus
(REUSE=∞, degenerate — the `r4c` failure again), and the dominant non-sink key-features are *rare*
(rate 0.02–0.05), not common — contradicting the "common marginal" premise. And the measured A=22.7×
came from FRA-**vs-head-ablation** (advantage = head-ablation hits *every* token the head touches), not
from a commonly-reused key-feature being suppressed — so reuse-ratio predicts the wrong baseline.
**Conclusion: we have a validated win/lose predictor (CCF∧LBNR) but NOT a magnitude predictor.** The
magnitude of A more plausibly tracks *head breadth* (how many positions/contexts the head is active at
that FRA leaves untouched) — left for future work.
