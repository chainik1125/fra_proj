---
author: Indranil Das
date: 2026-09-16
tags:
  - proposal
status: ready
---

> **STEP 0 PASSED (Sep 17).** Working conjunction found: password-recall format with ambiguous
> bigrams -- "The password for <A> <B> is <pay>." with each token shared across pairs (red fox->nine,
> blue fox->three, red owl->seven). Base gemma-2-2b: pair P(payload) 0.60-0.68 while either token with
> a novel partner is 0.03-0.08 (8-15x gap) -> a genuine AND. See [[ladder_log]] "B1 v3". The naive
> constructions failed (cross-concept copy ~0; single compound key leaks via the payload-adjacent
> token); the winning recipe = strong recall binding + token ambiguity. Proceed to the intervention.

## B1 -- Controlled conjunction anchor (the figure that proves single-feature cannot, FRA can)

This is the clean, cheap, controlled experiment that answers Dmitry's objection head-on before any
messy real case. It extends the semantic-filter machinery ([[semantic_filter_findings]]) but changes
the ONE thing that made single-feature tie us: the target is now a **conjunction**, not a single
concept. See [[plan_B]] B1.

### Mechanism note (why the naive plant is NOT conjunctive)

An induction head copies the token that *follows* a matched key. So a plant like
`The password is <A> <B> <PAYLOAD> ... Remember: <A>` copies whatever follows the matched A -- it is a
**single**-concept (A) trigger, which is exactly why the semantic-filter task tied single-feature. A
genuine cell needs the *attention itself* to depend on two DIFFERENT content features at the two
endpoints: query-content = A, key-content = B, with A != B. That is the thing to build and to verify,
not assume.

### Step 0 -- feasibility screen (forward passes only, cheap; gate before any intervention)

Two candidate ways to get a genuine A(query) x B(key) cell; screen both, keep whichever fires:

- **(i) Cross-concept semantic induction.** Plant `... the <B-word> holds <PAYLOAD> ... Remember the
  <A-word>:` where A and B are DIFFERENT but paired concepts (A = guard/captain/sentry; B =
  vault/gold/treasure). Question: does an A-query attend back to the B-key and copy PAYLOAD? The
  semantic filter proved *same*-concept transfer; this screens whether *cross*-concept A->B copy
  exists in base Gemma. If P(PAYLOAD) >= 0.2 for A-queries, we have the cell.
- **(ii) Two-route AND / compound key.** Plant the payload behind a two-token compound key `<A><B>`
  and query the compound; a single marginal token match should NOT reproduce it. Screen P(PAYLOAD)
  for compound-query vs A-only-query vs B-only-query.

Keep the construction where the payload fires on the pair (>= 0.2) but each marginal alone stays low
(< ~0.1). If neither fires, that is the boundary result and we fall back to the two-route removal
framing (below), which still beats single-feature on completeness-at-matched-collateral.

### Step 0 results (run Sep 16-17, gemma-2-2b base, H100; scripts/54-56)

Three screens, forward passes only (P of the target payload; NSEED 6-8):

**v1 (scripts/54)** -- the two naive constructions, both FAIL:
- (i) cross-concept induction (guard->vault etc.): pair = 0.002-0.004 (~0). Base gemma does NOT copy
  across concepts A->B. Dead.
- (ii) single compound key (red fox->nine, ONE mapping): pair = 0.72-0.87 (strong!) but B_only =
  0.40-0.51 -- the token adjacent to the payload leaks it via ordinary induction. Single-concept, not
  an AND.

**v2 (scripts/55)** -- ambiguous bigrams in a few-shot LIST format ("red fox: nine. blue fox: three.
red owl: seven."): the conjunction SIGNAL appears (pair > either novel-partner marginal in all 3) but
binding is far too weak: pair = 0.08-0.15. List format binds poorly.

**v3 (scripts/56)** -- WINNER: password-recall format + ambiguous bigrams. Body =
`... The password for <A> <B> is <pay>.` for three token-sharing pairs, then
`Remember the password for <A> <B>:`.

| group | pair P(pay) | A + novel partner | novel partner + B | conjunction |
|---|---:|---:|---:|:--:|
| red fox -> nine   | 0.601 | 0.045 | 0.038 | YES |
| iron gate -> four | 0.675 | 0.082 | 0.072 | YES |
| blue moon -> eight| 0.656 | 0.036 | 0.031 | YES |

Pair fires at ~0.6-0.68; either token with any other partner ~0.03-0.08 (8-15x gap). Neither token
alone carries the payload -- only the co-occurrence. This is the genuine AND. Recipe = strong recall
binding (from v1 ii) + token ambiguity (from v2).

### Concrete intervention protocol (implementation-ready, uses scripts/47 machinery)

Task (per group, e.g. red/fox/nine): plant the v3 body (3 token-sharing pairs) as context, then probe.

Probe families (all share the SAME planted context; only the final `for <qa> <qb>:` changes):
- **AB (the target)**: `red fox` -> emits `nine`. The behaviour to remove.
- **A-with-other (collateral A)**: `red owl` -> emits `seven`; `red cat` -> low. Must be PRESERVED.
- **other-with-B (collateral B)**: `blue fox` -> emits `three`; `gray fox` -> low. Must be PRESERVED.
- **payload-elsewhere (collateral pay)**: legit text ending in `nine` unrelated to the task. For the
  payload-suppress baseline's collateral.

Attribution + intervention, located on ONE AB prompt, applied to all probes:
- **FRA**: run the FRA tensor on the induction heads for the AB prompt; select the cells on the
  (query = final `fox` position) x (key = demo `fox`/`red fox` positions) block that carry the
  red-fox->nine retrieval; subtract those cells from attn scores (patch_fra, scripts/47). Content-
  addressed so it re-locates on each probe.
- **single SAE feature, ADDITIVE (Dmitry's baseline, exact form)**: attribution = harvest resid at the
  read layer on AB (payload present) vs a no-payload contrast, encode with the SAE, rank features by
  activation difference, take the TOP-1. Intervention = subtract c * (that one decoder vector) from the
  residual at all positions (additive, NOT projection, NOT directional, NOT multi-feature); sweep c on
  a grid; pick the point matching FRA's payload-removal. Do this for the best-of {feature ranked on A,
  feature ranked on B}. This feature necessarily fires on its whole set (all red-pairs or all fox-
  pairs), so it damages the corresponding collateral set.
- **DoM ablation**: mean(AB resid) - mean(no-payload resid), ablate that direction; grid c.
- **payload-suppress**: subtract the `nine` unembed direction at the last layer; grid.
- **oracle**: position-mask the exact AB query->key attention (removal ceiling; collateral undefined).

Metric: for each method, sweep its strength; record (payload-removal on AB, collateral KL on each of
A-with-other / other-with-B / payload-elsewhere). Read collateral AT MATCHED payload-removal (e.g. 50%,
70%). Report per-group, and the **WORST-case collateral over the three collateral sets** -- that is the
number that decides it. Pre-check per group: position-mask the pair attention; if payload drops, it is
attention-routed and in scope for FRA (expected yes).

### Experiment (once a construction passes Step 0)

Model: Gemma-2-2b (base). SAEs: GemmaScope 65k residual on the top ~25 induction heads. Same infra as
`g4_65k.py` / [[semantic_filter_findings]] (scripts/47).

Three held-out probe families:

- **AB (pair present)** -> model emits PAYLOAD. This is what we remove.
- **A-only** (concept A, no B) -> benign, NO payload. Collateral set 1.
- **B-only** (concept B, no A) -> benign, NO payload. Collateral set 2.

Interventions, all located on ONE AB prompt then applied held-out:

- **FRA**: cut the top cells of the (A-query-feature x B-key-feature) block only.
- **Single SAE feature, additive** (the baseline that beat us before): subtract the A feature; and
  separately subtract the B feature; grid-sweep the coefficient. Whichever single feature it removes,
  it fires on that feature's WHOLE set.
- **DoM ablation**: contrast AB vs no-payload, ablate the mean direction.
- **payload-suppress** (removes the payload feature -> hits all legit payload uses); **position/
  attention oracle** (removal ceiling).

Measure: suppression of PAYLOAD on held-out AB prompts, and **collateral (held-out KL) on the A-only
and B-only sets** (and a payload-only set for the payload-suppress baseline), read at matched
suppression. The A-only / B-only sets are the whole point: they are constructed to be exactly the
benign text a single-feature removal is forced to damage. Report **worst-case** over the collateral
sets -- that is where FRA's cell wins and every marginal baseline is forced to pay.

### Rationale

The password task in [[semantic_filter_findings]] is gated by a **single** concept (the trigger word),
so removing that one SAE feature does the same job -- which is why Dmitry's corrected eval shows
single-feature ties/beats FRA. A conjunction breaks that tie *by construction*:

- To kill AB, single-feature removal must remove the A feature (damages every A-only sentence) OR the
  B feature (damages every B-only sentence). It has no third option -- a single feature cannot see
  "A **and** B".
- The FRA cell (A-query x B-key) fires only where an A-token attends to a B-token, i.e. only on the
  conjunction, so it can suppress AB while leaving A-only and B-only untouched.

This is the one thing FRA can express that neither single-feature nor DoM can: a cell. The magnitude
law predicts the size of the advantage: A ~ reuse(A-only or B-only endpoint) / reuse(AB conjunction) --
large here because A and B are individually common but their gated co-occurrence is rare.

### Expected result

- FRA reaches the target suppression on held-out AB with collateral on A-only AND B-only near the
  clean floor.
- Single-feature (best of remove-A / remove-B) reaches the same suppression only by paying large
  collateral on whichever set it targets -- WORST-case collateral much higher than FRA (predict >5x,
  in line with the semantic-filter Pareto ratios but now on the worst-case, not the mean).
- DoM: broad, between the two.
- Oracle: ceiling; FRA close to it.

**Kill-criterion**: if single-feature, tuned additively with a full coefficient sweep, matches FRA's
worst-case collateral (over A-only and B-only) at matched suppression, this is a boundary point --
record it, do not tune to force a win. But this is the case FRA is theoretically built to win, so a
loss here would mean the conjunction was not actually attention-routed (check: does the payload survive
position-masking the A->B attention? if masking kills it, it is routed and FRA should win).

### Cost / status

Cheap: reuses all existing machinery; only the task templates and the A-only/B-only collateral sets are
new. First runnable target. Feeds the Sep 18 abstract claim directly.
