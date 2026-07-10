# fra_pii — THEORY NOTE: should FRA win on SSN-emit suppression?

Mechanistic account of the SSN task: suppress the model EMITTING a target in-context SSN
(name→ssn) while preserving (a) that SSN as a lookup KEY (ssn→name), (b) a sibling record's
SSN emission, (c) general behaviour. Structured as: a-priori prediction from the win-checklist →
reconcile with the Stage-2 NO-OP and the diag ORACLE → SAE-baseline / confirm-refute conditions.

TL;DR — **A-priori: SPLIT (emit/lookup separable by query-role; emit/sibling NOT — A→1 on the sibling
axis).** **CORRECTION (this revision):** an earlier draft concluded "no bounded FRA surface / Tier-1
unmet" — that was an **artifact of testing FRA only in its degenerate SINGLE-EDGE mode**. The
single-edge oracle (name 0.000, brace 0.000, digit 0.126 top-12) upper-bounds only the single-edge
family; **FRA's native edit is POSITION-INVARIANT** (subtract c·u^μ_q·u^ν_k·ω_μν at *every* (q,k)
where the conjunction fires), a strictly larger family whose true ceiling is the **ALL-POSITIONS
oracle** — zero attention onto the digit key-content from *all* queries at *all* layers. Because the
SSN is **random** (its digits can enter the computation *only* by being attended from context), the
all-positions oracle **MUST suppress emit** — so **Tier-1 is MET** and SSN-emit is a **distributed /
multi-hop attention circuit**, not a non-edge. The live question is now **Tier-3 selectivity, on two
axes**: emit-vs-**lookup** may be separable (lookup reads digits under a *different query role* = a
different cell) — the one axis where FRA's content-cut could beat content-ablation/SAE/the raw
all-positions oracle (all of which break lookup); emit-vs-**sibling** is NOT separable (shared generic
digit-key conjunction → A→1). FRAAgent is running the position-invariant content cut now.

---

## 1. A-priori prediction from the win-checklist

Task endpoints at the emit step ("… The SSN of Marcus Webb is ▮"):
- **query** = the answer position, "about-to-emit-the-ssn-value". A **format/ROLE** feature
  (the JSON-ssn-value slot / "SSN of X is" template). It is *identical across every record's
  emit* — the which-record work is done upstream by name-matching that resolves identity
  **positionally** (attend to wherever "Marcus Webb" was mentioned, copy the adjacent ssn field).
- **key** = SSN **digit tokens** ("5","3","1",…). Maximally **generic content** — every number,
  price, id in the stream reuses them.

Clause scores:

| clause | verdict a-priori | reason |
|---|---|---|
| 1 EDGE-ROUTED | plausibly PASS | emit looked like a large answer→digit induction/copy edge (raw attn L18H6≈0.75). |
| 2 LOAD-BEARING (LBNR, causal) | **at risk** | short in-context sequence + strong structure → redundant routes; needs causal (not raw-attn) confirmation. |
| 3 DIRECT CONSUMPTION | at risk | is the digit OV-transported from the digit key at the answer step, or already resident (moved onto the name/record token earlier)? |
| **4 CONJUNCTION-SPECIFIC w/ DISTINCTIVE-CONTENT discriminator** | **SPLIT — sibling FAIL / lookup OPEN** | The cut is the CONJUNCTION (emit-role query μ × digit-key ν), which fires only where BOTH fire — so "generic digit key" alone does *not* imply non-selective. **Sibling FAIL:** μ_emit fires at Bob's answer position → shared conjunction → A→1. **Lookup OPEN:** lookup reads the digits under a *different* query role (match-a-queried-SSN, not copy-this-person's-SSN); the cell fires there only if μ_emit fires at the lookup answer position — an empirical query-feature-identity question, NOT settled by the key being generic. |
| 5 REACH | n/a until 1–4 hold | — |

**Magnitude-law prediction: split by axis (A ≈ reuse(marginal)/reuse(conjunction)).** The relevant
denominator is reuse of the *conjunction* (emit-role μ × digit ν), NOT reuse of the generic digit key.
- **Sibling axis: A→1.** The (emit-role × digit) conjunction recurs on Bob's emission (same task, same
  role) — high reuse ⇒ no collateral advantage. Same class as the shared-endpoint sibling test
  (`fra_win/out/pii_sibling/shared_endpoint_t2.json`: FRA on-target removal ≈ 0, fra_on 0.757 ≈
  red_base 0.756) and box-retrieval (A=7.9× partial only at weak base).
- **Lookup axis: A OPEN, possibly ≫1.** If μ_emit is emit-specific (does not fire in the lookup role),
  reuse(conjunction | lookup) ≈ 0 while reuse(marginal digit) is high → A large: FRA spares lookup
  where every content-deletion baseline (which removes the digit *marginal*) must break it. This is
  the use-without-disclosure win, and it is decided by the query-feature-identity check, not a-priori.

**Is the emit/lookup dissociation even expressible as separable cells?** Partly:
- emit-query-role ("about to emit a DIGIT") ≠ lookup-query-role ("about to emit a NAME"). These are
  different query-side features, so a query-role-conditioned FRA cut is **inert on lookup** →
  lookup preserved. *This* axis FRA can express — and it is the one axis where content-ablation and
  digit-position SAE steering must pay (they delete the digit content that lookup needs to match).
- but the emit/**sibling** axis is NOT separable: Bob's emit shares the identical (emit-role ×
  digit-key) conjunction, so any cut strong enough to disarm Marcus fires on Bob too. A position
  patch spares Bob (different positions); the content-addressed cut cannot.
- The task requires preserving **both** lookup and sibling. FRA can, in principle, get lookup
  (query-role differs) but not sibling (generic conjunction). Net: not a win.

---

## 2. Reconcile with the Stage-2 NO-OP + the diag ORACLE

**Stage-2 (`pii_cut.py`, 8 held-out DBs):** FRA top-20 (answer→first-digit) cells, c binary-searched
for 0.8 removal, **capped at the c=24 ceiling** and only dropped emit_p ~0.96→~0.87–0.95 (≈8%);
emit_ok=True on all 8. The head-line "FRA emit-removed 0.00 / lookup 1.00" is a **vacuous** win: it
preserves lookup because it removed *nothing*. Baselines: content-ablation removed 1.00 / lookup
0.75; output-logit-suppress removed 1.00 / lookup 0.00. So the dissociation was never actually
tested by FRA — it was a removal-gated no-op masquerading as a clean separation.

**The two pre-registered hypotheses made different ORACLE predictions:**
- REACH failure (SAE misses a *real* answer→digit edge) → **oracle SUPPRESSES**, FRA falls short of it.
- CONTENT / EDGE-WRONG failure → **oracle also weak**.

**The diag result (`pii_sweep_diag.json`, `rs-pii-diag_run.log`) settles it — against reach:**

```
base emit P(first)      = 0.967
per-head causal drop (zero answer→digit attn):  L18H6 0.009, L15H0 0.002, rest ≈0 or NEGATIVE
ORACLE (top-6 heads, zero answer→ALL-digit attn): emit_supp 0.02  emit_ok True  look_ok True  sib_ok True  gen_kl 0.000
FRA-cut-ALL  c=10/30/60:                           emit_supp 0.018 (== oracle)
FRA M×c sweep (M∈{50,200,1000,ALL}×c∈{8,20,45}):   emit_supp flat 0.014–0.026, emit_ok True everywhere
```

**Diagnosis (corrected): a DISTRIBUTED / multi-hop attention circuit — NOT a non-edge.** These
numbers show the emit is not a *single localized* answer→digit edge; an earlier draft over-read that
as "no QK surface." The correction (see the scope discussion below): the single-edge oracle bounds
only single-edge cuts, and FRA's native mode is position-invariant. Reading the facts correctly:
1. **The single-edge oracle fails — and it bounds only the single-edge family.** Zeroing
   answer→digit scores on the answer query (bypassing SAE + soft-cap) drops emit 0.02. This says the
   *answer→digit edge is not solely load-bearing* — NOT that no attention cut works. The provable
   ceiling is the **all-positions oracle** (digit content invisible to all queries, all layers),
   which MUST suppress because the SSN is random (below). So the correct reading is *distributed*,
   not *absent*.
2. **Raw attention is a red herring.** L18H6 carries 0.75 raw answer→first-digit attention but 0.009
   causal drop — the raw-attn-vs-causal dissociation THEORY.md warns about. Heads chosen by raw
   attention; causally no *single* one carries emit (max drop 0.009). Consistent with a distributed
   bank, not with a non-edge.
3. **FRA-cut-ALL == the single-edge oracle (both 0.018).** This confirms FRA is *faithful at the
   single-edge scope* (full reach on that edge) — it does not mean FRA is out of moves, because FRA
   is not restricted to one edge. The flat M×c sweep is flat because every cell in it lives at the
   *same* single edge; widening the SCOPE (position-invariant), not the M or c, is the untried lever.

Mechanistically the retrieved SSN value is transported by a **distributed / multi-hop** attention
process — likely within-record name↔ssn binding at earlier layers feeding later copy hops, spread
thin across a head bank (per-head drops tiny, top-12 = 0.126). No *single* answer→key edge disarms
it, but a **position-invariant content-conjunction cut** (FRA's native mode) that makes the digit
content attention-invisible at every hop provably can (all-positions oracle). So this is NOT a
clause-3 non-edge; it is an **edge-routed-but-distributed** circuit whose cut lives at the
conjunction scope. The remaining question is clause-4 *selectivity*, not clause-1/3 existence.

### The box→frog contrast — single-edge-cuttable vs only-position-invariant-cuttable
The campaign's **banked retrieval win (~1100×)** is single-token forward retrieval — box→frog,
"the box contains a **frog**" / "key X → value **Y**". There the value is ONE token sitting AT the
attended key position, so a **single answer→value edge OV-copies the whole value**: the single-edge
oracle works, the value token is a distinctive-content discriminator, and FRA both **suppresses AND
wins on every axis** (A≈1100×, all tiers pass). SSN-emit differs in scope AND selectivity:
- **Multi-token value + record indirection.** "531-42-8817" → `5,3,1,-,4,2,…` inside a nested
  multi-record JSON keyed by name; retrieval is a distributed name↔ssn binding across many hops, so
  no *single* answer→value edge carries it — the cut lives at the **position-invariant conjunction
  scope**, not the single-edge scope. (Cuttable, just not the way box→frog is.)
- **Generic key.** frog is a distinctive value token (own content feature); an SSN digit is a
  generic token reused by every number — so even the position-invariant cut is non-selective across
  siblings (clause 4, A→1 on the sibling axis), whereas box→frog's value key is its own discriminator.

So box→frog and SSN-emit are BOTH edge-routed and both give FRA a surface; they split on (i) scope
(single-edge vs position-invariant) and (ii) selectivity (distinctive-value-key vs generic-digit-key).
SSN-emit is the entity-attribute *binding* class (synth-boxes: distributed edge, LBNR-R=−0.89), not
flat single-token retrieval — which is why it needs FRA's native position-invariant mode and why its
sibling axis collapses to A→1.

**Consequence for the campaign fork:** clause-1/3 (edge existence / direct consumption) is now
satisfied at the conjunction scope — the all-positions oracle suppresses — so the live question is
**clause 4, on two axes**: emit-vs-lookup (a query-role discriminator may exist → FRA's shot at a
win) and emit-vs-sibling (generic digit key → A→1, no discriminator). The a-priori "split" verdict of
§1 stands; the earlier "no surface" over-ride was a single-edge-scope artifact.

### FRA ⊇ attention ablation — but at TWO scopes (the single-edge trap, corrected)
Cutting the FRA cells subtracts Σ_{μν} u^μ_q u^ν_k ω^h_{μν} = the FRA-reconstructed content score
Ŝ^h[q,k]; in the perfect-reconstruction limit Ŝ = the full pre-cap score, so at large c it drives
that attention to 0 = attention ablation. **The subtlety that a first draft missed: this identity has
a SCOPE, and FRA is defined at the *conjunction* scope, not the single-edge scope.**

- **Single-edge scope.** Cut ALL cells at ONE pair (q,k) ⇒ ablate that one attention edge. Its
  destructive limit is the **single-edge oracle**. Empirically FRA-cut-ALL == the answer→digit
  single-edge oracle to measured precision (both 0.0184) — the identity holds. **But this is the
  degenerate corner**, and the harness (`fra_delta_topM` masks `qq==qpos & kk∈kpositions`) only ever
  cut here. Its ceiling (0.126, top-12) bounds *only single-edge cuts*.
- **Conjunction / position-invariant scope — FRA's ACTUAL native mode.** Cut the cell (μ,ν)
  *wherever it fires*: subtract c·u^μ_q·u^ν_k·ω_{μν} at EVERY (q,k) with u^μ_q>0 ∧ u^ν_k>0. This is a
  **strictly larger** family than any single-edge cut. Its destructive limit is the **ALL-POSITIONS
  oracle**: make the digit key-content attention-invisible from *all* queries at *all* layers =
  cut-ALL-cells-at-all-positions. **The single-edge oracle does NOT upper-bound this family** — that
  was the error.

> **Corrected statement:** FRA ⊇ attention ablation *at the conjunction scope*; the true destructive
> limit to peel back from is the **all-positions** oracle (content invisible everywhere), not the
> single-edge oracle. FRA at M<ALL / restricted-c is the content-addressed sub-family nested inside
> **that**.

**Why the all-positions oracle MUST suppress here (Tier-1 met).** The target SSN is *random*: its
digits carry no prior, so they can enter the computation ONLY by attention pulling them off the
digit key-positions — at whatever hop, however distributed. Zero all such attention everywhere and
the information cannot propagate ⇒ emit (and lookup, and sibling if you include their digits) must
break. So a suppressing position-invariant cut provably exists; the single-edge null was a scope
artifact, not a "non-edge."

**Methodological corollary (F5 mitigation, now correctly scoped).** F5: magnitude-ranking ⊥ causal
effect, so forward top-M is unreliable (the whole M×c sweep is flat). Guide instead **from the
destructive limit — the ALL-POSITIONS cut** (= all-positions oracle, the provable maximal effect) —
and causally **PEEL BACK** cells/positions to the minimal subset that still disarms emit while
sparing lookup/siblings. Trust the support, not the ranking; validate each prune causally. Peeling
from the single-edge limit (what a first pass would do) cannot even reach suppression here — you must
peel from the position-invariant limit.

---

## 3. SAE baseline + FRA-vs-SAE, and confirm/refute

**SAE single-feature steering prediction (`pii_sweep_sae.json`, pending):** the SAE hooks ablate the
top-k features **on the target's ssn-digit positions** — i.e. delete digit content where it sits.
Unlike answer→digit *attention* (inert), deleting the source content should actually starve the
copy, so **SAE will suppress emit** (this is the content-ablation mechanism, positionally gated).
Because it is gated to the target's digit positions it will **spare the sibling** (different
positions) but **break lookup** (the ssn→name match needs exactly those digits) — mirroring
content-ablation's 0.75 lookup / output-suppress's 0.00 lookup. Expect the same shape here:
suppress-emit yes, lookup-collateral yes, sibling-spared yes (positional).

**FRA-vs-SAE at matched suppression (now DEFINED, once FRA uses its position-invariant mode).** The
comparison was only "undefined" while FRA was tested single-edge; with the position-invariant cut it
has an operating point (the all-positions oracle proves suppression exists). At matched emit
suppression the comparison runs on **two collateral axes**, and they point opposite ways:
- **emit-vs-lookup — FRA's shot.** SAE/content-ablation delete the digit *representation* at source,
  so lookup (ssn→name match needs those digits) breaks. FRA cuts the (emit/copy-query-role × digit)
  *cell*; if lookup reads the same digits under a **different query role**, that is a different cell,
  left intact → **FRA spares lookup where SAE cannot.** This is the one axis where FRA can beat every
  content-deletion baseline AND the raw all-positions oracle (all three break lookup).
- **emit-vs-sibling — FRA loses (or ties).** Bob's emit is the same (emit-role × generic-digit)
  conjunction, so a position-invariant cut fires on it too → sibling breaks. SAE/oracle spare the
  sibling by *positional* gating (act only at the target's digit positions). A→1 on this axis.

**CONFIRM "generic-digit-key → no FRA advantage on the SIBLING axis":** at matched emit suppression,
FRA sib_ok drops and sibling collateral ≥ the positionally-gated baselines (A≈1). **REFUTE (the win):**
FRA at matched suppression keeps **lookup_ok=1** while SAE/content-ablation/all-positions-oracle all
drop it — this is the query-role separability, and it is a *genuine* FRA-only capability even though
the sibling axis is lost. The crux the empirical run decides: does the load-bearing digit-attending
hop use an **emit-specific query role** (→ lookup spared, REFUTE-the-loss on that axis) or a **shared
name↔ssn binding role** that also fires in lookup (→ FRA breaks lookup too, no advantage)?

---

## 4. The re-locate result, correctly scoped (FRAAgent)

FRAAgent's single-edge key-position scan:

```
oracle key-position scan (drop in P(first digit) from zeroing answer→k attn, single-query):
  name-token oracle    = 0.000
  brace/struct oracle  = 0.000
  digit oracle (top-12)= 0.126        best single head = 0.009
```

**What this does and does NOT show.** It shows the emit is **not a single localized answer→key
edge** — name/brace 0.000 refute the "answer→name/record carrier" hop, and the digit axis is a weak
**distributed bank** (top-12 = 0.126 ≫ best single 0.009: the STOCKTAKE-T1 redundant-head-bank
signature). It does **NOT** show "no FRA surface" — that inference (in an earlier draft) mistook the
single-edge oracle for the ceiling of the whole FRA family. FRA's native cut is position-invariant;
its ceiling is the **all-positions** oracle, which **must** suppress (random SSN, §2). So:

**Re-scoped three-tier gate.** Evaluate Tier 1 with the ALL-POSITIONS oracle, not the single-edge one:
- **Tier 1 — SURFACE: MET.** All-positions digit oracle provably suppresses emit ⇒ a suppressing
  position-invariant cut exists. SSN-emit is edge-routed, distributed/multi-hop. (The single-edge
  0.126 was the wrong ceiling.)
- **Tier 2 — DIRECT CONSUMPTION: MET in substance.** The random SSN can reach the output only through
  attention onto the digit content, so making that content attention-invisible disarms it — the
  behaviour *is* transported by the (distributed) attention, not recomputed independently.
- **Tier 3 — CONTENT DISCRIMINATOR: the live question, on TWO axes.**
  - *emit-vs-lookup:* possibly SEPARABLE — lookup reads the digits under a different query role ⇒ a
    different (role × digit) cell ⇒ FRA's emit-role cut can spare it. FRA's one shot at a capability
    no baseline has.
  - *emit-vs-sibling:* NOT separable — Bob's emit shares the generic (emit-role × digit) conjunction
    ⇒ A→1. Positionally-gated baselines (SAE/oracle) win this axis; FRA loses it.

### The open measurement (FRAAgent running the position-invariant cut)
The single remaining crux is **which query role carries the load-bearing digit-attending hop**:
- an **emit-specific role** ("about to copy a value") that does NOT fire in the lookup context ⇒
  FRA's content cut disarms emit and **spares lookup** ⇒ REFUTE the loss on the lookup axis (real
  FRA-only win there); sibling still breaks.
- a **shared name↔ssn binding role** that ALSO fires when lookup matches the given SSN ⇒ cutting it
  breaks lookup too ⇒ no advantage over content-ablation; FRA reduces to the all-positions oracle.

Also worth measuring: the **all-position digit-oracle ceiling** (confirms Tier-1 quantitatively) and
whether peeling back from it (§2 algorithm) finds any target-*and*-emit-specific subset — expected
to fail on the sibling axis (no content discriminator) but possibly succeed on the lookup axis.

**Net verdict — OPEN, do not stamp.** SSN-emit is an **edge-routed distributed circuit** (Tier-1/2
met via the position-invariant mode) with a **split clause-4**. One axis is decided a-priori, one is
NOT:
- **emit-vs-sibling: FAIL (A→1), decided.** μ_emit fires at Bob's answer position → shared conjunction.
- **emit-vs-lookup: OPEN, pending the query-feature-identity check.** Two outcomes:
  - **(a) μ_emit is emit-specific** (does not fire in the lookup role) → FRA disarms emit AND spares
    lookup → a **genuine use-without-disclosure WIN on this axis, where SAE/content-ablation lose**
    (they delete the digit marginal → break lookup). Sibling still breaks.
  - **(b) μ_emit is a generic retrieve-role shared with lookup** → cutting the cell breaks lookup too
    → FRA reduces to the all-positions oracle, no advantage → *then* it is the settled negative.

So the campaign verdict is **conditional on FRAAgent's all-positions content-cut + query-feature
check**, not closed. The earlier "no bounded surface / settled negative" was doubly wrong: (i) it used
the single-edge oracle (0.126) as the ceiling for the strictly-larger position-invariant family, and
(ii) it read "generic digit key" as "non-selective," ignoring that the cut is the CONJUNCTION μ×ν —
which is what leaves the lookup axis alive. The box→frog contrast (single-token value, all axes win)
and the FRA⊇attention-ablation formalization stand regardless.

---

## One-sentence campaign placement

SSN-emit is an **edge-routed but DISTRIBUTED / multi-hop** retrieval circuit whose cut lives at the
position-invariant CONJUNCTION scope (the single-edge oracle nulls — name 0.000, brace 0.000, digit
0.126 top-12 — bound only single-edge cuts; the all-positions oracle MUST suppress because the SSN is
random, so a suppressing FRA cut provably exists); selectivity then splits by the conjunction gate
(μ_emit × ν_digit), NOT by the generic key alone — **emit-vs-sibling FAILS** (μ_emit fires at Bob's
answer position, A→1) while **emit-vs-lookup is OPEN**, a genuine use-without-disclosure win **iff**
the empirical query-feature check shows μ_emit is emit-specific and not shared with the lookup role
(if it is, FRA breaks lookup too and it's the settled negative); contrast box→frog (single-token
value, live answer→value edge, all axes win, ~1100×). Verdict: **conditional, not closed** — pending
FRAAgent's all-positions content-cut + query-feature-identity check.

---
*Numbers: `pii_sweep_diag.json`, `pii_sweep_fra.json`, `rs-pii-diag_run.log`, Stage-2
`pii_cut_results.json`/`rs-pii-cut5_run.log`, `fra_win/out/pii_sibling/shared_endpoint_t2.json`
(HF `dmanningcoe/fra-phase1-steering-data`, prefix `fra_pii/`).*
