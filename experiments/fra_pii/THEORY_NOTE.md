# fra_pii — THEORY NOTE: should FRA win on SSN-emit suppression?

Mechanistic account of the SSN task: suppress the model EMITTING a target in-context SSN
(name→ssn) while preserving (a) that SSN as a lookup KEY (ssn→name), (b) a sibling record's
SSN emission, (c) general behaviour. Structured as: a-priori prediction from the win-checklist →
reconcile with the Stage-2 NO-OP and the diag ORACLE → SAE-baseline / confirm-refute conditions.

**WIN CRITERION (behavioral, conjunctive, one operating point).** A method WINS the use-without-
disclosure task iff, at a **single** intervention setting, it drives **emit_ok=False** (target SSN no
longer greedily emitted — disclosure actually prevented) **AND** keeps **look_ok=True** (ssn→name
lookup preserved). Partial P(first-digit) suppression is NOT a win; look_ok/sib_ok measured at a
*non-disarming* setting is NOT evidence of selectivity.

TL;DR — **NOT YET SETTLED; three-way conditional on round-5 (SVD).** Robust facts (round-4): SSN
recall is a **distributed, redundant-bank, content-routed** attention circuit (all-positions oracle
disarms it, emit_supp **0.970**) with a **generic, cross-record-shared digit key**. Scored against the
bar above:
- **FRA (position-invariant content cut): does NOT clear the bar — REACH-CAPPED.** emit_supp ~0.63–0.65,
  flat across M(100→41446 pairs=ALL) × c(8→45), **emit_ok=True at every config** → disclosure never
  prevented (fails conjunct 1). Its look_ok=True/sib_ok=True are at a **non-disarming** point → **weak
  evidence, NOT a selectivity win** (and sib_ok is confounded: Marcus & Elena share the digit tokens
  5,3,1,4,2, so a too-weak cut trivially "preserves" the sibling). The **reach ceiling — SAE
  reconstructs only ~65% of the emit conjunction's score** vs the 0.97 oracle — is the crux quantity.
- **All-positions ORACLE: emit_ok=False but look_ok=FALSE** → fails conjunct 2. Attention removal is
  *symmetric* (kills read-to-emit AND read-to-match), so even the perfect score-space intervention
  cannot dissociate emit from lookup.
- **SAE single-feature (L18 k=1): emit_ok=False AND look_ok=True → clears the bar — but POSITIONALLY**
  (position-locked hook at the target's digit slots) and **cross-record UNCONFIRMED** (sib-metric bug).
  Content-deletion is *directional* (kills transport-out for emit, spares name-output for lookup).

**μ_emit ≠ μ_lookup (query-feature distinctness) is NECESSARY but NOT SUFFICIENT** for an FRA win:
even if the emit-query role is distinct from the lookup-query role, FRA must *also* cross the reach
ceiling (0.65 → emit_ok=False) to prevent disclosure. Round-5 SVD (dimension-reduction of ω → r_eff,
a more complete rank-r edit) decides which of three worlds we are in (see §4).

**Structural contrast with box→frog** (single-token value, concentrated on one live answer→value edge
→ FRA reconstructs ~full score → 1100× win): SSN-emit's value is multi-token and smeared across a
redundant head-bank, so the SAE-cell basis under-reconstructs the score → the reach cap. Whether that
cap is *fundamental* (FRA loses) or *basis-limited* (a completer edit crosses it) is exactly what
round-5 tests. Caveats: SAE cross-record sibling **unconfirmed** (sib-metric bug) + SAE win is only
positional; FRA sib_ok confounded by shared digit tokens; SVD r_eff pending; single fixed DB.

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

> **HOW IT RESOLVED (round-4, see §3–4):** the a-priori was right that the emit/lookup dissociation
> is real and that content-deletion baselines pay on lookup — but WRONG that FRA would be the tool to
> exploit it. FRA is **reach-capped** (0.65 < disarm) before the query-role gate can matter, and the
> dissociation is realised by **SAE residual content-deletion** (directional: kills transport-out,
> spares name-output), not by an FRA query-role-conditioned cut. The lookup axis was won — by SAE.

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

**Consequence for the campaign fork:** clause-1/3 (edge existence / direct consumption) is satisfied
at the conjunction scope — the all-positions oracle suppresses — but FRA fails on **clause-5 REACH**
(§3: the SAE cut caps at 0.65 vs the 0.97 oracle), so it never reaches the selectivity question at
all. The two-axis selectivity is then decided **against FRA**: the oracle disarms but breaks lookup
(symmetric attention removal), and the emit/lookup dissociation is won by **SAE residual
content-deletion** — see §3–4. (The earlier "no surface" over-ride was a single-edge-scope artifact;
the real blocker is reach, one clause further down.)

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
destructive limit — the ALL-POSITIONS cut** (the correctly-scoped maximal FRA effect) — and causally
**PEEL BACK**; trust the support, not the ranking. The general algorithm stands; on THIS task it still
cannot deliver a behavioral win, but for a *reach* reason, not a scope one: the position-invariant
FRA limit reaches only emit_supp ~0.65 (SAE reconstructs ~65% of the score, §3) — partial, never
behavioral disarm — versus the 0.97 *attention* oracle. So peel-back's ceiling here is the reach cap,
and the disarm has to come from residual-space content-deletion instead (§3).

---

## 3. The result: three methods, three outcomes (round-4, real numbers)

All on the fixed 3-record DB, target = record[0], base emit P(first)=0.967. `emit_ok`=True means the
SSN still emits (NOT disarmed); `look_ok`=True means lookup preserved.

| method | mechanism (space) | emit_supp | **emit_ok=F?** (disarm) | **look_ok=T?** (keep) | clears bar? | source |
|---|---|---|---|---|---|---|
| **all-positions ORACLE** | attention removal (score) | **0.970** | **YES** | **NO** | ✗ (breaks lookup) | round-4 |
| **FRA pos-invariant cut** | QK cell-cut (score) | **0.62–0.65** (flat M×c) | **NO** (∀ config) | (T, but vacuous†) | ✗ (never disarms) | `pii_sweep_fra4-allpos.json` |
| **SAE single-feature** (L18 k=1) | source-content delete (residual) | **0.956** | **YES** | **YES** | ✓ but positional‡ | `pii_sweep_sae*.json` |

†FRA's look_ok=True / sib_ok=True are measured at a **non-disarming** operating point (emit_ok=True),
so they are **not** evidence of selectivity. sib_ok is doubly confounded: Marcus (531-42-8817) and the
sibling Elena (624-19-5530) **share the digit tokens 5,3,1,4,2**, so a cut too weak to break emit
trivially "preserves" the sibling too — indistinguishable from content-selectivity. ‡SAE clears the
bar only via a **position-locked** hook at the target's digit slots; cross-record sibling is
**UNCONFIRMED** (sib metric returned `None`/inconsistent in several runs — bug).

**Reading the three rows — the whole story is here:**
- **FRA is REACH-CAPPED (clause-5 failure).** The position-invariant content cut caps at emit_supp
  **0.62–0.65, dead flat** across M∈{100,500,2000,ALL=41446 pairs} × c∈{8,20,45} (gen_kl 0.005), and
  **emit_ok=True at every single config** — it never behaviorally disarms. Since cutting *all* pairs
  at max c still stops at 0.65 while the oracle (which zeros attention outright, bypassing the SAE
  reconstruction and soft-cap) reaches 0.97, the ~0.32 gap **is** the SAE-unexplained score fraction
  (error terms + soft-cap headroom). This is exactly THEORY.md clause-5: SAE reconstructs only ~65% of
  the emit conjunction's score, so the faithful cut can't clear the softmax margin. **Not a locate
  failure — the conjunction is right, the basis is too coarse to act on it.** (It vindicates the scope
  correction: position-invariant reaches 0.65 ≫ the single-edge 0.018, but still caps below disarm.)
- **The ORACLE disarms emit but BREAKS LOOKUP.** Attention-invisibility of the digit positions is
  **symmetric**: it kills reading-them-to-emit AND reading-them-to-match-in-lookup. So the strongest
  possible score-space intervention (the whole family's supremum) **cannot do the dissociation** —
  look_ok=False. This kills my earlier "FRA could spare lookup by query-role" hypothesis at the root:
  even a perfect attention cut can't, because the collateral is not about query-role, it's about the
  digits being *unreadable* to every consumer once you remove attention to them.
- **SAE CLEARS THE BEHAVIORAL BAR — but positionally.** Deleting the digit *representation* at the
  source positions (L18 k=1, emit_supp 0.956, emit_ok=False, look_ok=True) disarms emit while
  preserving lookup — the only method that satisfies both conjuncts at one operating point. It works
  where the oracle fails because content-deletion is **DIRECTIONAL**: emit must *transport that
  specific content forward* to the output (gone ⇒ emit dies), whereas lookup's output is the *name*
  and survives partial source-digit deletion. BUT the hook is **position-locked** (it acts at the
  target's digit slots), so this is a *positional* disclosure control, and **cross-record sibling is
  unconfirmed** (metric bug). It is not (yet) a demonstrated *content-addressed* selectivity — that is
  what a clean sib re-measure would establish.

**So the emit/lookup dissociation EXISTS and one method (SAE) clears the behavioral bar — positionally.**
The FRA-vs-SAE matched-suppression comparison is moot *today*: FRA has no operating point at which emit
is disarmed (reach cap), so there is no matched point. My earlier "query-role separability = FRA's shot"
was wrong: FRA never reaches the operating point where query-role could matter, and the mechanism that
clears the bar is content-direction, not query-role. Whether FRA can *also* clear it (content-addressed,
not position-locked) hinges on beating the reach ceiling — round-5.

---

## 4. Gate outcome so far, the three-way round-5 fork, and the campaign boundary lesson

**Gate, as of round-4.**
- **Tier 1 — SURFACE: MET.** The all-positions oracle disarms emit (0.970) — a suppressing
  position-invariant intervention exists. SSN-emit is edge-routed but **distributed / redundant-bank**
  (the single-edge scan — name 0.000, brace 0.000, digit 0.126 top-12, best head 0.009 — showed only
  that it isn't *localized*, not that it's absent).
- **Clause-5 — REACH: FRA does not clear it (yet).** The faithful position-invariant FRA cut caps at
  **0.62–0.65 (flat, emit_ok=True ∀)** against a 0.97 oracle ceiling — the SAE-cell basis reconstructs
  only ~**65%** of the emit conjunction's score. FRA locates the right conjunction but cannot act on
  enough of it to clear the softmax margin. **This ~65% is the crux quantity of the whole task.**
- **SELECTIVITY (the dissociation): cleared by SAE positionally; UNSETTLED for content-addressing.**
  SAE clears the behavioral bar (0.956 disarm + lookup kept) but position-locked; the oracle disarms
  yet breaks lookup; FRA never disarms. No *content-addressed* method has been shown to clear the bar.

**The three-way fork round-5 (SVD of ω → r_eff / a more complete rank-r edit) decides.** Whether FRA's
0.65 is a fundamental content-conjunction ceiling or a basis-completeness artifact is exactly what a
completer edit tests:
- **(a) SVD crosses emit_ok=False while look_ok stays True → GENUINE use-without-disclosure WIN** — and
  a *content-addressed* one, beating SAE's merely-positional win (the axis SAE only wins by position).
- **(b) SVD still stuck at ~0.65 (emit_ok=True) → "SELECTIVITY WITHOUT SUFFICIENCY."** FRA correctly,
  content-selectively targets the emit conjunction but cannot remove enough of it to prevent disclosure
  — a real, nameable partial result, distinct from both the vacuous single-edge null and a clean win.
  The reach cap is then a genuine property of distributed content-routed recall in the SAE basis.
- **(c) Crossing emit_ok=False also drops look_ok → SETTLED NEGATIVE.** The emit and lookup reads of
  the digits are not separable by any content-conjunction cut; the dissociation is SAE-positional only.

Necessary-but-not-sufficient reminder: even outcome (a) requires BOTH μ_emit≠μ_lookup (query-feature
distinctness, so the cut spares the lookup role) AND crossing the reach ceiling. (b) and (c) differ by
whether the lookup role survives once you *do* cross. Keep all three live until round-5 lands.

**Why box→frog is a clean FRA win and SSN-emit is at best reach-stressed — the structural inverse.**

| | box→frog (banked ~1100× FRA win) | SSN-emit (this task) |
|---|---|---|
| value | single token ("frog") | multi-token digit string |
| edge | one live, concentrated answer→value edge | distributed redundant head-bank, multi-hop |
| SAE-cell score reconstruction | ~full → cut clears the margin | ~65% → cut reach-caps below disarm |
| discriminator | distinctive value token (own feature) | generic, cross-record-shared digit key |
| bar cleared so far | **FRA** (score-space, content-addressed) | **SAE only, positionally** (FRA pending round-5) |

**Campaign boundary lesson (robust part).** FRA's score-space attention-cut is the right tool when the
behaviour is a *concentrated, content-keyed, single/low-hop* attention edge (box→frog, induction,
copy-suppression, acronym). SSN-emit is the opposite substrate — *multi-token, distributed,
redundant-bank, content-routed recall* — and there two things are already established regardless of
round-5: (i) the SAE-cell basis under-reconstructs the score, so the faithful FRA cut **reach-caps at
0.65** (no behavioral disarm at any M/c); (ii) attention removal is **symmetric**, so even the oracle
cannot separate emit from lookup — the dissociation, where it has been achieved at all, lives in
**residual/content space** (SAE, directional deletion). What round-5 decides is whether a completer
(SVD/rank-r) content-addressed edit can beat the reach cap; that toggles SSN-emit between a
content-addressed FRA win (a), a "selectivity-without-sufficiency" partial (b), and a settled negative
(c). Either way it maps a new boundary edge: **distributed content-routed recall is where score-space
FRA is at least reach-stressed and residual-space deletion is the safer disclosure control.**

**Caveats (do not over-claim).** (i) SAE cross-record **sibling** preservation is **unconfirmed** —
the sib metric returned `None`/inconsistent values in several SAE runs (bug), so "disarm Alice keep
Bob" is not yet demonstrated for SAE, only "disarm Alice keep lookup." (ii) The SVD/effective-rank
(`r_eff`) analysis of the emit conjunction is **pending** (would quantify the distribution directly).
(iii) Single fixed 3-record DB — held-out-DB transfer not yet re-run at the position-invariant scope.
None of these threaten the headline (FRA reach-cap 0.65 vs oracle 0.97 vs SAE 0.956 are robust across
the full M×c grid); they bound the *sibling* and *generality* claims.

---

## One-sentence campaign placement

SSN-emit is a **distributed, redundant-bank, content-routed** recall circuit with a generic,
cross-record-shared digit key: the behavioral win bar (emit_ok=False AND look_ok=True at one setting)
is cleared **only by a position-locked SAE feature (0.956 disarm, lookup kept; cross-record
unconfirmed)** — the all-positions attention oracle disarms but breaks lookup (symmetric removal), and
**FRA's content-conjunction cut is REACH-CAPPED at 0.65 (emit_ok=True ∀ M×c), never preventing
disclosure**, because the SAE-cell basis reconstructs only ~65% of the emit conjunction's score; so
the verdict is **three-way conditional on round-5 (SVD/rank-r)** — (a) a completer edit crosses the
cap with lookup intact = content-addressed FRA WIN, (b) it stays capped = "selectivity without
sufficiency", (c) crossing also breaks lookup = settled negative — the structural inverse of box→frog
(single-token, concentrated live edge → FRA 1100×), and already the boundary lesson that distributed
content-routed recall is where score-space FRA is reach-stressed and residual-space deletion is safer.

---
*Numbers (HF `dmanningcoe/fra-phase1-steering-data`, prefix `fra_pii/results/`): FRA position-invariant
`pii_sweep_fra4-allpos.json` (0.62–0.65 flat, emit_ok=T ∀); SAE `pii_sweep_sae-5.json` (L18 k=1 =
0.956 disarm); all-positions oracle 0.970 disarm / look_ok=F (round-4, per orchestrator); single-edge
scan `pii_sweep_diag.json`/`rs-pii-diag_run.log`; Stage-2 `pii_cut_results.json`;
`fra_win/out/pii_sibling/shared_endpoint_t2.json`. Caveats: SAE sib-metric bug, r_eff pending, single DB.*
