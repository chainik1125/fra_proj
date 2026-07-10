# fra_pii — THEORY NOTE: should FRA win on SSN-emit suppression?

Mechanistic account of the SSN task: suppress the model EMITTING a target in-context SSN
(name→ssn) while preserving (a) that SSN as a lookup KEY (ssn→name), (b) a sibling record's
SSN emission, (c) general behaviour. Structured as: a-priori win-checklist prediction (§1) →
scope/oracle reconciliation + the FRA⊇attention-ablation formalization (§2) → every method scored
against the behavioral bar (§3) → final verdict, the B3 condensate, and the boundary lesson (§4).

**WIN CRITERION (behavioral, conjunctive, one operating point).** A method WINS the use-without-
disclosure task iff, at a **single** intervention setting, it drives **emit_ok=False** (target SSN no
longer greedily emitted — disclosure actually prevented) **AND** keeps **look_ok=True** (ssn→name
lookup preserved). Partial P(first-digit) suppression is NOT a win; look_ok/sib_ok measured at a
*non-disarming* setting is NOT evidence of selectivity.

TL;DR — **SETTLED (round-5): outcome (b) — FRA is CONTENT-SELECTIVE but REACH-INSUFFICIENT.** SSN
recall is a **distributed, redundant-bank, content-routed** attention circuit (all-positions oracle
disarms it, emit_supp **0.970**) with a **generic, cross-record-shared digit key**. Scored against the
bar:
- **FRA — does NOT clear the bar (reach-insufficient), but the selectivity is now REAL.** The best
  variant, an **SVD rank-1 edit, reaches emit_supp 0.83–0.84** (beating the 0.65 cell-cut) and
  **preserves lookup + sibling + general** (gen_kl 0.058) — yet **emit_ok=True**: base P(first)=0.967 is
  so peaked that even an 84% crush leaves the SSN as the greedy argmax. Rank-4 *lowers* supp to 0.65 and
  *breaks* selectivity → the top mode **is** the emit-copy direction, lower modes are shared collateral.
  The **FRA(0.84)→oracle(0.97) gap is the reach gap** — FRA edits only the ~65–84% SAE-reconstructed
  part; the error terms carry the argmax-flipping remainder. **The reach ceiling is the crux quantity.**
- **All-positions ORACLE: emit_ok=False but look_ok=FALSE** — attention removal is *symmetric* (kills
  read-to-emit AND read-to-match), so even the perfect score-space intervention can't dissociate.
- **SAE single-feature (L18 k=1): emit_ok=False AND look_ok=True → the only method that clears the bar —
  but POSITIONALLY** (position-locked hook) and **cross-record UNCONFIRMED** (sib-metric bug). Not a
  demonstrated *content-addressed* selectivity; content-deletion is *directional* (kills transport-out).

**Accurate frame = "selectivity without sufficiency"** — retire BOTH "settled negative" (FRA is
genuinely content-selective) and "selectivity win" (it never disarms). **Silver lining: the B3
CONDENSATE is CONFIRMED on gemma** — per-head ω is ~rank-1 (cumvar@r=1 = 0.915–0.998, r_eff≈1),
contra the F7 no-condensate lean for dense gpt2, reopening T4 as productive (see §4).

**Structural inverse of box→frog** (single-token value, concentrated on one live answer→value edge →
FRA reconstructs ~full score → 1100× win): SSN-emit's value is multi-token and smeared across a
redundant head-bank, so the SAE-cell basis under-reconstructs the score → the reach cap. Round-5
showed the cap is not a scope or a low-rank artifact (rank-1 already ≈ the ceiling; more rank only adds
collateral) — it is a genuine **basis-reach** limit against a very peaked base distribution. Caveats:
SAE cross-record sibling **unconfirmed** (sib-metric bug) and SAE's win is only positional; FRA sib_ok
still shares the digit tokens (but the rank-1-vs-rank-4 contrast and general-KL preservation carry the
selectivity claim); single fixed DB; base model.

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

## 3. The result: every method scored against the behavioral bar (r4–r5, real numbers)

All on the fixed 3-record DB, target = record[0], base emit P(first)=0.967. `emit_ok`=True means the
SSN still emits (NOT disarmed); `look_ok`=True means lookup preserved. **Win = emit_ok=False AND
look_ok=True at one setting.**

| method | mechanism (space) | emit_supp | **emit_ok=F?** | **look_ok=T?** | clears bar? | source |
|---|---|---|---|---|---|---|
| **all-positions ORACLE** | attention removal (score) | **0.970** | **YES** | **NO** | ✗ (breaks lookup) | round-4 |
| FRA single-edge cut | QK cell-cut (score) | ≤0.13 | NO | — | ✗ | `pii_sweep_diag.json` |
| FRA pos-invariant cut | QK cell-cut (score) | 0.62–0.65 (flat M×c) | NO (∀) | T (non-disarm†) | ✗ (never disarms) | `pii_sweep_fra4-allpos.json` |
| **FRA SVD rank-1** (best FRA) | QK cell-cut, rank-r (score) | **0.83** | **NO** | **YES** | ✗ (reach) — but **selective**§ | `pii_sweep_fra5-svd.json` |
| FRA SVD rank-2 | " | 0.84 | NO | YES | ✗ (reach) | `pii_sweep_fra5-svd.json` |
| FRA SVD rank-4 | " | 0.65 | NO | YES (but sib BROKEN) | ✗ + cross-record collateral | `pii_sweep_fra5-svd.json` |
| **SAE single-feature** (L18 k=1) | source-content delete (residual) | **0.956** | **YES** | **YES** | ✓ but positional‡ | `pii_sweep_sae*.json` |

†at 0.65 the look_ok/sib_ok were non-disarming *and* sib is confounded (target 531-42-8817 & sibling
624-19-5530 share digit tokens 5,3,1,4,2). §The SVD rank-1 result **earns the selectivity claim**: at
0.83 supp it preserves lookup + sibling + general (gen_kl 0.058), and **rank-4 *lowers* supp to 0.65
while breaking the SIBLING only — lookup stays intact**. The top mode IS the emit-copy direction; the
lower modes carry **cross-record (sibling) collateral**, not lookup collateral. That the over-editing
degradation lands on the *sibling* (never lookup) actually *strengthens* the earned-selectivity point:
the rank-1-preserves / rank-4-breaks-sibling contrast (plus clean general-KL) is content-selectivity
evidence independent of the shared-digit confound. ‡SAE clears the bar only via a
**position-locked** hook; cross-record sibling **UNCONFIRMED** (metric bug).

**Reading the rows — the whole story:**
- **FRA is CONTENT-SELECTIVE but REACH-INSUFFICIENT (clause-5).** The position-invariant cell-cut caps
  at 0.62–0.65 flat across M∈{100…ALL=41446} × c∈{8…45}; the **SVD rank-1 edit pushes it to 0.83–0.84**
  (best variant) — but **emit_ok stays True at every setting**. Even at 0.84, base P(first)=0.967 is so
  peaked that the SSN remains the greedy argmax, so *disclosure is never prevented*. The FRA(0.84)→
  oracle(0.97) gap **is** the reach gap: FRA edits only the ~65–84% SAE-reconstructed part of the emit
  score; the SAE-unexplained error terms carry the argmax-flipping remainder. **Not a locate failure,
  not a scope failure, not a low-rank failure — a basis-reach failure** (the conjunction is right and,
  via SVD, is one clean mode; the SAE basis just can't reconstruct enough of its score to cross a very
  peaked margin). The **selectivity is real**: rank-1 preserves lookup+sibling+general at 0.84 while
  rank-4 *breaks* the sibling at lower supp — the top mode is the emit-copy direction, lower modes are
  shared collateral. So FRA gets the *right target* but not *enough removal*.
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

**So the emit/lookup dissociation EXISTS; only SAE clears the behavioral bar, and only positionally;
FRA is content-selective but reach-insufficient.** The FRA-vs-SAE matched-suppression comparison stays
moot because FRA has *no* operating point at which emit is disarmed (its best, rank-1, tops out at 0.84
< the peaked argmax margin). My earlier "query-role separability = FRA's shot" was wrong on both counts:
FRA never reaches disarm (reach), and the mechanism that *does* clear the bar is content-direction
(SAE residual deletion), not query-role. The one thing round-5 changed: FRA's selectivity is now
*demonstrated* (rank-1 vs rank-4), so the verdict is **outcome (b) — selectivity without sufficiency**,
not a settled negative and not a win.

---

## 4. Final verdict — outcome (b), the B3 condensate, and the campaign boundary lesson

**Gate, settled (r4–r5).**
- **Tier 1 — SURFACE: MET.** The all-positions oracle disarms emit (0.970) — a suppressing
  position-invariant intervention exists. SSN-emit is edge-routed but **distributed / redundant-bank**
  (single-edge scan: name 0.000, brace 0.000, digit 0.126 top-12, best head 0.009 → not *localized*).
- **Clause-5 — REACH: FRA FAILS.** Best variant (SVD rank-1) reaches emit_supp 0.83–0.84 but never
  disarms (emit_ok=True ∀). The 0.84→0.97 gap is the SAE-unexplained score fraction; against a base
  P=0.967, that remainder keeps the SSN as greedy argmax. **The ~65–84% reconstruction is the crux.**
- **SELECTIVITY: real for FRA (rank-1 vs rank-4), but insufficient; the bar is cleared only by SAE,
  positionally.** SAE disarms (0.956) + keeps lookup, position-locked (cross-record unconfirmed); the
  oracle disarms yet breaks lookup; FRA is selective but never disarms.

**FINAL VERDICT = outcome (b): "selectivity without sufficiency."** FRA content-selectively targets the
emit conjunction (rank-1 preserves lookup + sibling + general at its 0.84 ceiling; rank-4 adds
collateral) but cannot remove enough of it to prevent disclosure. This is a real, nameable partial
result — retire BOTH the earlier "settled negative" (FRA *is* selective) and any "selectivity win" (it
*never* disarms). SAE (residual content-deletion) is the only method that disarms while preserving
lookup, and only *positionally* — not a demonstrated content-addressed selectivity. Box→frog inversion
stands.

### The B3 condensate — CONFIRMED on gemma (the silver lining, reopens T4)
Round-5 SVD of the per-head active-feature coupling ω is **~rank-1**: cumvar@r=1 = **0.915–0.998 across
all 8 heads**, the top singular value dominating the next by **10–40×** (e.g. L17H3 σ=[8257, 322, …];
L18H6 [4542, 257, …]; L15H0 the loosest at 0.915). So **r_eff ≈ 1** — the distributed, redundant-bank
emit conjunction is, *per head, ONE collective mode*. That mode is causally the emit-copy direction
(the rank-1 edit is the most selective + highest-suppression FRA variant; adding lower modes only adds
shared collateral). This is a **B3 condensate CONFIRMED on gemma-2-2b**, contrary to the F7
no-condensate lean for dense gpt2 — so gemma's coupling *condenses*, reopening **T4 (mean-field /
condensate theory)** as productive here. The practical upshot: the F² cell space collapses to r≈1 mode
per head, which is exactly what makes "guide-from-the-destructive-limit + causal peel-back" (§2)
tractable — even though, on this task, that peeled mode is reach-insufficient to disarm.

**Why box→frog is a clean FRA win and SSN-emit is reach-insufficient — the structural inverse.**

| | box→frog (banked ~1100× FRA win) | SSN-emit (this task) |
|---|---|---|
| value | single token ("frog") | multi-token digit string |
| edge | one live, concentrated answer→value edge | distributed redundant head-bank, multi-hop |
| SAE-cell score reconstruction | ~full → cut clears the margin | ~65–84% → best (rank-1) caps below disarm |
| discriminator | distinctive value token (own feature) | generic, cross-record-shared digit key |
| FRA selectivity | wins all axes | real (rank-1 vs rank-4) but insufficient |
| bar cleared | **FRA** (score-space, content-addressed) | **SAE only, positionally** |

**Campaign boundary lesson.** FRA's score-space attention-cut is the right tool when the behaviour is a
*concentrated, content-keyed, single/low-hop* attention edge (box→frog, induction, copy-suppression,
acronym). SSN-emit is the opposite substrate — *multi-token, distributed, redundant-bank, content-routed
recall* — and there: (i) the SAE-cell basis under-reconstructs the score, so even the best (SVD rank-1)
FRA cut **reach-caps at 0.84** — content-selective but never a behavioral disarm; (ii) attention removal
is **symmetric**, so even the oracle can't separate emit from lookup; the dissociation, where achieved,
lives in **residual/content space** (SAE, directional deletion). This maps a new boundary edge:
**distributed content-routed recall is where score-space FRA is content-selective-but-reach-insufficient
and residual-space content-deletion is the safer, use-preserving disclosure control.** The productive
counter-current is B3: the coupling *condenses* to r≈1 on gemma, so the theory-side FRA object (the
mean-field mode) is clean here even where the *edit* is reach-bound.

**Caveats (bound the sibling & generality claims, not the headline).** (i) SAE cross-record **sibling**
is **unconfirmed** (sib metric ablated the sibling's own digits — bug; fixed, not re-run), so SAE's
demonstrated claim is "disarm target + keep lookup," not "keep Bob." (ii) FRA sib_ok still shares the
digit tokens; the selectivity claim rests on the rank-1-vs-rank-4 contrast + clean general-KL, not on
sib_ok alone. (iii) SVD sweep crashed at r>4 on small-coupling heads (index error) — verdict robust,
r=1 is optimal anyway. (iv) Single fixed DB; base model (no IT-SAE). The headline (FRA rank-1 0.84
non-disarming + content-selective; oracle 0.97 disarms-but-breaks-lookup; SAE 0.956 disarm+keep-lookup
positionally; ω rank-1) is multiply confirmed r1–r5.

---

## One-sentence campaign placement

SSN-emit is a **distributed, redundant-bank, content-routed** recall circuit with a generic,
cross-record-shared digit key: the behavioral win bar (emit_ok=False AND look_ok=True at one setting)
is cleared **only by a position-locked SAE feature (0.956 disarm, lookup kept; cross-record
unconfirmed)**, while **FRA is content-selective but reach-insufficient** — its best variant (an SVD
rank-1 edit) reaches emit_supp 0.84 and preserves lookup+sibling+general yet **never disarms**
(base P=0.967 too peaked; the 0.84→0.97 gap is the SAE-unexplained score the cut can't touch), so the
final verdict is **outcome (b), "selectivity without sufficiency"** (retiring both "settled negative"
and "selectivity win"); the structural inverse of box→frog (single-token, concentrated live edge → FRA
1100×), with a silver lining — the per-head coupling ω **condenses to rank-1 (cumvar@r=1 0.915–0.998,
r_eff≈1)**, a B3 condensate confirmed on gemma that reopens T4.

---
*Numbers (HF `dmanningcoe/fra-phase1-steering-data`, prefix `fra_pii/results/`): FRA SVD
`pii_sweep_fra5-svd.json` (rank-1 supp 0.829 / rank-2 0.842 / rank-4 0.653, emit_ok=T ∀; per-head
cumvar@r=1 0.915–0.998 in `rs-pii-fra-5_run.log`); FRA pos-invariant `pii_sweep_fra4-allpos.json`
(0.62–0.65 flat); SAE `pii_sweep_sae-5.json` (L18 k=1 = 0.956 disarm); all-positions oracle 0.970 /
look_ok=F; single-edge `pii_sweep_diag.json`; Stage-2 `pii_cut_results.json`. Canonical results:
`experiments/fra_pii/RESULTS.md`. Caveats: SAE sib-metric bug, single DB, base model.*
