# fra_pii — THEORY NOTE: should FRA win on SSN-emit suppression?

Mechanistic account of the SSN task: suppress the model EMITTING a target in-context SSN
(name→ssn) while preserving (a) that SSN as a lookup KEY (ssn→name), (b) a sibling record's
SSN emission, (c) general behaviour. Structured as: a-priori prediction from the win-checklist →
reconcile with the Stage-2 NO-OP and the diag ORACLE → SAE-baseline / confirm-refute conditions.

TL;DR — **A-priori: LOSE (clause-4 fail, A→1). Empirically: SETTLED NEGATIVE — no bounded FRA
surface.** The fork "reach vs content-addressing" has a hidden third leaf and SSN-emit lands in it.
The diag ORACLE **upper-bounds any single-QK-edge score-cut** (attention-edge ablation is the M=ALL
corner of FRA — §2 identity), and the causal re-locate scan makes it decisive: **name-oracle 0.000,
brace-oracle 0.000, digit-oracle 0.126 (top-12), best single head 0.009.** So the emit of a retrieved
value is **not a cuttable answer→key QK edge** — the answer→name/record multi-hop story is refuted,
and the digit axis is at best a weak **redundant head-bank** (Tier-1 miss: no bounded edge). This is
the **clause-3 direct-consumption boundary**: the value is assembled/transported off the single-edge
axis. FRA is a QK-score-cut, so it has **no surface** here — before clause-4 even arises. Not reach
(FRA-cut-ALL matches the oracle to 4 decimals), not content-selectivity (never reached). One open
label — the all-208-head digit ceiling — decides (a) non-selective redundant-bank vs (b) resident/MLP;
**either way the verdict is the same.**

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
| **4 CONJUNCTION-SPECIFIC w/ DISTINCTIVE-CONTENT discriminator** | **FAIL** | **both endpoints generic-ROLE.** (emit-role query × generic-digit key) recurs on every sibling's emit and overlaps digit-reading in lookup. This is the box-retrieval / entity-PII class (THEORY.md sharpened corollary). |
| 5 REACH | n/a until 1–4 hold | — |

**Magnitude-law prediction: A ≈ reuse(marginal)/reuse(conjunction) → ~1.** reuse(conjunction|eval)
is HIGH — the (emit-role × digit-key) conjunction is *not* rare on the eval distribution; it fires
on Bob's emission and on every digit-reading step. Same class as the shared-endpoint sibling test
(`fra_win/out/pii_sibling/shared_endpoint_t2.json`: FRA on-target removal ≈ 0, fra_on 0.757 ≈
red_base 0.756 — a degenerate "no-op at strong base") and box-retrieval (A=7.9× partial only at
weak base). No collateral WIN available by construction.

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

**Diagnosis: CLAUSE-3 (direct-consumption) boundary — a THIRD case, NOT reach, NOT (yet)
content-addressing.** The fork the harness pre-registered ("reach failure → oracle suppresses;
content failure → oracle weak") has a hidden third leaf: the oracle can be weak because **no single
answer→key-token QK edge carries the behaviour at all**. That is what we see. Three facts pin it:
1. **The oracle fails, and the oracle upper-bounds the whole class of QK-score-cuts.** A
   position-patch that zeros the answer→digit scores outright — bypassing the SAE basis *and* the
   soft-cap entirely — drops emit only 0.02. Any FRA cell-cut is a *weaker, content-addressed*
   member of the same "subtract from these QK scores" family; the oracle is its supremum. Oracle
   fails ⇒ the entire family fails ⇒ this is not "SAE missed a real edge" (reach) and not "the edge
   fires on siblings" (content) — it is *no single QK edge is the carrier*.
2. **Raw attention is a red herring.** L18H6 carries 0.75 raw answer→first-digit attention but 0.009
   causal drop — the exact raw-attn-vs-causal dissociation THEORY.md warns about ("raw attention
   selects positional/sink heads"). The candidate heads were chosen by raw attention; causally none
   of them carry emit (max drop 0.009).
3. **FRA is faithful, not reach-limited.** FRA-cut-ALL (M=ALL, c up to 60) lands at emit_supp 0.018
   = the oracle's 0.018, and the whole M×c frontier is flat. If FRA were reach-limited it would fall
   *short* of the oracle; instead it **matches** the oracle and inherits its null. FRA has full reach
   on this edge — the edge is simply not the behaviour's carrier.

Mechanistically this is the **clause-3 "direct consumption" failure generalised**: clause 3 demands
the behaviour BE a located edge's OV-transported content, not a downstream computation that merely
reads it (greater-than: real QK edge, MLP computes the output). Here the retrieved SSN value is
**assembled/transported by a process no single score-cut can disarm** — the plausible routes are
(i) **multi-hop**: within-record binding moves the ssn onto the name/record token at earlier layers,
then answer→name (or the template) carries it, so by the answer step the digit is already resident;
(ii) **distributed** across many weak heads/positions; (iii) **MLP-mediated multi-token digit copy**.
On any of these, cutting the answer→digit QK edge controls (at most) one hop of the transport, never
the emission. The located key (digit) is wrong AND the transport is not single-edge OV — so it fails
clause 3 whether or not a better key exists.

### The box→frog contrast — why some retrievals ARE cuttable and this one is not
The campaign's **banked retrieval win (~1100×)** is single-token forward retrieval — box→frog,
"the box contains a **frog**" / "key X → value **Y**". There the value is ONE token sitting AT the
attended key position, so a **single answer→value edge OV-copies the whole value**: the edge is
load-bearing (oracle/edge-cut works) and the value token is a distinctive-content discriminator
(A≈1100×, clause 3 and 4 both pass). SSN-emit breaks BOTH structural preconditions:
- **Multi-token value.** "531-42-8817" tokenises to `5,3,1,-,4,2,…`; no single answer→value edge
  transports the string. Even the *first* digit's prediction (P=0.967) is not carried by
  answer→first-digit (oracle 0.02) — it rides on a resident/structural signal, not a fresh copy.
- **Record-structure indirection.** The ssn is a *field* inside a JSON record keyed by name
  (`{"name":…,"ssn":…,"city":…}`) in a *multi-record* DB. Retrieval is name→record→ssn-field — a
  multi-hop bind that lands the value on an intermediate carrier before the answer step. Flat
  box→frog has no such indirection.

So the same head-line ("cut the answer→value edge") that works on box→frog is a **category error**
here: SSN-emit is entity-attribute *binding* (synth-boxes class: distributed/weak edge,
LBNR-R=−0.89, A=5133× artifact = not load-bearing), not flat single-token retrieval. Multi-token +
nested-record indirection is exactly what pushes a retrieval from the cuttable-single-edge regime
into the distributed/multi-hop regime where clause 3 fails.

**Consequence for the campaign fork:** clause 4 is never reached — you cannot ask "is the conjunction
content-selective?" of a behaviour that has no single load-bearing edge. The a-priori clause-4 LOSE
(A→1) stands as the *eventual* verdict, but the *proximate* empirical cause is clause 3.

### FRA ⊇ attention-edge ablation: the M=ALL identity (and the peel-back algorithm it licenses)
The oracle==FRA-cut-ALL coincidence is not luck — it is a **structural identity**, and it reframes
what FRA *is*. Cutting ALL cells at a token pair (q,k) subtracts

  Σ_{μν} u^μ_q u^ν_k ω^h_{μν}  =  the FRA-reconstructed content score  Ŝ^h[q,k].

In the perfect-reconstruction limit (the feature×error and error×error terms → 0) Ŝ = the full
pre-cap score S^h[q,k] (up to bias terms), so subtracting it at large c drives attention to k → 0 —
**exactly attention-token-pair ablation of the edge (q,k).** Therefore:

> **Attention-map ablation of a token pair is the M=ALL, large-c corner of FRA.** FRA ⊇
> attention-edge-ablation; FRA at M<ALL is a strictly finer, **content-addressed sub-family nested
> inside** that ablation (it removes only the cells whose conjunction you name, wherever it fires).

**Empirical confirmation on this task:** FRA-cut-ALL reproduced the position-patch oracle to the
measured precision (both 0.0184) — the reconstruction error on this edge is negligible, so the M=ALL
limit *is* the edge ablation. This makes the oracle a legitimate supremum for the whole FRA family on
this edge (the fact-1 argument above is exact, not approximate).

**Methodological corollary (the useful part — a concrete F5 mitigation).** F5 established that FRA
magnitude-ranking ⊥ causal effect, so **forward top-M selection is unreliable** (top cells need not
be the causal ones; here the whole M×c sweep is flat regardless of M). The identity licenses the
opposite algorithm — **guide from the destructive limit**: start at cut-ALL (= the oracle, causally
the maximal effect available at this edge) and **causally PEEL BACK** cells, pruning from the full
support to the minimal subset that still suppresses the target while sparing siblings/lookup. Trust
the **support** (which cells are non-zero at the edge), not the **ranking**; validate each pruning
step causally. This is exactly the F5 "union restrictor + causal-validation-inside" prescription with
a runnable procedure. (It does not manufacture a win on SSN-emit — the edge's cut-ALL ceiling is
itself too low / non-selective — but it is the general tool the null here motivated.)

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

**FRA-vs-SAE at matched suppression:** *structurally undefined* — and §4 shows this is now settled,
not pending. FRA has no operating point that suppresses emit (frontier flat at 0.02) and the
re-locate found no bounded edge to move to (name/brace 0.000; digit a weak redundant bank), so there
is no matched-removal point at which to compare collateral. SAE reaches suppression by deleting the
digit representation at source; FRA/attention-cut cannot reach it at any bounded edge. Both are only
*positionally* selective (act at the target's digit positions) — spare the sibling, break lookup —
because the digit key is generic (clause 4); neither is content-selective.

**CONFIRM "generic-digit-key → no FRA collateral advantage":** after re-locating a load-bearing
edge, FRA at matched emit-suppression shows sib_ok dropping and general/sibling collateral ≥ the
position-patch (A≈1 on the sibling axis) — i.e. the win-checklist clause-4 prediction holds once
you get to it. **The current no-op is a weaker, upstream confirmation**: even the oracle can't
suppress on the digit edge, so FRA has nothing to be advantaged *about*.

**REFUTE:** FRA reaches an operating point that suppresses target emit AND keeps sib_ok=1 AND
lookup=1 at matched removal while SAE/ablation cannot — this would require a distinctive-content
discriminator (e.g. the causal key is the **NAME** token, which IS distinctive content: an
(emit-role query × Marcus-name key) conjunction is selective for Marcus vs Bob). This is the one way
the task could flip to a win, and it is exactly what the re-location measurement below tests.

---

## 4. The re-locate result: Tier-1 UNMET on every key hypothesis (FRAAgent)

The gate for FRA to have any surface is **three-tier** (each strictly harder). FRAAgent's causal
key-position scan has now run it, and the answer→name hypothesis is **REFUTED**:

```
oracle key-position scan (drop in P(first digit) from zeroing answer→k attn):
  name-token oracle    = 0.000
  brace/struct oracle  = 0.000
  digit oracle (top-12)= 0.126        best single head = 0.009
```

- **Tier 1 — SURFACE (a single / small-enumerable answer→key QK edge on which the ORACLE SUCCEEDS):
  UNMET on all three key hypotheses.** name 0.000, brace 0.000 kill the multi-hop answer→name/record
  story outright — the value is *not* fetched by the answer step attending to a carrier token. The
  digit axis is where the (weak) routing lives, but no bounded edge carries it: best single head
  0.009, top-12 only 0.126. So **SSN-value emission is not a cuttable answer→key QK edge.** For the
  emit-role query × generic-digit key, this is also clause-4-doomed even where it has mass.
- **Tier 2 — DIRECT CONSUMPTION (clause 3):** contingent on the open sub-question below.
- **Tier 3 — CONTENT DISCRIMINATOR (clause 4, the WIN):** moot — the only distinctive-content
  candidate was the NAME key, and the name oracle is 0.000, so there is no name-keyed edge to be
  selective on. The digit key is generic ⇒ A→1 even if a surface existed.

### The one open sub-question, and the verdict conditional on it
The digit oracle's shape (tiny per-head, many heads, top-12 = 0.126 ≫ best single 0.009) is the
**STOCKTAKE-T1 redundant-head-bank signature** (distributed attention across a bank), not obviously
resident/MLP. FRAAgent is measuring the **all-208-head digit-oracle ceiling** to decide:

- **(a) all-head digit oracle → ~1  ⇒ REDUNDANT-BANK (attention-routed but distributed).** Emit *is*
  answer→digit attention, spread thin across the whole head bank. Via the M=ALL identity, FRA's
  cut-ALL **union across the full bank** would then suppress — so a *surface exists*, but only as the
  maximal union, and it is **non-selective by construction**: the key is a generic digit, so cutting
  the bank kills digit-reading everywhere (siblings, lookup, any number) → A→1. Peel-back (§2) finds
  no target-specific subset because there is no content discriminator to peel toward. Fails Tier 3.
- **(b) all-head digit oracle stays low  ⇒ RESIDENT / MLP.** The value is already in the stream by
  the answer step (multi-hop bind onto the digit-carrier at earlier layers) or MLP-assembled — a
  clean **clause-3 non-edge** negative. No QK surface at any head count. Fails Tier 1 outright.

**Either branch, the headline is the same:** SSN-emit fails at **Tier 1/2** — it is not a bounded
edge-routed behaviour (and even the redundant-bank surface is non-selective). Therefore **SAE
content-ablation, which deletes the digit REPRESENTATION at the source position, suppresses where the
entire QK-score-cut family (FRA and attention-edge ablation alike) cannot** — because it acts on the
value's existence in the residual, not on any single head's reading of it. And **neither is
content-selective**: the digit key is generic (clause 4), so both rely on *positional* gating (act at
the target's digit positions) for target-specificity, which spares the sibling but breaks lookup.
This is the exact inverse of **box→frog**, where a single-token value on a live answer→value edge
passes all tiers and FRA both suppresses and wins (~1100×).

**Net verdict:** FRA-vs-SAE matched-suppression is not merely undefined-pending-relocate — the
relocate has run and there is **no bounded FRA surface** (Tier-1 miss on name/brace; digit axis
either resident/MLP or a non-selective redundant bank). SSN-emit is a **settled negative** for the
content-addressed win, awaiting only the all-head ceiling to label it (a) vs (b).

---

## One-sentence campaign placement

SSN-emit is a-priori a clause-4 loss (positional identity, generic-role endpoints, A→1), but the
*proximate*, now-measured cause is one clause earlier and blunter — a **clause-3 direct-consumption
failure with no bounded QK surface**: unlike single-token box→frog retrieval (banked ~1100× win,
where one answer→value edge OV-copies the whole value and all tiers pass), the multi-token SSN inside
a nested multi-record JSON is assembled/transported off the single-QK-edge axis, so the causal
re-locate finds name-oracle 0.000, brace-oracle 0.000 and only a weak distributed digit bank
(top-12 = 0.126, best head 0.009) — attention-edge ablation being the M=ALL corner of FRA, this means
**no bounded FRA cut carries the behaviour**; SAE content-ablation (delete the digit representation at
source) suppresses where the whole QK-score-cut family cannot, and neither is content-selective
(generic digit key, clause 4) — a settled negative, with only the all-head digit ceiling left to
label it redundant-bank vs resident/MLP.

---
*Numbers: `pii_sweep_diag.json`, `pii_sweep_fra.json`, `rs-pii-diag_run.log`, Stage-2
`pii_cut_results.json`/`rs-pii-cut5_run.log`, `fra_win/out/pii_sibling/shared_endpoint_t2.json`
(HF `dmanningcoe/fra-phase1-steering-data`, prefix `fra_pii/`).*
