# fra_pii — THEORY NOTE: should FRA win on SSN-emit suppression?

Mechanistic account of the SSN task: suppress the model EMITTING a target in-context SSN
(name→ssn) while preserving (a) that SSN as a lookup KEY (ssn→name), (b) a sibling record's
SSN emission, (c) general behaviour. Structured as: a-priori prediction from the win-checklist →
reconcile with the Stage-2 NO-OP and the diag ORACLE → SAE-baseline / confirm-refute conditions.

TL;DR — **A-priori: LOSE (clause-4 fail, A→1).** **Empirically: LOSE for a blunter, earlier reason —
a THIRD case the fork "reach vs content-addressing" doesn't contain.** The diag ORACLE (which
**upper-bounds any single-QK-edge score-cut**) fails: zeroing answer→ssn-digit attention across the
whole retrieval head-set drops emit only 0.02. So the emit of a retrieved value is **not reducible
to a single answer→key-token QK edge at all** — this is the **clause-3 "direct consumption" boundary
in a new guise** (the value is assembled/transported off the single-edge axis: multi-hop
answer→name→value, distributed heads, or MLP-mediated multi-token digit copy). FRA is a QK-score-cut,
so when no single QK edge carries the behaviour, FRA has **no surface** — before the clause-4
content-addressing question even arises. Not reach (FRA matches the oracle exactly), not (yet)
content-selectivity (never reached).

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

**FRA-vs-SAE at matched suppression:** currently *undefined* — FRA has no operating point that
suppresses emit (frontier flat at 0.02), so there is no matched-removal point at which to compare
collateral. The comparison is vacuous until the emit edge is correctly re-located. If/when it is:
FRA's *only* structural advantage is lookup-preservation (it would cut the emit-query's *reading*
of the source, not the digit *content*, so lookup's match survives), while it *loses* the sibling
axis (generic conjunction). SAE/ablation/oracle win sibling by positional gating and lose lookup.

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

## 4. What a proper re-locate must FIND for FRA to have any surface (FRAAgent is running it)

The no-op is uninterpretable as an FRA *verdict* until the edge is re-located, because the harness
cut a non-load-bearing edge. But the oracle result already sets the bar. For FRA to have surface at
all, the re-locate must clear a **three-tier** gate — each tier is a strictly harder ask, and the
current data says tier 1 is not yet met:

- **Tier 1 — SURFACE (a defined suppression operating point): a single, or small enumerable,
  answer→key-token QK edge on which the ORACLE SUCCEEDS.** Run a **causal key-position scan**: for
  the emit prompt, zero answer→k attention (per head-set) for k over ALL key positions — name
  tokens, `{`/field-name/delimiter tokens, digits — and record the drop in P(first digit). Also test
  the **earlier within-record binding edge** (ssn-field-slot-query × digit-key at the layer where the
  ssn is bound onto its carrier). If NO single edge (or small union) makes the oracle suppress, the
  transport is genuinely distributed/multi-hop/MLP → **FRA has no surface, full stop** (position-patch
  or content-ablation is the only tool, and this is the final verdict). *This is the tier the flat
  0.02 frontier says is currently unmet on the answer→digit axis.*
- **Tier 2 — DIRECT CONSUMPTION (clause 3): the emit must BE that edge's OV-transported content.**
  Confirm the oracle on the re-located edge *suppresses* (not merely dents), and that emit is not
  reconstructed by a downstream MLP. **Attention-vs-resident test:** mean-patch the resid at the
  carrier (name/record) position, or ablate the answer position's attention entirely, to check
  whether the value is already resident before the answer step (multi-hop) vs freshly copied.
- **Tier 3 — CONTENT DISCRIMINATOR (clause 4, the WIN, not just surface): the load-bearing edge's
  discriminating endpoint must be DISTINCTIVE CONTENT, not generic role.** The only candidate in this
  task is the **NAME token** as the key of the binding edge — (query × Marcus-name-key) is selective
  for Marcus vs Bob; (ssn-field-slot-query × generic-digit-key) is not. **Query/key-feature identity
  across records:** build FRA on the emit prompt for Marcus vs Bob; if the top load-bearing endpoint
  feature is the SAME id for both → generic ROLE → clause-4 fail (A→1) even with surface. Only a
  name-keyed load-bearing edge opens the REFUTE path.

**Net:** Tier-1 miss ⇒ no surface (current reading). Tier-1 pass, Tier-3 miss ⇒ FRA can *suppress*
but not *win* (A→1, fires on siblings; same as synth-boxes). All three ⇒ the only world where FRA
beats the baselines here. FRA-vs-SAE matched-suppression is undefined until at least Tier 1 clears.

---

## One-sentence campaign placement

SSN-emit is a-priori a clause-4 loss (positional identity, generic-role endpoints, A→1), but the
*proximate* empirical cause is one clause earlier and blunter — a **clause-3 direct-consumption
failure**: unlike single-token box→frog retrieval (banked ~1100× win, where one answer→value edge
OV-copies the whole value), the multi-token SSN inside a nested multi-record JSON is
assembled/transported off the single-QK-edge axis, so the diag oracle — which upper-bounds *every*
QK-score-cut — drops emit only 0.02 and FRA-cut-ALL matches it exactly, meaning **no single edge
carries the behaviour for FRA to act on**; a position-patch (or content re-location) on the true
carrier, not a content-addressed cell-cut, is the tool, and FRA gets no surface here unless a
re-locate finds a single, load-bearing, name-keyed binding edge.

---
*Numbers: `pii_sweep_diag.json`, `pii_sweep_fra.json`, `rs-pii-diag_run.log`, Stage-2
`pii_cut_results.json`/`rs-pii-cut5_run.log`, `fra_win/out/pii_sibling/shared_endpoint_t2.json`
(HF `dmanningcoe/fra-phase1-steering-data`, prefix `fra_pii/`).*
