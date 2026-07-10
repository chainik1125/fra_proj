# fra_pii — THEORY NOTE: should FRA win on SSN-emit suppression?

Mechanistic account of the SSN task: suppress the model EMITTING a target in-context SSN
(name→ssn) while preserving (a) that SSN as a lookup KEY (ssn→name), (b) a sibling record's
SSN emission, (c) general behaviour. Structured as: a-priori prediction from the win-checklist →
reconcile with the Stage-2 NO-OP and the diag ORACLE → SAE-baseline / confirm-refute conditions.

TL;DR — **A-priori: LOSE (clause-4 fail, A→1).** **Empirically: LOSE for a blunter reason than
predicted — the located edge is not even load-bearing (clause 2/3), so FRA never reaches the
clause-4 test.** The diag ORACLE settles the reach-vs-content fork against BOTH: it is an
**edge-location failure**, not a reach failure and not (yet) a content-addressing failure.

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

**Diagnosis: EDGE-LOCATION failure (clause 2/3), NOT reach, NOT (yet) content-addressing.**
Three facts pin it:
1. **The oracle fails.** A position-patch that zeros the answer→digit scores outright — bypassing
   the SAE basis *and* the soft-cap entirely — drops emit only 0.02. So the SAE "just missing" a
   real edge is ruled out: there is no strong causal edge there for it to miss.
2. **Raw attention is a red herring.** L18H6 carries 0.75 raw answer→first-digit attention but 0.009
   causal drop — the exact raw-attn-vs-causal dissociation THEORY.md warns about ("raw attention
   selects positional/sink heads"). The candidate heads were chosen by raw attention; causally none
   of them carry emit (max drop 0.009).
3. **FRA is faithful, not reach-limited.** FRA-cut-ALL (M=ALL, c up to 60) lands at emit_supp 0.018
   = the oracle's 0.018, and the whole M×c frontier is flat. If FRA were reach-limited it would fall
   *short* of the oracle; instead it **matches** the oracle and inherits its null. FRA has full reach
   on this edge — the edge is simply not the behaviour's carrier.

Mechanistically: the first SSN digit is **not** produced by the answer position attending to the
in-context digits at these heads. It is retrieved by a route that doesn't pass through
answer→digit — almost certainly **multi-hop / positional**: within-record binding moves the ssn
onto the name/record token at earlier layers, and answer→name (or the template itself) carries it
forward, so by the answer step the digit is already resident. This is the greater-than pattern
(clause-3: real-looking QK edge, behaviour computed off it) crossed with positional identity — the
located key (digit) is wrong; the causal key is the **name/record token**.

**Consequence for the campaign fork:** clause 4 is never reached. You cannot ask "is the conjunction
content-selective?" of an edge that isn't load-bearing. The a-priori clause-4 LOSE stands as the
*eventual* verdict, but the *proximate* empirical cause is one clause earlier.

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

## 4. Recommended measurement (empirical agents are not running this)

The no-op is currently uninterpretable as an FRA verdict because the harness cut a non-load-bearing
edge. Before any further FRA/SAE frontier work, **re-locate the causal emit edge**:

1. **Causal key-position scan.** For the emit prompt, zero answer→k attention (one head-set) for k
   ranging over ALL key positions — name tokens, record delimiters, field-name tokens, digits — and
   record the drop in P(first digit). Expect the **name/record token**, not the digit, to be the
   load-bearing key. Re-rank heads causally on THAT edge (not raw attention).
2. **Attention-vs-resident test.** Ablate the answer position's *attention entirely* (or mean-patch
   the resid at the name position) to check whether emit is attention-routed at the final step at
   all, or whether the digit is already resident by the answer step (multi-hop). This decides
   clause-3.
3. **Query-feature identity across records.** Build FRA on the emit prompt for Marcus vs Bob; check
   whether the top query-side feature at the answer position is the SAME feature id. Same id →
   generic ROLE (clause-4 fail confirmed at the feature level, independent of any oracle). Different
   id keyed to the name → a content discriminator exists and the REFUTE path is open.
4. Only if 1–3 find a load-bearing, content-keyed edge does the FRA-vs-SAE matched-suppression
   comparison become defined.

---

## One-sentence campaign placement

SSN-emit is the entity-PII / box-retrieval failure of the win-checklist: identity is resolved
**positionally** and both nominal conjunction endpoints (emit-role query, generic-digit key) are
generic-role — so a-priori A→1 — but empirically the FRA cut is a **faithful no-op** because the
harness located a **non-load-bearing** edge (the diag oracle zeroing answer→digit drops emit only
0.02 and FRA-cut-ALL matches it exactly, ruling out reach), meaning the behaviour fails clause 2/3
*before* the clause-4 content-addressing question arises, and a position-patch on the correctly
re-located name/record edge — not a content-addressed cell-cut — is the tool this task calls for.

---
*Numbers: `pii_sweep_diag.json`, `pii_sweep_fra.json`, `rs-pii-diag_run.log`, Stage-2
`pii_cut_results.json`/`rs-pii-cut5_run.log`, `fra_win/out/pii_sibling/shared_endpoint_t2.json`
(HF `dmanningcoe/fra-phase1-steering-data`, prefix `fra_pii/`).*
