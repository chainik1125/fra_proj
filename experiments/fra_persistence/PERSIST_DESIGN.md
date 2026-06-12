# PERSIST_DESIGN.md — the concrete, pre-registered design for the persistence toy (Phase A)

*DESIGN/THEORY agent. Spec = `CAMPAIGN.md` (THE THESIS / OBSTACLE / TOY / LOCKED M1–M3 + WIN/FAIL).
This sharpens that spec into a runnable design + the formal M2 metric, confirms/refines the locked
thresholds, and names the single biggest misleading-WIN and misleading-FAIL risks. NO GPU here; the
EVALUATOR (Phase B) builds + runs. Reuses `fra/core/fra._build_fra_result` + the `j1/j2/j6/j7` hook
machinery verbatim — same gpt2-small + `gpt2-small-res-jb` resid_pre toolkit as induction.*

---

## 0. ONE-PARAGRAPH FRAME (what this measures and why it's the crux)

The thesis: an FRA cell-cut on the dominant `(qFi × kFj)` cell of a word-association `A→B` is a
**weight-space, support-gated edit** — applied *unconditionally* wherever those two SAE features fire,
it should remove `A→B` at EVERY appearance of A (novel contexts, unseen positions), with no
inference-time detection, while leaving A's other uses intact. A token-mask can't (it must detect A at
each occurrence); a feature-ablation can't (it kills A entirely). **The non-trivial obstacle** (from the
induction anchor: the dominant cell was the OFF-diagonal `qF423 × kF13032`, distinct query/key features):
the SAE may pick *different* features for "the same" A→B at different contexts. If so, one cell-cut
misses appearances → persistence needs the union of cells (bounded+identifiable = still a win;
unbounded/context-dependent = a real, important FAIL). **M2 is the foundational measurement: how
consistent is the A→B cell across A's appearances?** M1 (persistence) and M3 (comparison vs token-mask /
union / benign-use) are only interpretable *given* M2.

Honest prior (from induction R²<0, off-diagonal dominant cell, top-3 cells carrying only ~20% of edge
score): the **INFORMATIVE-NEGATIVE is the most likely outcome** — SAE inconsistency limits single-cell
persistence. The design is built so that negative is *cleanly measured and itself valuable* (it
quantifies the obstacle and motivates path-2's cleaner decomposition), NOT a noisy null.

---

## 1. THE TOY, MADE CONCRETE

### 1.1 The association A→B (single-token, novel/in-context — NOT parametric)

- **A and B must each be a single GPT-2 token AND novel/low-frequency**, so the copy behaviour is
  *in-context induction*, not a parametric bigram the model already knows. We do NOT hand-pick " dax"/
  " blicket" blindly — the EVALUATOR runs a **token-validity + induction-copy gate** (§1.5) and selects
  the A/B pair from a candidate pool that passes it.
- **Candidate pool (pre-registered construction, not the final pair):** draw single-token candidates two
  ways and union them, then filter (§1.5):
  1. *Random mid-vocab tokens* (the induction regime that already works in `j1`): `torch.randperm(40000)+1000`,
     keep those whose decoded string round-trips to a single token under `tok.encode(decoded)` (length 1).
  2. *Nonce-like real tokens* (rare but lexical, so carriers read naturally): from a fixed list of
     low-frequency single-token strings — e.g. `" dax"," blicket"," wug"," fep"," zorp"," glorp",
     " trell"," kresh"," vung"," plим"`-style nonces that tokenize to length 1 (EVALUATOR keeps only
     the length-1 ones; the random-token branch is the guaranteed fallback so the experiment can ALWAYS
     run even if every nonce is multi-token).
- **B is chosen from the SAME pool, disjoint from A** (so `P(B|A)` baseline ≈ 0 absent induction — B is
  not a generic high-prior continuation). Pre-register: **B must have parametric-prior `P(B | "…A")` <
  0.02** in a *no-prior-occurrence* prompt (A appears once, never followed by B) — this certifies the
  copy is in-context, not parametric. Reject the pair otherwise.
- **Pre-register 3 A→B pairs**, not one: `{(A1,B1),(A2,B2),(A3,B3)}` all passing §1.5, to show M2/M1 are
  not a single-pair artifact. Headline numbers are reported per-pair AND pooled.

### 1.2 The carrier templates (the STRESS design — this is where misleading-WINs die)

Each prompt embeds the induction pattern **`… A … B … [carrier] … A`** so that:
- the FIRST `A…B` is the induction primer (establishes A→B in context),
- the SECOND `A` (the "probe-A") is the query whose copy-prob `P(B | probe-A)` is the ground truth,
- the carrier text between/around varies so **probe-A appears at many positions and many left-contexts**.

**THE STRESS RULE (locked):** the carriers MUST vary probe-A's *immediate left context* (the preceding
1–3 tokens) and its absolute position, so that A's SAE feature *could* drift if the SAE is
context-sensitive. If probe-A always sits in the same local n-gram (e.g. always "the A"), its SAE feature
is trivially constant and M2 reports a fake-high consistency — the **misleading WIN** (§5). Concretely
each carrier draws probe-A's left neighbour from a varied set: `{". ", " the", " a", " and", " when",
" near", " after", " my", " some", " then"}` and varies the number of filler tokens before probe-A.

**Carrier template family (pre-registered; `{A}`/`{B}` substituted, `{f*}` = filler tokens drawn from a
fixed neutral filler bank of common single tokens):**

```
T1 (adjacent primer):   "{A} {B}. {f1} {f2} {f3}. {lc} {A}"
T2 (long carrier):      "{A} means {B}. {f1} {f2} {f3} {f4} {f5} {f6} {f7} {f8}. {lc} {A}"
T3 (intervening clause):"The {A} gave a {B}. {f1} {f2} {f3}, {f4} {f5} {f6}; {lc} {A}"
T4 (list carrier):      "{A} {B}, {f1} {f2}, {f3} {f4}, {f5} {f6}. {lc} {A}"
T5 (question carrier):  "Why {A} {B}? {f1} {f2} {f3} {f4} {f5}. {lc} {A}"
T6 (double-distractor): "{A} {B}. {g1} {h1}. {f1} {f2} {f3} {f4}. {lc} {A}"   # g1/h1 = OTHER nonce decoys
```

- `{lc}` = the varied left-context token (the STRESS lever).
- `{f*}` = filler tokens sampled (fixed seed) from a **neutral common-token bank** that does NOT contain
  A or B (the EVALUATOR builds the bank from frequent single GPT-2 tokens, e.g. `" people"," time"," was"," with"," it"," they"," some"," many"," day"," place" …`).
- T6's `{g1}{h1}` injects OTHER novel tokens (decoy associations) so the head must content-address A,
  not just "the most recent nonce."
- **probe-A is always the LAST token** (the next-token prediction is the ground-truth target), so
  `P(B | probe-A)` is read from the final-position logits — clean, judge-free.

### 1.3 LOCATE vs HELD-OUT split (disjoint carriers AND positions — the generalization test)

Per the carry-over rule "extraction data disjoint from eval" (the val-extract DoM leak):

- **LOCATE set** = templates **{T1, T2, T3}** × the left-context set × filler seeds → **n_locate ≈ 20**
  appearances. Used ONLY to find the dominant cell(s) and (for M3-union) the top-k.
- **HELD-OUT set** = templates **{T4, T5, T6}** × *different* left-context tokens (split the `{lc}` bank
  in half: LOCATE uses half, HELD-OUT uses the other half) × *different* filler seeds → **n_holdout ≈ 20**
  appearances. Used ONLY to test persistence (copy-prob drop) and consistency.
- This makes held-out differ from locate in **template structure, left-context token, filler content,
  and absolute position** — a genuine generalization test, not a re-skin.
- **Total ≈ 40 appearances** of probe-A across contexts (≥ the "30–50" the spec asks for). Pooled over 3
  A→B pairs → ~120 appearances for the distribution stats. The induction head L5H5 (re-confirmed argmax in
  `j1`/induction anchor) is the locus; the same multi-head set `[(5,5),(6,9),(5,1),(7,10),(7,2)]` from
  `j6/j7` is available as a robustness cross-check (report L5H5 as primary).

### 1.4 GROUND-TRUTH metric (judge-free)

For an appearance with probe-A at the final position `q`:
- `copyprob = softmax(logits[q])[B_token]` — the induction copy-probability, exactly the `j1/j2` metric
  but read at the probe-A position with target = B.
- **Removal fraction** for an intervention: `rem = 1 − copyprob_intervened / copyprob_clean` (per
  appearance, then averaged; clipped to `[0,1]` for reporting, raw kept).
- Sanity floor: only appearances with `copyprob_clean ≥ 0.15` enter the removal stats (the model must
  actually be doing the induction there — same `base_resp` guard as `j6`).

### 1.5 The induction-copy GATE (run FIRST, before any M1/M2 — pair-selection, pre-registered)

For each candidate (A,B) and the 6 templates:
1. Both A and B encode to exactly one token; A ≠ B.
2. `mean copyprob_clean(B | probe-A)` over a screening set of carriers **≥ 0.30** (the model reliably
   induction-copies B — same bar as `j7`).
3. Parametric-prior check: in a *no-primer* prompt where A appears but B never follows, `P(B | …A) <
   0.02` (certifies in-context, not parametric).
4. **A's benign-use check:** A must also appear in ≥1 *non-A→B* carrier where its top-1 logit is NOT B
   (so "benign use of A" is well-defined for the M3 preservation control).
- Pick the **3 pairs with the highest gate-2 copyprob** that pass 1/3/4. If <3 nonce pairs pass, fill
  from the random-mid-vocab branch (guaranteed to pass, per `j1`). **Record the chosen pairs + their
  gate numbers in the results JSON** (anti-post-hoc: pairs are frozen before M1/M2 are computed).

---

## 2. M2 — THE SAE-CONSISTENCY METRIC (the crux), FORMALIZED

For every probe-A appearance `a` (final-position query `q_a`, against the induction key position `k_a` =
the position of B in that appearance's primer), the FRA tensor gives the per-feature-pair contributions
`S[q_a, k_a, i, j]`. **Build the per-appearance dominant cell + the distributions below. Everything is
computed on the SAME `_build_fra_result` call the EVALUATOR already uses (`top_k=None`, resid_pre).**

### 2.1 Per-appearance dominant cell (exact extraction)

For appearance `a`, restrict the sparse FRA to the induction edge `(q_a, k_a)` (the locations where
`qq==q_a & kk==k_a`, exactly as `j2.edge_idx`). Aggregate over that single edge:
- `cell_score_a[i,j] = Σ_{n at edge a} val[n]·𝟙[ii[n]=i, jj[n]=j]` (one cell may have multiple terms;
  coalesced so usually one).
- **Dominant cell** `(i*_a, j*_a) = argmax_{i,j} |cell_score_a[i,j]|`. (Match the anchor convention:
  rank by **|score|**, the score-magnitude carried, NOT raw — so a strongly-*negative* anti-edge cell is
  not hidden.)
- **edge_coverage_a** = `|cell_score_a[i*,j*]| / Σ_{i,j}|cell_score_a[i,j]|` — what fraction of THIS
  appearance's edge the top cell carries (a within-appearance concentration; if low, even one appearance
  isn't single-cell — a stronger negative than cross-appearance drift).
- **Record per appearance:** `(i*_a, j*_a, cell_score_a[i*,j*], edge_coverage_a, copyprob_clean_a)`.

> Multi-edge subtlety (locked): each appearance has ONE induction edge (one primer B), so `k_a` is
> unambiguous. The EVALUATOR identifies `k_a` as the key position whose token == B in that prompt (there
> is exactly one B). No ambiguity, unlike a generic sentence.

### 2.2 CONCENTRATION (the headline M2 number)

Over all N appearances (pooled, and per LOCATE/HELD-OUT), tally the multiset of dominant cells
`{(i*_a, j*_a)}`:
- **top1_coverage** = `(# appearances whose dominant cell == the globally-most-frequent cell) / N`. This
  is the spec's "does the top-1 cell carry the association at ≥X% of appearances."
- **Cumulative coverage curve** `cov(k)` = fraction of appearances whose dominant cell is in the
  **top-k** most-frequent cells, for `k=1,2,3,5,10`. Report `cov(1), cov(3), cov(5)` explicitly.
- **n_cells_for_90** = smallest `k` with `cov(k) ≥ 0.90` — the union size needed to catch ~all
  appearances. This single integer is the **bounded-vs-unbounded verdict**: small (≤3) = bounded win;
  large/≈N = context-dependent/unbounded fail.
- **Weighted variant (report both):** weight each appearance by `cell_score_a[i*,j*]` (its causal
  magnitude), not just count — so a cell dominant only at low-copyprob appearances doesn't inflate
  coverage. Headline = the **count** version; the magnitude-weighted version is the robustness check.

### 2.3 q-side vs k-side feature consistency (the diagnostic that EXPLAINS a low concentration)

The anchor's lesson: the cell is OFF-diagonal, so q-feature and k-feature can drift *independently*.
Decompose the (in)consistency:
- **q-side consistency** = top1 frequency of the **query feature** `i*_a` alone (marginal over j):
  is " A at the query position" the same SAE feature across contexts? `qtop1 = max_i (#{a: i*_a=i})/N`.
- **k-side consistency** = top1 frequency of the **key feature** `j*_a` alone: `ktop1 = max_j (#{a:
  j*_a=j})/N`. (k is the "token-that-was-copied / prev-was-A" feature at B's position — the anchor's
  `kF13032`.)
- **Joint vs marginal:** if `top1_coverage ≪ qtop1·ktop1·N`-ish (joint much less concentrated than the
  product of marginals would predict), the q and k drifts are *correlated* (context picks a coordinated
  pair); if `top1_coverage ≈ qtop1` and `qtop1 < ktop1`, the **q-side is the culprit** (A's query-feature
  drifts with left-context — exactly what the STRESS carriers probe). Report which side dominates the
  inconsistency; this is the most scientifically informative byproduct and the direct handle for path-2.
- **Drift-vs-stress correlation:** because we logged each appearance's `{lc}` (left-context) and
  template, report whether `i*_a` changes *with* `{lc}` (the intended stressor) — a contingency
  table `{lc} × i*_a`. If A's query feature is constant across all `{lc}`, that's the honest "SAE is
  consistent here" finding; if it tracks `{lc}`, that's the obstacle, *localized to its cause*.

### 2.4 The decision rule (one cell suffices / union / unbounded)

- **"One cell suffices"** iff `top1_coverage ≥ 0.70` AND the top-1 cell on the LOCATE set is also the
  top-1 on HELD-OUT (the cell generalizes, not memorized).
- **"Bounded union suffices"** iff `n_cells_for_90 ≤ 3` (small, identifiable union) even if top-1 < 0.70.
- **"Context-dependent / unbounded"** iff `cov(3) < 0.40` OR `n_cells_for_90 ≥ 0.5·N` (the union is as
  big as the data — no compression). This is the INFORMATIVE-NEGATIVE branch.

---

## 3. M1 (PERSISTENCE) + M3 (COMPARISON), MADE RUNNABLE

All interventions reuse the **content-addressed score-delta hook** from `j2`/`j6`/`j7`: build a
`[seq,seq]` delta `d` and subtract it from `hook_attn_scores[0,H]` (multi-head: subtract per head). The
ONLY difference across interventions is *which* `(q,k)` entries `d` is nonzero at — that is the whole
point (unconditional feature-gated vs position-gated).

### 3.1 LOCATE the cell (on the LOCATE set only)

- Run FRA per LOCATE appearance, get each `(i*_a, j*_a)` (§2.1). The **located cell** `(I, J)` = the
  most-frequent dominant cell on LOCATE (top-1). For the union variants, also keep the LOCATE top-3.

### 3.2 M1 — the UNCONDITIONAL cut (the FRA-specific operation)

For a HELD-OUT appearance, build `d` by summing the FRA contributions of the **located cell `(I,J)`
wherever those two features fire** — i.e. over ALL `(q,k)` pairs in that held-out prompt where
`ii==I & jj==J` (NOT just the known induction edge). This is `j2.pairs_delta([(I,J)])` applied with the
LOCATE-derived `(I,J)`, on the held-out prompt's own FRA. Subtract via the hook; read
`copyprob_intervened(B|probe-A)`. **Report `rem_holdout` = mean removal over held-out appearances.**
- *This is the persistence claim:* a cell located on T1/T2/T3 removes A→B on T4/T5/T6 at novel
  positions/left-contexts, with no per-occurrence detection — the cut fires wherever its features fire.
- Also report `rem_locate` (in-distribution) as the ceiling; persistence ratio = `rem_holdout/rem_locate`.

### 3.3 M3a — TOKEN-MASK competitor (must FAIL on held-out)

The deployable token-mask: cut attention **to A's position(s) at the LOCATE positions only** — i.e.
zero the head's score from the probe-query to the key position(s) of A *as identified on the LOCATE set*.
Operationally, the token-mask is given the LOCATE primer's `(q,k)` induction edge and masks THAT
position pattern; on held-out it has **no detector**, so it either (i) masks nothing (positions differ →
`rem_holdout ≈ 0`) or (ii) if we grant it an oracle A-detector, it becomes the position-patch oracle
(`j2` baseline) which DOES work but **requires inference-time detection of A** — the thing FRA avoids.
Report BOTH framings:
- **token-mask (no detector)**: apply the LOCATE-position mask verbatim to held-out → expected
  `rem_holdout ≈ 0`. This is the honest competitor (a fixed weight-free position mask doesn't generalize).
- **token-mask (oracle detector)**: position-patch the held-out A's actual induction edge (`j2` oracle).
  This works but is flagged as *requiring detection* — not a weight edit. FRA's claim is "matches the
  oracle's removal WITHOUT the detector."
- **M3a verdict number:** `rem_holdout(FRA) / rem_holdout(token-mask, no detector)` — pre-registered to be
  ≥ 2× for the WIN (the spec's "≥2× worse held-out removal").

### 3.4 M3b — UNION-of-top-k (how many cells to catch all appearances)

Apply the cut for the LOCATE top-k cells (`k=1,2,3`) unconditionally on held-out; report `rem_holdout(k)`
as a function of k. The marginal gain `rem_holdout(3) − rem_holdout(1)` quantifies how much the
inconsistency costs and whether a *bounded* union recovers persistence (ties directly to M2's
`n_cells_for_90`).

### 3.5 M3c — BENIGN-USE preservation control (association-specificity)

Construct **benign carriers**: prompts where A appears but is NOT in an A→B induction context (no B
primer; A used in an ordinary continuation, from gate-4). Metrics under each intervention:
- **collateral_KL** = KL(clean ‖ intervened) of the full next-token distribution at A's benign
  positions (the `j7` per-position KL).
- **benign top-1 preserved** = fraction of benign positions whose argmax next-token is unchanged.
- Compare three interventions: **FRA cell-cut** (should be ~0 collateral — it only fires on the A→B
  cell), **token/feature-ABLATION of A** (zero A's SAE query feature `i*` everywhere → should wreck A's
  benign uses, high KL), **linear steer on A's direction** (the `j6/j7` ActAdd-cue baseline → broad
  collateral). Pre-registered: **FRA collateral_KL ≪ feature-ablation collateral_KL** (the spec's
  "FRA collateral on non-A→B << feature-ablation"); report the ratio.

### 3.6 Numbers to report (the results JSON contract for the EVALUATOR)

```
chosen_pairs:        [(A,B,gate_copyprob,parametric_prior), ×3]
per_pair & pooled:
  M2: top1_coverage, cov(1), cov(3), cov(5), n_cells_for_90,
      qtop1, ktop1, q_vs_k_culprit, cell_hist (top-10 cells w/ counts),
      lc_x_qfeat_contingency, locate_top1_cell, holdout_top1_cell, top1_locate==top1_holdout?
  M1: rem_locate, rem_holdout, persistence_ratio(rem_holdout/rem_locate),
      base_copyprob_locate, base_copyprob_holdout
  M3a: rem_holdout_tokenmask_nodetector, rem_holdout_tokenmask_oracle, FRA_vs_mask_ratio
  M3b: rem_holdout(k=1,2,3), union_gain
  M3c: FRA_collat_KL, featablate_collat_KL, linsteer_collat_KL, benign_top1_preserved_each, ratios
  robustness: same headline numbers on the 5-head set (secondary)
VERDICT: WIN / INFORMATIVE-NEGATIVE / AMBIGUOUS  (by §4 rule)
```

---

## 4. CONFIRM / REFINE the LOCKED WIN/FAIL (with justification)

The CAMPAIGN.md WIN requires, jointly: (a) top-1 (or k≤3) cell-cut removes **≥70%** at HELD-OUT; (b) SAE
**concentration high (top-3 ≥ 80%)**; (c) token-mask **fails on held-out (≥2× worse)**; (d) A's benign
uses preserved (FRA collateral ≪ feature-ablation). INFORMATIVE-NEGATIVE = top-1 covers <40% / held-out
needs a large/unbounded union.

**I CONFIRM the structure and all four WIN clauses, with three refinements (justified):**

1. **CONFIRM the ≥70% held-out removal (M1a) and ≥2× token-mask gap (M3a) and benign-preservation
   (M3c) as written.** These are the FRA-specific, non-deflatable clauses (generalization +
   no-detector + specificity) and they are the heart of the thesis. Do not loosen.

2. **REFINE clause (b): make it a TWO-NUMBER gate, top-1 AND cov(3), with the bounded-union escape
   explicit.** As written, "top-3 ≥ 80%" mixes two regimes. Lock instead:
   - **Clean WIN (single-cell persistence):** `top1_coverage ≥ 0.70` AND `rem_holdout(k=1) ≥ 0.70`.
   - **Bounded-union WIN (still a real, weaker win — name it as such):** `top1_coverage` in `[0.40,
     0.70)` BUT `cov(3) ≥ 0.80` AND `n_cells_for_90 ≤ 3` AND `rem_holdout(k≤3) ≥ 0.70`. This is the
     spec's "bounded+identifiable = still a win" branch, now operationalized. The single-cell WIN is the
     headline; the bounded-union WIN is reported as "persistence holds via a small identifiable cell-set
     (k≤3), not a single cell."
   - *Justification:* the induction anchor (off-diagonal, top-3 carry only ~20% of edge score) makes a
     clean single-cell WIN unlikely; splitting the gate prevents a near-miss from being mislabeled and
     captures the genuinely-valuable bounded-union outcome the spec already anticipated.

3. **REFINE the INFORMATIVE-NEGATIVE boundary to remove the gap and add a magnitude floor.** Lock:
   - **INFORMATIVE-NEGATIVE iff** `cov(3) < 0.40` OR `n_cells_for_90 ≥ 0.5·N` OR
     `rem_holdout(k≤3) < 0.40`. (The held-out-removal floor catches the case where the cell is *concentrated
     but causally weak* — exactly the anchor's "selective but weak, 20% of edge" failure mode, which a
     pure consistency metric would miss.)
   - **AMBIGUOUS** = the band between WIN and INFORMATIVE-NEGATIVE (e.g. `cov(3)∈[0.40,0.80)` with
     `rem_holdout∈[0.40,0.70)`): report as "partial persistence," do NOT claim either verdict, hand to
     RED-TEAM/PLANNING. (Pre-registering AMBIGUOUS is the anti-post-hoc move — it forbids rounding a
     middling result up to WIN.)
   - *Justification:* the original FAIL spec was silent on the [40%,70%] top-1 / [40%,80%] cov(3) middle
     and silent on the concentrated-but-weak case the anchor actually produced. These two additions make
     the verdict total and honest.

4. **ADD a pre-registered SANITY gate that must pass for ANY verdict to count (not WIN-specific):**
   `base_copyprob` (clean induction) ≥ 0.30 on both LOCATE and HELD-OUT, and the located cell is
   **non-sink** (reuse the anchor's CCF non-sink screen: the q/k features are NOT in the >50%-of-generic-
   sentences active set). *Justification:* if base copy-prob is low the removal fraction is noise; if the
   "dominant cell" is a sink/positional feature, any "consistency" is trivial (a sink fires everywhere) —
   that would be a misleading WIN. This gate is mandatory and reported up front.

**Net:** thresholds 70% / 80% / 2× / benign-≪ are CONFIRMED as the bar; the refinements add (i) the
bounded-union WIN tier, (ii) a total + magnitude-floored negative boundary, (iii) an AMBIGUOUS band, and
(iv) a mandatory base-copyprob + non-sink sanity gate. None of these loosen the WIN; they sharpen what
counts and close honesty gaps.

---

## 5. THE BIGGEST MISLEADING-WIN and MISLEADING-FAIL RISKS (for RED-TEAM + EVALUATOR to guard)

### 5.1 SINGLE BIGGEST MISLEADING-WIN: trivially-constant A feature (n-gram lock-in / sink)

If probe-A always sits in the **same local n-gram** (e.g. always preceded by " the"), its SAE
query-feature is constant *for a trivial reason* — local-context memorization, not a genuine
"A-content is one feature." Then `top1_coverage` is ~1.0 and persistence "works," but the claim
"FRA pins the A→B association persistently" is **fake**: you'd get the same constant cell for any
fixed-n-gram token. A second route to the same false win: the dominant cell is a **sink/positional
feature** (fires everywhere, so cutting it removes induction-everywhere and looks persistent + specific).

**Design defenses (built in above):**
- **The STRESS RULE (§1.2):** carriers *deliberately* vary probe-A's left-context `{lc}` and position so
  the SAE feature *could* drift. High consistency *despite* varied left-context is the real claim; the
  `{lc} × i*_a` contingency table (§2.3) certifies the feature is invariant to the stressor, not locked
  to an n-gram.
- **Non-sink gate (§4.4):** the located cell's q/k features must pass the CCF non-sink screen.
- **Decoy associations (T6):** other nonces in-context force content-addressing of A, not "most-recent-
  nonce."
- **RED-TEAM ask:** re-run M2 with the STRESS lever *removed* (fixed `{lc}=" the"`). If consistency only
  appears in the fixed-n-gram condition, the WIN is the artifact — report that delta.

### 5.2 SINGLE BIGGEST MISLEADING-FAIL: cell-indexing / edge-identification fragmentation

The likeliest *fake negative*: the per-appearance dominant cell is recorded as "different" across
appearances when it is **really the same association**, due to (i) the FRA top-k truncation splitting a
near-tie between two co-dominant cells differently per appearance (the anchor's top-3 carry only 20% — so
ranks are unstable), (ii) mis-identifying the key position `k_a` (B can appear more than once if a
template repeats it; or BOS/position offsets shift indices), or (iii) using raw `top_k` (the FRA built
with `top_k=20` per position could drop A's feature at some positions, fragmenting the cell set). Any of
these inflates the cell count → fake "context-dependent/unbounded" FAIL.

**Design defenses:**
- **Use `top_k=None` (all features)** in `_build_fra_result` for the M2 decomposition (as `j2/j6/j7` do)
  — no truncation-induced fragmentation. (top-k is only a speed knob; gpt2-small sequences are short.)
- **Tie-aware concentration:** report `cov(k)` and the magnitude-weighted variant, not just top-1, so a
  2-way near-tie reads as `cov(2)≈1`, NOT as inconsistency. Also report the **stability of the dominant
  cell to ±1 in the rank** (is cell #2 the same set across appearances?).
- **Unambiguous `k_a`:** templates place exactly one B per primer; the EVALUATOR asserts exactly one
  key position has token==B and errors loudly otherwise (no silent mis-edge).
- **Canonicalize indices** off the BOS-prepended token list used for the forward pass (same indexing the
  hook uses), and assert `probe-A == final position`.
- **RED-TEAM ask:** before accepting INFORMATIVE-NEGATIVE, verify it survives the magnitude-weighted +
  tie-aware metric and the all-features (`top_k=None`) decomposition — i.e. the fragmentation is *real*
  (the SAE genuinely uses different features), not a binning artifact. The honest negative is "the SAE
  uses genuinely distinct, causally-comparable cells per context"; the artifact is "ranks wobble on a
  near-tie."

### 5.3 (secondary) the concentrated-but-weak trap

The anchor's actual result: cell is selective (129×) but carries only ~20% of the edge → cutting it
barely moves behavior. A consistency-only metric would call this a WIN ("top-1 covers 90%!") while
`rem_holdout` is ~0.2. **Guarded by the §4.3 `rem_holdout(k≤3) ≥ 0.40` floor in the negative boundary and
the `≥ 0.70` removal in the WIN** — concentration without causal removal is explicitly NOT a win.

---

## 6. HARNESS NOTES FOR THE EVALUATOR (so Phase B is a thin build)

- **Reuse verbatim:** `from fra.core.fra import _build_fra_result`; the FRA call is
  `_build_fra_result(model, L, H, feats, sae.W_dec.float(), dev, top_k=None, rms_activations=x_hat,
  dec_norms=None, chunk_size=16)` with `feats = sae.encode(resid_pre).float()` and
  `x_hat = feats@W_dec + b_dec` — identical to `j2/j6/j7`. SAE = `SAE.from_pretrained("gpt2-small-res-jb",
  "blocks.5.hook_resid_pre")`. Model = `HookedTransformer.from_pretrained("gpt2")`.
- **The cut hook** = `j2.pairs_delta` + the `hook_attn_scores` subtract hook (single-head L5H5 primary;
  the multi-head `byL` loop from `j6/j7` for the robustness pass).
- **`pairs_delta([(I,J)])` is exactly the unconditional cut** — it sums FRA values for that `(i,j)` over
  ALL `(q,k)`, which is "fire wherever the features fire." That is the M1 operation; no new mechanism.
- **Per-appearance dominant cell** = `j2.edge_idx`/`edge_top_pairs(e, 1)` ranked by `|val|`.
- **Compute:** gpt2-small is tiny — one short job per A→B pair on one small GPU (L4), pod `rs-persist-*`,
  HF prefix `fra_persist/{code,results}`, `RP_API_KEY_MATS`. `ckpt()` partial-JSON inside the appearance
  loop + traceback-upload (carry-over rule). Parse-gate before upload. Budget-trivial; no grid needed.
- **Determinism:** fix all seeds (token sampling, filler draws, `{lc}` split). Record chosen pairs +
  gate numbers in JSON BEFORE M1/M2 so pre-registration is auditable.

---

## 7. EXPECTED OUTCOME (honest)

Given induction's off-diagonal dominant cell, R²<0, and top-3-carry-only-20%, the **most probable result
is the INFORMATIVE-NEGATIVE**: the q-side feature for A drifts with left-context (the STRESS lever bites),
`top1_coverage` lands in the 30–60% range, `n_cells_for_90` is moderate-to-large, and single-cell
`rem_holdout` is weak — so persistence requires a union whose size measures the obstacle. **That negative
is the deliverable:** it quantifies *how* SAE inconsistency limits path-1 persistence, localizes it to
the q-side vs k-side, and hands path-2 a concrete target (a decomposition where A's content is one stable
feature). The design ensures: (i) if a clean or bounded-union WIN exists, the gates catch it and the
STRESS/non-sink screens certify it isn't an n-gram/sink artifact; (ii) if it's a negative, the tie-aware
+ all-features + magnitude-weighted controls certify it's a *real* SAE-inconsistency finding, not a
binning artifact. Either way the result is clean and pre-registered.
