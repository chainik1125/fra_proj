# CIRCUITS.md — FRA × Circuit-Tracing: the selected circuits + pre-registered "FRA adds X" claims

*SELECTION+THEORY agent, FRA × CIRCUIT-TRACING track (`CAMPAIGN.md`), 2026-06-12. This is the SPEC the
EVALUATOR runs against. Each circuit below carries a CONCRETE, FALSIFIABLE "FRA adds X" claim tied to one of
the A1–A5 criteria, a FAITHFULNESS metric (the top-k cells must reconstruct the head's QK score and/or its
causal task effect — not cherry-picked), and a runnability note (model + SAE + which existing toolkit it
re-points). I do NOT run GPU or touch git — the orchestrator commits, the EVALUATOR runs.*

---

## 0. THE THESIS, RESTATED AS A TEST (so every claim below is falsifiable the same way)

Classic circuit tracing localizes WHICH heads carry a behavior and reads off the attention PATTERN, but treats
the QK SCORE `S[q,k]` as a black box (only intervention = head/edge ablation). FRA decomposes
`S[q,k] = Σ_{μν} u^μ_q u^ν_k ω_{μν}` into CONTENT×CONTENT feature-pair CELLS. So a "FRA adds X" claim is only
real if it clears BOTH gates, on the operating regime, with a symmetric red-team:

- **FAITHFULNESS (necessary, run FIRST per circuit):** the top-k cells reconstruct the head's pre-softmax QK
  score on the circuit's defining edge. Concretely (the `j1_induction_explore.py` pattern, verbatim): sum the
  sparse FRA tensor over its feature dims → `recon[q,k]`, compare to `cache["...hook_attn_scores"][0,H]`.
  Report **two numbers**: `corr(all causal scores)` and the edge-restricted `R² of S from top-k cells`
  (1 − ‖S_edge − Σ_top-k cells‖² / ‖S_edge − bias‖²). A decomposition that does not reconstruct the score is a
  story, not a result — that circuit is dropped, not narrated.
- **CAUSAL (the "adds" payload):** the A-criterion claim — cut the specific top-k CELL(s) vs ablate the whole
  HEAD/edge, on a ground-truth task metric. The cell-cut is the per-edge score-delta subtracted at
  `hook_attn_scores` (the `j1`/`j2`/`j13` `pair_delta_map` machinery, already written).
- **RED-TEAM (symmetric, mandatory):** the dominant cell must be the HYPOTHESIZED content, not (i) a
  generic-role/sink feature (false positive), nor (ii) noise that fails to reconstruct (false negative). The
  CCF non-sink-mass screen (`g_screen.py` `ccf()` + `sink_set()`) is the standing false-positive guard.

**Falsifier template (applies to every circuit):** the claim FAILS if — the FRA decomp does not reconstruct
`S` on the edge (low R²/corr), OR the dominant cell is a generic-role/positional/sink feature rather than the
hypothesized content conjunction, OR cutting the cell does not move the task metric more selectively than the
head-ablation baseline.

---

## 1. TOOLKIT INVENTORY (what EXISTS — the EVALUATOR re-points, does NOT rebuild)

| capability | where | reuse for circuits |
|---|---|---|
| 4D FRA tensor `S[q,k,i,j]` (sparse) | `fra/core/fra.py::_build_fra_result` / `get_sentence_fra_batch`; `fra/fra_func.py` mirror | per-head QK cell decomposition, all circuits |
| reconstruction faithfulness (sum cells → `recon[q,k]`, corr vs `hook_attn_scores`) | `experiments/fra_win/jobs/j1_induction_explore.py` (lines ~82–100) | the FAITHFULNESS gate, all circuits |
| surgical cell-cut + selectivity matrix `M[target,affected]` | `j2_selectivity.py` (`pair_delta_map`, `patched_copyprob`) | the A3 surgical-cut test, all circuits |
| IOI name-mover END→IO harness + cell-cut + ActAdd baseline | `j13_ioi.py` | IOI directly (already touched) |
| CCF (non-sink content-mass) + LBNR (cut-edge load-bearing R) screen | `g_screen.py` (`ccf`, `sink_set`, `causal_heads`, `cut`) | the red-team false-positive guard + head-find, all circuits |
| binding circuit #0 oracle cell-cut (R_gen / sibling-bleed / order-shuffle content-vs-position) | `experiments/fra_organisms2/jobs/binding_precheck.py` + `BINDING_LOG.md` | the A2 content-vs-position TEMPLATE, all circuits |

**Models + SAEs the toolkit supports as-is** (do NOT pick a circuit outside this set):
- **gpt2-small** + `gpt2-small-res-jb` residual SAEs at `blocks.{L}.hook_resid_pre` (used in `j1`/`j2`/`j13`/
  `g_screen`). RoPE off, LayerNorm path. **TINY → cheap → run first.** This is where induction, IOI,
  docstring, successor, greater-than all live.
- **gemma-2-2b** + `gemma-scope-2b-pt-res-canonical` (RoPE + softcap + RMSNorm, all handled in `fra_func.py`).
  Heavier; reserve for cross-model confirmation only.

**Consequence for selection:** all five selected circuits are run on **gpt2-small + res-jb**, the toolkit's
home turf. No new SAE training, no new model port. The EVALUATOR copies `j1`'s reconstruct + `j2`'s cell-cut +
`g_screen`'s CCF, changes the prompt builder + the edge `(Q,K)` + the task metric.

---

## 2. THE SELECTED CIRCUITS (ranked: faithfulness-anchor first, then richest-added-insight)

Ordering rationale: **(#0 binding)** is the WORKED example — already run, A2+A3 both banked — the template for
the claim shape. **(#1 induction)** is the FAITHFULNESS ANCHOR: cleanest QK, run FIRST to prove the FRA decomp
reconstructs a known-simple QK and the cell-cut is selective; it is also the banked behavioral win so A3 is
already partly in hand. **(#2 IOI)** is the richest added insight (what does S-inhibition MATCH?). **(#3
docstring)** is a clean attn-only A1/A2. **(#4 greater-than)** is the deliberate NEGATIVE control (FRA should
show the QK adds little). Successor heads are listed as a stretch alternate, not a core pick (justified in §4).

---

### CIRCUIT #0 (WORKED — fold-in template) — IN-CONTEXT BINDING (Mixing Mechanisms)

- **THE CIRCUIT:** in-context variable binding / bound-entity retrieval. Paper: *Mixing Mechanisms* (Gur 2025).
  Model: **gemma-2-2b-it** (the one exception to the gpt2-small rule — it is already done). Key head(s):
  **L22H4** (dominant retrieve-bound-entity head; mean answer-step P-drop 0.562), secondary **L18H6** (0.156).
  Published-pattern story: at the "Who has the {value}?" answer step, the head attends from the query position
  to the ENTITY token bound to the queried value, and copies it.
- **THE OPEN QUESTION the pattern leaves unresolved:** is that QK edge addressed by entity/value CONTENT
  (lexical), or by the binding's SLOT/POSITION? The attention pattern alone cannot tell these apart — both
  predict "attend to the right entity" on the canonical prompt. This is the A2 content-conjunction question.
- **THE "FRA ADDS X" CLAIM (A2 content-conjunction-ID + A3 surgical-cut):** FRA's QK decomp surfaces an
  (entity-feature × value-feature) content cell, and cutting that cell is CONTENT-addressed and surgical.
  **RESULT (banked, run-2 `8aqp5q9l62v0tk`):** order-shuffle audit `drop_content 0.952` vs `drop_position
  0.0008`, **29/29 = 100% content-addressed**; oracle cell-cut `R_gen = 0.900` (answer-step load-bearing,
  inverse of the injection signature R_prefill); **sibling-bleed 0.0009** (cutting the target binding's cell
  leaves the other three bindings' retrieval intact → surgical, A3). FRA resolved the lexical-vs-positional
  question the pattern couldn't, and the cell-cut is far more selective than head-ablation (which would kill
  all four bindings).
- **FAITHFULNESS:** the order-shuffle is itself the faithfulness+causal test (content cut moves the metric,
  position cut does not, 29/29). Reconstruction-R² on L22H4's edge was not separately reported → the only
  open item to backfill if this becomes a headline figure.
- **STATUS:** DONE. Use as the claim-shape template for #1–#4: *"the pattern says 'attends to X'; FRA says
  WHICH content cell; the cell-cut is content-addressed AND surgical (sibling-clean); head-ablation is not."*

---

### CIRCUIT #1 — INDUCTION HEADS (the FAITHFULNESS ANCHOR — run FIRST)

- **THE CIRCUIT:** induction. Paper: Olsson et al. 2022 (in-context learning / induction heads). Model:
  **gpt2-small**, SAE `gpt2-small-res-jb` @ `hook_resid_pre`. Key head: **L5H5** (the `j2`/`j1` induction head;
  EVALUATOR re-confirms argmax over `mean attn on induction edge` per `j1`, top-8 printed). Published-pattern
  story: prev-token head writes "the token before me" into position; the induction head's QK then MATCHES the
  current token against the key that holds "I am preceded by the current token," attends `[A][B]...[A]→[B]`,
  and copies `[B]`. The textbook clean QK.
- **THE OPEN QUESTION:** the pattern says "matches prev-token" but does the QK do **token-identity match**
  (the literal copied token's feature) or a generic **position/recency** feature? And is the head doing ONE QK
  job or several (A1)? The pattern can't separate token-content from a same-distance positional stripe.
- **THE "FRA ADDS X" CLAIM (A5 quantitative-faithful-account — PRIMARY — + A2 token-vs-content):**
  1. **A5 (the anchor):** the top-k FRA cells RECONSTRUCT L5H5's pre-softmax score on the induction edges with
     **high R²** and **per-edge the dominant cell is a SELF cell `i == j`** (the same SAE feature on query and
     key side = the token-identity match), not a sink/positional feature. This VALIDATES that the FRA decomp
     faithfully recovers a known-simple QK — the precondition for trusting it on IOI/docstring.
  2. **A2:** the dominant cell tracks the COPIED TOKEN's content (it changes when the token changes), i.e.
     token-identity, not a fixed position-`Δ` cell.
  3. **A3 (already partly banked):** cutting edge t*'s top-m cells suppresses copying of token t* ONLY, with
     ~0 collateral on other tokens (the `j2` diagonal-dominance selectivity matrix `M[target,affected]`).
- **FALSIFIABLE PREDICTION + FALSIFIER:** dominant cells on the induction stripe are SELF (`i==j`) content
  cells; reconstruction corr/R² is high on the edges; the `M` matrix is diagonally dominant.
  **FALSIFIED if** — reconstruction R² is low (decomp is noise), OR the dominant cell is a sink/generic
  feature flagged by CCF, OR `M` is NOT diagonal (the "content-addressed cut" actually bleeds across tokens =
  it was positional all along). The honest red-team here is real: if the dominant cell is a same-`Δ`
  positional feature that merely co-varies with the token, A2 fails and the win is downgraded to "FRA confirms
  the position story," NOT a content win.
- **FAITHFULNESS METRIC:** (i) `corr(all causal S, recon)` AND edge-restricted `R²(S_edge | top-k cells)`
  ≥ 0.7 (anchor bar); (ii) causal: `M[target,affected]` diagonal-dominance ratio = mean on-diagonal Δcopyprob
  / mean off-diagonal Δcopyprob ≫ 1 (the banked ~15× selectivity should reproduce as the cell-cut number);
  (iii) cell-cut vs whole-head-ablation: head-ablation collateral / cell-cut collateral (the selectivity
  multiple).
- **RUNNABILITY:** fully in hand. `j1_induction_explore.py` already computes the reconstruction + dominant
  pair; `j2_selectivity.py` already computes the selectivity matrix. The EVALUATOR runs these two AS-IS to
  produce the anchor numbers (R², diagonal-dominance), then adds the A2 token-swap check (vary `R` tokens,
  confirm the dominant cell index tracks the token). **Cheapest possible first experiment in the whole track.**

---

### CIRCUIT #2 — IOI: WHAT DOES S-INHIBITION MATCH? (richest added insight)

- **THE CIRCUIT:** Indirect Object Identification. Paper: Wang et al. 2022. Model: **gpt2-small**, `res-jb`.
  Prompt: *"When Mary and John went to the store, John gave a drink to" → " Mary"*. Key heads (two distinct QK
  jobs, both candidates):
  - **Name-mover heads L9H9 / L9H6 / L10H0** — attend END→IO and copy the IO name (the `j13_ioi.py` edge).
  - **S-inhibition heads L7H3 / L7H9 / L8H6 / L8H10** — attend END→S2 and write a signal that *suppresses* the
    name-movers' attention to the DUPLICATED (subject) name, so the movers land on the IO instead.
- **THE OPEN QUESTION (THE prize, A2):** the published-pattern story for S-inhibition is "attends to the
  repeated name and signals the movers to avoid it" — but it is OPEN/ambiguous WHAT the S-inhibition QK
  actually MATCHES at the S2 key: **name-IDENTITY** (the literal token "John"), **POSITION** (the second-name
  slot), or **DUPLICATE-TOKEN** content (a "this token appeared before" feature, à la the duplicate-token
  head)? Wang's pattern-level analysis cannot resolve this — all three predict "attend to S2." This is exactly
  the binding lexical-vs-positional question, transplanted to the canonical circuit.
- **THE "FRA ADDS X" CLAIM (A2 content-conjunction-ID — PRIMARY — + A1 sub-mechanism):**
  - **A2:** FRA's QK decomp of an S-inhibition head's END→S2 edge is dominated by a **DUPLICATE-TOKEN /
    name-identity content cell** (query = "I am the END after a repeated name" feature; key = the S2-name
    feature), NOT a generic second-position cell. Decisive test (the binding order-shuffle transplanted): swap
    which name is duplicated / move S2 to a different slot — if the dominant cell follows the NAME-CONTENT it
    is identity/duplicate-addressed; if it follows the SLOT it is positional.
  - **A1:** the name-mover edge END→IO may decompose into MULTIPLE separable cells (a name-identity copy cell +
    a "this is the un-inhibited name" cell), showing the mover does more than one QK job.
- **FALSIFIABLE PREDICTION + FALSIFIER:** the S-inhibition END→S2 edge's top cells are name/duplicate-token
  content cells that survive the order-shuffle by CONTENT; cutting them recovers the name-mover's attention to
  S2 (raises logit_diff toward the no-inhibition baseline) MORE selectively than ablating the whole
  S-inhibition head. **FALSIFIED if** — the dominant S2 cell is a generic position/sink feature (CCF flags
  it), OR it does not reconstruct the edge score, OR the order-shuffle shows it is slot- not content-addressed
  (then S-inhibition is positional and FRA's "added" answer is the NEGATIVE one — still a real, publishable
  resolution of the open question, just the other way).
- **FAITHFULNESS METRIC:** (i) reconstruction R² of the S-inhibition head's END→S2 score from top-k cells;
  (ii) **causal, on the real task:** logit_diff(IO − S) recovery when the S2 cell is cut vs when the whole
  S-inhibition head is ablated — FRA's claim needs `Δlogit_diff(cell-cut) / Δlogit_diff(head-ablate)` to be a
  large fraction (the cell carries most of the head's inhibition effect) WHILE leaving the name-movers' other
  behavior intact (selectivity vs the `j13` ActAdd-on-IO baseline, already coded). The name-mover END→IO cut
  is already partly in `j13_ioi.py` (FRA reduces logit_diff via the QK edit) — extend it from "the edge" to
  "the named content cell" and add the S-inhibition head.
- **RUNNABILITY:** `j13_ioi.py` already builds the IOI prompts, finds name-mover edges, FRA-decomposes the
  END→IO edge, cuts the top pairs, and compares to ActAdd. The EVALUATOR ADDS: (a) the S-inhibition heads
  (L7/L8) and their END→S2 edge; (b) the order-shuffle name/slot audit (copy the binding `binding_precheck.py`
  content-vs-position logic); (c) the CCF non-sink screen on the dominant cell. Second experiment after the
  induction anchor passes.

---

### CIRCUIT #3 — DOCSTRING (clean attn-only A1/A2)

- **THE CIRCUIT:** the docstring circuit. Source: Heimersheim & Janiak, *A circuit for Python docstrings in a
  4-layer attention-only transformer* (Interpretability/Apollo). Model: **attn-only-4L** is the native model,
  BUT to stay on the supported toolkit the EVALUATOR reproduces the docstring TASK on **gpt2-small** (the task
  generalizes; gpt2-small completes a `def f(self, param_a, param_b):  """docstring ... :param param_` →
  `param_b` next-argument-name pattern). Key heads: the "argument-mover" / induction-like heads that, at the
  `:param ` position, attend back to the NEXT not-yet-documented argument name in the signature and copy it.
- **THE OPEN QUESTION:** the pattern says "attends to the next argument" — but is the QK matching the
  ARGUMENT-NAME content, the comma/positional DELIMITER structure ("the token after the last :param"), or an
  induction-style prev-token match? A1: is one head doing both the delimiter-tracking and the name-copy, or
  are these separable cells?
- **THE "FRA ADDS X" CLAIM (A1 sub-mechanism + A4 cross-circuit-reuse):**
  - **A1:** the argument-mover's QK decomposes into ≥2 separable cells — a DELIMITER/structure cell (`:param`
    or `,` feature) that does the slot-tracking and an ARG-NAME content cell that does the copy — demonstrating
    the single head runs two QK jobs (the pattern shows only "attends to the right argument").
  - **A4:** the copy cell is the SAME induction self-cell (`i==j` token-identity) found in Circuit #1 —
    induction cells RE-USED inside the docstring circuit (the cross-circuit feature-cell reuse criterion). This
    is the cheapest A4 evidence in the track because #1 already labels that cell.
- **FALSIFIABLE PREDICTION + FALSIFIER:** the argument-mover edge has a structure/delimiter cell AND a
  name-content cell; the name-content cell index MATCHES an induction self-cell from #1; cutting the name cell
  drops next-arg copy-prob while cutting the delimiter cell mis-routes the slot (a double dissociation).
  **FALSIFIED if** — the edge is a single inseparable cell (no A1), OR the "reused" cell is not actually the
  same feature index as #1 (no A4), OR neither cut moves the task (no reconstruction/causality).
- **FAITHFULNESS METRIC:** reconstruction R² of the argument-mover score from top-k cells; causal double
  dissociation — `Δ(next-arg copy-prob)` under name-cell-cut vs delimiter-cell-cut should be off-diagonal
  (each cut hits its own job); A4 = exact feature-index overlap between the docstring copy cell and the #1
  induction self-cell.
- **RUNNABILITY:** gpt2-small + `res-jb`, same harness as #1. The only new code is the docstring prompt
  builder (a templated `def`/`:param` string) and the next-arg-name metric; the FRA decomp + cell-cut + the
  #1 cell-index comparison are all reuse. Third experiment. Lower priority than IOI but the cheapest A4.

---

### CIRCUIT #4 — GREATER-THAN (the deliberate NEGATIVE control — the honest boundary)

- **THE CIRCUIT:** greater-than / year completion. Paper: Hanna et al. 2023. Model: **gpt2-small**, `res-jb`.
  Prompt: *"The war lasted from the year 17YY to the year 17" → a two-digit number > YY*. The mechanism is
  **largely MLP** (the MLPs compute the `> YY` numeric comparison); attention's job is mostly to MOVE the `YY`
  digits to the final position, not to compute the comparison. v1 flagged it **G-post/D2** (the load-bearing
  computation is post-attention / MLP).
- **THE OPEN QUESTION:** does the QK of the digit-moving heads carry any CONTENT-specific comparison signal, or
  is it a generic "move the year digits" positional/structure edge? If FRA is honest, it should show the QK
  adds LITTLE here — the content work is in the MLP, off the QK manifold.
- **THE "FRA ADDS X" CLAIM (A5, the NEGATIVE direction — the boundary):** FRA should show the digit-moving
  head's QK is dominated by a GENERIC year-digit / positional cell (NOT a magnitude/ordinal "greater-than"
  content cell), and that cutting any single QK cell does NOT move the `> YY` probability mass (because the
  comparison is computed downstream in MLP). The "added value" is the HONEST BOUNDARY: FRA correctly reports
  that QK-resolution buys little where the circuit is not QK-driven, calibrating that the positive claims on
  #1/#2/#3 are not just storytelling-on-any-circuit.
- **FALSIFIABLE PREDICTION + FALSIFIER:** the year-head QK reconstructs as a positional/digit-structure cell
  with HIGH reconstruction (the QK is real, it just moves digits) but LOW causal cell-cut effect on the
  greater-than metric (cutting the cell does not change which year is preferred — the MLP recomputes). The
  CONTROL would FAIL ITS PURPOSE (i.e. greater-than would NOT be a clean negative) if a single QK cell encoded
  the ordinal comparison and cutting it flipped the prediction — in which case greater-than is actually a
  QK-content circuit and becomes a 5th positive. Either outcome is informative; the prediction is that it's
  the negative.
- **FAITHFULNESS METRIC:** reconstruction R² (expected high — the QK is a real digit-mover); causal cell-cut
  Δ`P(year > YY)` (expected ≈ 0 — the negative); contrast against the head-ablation Δ (expected: head-ablation
  hurts MORE than any cell, because moving the digits matters but no single content cell does). The diagnostic
  ratio is `cell-cut effect / head-ablation effect ≈ 0` here vs `≈ large fraction` on #1/#2 — that contrast IS
  the boundary result.
- **RUNNABILITY:** gpt2-small + `res-jb`. New code: the year-prompt builder + the `P(year > YY)` metric (sum
  probability over the valid greater years); everything else reuse. Run AFTER ≥2 positives land (it is only
  interpretable as a contrast). Fourth experiment.

---

## 3. THE RANKED RUN ORDER + THE RECOMMENDED FIRST EXPERIMENT

| rank | circuit | criterion | why this slot | cost |
|---|---|---|---|---|
| #0 | binding (gemma-2-2b-it) | A2 + A3 | DONE — the worked template (content-addressed 29/29, R_gen 0.90, bleed 0.0009) | banked |
| **#1** | **induction L5H5 (gpt2)** | **A5 (anchor) + A2 + A3** | **FAITHFULNESS ANCHOR — proves the FRA decomp reconstructs a known-simple QK before we trust it on IOI; banked behavioral win so A3 connects** | **tiny / first** |
| #2 | IOI S-inhibition + name-mover (gpt2) | A2 (richest) + A1 | the prize open question (what does S-inhibition MATCH?); `j13_ioi.py` already 60% there | small |
| #3 | docstring task on gpt2 | A1 + A4 | clean two-job head + the cheapest cross-circuit-reuse (reuses #1's labeled cell) | small |
| #4 | greater-than (gpt2) | A5-negative | the honest boundary / calibration control; only interpretable after ≥2 positives | small |

### RECOMMENDED FIRST EXPERIMENT (for PLANNING to confirm, EVALUATOR to run)

**Circuit #1, the induction faithfulness anchor — run `j1_induction_explore.py` + `j2_selectivity.py` AS-IS on
gpt2-small + `gpt2-small-res-jb`, and report the three anchor numbers:**

1. **FAITHFULNESS:** `corr(all causal S, FRA recon)` and the edge-restricted `R²(S_edge | top-k cells)` on
   L5H5's induction edges. **Anchor bar: R² ≥ 0.7 and the per-edge dominant cell is a SELF cell `i==j`.**
   *If this fails, the whole track is suspect — stop and debug the decomp before any other circuit.*
2. **A2 token-tracking:** re-run with a fresh random token set `R`; confirm the dominant cell INDEX tracks the
   copied token (content), and is not a fixed position-`Δ` cell. CCF non-sink check on it.
3. **A3 surgical-cut:** the `j2` selectivity matrix `M[target,affected]` diagonal-dominance ratio, and the
   cell-cut-vs-head-ablation collateral multiple (the banked ~15× should reproduce from the cell-cut).

This is the cheapest decisive experiment in the track (gpt2-small, two existing scripts, minutes on an L4),
and it gates everything else: a passing anchor licenses trusting the FRA decomp on IOI's harder open question;
a failing anchor means fix the reconstruction first. **Faithfulness FIRST, exactly per the carry-over rule.**

---

## 4. CIRCUITS CONSIDERED AND NOT IN THE CORE 5 (with justification)

- **Successor heads (Gould et al.) — STRETCH ALTERNATE, not core.** A2-ordinal is attractive (which ordinal
  feature does the QK match — "the day after," "the next number"?), but (a) the cleanest successor results are
  on Pythia/larger models, not gpt2-small + `res-jb`, raising the runnability cost (would need a SAE the
  toolkit doesn't already load), and (b) docstring already banks the "ordinal/next-in-a-sequence" flavor via
  the next-argument copy, on the supported stack, AND delivers A4 for free. **Slot successor only if a
  positive needs a 5th and gpt2-small shows a clean successor head; otherwise skip.**
- **Copy-suppression L10H7 (banked behavioral win) — OPTIONAL companion, not a circuit pick.** The organisms
  SYNTHESIS flags a copy-suppression QK feature-pair decomposition as a cheap characterization companion (gpt2
  + res-jb). It is a labeled-cell artifact on an already-banked win, not an open-question resolution, so it
  does NOT compete with #1–#4 for a core slot. The orchestrator MAY slot it as a low-priority writeup
  companion alongside #1 (same model+SAE, near-zero marginal cost) but it is not required.
- **Factual recall / ROME edits — EXCLUDED.** Resolved as a NO-GO in the organisms campaign (relation-keyed,
  not subject-keyed; D3 conjunction-recurrence). It is an MLP-pre-bake circuit, off the QK-resolution thesis.

---

## 5. WHY THESE FIVE ARE THE RIGHT SET (the meta-justification)

1. **They are all on ONE supported stack** (gpt2-small + `gpt2-small-res-jb`, plus the already-done gemma
   binding) — zero new SAEs, zero new model ports, maximal toolkit reuse. Budget-conscious by construction.
2. **They span the A-criteria without overlap:** #1 = A5 faithfulness anchor (+A2/A3), #2 = A2 prize (+A1),
   #3 = A1 sub-mechanism + A4 reuse, #4 = A5-negative boundary, #0 = A2+A3 worked template. Every criterion
   A1–A5 is exercised, and the NEGATIVE control (#4) is built in so the positives are calibrated, not
   storytelling.
3. **They are ordered faithfulness-first:** the anchor (#1) must reconstruct a known-simple QK before any
   harder open-question claim (#2 IOI) is trusted — the hard carry-over rule. The IOI open question (what does
   S-inhibition MATCH?) is the single richest thing FRA can ADD to the canonical circuit, but it is only
   credible AFTER the anchor passes.
4. **Each claim is falsifiable the same way** (§0 template): low reconstruction R² → drop; generic/sink/
   positional dominant cell → red-team kills it; non-selective cut → no A3. The IOI and binding claims can
   even resolve to the NEGATIVE (positional/slot-addressed) and still be publishable resolutions of a
   genuinely open question — the claim is "FRA RESOLVES it," not "FRA confirms the content story."
