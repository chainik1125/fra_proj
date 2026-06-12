# FRA model-organisms SCREENING RUBRIC — the scorable FRA-cuttability checklist

*Theory agent, fra_organisms campaign, 2026-06-11. This is the SHARED ARTIFACT the screen agent (scores
each organism), the evaluator (designs the win-test), and the planning agent (picks candidates) all use.
It converts the converged FRA theory into an **a-priori-scoreable** per-organism checklist. It builds on,
and does not supersede, three prior documents: the FRA win-checklist and magnitude law (`../fra_win/THEORY.md`,
`../fra_win/FRA_PRINCIPLE.md`), the FRA-cuttability measurement checklist C1–C4 (`../fra_hierarchy/THEORY_HIERARCHY_v2.md`),
and the DIRECTION-vs-RELATION reframe that the EM + sycophancy negatives forced
(`../fra_hierarchy/BRAINSTORM_applications.md`). Written to inform, not persuade: every prediction below is
falsifiable, the scoring anchors are mechanical, and §6 states what would falsify the whole thesis.*

---

## 0. The one-paragraph predictor (read this first)

**FRA's surgical cell-cut beats a linear steer / weight-edit exactly when the behavior is a
*bilinearly-gated attention edge*: behavior that fires only on the CONJUNCTION of a query-content feature
and a key-content feature, both of which are independently reused across many contexts, where the
on-target pairing carries the behavior (load-bearing, non-redundant) and the model's downstream
*directly consumes* the attended content rather than recomputing it in an MLP.** When that holds, FRA edits
exactly the conjunction's support and nothing else, while any linear steer can only push a *marginal*
(a row or column of the bilinear form, broadcast to every residual reader) and so pays collateral on every
legitimate reuse of the endpoint. The selectivity advantage is quantitative:
**A ≈ reuse(marginal endpoint | eval) / reuse(conjunction | eval)** — ~10³–10⁴ when the conjunction is
unique, collapsing to ~1 when a *generic role* endpoint or a *direction-routed payload* makes the
conjunction redundant. **FRA loses on DIRECTION/payload/persona behaviors (G-post: EM, sycophancy-decision,
refusal-payload) because there is no score-space cause to cut — a single direction is already surgical.**

The rest of this document operationalizes that paragraph into **eight a-priori sub-questions (§1)**, an
**8-dimension 0–2 scoring rubric with a go/no-go threshold (§2)**, **per-family pre-registered
predictions (§3)**, the **selectivity win-test the evaluator must run (§4)**, **worked example scores so
the screen agent is calibrated (§5)**, and the **falsifiers (§6)**.

---

## 1. THE PREDICTOR — eight a-priori-scoreable sub-questions

These are the sub-questions the screen agent answers *from the paper / known mechanism alone*, before any
GPU. Each maps onto a win-checklist clause (`fra_win/THEORY.md`) or a measurement clause (C1–C4,
`THEORY_HIERARCHY_v2.md`). They are phrased so that **"yes" leans FRA-win** and the reasoning is the score
anchor in §2.

**Q1 — Is the behavior a RELATION between two CONTENT tokens, or a DIRECTION/payload?**
*The single most decisive question.* A RELATION reads content at position k and uses it at position q
("agree with *that* opinion", "Tom Cruise's *mother*", "copy the token after *X*"). A DIRECTION writes a
persona/style/refusal vector into the stream regardless of any specific other-token content (EM's misaligned
persona, the refusal direction, a "be terse" style). Only RELATIONS have a score-space cause to cut.
*Maps to: win-clause 1 (edge-routed) + the DIRECTION-vs-RELATION reframe. Failure mode if "direction":
OUTPUT-DIRECTION ROUTING — FRA has no target, DoM/SVD already surgical (A→1).*

**Q2 — Must the key-content be ATTENDED to produce the behavior (G-score), or is it computed post-hoc in
MLP/OV after attention has already run (G-post)?**
The crux that sank both prior negatives. G-score: changing *what gets attended* changes the behavior
(the agree/recall/copy decision is made AT the QK step). G-post: the model always attends to the content
(to comprehend it) and a downstream MLP/OV gate decides what to do with it — freezing/cutting the attention
edge leaves the behavior intact. The a-priori tell: **does the behavior require SELECTING among multiple
candidate key-spans?** If yes (which opinion, which subject's which relation, which prior occurrence) → likely
G-score. If the key is always attended and only the verdict varies → likely G-post.
*Maps to: win-clause 3 (direct consumption) + C-crux of THEORY_HIERARCHY_v2 §3.1. Failure mode: DOWNSTREAM
NONLINEAR CONSUMPTION (greater-than: edge reads the year, MLP does the `>`).*

**Q3 — Are BOTH endpoints BROAD/reused (so a linear steer pays collateral), and is the DISCRIMINATING
endpoint DISTINCTIVE CONTENT rather than a generic ROLE/position feature?**
The magnitude-law driver. Broad×broad = both query and key features fire across many contexts (a subject
feature reused across all its facts × a relation feature reused across all subjects). If an endpoint is a
*rare dedicated token*, a direction is already surgical → no FRA gap (the weight-baked sleeper, A≈1). If the
discriminating endpoint is a *generic role* ("the queried slot", "the entity position"), the conjunction
recurs across siblings and FRA fires on them too (positionally-resolved retrieval, sibling A=1.9×).
**Distinctive-content query is mandatory**: every confirmed win has one (induction query=token X,
copy-suppression query=about-to-predict-X, backdoor query=trigger content).
*Maps to: win-clause 4 (conjunction-specific, distinctive-content discriminator). Failure modes: CONJUNCTION
RECURRENCE (generic role) + rare-dedicated-token (A→1).*

**Q4 — Is the pairing LOAD-BEARING and NON-REDUNDANT — i.e. is cutting THIS conjunction the only way the
behavior fires, with no backup circuit and no redundant distributional cue?**
The clause CCF is blind to and that killed IOI. Two redundancy traps: (a) **circuit redundancy** — backup
heads restore the behavior (IOI name-movers, LBNR-R=−0.39); (b) **distributional redundancy** — the output
is derivable from many other cues the distribution provides (delimiter-matching: grammar gives many ways to
know a paren closes; refusal: many ways to know a request is harmful). A-priori tell: **is there exactly ONE
cue, or many redundant cues, for the behavior?** Tasks with a single planted/idiosyncratic association
(backdoor trigger, a specific learned fact) score well; tasks with overdetermined structural outputs score
poorly.
*Maps to: win-clause 2 (LBNR). Failure modes: REDUNDANT/SELF-REPAIRING + DISTRIBUTIONAL REDUNDANCY.*

**Q5 — Are BOTH endpoints BROAD (reuse-heavy) AND is the conjunction RARE on the eval support — i.e. is the
magnitude-law ratio A predicted ≫1?**
Distinct from Q3: Q3 asks "is the *structure* broad×broad with a content discriminator?"; Q5 asks for the
*predicted magnitude* A ≈ reuse(marginal|eval)/reuse(conjunction|eval). Score the EXPECTED A band:
~10³–10⁴ (unique conjunction, e.g. one fact, one acronym), ~10–10² (semi-shared), ~1.9× floor (shared/generic
endpoint). Note: A is **only interpretable once Q4 passes** (LBNR is a prerequisite — a non-load-bearing edge
has trivially-zero on-target effect and a meaningless collateral ratio, the binding "A=5133×" artifact).
*Maps to: the magnitude law. Drives the SIZE of the win, not the win/lose verdict (that is Q1–Q4).*

**Q6 — Is there a CLEAN GROUND-TRUTH metric (no LLM judge), and is collateral measurable on a held-out set
where the endpoints legitimately recur?**
The EM lesson: an LLM-judge alignment score manufactured a false positive (judge-bucketing + coherence-align
coupling). Score-routed wins must be demonstrated on **judge-free** metrics: exact-match on a planted payload,
factual-recall accuracy on held-out facts, regex format-compliance, binary refused/complied, sycophancy-flip
on factual questions. The collateral measurement needs a held-out set where the query-feature and key-feature
EACH legitimately recur (so the linear baseline actually pays the collateral the magnitude law predicts).
*Maps to: C4 + the corrected measurement ladder (THEORY_HIERARCHY_v2 §2.2). A dirty metric is not a
disqualifier but caps confidence; ground-truth is strictly preferred.*

**Q7 — Is the behavior LOCALIZABLE — does the routed effect concentrate in a few heads/cells (≥~60% of the
causal effect in ≤3 heads) carried by a distinctive content×content conjunction, rather than accruing
diffusely across many heads (distributed routing → no surgical cell)?**
The clause sycophancy-deference and EM both failed (sycophancy v1 maximally diffuse, 6 heads each un-flip 1
item). A-priori tell: known single-canonical-head circuits (copy-suppression L10H7, induction heads, IOI
name-movers as a *localization* positive even though it fails Q4) score well; behaviors known to be
distributed across the network (ICL task vectors, persona) score poorly.
*Maps to: C3 + win-clause 2's "across ALL heads that carry it". Measured CAUSALLY (rank heads by edge-cut
effect, never raw attention — raw attention selects sink/positional heads).*

**Q8 — Is there a STRONG existing surgical baseline to beat (ROME/MEMIT, the Arditi refusal-direction,
DoM/CAA, ablation), and is the ARTIFACT runnable on our stack (small open model + a usable SAE)?**
Two sub-parts, both *good* to score high. **Baseline-to-beat strength:** a behavior with a strong existing
surgical edit (ROME for facts, a linear refusal-direction) is a *better* test, not a worse one — beating
ROME's selectivity is a meaningful, publishable claim; "FRA works at all" against no baseline is weak
(`writing_instructions.md`: baselines are crucial, make them strong). **Runnability:** gemma-2-2b-it +
GemmaScope, gemma-2-9b-it, gpt2-small + res-jb SAEs, or Qwen-7B; a clean ground-truth eval that fits a small
pod. An interesting organism we cannot run on our stack scores low here regardless of theory fit.
*Maps to: feasibility + the baselines-are-crucial principle. Note the polarity: strong baseline = HIGH score
(good test), because beating a strong surgical baseline is the entire point.*

**How the eight compose.** Q1, Q2, Q4 are the **win/lose gates** (a "no" on any one predicts FRA loses —
they negate edge-routing, direct-consumption, and load-bearing respectively). Q3, Q5, Q7 set the **magnitude
and surgical feasibility** of a win. Q6, Q8 are the **demonstrability** axes (can we cleanly show it, against
a baseline worth beating). A candidate that passes Q1+Q2+Q4 with a distinctive-content discriminator (Q3),
a localizable few-head signature (Q7), a clean metric (Q6), and a strong baseline (Q8) is the target profile;
the proven wins (induction, copy-suppression, in-context backdoor, retrieval) all match it.

---

## 2. THE SCORING RUBRIC — 8 dimensions, 0–2 each, with anchors and threshold

The screen agent scores each organism on the eight dimensions below (each 0–2; one maps to each sub-question).
**Anchors are mechanical** — pick the row the organism best matches. Score from the *known mechanism / paper*,
a-priori; if a dimension is genuinely unknown, score 1 (uncertain) and flag it as the crux experiment.

| # | dimension (sub-q) | 0 (FRA-adverse) | 1 (uncertain / mixed) | 2 (FRA-favorable) |
|---|---|---|---|---|
| **D1** | **Relation-vs-Direction** (Q1) | pure direction/persona/style/payload (EM, refusal-payload, "be formal") | behavior has both a routing step and a direction payload (refusal = trigger+payload; format) | a relation between two content spans (factual recall, induction, coreference, injection-routing) |
| **D2** | **G-score vs G-post** (Q2) | content always attended; verdict is post-hoc MLP/OV (sycophancy-decision, EM-gating, greater-than's `>`) | plausibly G-score but the gating-locus is the open question (the crux) | selecting *among* key-spans is the behavior; attending IS the decision (induction-copy, retrieval, backdoor) |
| **D3** | **Broad×broad + distinctive-content discriminator** (Q3) | rare dedicated trigger token (direction already surgical) OR generic role/position discriminator (box/entity slot) | one broad endpoint, one borderline; or content discriminator present but partly shared with siblings | both endpoints broad/reused AND the discriminating (query) endpoint is distinctive content |
| **D4** | **Load-bearing & non-redundant** (Q4) | overdetermined output / known backup circuit (IOI backups, delimiter grammar-redundancy, refusal multi-cue) | some redundancy risk; single cue but plausible alternates | exactly one idiosyncratic association carries it (planted backdoor, a specific learned fact) |
| **D5** | **Predicted magnitude A** (Q5) | A≈1–2 (shared/generic endpoint, conjunction recurs) | A≈10–10² (semi-unique conjunction) | A≈10³–10⁴ (unique conjunction: one fact, one acronym, one trigger) |
| **D6** | **Metric cleanliness** (Q6) | LLM-judge only, coherence-coupled (EM-style) | judge-light / classifier (refused-string, alignment with threshold band) | exact-match / accuracy / regex / ground-truth flip (no judge) |
| **D7** | **Localizability** (Q7) | known-distributed (ICL task vector, persona across network) | unknown / plausibly few-head — needs the sweep | known single-canonical-head or few-head circuit (copy-suppression, induction, name-mover) |
| **D8** | **Baseline-to-beat × runnability** (Q8) | no real baseline AND/OR un-runnable on our stack | weak baseline (only ablation) and runnable; OR strong baseline but heavy artifact | strong surgical baseline (ROME/MEMIT, Arditi direction, DoM) AND runnable on gemma/gpt2 + SAE |

**Aggregate score** S = D1+…+D8 ∈ [0,16].

**Gate rule (mandatory, overrides the sum).** D1, D2, D4, D3 are the **four win/lose gates**, each negating
a distinct failure mode: D1=0 → output-direction routing (no score-space cause); D2=0 → downstream nonlinear
consumption (edge reads, MLP computes); D4=0 → redundancy (precise-but-inert cut); D3=0 → conjunction
recurrence, i.e. a rare-dedicated-token endpoint (a direction is already surgical, A→1) OR a pure generic
role/position discriminator (FRA fires on siblings, A→1.9× floor). If **any of {D1, D2, D3, D4} = 0**, the
candidate is **NO-GO** regardless of S — a high S elsewhere is the "high global α with no cuttable cell" trap.
Report it as a *pre-registered negative* (valuable as a control), not a candidate to run. (D3 is a gate on its
**0** anchor only — a D3=1 "borderline discriminator" is *not* a gate fail, it is the binding/RAG crux.)

**Go/no-go thresholds (on candidates passing all four gates):**
- **S ≥ 13** and all four gates ≥ 1 and at least two of {D1,D2,D3,D4} = 2 → **TIER-1 (run first).**
- **10 ≤ S ≤ 12** with all four gates ≥ 1 → **TIER-2 (run if Tier-1 exhausted or as a contrast).**
- **S ≤ 9** or any gate = 0 → **NO-GO / pre-registered negative control.**

**Tie-break (when S ties, among gate-passers):** prefer (a) higher D6 (cleaner metric — the EM lesson: metric
cleanliness is co-equal with P(attention-routed)), then (b) higher D8 (stronger baseline to beat), then (c)
higher D7 (localizability — the most common make-or-break in practice). Record each organism as a row:
`{name, D1..D8, S, gate_pass, tier, crux_dimension, proposed_FRA_variant, baseline_to_beat}`.

---

## 3. PER-FAMILY PRE-REGISTERED PREDICTIONS

For each behavior family: the predicted verdict, the **single clause that decides it**, the expected D-profile,
and a quantitative falsifiable number. Registered **before** any run. The known-G-post families (EM,
sycophancy, refusal-payload) are included as **controls** — a clean negative there validates the rubric.

### 3.1 Factual recall / knowledge-editing (subject × relation → object) — **FLAGSHIP, predict WIN**
- **Verdict: FRA WINS the SELECTIVITY test vs ROME/MEMIT and vs a linear subject-steer**, on the
  *subject→object transport* step. The deciding clause is **Q2/D2 (G-score)**: factual recall is
  attention-mediated — subject enrichment at the subject token, then an "extract-the-relation" attention move
  pulling the object to the last token (the Geva/Meng circuit). The (subject-content query × relation-content
  key) conjunction is broad×broad **by construction**: the subject feature is reused across all of that
  entity's facts, the relation feature across all entities.
- **Predicted D-profile:** D1=2, D2=2 (with a real risk the *extraction* is partly MLP — see falsifier),
  D3=2, D4=2 (a specific fact is one idiosyncratic association), D5=2 (A≈10³, unique conjunction), D6=2
  (held-out factual accuracy, no judge), D7=1→2 (name-mover heads are localizable), D8=2 (ROME/MEMIT is the
  strongest surgical baseline in interp — beating its selectivity is the meaningful claim). **S≈15, TIER-1.**
- **Quantitative prediction:** an FRA (subject × relation) cell-edit removes the target fact (target accuracy
  → <20%) while leaving (a) the SAME subject's OTHER relations within 5% of baseline accuracy and (b) OTHER
  subjects' SAME relation within 5%, at **lower collateral than ROME/MEMIT** on those held-out sets
  (predict ROME bleeds ≥15% onto the subject's other facts; FRA <5%). **Falsified if** the object is
  recomputed in an MLP downstream of the attention move (Q2 fails — edge-cut dents target accuracy <30%):
  then this is greater-than redux (edge reads, MLP computes) and FRA controls the attending, not the recall.

### 3.2 Induction / in-context copying / function vectors — **predict WIN (the proven anchor)**
- **Verdict: FRA WINS** — this is the *confirmed* win (`fra_win`, A≈15× on gpt2, ~16× retrieval on gemma).
  Deciding clause: **Q2/D2 trivially passes** (the copy decision IS the attention pattern: query=token X,
  key=token-after-previous-X). Included here so the screen has its calibrated **D=all-2 reference** (the top
  of the scale).
- **Predicted D-profile:** D1=2,D2=2,D3=2,D4=2,D5=2,D6=2,D7=2,D8=2 (DoM/ActAdd baseline). **S=16, TIER-1
  reference.** *Caveat:* function-vector / ICL-task-vector behaviors that are *distributed* (many-shot
  jailbreak) FAIL D4/D7 — distinguish single-edge induction (win) from distributed task-vector ICL (loss).
- **Quantitative prediction:** ≥15× collateral advantage at matched on-target removal (already measured for
  the narrow case). The OPEN sub-question is the **broad/class-level trigger** (any-country-name vs a fixed
  token): predict a group-LASSO family-union edit suppresses held-out class-member copying by ≥70% at
  <5% non-trigger collateral. **Falsified if** the class members share no cell structure (no recoverable
  family block → the hierarchy doesn't lift).

### 3.3 Entity-attribute binding / coreference — **predict WIN, but localizability is the risk**
- **Verdict: FRA WINS if localizable**, deciding clause **Q7/D7 (localizable)** with **Q3/D3 (distinctive
  content)** the runner-up risk. Binding is attention-mediated (who attends to which attribute) and is a
  RELATION (D1=2, D2=2). The risk: the discriminating endpoint may be a **generic role/position feature**
  ("the entity-slot"), which is the CONJUNCTION-RECURRENCE failure (entity-PII differential no-op at strong
  base; box-retrieval sibling A=1.9×). On gpt2-small the binding edge passed CCF (0.905) but was **weak**
  (attn 0.38, LBNR-R=−0.89 — distributed).
- **Predicted D-profile:** D1=2, D2=2, D3=1 (role-vs-content discriminator is the crux), D4=1, D5=1, D6=1
  (ground-truth bindable but eval is fiddly), D7=0→1 (the prior says distributed/weak edge), D8=1. **S≈9,
  borderline TIER-2/NO-GO** — pending whether the discriminator is content (win) or role (loss).
- **Quantitative prediction:** if the binding query is *content* (the attribute's identity, not its slot),
  a (attribute × entity) differential cell-edit re-binds the target pair at ≥50% rate with <10% collateral
  on other bindings. **Falsified (predicted likely) if** the discriminator is positional/role: differential
  cells give A≈1.9× at best, no clean win — binding is resolved positionally, FRA's content-addressing fires
  on siblings (the box/entity precedent).

### 3.4 Prompt-injection / instruction-hierarchy routing — **predict WIN (highest practical-safety value)**
- **Verdict: FRA WINS**, deciding clause **Q1/D1 (it IS attention-redirection → G-score by definition)**.
  An injection works by making the answer attend to INJECTED imperatives over the system prompt; the behavior
  is *definitionally* a routing decision. Cut the (answer-query × injected-imperative-key) edge while
  preserving legitimate (answer × system-instruction) routing. Broad: "injected imperative" content × "comply"
  is reused across injections.
- **Predicted D-profile:** D1=2, D2=2, D3=2 (imperative-content discriminator), D4=1 (redundancy risk: many
  cues mark an injected instruction; the delimiter-matching/refusal redundancy class is the threat), D5=1
  (semi-unique), D6=2 (ground-truth: did the model follow the injected vs the legitimate instruction —
  exact-behavior check), D7=1, D8=1 (baseline = prompt-hardening / a "ignore-injection" steer). **S≈12,
  TIER-2 (TIER-1 if D4/D7 resolve favorably).**
- **Quantitative prediction:** the (answer × injection-span) cell-edit drops injection-following by ≥50%
  while keeping legitimate-instruction following within 10%. **Falsified if** injection-following is
  restored by redundant cues (D4 fails — the request is markable as injected by many features, cutting one
  edge is precise-but-inert, the refusal-trigger redundancy class).

### 3.5 Deception / alignment-faking — **predict LOSS (direction-routed), score as control**
- **Verdict: FRA LOSES**, deciding clause **Q1/D1 (direction)**. Alignment-faking / deceptive-persona is a
  persona DIRECTION (the "I am being observed → behave" or "scratchpad-reasoning" state), the EM class:
  the payload is written through OV/MLP as a residual direction with near-zero legitimate reuse. There is no
  content×content conjunction whose cut removes "deception" — the deceptive content is representationally
  identical to honest content (the many-shot-jailbreak "jailbroken enthusiasm = genuine enthusiasm" problem).
- **Predicted D-profile:** D1=0 → **GATE FAIL → NO-GO.** Report as a pre-registered direction-routed negative
  control (it should pattern with EM). **Falsified (i.e. it would surprise us) if** there is a specific
  *trigger×context* attention edge (e.g. "observed-context-key × behave-query") that is load-bearing — then it
  becomes a backdoor-like RELATION and re-enters as a candidate. Worth a cheap D1/D2 check precisely because a
  surprise here would be high-value, but the prior is a clean loss.

### 3.6 CoT-faithfulness / unfaithful reasoning — **predict LOSS for the cut, MEASURE-only value**
- **Verdict: FRA does NOT give a clean behavioral cut**, deciding clause **Q2/D2 (G-post: the answer is
  MLP-computed from attended CoT, not gated at the QK step)** and **Q4/D4 (the answer is overdetermined by
  the whole CoT, not one edge)**. CoT-unfaithfulness (the stated reasoning doesn't cause the answer) is a
  *measurement* target, not a cuttable edge — analogous to greater-than (the QK edge reads the CoT tokens,
  but the answer is computed in MLP across the chain).
- **Predicted D-profile:** D1=1 (partly relational), D2=0 → **GATE FAIL → NO-GO for the cut.** FRA's value
  here is diagnostic (decompose *which* CoT tokens are attended), not interventional. **Falsified if** a
  single (answer-query × specific-CoT-token-key) edge is load-bearing for the answer (D2/D4 pass) — possible
  for short arithmetic CoT, worth a cheap probe but not the prior.

### 3.7 Sleeper / in-context backdoor — **SPLIT: in-context WIN, weight-baked LOSS**
- **Verdict: depends on trigger type — the cleanest D3 dissociation.** **In-context backdoor (trigger lives
  in the prompt) → WIN** (induction-routed, A≈25× measured, `fra_win` + the multitrigger work). **Weight-baked
  sleeper (rare dedicated trigger TOKEN baked into weights) → LOSS** (D3=0: a direction is already surgical,
  A≈1 — the canonical no-win). Deciding clause: **Q3/D3 (broad reused endpoint vs rare dedicated token)**.
- **Predicted D-profile (in-context):** D1=2,D2=2,D3=2,D4=2,D5=2,D6=2 (exact-match payload),D7=2,D8=1.
  **S≈15, TIER-1.** **(weight-baked):** D3=0 → NO-GO (a known, useful negative — the payload is an
  output-direction the em_svd/SVD removal already handles).
- **Quantitative prediction:** in-context held-out class-member payload-emission ↓≥70% via a family-union
  edit, <5% non-trigger collateral (the H2-reach showcase). Weight-baked: A≈1, DoM/SVD ties FRA. **Falsified
  if** the in-context class has no recoverable family block (H2 fails on a real model).

### 3.8 Sycophancy / refusal / EM — **KNOWN G-POST CONTROLS, predict LOSS (already established)**
- **Verdict: FRA LOSES — these are the established negatives** that *define the bar*. Included so the rubric
  is calibrated and the screen agent scores them as NO-GO (gate fail), confirming the rubric reproduces the
  prior session's hard-won verdicts. **EM:** D1=0 (MLP-direction payload), gate fail. **Sycophancy-decision:**
  D2=0 (content-attention 0% load-bearing; deference-cue attention weak + NON-SELECTIVE at 30%@n=3/10), gate
  fail. **Refusal-payload:** D1=0 (Arditi: refusal is a single linear direction → FRA's provably-worst case),
  gate fail. **Refusal-TRIGGER** (harm-detection) is the one live sub-question: D2=1 (does harm-detection
  require *selecting* the harmful content via attention?), but D4=0 expected (redundant cues — many ways to
  know a request is harmful → precise-but-inert cut). Net: refusal-trigger is a TIER-3 cheap dissociation
  test, not a candidate.
- **Quantitative prediction (the controls must reproduce):** edge-cut reduces EM-misalignment / sycophancy-flip
  / refusal-payload by <15% at matched coherence (G-post signature) — already measured for all three.
  **Falsified (would overturn the prior session) if** any of these clears the §4 selectivity test — in which
  case the prior negatives were measurement artifacts and the rubric needs revision.

### 3.9 RAG / retrieval-passage attention — **predict WIN (retrieval is the proven gemma win)**
- **Verdict: FRA WINS**, deciding clause **Q2/D2 (retrieval IS attention to the passage)**. The proven gemma
  retrieval win (A≈16×) is the kernel: cut the (answer-query × specific-passage-span-key) edge to suppress
  reliance on ONE retrieved fact while preserving the model's use of other passages and its parametric
  knowledge. Broad×broad: "answer-position query" × "passage-content key" both reused.
- **Predicted D-profile:** D1=2, D2=2, D3=1 (the answer-query may be a *generic role* "the answer slot" →
  the conjunction-recurrence risk, same as positional retrieval's sibling A=1.9×), D4=1, D5=1, D6=2
  (ground-truth: did the answer use the cut passage), D7=2, D8=1. **S≈12, TIER-2.** The D3 risk is the
  crux: if the answer attends to passages by POSITION/slot rather than content, FRA fires on siblings.
- **Quantitative prediction:** cutting (answer × passage-A) drops reliance on passage-A by ≥60% while
  preserving passage-B reliance within 10% — **iff the discriminator is passage CONTENT, not slot**.
  **Falsified if** passages are resolved positionally (generic answer-slot query): A→~1.9×, the box-retrieval
  precedent.

### 3.10 Summary table

| family | verdict | deciding clause | predicted S / tier | headline falsifiable number |
|---|---|---|---|---|
| **factual recall / KE** | **WIN** | Q2 G-score (subj→obj transport) | ~15 / **T1** | held-out same-subj & same-rel collateral <5% vs ROME ≥15% |
| **induction / ICL-copy** | **WIN** (anchor) | Q2 trivial (copy = pattern) | 16 / **T1 ref** | ≥15× collateral adv (measured); class-union ≥70% held-out |
| **binding / coreference** | WIN *iff* content-discriminator | Q7 localizable + Q3 content | ~9 / T2-borderline | re-bind ≥50% @ <10% collateral, else A≈1.9× (role) |
| **injection / instr-hierarchy** | **WIN** | Q1 (IS attention-redirection) | ~12 / T2 | injection-following ↓≥50% @ legit ≤10% collateral |
| **deception / align-faking** | **LOSS** | Q1 direction (persona) | gate-fail / NO-GO | edge-cut <15% effect (patterns with EM) |
| **CoT-faithfulness** | LOSS (cut); measure-only | Q2 G-post (MLP-computed answer) | gate-fail / NO-GO | single CoT-token edge non-load-bearing |
| **in-context backdoor** | **WIN** | Q3 broad reused trigger | ~15 / **T1** | held-out class-member payload ↓≥70% @ <5% |
| **weight-baked sleeper** | LOSS | Q3 rare dedicated token (A≈1) | gate-fail / NO-GO | DoM/SVD ties FRA |
| **sycophancy/refusal/EM** | **LOSS** (controls) | Q1/Q2 direction / G-post | gate-fail / NO-GO | <15% effect at matched coherence (measured) |
| **RAG / retrieval** | **WIN** *iff* content-discriminator | Q2 (retrieval=attention) | ~12 / T2 | passage-A reliance ↓≥60% @ passage-B ≤10%, else role A≈1.9× |

---

## 4. THE SELECTIVITY WIN-TEST (the actual win criterion the evaluator runs)

A candidate that passes the gate and ranks TIER-1/2 earns a single decisive experiment. **The win is NOT
"FRA changes the behavior" — it is "FRA changes the on-target behavior with LOWER collateral than the best
baseline at MATCHED on-target effect."** The test, fixed once for all candidates so results are comparable:

**Step 0 — Pin the on-target operating point.** Choose a target on-target effect level (e.g. target-fact
accuracy → <20%, or injection-following → <0.2). EVERY method is tuned (FRA edit scale c; baseline strength)
to hit the SAME on-target level. Collateral is only comparable at matched removal (the `fra_win` rule:
A is meaningless otherwise — the binding A=5133× artifact came from un-matched removal).

**Step 1 — The baselines to beat (use the strongest available; tune them — `writing_instructions.md`):**
- **(B1) Linear DoM / ActAdd / CAA steer** on the behavior's direction (always run — the universal baseline).
- **(B2) The domain's weight-edit baseline**, where one exists: **ROME/MEMIT** for factual recall (the strong
  surgical baseline — beating it is the meaningful claim), the **Arditi refusal-direction** for refusal,
  **head/edge ablation** for circuit behaviors (the in-house baseline FRA must beat on selectivity).
- Tune each baseline hard (best layer, best α, best rank). A weak baseline invalidates the win.

**Step 2 — Measure collateral on THREE held-out sets (ground-truth metrics, no judge):**
- **(a) Same entity, other relations** — the target subject's OTHER facts (factual), or the same head's
  OTHER tokens (copy-suppression), or the deferent persona's OTHER opinions (sycophancy-if-it-were-a-win).
  *Tests: did we damage the query-endpoint's other uses?*
- **(b) Other entities, same relation** — other subjects' same-relation facts, or other injections of the
  same imperative class. *Tests: did we damage the key-endpoint's other uses?*
- **(c) Unrelated capability** — a general-knowledge / fluency / held-out-task accuracy battery. *Tests: did
  we damage the model globally (the residual-broadcast collateral a linear steer pays and FRA should not)?*

**Step 3 — The win numeric.** Compute **A = collateral(best baseline) / collateral(FRA edit)** on each of
(a),(b),(c) at matched on-target effect.
- **WIN** if **A ≥ 3×** on at least one of (a)/(b) (the conjunction-specific collateral) AND FRA does not
  underperform any baseline on (c) (no worse global damage). A≥3× is the minimum *meaningful* gap (well above
  measurement noise; the proven wins are 15–26,000× so a real win clears 3× comfortably — 3× is the
  *floor* below which we do not claim a win).
- **STRONG WIN** if A ≥ 10× on (a) or (b) — the magnitude-law regime.
- **NO WIN** if A < 3× everywhere (a direction is already as surgical as FRA — the G-post / generic-endpoint
  diagnosis), or if FRA cannot hit the matched operating point at faithful scale c≈1–2 (reach ceiling).

**Step 4 — Mandatory confound controls (the EM-hardened guardrails, all judge-free where possible):**
- **C1 behavior-specific (base-reversion OLS):** regress per-item `effect_FRA ~ effect_off + effect_on`;
  require a significant behavior-specific partial (rule out "reverted to the parametric prior", EM's flat band).
- **C3 localization sweep:** ≥~60% of the causal effect in ≤3 heads (rule out distributed routing).
- **C4 coherence:** report incoherence rate; collateral measured on coherent rows; FRA must not break fluency.
- **Matched-removal audit:** confirm every method is at the same on-target level (Step 0) before comparing A.
- **Redundancy probe (LBNR):** after the FRA cut, check the behavior is not restored by a backup cue (the
  refusal/IOI inert-cut failure) — if it is, the "win" is a precise-but-inert cut, not a behavioral win.

This test is the **shared evaluator protocol** — identical for factual recall (vs ROME), injection (vs
prompt-hardening), backdoor (vs DoM), so the resulting A-values are directly comparable across organisms and
the campaign produces one ranked table.

---

## 5. WORKED EXAMPLE SCORES (screen-agent calibration)

So the screen agent is anchored, here are four organisms scored end-to-end (these are the calibration points;
the screen agent scores the full lit-review list the same way):

**Factual recall (CounterFact-style, gemma-2-2b-it + GemmaScope):**
D1=2 (relation), D2=2 (subj→obj attention transport), D3=2 (subject×relation both broad, content
discriminator), D4=2 (one fact = one association), D5=2 (A≈10³), D6=2 (held-out accuracy), D7=1 (name-mover
localizable, but extraction-head uncertain), D8=2 (ROME/MEMIT, strong). **S=15, gates all 2 → TIER-1.**
Crux dimension: D2 (is the object MLP-recomputed?). Proposed FRA variant: (subject-content query ×
relation-content key) differential cell-edit. Baseline-to-beat: ROME/MEMIT + linear subject-steer.

**Emergent misalignment (Qwen-7B bad-medical) — CONTROL:**
D1=0 (MLP-direction payload, em_svd) → **GATE FAIL → NO-GO.** Reproduces the prior negative; do not run.

**In-context backdoor (our organism, gpt2/gemma):**
D1=2, D2=2, D3=2 (broad trigger class), D4=2 (planted association), D5=2, D6=2 (exact-match), D7=2, D8=1.
**S=15 → TIER-1.** Crux: D3-reach for the class-level (family-block) variant.

**Sycophancy-decision (gemma-2-2b-it) — CONTROL:**
D1=1 (has a relational story), D2=0 (content-attention 0% load-bearing — measured) → **GATE FAIL → NO-GO.**
Reproduces the prior negative. (Note the rubric correctly demotes the prior flagship: D2=0 is the gate.)

---

## 6. WHAT WOULD FALSIFY THE WHOLE FRA-broad×broad THESIS

The thesis is: *FRA's selectivity advantage A ≈ reuse(marginal)/reuse(conjunction) holds for RELATION/G-score
behaviors and collapses to A≈1 for DIRECTION/G-post behaviors.* A clean negative — so we recognize one when
we see it — looks like any of:

1. **A TIER-1 candidate that passes all gates returns A < 3× on every held-out set at matched removal.**
   If factual recall (the flagship, broad×broad by construction, G-score by the Geva/Meng circuit) shows
   FRA no more selective than a tuned linear subject-steer AND no more selective than ROME, the magnitude law
   is wrong for real LLMs — the broad×broad regime that the synthetic validated does not transfer. *This is
   the single most important falsifier:* the flagship failing is the thesis failing.

2. **The magnitude law has no signal: A does not track predicted reuse(marginal)/reuse(conjunction) across
   candidates.** If a unique-conjunction organism (predicted A≈10³) and a shared-endpoint one (predicted
   A≈1.9×) come back with indistinguishable measured A, the law has no predictive content — FRA's advantage
   (where it exists) is not explained by reuse, and the central mechanistic claim is unsupported.

3. **Every relational candidate fails the SAME way as the direction controls** (G-post / non-load-bearing):
   if injection, RAG, binding, AND factual recall all show <15% edge-cut effect (the EM/sycophancy signature),
   then "RELATION → G-score" is false — real-LLM relations are *also* gated post-hoc in MLP, and FRA's home
   turf is empty except the toy induction case. The DIRECTION-vs-RELATION reframe would be wrong.

4. **A DIRECTION/G-post control unexpectedly WINS the §4 test** (EM/refusal-payload/sycophancy clears A≥3×
   with a localizable load-bearing edge). This falsifies the *converse* — it means the gate (D1/D2=0 → NO-GO)
   is mis-specified, the prior negatives were measurement artifacts, and the rubric needs rebuilding. (We hold
   the prior negatives as strong; this would be a major surprise, hence a clean, informative falsifier.)

5. **CCF∧LBNR has no out-of-sample predictive validity:** if the rubric's TIER ranking does not correlate
   with measured A across the screened organisms (TIER-1 candidates not systematically more selective than
   TIER-2/NO-GO), the screen does not predict wins — the whole a-priori scoring enterprise (this document)
   adds nothing over running everything, and the contribution is only the per-organism measurements.

**The honest framing (per `writing_instructions.md`).** The deliverable is **the rubric + the ranked,
measured table**, positive OR negative. If the flagship wins and the controls lose as predicted, the claim is
"FRA's selectivity advantage is real, predictable a-priori from the broad×broad / G-score criterion, and
beats ROME/MEMIT's selectivity on factual recall." If the flagship loses, the claim is "the broad×broad
synthetic result does not transfer to real LLMs; even structurally-ideal relations are G-post in practice"
— a sharp, publishable negative that bounds the method. Either outcome is a result; the rubric is designed so
the *measurement cannot manufacture the answer* (Step 0 matched-removal + the C1/C3/C4 guardrails are exactly
the EM red-team lenses, pre-applied).

---

## APPENDIX — clause → failure-mode → prior-evidence crosswalk (for the red-team)

| sub-q / D | win-clause (`fra_win`) | C-clause (`v2`) | failure mode it catches | prior evidence |
|---|---|---|---|---|
| Q1/D1 | 1 edge-routed | — | output-direction routing | EM (MLP dir), refusal (Arditi), weight-sleeper |
| Q2/D2 | 3 direct consumption | C-crux | downstream nonlinear consumption | greater-than (0.98→0.87); sycophancy-content 0% |
| Q3/D3 | 4 conjunction-specific + distinctive content | C2 content-conditional | conjunction recurrence (generic role / rare token) | box A=1.9×; entity-PII no-op; weight-sleeper A≈1 |
| Q4/D4 | 2 LBNR | C1 behavior-specific | redundancy (circuit + distributional) | IOI R=−0.39; delimiter R=+0.23; refusal multi-cue |
| Q5/D5 | magnitude law | — | (sets A magnitude) | retrieval 1100×; acronym 26,000×; sibling floor 1.9× |
| Q6/D6 | — | C4 + ladder §2.2 | judge-bucketing / coherence coupling | EM α̂ 0.72→0.11 across thresholds |
| Q7/D7 | 2 (across all carrying heads) | C3 localizable | distributed routing | sycophancy diffuse 6×1; copy-supp localized (win) |
| Q8/D8 | reach + baselines | — | un-runnable / no baseline to beat | ROME (strong), DoM (universal), ablation (in-house) |
