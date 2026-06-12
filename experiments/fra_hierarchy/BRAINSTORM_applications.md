# FRA broad×broad — alternative applications brainstorm (post EM+sycophancy negatives)

## What the two negatives actually taught us: DIRECTION vs RELATION
The campaign tested two ALIGNMENT behaviors (EM misalignment, sycophancy) and both came back G-post.
That's not bad luck — it's a CLUE about the wrong class. The sharpened criterion:

**FRA's surgical cell-cut beats a linear steer iff the behavior is RELATIONAL/ASSOCIATIVE and the relation
is carried by the attention SCORE (G-score), AND both endpoints are broad (reused) features.**

A behavior is FRA-cuttable iff ALL of:
1. **G-score** — the decision is "does query-content attend to key-content," made AT the QK step, not a
   post-hoc OV/MLP direction. (EM: MLP-direction payload. Sycophancy: deference baked into the residual by
   the answer position. Both G-post → FRA loses.)
2. **Broad×broad reuse** — both the query feature and key feature fire across MANY contexts, so a linear
   steer removing either has high collateral. Selectivity advantage ∝ reuse (the magnitude law).
3. **Specific-pairing target** — you want to remove ONE association (this query-content × this key-content)
   and preserve both features' other uses. (A linear steer can't: it removes a whole direction.)
4. **The behavior IS the attention** — relation / binding / retrieval / induction: inherently attention-shaped.

### The diagnostic: which behaviors are DIRECTION (FRA loses) vs RELATION (FRA wins)?
- DIRECTION / payload / persona (G-post — a linear steer is already near-optimal, FRA has no edge):
  EM misalignment, sycophancy-decision, style/persona, **refusal** (Arditi: refusal is a SINGLE linear
  direction → FRA is provably the WRONG tool here; DROP refusal from the candidate list).
- RELATION / binding / retrieval (G-score, attention-native — FRA's home turf):
  factual recall (subject→object), in-context induction/copying, entity-attribute binding, coreference,
  instruction/injection routing, RAG passage-attention. THE ONE PROVEN WIN (fra_win) is induction-retrieval.

CONCLUSION: pivot from alignment-behaviors (mostly direction-routed) to RELATIONAL behaviors. This also
matches the PI's original intuition: "the most interesting consequences happen when features are associated
with many different words at higher levels of the hierarchy" = a broad subject feature × a broad relation feature.

## Ranked alternative applications (fit the G-score + broad×broad + surgical criterion)

### 1. FLAGSHIP — Surgical factual-association editing (subject-feature × relation-feature cell)
Cut "Tom Cruise ⊗ mother-of → Mary Lee Pfeiffer" by editing the (subject-content query × relation-content key)
cell, WITHOUT damaging (a) Tom Cruise's OTHER facts or (b) other people's mothers. This is broad×broad BY
CONSTRUCTION: subject feature (reused across all his facts) × relation feature (reused across all subjects).
- WHY it should be G-score: factual recall is attention-mediated (subject enrichment at the subject token,
  then an "extract the relation" attention move to the last token — the Geva/Meng circuit). The object is
  pulled via attention → cutting the subject×relation conjunction should block THAT fact's retrieval.
- SELECTIVITY TEST (the win): vs ROME/MEMIT (rank-1 MLP edit) and vs a linear "Tom Cruise" steer — FRA should
  leave more of the subject's other facts + other subjects' same-relation facts intact (lower collateral).
  Ground-truth metric (no judge): does the edited fact change AND do held-out facts survive (accuracy).
- Strong existing baselines to beat (ROME/MEMIT/linear). Clear broad×broad story. gemma-2-2b-it + known facts.
- RISK: factual recall may have a strong MLP-extraction step (also partly G-post) — but the SUBJECT→OBJECT
  transport is attention; the test is whether the conjunction-cut is more surgical than the alternatives.

### 2. Attention-hijack / prompt-injection defense (instruction-source × directive cell)
Jailbreaks/injections work by making the answer attend to INJECTED instructions over the system prompt. The
behavior IS attention redirection → definitionally G-score. Cut the injection-content × answer edge while
keeping legitimate instruction-following. Practical safety value; clean selectivity test (block injection,
preserve benign instructions). Broad: "injected-imperative" content × "comply" — reused across many injections.

### 3. Induction-backdoor disarm (generalize the PROVEN fra_win)
The one place FRA already won (fra_win: suppress a cue's induction ~15× less collateral than ActAdd). Generalize:
in-context backdoors / repeated-pattern exploits where the trigger fires via INDUCTION (attend-to-prev-occurrence).
Cut the trigger-token induction edge → disarm the backdoor without touching the model's other induction. Lowest
risk (we have the mechanism), but narrower novelty (extends existing result).

### 4. Entity-attribute binding fix (entity × attribute cell)
Multi-entity contexts where the model binds the wrong attribute to the wrong entity (known failure). The binding
is attention-mediated. FRA edits WHO attends to WHICH attribute. Cleaner G-score story than alignment; harder to
build a ground-truth eval.

## RECOMMENDATION
Lead with #1 (factual-association editing) — it is the textbook broad×broad application, plausibly G-score,
has ground-truth metrics + strong baselines (ROME/MEMIT/linear) to demonstrate the SELECTIVITY win, and directly
realizes the PI's "broad subject × broad relation, high in the hierarchy" intuition. #2 (injection defense) is the
highest practical-safety upside and most cleanly G-score. #3 is the safe extension of the proven win.
DROP refusal (it's a linear direction → FRA's worst case).
