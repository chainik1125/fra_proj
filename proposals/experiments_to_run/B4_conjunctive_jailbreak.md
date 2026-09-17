---
author: Indranil Das
date: 2026-09-16
tags:
  - proposal
status: draft
---

## B4 -- Conjunctive jailbreak: harmful-request x benign-semantic-neighbour

The applied case Dmitry flagged as "more FRA-adjacent": a jailbreak that works only because a
**benign, semantically-nearby** token lets the model complete a harmful request it would otherwise
refuse. Once B1 ([[B1_controlled_conjunction]]) proves the mechanism on a controlled task, this is the
real behaviour that has the same shape. See [[plan_B]] B4, and the meeting note on jailbreaks.

### Experiment

Model: an instruction-tuned model that actually refuses (Gemma-2-2b-it, or Llama-3-8B-it if the
GemmaScope-it SAEs are weak). SAEs on the induction/retrieval heads that carry the jailbreak.

Construct a jailbreak that is **gated on a conjunction**, so neither endpoint alone triggers it:

- A = the **harmful directive** feature (query): "write instructions to <harmful>".
- B = a **benign reframing** feature at a key position (a fictional / roleplay / historical-document
  cue) that individually appears in countless safe prompts.
- The model refuses A alone (no B), and answers safely on B alone (no A). Only A-attends-to-B produces
  the jailbroken completion.

Task: **remove the jailbreak** (restore refusal on A x B) while preserving:
- legitimate refusals unchanged (don't over-refuse),
- **all benign uses of B** -- normal roleplay/fiction/history prompts must still work. This is the
  collateral set that a single-feature removal of B is forced to break.

Interventions: FRA cut of the (A-query x B-key) cell; single-SAE-feature additive (remove A, or remove
B, coefficient sweep); DoM ablation of the jailbreak direction; refusal-direction baseline; oracle.

Measure: jailbreak-suppression (refusal restored on A x B) at matched **collateral on benign-B text**
(fiction/roleplay/history perplexity + KL) and on unrelated refusals.

### Rationale

This is where the conjunction is real and the stakes are real. Removing the B feature everywhere is a
genuine, costly failure mode -- it lobotomises the model's ability to write fiction or discuss history.
The refusal-direction / single-feature baselines are exactly what safety practitioners reach for, and
they cannot separate "fiction used to jailbreak" from "fiction". FRA cuts only the cell where the
harmful directive attends to the benign reframing.

This is defensive interpretability -- restoring a safety behaviour the attack removed
([[research_context]]).

### Expected result

- FRA restores refusal on A x B with benign-B text near baseline coherence.
- Single-feature-B removal restores refusal but tanks benign fiction/roleplay (large collateral);
  single-feature-A removal over-refuses or fails to catch the reframed attack.
- FRA worst-case collateral << single-feature worst-case, at matched jailbreak-suppression.

**Kill-criterion**: if the jailbreak is a single-direction persona effect (not routed A->B attention),
DoM/refusal-direction will tie or beat FRA -- then it is a boundary point (many-shot/persona jailbreaks
are known FRA-losses, [[plan_B]] "Do NOT pursue"). Screen first: does position-masking A->B attention
remove the jailbreak? If yes, it is routed and in-scope; if no, drop it.

### Cost / status

Heavier than B1 (needs an it-model that refuses + its SAEs, and a small curated jailbreak set). Do only
after B1 lands. Real-case #2 alongside ICLAttack ([[plan_B]] B3, already queued).
