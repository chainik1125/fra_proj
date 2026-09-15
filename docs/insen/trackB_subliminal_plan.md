---
author: Indranil Das
date: 2026-09-15
tags:
  - proposal
  - in-progress
---

## Track B: subliminal learning as a trained-in behaviour for FRA

Research context: defensive interpretability research (see [[research_context]]). Dmitry proposed
subliminal learning (Cloud et al., arXiv 2507.14805) as a candidate for the "trained-in, natural"
behaviour the semantic-filter result ([[semantic_filter_findings]]) is missing.

Subliminal learning: a student fine-tuned on a teacher's outputs acquires the teacher's trait
(e.g. owl-preference, or misalignment) even when the training data is semantically unrelated (number
sequences), and only when teacher and student share the same base model. The trait rides
model-specific statistical patterns, not content.

### Why it fits, and the risk

Fits: the trait is baked into the student's WEIGHTS by fine-tuning and is safety-relevant (a
misaligned teacher silently infecting a student) -- exactly the "not planted" property we lack.

Risk: the trait is likely a *direction/disposition* ("likes owls", "is misaligned"), not a
*content-query x content-key attention edge*. Directions are FRA's known losing regime (sleeper, EM,
sycophancy all came back "direction, DoM wins"). So the first question is NOT "run FRA on it" but
"does this behaviour even have the shape FRA needs".

### The go/no-go (cheapest first)

**Stage 0 -- replicate the transfer (needed before anything).**
Teacher = student base = gemma-2-2b-it (shared init, required by the paper). Give the teacher a trait
by system prompt ("You love owls"). Generate ~3-10k number-only sequences, filtered to digits/commas
so no semantic leakage. Fine-tune a student (gemma-2-2b-it, LoRA) on those sequences. Eval: does the
student's owl-preference rise above a control student trained on a neutral teacher's numbers?
GO iff the trait transfers (else the phenomenon isn't reproduced on this model and we stop).

**Stage 1 -- direction vs relation (the decision).**
On the trait-bearing student:
- Build a difference-of-means direction for the trait (owl vs non-owl prompts). Does DoM ablation
  remove the trait cleanly at low collateral? If YES -> it's a direction, FRA has no structural
  edge, report that and stop (this is a DoM story).
- Look for an attention edge: is there a query-content x key-content cell whose FRA score tracks the
  trait? Use the same machinery as the semantic-filter pipeline. If a cell carries it and DoM
  struggles -> it's FRA's turf, proceed to a full comparison.

Only if Stage 1 says "relation" do we run the full FRA-vs-baselines comparison.

### Cost / feasibility

Heavier than Track A: teacher generation + student fine-tune + eval, then the diagnosis. All on the
free NCSA allocation, >=40GB GPU. Stage 0 is the gate; do not build the full comparison until the
trait replicates and Stage 1 says relation.

### Status

Plan only. Track A (LoRA-baked concept association, `scripts/49_lora_bake_concept.py`) runs first and
in parallel; it is the cleaner test of the same "planted -> trained-in" jump and reuses the whole
semantic-filter pipeline. Track B Stage 0 script to be written once Track A is launched.
