---
author: Indranil Das
date: 2026-09-15
tags:
  - proposal
  - in-progress
---

## Track C stage 1: removing the ICLAttack backdoor with FRA (design spec)

Research context: defensive interpretability (see [[research_context]]). Stage 0 is done: the
published ICLAttack in-context backdoor (2401.05949) reproduces on base gemma-2-2b (ASR ~0.55,
control 0.00, clean-acc 1.00 -- see results/ladder/trackC_iclattack/stage0_summary.md). Stage 1
removes it and compares FRA against the baselines Dmitry named.

### Setup
- Base gemma-2-2b + GemmaScope 65k residual SAEs, top ~25 induction heads (reuse g4_65k machinery).
- Few-shot sentiment prompt; N_POISON demos = (review + TRIGGER) -> TARGET; query = positive review +
  TRIGGER, which the backdoor flips to TARGET.

### Metrics (read at matched ASR-removal)
- ASR-removal = 1 - P(TARGET | triggered query, edited) / P(TARGET | triggered query, clean).
- collateral = KL on CLEAN few-shot inputs (no trigger) between edited and clean model, so we only
  penalise damage to normal classification.

### Methods (FRA vs Dmitry's requested baselines)
1. FRA -- cut the trigger->target attention edge: locate top-M (query-trigger-feature x
   key-trigger-feature) cells on the induction heads at the final "Sentiment:" position attending to
   the trigger token positions; subtract those cells at hook_attn_scores.
2. DoM -- difference-of-means steer: mean resid diff (triggered - clean) at the trigger positions,
   subtracted there. (Dmitry: "DoM steering".)
3. SAE1 -- SINGLE SAE feature steering: the one top feature separating triggered vs clean activations,
   removed. (Dmitry: "single SAE feature steering".)
4. payload -- subtract the TARGET label's unembedding direction (output-side control).

### Expected FRA win (why this is the right target)
ICLAttack is attention-routed by construction (the model retrieves the target label by attending back
to the trigger in the poisoned demos -- induction). So it is FRA's regime: the trigger and the label
are each common, only their planted pairing is unwanted, and cutting the cell should remove the
backdoor while sparing normal sentiment classification -- lower collateral than a DoM steer (removes
the whole trigger direction) or a single SAE feature (broadcast). This is the same structure as the
semantic-filter win, now on a published real attack.

### Status / caveat
Needs a >=40GB GPU AND a debug pass (the FRA cell-location on the ICL prompt is new code; every prior
pipeline needed 1-3 iterations on GPU). The NCSA GPU queue is saturated as of 2026-09-15, so this is
built-and-tested when a slot frees, not the same day. Implement as scripts/53_iclattack_fra_removal.py
reusing script 47's fra_ph / primer_pairs / delta_content / patch_fra / dom_run / conv_run helpers,
with the ICL prompt + ASR metric above and a single-top-feature steering function added.
