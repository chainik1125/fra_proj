---
author: Indranil Das
date: 2026-09-18
tags:
  - validation
---

## Validation notes for the conjunction result (Dmitry's Sep 18 checklist)

Answers to the validation items raised in the Sep 18 call, and honest open items for Saturday.

### 1. Hook-point consistency (FRA vs single-SAE-feature)

- **Same SAE, same read hookpoint.** Both methods read features from the SAME SAE at
  `blocks.L.hook_resid_pre` (gpt2-small-res-jb for GPT-2; GemmaScope-res for gemma). FRA's QK feature
  decomposition (`_build_fra_result`) and the single-feature attribution both encode `hook_resid_pre`.
  So the feature basis is identical -- the comparison is "same information, different technique."
- **Intervention target differs by method, as it must.** FRA edits `attn.hook_attn_scores` (the QK
  score -- its defining operation); single-feature edits `hook_resid_pre` (residual). This is inherent:
  FRA is a QK-score edit, single-feature is a residual edit.
- **Sweep (scripts/71):** single-feature at `hook_resid_pre` for every layer 1-11. FRA beats the best of
  these by 6.4x.
- **OPEN ITEM (do for Saturday):** Dmitry wants the sweep to also cover `ln1.hook_normalized` (post-LN1,
  the direct attention input) and `hook_resid_post` (end of block), not only `hook_resid_pre` (pre-LN1).
  Our current SAEs are trained at `resid_pre`; adding resid_post / ln1-normalized needs SAEs at those
  hookpoints. Minimum acceptable to Dmitry: LN1-normalized (the pre-attention one). Note the distinction:
  `resid_pre` is pre-LN1; `ln1.hook_normalized` is post-LN1. We must confirm which our SAE corresponds to
  and, ideally, show the FRA advantage holds at LN1-normalized AND resid_post.

### 2. Token consistency (all-token vs specific-token)

- **FRA** (`patch`) subtracts the located feature-pair's contribution from attention scores **wherever
  that pair appears** (content-addressed) -- effectively all positions, but the edit is zero except where
  the query-content x key-content pair actually fires. So it is gated by the pair, not by a hand-picked
  token list.
- **single-feature** (`feat_add`) subtracts the decoder direction at **all positions unconditionally**
  (the standard additive SAE steer).
- So neither is a hand-picked token subset; both act at all positions. FRA's gating *is* the feature-pair
  (that is the mechanism, not a cheat). **OPEN ITEM:** for a strict like-for-like Dmitry may want the
  single feature gated to the same positions FRA edits -- easy variant to add; expect it only helps the
  baseline slightly.

### 3. SAE feature-selection (the detection stage)

- **Attribution:** contrast the target condition (payload present) vs a matched condition without it
  (same context, the target's answer absent -- e.g. same subject, different attribute). Take the residual
  at the pre-attention hookpoint (`hook_resid_pre`, last position), encode with the SAE, and pick the
  feature by the largest activation difference: `top_feat = argmax(enc(target) - enc(contrast))`.
- **We used TOP-1** (the strictest single-feature baseline), additive, coefficient-swept. Dmitry's fuller
  recipe is top-10 with a coefficient grid -- straightforward extension if we want the strongest baseline
  (`feat1` -> best-of-top-10).
- Intervention: subtract `c * unit(W_dec[top_feat])` scaled by residual norm, at `hook_resid_pre`.

### 4. The flat global-payload / DoM lines (Dmitry's "suspicious")

- **Cause found:** the baseline coefficient grid was too coarse/strong -- even the SMALLEST coefficient
  already removed 96-100% of the payload, so every point piled up at removal ~= 1.0 and the collateral
  interpolated at 30/50/70% just returned the value at the lowest-available (96%) point -> a flat line.
- **Fix:** much finer low-end coefficient grids (e.g. pay/dom/feat1 now start at 0.02) so every method
  samples the 20-95% removal range. Re-running (scripts/65). This may reduce FRA's *reuse-axis* advantage
  at matched removal (the general-text axis -- FRA-QK 0 vs single-feature ~1.0 -- is unaffected). Honest
  numbers to follow.

### 5. Induction-head ablation baseline (retrieval-relevant)

- **TODO:** add "ablate the whole induction head(s)" as a baseline (Dmitry: the natural baseline for a
  retrieval/copy task). Compare its removal/collateral trade-off to FRA. Building this next (scripts/72).

### 6. Safety-relevance

- Current setting is a synthetic conjunctive password / fact-injection. Push toward a safety-relevant
  instantiation (poisoned in-context fact that shares an entity with legit facts; see B1_real). Dmitry is
  OK with it as an *illustrative* section if the KL ratios are large and transparent, with the Llama-8B
  sleeper agent as the main result.

### Strategy (from the call)
Push synthetic until Saturday; then pivot to Llama-8B sleeper agents (required). Synthetic = main section
if strong, else illustrative appendix. Abstract submitted (Dmitry); Indranil has OpenReview author access.
