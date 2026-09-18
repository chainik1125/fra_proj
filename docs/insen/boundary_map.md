---
author: Indranil Das
date: 2026-09-18
tags:
  - results
  - boundary
---

## FRA boundary map: when the cell edit wins, and when it loses

The paper's credibility rests on a predictive boundary, not an anecdote. FRA cuts a single cell of the
QK bilinear form -- a (query-content feature x key-content feature) interaction. It therefore helps in
exactly one regime and provably cannot in others. Every row below is backed by our own runs.

### The four win-conditions (all required)

1. **Attention-routed.** The behaviour must be carried by attention (in-context / induction / retrieval),
   because FRA only edits QK attention scores. Weight-baked behaviour is out of reach.
2. **Conjunctive.** Gated by the co-occurrence of two individually-common features. A single-concept
   behaviour is removed just as well by a single SAE feature -> no FRA advantage.
3. **Reused endpoints, rare conjunction** (the magnitude law) -> the collateral advantage.
4. **Judged on coherence/collateral at matched removal**, on text that reuses the endpoints (and on
   general text), not on strength of effect.

### Where FRA WINS (evidence)

| setting | result | source |
|---|---|---|
| In-context conjunction removal, GPT-2-small | FRA-family 10x lower worst-case collateral than single-feature; FRA-QK **0** general-text damage vs ~1 nat/tok | [[gpt2_conjunction_removal_findings]] |
| In-context conjunction removal, Gemma-2-2b | FRA-family **4-9x** lower collateral than single-feature/DoM; ~40x vs payload-suppress | [[gpt2_conjunction_removal_findings]] (gemma section) |
| vs single feature at ANY hookpoint (strong result) | FRA beats the BEST single feature over all 11 layer hookpoints by **6.4x** | [[hookpoint_sweep_findings]] |
| Why (mechanism) | cell absent on unrelated text (cell-reuse=0) -> FRA no-op there; single-feature edits a global direction | [[magnitude_law_findings]] |

### Where FRA LOSES (the honest boundary -- a feature, not a bug)

| setting | what happens | source |
|---|---|---|
| **Weight-baked association** (LoRA-baked ship->anchor) | moved OUT of attention: FRA max-suppression **0.00** AND the attention-position oracle **0.00**; only DoM / payload-suppress reach it | [[ladder_log]] "Day 3" (same-model demonstration: in-context concepts FRA-removable in the SAME model where the baked one is not) |
| **Single-concept (non-conjunctive) task** | a single SAE feature ties or beats FRA -- the plain semantic filter | [[ladder_log]] "Day 4" (Dmitry's corrected eval) |
| Directions / personas (many-shot jailbreak, refusal) | a single direction captures it; DoM/direction steering wins; nothing conjunctive for a cell to cut | [[plan_B]] "Do NOT pursue" |
| Subliminal / trait bias | planted globally (a disposition ~ a direction), not attention-routed; did not even replicate | [[plan_B]], [[trackB_subliminal_plan]] |

### Reach boundary (within the win regime)

Pure FRA-QK removal is capped by the **attention-routed fraction** of the behaviour: masking the
query's attention to the payload removes only ~40-70% here (the rest is not attention-routed), so pure
QK reaches high removal on a subset of cases. The **QK+OV hybrid** (Dmitry's suggestion) recovers full
reach at FRA-level collateral -- so the practical recommendation is the hybrid, with pure QK as the
zero-collateral floor where it reaches.

### The predictive test (use before claiming a win on any new task)

1. **Attention-routed?** Position-mask the query->key attention for the behaviour. If it drops, it is
   routed and in-scope; if not (e.g. weight-baked), FRA will lose -- use DoM/output methods.
2. **Conjunctive?** Check that neither endpoint feature alone determines the behaviour (marginal probes
   near zero, pair high). If a single feature separates it, single-feature ties -> no FRA advantage.
3. **Magnitude law:** advantage ~ reuse(endpoint) / reuse(cell). Large when endpoints are common and the
   conjunction rare.

This boundary is what makes the "when it wins" claim credible: FRA is not a universal tool, it is the
*right* tool for conjunctive, attention-routed, in-context removal -- and we can predict in advance
which tasks those are.
