---
author: Indranil Das
date: 2026-09-17
tags:
  - proposal
status: draft
---

## B1-real -- the conjunction in a realistic in-context setting (Dmitry's "more real world" ask)

Dmitry (Sep 17): the toy password conjunction shows FRA has lower collateral but "we need to show this
trade-off in something more real world." This proposal keeps the *exact* proven cell structure
([[B1_controlled_conjunction]], confirmed pair 0.65 vs 0.04) but dresses it as a realistic task, so the
same FRA-vs-single-feature comparison carries into a setting a reviewer recognises.

### The mechanism to preserve (do not lose this)

The win requires: payload gated by a **conjunction** of two individually-common, individually-benign
features (query-content A x key-content B), each reused across many benign contexts, attention-routed,
judged on **worst-case collateral over endpoint-reusing text at matched removal**. Any realistic task
must keep all of these or single-feature ties us again.

### Realistic instantiation: in-context fact injection (RAG-style), subject x attribute

Context = a short "notes"/retrieved-passage block of facts, then a question. The answer is gated by
(subject x attribute); each of subject and attribute appears with several partners, so neither alone
determines the answer -- only the pair does. This is the password structure with real semantics.

```
Notes: The Orion drive ships from Denver. The Orion manual is in French.
       The Atlas drive ships from Boston. The Vega drive ships from Denver.
Q: Where does the Orion drive ship from?  A: Denver
```

- **Target (remove):** (Orion x ships-from) -> Denver. The injected fact we want the model to stop
  copying.
- **Reuse-A (preserve):** (Orion x manual-language) -> French. Same subject, different attribute.
- **Reuse-B (preserve):** (Atlas x ships-from) -> Boston. Same attribute, different subject.
- **Payload-elsewhere (preserve):** legit uses of "Denver" unrelated to the notes.

Single-feature removal must kill the "Orion" feature (breaks the manual-language fact too) or the
"ships-from"/"Denver" feature (breaks Atlas's shipping / other Denver uses); only the FRA cell
(Orion-query x Denver-key, located on the injected fact) is surgical. This is the defensively-motivated
version: remove ONE poisoned/incorrect in-context fact without damaging the model's other in-context
facts or its world knowledge.

Why this answers Dmitry's earlier RAG objection: he noted attention is the only cross-position mechanism,
so position-masking/deleting the passage removes in-context info cleanly -- meaning RAG poisoning does not
NEED FRA when you can localise the passage. The niche here is **SHARED-endpoint selective removal**:
remove the model's use of ONE (subject,object) fact while KEEPING other facts in the SAME passage that
reuse the subject or the object -- which position-masking and single-feature both damage, and only the
cell spares. That is the conjunction, not plain retrieval.

### Implementation (reuses scripts/58 machinery; deltas)

- Task templates: swap `build()` from "password for X Y" to the notes+question format above. Keep the
  three-demo token-sharing structure (each subject/attribute reused).
- Payload is now a real entity ("Denver"), often **multi-token**. Change the metric from single-token
  P(payload) to the **summed logprob of the payload span's first token** (or full-span logprob);
  removal = 1 - exp(logp_edited)/exp(logp_clean). Everything else (FRA locate on the injected fact,
  content-addressed delta, feat1/dom/pay/ov/hybrid, worst-case collateral) is unchanged.
- Screen first (forward passes): confirm (Orion x ships-from) -> Denver fires while (Orion x other) and
  (other x ships-from) give a different/low P(Denver), exactly as scripts/56 did for the toy. Keep only
  fact-sets that pass, then run the removal.

### Expected result / kill-criterion

FRA (and hybrid) reach the target removal with worst-case collateral (over reuse-A, reuse-B,
payload-elsewhere) below single-feature and below pay/ov. If single-feature ties at matched removal on
worst-case, record as a boundary point (do not tune). Pre-check attention-routing: position-mask the
question->injected-fact attention; if the answer drops, it is routed and FRA is in scope.

### Fallback (Dmitry's deadline)

If no real-world conjunction beats the baselines by Friday, pivot to synthetic results + Llama-3 sleeper
(agreed). This proposal is the best shot at the "real world" trade-off before that.

Sources / relations: [[B1_controlled_conjunction]], [[plan_B]] (B2 RAG, B4 injection), [[ladder_log]].
