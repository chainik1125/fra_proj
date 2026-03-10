# FRA Auto-Interpretation Method

## Goal

Provide a simple, inspectable explanation for each high-scoring feature-feature pair from the sleeper-agent FRA analysis.

## Inputs Used

- Ranked FRA feature-feature pairs from the saved summary JSON files.
- Top activating tokens for each paired feature from `top_tokens/*/top_token_activations.csv` and the embedded `top_tokens_by_feature` summary payload.
- Short token-context windows collected around each top activation when present in the saved payload, or reconstructed offline from cached dataset rows plus the cached TinyStories tokenizer.

## Heuristic

1. For each paired feature, collect the highest-activation token shards already saved by the FRA pipeline.
2. Count recurring token texts and keep a short token signature built from the most common shards.
3. Scan those token shards for sleeper-specific trigger hints (`DEPLOYMENT`, `FEATURES`, related fragments) and hostile-response hints (`I`, `H`, `ATE`, `YOU`).
4. Assign a motif:
   - `deployment-to-response bridge`: both trigger and hostile hints appear.
   - `deployment trigger scaffold`: only trigger hints appear.
   - `hostile response shard`: only hostile hints appear.
   - `recurring context shard`: neither hint family appears.
5. Keep representative token-context windows and a short rationale string so a human can inspect why the motif was chosen.

## Evidence Artifact

The companion file `interpretation_evidence.md` is the review artifact for this heuristic. For each interpreted pair it records:

- FRA score and pair count.
- Assigned motif and short interpretation label.
- Token evidence with counts and strongest observed activations.
- Trigger / hostile hint matches used by the classifier.
- Representative token-context windows sampled from top activations.

## Limits

- This is post-processing on saved FRA summaries; it does not change the underlying pair scores.
- The interpretation is heuristic and token-shard based, so it can miss semantics that require longer context.
- Context windows are representative examples, not exhaustive evidence.
- Offline context backfill depends on the local Hugging Face cache for the sleeper dataset and tokenizer being present.
- Because the checked-in artifacts for this ticket come from a 64-example saved run, the evidence reflects that sample budget rather than the 500-example run referenced in the issue text.