---
author: Insen
date: 2026-09-02
tags:
  - results
---

## The QK channel's candidate pairs are not the pairs it ranked

A short note on one step in `scripts/select_features.py`. It needs no new
experiment -- it is visible in the code and quantifiable on the pipeline's own
scoring function -- and the fix is small. We read this as a design choice with a
consequence that was probably unintended, not as a bug.

## What the code does

`rank_qk_diff` scores every ordered feature pair:

```text
score[l, m] = |QK_total[l, m]| * |mean_dep[Z_q(l) Z_k(m)] - mean_cln[Z_q(l) Z_k(m)]|
```

and returns the pairs sorted by that score. `_get_tuples_diff` then converts that
ranking into the candidate tuples the QK channel actually steers with:

```python
q_feats, k_feats = _top_unique_from_pairs(top_pairs_q, top_pairs_k, args.top_k)

if channel == "qk":
    n = min(len(q_feats), len(k_feats), args.top_k)
    return [[(q_feats[i], "Q"), (k_feats[i], "K")] for i in range(n)]
```

`_top_unique_from_pairs` walks the ranked pair list and independently accumulates
the first `top_k` **distinct** query features and the first `top_k` **distinct**
key features. The two lists are then **zipped by index**.

## The dedup is well motivated; the zip is the problem

The dedup step has a clear rationale. The literal top-K pairs are dominated by a
few high-scoring features appearing over and over against many partners, so
taking pairs verbatim would yield K candidates that are largely the same
intervention. Forcing distinct features on each side is a reasonable way to get
K genuinely different candidates.

The zip afterwards is the step with no supporting argument. Having built two
marginal orderings -- best query features, best key features -- pairing them by
position is arbitrary. The i-th best query feature and the i-th best key feature
come from *different rows* of the pair ranking and were, in general, never scored
against each other. Nothing in the scoring function says they belong together.

## Consequence, measured on the pipeline's own metric

Seed-0 `ln1` SAE (d_sae 1536, k 32, paper defaults), 200 paired prompts,
`rank_qk_diff` called exactly as `_ensure_qk_diff` calls it. Running
`_top_unique_from_pairs` and zipping as the QK channel does:

| tuple | (Q, K) | pair score | rank of that pair, by their own score |
|---:|---|---:|---:|
| 0 | (1114, 1232) | 171.4 | #0 |
| 1 | (1337, 508) | 37.16 | #27 |
| 2 | (508, 1241) | 4.068 | #706 |
| 3 | **(1241, 760)** | **0.086** | **#53,421 of 2,359,296** |
| 4 | (1315, 72) | 33.71 | #30 |

Tuple 3 is in the top-5 candidate set while the scoring function that produced
the candidate set ranks it fifty-three thousandth. The zip does not merely
discard pair structure -- it can manufacture pairs the ranking actively rejects.

Tuple 0 is unaffected, because the first unique Q feature and the first unique K
feature happen to be the top pair. The distortion grows with `i`, and the paper's
figure-2 sweeps use `top_k = 20`.

## Suggested fix

Select top-K **pairs** subject to a diversity constraint, instead of top-K
queries and top-K keys zipped. For instance, walk the ranked pair list and accept
a pair only if neither of its features has already been used:

```python
def top_diverse_pairs(pairs_q, pairs_k, top_k):
    out, seen_q, seen_k = [], set(), set()
    for q, k in zip(pairs_q, pairs_k):
        if q in seen_q or k in seen_k:
            continue
        out.append((int(q), int(k)))
        seen_q.add(q); seen_k.add(k)
        if len(out) >= top_k:
            break
    return out
```

This keeps the property the dedup was after -- K candidates that are not the same
intervention repeated -- while guaranteeing every candidate is a pair the ranking
actually scored highly. It is a few lines and changes no other stage.

## Why it matters beyond tidiness

FRA-QK's stated novelty is that it resolves attention into feature *pairs*. If
the candidates handed to the intervention are re-zipped marginals, then the pair
structure is doing no work at selection time either -- and any method that ranked
individual query-side and key-side features would produce the same candidates.
That is worth knowing independently of whether the steering results change.

Whether the results change is an empirical question we have not answered. Tuple 0
is unaffected, so a `--mode winner` run that happens to select it is unaffected
too. The exposure is in `--mode topk`, which sweeps all 20.

## Open question: the ranking may favour frequent features over selective ones

Separately, three features in our SAE are far more trigger-localised than
anything the pair ranking surfaces -- feature 616 activates 1097x more on the
trigger token than elsewhere and fires on 1.44% of deployment positions versus
0.16% of clean ones; 1079 and 832 are similar. None appear in the top pairs.

A plausible reason is in the score itself. `diff_outer` is built from
`Z_q(l) = sum_q f[q, l]`, a **position sum**, so a feature that fires on very few
positions contributes a small `Z` however selective it is, and the product
`Z_q Z_k` compounds it. The ranking would then be biased toward frequently-firing
features over sharply selective ones -- which is the opposite of what a trigger
search wants.

We have not tested this. A normalised variant (mean over firing positions rather
than sum, or a max) would be the obvious check.

## Reproducibility note

`rank_qk_diff` is sensitive to its masks in a way worth flagging. Calling it with
`key_mask=query_mask` -- rather than leaving `key_mask=None` as `_ensure_qk_diff`
does -- changes the ranking completely: the top six pairs become diagonal
self-pairs `(l, l)` of frequent prompt-format features (`' Summary'`,
`' Features'`, `' Random'`) with essentially no deployment selectivity, and the
trigger feature disappears from the top of the list. We hit this ourselves before
matching the call exactly. Anyone re-running the selection should copy
`_ensure_qk_diff`'s signature rather than assume symmetric masking.

## Related

- [[fra_toy_steering_pareto]] -- the two-claim split (intervention site vs feature
  identification) that motivated looking at how the pair reaches the intervention.
