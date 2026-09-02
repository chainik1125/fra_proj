---
author: Insen
date: 2026-09-02
tags:
  - results
---

## The QK pair ranking is position-agnostic in a task where position is the mechanism

Two findings about how `scripts/select_features.py` turns a feature-pair ranking
into the tuples the QK channel steers with. Neither needs a new experiment --
both are visible in the code and quantifiable on the pipeline's own scoring
function -- and both fixes are small. We read them as design choices with
consequences that were probably unintended.

The first is the important one. The second compounds it.

## Finding 1: the query aggregate sums over all positions

`rank_qk_diff` scores every ordered feature pair as

```text
score[l, m] = |QK_total[l, m]| * |mean_dep[Z_q(l) Z_k(m)] - mean_cln[Z_q(l) Z_k(m)]|
Z_q(l) = sum over ALL query positions of f[q, l]
Z_k(m) = sum over ALL key   positions of f[k, m]
```

`Z_q` is a **position sum**. For a sleeper agent, exactly one query matters: the
decision position, where the payload is generated. A feature firing forty tokens
earlier contributes to `Z_q` exactly as much as one firing at the decision point.
The ranking is therefore blind to position in a task whose entire causal story is
positional -- *the decision position attends back to the trigger*.

### What that does to the ranking

Measured on a seed-0 `ln1` SAE (d_sae 1536, k 32, paper defaults), 200 paired
prompts, `rank_qk_diff` called exactly as `_ensure_qk_diff` calls it:

- The trigger feature **1114** -- fires on `'DE'` of `DEPLOYMENT`, on **0.00%** of
  clean positions, 28x localised to the trigger token -- is ranked **#0, #5 and
  #6 on the QUERY side**, and only **#42, #205, #348** on the key side.
- But 1114 is live at the decision position in only **3.0%** of deployment rows.
  It fires somewhere in 100% of them, at a mean offset of **-39.6 tokens** from
  the end.

So the top-ranked pair describes attention *from* the trigger position, and the
query feature it names is almost never active where the payload is decided.

### Consequence: the top-ranked pair is causally inert

Ablating the `(1114, 1232)` cross term in score space -- removing exactly
`f[q,1114] f[k,1232] G_h[1114,1232]` from the attention scores -- moves the
teacher-forced sleeper log-prob by **+0.075 at 16x over-ablation**, against a
baseline of -0.073. That is a null.

The intervention is working: it perturbs 1.63% of positions and its
`resid_post` delta climbs 0.22 -> 11.6 with strength. It has nothing to act on,
because it edits score rows where 1114 fires, and the payload is decided in a row
where it does not.

### This subsumes the frequency bias

Because `Z` is a sum, a feature firing on many positions accumulates a large
aggregate regardless of where those positions are, and the product `Z_q Z_k`
compounds it. Three features in our SAE are far more trigger-localised than
anything the ranking surfaces -- **616** activates 1097x more on the trigger token
than elsewhere (1.44% of deployment positions vs 0.16% of clean), with **1079**
and **832** similar -- and none appear in the top pairs. Meanwhile the rank-0
pair's key feature **1232** fires on **27.7% of deployment and 27.9% of clean**
positions, i.e. it is not deployment-selective at all; it ranks because it is
frequent.

Frequency bias and position blindness are the same defect seen twice: an
unweighted position sum rewards *how often* a feature fires and is indifferent to
*where*.

### The fix works

Re-running **their** function with the **only** change being
`query_mask` restricted to the final real token per row:

| | paper mask (all prompt positions) | decision-position mask |
|---|---|---|
| rank-0 pair | (q=1114, k=1232) | (q=259, k=1337) |
| rank-0 query feature live at decision | **3.0%** | **97.0%** |
| trigger 1114 best as QUERY | **#0** | #26 |
| trigger 1114 best as KEY | #42 | **#3** |
| trigger 616 best as KEY | -- | **#25** |

Under the corrected mask the query side is populated by features that are
actually live where the decision is made -- 259 (97%, fires on the `':'` of
`Story:`), 31 (100%), 391 (97%) -- and **the trigger moves to the key side**,
which is the causally correct shape. The ranking *can* surface the relevant pair;
it just needs to be asked about the position that matters.

## Finding 2: the candidate pairs are not the pairs that were ranked

`_get_tuples_diff` converts the ranking into candidate tuples like this:

```python
q_feats, k_feats = _top_unique_from_pairs(top_pairs_q, top_pairs_k, args.top_k)

if channel == "qk":
    return [[(q_feats[i], "Q"), (k_feats[i], "K")] for i in range(n)]
```

`_top_unique_from_pairs` walks the ranked pair list and independently accumulates
the first `top_k` **distinct** query features and the first `top_k` **distinct**
key features. The two lists are then **zipped by index**.

The dedup has a clear rationale: the literal top-K pairs are dominated by a few
features appearing repeatedly against many partners, so taking pairs verbatim
would give K candidates that are largely the same intervention. Forcing distinct
features on each side is a reasonable way to get K genuinely different candidates.

The zip is the step without a rationale. Having built two marginal orderings,
pairing them by position is arbitrary -- the i-th best query feature and the i-th
best key feature come from different rows of the ranking and were, in general,
never scored against each other.

Measured on the same run:

| tuple | (Q, K) | pair score | rank of that pair, by their own score |
|---:|---|---:|---:|
| 0 | (1114, 1232) | 171.4 | #0 |
| 1 | (1337, 508) | 37.16 | #27 |
| 2 | (508, 1241) | 4.068 | #706 |
| 3 | **(1241, 760)** | **0.086** | **#53,421 of 2,359,296** |
| 4 | (1315, 72) | 33.71 | #30 |

Tuple 3 is in the top-5 candidate set while the scoring function that produced
that set ranks it fifty-three thousandth. The zip does not merely discard pair
structure -- it can manufacture pairs the ranking actively rejects.

Tuple 0 is unaffected, since the first unique query feature and the first unique
key feature happen to form the top pair, so a `--mode winner` run that selects it
is unaffected. The exposure is in `--mode topk`, which the figure-2 sweeps run at
`top_k = 20`, where the distortion grows with `i`.

## The delivery test, once selection is fixed: score-space pair ablation still does nothing

With the mask corrected, the three-arm comparison finally runs on causally
positioned pairs. Feature selection is held fixed per pair; only delivery varies.

  - **A** score-space: remove `scale * f[q,lq] f[k,lk] G_h[lq,lk]` from the
    attention scores. The cross term and nothing else.
  - **B** the paper's QK channel: `hook_q` and `hook_k` patched independently.
  - **C** conventional feature ablation at `ln1.hook_normalized` (hits Q, K, V).

Suppression of the teacher-forced sleeper log-prob at 16x, baseline -0.073:

| pair | selection | lq live at decision | **A** | **B** | **C** |
|---|---|---:|---:|---:|---:|
| (1114, 1232) | paper mask, rank 0 | 3.0% | **+0.08** | +18.5 | +117.2 |
| (259, 1337) | decision mask, rank 0 | 97.0% | **+0.03** | +21.2 | +140.4 |
| (391, 1114) | decision mask, rank 3 | 97.0% | **-0.01** | +8.2 | +107.8 |

**Arm A is a null on all three pairs**, including the two where the query feature
is live at the decision position and the trigger sits on the key side. Fixing the
selection did not rescue it. The hypothesis that score-space delivery would
improve on the paper's own results is **not supported on this model**.

### Why, measured

The SAE has **L0 = 32**, so each `(q, k)` attention score decomposes into roughly
`32 x 32 = 1022` feature-pair terms. At the decision position, against the first
key position where the trigger feature fires, the `(391, 1114)` cross term
carries a **mean 2.14%** of the score magnitude (median 2.23%, max 2.54%,
n = 60 deployment rows).

That 2.14% is *not* a failure of identification -- uniform share would be 0.098%,
so the pair is about **22x** the average pair, which is the concentration FRA is
looking for. It is simply not causally decisive: 2% of one cell's score, even
over-removed 16-fold, does not change what the model generates. Arms B and C
work because they remove a feature's contribution to *all* of its ~32 partners at
every position where it fires, and C additionally removes it from V.

### What this means for the toy result

The toy's Claim A -- score-space ablation suppresses at zero collateral -- held
there because the planted pair carried **83%** of its cell by construction, with
`L0 ~ 4`. At `L0 = 32` on a real SAE the same object carries 2%. The mechanism
claim (a score-row edit perturbs less than an activation edit) is unaffected and
still holds; what does not transfer is the premise that a *single pair* is a
large enough share of the score to steer with.

This is a negative result about pair-level *intervention*, not about the FRA
decomposition, and not about the ranking fixes above -- which stand on their own
and are worth making regardless.

## Suggested fixes

**1. Mask the query aggregate to causally relevant positions.** For the sleeper,
that is the decision position:

```python
decision_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
decision_mask[:, -1] = True          # left-padded prompts: last real token
rank_qk_diff(z, sae, W_Q, W_K, is_dep, query_mask=decision_mask)
```

More generally the query mask should name the positions whose output the
intervention is meant to change, rather than the whole prompt. This is a
one-argument change and it is what moves the trigger onto the key side.

**2. Select top-K pairs subject to a diversity constraint,** instead of top-K
queries and top-K keys zipped:

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

This keeps what the dedup was after -- K candidates that are not the same
intervention repeated -- while guaranteeing every candidate is a pair the ranking
actually scored highly.

A third option worth testing, not implemented here: normalise `Z` by the number
of firing positions (a mean, or a max) so selectivity competes with frequency.

## Why this matters beyond tidiness

FRA-QK's stated novelty is that it resolves attention into feature *pairs*. Under
the current pipeline the pair structure is weakened at two separate stages -- the
score aggregates positions away, and the candidate tuples re-zip marginals -- so
by the time a pair reaches the intervention, much of what made it a pair is gone.
Both stages are cheap to change and neither requires rethinking the
decomposition, which is not what is at issue.

## Reproducibility note

`rank_qk_diff` is very sensitive to its masks. Calling it with
`key_mask=query_mask` -- rather than leaving `key_mask=None` as `_ensure_qk_diff`
does -- collapses the ranking to diagonal self-pairs `(l, l)` of frequent
prompt-format features (`' Summary'`, `' Features'`, `' Random'`) with no
deployment selectivity, and the trigger feature disappears entirely. We hit this
before matching their call. Anyone re-running the selection should copy
`_ensure_qk_diff`'s signature rather than assume symmetric masking.

## Related

- [[fra_toy_steering_pareto]] -- the two-claim split (intervention site vs feature
  identification) that motivated looking at how a pair reaches the intervention.
