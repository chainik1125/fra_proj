---
author: Insen
date: 2026-09-01
tags:
  - results
---

## Signed and L1 FRA aggregates are identical within a head

Short note. A structural fact about FRA that we hit while designing recovery
metrics for the toy experiment, and which turns out to say something about
Dmitry's note 03 (`03_ov_concentration_finding.md`, on branch
`upstream/dmitry/ov` under `experiments/tinystories_sleeper/tracing_feature/qk_vs_ov/working_notes/`)
rather than about our toy.

## Claim

For a **single attention head**, aggregating FRA over positions, the signed
aggregate and the L1 aggregate are equal up to a per-pair sign:

```text
| sum_{q,k} FRA_QK[q,k,l,m] |  ==  sum_{q,k} | FRA_QK[q,k,l,m] |
```

for every feature pair $(\lambda, \mu)$. The two therefore induce **identical
rankings** over feature pairs. This holds on any model, for any head, whenever
feature activations are non-negative.

## Why

The QK decomposition factorises as

```text
FRA_QK[q,k,l,m] = f[q,l] * f[k,m] * G[l,m],    G = (W_dec W_Q)(W_dec W_K)^T / sqrt(d_head)
```

`G` carries no position indices: it is a function of the trained weights alone.
So for a *fixed* pair $(\lambda, \mu)$, the only position-dependent factor is
`f[q,lambda] * f[k,mu]`, and if activations are non-negative that factor is
non-negative for every $(q, k)$. Hence

```text
sign( FRA_QK[q,k,l,m] ) = sign( G[l,m] )    for all (q, k)
```

Every term in the sum over positions carries the same sign, so nothing can
cancel, and summing gives $\pm$ the sum of absolute values. The claim follows.

The non-negativity condition is satisfied by construction for ReLU and JumpReLU
SAEs. For TopK SAEs it holds whenever the retained top-$k$ pre-activations are
positive, which is the ordinary case rather than a guarantee.

The OV path has the same structure. With $A \ge 0$ (post-softmax) and
$f \ge 0$, the signed contribution $A[q,k] \, f[k,\lambda] \, (W_{dec}[\lambda]
W_V W_O)$ is a non-negative multiple of a fixed vector per feature, so its
projection onto any fixed readout direction also has constant sign across
positions.

## What breaks it

Cancellation requires two contributions to the *same* feature pair with opposite
signs. Within one head and over positions that is impossible, by the above. It
becomes possible as soon as you aggregate over something that carries its own
$G$:

- **across heads** -- each head has its own `G_h`, and `sign(G_h[lambda,mu])`
  may differ between heads
- **across routes / pre-features** in a two-stage decomposition, for the same reason
- **across layers**

## Consequence for note 03

Note 03 reports that the signed sum
is *anti*-predictive of measured ablation impact (Spearman $-0.39$) while
concentration over $|\cdot|$ is strongly predictive ($+0.86$), and reads this as
evidence that concentration is the right aggregation.

The result above sharpens that. The anti-predictiveness **cannot** be a
within-head effect, because within a head the signed and absolute aggregates are
provably the same up to sign and would produce the same correlation. It is
therefore *entirely* a cross-head (and cross-route) cancellation phenomenon.

That is a stronger statement than the note makes, and it is a mechanism rather
than an observed correlation: features whose attribution is spread across heads
that disagree in sign are exactly the ones whose signed sum understates them.
The note's own explanation -- "L1 helps because it penalises cancellation" -- is
correct, and this pins down *where the cancellation lives*.

## Consequence for metric design

Reporting "signed and absolute rankings agree" is not evidence about anything
when the aggregate is over positions within one head. It is a tautology. Our
toy hit exactly this: at the aggregate scope both rankings put the planted pair
at rank 1, and we briefly read that as agreement before checking. Verified
numerically at `max | |signed| - L1 | = 0`, not merely
small.

Two places the comparison *is* meaningful:

- **within a single $(q,k)$ cell**, where the competing entries are different
  pairs $(\lambda,\mu)$ with different signs of `G`
- **across heads**, which is Dmitry's setting

Pinned as a regression test in `tests/toy/test_recovery.py::test_signed_and_l1_aggregation_are_degenerate`
so the agreement is never re-reported as a finding.

## Scope and caveats

- The claim is about aggregation **over positions**, for a **fixed feature pair**,
  within **one head**. It says nothing about which aggregation predicts causal
  impact once you leave that setting -- that is note 03's question and its answer
  stands.
- We have verified it on the toy only. The proof does not depend on anything
  toy-specific, but it does depend on $f \ge 0$, which is worth confirming for
  the specific SAE in any setting where it is applied.
- It concerns the QK path directly; the OV argument above is sketched rather than
  tested.
