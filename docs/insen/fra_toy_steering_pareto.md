---
author: Insen
date: 2026-09-02
tags:
  - results
---

## Two separable claims: where you intervene, and which pair you pick

The earlier intervention work ([[fra_toy_intervention]]) showed the FRA-identified
pair is causally load-bearing, and that matched-random and runner-up pairs are
not. That is **specificity**. It is not **better** -- nothing was compared
against, and "better than what?" is the comparison the FRA paper's entire Pareto
framing rests on.

This note builds that comparison. It reports two claims that are routinely
conflated, and separating them is the point:

- **Claim A -- about the intervention SITE.** Ablating in score space perturbs
  *nothing* outside the target row, exactly. Activation-space and residual-space
  edits change what position `q*` *is*, and therefore corrupt every later
  position that attends to it. This is structural: **any** score-row
  intervention has it. It is not FRA's.
- **Claim B -- about FRA.** FRA is what tells you *which* pair to ablate. That
  is the recovery result (rank 1 of 10,000) and the specificity result
  (matched-random and runner-up do nothing).

Together: **FRA identifies the pair; score-space intervention delivers it at zero
collateral.** Neither half is sufficient alone. Claim A is arguably the more
valuable finding, because it generalises beyond FRA entirely -- and it is exactly
what the paper's Figure 1(c) depicts and never runs.

## Setup

Three interventions on the same target feature, nested by construction:

| method | edit | breadth |
|---|---|---|
| `fra_pair` | subtract `scale * f[q,l*] f[k,m*] G[l*,m*]` from the QK scores | one feature *pair*, QK only |
| `qkv` | remove `alpha * f[t,l*] W_dec[l*]` from the input to `W_Q`/`W_K`/`W_V` | the paper's "QK->QK"; hits Q, K *and* V |
| `residual` | subtract `alpha * f[t,l*] W_dec[l*]` from the residual stream | conventional SAE ablation; also survives on the skip connection |

Strength swept over 14 values from 0 to 6; `rho` in `{0, 0.4}` (the defensible
range from the multi-seed sweep); 3 seeds each, 6 runs total.

Pareto axes, the toy analogue of the paper's Figure 2: collateral on x (lower is
less damage), target-behaviour accuracy on y (lower is more suppression). So
lower-left is better.

![Steering Pareto frontier](../../results/figures/pareto.png)

## Claim A: verified, exactly, on every run

| rho | seed | `fra_pair` max KL | max frac perturbed | baselines max KL | baselines frac |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 0 | **0.000e+00** | **0.000** | 1.280e-01 | 0.477 |
| 0.00 | 1 | **0.000e+00** | **0.000** | 6.829e-01 | 0.483 |
| 0.00 | 2 | **0.000e+00** | **0.000** | 2.401e-02 | 0.485 |
| 0.40 | 0 | **0.000e+00** | **0.000** | 3.030e-03 | 0.477 |
| 0.40 | 1 | **0.000e+00** | **0.000** | 1.452e-01 | 0.483 |
| 0.40 | 2 | **0.000e+00** | **0.000** | 3.106e+00 | 0.485 |

Zero, not small, at every strength including 6x over-ablation, on all six runs.

At matched suppression (>= 99% of baseline query accuracy removed):

| rho | `fra_pair` | `qkv` | `residual` |
|---:|---|---|---|
| 0.00 | **0.00e+00** (5.7e-04 to 2.6e-03 for baselines across seeds) | 5.7e-04 - 2.6e-03 | 5.7e-04 - 2.6e-03 |
| 0.40 | **0.00e+00** (6.9e-05 to 3.4e-04 for baselines) | 6.9e-05 - 3.4e-04 | 6.9e-05 - 3.4e-04 |

### Why, mechanically

The perturbed set under residual steering is **exactly** `{positions >= q*}` --
asserted with `torch.equal`, not approximately. That is the set of positions that
can causally attend to `q*`.

Residual and activation edits change *what `q*` is*, so its value vector changes
and every later position reading from it sees the corruption. A score-row edit
changes *where `q*` looks*, not what it is, so nothing downstream can observe it.
Algebraically, `f[q,l*] = 0` for every `q != q*`, so the score delta
`f[q,l*] f[k,m*] G[l*,m*]` is identically zero on every other row -- independently
of the weights, which is why it is seed-invariant.

Pinned as regression tests in `tests/toy/test_steering.py`, parametrised over
scale 1.0 / 3.0 / 6.0.

## Claim B: FRA picks the pair

Claim A says score-space intervention is clean. It says nothing about *which*
pair to clean. Ablating an arbitrary pair in score space is equally
zero-collateral and equally useless -- it suppresses nothing.

That half is carried by earlier results: the planted pair is rank 1 of 10,000 in
the aggregate ([[fra_toy_rho_sweep]]), and ablating it destroys the behaviour
while a magnitude-matched random pair and the aggregate runner-up do not
([[fra_toy_intervention]]).

The honest joint statement is: **the site gives you zero collateral, the
identification gives you the effect.** A paper claiming FRA-guided steering is
better than conventional steering is implicitly claiming both, and only the
second is about FRA.

## The paper's own "QK->QK" baseline is not intermediate

`qkv` and `residual` have **identical collateral at every strength and every
seed** -- to the printed precision, in all 12 comparisons. They differ only at
`q*` itself, which is a target position, so the difference never appears on the
collateral axis.

This is a direct finding about the paper's methodology. Its middle option is
presented as a narrower intervention than conventional additive steering, but on
a collateral axis it buys nothing: it edits the same activations, and the only
thing it spares is the skip connection at the position being targeted anyway.
The narrowness that matters is score-space versus activation-space, not
"activations feeding attention" versus "the residual stream".

`residual` is marginally *more* suppressive than `qkv` at equal collateral (e.g.
44.34% vs 48.34% residual query accuracy at strength 0.85, rho=0), because it
also removes the feature from what the unembedding reads at `q*`. So the paper's
intermediate option is weakly dominated by the baseline it is meant to improve on.

## What the strength axis costs

The suppression side is where seeds vary, and where FRA pays:

| rho | strength for >=99% suppression, `fra_pair` | baselines |
|---:|---|---|
| 0.00 | 1.25, 1.25, 1.50 | 1.25, 1.25, 1.50 |
| 0.40 | 3.00, 2.00, 2.00 | 1.25, 1.25, 1.25 |

At `rho = 0` FRA needs the same strength as the baselines. At `rho = 0.4` it needs
1.6x to 2.4x more. This is the magnitude threshold from
[[fra_toy_intervention]]: the planted coupling shrinks with overlap while the
score margin grows, so a unit-scale ablation removes proportionally less.

It does not cost FRA the frontier, because collateral stays exactly zero at any
strength. But it is a real asymmetry and it will matter anywhere the intervention
strength is itself constrained.

## Methodological note: the specified collateral axis was saturated

The first version of this experiment measured collateral as **accuracy at
non-target positions**, which reported `+0.00` damage for every method at every
strength up to 1.5x. Read literally that is a null -- all three methods equally
surgical -- and it is wrong.

The diagnostic: under residual steering the maximum logit change at non-target
positions is **15.16**, affecting **50.6%** of them, while accuracy there is
unmoved. The label away from the query position is a single constant token, and
the model predicts it robustly enough that a 15-logit perturbation does not flip
the argmax.

So the accuracy form of the axis has no dynamic range in this toy, and the note's
result depends on having switched to mean `KL(base || intervened)`, which is also
closer to what the paper's coherence axis is doing. Both forms are recorded in
`results/pareto.json`.

Generalisable lesson: an accuracy-based collateral axis will understate steering
damage wherever the untargeted behaviour is easy. Coherence-style distributional
measures are not a stylistic preference, they are what makes the axis measurable.

## Caveats

- 3 seeds, `rho` in `{0, 0.4}`, single head, oracle features, Stage A only.
- Collateral is measured at *other positions*. A real coherence axis also covers
  damage at the target position itself -- the model still has to say something
  sensible there -- and this cannot see that.
- Claim A's zero is structural, so it will hold for any score-space intervention
  on any model. What will *not* transfer automatically is the toy's clean
  separation between "target position" and "everything else": in a real model,
  the feature being targeted fires at many positions, and the target row is not a
  single row.
- Baseline collateral magnitude varies by two orders of magnitude across seeds
  (2.4e-02 to 6.8e-01 at `rho = 0`). The qualitative claim is robust; the
  quantitative gap is not a stable number.

## Related

- [[fra_toy_intervention]] -- specificity, and the magnitude-vs-redundancy result.
- [[fra_toy_rho_sweep]] -- the recovery curve and the multi-seed admission result.
- [[fra_signed_l1_degeneracy]] -- signed and L1 aggregates coincide within a head.
