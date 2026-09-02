---
author: Insen
date: 2026-09-01
tags:
  - results
---

## Ablating the recovered FRA edge destroys the behaviour, at every overlap

The causal half of Dmitry's step 4, and brief section 6 metrics 5-6. Attribution
is only meaningful if intervention confirms it -- FRA is an *intervention* paper,
and attribution without a causal test is what sank its Case 2.

This also runs the experiment FRA's own Figure 1(c) depicts and the paper never
performs. That figure shows cutting a single feature-pair interaction while
preserving the others; the paper's "QK->QK" intervention instead edits the shared
input to `W_Q`, `W_K` and `W_V` for all heads, which is not a per-pair QK
intervention.

**Headline.** The planted pair is the causal carrier at *every* rho. Where a
scale-1 ablation looks weak at high overlap, that is a magnitude threshold, not
loss of causal role: over-ablating collapses the behaviour completely at every
rho tested. Meanwhile competitors -- including one the margin metric ranks as
*tied with* the planted pair -- barely matter.

## How the intervention works

Gate 3 established that the pre-softmax score is exactly the sum of the FRA
tensor over feature pairs, so zeroing one entry and re-summing is exactly
subtracting one term:

```text
s'[q,k] = s[q,k] - scale * f[q,lambda] * f[k,mu] * G[lambda,mu]
```

Applied as a hook on `hook_attn_scores`, which lets softmax, the OV path with the
*new* pattern, and the unembed all run downstream. That makes it a genuine causal
intervention on one feature pair, not a re-scoring of a frozen pattern. Verified
to remove exactly the analytic pair contribution, to 0.0.

## Four arms

| rho | planted | random (matched) | agg runner-up | cell runner-up | agg r-u ratio |
|---:|---:|---:|---:|---:|---:|
| 0.00 | **-78.81** | +0.00 | +0.00 | +0.00 | 4.82 |
| 0.10 | -21.58 | +0.00 | +0.00 | +0.00 | 2.77 |
| 0.20 | -19.24 | +0.00 | +0.00 | +0.00 | 1.61 |
| 0.40 | -19.24 | -1.40 | +0.00 | -0.39 | 1.50 |
| 0.60 | -21.58 | -0.10 | +0.00 | -1.86 | 1.45 |
| 0.80 | -22.46 | +0.00 | -0.29 | **-6.54** | 0.98 |

Change in held-out query accuracy, percentage points, chance 12.5%, n=1024. All
six points pass the Gate 2 admission criterion.

The **random** arm is rescaled to remove the same total score mass as the planted
arm, asserted equal to <0.1%. Without that rescaling it would show no effect for
the trivial reason that it barely perturbed anything; matching the mass makes it
a test of *where* the mass was removed.

The **aggregate runner-up** arm is structurally uninformative, and is kept in
order to show that. It is selected over all `(q,k)`, so its query feature is
usually inactive at `q*` -- its contribution at the planted cell is then
identically zero. Measured live fraction at `(q*, k*)`: 5.7%, 12.9%, 0.3%, 0.3%,
2.1%, 2.1%. At rho=0.2 and 0.4 it selected `(10,10)`, a distractor-distractor
pair live in 0.3% of sequences.

The **cell runner-up** arm fixes that: the largest live competitor at each
sequence's own `(q*, k*)`, chosen per sequence, so both features are active at
the relevant positions by construction. It uses 47 to 89 distinct pairs as rho
rises, and the winner is almost always `(lambda*, nu_c)` -- the planted query
feature paired with the *content* feature rather than `mu*`. Competitors compete
on the key side, not the query side.

## The margin metric understates causal identification

At rho=0.8 the aggregate runner-up ratio is 0.98: by magnitude the planted pair
is *tied with or marginally beaten by* a competitor. Causally it is not close:

- planted **-22.46 pp**
- best live competitor **-6.54 pp** (3.4x less)
- aggregate runner-up **-0.29 pp** (77x less)

So FRA still identifies the causally correct pair where the margin metric reports
a tie. The metric is fragile; the identification is not. This qualifies the
headline of [[fra_toy_rho_sweep]], which measured margin collapse and could not
distinguish the two.

## The scale sweep: magnitude, not redundancy

At scale 1 the ablation costs 79 pp at rho=0 but only ~20 pp at rho >= 0.1, and
that plateau is flat from 0.1 to 0.8. Two hypotheses survived the four-arm run:

- **magnitude** -- `removed_L1` fell 34% (tracking `G[lambda*,mu*]`) while the
  score margin grew, so scale 1 is no longer enough removal
- **redundancy** -- the rule is carried by correlated coordinates of a
  non-orthogonal decomposition and removing one cannot help

Scaling the ablation past 1 separates them. Post-ablation accuracy:

| rho | scale 1 | 1.5 | 2 | 3 | 4 |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 19.43% | 0.10% | 0.10% | 0.10% | 0.10% |
| 0.10 | 77.34% | 5.66% | 0.20% | 0.00% | 0.00% |
| 0.20 | 80.08% | 11.72% | 0.49% | 0.00% | 0.00% |
| 0.40 | 78.22% | 16.80% | 2.15% | 0.00% | 0.00% |
| 0.60 | 76.86% | 19.34% | 3.32% | 0.00% | 0.00% |
| 0.80 | 77.44% | 27.54% | 6.64% | 0.00% | 0.00% |

**Magnitude wins, unambiguously.** Behaviour collapses to 0% -- below the 12.5%
chance floor, i.e. actively wrong -- at every rho once the ablation is strong
enough. The redundancy story is dead: if correlated coordinates were carrying the
rule, no amount of removing one of them would help, and removing more would not
help monotonically.

The residual rho-dependence is in how *much* over-ablation is needed: scale 1.5
suffices at rho=0, scale 2-3 at rho=0.8. That is a real but different kind of
degradation -- about required intervention strength, not about which pair to
intervene on.

This also resolves the "softmax got sharper" alternative in the same run, since
that hypothesis *is* the magnitude hypothesis. Corroborating, from the multi-seed
sweep: `|W_Q|` grows from ~22 at rho=0 to ~26 at rho=0.6 while `G[lambda*,mu*]`
falls, consistent with the model compensating by scaling weights up.

## Structure of the residual failures

Post-ablation accuracies repeat exactly across rho (801/1024 at two points,
825/1024 at two others), which suggested a specific failing subset. The
evaluation batch *is* identical across rho -- `make_directions` draws the same
shapes whatever rho is, so only the directions change -- making this a paired
comparison. But the failing sets are similar, not identical:

- Jaccard of failing sets between adjacent rho: 0.68 to 0.84
- between rho=0 and everything else: 0.23 to 0.28

So rho=0 fails on a qualitatively different set, and above it the set drifts
smoothly. The repeated counts were coincidences of size.

Two weak but consistent conditioning variables:

- **query/key gap.** Short gaps survive ablation better: 83-87% at gap 1 versus
  72-76% at gap 13-32, at every rho. `corr(correct, gap)` is -0.05 to -0.07 above
  rho=0, and -0.142 at rho=0. More competing positions means a competitor more
  easily wins the argmax once the planted contribution is removed.
- **content feature.** Spread across `nu_c` grows from 10.7 pp at rho=0.1 to
  25.9 pp at rho=0.4-0.8; `nu_0` survives best (~91%), `nu_4` worst (~65%).

## Caveats

- Single seed for the intervention experiments. The multi-seed sweep covers the
  recovery metrics, not the ablations.
- `train_cached` on a cache hit does not advance the DGP generator, so scripts
  08/09 evaluate on a different batch than 06/07 -- 19.43% versus 20.02% for the
  same intervention. Each script is internally consistent and no conclusion
  depends on it, but exact percentages are not comparable across the two groups.
- The 3-arm and 4-arm runs agree to 1e-12 on every shared arm and select the same
  random and runner-up pairs, so training is deterministic given the seed.
- Stage A only; no SAE anywhere.

## Related

- [[fra_toy_rho_sweep]] -- the recovery curve these interventions qualify.
- [[fra_signed_l1_degeneracy]] -- why signed and L1 aggregates coincide in a head.
