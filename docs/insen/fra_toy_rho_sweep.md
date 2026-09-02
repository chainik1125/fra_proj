---
author: Insen
date: 2026-09-01
tags:
  - results
---

## FRA recovery degrades with feature overlap while the circuit stays perfect

Stage A of the toy experiment: a planted QK edge in a data-generating process
with ground-truth features, a 1L/1H attention-only transformer trained on it,
and oracle FRA computed with the true feature directions as `W_dec`. No SAE
anywhere. The question is whether FRA recovers the edge we planted, and under
what conditions it stops.

**Headline.** It does recover it, cleanly, at zero overlap. As overlap rises the
recovery margin collapses to parity while the model's circuit remains *perfectly
intact* -- 100% attention concentration on the planted key at every $\rho$
tested. The degradation is therefore about FRA, not about the model failing to
learn the edge.

![FRA recovery vs feature overlap](../../results/figures/rho_sweep.png)

## Result

Single seed, `rho` swept over the `common` geometry (every pair of feature
directions has cosine exactly $\rho$). Retrained at each point, since the data
geometry changes.

| $\rho$ | held-out acc | Gate 2 | admitted | agg rank | agg runner-up | agg mass | cell rank-1 | cell runner-up | circuit runner-up | `G[l*,m*]` |
|---:|---:|---:|:--:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 98.83% | 100% | yes | 1 | 3.86 | 0.0944 | 100% | 9.44 | 5.83 | 20.55 |
| 0.10 | 99.80% | 100% | yes | 1 | 2.09 | 0.0280 | 100% | 3.55 | 3.38 | 16.16 |
| 0.20 | 99.80% | 100% | yes | 1 | 1.11 | 0.0200 | 100% | 2.83 | 2.68 | 15.23 |
| 0.40 | 99.80% | 100% | yes | 1 | 1.04 | 0.0181 | 100% | 2.38 | 2.21 | 14.48 |
| 0.60 | 99.80% | 100% | yes | 1 | 1.24 | 0.0167 | 96% | 1.98 | 1.87 | 13.96 |
| 0.80 | 100.00% | 100% | yes | 1 | 1.02 | 0.0141 | 83% | 1.53 | 1.54 | 13.61 |

Raw data in `results/sweep_rho.json`; regenerate with `scripts/04_rho_sweep.py`
and `scripts/05_plot_sweep.py`.

## The curve is not confounded

The obvious way this experiment fails is that overlapping features make the
*task* harder, the model learns a worse circuit at high $\rho$, and a declining
recovery curve just tracks a declining model. Then "FRA stopped finding the
edge" and "the model stopped forming the edge" are indistinguishable.

That did not happen. At every $\rho$, including 0.8:

- held-out query accuracy is 98.8-100%
- Gate 2 `argmax_is_key` is **100.00%** -- at every $\lambda^*$ query position,
  in every evaluated sequence, the single most-attended key is the planted
  $\mu^*$ position
- the planted coupling `G[lambda*,mu*]` stays large, falling only from
  20.55 to 13.61

All six points pass admission. The model learns the same clean skip-trigram at
every overlap; only FRA's ability to *isolate* it degrades. This is the
favourable outcome -- the weaker fallback claim ("FRA degrades no faster than
the underlying circuit does") is not needed.

## What actually degrades

Not the planted edge. The planted coupling falls 34% (20.55 to 13.61) while the
circuit runner-up ratio falls 73% (5.83 to 1.54). The edge is not weakening
anything like fast enough to explain the collapse -- **the competitors are
rising**.

Measuring the off-diagonal entries of `G` directly:

| $\rho$ | `G[l*,m*]` | off-diag mean | off-diag std | largest competitor |
|---:|---:|---:|---:|---:|
| 0.00 | 20.55 | -0.005 | 0.177 | 3.52 |
| 0.40 | 14.48 | 0.172 | 0.555 | 6.55 |
| 0.80 | 13.61 | 0.058 | 0.873 | 8.84 |

The planted coupling falls 34%; the *spread* of competitor couplings rises
**4.9x** (0.177 to 0.873) and the largest competitor rises 2.5x. Those two
together account for the 3.8x collapse in the runner-up ratio, and the second
dominates.

Note what is *not* happening. Under the `common` geometry each direction is
`v_i = sqrt(1-rho) * o_i + sqrt(rho) * c` with `o_i` orthonormal and `c`
shared, which contributes a term `rho * (c W_Q).(c W_K) / s` identical for
every entry -- a uniform rank-1 background. The obvious prediction is that this
background is what lifts the competitors. **It is not.** That term measures
0.000, 0.152 and 0.026 at the three $\rho$ values, and the off-diagonal *mean*
tracks it -- small, and non-monotonic. The trained model has no reason to align
`W_Q` and `W_K` along the shared direction, so the background never becomes
large.

What grows is the variance, not the mean. The plausible source is the cross
terms `sqrt(rho(1-rho)) * [(o_l W_Q).(c W_K) + (c W_Q).(o_m W_K)]`,
which form an additive row-plus-column pattern with random signs: zero mean by
construction, but spread growing with $\rho$. That decomposition is consistent
with the measurements but has not been verified directly.

So the mechanical account of why FRA blurs under superposition is *dispersion,
not offset*: shared components make feature directions mutually indistinguishable
to the QK circuit, every pair acquires a spurious coupling drawn from a widening
distribution, and the largest spurious coupling climbs toward the planted one. A
method that reports per-pair magnitudes cannot separate a genuine interaction
from the upper tail of that distribution.

## Rank is the wrong metric; margin is the right one

The aggregate rank of $(\lambda^*, \mu^*)$ is **1 at every single $\rho$**,
including 0.8 where the runner-up ratio is 1.02 -- the planted edge is
essentially tied with an arbitrary competitor and would flip under noise.

A study using only the brief's metric 1 ("rank of the planted pair, target 1")
would have reported *no degradation at all* and concluded FRA is robust to
superposition. It is not; the metric is insensitive. The cell-scope rank is a
little more honest, decaying to 96% and 83% at the top of the range, but the
margin metrics are what show the effect.

Recommendation: report `runner_up_ratio` as primary. It is scale-free and
denominator-free, and it moves when a competitor actually catches up.

Mass fraction also declines (0.094 to 0.014) and is reported for comparability
with Dmitry's `qk_pair_concentration.json`, but it should not be read as a pure
FRA signal -- see the caveat below.

## Caveats

- **Single seed per point.** The aggregate runner-up ratio is non-monotonic near
  parity (1.11, 1.04, 1.24, 1.02), which is noise about a quantity pinned near
  1. The circuit-only and cell curves are cleanly monotonic and should be
  trusted more. Multi-seed error bars are the first thing to add.
- **One geometry family.** Only `common` (uniform pairwise cosine) was swept.
  The `subspace` mode, which produces heterogeneous overlap, is implemented and
  tested but not swept. A result that survives both would be about overlap
  rather than about one construction.
- **Mass fraction is denominator-dominated.** It divides by an L1 over every
  co-active pair, and at $\rho=0$ the 9,999 non-planted entries of `G` carry
  96.7% of that denominator. Raising $\rho$ inflates the denominator directly,
  via the same widening spread of spurious couplings described above, so part of
  the mass-fraction decline is the denominator moving rather than the edge
  blurring. This is why the runner-up ratio leads.
- **Stage A only.** No SAE line. The Stage A / Stage B gap -- the cost of SAE
  imperfection measured against ground truth -- remains the obvious next
  experiment, and the prediction is that Stage B declines earlier and faster.
- **No superposition at $\rho=0$.** With `n_feat = 100 < d_model = 128` every
  feature gets its own dimension, so $\rho$ here is *imposed* overlap rather
  than the *forced* overlap of a real model at `n_feat >> d_model`. That
  ratio is arguably the more faithful knob and is the natural second axis.
- **Full-rank head.** `d_head = d_model = 128`, so `W_QK` is unconstrained.
  Real heads have `d_head << d_model`, and `rank(G) <= d_head` structurally
  caps how concentrated a single cell can be. Held fixed here deliberately to
  keep it out of the $\rho$ curve; it is a separate axis worth sweeping.
- Exactness is $\rho$-independent and was verified at 0.0/0.2/0.5/0.8: since
  $f$ is planted and `x = f @ W_dec` by construction, the arithmetic never
  breaks. Only interpretability does.

## Related

- [[fra_signed_l1_degeneracy]] -- why signed and L1 aggregates coincide within a
  head, and what that implies for note 03.
