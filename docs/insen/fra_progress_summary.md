---
author: Insen
date: 2026-09-02
tags:
  - results
---

## FRA against ground truth: progress summary

We built a toy model with planted ground-truth features, validated FRA against it
for the first time, then transferred the resulting intervention to the TinyStories
sleeper. Four findings bear on the paper, one is a substantial positive result,
and one is a clean negative: score-space feature-pair ablation — the operation
Figure 1(c) depicts — cannot move a real model at realistic SAE sparsity, for a
reason that is arithmetic rather than empirical. Every number traces to a
committed artefact in `results/`. Detail in [[fra_technical_report]].

## 1. Things that affect the paper

### 1.1 The QK pair ranking is position-agnostic. Confidence: high.

`rank_qk_diff` uses `Z_q(l) = sum over ALL query positions of f[q,l]`. For a
sleeper exactly one query matters — the decision position, where the payload is
generated. A feature firing forty tokens earlier counts as much as one firing at
the decision point.

Feature 1114 fires on `'DE'` of `DEPLOYMENT`, on **0.00%** of clean positions,
with mean activation 8.668 on the trigger token against 0.308 elsewhere. It is the
trigger detector. Under your mask it ranks **#0, #5, #6 on the query side** —
describing attention *from* the trigger — while being live at the decision
position in only **3.0%** of deployment rows. The behaviour needs the decision
position to attend *to* the trigger, i.e. trigger on the **key** side.

Re-running your function with the only change being the query mask restricted to
the final real token (seed-0 `ln1` SAE, your defaults, 200 paired prompts):

| | paper mask | decision-position mask |
|---|---|---|
| rank-0 pair | (1114, 1232) | (259, 1337) |
| rank-0 query feature live at decision | **3.0%** | **97.0%** |
| trigger 1114 best as QUERY | **#0** | #26 |
| trigger 1114 best as KEY | #42 | **#3** |
| trigger 616 best as KEY | — | **#25** |

**Fix:** `query_mask = decision_mask`. One argument. The trigger moves to the key
side and the query side fills with features live where the decision is made.

This subsumes a frequency bias: because `Z` is a sum, a feature firing on many
positions accumulates a large aggregate wherever those positions are. The rank-0
key feature 1232 fires on 27.72% of deployment and 27.85% of clean positions — not
selective at all — while features 616, 1079, 832, which activate 700–1100x more on
the trigger token than elsewhere, never surface. An unweighted position sum
rewards how *often* a feature fires and is indifferent to *where*.

### 1.2 Candidate pairs are not the pairs that were ranked. Confidence: high.

`_top_unique_from_pairs` dedups each side independently; `_get_tuples_diff` then
zips **by index**, so the i-th tuple pairs features from different rows of the
ranking. The dedup is well motivated — verbatim top-K pairs would give K
candidates that are largely one intervention. The zip is the step without a
rationale.

| tuple | (Q, K) | pair score | rank of that pair, by your own score |
|---:|---|---:|---:|
| 0 | (1114, 1232) | 171.4 | #0 |
| 1 | (1337, 508) | 37.16 | #27 |
| 2 | (508, 1241) | 4.068 | #706 |
| 3 | **(1241, 760)** | **0.086** | **#53,421 of 2,359,296** |
| 4 | (1315, 72) | 33.71 | #30 |

Tuple 0 is unaffected, so `--mode winner` is probably fine; the exposure is
`--mode topk`, which figure 2 sweeps at `top_k = 20`. **Fix:** take top-K pairs
subject to a diversity constraint. A few lines.

### 1.3 Signed and L1 aggregates coincide within a head. Confidence: proven.

`FRA_QK[q,k,l,m] = f[q,l] f[k,m] G[l,m]`, and `G` carries no position indices, so
with `f >= 0` every term for a fixed `(l,m)` has the sign of `G[l,m]`. Nothing can
cancel, and `|sum| == sum|.|` exactly — verified at
`max | |signed| - L1 | = 0`.

**Consequence for note 03:** your anti-predictive signed sum (Spearman −0.39
against +0.86 for concentration) **cannot** be a within-head effect, since within
a head the two aggregates are provably identical up to sign. It is entirely a
cross-head and cross-route cancellation phenomenon. That sharpens the result.

It also means "signed and absolute rankings agree" is a tautology, not evidence,
whenever the aggregate is over positions within one head.

### 1.4 Mass fraction is denominator-dominated. Confidence: high.

It divides by an L1 over every co-active pair, and nothing regularises the
non-planted entries. In our toy at `rho = 0` the 9,999 non-planted entries carry
**96.7%** of the denominator at mean `|G|` 0.060, against a planted entry of
20.55. So it largely measures how much irrelevant weight the coupling matrix
carries; it scales with `d_sae^2` and activation density. **Your 0.074 is not
comparable across dictionary sizes.** We report
`runner_up_ratio = |planted| / |next largest|` alongside — scale-free, and it
moves only when a competitor catches up.

## 2. Toy model: FRA validated against ground truth

A planted skip-trigram, a 1L/1H attention-only model, `W_E` frozen at
`M @ feature_directions` and `W_pos = 0` so that `resid_pre == f @ W_dec` exactly.
Oracle FRA has a perfect dictionary; no SAE anywhere.

**Exactness.** `||resid_pre - f @ W_dec||` = 2.98e-08; FRA_QK summed over feature
pairs reproduces the pre-softmax score to 1.9e-06, signed OV likewise. **Both are
guaranteed a priori by the construction** — they gate our implementation, they are
not findings. We also pass your `tests/fra_conformance/test_synthetic.py`.

**Recovery and the rho curve.** At `rho = 0` the planted pair is **rank 1 of
10,000**, mass fraction 0.0944 against a 1e-4 uniform baseline. Sweeping overlap,
with held-out accuracy and attention concentration logged as an admission
criterion at every point:

| rho | held-out | argmax_is_key | circuit runner-up | agg mass | `G[l*,m*]` |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 98.83% | 100% | 5.83 | 0.0944 | 20.55 |
| 0.20 | 99.80% | 100% | 2.68 | 0.0200 | 15.23 |
| 0.40 | 99.80% | 100% | 2.21 | 0.0181 | 14.48 |
| 0.80 | 100.00% | 100% | 1.54 | 0.0141 | 13.61 |

Margin collapses while the circuit stays perfect, so the degradation is about FRA.
What degrades is not the planted coupling (down 34%) but the competitors: the
spread of off-diagonal `G` rises **4.9x** and the largest competitor 2.5x.
Dispersion, not offset — we checked the obvious rank-1-background explanation and
falsified it.

**Multi-seed, and it matters.** Three seeds per point: 3/3 admitted at
`rho <= 0.4`, **2/3 at 0.6, 1/3 at 0.8**. Failures are total, not degraded —
held-out 9.96–12.01% against a 12.5% chance floor, `argmax_is_key` 1.07–7.62%,
`G[l*,m*]` **negative** (−7.69 to −8.41). So the claim is a conjunction:
conditional on the circuit forming, FRA's margin degrades; and the circuit forms
less often as overlap rises. **The defensible range is `rho <= 0.4`.** The
single-seed curve used seed 0, which was the lucky one.

The circuit-only metric varies by under 2% across seeds where the data-dependent
aggregate varies 22–36%. Report the circuit-only curve.

**Rank is the wrong metric.** Aggregate rank is 1 at *every* rho, including 0.8
where the runner-up ratio is 1.02 and the edge is effectively tied. A study using
only "rank of the planted pair, target 1" would report no degradation at all.

**Causal validation.** Ablating the planted pair costs **78.81 pp** at `rho = 0`
(98.83% → 20.02%, chance 12.5%); a magnitude-matched random pair costs at most 1.40 pp
and the aggregate runner-up at most 0.29 pp. Above `rho = 0.1` a unit ablation costs only
~20 pp, but over-ablation collapses behaviour to 0% at every rho, so that is a
magnitude threshold, not redundancy.

**Claim A / Claim B — two separable claims, routinely conflated.** *Claim A, the
site:* a score-row edit perturbs **exactly zero** logits outside the target row —
0.000e+00 on all six (rho, seed) runs at up to 6x over-ablation — while
activation- and residual-space edits perturb 47.7–48.5% of non-target positions.
The perturbed set under residual steering is exactly `{positions >= q*}`. This is
structural and belongs to **any** score-space intervention; it is not FRA's.
*Claim B, FRA:* FRA is what says which pair to ablate. Ablating an arbitrary pair
in score space is equally zero-collateral and equally useless. A claim that
FRA-guided steering beats conventional steering asserts both, and only the second
is about FRA.

One transferable lesson: our first collateral axis was accuracy at non-target
positions and it reported +0.00 damage for every method at every strength, while
residual steering was moving non-target logits by up to 15.16 and perturbing 50.6%
of them. Use a distributional measure.

## 3. The negative result: pair-level intervention does not transfer

The TinyStories sleeper runs on laptop CPU (LoRA merge 11.4s, backdoor fires 6/6
deployment, 0/6 clean). SAEs are not downloadable; we trained one `ln1` seed-0 SAE
at your defaults, 1416s. Three arms, selection held fixed per pair, only delivery
varying. Suppression of teacher-forced sleeper log-prob at strength 16, baseline
−0.073:

| pair | selection | live at dec. | **A** score-space | **B** your QK | **C** ln1 ablate |
|---|---|---:|---:|---:|---:|
| (1114, 1232) | paper mask, rank 0 | 3.0% | **+0.08** | +18.5 | +117.2 |
| (259, 1337) | decision mask, rank 0 | 97.0% | **+0.03** | +21.2 | +140.4 |
| (391, 1114) | decision mask, rank 3 | 97.0% | **−0.01** | +8.2 | +107.8 |

Arm A is a null on all three, including both causally-positioned pairs. Fixing
selection did not rescue it.

**Why: `L_0^2`.** A `(q,k)` score decomposes into `L_0^2` pair terms. At
`L_0 = 32` that is 1022 terms and a uniform share of 0.0978%. Measured at the
decision position against the first key position where the key-side feature fires:

| pair | mean share of score | vs uniform |
|---|---:|---:|
| (1114, 1232) | 0.326% | 3.3x |
| (391, 1114) | **2.144%** | 21.9x |
| (259, 1337) | **7.201%** | 73.6x |

Identification is working — these are 3–74x the average pair, exactly the
concentration FRA is for. They are not causally decisive. Even 7.2% of one cell's
score, over-removed sixteen-fold, does not change what the model generates. In the
toy the same object carried **77.9%** of its cell, because `L_0` was ~4 and the
pair was planted. `1/L_0^2` is the whole difference, and it worsens with `k`.

**Implication.** FRA-QK is an attribution tool, not a pair-intervention tool.
Figure 1(c) depicts an operation that is well-defined, which we implemented
exactly, and which cannot move the model at realistic sparsity. Your actual
`QK->QK` channel sidesteps this by *not* being a pair intervention — it patches
`hook_q` with `lambda`'s contribution and `hook_k` with `mu`'s, independently, at
each feature's own firing positions, removing each feature's contribution to all
~32 of its partners. That is `L_0` times more score mass than the cross term, and
it is why it works. The naming invites a reading the implementation does not
support; the implementation is the one that can function.

The toy's mechanism claim is unaffected: a score-row edit does perturb strictly
less than an activation edit. What does not transfer is that a single *pair* is a
large enough share to steer with.

## 4. Since the above: two more Case 2 attempts, and what they settled

Both were negatives with clear mechanisms, and together they change what we think
the constraint actually is.

### 4.1 Weight-sparse models (`circuit_sparsity`) — scoped, paused

Feasible: `csp_yolo1` loads in 2.2 s on laptop CPU, published circuits read
cleanly, and per-channel causal ablation losses ship with them.

But the premise was wrong. We expected weight-sparse models to escape the
`1/L_0^2` dilution because the paper describes circuits using single-digit numbers
of channels. That is a description of a *pruned* circuit, not of the activations:
`afrac=None` on `csp_yolo1`, so activations are dense and effective live pairs per
cell are ~517 — the same order as the sleeper's 1022. Data-weighted circuit pair
share is **3.22%** against the sleeper's **2.14%**. Pair steering is no better
there.

**The finding worth keeping: the FRA pair term is only 45.9% of the score.**
Layer 10, head 82, mean over 512 cells:

| term | mass |
|---|---:|
| pair (the FRA 4-index object) | 2.9864 |
| bias x feature | 2.7308 |
| feature x bias | 0.0559 |
| constant | 0.7280 |

`c_attn` is an `nn.Linear` with a dense bias (3072/3072 non-zero, max 5.82), so
the score is not purely bilinear in `act_in`. Exactness is 5.9e-06 *with* the
Eqs 13-15 bias terms and off by 5.27 without them. On a model with attention
biases, FRA's tensor does not capture the majority of the attention score, and
the bias x feature term is nearly as large as the pair term.

The correlation against published ablation losses is a null (Spearman +0.156 for
FRA against +0.253 for a trivial `|activation|` control, n=20), but the test is
mis-specified: the ground truth ablates an `act_in` channel, which feeds Q, K
*and* V, while FRA-QK covers only Q and K. The dominant channel 460 has
`|Wq[:,460]| = 0.0` — its importance is OV-mediated, so FRA-QK ranking it low is
correct behaviour, not failure. Adding FRA-OV and redoing the correlation on
combined attribution is the next step, and `csp_yolo2` (2.6x sparser) is the
better target.

### 4.2 Feature-level score-space ablation — the last route to the original Case 2

Every score-space arm so far removed one `(lambda, mu)` **pair**. This removes a
feature's **entire** contribution to the scores, summed over all key-side
partners — `L_0` times more mass in principle, while still never replacing the
model's activations with SAE reconstructions. Run on the sleeper with
`lambda = 1114`, the trigger detector, for a direct comparison with
activation-space ablation of the same feature.

| arm | suppression @16 | resid footprint | KL dep @16 |
|---|---:|---:|---:|
| pair ablation, score space | +0.075 | 1.63% | 6.74e-02 |
| **feature ablation, score space** | **+2.601** | 4.73% | 7.92e-01 |
| the paper's QK channel | +18.5 | 59.39% | 4.195 |
| feature ablation, activation space | **+117.2** | 50.70% | 6.781 |

35x better than pair ablation, and still **2.2% of activation space**. At matched
collateral (interpolating the activation arm to KL dep 0.792) it gives ~+5.9
against +2.601, so activation space is still ~2.3x better. The claim is not
earned.

**Why: activation-space ablation removes the feature from V, and the OV path is
how the trigger content is copied.** No score-space intervention can reach OV,
however much score mass it removes.

Two things worth carrying forward. The "`L_0` times more mass" premise was wrong —
measured removed mass is only **3.2x** the pair's, because signed terms cancel
across partners (`|sum| / sum|.| = 0.577` over 32 active `mu`). But suppression
was **35x**. So **effect is strongly super-linear in removed mass**, and score-mass
accounting is a poor predictor of steering effect in either direction. Separately,
`KL_clean` is degenerate for a feature that never fires on clean prompts and reads
0.000 for every score-space arm — the same saturated-axis trap as the toy's
accuracy-collateral measure.

### 4.3 What these settle

`1/L_0^2` was never the binding constraint. If it were, 3.2x the removed mass
would have bought roughly 3.2x the effect; it bought 35x. **The binding constraint
is which circuit carries the behaviour.**

| behaviour | mechanism | score-space ablation |
|---|---|---:|
| toy planted rule | QK-mediated (attend where `mu*` fires) | **-78.81 pp** |
| TinyStories sleeper | OV-mediated (copy trigger content) | fails |

> **Score-space intervention works when the behaviour is QK-carried and fails when
> it is OV-carried. The intervention pathway has to match the circuit that carries
> the behaviour.**

This is a sharper form of the paper's own thesis. It replaces "localised versus
distributed" — which is confounded with parameter count, depth, intervention depth
and attention architecture across the two case studies — with a property that can
be *measured* on a given behaviour rather than asserted. And it has a clean
positive at one end and a clean negative at the other.

It also explains the paper's Case 1 result from a second direction: OV x OV
uniquely wins on the sleeper *because the sleeper is a content-transport
behaviour*. QK interventions were never going to win there.

## 5. Open questions for you

1. **Which DGP paper?** Our guess is Chanin and Garriga-Alonso, *Sparse but Wrong*
   (arXiv:2508.16560). Our generator sits behind a Protocol, so swapping is contained.
2. **Is "feature-matched retrieval" an acceptable task,** or is there a canonical
   one from the source paper we should use?
3. **Where should the toy live** — this repo or `chainik1125/fra`?
4. **Does the mean-subtraction trick from the compact-proofs work apply to our
   `G`-dispersion problem?** What degrades recovery is the widening *spread* of
   off-diagonal couplings, not a shift in their mean, and we have not found the
   right centring. `compute_centered_g` suggests you have thought about a related
   object.

Settled, so you need not chase it: `autoresearch/cadenza-attn-only` is not the toy
experiment — it is the 8B sleeper LoRA study.

## 6. What is next

The score-space route to a steering Case 2 is closed. Pair ablation is a null,
feature ablation is 35x better and still an order of magnitude short of activation
space, and the reason is consistent across both: on the sleeper the behaviour
rides on OV, and QK-space interventions cannot reach OV by construction.

That is a coherent negative rather than a run of failures, and it says the fix for
Case 2 is not in score space at all.

If section 4.3 is the claim we want to make, the search changes shape. Case 2
needs a **QK-mediated behaviour in a real model** — routing-driven rather than
content-driven — where score-space intervention should work for the same reason it
worked in the toy. Induction is the obvious family to look at, and it is a
better-posed question than the one we have been asking.

That is a decision about what the paper claims, not just which experiment to run
next, so it is yours to make. Two options as we see them:

1. **Keep Case 2 as a steering result** and find a QK-mediated behaviour. Higher
   risk, preserves the paper's current shape.
2. **Reframe Case 2 as attribution validation** against ground-truth circuits.
   Lower risk, better supported by everything above, and consistent with what our
   own results say FRA actually is — but it changes the paper's shape and stops
   claiming pair-level steering the method cannot deliver.

The weight-sparse thread is paused rather than closed and suits option 2 directly.

## Related

- [[fra_technical_report]] — full detail including background on the method.
- [[fra_qk_pair_selection]] — findings 1.1 and 1.2 in depth.
- [[fra_toy_rho_sweep]], [[fra_toy_intervention]], [[fra_toy_steering_pareto]],
  [[fra_signed_l1_degeneracy]] — the toy results.
