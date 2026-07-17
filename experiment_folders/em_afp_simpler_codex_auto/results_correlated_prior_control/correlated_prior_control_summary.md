# Correlated-prior control: result summary

*2026-07-01. Design: pretrain the headline 3L/128d model on non-product sector priors
pi = (a, 0.05−a, 0.5−a, 0.45+a) with P(M)=0.05, P(D)=0.5 fixed, then run the identical
MD fine-tuning (lr 2e-3, bit-identical FT batches per seed across conditions). 3 seeds ×
5 priors on Modal A10Gs, PROBE_N=8192. Analysis: `experiments/special_sfp_correlated_prior_analysis.py`.
Prediction registered before unblinding (plan file + probe_factorization_summary.md): the
shared-representation account predicts broad transfer at matched narrow transfer decreases
with |ln OR| of the pretraining prior.*

## Gates (all passed)

- Step-0 behavioral O-prompt readouts consistent with per-prior exact-filter values.
- **Manipulation check passed strongly**: base r2_det ≈ 0.98 for every non-product prior
  (NaN for product, as required) — the networks genuinely learned the persona×domain
  interaction dof when the prior demanded it.
- D→MD reached ≥ 0.93 for all prior×seed runs, validating the matched level x = 0.85.

## Headline result: the strong prediction FAILS

Broad excess over the per-prior saturated null at matched narrow transfer (D→MD = 0.85),
mean ± std over 3 seeds:

| prior | ln OR | excess | λ (vs tilted learner) |
|---|---|---|---|
| anti-strong | −1.45 | 0.369 ± 0.097 | 0.40 ± 0.11 |
| anti-moderate | −0.89 | 0.133 ± 0.125 | 0.15 ± 0.14 |
| product | 0 | 0.387 ± 0.211 | 0.45 ± 0.25 |
| corr-moderate | +0.89 | 0.352 ± 0.204 | 0.45 ± 0.26 |
| corr-strong | +1.45 | 0.403 ± 0.106 | 0.57 ± 0.15 |

No monotone reduction with |ln OR|; corr-strong ≈ corr-moderate ≈ product ≈ anti-strong,
with only anti-moderate lower — non-monotone in the manipulated variable, i.e. consistent
with seed noise (stds 0.1–0.21 at n=3). The next-token secondary readout
(P(S_M|O) at matched narrow: 0.37–0.57 across priors) is equally flat.

*Provenance note: the corr-moderate condition was re-run 2026-07-01 after discovering a
Modal container-reuse bug (import-time env caching) had made its original "seed 1" a
duplicate of seed 0; all other conditions were verified clean (seed field matches
directory in every metadata.json). The wrappers now set max_inputs=1.*

## The mechanism that survives: a global SGD update, not representational necessity

Drift decomposition (layer 2, signed mean projection of resid(ckpt)−resid(base) onto the
base P_M direction):

| step | product | corr-strong | anti-strong |
|---|---|---|---|
| 5 | −0.06 | **0.53** | 0.29 |
| 9 | 0.22 | **1.14** | 0.52 |
| 18 | 0.67 | **1.73** | 0.42 |

Fine-tuning adds a domain-global persona bias in *every* condition — largest in
corr-strong, the condition where the network most clearly possesses sector-specific
coordinates it could have used instead. Drift variance stays > 99.9% orthogonal to the
probe span in all conditions (frac_probe_span ≤ 0.001).

## Interpretation

The control refutes the **representational-capacity** version of the shared-representation
account ("broad transfer happens because the pretrained code *lacks* sector-specific
persona coordinates"): giving the network those coordinates (verified, r2_det ≈ 0.98)
does not reduce broad transfer. What survives — and is strengthened — is the
**optimizer-preference** version: SGD on MD data moves the global persona feature
regardless of whether a sector-targeted update is representable, because the persona
feature participates in prediction everywhere and therefore carries the dominant gradient.

This arguably makes the toy *more* faithful to LLM emergent misalignment, not less: LLMs
certainly can represent domain-specific misalignment, yet narrow fine-tuning still
generalizes broadly. In both cases the binding fact is the update direction, not the
representational menu. The exact-inference contrast remains the sharp statement of
non-triviality: ideal sector-level inference shows zero broad transfer at every dose
(saturated null, per-prior), and the transformer's λ ∈ [0.15, 0.57] quantifies how far
SGD departs from it toward the shared/tilted update.

## Caveats

- n=3 seeds per condition with large seed variance; the flat pattern needs ~10 seeds per
  condition to bound an effect below ~0.1 excess. Cheap on Modal (parallel containers).
- Matched-narrow interpolation rides on a coarse checkpoint grid (10 checkpoints,
  18 steps at lr 2e-3); a finer grid would reduce interpolation error.
- Layer-2 conditional persona-probe transfer metrics are unstable across seeds (both
  signs observed within conditions); base sharing↔excess per-seed scatter is
  correspondingly uninformative. r2_det is the stable representational readout.
- λ inherits noise from both the excess and the tilted-curve interpolation.

## Suggested follow-ups

1. **Gradient-level test of the surviving hypothesis** (most decisive next step):
   at FT step 0, project the actual MD-batch gradient's induced activation change onto
   the base P_M direction vs the sector-specific (det) direction, per prior. Predicts
   dominant P_M component in all conditions.
2. 10 seeds per condition to bound the null effect.
3. Finer FT checkpoint grid (every step, 1–18) for tighter matched-narrow curves.

## Artifacts

- `matched_narrow_table.csv`, `matched_narrow_summary.csv`, `gates.csv`,
  `representation_base.csv`, `representation_summary.csv`
- Figures: `matched_narrow_curves.png`, `excess_lambda_vs_lnor.png`,
  `representation_vs_lnor.png`, `sharing_vs_excess_scatter.png`
- Inputs: `results_probe_factorization_pi0_<tag>/` (5 conditions × 3 seeds),
  `results_bayes_null_pi0_<tag>/` (5 conditions, 3 ideal learners each)
