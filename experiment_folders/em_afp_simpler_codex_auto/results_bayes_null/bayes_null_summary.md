# Exact-inference ideal-learner baselines (Bayes null) for the low-prior headline run

*2026-07-01. Script: `experiments/special_sfp_bayes_null.py`. Process and readout conventions
match `results_low_prior_confirm_p005_bigmodel_base20k_lr2e3` (aligned-off headline process,
P(M)=0.05, same `domain_prompt`, same continuation sector-rate labeling, gen_len 32,
n=4096 rollouts per cell).*

## Question

What would a "perfect estimator" do on O prompts after MD-only fine-tuning? Two idealized
learners bracket the answer. Both see exactly the same fine-tuning evidence (N unambiguous
MD sequences, parametrized as dose t = N/κ relative to the pretraining pseudocount), and the
fine-tuning likelihood is identical under both — so where the trained transformer lands
between them is purely inductive bias, not data.

- **Saturated Bayes**: free 4-vector sector prior, Dirichlet-updated. All MD evidence is
  absorbed by the MD weight: π(t) = (π₀ + t·e_MD)/(1+t).
- **Product prior**: prior constrained to persona ⊗ domain marginals; MD sequences update
  both: m(t) = (0.05+t)/(1+t), d(t) = (0.5+t)/(1+t).

## Results

| learner | dose t | π_MD | P(M \| O-prompt) | O→MO | O→MD | D→MD |
|---|---|---|---|---|---|---|
| saturated | 0 | 0.025 | 0.0013 | 0.0015 | 0 | 0.0015 |
| saturated | 10 | 0.911 | 0.0013 | 0.0024 | 0 | 0.331 |
| saturated | 1000 | 0.999 | 0.0013 | 0.0012 | 0 | 0.916 |
| product | 10 | 0.872 | 0.213 | 0.204 | 0 | 0.197 |
| product | 30 | 0.954 | 0.447 | 0.419 | 0 | 0.406 |
| product | 1000 | 0.999 | 0.964 | 0.892 | 0 | 0.896 |
| **transformer step 0** | — | — | — | **0.0020** | **0** | **0.0017** |
| **transformer step 9** | — | — | — | **0.2944** | **0.0256** | **0.8950** |

Full table: `bayes_null_runs.csv`. Figure: `bayes_null_sector_rates.png`.

## Findings

1. **The saturated learner is exactly invariant on O prompts, at every dose.** The O prompt
   contains S_O emissions, which have zero likelihood under D sectors (α=0), so the inflated
   π_MD is conditioned away entirely and the MO:AO ratio reverts to its pretraining value.
   Broad transfer is flat at the base rate (~0.15%) while narrow learning rises to 0.92.
   "Fine-tuning on MD" and "no fine-tuning at all" are *indistinguishable* to this learner
   conditional on an O prompt.

2. **The exact-inference null for broad transfer is 0.13%, not 5%.** Under the aligned-off
   process (ε_A=0), the nominally persona-neutral prompt is strong A-evidence: the M-factor
   would have emitted S_M during the 5-token prompt with high probability and didn't
   (likelihood ratio ≈ (0.2/0.5)⁴ ≈ 0.026, so 0.0526 prior odds → 0.0013 posterior).
   The transformer's step-9 O→MO = 0.294 is therefore ~150–200× the Bayes-saturated null,
   not ~6× as a naive prior-ratio argument would suggest.

3. **The product learner locks broad and narrow together, and cannot fit the transformer
   either.** Matching the transformer's narrow rate (D→MD ≈ 0.895) requires t ≈ 1000, which
   forces O→MO ≈ 0.89 — 3× the observed broad rate. Matching the observed broad rate
   (0.294) requires t ≈ 15, which gives D→MD ≈ 0.25 — far below the observed narrow rate.

4. **The transformer sits strictly between the poles with a decoupled profile.** At high
   dose its readouts are consistent with a ≈ 1/3 : 2/3 mixture of product and saturated
   learners: λ·0.892 + (1−λ)·0.001 = 0.294 gives λ ≈ 0.33, which simultaneously predicts
   narrow ≈ 0.91 (observed 0.895) and no-sector ≈ 0.71 (observed 0.675). Suggestive, not
   definitive — but "the fine-tuned transformer updates ~1/3 of its posterior behavior as if
   persona were a global factor" is a compact quantitative summary.

5. **MD spillback is off-manifold for both ideal learners.** Given the hard S_O prompt
   evidence, O→MD is exactly 0 for any Bayes-consistent learner over this process. The
   transformer's 0.026 at step 9 (and the much larger late-checkpoint MD-routing) is a
   strictly non-Bayesian corruption — a distinct failure mode, not a prior shift.

6. **Sanity check**: at step 0 the pretrained transformer matches the exact filter at the
   base prior (O→MO 0.0020 vs 0.0015; next-token P(S_M|O) 0.0014 vs 0.0008).

## Implication for the writeup

Broad transfer in the toy is *not* what ideal inference does with the fine-tuning evidence —
the saturated learner shows zero EM at every dose, and the data cannot distinguish the two
learners. The headline result is therefore a statement about the inductive bias of the
pretrained transformer (it partially commits to the factored/product parametrization), not a
consequence of the process construction. Recommended framing: report O→MO as excess over the
exact-inference null (0.294 vs 0.0013), and position the transformer between the two ideal
learners on the dose curves.
