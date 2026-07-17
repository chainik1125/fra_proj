# Belief-probe factorization results (3 seeds, Modal A10G)

*2026-07-01. Script: `experiments/special_sfp_probe_factorization.py`, launched via
`cloud/modal_probe_factorization.py --seeds 0,1,2`. Headline low-prior config
(P(M)=0.05 product prior, 3L/128d model, 20k pretrain steps, MD FT at lr 2e-3).
Moore-Penrose (lstsq) probes from resid_post to exact Bayes sector posteriors,
per layer and FT checkpoint. Combined CSVs in this directory; per-seed figures in
seed{0,1,2}/.*

## Analytic premise (verified numerically each run)

With the product base prior and Kronecker sector structure, the exact Bayes posterior
factorizes as persona-belief × domain-belief for every sequence (max interaction gap = 0).
A factored/shared code is therefore Bayes-sufficient for pretraining; the interaction
degree of freedom is never exercised on-distribution.

## Findings

1. **The belief tracker survives fine-tuning intact.** Sector-posterior probe R² stays
   ≈ 0.99 at every FT checkpoint (layer 2). Stronger: probes *refit* on the fine-tuned
   model decode the O-prompt state as P_M ≈ 0.05 — the correct base-Bayes posterior —
   even at checkpoints where the model behaviorally emits S_M on O prompts at ~43%.
   Fine-tuning does not rewrite the belief computation.

2. **Fine-tuning adds a domain-global shift aligned with the persona readout.** The
   activation drift (resid(ckpt) − resid(base) on fixed base-process data) is > 99.9%
   orthogonal to the base probe span in variance, but its *mean* is systematically
   positive along the base P_M direction, growing monotonically (signed projection
   0 → 0.54 by step 18). The same shift appears under D and O prompts alike
   (base-probe decoded P_M at step 9: 0.53 vs 0.46) — one persona bias applied
   everywhere, not a sector-targeted update.

3. **Decoded persona tracks behavior quantitatively.** Base-probe decoded P_M on the
   O prompt rises 0.003 → 0.456 by step 9, vs behavioral P(S_M|O) = 0.433 at the same
   checkpoint. At late checkpoints the decoded P_D on O prompts also rises (0.37 by
   step 18) — the representational signature of the behavioral MD-routing/spillback.

4. **The base persona code is partially, not perfectly, shared — with real seed
   variance.** Persona belief is perfectly decodable within each domain condition
   (R² ≈ 1.0). Cross-domain persona-probe transfer (fit on P_D ≥ 0.9 contexts, test on
   P_D ≤ 0.1) is strong and symmetric at layers 0–1 (corr ≈ 0.67–0.81) but asymmetric
   and seed-dependent at layer 2: D→O ∈ {0.44, 0.98, 0.98}, O→D ∈ {−0.15, −0.09, 0.94}.
   Partial sharing is consistent with the behavioral result that the transformer sits
   between the saturated (sector-specific) and product (fully shared) ideal learners
   (λ ≈ 1/3 mixture, see results_bayes_null).

5. **No clean per-seed sharing↔transfer correlation at n=3.** The most-shared seed
   (seed 2) has the most persistent broad transfer (P(S_M|O) = 0.54 at step 18), but
   seed 0 (least shared) transfers strongly at step 9. Three seeds cannot settle this;
   the correlated-prior control is the proper causal test.

6. Probe directions rotate substantially during FT but uniformly across all four
   sectors (rot_cos ≈ 0.2 by step 9 for MD, MO, AD, AO alike) — global representational
   drift, not sector-specific rewiring.

## Interpretation

The probes support the shared-representation mechanism in a specific form: broad
misalignment is implemented as a **domain-global persona bias added on top of an intact
belief tracker**, not as a re-inferred sector posterior. Combined with the exact-inference
baselines (results_bayes_null): the behavioral update is inexpressible as any sector-prior
update (those revert on O prompts); representationally it is a shift along the persona
coordinate of the factored code — exactly the product-learner-like component, at partial
strength.

## Next

- Correlated-prior control: pretrain with non-product pi0 (e.g. 0.040, 0.010, 0.460, 0.490
  — note P(M)=0.05 and P(D)=0.5 must stay fixed; the family (a, 0.05−a, 0.5−a, 0.45+a)
  does this) where the Bayes posterior requires the interaction dof; shared-representation
  account predicts reduced broad transfer. Ready via
  `uv run --with modal modal run cloud/modal_probe_factorization.py --seeds 0,1,2 --pi0 ...`
- More seeds for the sharing↔transfer correlation (cheap on Modal: parallel containers).
