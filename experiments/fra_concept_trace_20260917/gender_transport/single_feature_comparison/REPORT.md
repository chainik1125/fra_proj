# Can one feature invert queen versus king?

Measured 17 September 2026, GPT-2 Small, matched prefixes `A female monarch is called a` and `A male monarch is called a`. No answer token is supplied. The outcome is M = logit(` queen`) − logit(` king`); inversion means the sign of M reverses relative to that prefix’s clean baseline. It does not mean the top generated word becomes queen or king.

**Result:** a single feature can invert the preference. On the same OV edge, adding male-related feature 25975 to the female prompt crosses at α≈5.35, and adding women-related feature 20446 to the male prompt crosses at α≈4.50. Both additions retain nonnegative coefficients. The corresponding four-feature edge edits cross at α≈1.99 and 1.34, respectively, but extrapolate the suppressed features below zero. Ordinary source-residual edits also invert both prompts. Across all positions, replacing only feature 15560 with its female-prompt coefficients inverts the male prompt even at unit strength. No pure unit deletion of a single feature in the 40-candidate scan inverts either prompt in any tested scope. Thus four-feature coordination is not necessary for inversion; this comparison does not establish a selective advantage of FRA.

## Interventions and feature selection

We reuse the previous L8H11 edge, source position 1 (`female/male`) to query position 5 (final `a`), and the OpenAI 32,768-latent TopK SAE (k=32), SAE Lens release `gpt2-small-resid-post-v5-32k`, training hook `blocks.7.hook_resid_post`. For the OV method, its normalized dictionary is transferred to `blocks.8.ln1.hook_normalized`, using the prior transfer audit. Ordinary residual edits use the actual training hook.

Three scopes are compared:

- **One OV edge:** change only the feature contribution to L8H11’s source-to-final-query message, with attention weights fixed.
- **Source residual:** edit the SAE coefficient at `female/male` in the post-layer-7 residual. All subsequent consumers of that source representation can be affected.
- **All-position residual:** apply the same feature-ID edit at every position in the six-token prefix, including the final query.

A unit donor edit substitutes the opposite-gender coefficient; α multiplies that change. A unit deletion sets the original feature coefficient to zero. The original SAE reconstruction error, mean and decoder bias are retained. Residual edits use each base token’s native decoder scale: Δx = α σ_base Σ_i (z_donor,i − z_base,i) D_i. OV edits retain the previous exact convention Δz_head = α A_base Σ_i (a_donor,i − a_base,i) D_i W_V, where a = σ z at ln1.

α>1 extrapolates beyond the donor. If an active feature is reduced to a zero donor coefficient, going beyond α=1 makes its effective coefficient negative. Such an edit is signed steering, not deletion. Adding a feature absent in the base can remain nonnegative above α=1, but is still beyond the observed donor magnitude. Re-encoding an edited residual can change other inferred SAE coefficients because this is an overcomplete dictionary; “one feature” describes the one decoder component edited.

The unit-strength scan tests all 40 features whose source coefficients differ between the two prompts. The strength sweep tests the four previously selected causal features, separately and jointly, at α = 0, 0.25, 0.5, 1, 2, 4, 8, 16. The first detected grid crossing is refined by ten bisections. These are local crossing brackets, not guaranteed global minima over a potentially nonmonotone response curve. No new feature selection uses the present sweep outcomes.

## Features

| Feature | Automatic annotation (hypothesis) | Female source z | Male source z |
| --- | --- | --- | --- |
| 25975 | references to male gender and masculinity | 0.0000 | 4.6695 |
| 20446 | references to women in various contexts | 3.5555 | 0.0000 |
| 15560 | references to specific individuals, particularly women involved in political or social contexts | 2.7406 | 0.0000 |
| 3281 | references to men or discussions related to their health and societal roles | 0.0000 | 3.1026 |

## Does one unit edit already invert the preference?

Counts of inversions among the 40 source-difference candidates. The linked raw results include every feature, including inactive-feature no-ops.

| Scope | Unit edit | Female → king preference | Male → queen preference |
| --- | --- | --- | --- |
| One OV edge | donor | 0/40 | 0/40 |
| One OV edge | delete | 0/40 | 0/40 |
| Source residual | donor | 0/40 | 0/40 |
| Source residual | delete | 0/40 | 0/40 |
| All-position residual | donor | 0/40 | 1/40 (15560) |
| All-position residual | delete | 0/40 | 0/40 |

For the four named gender-related features, individual unit deletions give:

| Base | Deleted feature | OV-edge M | Source-residual M | All-position-residual M |
| --- | --- | --- | --- | --- |
| female | 25975 | +1.2102 | +1.2102 | +1.2102 |
| female | 20446 | +1.0487 | +0.9492 | +0.9492 |
| female | 15560 | +1.0916 | +1.0535 | +0.2227 |
| female | 3281 | +1.2102 | +1.2102 | +1.2102 |
| male | 25975 | -0.9666 | -0.8413 | -0.8413 |
| male | 20446 | -1.3111 | -1.3111 | -1.3111 |
| male | 15560 | -1.3111 | -1.3111 | -1.3111 |
| male | 3281 | -1.2083 | -1.3653 | -1.3653 |

## Strength required for inversion

Each cell is the non-inverted/inverted α bracket at the first detected crossing. **†** marks a negative effective coefficient at the crossing. All other crossings retain nonnegative effective coefficients. Four-feature rows change four coefficients simultaneously; equal α is not an equal edit norm. Different scopes also have different consumers, so these comparisons establish feasibility, not an FRA selectivity advantage.

| Base | Feature(s) | One OV edge | Source residual | All-position residual |
| --- | --- | --- | --- | --- |
| female | 25975 | 5.344–5.348 | 4.336–4.340 | 4.336–4.340 |
| female | 20446 | 6.781–6.785 † | >16 (no grid crossing) | >16 (no grid crossing) |
| female | 15560 | 9.109–9.117 † | 3.666–3.668 † | 1.129–1.130 † |
| female | 3281 | 15.969–15.977 | >16 (no grid crossing) | >16 (no grid crossing) |
| female | All four | 1.984–1.985 † | 1.171–1.172 † | 0.700–0.700 |
| male | 25975 | 3.844–3.846 † | >16 (no grid crossing) | >16 (no grid crossing) |
| male | 20446 | 4.500–4.504 | 1.675–1.676 | 1.675–1.676 |
| male | 15560 | 5.527–5.531 | 3.207–3.209 | 0.610–0.611 |
| male | 3281 | 12.188–12.195 † | >16 (no grid crossing) | >16 (no grid crossing) |
| male | All four | 1.342–1.343 † | 0.869–0.869 | 0.390–0.390 |

## Probability curves for one edge

The following rows compare selected strengths of the single-feature and four-feature OV edits; every tested strength and threshold refinement is retained in results.json.

| Base | Feature(s) | α | P(queen) | P(king) | M | Negative coefficient? |
| --- | --- | --- | --- | --- | --- | --- |
| female | Baseline | 0 | 5.72% | 1.70% | +1.2102 | no |
| female | 25975 | 1 | 5.20% | 1.91% | +1.0011 | no |
| female | 25975 | 2 | 4.68% | 2.13% | +0.7861 | no |
| female | 25975 | 4 | 3.59% | 2.58% | +0.3313 | no |
| female | 25975 | 8 | 1.55% | 3.18% | -0.7214 | no |
| female | 20446 | 1 | 5.27% | 1.85% | +1.0487 | no |
| female | 20446 | 2 | 4.83% | 1.99% | +0.8852 | yes |
| female | 20446 | 4 | 3.98% | 2.31% | +0.5436 | yes |
| female | 20446 | 8 | 2.33% | 3.05% | -0.2723 | yes |
| female | 15560 | 1 | 5.32% | 1.79% | +1.0916 | no |
| female | 15560 | 2 | 4.94% | 1.87% | +0.9721 | yes |
| female | 15560 | 4 | 4.21% | 2.04% | +0.7259 | yes |
| female | 15560 | 8 | 2.87% | 2.42% | +0.1711 | yes |
| female | 3281 | 1 | 5.48% | 1.76% | +1.1355 | no |
| female | 3281 | 2 | 5.24% | 1.81% | +1.0608 | no |
| female | 3281 | 4 | 4.79% | 1.93% | +0.9117 | no |
| female | 3281 | 8 | 3.97% | 2.15% | +0.6128 | no |
| female | All four | 1 | 4.19% | 2.22% | +0.6361 | no |
| female | All four | 2 | 2.78% | 2.81% | -0.0110 | yes |
| female | All four | 4 | 0.79% | 3.86% | -1.5888 | yes |
| female | All four | 8 | 0.05% | 2.40% | -3.9089 | yes |
| male | Baseline | 0 | 2.03% | 7.54% | -1.3111 | no |
| male | 25975 | 1 | 2.56% | 6.73% | -0.9666 | no |
| male | 25975 | 2 | 3.18% | 5.93% | -0.6221 | yes |
| male | 25975 | 4 | 4.67% | 4.44% | +0.0505 | yes |
| male | 25975 | 8 | 7.63% | 2.32% | +1.1898 | yes |
| male | 20446 | 1 | 2.50% | 6.93% | -1.0198 | no |
| male | 20446 | 2 | 3.06% | 6.32% | -0.7253 | no |
| male | 20446 | 4 | 4.45% | 5.13% | -0.1411 | no |
| male | 20446 | 8 | 7.78% | 3.28% | +0.8649 | no |
| male | 15560 | 1 | 2.42% | 7.11% | -1.0766 | no |
| male | 15560 | 2 | 2.89% | 6.66% | -0.8368 | no |
| male | 15560 | 4 | 4.04% | 5.76% | -0.3527 | no |
| male | 15560 | 8 | 6.90% | 4.22% | +0.4916 | no |
| male | 3281 | 1 | 2.19% | 7.33% | -1.2083 | no |
| male | 3281 | 2 | 2.36% | 7.13% | -1.1045 | yes |
| male | 3281 | 4 | 2.74% | 6.70% | -0.8939 | yes |
| male | 3281 | 8 | 3.66% | 5.81% | -0.4625 | yes |
| male | All four | 1 | 3.96% | 5.50% | -0.3293 | no |
| male | All four | 2 | 6.76% | 3.78% | +0.5824 | yes |
| male | All four | 4 | 11.75% | 1.77% | +1.8929 | yes |
| male | All four | 8 | 16.76% | 0.24% | +4.2380 | yes |

## Checks and scope

- All 80 prior unit OV single-feature outcomes (40 per direction) are reproduced to within 1e−4 logits. Actual maximum discrepancy: 1.14e-05.
- Native SAE decode-after-deletion agrees with the analytic residual subtraction; maximum absolute discrepancy 3.34e-06.
- Zero-strength and inactive-component edits are exact no-ops. Single-edge OV edits leave every earlier query’s logits bitwise unchanged.
- Candidate probabilities are measured over the full vocabulary. The final-token KL and edit norms are saved for inspection but do not measure protected capability retention.
- This comparison uses only the original two prefixes. The single-feature thresholds have not been validated across paraphrases, and no matched-effect collateral comparison has been conducted.

## Reproduce

```bash
uv run --no-project --with sae-lens==6.46.1 --with transformer-lens==3.9.0 python experiments/fra_concept_trace_20260917/compare_single_transport.py
python3 experiments/fra_concept_trace_20260917/report_single_transport.py
```

[Protocol](protocol.json) · [All causal runs](results.json) · [Crossing brackets](thresholds.json) · [Baseline](baseline.json) · [Checks](checks.json) · [Manifest](manifest.json) · [Parent investigation](../REPORT.md).

Follow-up: [matched-inversion KL on the rest of the sentence](../matched_kl/REPORT.md) compares the methods at equal queen/king margins, with fixed continuations and a prediction-position SAE control.
