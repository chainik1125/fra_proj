# FRA versus SAE with matched token scopes and feature IDs

Measured 18 September 2026 on GPT-2 Small. All layer/head numbers are zero based.

**Result:** when the same four features are edited at every source token and FRA can affect every causal destination, after-answer KL is 0.006511 for head-11 FRA, 0.006811 for all-head FRA, and 0.032199 for SAE residual edits. The SAE/FRA ratios are 4.95× and 4.73×, respectively. The advantage survives these controls, but the earlier hundred-fold gap came from a different comparison that restricted FRA to one destination. Its magnitude should not be carried over to the matched-scope result.

## What is held constant

The main comparison uses exactly the same four feature IDs (25975, 20446, 15560, 3281) for both methods, the same clean/opposite-gender donor prompts, the same source-token set, and the same target queen/king effect. A separate single-feature comparison uses feature 15560 in both methods. Feature sets are frozen from the preceding investigation; no features are selected using continuation KL.

For `A₀ female/male₁ monarch₂ is₃ called₄ a₅ …`, define the edited source set S. The main FRA variants change the selected features’ OV contributions from every k in S to **every causal destination q≥k**, at layer 8. There is no manual destination restriction. One variant edits only the previously selected head 11; another edits all 12 heads. SAE edits change those same feature components at those same token positions in the post-layer-7 residual. “All” means all tokens at this layer, not all layers.

This matches source-feature support, not the representation space of the intervention: a residual node edit can affect Q, K, V, other heads and subsequent residual consumers. A V-only edit restricts the channel through which the change is delivered. The all-head variant separates head restriction from this V-versus-residual distinction.

The OpenAI TopK SAE has 32,768 latents and k=32 (`gpt2-small-resid-post-v5-32k`, `blocks.7.hook_resid_post`). FRA uses the same normalized dictionary transferred to `blocks.8.ln1.hook_normalized`. Transfer support/coefficient agreement is audited for every input. The source coefficients change by α(z_donor−z_base), using the **base** SAE standard deviation at each site in both methods, while retaining mean, decoder bias and reconstruction error. This slightly standardizes the earlier FRA convention that included the donor standard deviation; numerical agreement is checked.

```text
SAE:     Δx[k]   = α σ_base[k] Σ_i (z_donor[k,i] − z_base[k,i]) D_i,  k∈S
FRA-OV:  Δz[q,h] = Σ_(k∈S, k≤q) A_base[h,q,k] (Δx_ln1[k] W_V[h])
```

With all destinations permitted, FRA-OV is algebraically equivalent to changing those feature contributions directly in V at the allowed source tokens and heads. That equivalence is verified against an independent value-hook implementation. These experiments concern feature-resolved V/OV control, not a QK cell or concept-conditioned destination gate.

## Matched effect and KL measurement

Each intervention has its own calibrated α. The primary target is M = logit(queen)−logit(king) = −0.5 for the female prompt and +0.5 for the male prompt, giving 1.65:1 odds in favor of the opposite-gender answer. A stronger four-feature comparison uses ±1 (2.72:1 odds). The first crossing on α∈{0,.25,.5,1,2,4,8,16} is refined with thirteen bisections. No crossing means **unmatched**, not a zero-KL success. Odds are matched; absolute answer probabilities are retained separately.

The same three continuations as the preceding KL experiment are teacher-forced: `and usually rules a kingdom.`, `and lives in a royal palace.`, and `and performs official duties.` Each is tested with both `queen` and `king` supplied, for both source genders. Clean and edited models receive identical tokens in each case. The donor differs only in the source gender word; it receives the same supplied answer/tail. This yields 12 sentence variants per successful two-direction comparison.

Full-vocabulary KL(clean‖edited) is computed in float64, in nats per prediction position. The answer prediction at query 5 is excluded. “After answer” covers predictions of every supplied suffix token, including punctuation. “Rest” includes those and queries 0–4. Means weight each sentence variant equally. A pooled value is presented as a matched head-to-head only if both directions reached the target.

## Main four-feature result: KL after the answer

| Identical edited source-token set | FRA OV, head 11 | FRA OV, all 12 heads | SAE residual | SAE / head-11 FRA |
| --- | --- | --- | --- | --- |
| Gender token only: {1} | 0.004651 | 0.007050 | unmatched in one direction | — |
| Prediction token only: {5} | unmatched | unmatched | 0.025708 | — |
| Both selected tokens: {1,5} | 0.004740 | 0.007120 | 0.027669 | 5.84× |
| All six prefix tokens: {0,…,5} | 0.006402 | 0.006582 | 0.028292 | 4.42× |
| All tokens in full sentence | 0.006511 | 0.006811 | 0.032199 | 4.95× |

## Main four-feature result: KL over all other predictions

| Identical edited source-token set | FRA OV, head 11 | FRA OV, all 12 heads | SAE residual | SAE / head-11 FRA |
| --- | --- | --- | --- | --- |
| Gender token only: {1} | 0.004539 | 0.007028 | unmatched in one direction | — |
| Prediction token only: {5} | unmatched | unmatched | 0.013729 | — |
| Both selected tokens: {1,5} | 0.004540 | 0.006984 | 0.025377 | 5.59× |
| All six prefix tokens: {0,…,5} | 0.005523 | 0.006592 | 0.022961 | 4.16× |
| All tokens in full sentence | 0.005579 | 0.006717 | 0.025005 | 4.48× |

For the prefix scope, only sources 0–5 are edited; FRA still transmits their changed value content to all later destinations. For the sentence scope, source positions in the supplied answer and continuation are also eligible. Thus the prefix row is not secretly limited to prefix destinations.

## How much did the original destination restriction contribute?

The following all use four features, head 11, and the same target inversion, but intentionally vary source/destination support. They isolate the restriction that was present in the earlier headline result. These are not all identical-source-set comparisons.

| Head-11 OV scope | Before answer | After answer | Rest |
| --- | --- | --- | --- |
| Source 1 → query 5 only (old intervention) | 0 (exact) | 0.000211 | 0.000114 |
| Source 1 → all causal queries | 0.004457 | 0.004651 | 0.004539 |
| All causal sources → query 5 only | 0 (exact) | 0.000227 | 0.000122 |
| All sources → all causal queries | 0.004570 | 0.006511 | 0.005579 |

The prediction-only source row in the main table means editing features **carried by token 5** and allowing them to reach all causal readers. It is not the original source-1→query-5 edge. Editing a token’s own V contribution can be too weak to invert the answer even when directly editing its residual representation succeeds.

## Directional results and answer probabilities

Each row averages the six continuation variants for one source gender. Probabilities are measured at the calibrated prefix. † means an effective feature coefficient becomes negative on the calibration prefix; α>1 is extrapolation, not deletion.

| Source | Scope | Method | α | P(queen) | P(king) | After-answer KL | Rest KL |
| --- | --- | --- | --- | --- | --- | --- | --- |
| female | source | fra_h11 | 2.1722 † | 2.120% | 3.495% | 0.004741 | 0.005276 |
| female | source | fra_all_heads | 2.9498 † | 2.269% | 3.741% | 0.007675 | 0.008573 |
| female | source | sae_residual | 1.4443 † | 2.063% | 3.402% | 0.050522 | 0.050640 |
| female | prediction | fra_h11 | unmatched | — | — | — | — |
| female | prediction | fra_all_heads | unmatched | — | — | — | — |
| female | prediction | sae_residual | 11.3286 † | 0.417% | 0.687% | 0.050174 | 0.026789 |
| female | source_and_prediction | fra_h11 | 2.1512 † | 2.117% | 3.490% | 0.004843 | 0.005268 |
| female | source_and_prediction | fra_all_heads | 2.9286 † | 2.267% | 3.737% | 0.007877 | 0.008611 |
| female | source_and_prediction | sae_residual | 1.1157 † | 2.126% | 3.506% | 0.028602 | 0.027094 |
| female | prefix | fra_h11 | 1.8784 † | 2.096% | 3.456% | 0.006949 | 0.006471 |
| female | prefix | fra_all_heads | 2.2589 † | 2.387% | 3.935% | 0.007972 | 0.008566 |
| female | prefix | sae_residual | 0.8569 | 2.226% | 3.670% | 0.031093 | 0.025935 |
| female | sentence | fra_h11 | 1.8784 † | 2.096% | 3.456% | 0.007113 | 0.006556 |
| female | sentence | fra_all_heads | 2.2589 † | 2.387% | 3.935% | 0.008284 | 0.008736 |
| female | sentence | sae_residual | 0.8569 | 2.226% | 3.670% | 0.036491 | 0.028752 |
| male | source | fra_h11 | 1.5497 † | 6.318% | 3.832% | 0.004560 | 0.003802 |
| male | source | fra_all_heads | 1.9711 † | 5.779% | 3.505% | 0.006424 | 0.005484 |
| male | source | sae_residual | unmatched | — | — | — | — |
| male | prediction | fra_h11 | unmatched | — | — | — | — |
| male | prediction | fra_all_heads | unmatched | — | — | — | — |
| male | prediction | sae_residual | 3.3541 | 5.362% | 3.252% | 0.001241 | 0.000669 |
| male | source_and_prediction | fra_h11 | 1.5319 † | 6.326% | 3.837% | 0.004637 | 0.003811 |
| male | source_and_prediction | fra_all_heads | 1.9273 † | 5.815% | 3.527% | 0.006364 | 0.005358 |
| male | source_and_prediction | sae_residual | 0.9455 | 6.283% | 3.811% | 0.026737 | 0.023660 |
| male | prefix | fra_h11 | 1.2829 † | 6.408% | 3.887% | 0.005855 | 0.004574 |
| male | prefix | fra_all_heads | 1.3537 † | 5.810% | 3.524% | 0.005193 | 0.004618 |
| male | prefix | sae_residual | 0.5986 | 6.169% | 3.741% | 0.025492 | 0.019986 |
| male | sentence | fra_h11 | 1.2829 † | 6.408% | 3.887% | 0.005908 | 0.004602 |
| male | sentence | fra_all_heads | 1.3537 † | 5.810% | 3.524% | 0.005339 | 0.004697 |
| male | sentence | sae_residual | 0.5986 | 6.169% | 3.741% | 0.027906 | 0.021259 |

## Same single feature in every method: 15560

This controls feature count and identity as well as source support. These results use the primary ±0.5 target.

| Edited source-token set | FRA OV, head 11 | FRA OV, all 12 heads | SAE residual | SAE / head-11 FRA |
| --- | --- | --- | --- | --- |
| Gender token only: {1} | 0.007195 | 0.008939 | unmatched in one direction | — |
| Prediction token only: {5} | unmatched | unmatched | 0.025708 | — |
| Both selected tokens: {1,5} | 0.007588 | 0.010739 | 0.019093 | 2.52× |
| All six prefix tokens: {0,…,5} | 0.013406 | 0.009091 | 0.024567 | 1.83× |
| All tokens in full sentence | 0.013927 | 0.009883 | 0.031790 | 2.28× |

## Stronger four-feature inversion: ±1 margin

| Edited source-token set | FRA OV, head 11 | FRA OV, all 12 heads | SAE residual | SAE / head-11 FRA |
| --- | --- | --- | --- | --- |
| Gender token only: {1} | 0.007750 | 0.012103 | unmatched in one direction | — |
| Prediction token only: {5} | unmatched | unmatched | 0.029808 | — |
| Both selected tokens: {1,5} | 0.007895 | 0.012245 | 0.046836 | 5.93× |
| All six prefix tokens: {0,…,5} | 0.010681 | 0.011258 | 0.049215 | 4.61× |
| All tokens in full sentence | 0.010857 | 0.011632 | 0.055278 | 5.09× |

## Forced-answer sensitivity

Primary four-feature results split according to whether the supplied answer agrees with the source gender. Each matched pooled cell averages six sentence variants.

| Scope | Method | After-answer KL, original answer | After-answer KL, opposite answer |
| --- | --- | --- | --- |
| source | fra_h11 | 0.003493 | 0.005808 |
| source | fra_all_heads | 0.006236 | 0.007864 |
| source | sae_residual | unmatched in one direction | unmatched in one direction |
| source_and_prediction | fra_h11 | 0.003548 | 0.005932 |
| source_and_prediction | fra_all_heads | 0.006319 | 0.007921 |
| source_and_prediction | sae_residual | 0.022445 | 0.032894 |
| prefix | fra_h11 | 0.004576 | 0.008229 |
| prefix | fra_all_heads | 0.005371 | 0.007794 |
| prefix | sae_residual | 0.020300 | 0.036284 |
| sentence | fra_h11 | 0.004637 | 0.008384 |
| sentence | fra_all_heads | 0.005601 | 0.008022 |
| sentence | sae_residual | 0.023096 | 0.041301 |

## Checks and limits

- The largest mismatch from the requested queen/king margin in any full-sentence evaluation is 0.000159 logits. Prefix-only and teacher-forced full-sequence answer predictions agree.
- Independently editing V reproduces the all-destination FRA implementation to within 1.34e-05 absolute logit error.
- Across all measured inputs, ln1 transfer changes 0 token supports; maximum coefficient difference is 0.000156. Exact feature-plus-error accounting remains at the actual attention input.
- Zero edits are exact no-ops. Logits before the earliest direct write are bitwise unchanged. In particular, removing a destination mask does not retain the earlier zero-KL guarantee for prefix tokens.
- These are donor-coefficient substitutions and extrapolations, not removal experiments. Signed coefficients are explicitly flagged.
- KL measures distributional change rather than harm. Matching queen/king odds alone does not match absolute probability mass; inspect the reported probabilities before interpreting a large ratio.
- The all-head OV control is still V-only and leaves QK and the residual bypass unchanged. It is not equivalent to a full residual SAE edit, nor does this experiment establish that FRA outperforms arbitrary localized vector steering or attention masks.
- The result is limited to two source prefixes and three constructed continuations. Free-running generation, broader feature families and automatic content-conditioned destination selection remain untested.

## Reproduce

```bash
uv run --no-project --with sae-lens==6.46.1 --with transformer-lens==3.9.0 python experiments/fra_concept_trace_20260917/compare_matched_scopes.py
python3 experiments/fra_concept_trace_20260917/report_matched_scopes.py
```

[Protocol](protocol.json) · [Baseline](baseline.json) · [Calibration](calibration.json) · [Full calibration sweep](calibration_scan.json) · [Per-token continuation results](continuations.json) · [Aggregates](summary.json) · [Checks](checks.json) · [Manifest](manifest.json).

The completed run contains 1934 edited forwards, 102 calibration conditions (87 matched), and 522 sentence evaluations.

Supplemental calibration check: every unmatched case within 0.1 logits of its target was sampled at 65 evenly spaced strengths between the neighboring coarse-grid values. The male source-only residual edit reached a sampled maximum margin of 0.492357 (four, α=2.1250), 0.444278 (single_15560, α=9.2500). Neither crossed +0.5. This is a denser local check, not a proof of a global maximum. [Protocol](refinement_protocol.json) · [Measurements](near_miss_refinement.json). Reproduce with `uv run --no-project --with sae-lens==6.46.1 --with transformer-lens==3.9.0 python experiments/fra_concept_trace_20260917/refine_scope_near_misses.py`, then rerun this report generator.
