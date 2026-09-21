# Matched-inversion KL on the rest of the sentence

Measured 18 September 2026. GPT-2 Small, OpenAI 32k TopK SAE (k=32), post-layer-7 dictionary; OV measurements use its audited transfer to layer-8 ln1. Layer/head numbering is zero based.

**Result:** at the matched primary target, the all-prefix single-feature residual edit has 116.4× the four-feature FRA edit’s KL on predictions after the answer, and 168.2× its KL averaged over all other positions. A single-feature edit confined to the same OV edge has essentially the same low KL as the four-feature edit. The query-local SAE control also has higher after-answer KL, although the gap is strongly asymmetric between the two gender directions. These observations support lower collateral for the tested OV interventions, rather than a need for four-feature coordination; they do not establish an advantage over every equally localized attention or vector intervention.

Follow-up: [the matched-feature, matched-source-scope comparison](../matched_scopes/REPORT.md) removes the extra destination restriction from FRA. Its all-token advantage is about 4.9× for head 11 and 4.7× across all heads, rather than the hundred-fold gap in the earlier differently scoped comparison below.

## Comparison

Each method is calibrated to the same queen-minus-king logit margin at `A female/male monarch is called a`. The primary targets are −0.5 for the female prompt and +0.5 for the male prompt: the desired opposite-gender answer has e^0.5 ≈ 1.65 times the probability of the original-gender answer. This matches the queen/king odds, not the absolute probability mass assigned to these two words. A stronger sensitivity check uses margins −1 and +1 (odds ≈ 2.72). All outcome tables exclude the answer prediction itself.

The primary teacher-forced sentence family is:

```text
A {female/male} monarch is called a {queen/king} and usually rules a kingdom.
A {female/male} monarch is called a {queen/king} and lives in a royal palace.
A {female/male} monarch is called a {queen/king} and performs official duties.
```

Both answer choices are supplied for both source genders. Clean and edited models receive exactly the same tokens in each comparison. The counterfactual donor also receives that same answer and continuation and differs only at the source gender word. Thus future teacher-forced answers cannot be used to produce the earlier inversion. Supplying an answer is not counted as generation success.

At every other prediction position, we compute **KL(clean next-token distribution ‖ edited next-token distribution)** over the full vocabulary, using float64 and natural logarithms. Before-answer positions are queries 0–4; query 5 predicts queen/king and is excluded. After-answer positions predict all supplied suffix tokens, including punctuation. The final punctuation’s prediction of an unsupplied next token is excluded. The initial `A` has no preceding prediction. Tables report means in nats per prediction position, with equal weighting of the twelve complete sentence variants unless a subgroup is named. These are conditional KL measurements along fixed histories, not an estimated KL between full free-running sentence distributions.

## Primary result: matched 1.65:1 inverted odds

| Intervention | Before answer | After answer | All other positions | Relative to four-feature FRA |
| --- | --- | --- | --- | --- |
| FRA OV, four features | 0 (exact) | 0.000211 | 0.000114 | 1.00× |
| FRA OV, single feature | 0 (exact) | 0.000212 | 0.000114 | 1.00× |
| SAE single feature, source token | 0.266602 | 0.049796 | 0.148645 | 1308.42× |
| SAE 15560, all prefix positions | 0.013060 | 0.024567 | 0.019112 | 168.23× |
| SAE 15560, prediction position only | 0 (exact) | 0.025708 | 0.013729 | 120.85× |
| SAE 15560, whole sentence | 0.013060 | 0.031790 | 0.022888 | 201.47× |

The zero pre-answer KL for FRA is structural: the edit only changes an attention output at query 5 and cannot affect earlier positions. The prediction-position SAE control receives the same positional restriction and also has exactly zero pre-answer KL. The after-answer column therefore gives a more informative comparison of propagated changes than the prefix alone.

The all-prefix SAE method edits only the original six-token prefix when evaluating the full sentence; this isolates propagation after the answer. The whole-sentence SAE method continues editing feature 15560 at every position, including the supplied answer and suffix. These are different scopes and are reported separately.

## Each direction separately

| Source gender | Intervention | Before answer | After answer | All other positions | Relative to FRA |
| --- | --- | --- | --- | --- | --- |
| female | FRA OV, four features | 0 (exact) | 0.000198 | 0.000106 | 1.00× |
| female | FRA OV, single feature | 0 (exact) | 0.000191 | 0.000103 | 0.97× |
| female | SAE single feature, source token | 0.494319 | 0.070310 | 0.263750 | 2482.71× |
| female | SAE 15560, all prefix positions | 0.016381 | 0.026540 | 0.021643 | 203.72× |
| female | SAE 15560, prediction position only | 0 (exact) | 0.050174 | 0.026789 | 252.17× |
| female | SAE 15560, whole sentence | 0.016381 | 0.036420 | 0.026792 | 252.20× |
| male | FRA OV, four features | 0 (exact) | 0.000225 | 0.000121 | 1.00× |
| male | FRA OV, single feature | 0 (exact) | 0.000233 | 0.000125 | 1.04× |
| male | SAE single feature, source token | 0.038885 | 0.029282 | 0.033541 | 277.25× |
| male | SAE 15560, all prefix positions | 0.009740 | 0.022595 | 0.016582 | 137.07× |
| male | SAE 15560, prediction position only | 0 (exact) | 0.001241 | 0.000669 | 5.53× |
| male | SAE 15560, whole sentence | 0.009740 | 0.027159 | 0.018983 | 156.92× |

## Calibration strengths and intervention definitions

A donor-sized edit (α=1) replaces selected SAE coefficients by their opposite-gender values. Larger α extrapolates. OV changes only the selected feature contributions along L8H11 from source position 1 to query position 5, keeping base attention weights. Residual edits change the relevant decoder components at post-layer 7, using each base token’s native SAE normalization scale and retaining the original mean, decoder bias and reconstruction error. “Single feature” names one edited decoder component, not an assertion that other SAE coefficients stay fixed after re-encoding.

The single-feature source and OV methods use male-related 25975 in the female prompt and women-related 20446 in the male prompt, as in the previous comparison. Broad residual methods use 15560. The query-only control tested all four previously selected features [25975, 20446, 15560, 3281], selecting a reachable candidate without consulting continuation outcomes. All reachable query-local edits tie at zero pre-answer KL, so candidate order resolves ties. † indicates at least one negative effective coefficient on the calibration prefix.

| Source | Intervention | Features | α | Actual margin | P(queen) | P(king) |
| --- | --- | --- | --- | --- | --- | --- |
| female | FRA OV, four features | 25975,20446,15560,3281 | 2.6498 † | -0.500035 | 1.956% | 3.224% |
| female | FRA OV, single feature | 25975 | 7.2185 | -0.499971 | 1.891% | 3.118% |
| female | SAE single feature, source token | 25975 | 6.2219 | -0.499955 | 2.223% | 3.665% |
| female | SAE 15560, all prefix positions | 15560 | 1.3832 † | -0.500113 | 2.196% | 3.622% |
| female | SAE 15560, prediction position only | 15560 | 11.3286 † | -0.500037 | 0.417% | 0.687% |
| female | SAE 15560, whole sentence | 15560 | 1.3832 † | -0.500113 | 2.196% | 3.622% |
| male | FRA OV, four features | 25975,20446,15560,3281 | 1.9009 † | +0.500002 | 6.467% | 3.923% |
| male | FRA OV, single feature | 20446 | 6.4055 | +0.500053 | 6.457% | 3.916% |
| male | SAE single feature, source token | 20446 | 2.6740 | +0.499961 | 5.938% | 3.602% |
| male | SAE 15560, all prefix positions | 15560 | 0.9632 | +0.499979 | 6.014% | 3.648% |
| male | SAE 15560, prediction position only | 15560 | 3.3541 | +0.500046 | 5.362% | 3.252% |
| male | SAE 15560, whole sentence | 15560 | 0.9632 | +0.499979 | 6.014% | 3.648% |

Equal α is not equal edit magnitude or equal causal effect; the KL comparison uses the matched margins above. This is a comparison among the specified interventions, not a search for the best possible residual steer.

The female query-only SAE control reaches the desired odds partly while reducing total queen/king probability mass: P(king)=0.687%, versus 3.224% for four-feature FRA. It is therefore a weaker answer-probability match, despite the identical log-odds target. The source and all-prefix single-feature edits retain comparable or slightly higher P(king) (3.665% and 3.622%). The query-control ratio should not be interpreted as a comparison matched on absolute answer probability.

## Does the forced answer change the result?

These primary-target subgroup results separate semantically consistent clean sentences (`female … queen`, `male … king`) from the opposite forced answers. Each cell averages six sentence variants.

| Intervention | Rest KL, original answer | Rest KL, opposite answer | After-answer KL, original | After-answer KL, opposite |
| --- | --- | --- | --- | --- |
| FRA OV, four features | 0.000101 | 0.000126 | 0.000187 | 0.000235 |
| FRA OV, single feature | 0.000104 | 0.000124 | 0.000194 | 0.000230 |
| SAE single feature, source token | 0.144920 | 0.152371 | 0.042801 | 0.056791 |
| SAE 15560, all prefix positions | 0.014987 | 0.023238 | 0.016779 | 0.032356 |
| SAE 15560, prediction position only | 0.017361 | 0.010097 | 0.032551 | 0.018864 |
| SAE 15560, whole sentence | 0.017998 | 0.027777 | 0.022506 | 0.041074 |

## Stronger inversion: matched 2.72:1 odds

Missing entries mean that the intervention did not reach the stronger target on the fixed α grid through 16; they are not included as successful comparisons.

| Source | Intervention | After-answer KL | All other positions | Relative to FRA |
| --- | --- | --- | --- | --- |
| female | FRA OV, four features | 0.000375 | 0.000201 | 1.00× |
| female | FRA OV, single feature | 0.000348 | 0.000187 | 0.93× |
| female | SAE single feature, source token | unmatched | unmatched | — |
| female | SAE 15560, all prefix positions | 0.043974 | 0.034443 | 171.03× |
| female | SAE 15560, prediction position only | 0.056964 | 0.030376 | 150.84× |
| female | SAE 15560, whole sentence | 0.057599 | 0.041549 | 206.32× |
| male | FRA OV, four features | 0.000387 | 0.000208 | 1.00× |
| male | FRA OV, single feature | 0.000391 | 0.000210 | 1.01× |
| male | SAE single feature, source token | 0.075831 | 0.082706 | 397.29× |
| male | SAE 15560, all prefix positions | 0.044404 | 0.032595 | 156.57× |
| male | SAE 15560, prediction position only | 0.002652 | 0.001430 | 6.87× |
| male | SAE 15560, whole sentence | 0.052675 | 0.036925 | 177.37× |

## Verification and limits

- Every successful calibration and every full-sentence rerun reaches its intended margin to within 0.000116 logits. Prefix-only and full-sequence baseline answer logits agree within 1e−4.
- The same tokens are supplied to the clean and edited model. Zero edits are exact no-ops. Earlier logits are bitwise unchanged for the OV and query-local SAE interventions.
- Full-vocabulary KL is computed in float64; negligible negative roundoff is clipped only after checking it is above −1e−10. Per-position distributions of KL and actual-token log probabilities are retained.
- KL measures distributional change, not whether every change is harmful. The protocol does not assess a broad set of preserved capabilities or free-running continuations.
- Conclusions apply to two source prefixes, three short hand-written continuations, the selected features, and the specified intervention scopes. An advantage over these baselines would not by itself establish a general FRA advantage or a royalty-specific feature conjunction.

## Reproduce and inspect

```bash
uv run --no-project --with sae-lens==6.46.1 --with transformer-lens==3.9.0 python experiments/fra_concept_trace_20260917/compare_transport_kl.py
python3 experiments/fra_concept_trace_20260917/report_transport_kl.py
```

[Protocol](protocol.json) · [Calibration](calibration.json) · [Calibration sweep](calibration_scan.json) · [Every token measurement](continuations.json) · [Aggregates](summary.json) · [Checks](checks.json) · [Manifest](manifest.json) · [Previous single-feature experiment](../single_feature_comparison/REPORT.md).

The run used 529 edited forward passes and produced 138 full-sentence evaluations. The re-downloadable layer-4 and layer-5 weights from this experiment’s temporary cache were removed to free disk space, after matching their hashes against the saved manifest; current model and layer-7 weights were retained. See [cache cleanup record](cache_cleanup.json).
