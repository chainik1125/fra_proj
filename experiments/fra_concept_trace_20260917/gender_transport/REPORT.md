# A causal gender-transport path in GPT-2 Small

Measured 17 September 2026. All layers and heads are zero based.

**Result:** the strongest content path in the adaptive scan is layer 8, head 11, from the gender token to the final `a`. Four input-SAE features with gender-related annotations recover approximately the whole effect of this message. Unit coefficient substitution changes the queen/king preference without flipping it on the primary prompt; doubling the substitution flips the relative preference in both directions. The same intervention also changes other gender-dependent answers, so this is evidence for a general gender-transport path, not a demonstrated FRA selectivity advantage.

## What was measured

The paired prefixes are `A female monarch is called a` and `A male monarch is called a`. They have six aligned tokens, differ only at position 1, and receive no BOS or answer token. The query is position 5, the final `a`. The outcome is M = logit(` queen`) − logit(` king`); probabilities use the full vocabulary softmax. Baseline M is 1.210249 for female and -1.311116 for male (difference 2.521365).

A donor-effect fraction is (M_edited − M_base)/(M_donor − M_base). It is relative to changing the actual input word, can be negative or exceed 100%, and is not additive across interventions. Each reported causal outcome comes from a full downstream model rerun.

The top baseline next token is a quotation mark on both prompts. These are next-token distribution measurements, not demonstrations of greedy sentences changing from queen to king.

## Locate transport before choosing feature labels

The scan patches Q, K, attention patterns, V, attention outputs, MLP outputs and residuals in both directions, at every layer. Layers 8, 11 and 6 were selected by mean bidirectional attention-output effect; all their heads were tested, then all causal edges in the two strongest heads per layer. The path is strongest within this adaptive scan, not proven strongest among every edge in all 12 layers. The localization stage contains 1,704 interventions.

Selected all-position layer patches, expressed as donor-effect fractions:

| Layer | Patched quantity | Female → male | Male → female |
| --- | --- | --- | --- |
| 5 | pattern | -0.06% | 0.09% |
| 5 | v | -1.18% | -0.74% |
| 5 | attn_out | -1.48% | -0.34% |
| 6 | pattern | -0.71% | 0.71% |
| 6 | v | 8.88% | 23.26% |
| 6 | attn_out | 8.31% | 22.79% |
| 8 | pattern | -0.44% | 0.18% |
| 8 | v | 24.50% | 53.23% |
| 8 | attn_out | 25.61% | 52.19% |
| 11 | pattern | -1.23% | -0.06% |
| 11 | v | 14.99% | 35.48% |
| 11 | attn_out | 16.61% | 30.67% |

For the single L8H11 edge 1 → 5, the base attention weights are 0.34851 (female) and 0.33074 (male). Replacing only its value content while keeping the base attention pattern gives:

| Base prefix | ΔM | Donor-effect fraction | P(queen) | P(king) |
| --- | --- | --- | --- | --- |
| female | -0.527036 | 20.90% | 4.37% | 2.21% |
| male | +0.930655 | 36.91% | 3.79% | 5.55% |

Joint counterfactual patches give the following fractions. These replace native activations with clean donor activations; they are localization oracles and do not establish independently additive paths.

| Patched quantity | Layers | Female → male | Male → female |
| --- | --- | --- | --- |
| pattern | 8,11,6 | -2.14% | 0.19% |
| pattern | all 12 | -0.73% | 2.64% |
| v | 8,11,6 | 51.50% | 82.85% |
| v | all 12 | 101.27% | 100.02% |
| attn_out | 8,11,6 | 52.53% | 79.41% |
| attn_out | all 12 | 100.00% | 100.00% |
| mlp_out | 8,11,6 | 1.92% | 0.89% |
| mlp_out | all 12 | 88.64% | 91.25% |

Swapping all attention patterns transfers under 3% of the contrast in either direction; swapping all values transfers approximately 100%. This supports a content-change account for this particular contrast. It does not show that routing is dispensable: both genders may use essentially the same necessary route. A large early MLP patch effect also does not by itself identify where royalty and gender are combined.

## Resolve the selected OV message into features

SAE Lens release: `gpt2-small-resid-post-v5-32k`, checkpoint `blocks.7.hook_resid_post`, OpenAI TopK with k=32 and 32,768 latents. Its layer-normalizing dictionary is applied directly at `blocks.8.ln1.hook_normalized`. This is a **transferred dictionary**, not an SAE independently trained at ln1. Feature IDs below belong to layer 7 of this release; they are not the IDs from earlier layer-5 ablations.

Write the normalized attention input as x_k = Σ_i a_ki D_i + b_k + e_k. Here a = SAE activation × the SAE normalization standard deviation, b includes the normalization mean and decoder bias, and e is the reconstruction residual. The implemented edit is

```text
Δz[q,h] = α A[h,q,k] Σ_(i in S) (a_donor[k,i] − a_base[k,i]) D_i W_V[h]
```

The model then applies W_O and the remaining network normally. The base attention pattern, input tokens and reconstruction residual are retained for feature-only edits. At α=1 this substitutes the donor feature coefficients only along one OV message. α>1 extrapolates beyond the donor and can imply negative effective coefficients: it is steering, not literal feature deletion. This is source-feature-resolved OV transport; no downstream Queen SAE feature or royal-query × gender-key conjunction is established.

Features were ranked using the female-prompt gradient times their male-minus-female message contribution, before testing individual interventions. Annotations were fetched after numerical selection and all causal tests. The four largest predicted donor-directed effects were:

| Feature | Automatic annotation (hypothesis) | Female activation | Male activation | Female → male ΔM | Male → female ΔM |
| --- | --- | --- | --- | --- | --- |
| [25975](https://www.neuronpedia.org/gpt2-small/7-res_post_32k-oai/25975) | references to male gender and masculinity | 0.0000 | 4.6695 | -0.2091 | +0.3445 |
| [20446](https://www.neuronpedia.org/gpt2-small/7-res_post_32k-oai/20446) | references to women in various contexts | 3.5555 | 0.0000 | -0.1616 | +0.2913 |
| [15560](https://www.neuronpedia.org/gpt2-small/7-res_post_32k-oai/15560) | references to specific individuals, particularly women involved in political or social contexts | 2.7406 | 0.0000 | -0.1187 | +0.2345 |
| [3281](https://www.neuronpedia.org/gpt2-small/7-res_post_32k-oai/3281) | references to men or discussions related to their health and societal roles | 0.0000 | 3.1025 | -0.0748 | +0.1028 |

The activations above are native normalized SAE coefficients z; interventions restore each token’s actual scale via a. Individual causal effects need not sum to the joint effect.

Main intervention outcomes:

| Base | Edit | P(queen) | P(king) | M | Donor-effect fraction |
| --- | --- | --- | --- | --- | --- |
| female | Baseline | 5.72% | 1.70% | +1.2102 | -0.00% |
| female | Whole value-content swap | 4.37% | 2.21% | +0.6832 | 20.90% |
| female | Four features, α=1 | 4.19% | 2.22% | +0.6361 | 22.77% |
| female | Four features, α=2 | 2.78% | 2.81% | -0.0110 | 48.44% |
| female | Four features, α=4 | 0.79% | 3.86% | -1.5888 | 111.01% |
| male | Baseline | 2.03% | 7.54% | -1.3111 | 0.00% |
| male | Whole value-content swap | 3.79% | 5.55% | -0.3805 | 36.91% |
| male | Four features, α=1 | 3.96% | 5.50% | -0.3293 | 38.94% |
| male | Four features, α=2 | 6.76% | 3.78% | +0.5824 | 75.10% |
| male | Four features, α=4 | 11.75% | 1.77% | +1.8929 | 127.07% |

At unit strength, the four features recover 108.9% (female), 105.5% (male) of the full message’s causal margin effect. Slight overshoot is consistent with the remaining content partly opposing these features; these ratios are not explained-variance measures.

Feature/bias/error accounting at unit strength:

| Component patched | Female → male ΔM | Male → female ΔM |
| --- | --- | --- |
| all_sae_features | -0.573603 | +0.991919 |
| bias_only | -0.000003 | +0.000000 |
| error_only | +0.044152 | -0.060371 |
| features_bias_error | -0.527038 | +0.930652 |
| full_content_patch | -0.527036 | +0.930655 |

## Reuse the frozen path and features

The same L8H11 source-to-final-query rule and four feature identities were applied without a royalty gate. The donor coefficients were measured in each matched prompt. These are small exploratory checks on hand-written templates, not a statistical benchmark. Most templates retain the same `A female/male` prefix and therefore the same causal source activations; the title template also changes that prefix.

Unit-strength four-feature donor-effect fractions:

| Female version of prefix | Measured pair | Female → male | Male → female |
| --- | --- | --- | --- |
| A female ruler is called a | queen / king | 23.45% | 38.41% |
| A female sovereign is called a | queen / king | 19.78% | 35.94% |
| The female monarch is called a | queen / king | 21.53% | 43.46% |
| A female monarch is known as a | queen / king | 19.07% | 41.15% |
| The title of a female monarch is | queen / king | 17.08% | 34.53% |
| A female parent is called a | mother / father | 37.52% | 40.56% |
| A female sibling is called a | sister / brother | 23.30% | 34.23% |
| A female child is called a | girl / boy | 24.11% | 41.51% |

The edit moves all five royalty paraphrases in the donor direction, but it also shifts mother/father, sister/brother and girl/boy. Thus the path has general gender relevance. These controls provide evidence against royalty-specific selectivity of the present ungated rule. They are separate prompts; they do not yet test preserving two downstream uses of the same source token within one sentence.

## QK check on the causal edge

The twelve largest absolute input-feature × input-feature score terms on the female L8H11 edge were each removed at unit strength, with the softmax and downstream network rerun. The largest absolute resulting margin change is 0.043952 logits, compared with 0.527036 logits for the whole content swap. These terms are not selected for gender/royalty labels; this is a limited check, not an exhaustive search over possible QK edits. See [all QK terms and outcomes](causal_edge_qk.json).

## Numerical checks and interpretation

- The ln1 transfer changes no active supports on either primary prefix. The largest native coefficient difference from encoding post-layer-7 residuals is 0.000143. This audit covers these two prefixes, not a broad corpus.
- Relative squared input reconstruction error is 3.80% (female), 3.86% (male). Reconstruction error is kept explicitly.
- Feature + bias + error reconstructs the donor-minus-base value vector to maximum absolute error 2.15e-06. Adding those components reproduces the independently localized full message effect.
- Self-patches are exact no-ops. Embedding-residual patches recover the donor. Joint pattern/value patches agree with attention-output patches at every layer and in both directions. All selected-edge feature edits leave every earlier query’s logits bitwise unchanged.

The earlier layer-5 result was a weak site for this contrast. A meaningful, interpretable OV effect appears at layer 8. This addresses the concern that no feature-resolved transport exists, but it does not demonstrate the special advantage sought in the synthetic conjunction experiments. We have not identified a causal royalty-query × gender-key cell, tested a shared source such as `queen` with multiple protected uses, or compared to token masks and source-feature edits at matched target effect. The next discriminating experiment needs those conditions, with the same allowed gating for each method.

## Reproduction and artifacts

From the repository root:

```bash
uv run --no-project --with sae-lens==6.46.1 --with transformer-lens==3.9.0 python experiments/fra_concept_trace_20260917/localize_gender_transport.py
uv run --no-project --with sae-lens==6.46.1 --with transformer-lens==3.9.0 python experiments/fra_concept_trace_20260917/resolve_gender_transport.py
python3 experiments/fra_concept_trace_20260917/report_gender_transport.py
```

[Protocol](PROTOCOL.md) · [Localization](localization.json) · [Frozen feature ranking](feature_ranking.json) · [Feature interventions](feature_results.json) · [Validation](validation.json) · [Joint patches](joint_patches.json) · [Localization manifest](manifest.json) · [Resolution manifest](resolution_manifest.json) · [Feature checks](feature_checks.json). The manifests retain package versions and model/SAE checkpoint hashes; label source URLs and hashes are in [path_feature_labels.json](path_feature_labels.json).

The first resolution attempt terminated during model loading without an explicit Python error. The completed attempt disabled global gradient tracking before loading, and enabled it only for the single ranking gradient. Its outputs and assertions completed successfully; no inference outcomes from the failed attempt enter this report.

Follow-up: [single-feature inversion comparison](single_feature_comparison/REPORT.md) tests individual feature substitution/deletion along this edge and at the source/all-position residuals, with strength sweeps against the four-feature edit.
