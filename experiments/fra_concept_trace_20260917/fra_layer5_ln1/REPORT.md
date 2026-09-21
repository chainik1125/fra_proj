# FRA at attention layer 5: female monarch

**Finding:** Royalty and gender candidates have nonzero cross-token QK terms and signed OV messages at attention layer 5. The tested feature paths have small effects on the final queen-versus-king prediction. This first layer does not establish a strong control path or an FRA advantage.

The main input is exactly `A female monarch is called a`, six tokens without BOS. The matched control substitutes `male`. No answer token is supplied to these forwards. Layers and positions are zero based.

## Hook and SAE correction

The prior layer-5 post-block SAE is downstream of attention layer 5. It cannot be reused as that attention layer’s input dictionary. Here, **input features are measured directly at `blocks.5.ln1.hook_normalized`**, using the normalized pretrained **layer-4 post-block dictionary transferred to that site**. Input feature IDs therefore differ from the previous layer-5 post-block IDs. The original layer-5 dictionary is used only as an output readout.

The SAE Lens 6.46.1 catalogue check found no directly trained ln1 release; the repository has a LocalLn1SAE wrapper but its example GPT-2 checkpoint is absent. This is **not an independently trained ln1 SAE**. Both the original SAE and the target model site normalize their input; the transfer was tested rather than assumed. [Catalogue audit](availability.json).

Across 20 saved examples / 239 token positions, direct ln1 encoding had **zero changed active feature sets** relative to encoding the original layer-4 residual. Maximum coefficient difference: 0.000213623. [Transfer and reconstruction audit](transfer_audit.json).

This check supports reuse of the dictionary’s feature identities on these examples. It is not a broad new SAE quality benchmark. Labels remain automatic hypotheses; a Queen-labelled feature is not assumed to encode female royalty exclusively.

## Which features are present before attention?

Five royalty and fifteen gender-labelled candidates are active somewhere in the two prefixes. They were selected from the layer-4 labels before path intervention outcomes, with mixed descriptions retained. The earlier layer-5 mask was not silently transferred across dictionaries. [Frozen masks and descriptions](feature_groups.json).

| Token | Input feature | Automatic label | Female-prefix activation | Male-prefix activation |
|---|---:|---|---:|---:|
| female/male | [28409](https://www.neuronpedia.org/gpt2-small/4-res_post_32k-oai/28409) | mentions of females in various contexts | 8.1192 | 7.1370 |
| female/male | [11431](https://www.neuronpedia.org/gpt2-small/4-res_post_32k-oai/11431) | references to female identity and gender | 6.7904 | 0.0000 |
| female/male | [18349](https://www.neuronpedia.org/gpt2-small/4-res_post_32k-oai/18349) | references to gender, particularly related to male and female distinctions | 5.1282 | 6.6900 |
| female/male | [5566](https://www.neuronpedia.org/gpt2-small/4-res_post_32k-oai/5566) | references to gender, particularly males | 0.0000 | 4.0280 |
| monarch | [2790](https://www.neuronpedia.org/gpt2-small/4-res_post_32k-oai/2790) | references to royalty, particularly the word "Queen" and its related contexts | 3.6027 | 3.3868 |
| monarch | [20633](https://www.neuronpedia.org/gpt2-small/4-res_post_32k-oai/20633) | occurrences of the word "Queen" in various contexts | 2.6730 | 2.5699 |
| monarch | [17001](https://www.neuronpedia.org/gpt2-small/4-res_post_32k-oai/17001) | terms related to royalty and royal family members | 2.9292 | 2.9421 |
| monarch | [21991](https://www.neuronpedia.org/gpt2-small/4-res_post_32k-oai/21991) | references to royalty and leadership positions | 2.1496 | 2.3786 |
| monarch | [804](https://www.neuronpedia.org/gpt2-small/4-res_post_32k-oai/804) | references to women's rights and issues related to gender equality | 2.6674 | 1.5261 |
| is | [923](https://www.neuronpedia.org/gpt2-small/4-res_post_32k-oai/923) | names and titles of historical or fictional monarchs | 1.4081 | 1.8528 |

Royalty / Queen-labelled features are already active at `monarch` before this attention block, in **both** female and male prompts. At the final `a`, none of the selected input royalty or gender candidates is active. Thus the measured concept–concept QK pairs occur upstream of the answer position; the final query may use other features to read those concepts.

## What is exact

At the actual post-LayerNorm input, write `x = sum_i a_i D_i + b + e`, where `a_i` includes the SAE decoder’s measured normalization scale, `b` its decoded bias/mean, and `e` the reconstruction error. Model LayerNorm is upstream of this representation and upstream of the interventions.

- **QK:** the feature pair term is `(a_qi D_i W_Q) · (a_kj D_j W_K) / sqrt(d_head)`. The full score includes feature×feature, feature×bias/error, bias/error×feature, and bias/error×bias/error terms, including model projection biases. All nine grouped terms are saved.
- **OV:** the feature message is `A[h,q,k] * a_ki D_i W_V[h] W_O[h]`. It is an exact linear decomposition with the observed attention pattern, plus bias and error messages. OV alone does not supply a second bilinear royalty×gender interaction.
- Signed OV projection onto the **post-block Queen feature’s preactivation** uses its clean observed normalization. The residual skip and MLP contributions are separate. This projection is not a final-logit attribution, and the TopK feature activation is nonlinear.
- Causal reruns recompute softmax and all downstream model layers and SAE output normalization. QK-only edits leave V intact. OV-only edits preserve the clean pattern. Combined edits subtract OV contributions using the **edited** pattern.

Maximum score reconstruction error: **1.41e-05**. Maximum attention-output reconstruction error: **9.78e-06**. The score-level four-corner mixed difference agrees with the corresponding FRA pair sum within **1.27e-06**. These identities include bias and SAE error; the selected semantic features alone are not a complete reconstruction.

## Explicit QK interactions: monarch query ← female key

For example, input Queen-labelled feature **20633 at monarch** and female-identity feature **11431 at female** contribute **+0.224506** to head 10’s score. But that head assigns only **0.478%** attention to this edge. Removing this single term changes the final queen-minus-king logit margin by only **+0.0000248**. A visibly nonzero score term is not sufficient evidence of behavioral control.

| Head | Sum of royalty-query × gender-key terms | Clean attention to female | Attention after removing that sum | Final margin change when only this head’s sum is removed |
|---:|---:|---:|---:|---:|
| 0 | -0.067773 | 0.2521% | 0.2697% | -0.000010 |
| 1 | -0.052167 | 0.0004% | 0.0004% | -0.000003 |
| 2 | +0.120091 | 31.2214% | 28.7024% | +0.003340 |
| 3 | -0.039311 | 14.6608% | 15.1595% | +0.000090 |
| 4 | -0.271277 | 10.3339% | 13.1314% | +0.000092 |
| 5 | +0.150069 | 0.1219% | 0.1049% | -0.000010 |
| 6 | +0.106197 | 8.0628% | 7.3099% | -0.000013 |
| 7 | -0.073588 | 1.7285% | 1.8580% | +0.000065 |
| 8 | -0.119084 | 0.9626% | 1.0830% | -0.000184 |
| 9 | -0.108938 | 2.5093% | 2.7901% | +0.000006 |
| 10 | +0.186127 | 0.4782% | 0.3973% | +0.000012 |
| 11 | +0.301157 | 4.1219% | 3.0831% | +0.002988 |

The all-cross-token condition removes both royalty-query × gender-key and gender-query × royalty-key terms wherever they are active, excluding self-position pairs. Broader “every term involving a group” controls also include interactions with other features, bias/error components, and self positions.

## Explicit OV messages

Gender feature **11431 at female → monarch through head 2** has a residual message norm of **0.633705**, and a signed **+0.002772** contribution to the clean Queen-feature preactivation at monarch. Removing that message changes the downstream queen-minus-king margin by **+0.006157**: its direct Queen-feature projection and downstream behavioral effect do not have the same sign.

A different path, gender/equality-labelled feature **804 at monarch → is through head 10**, supports the final queen-versus-king margin locally. Removing it changes that margin by **−0.009642**. These are measured but small effects, and the labels are provisional.

## Causal group and path interventions

Metric: `M = logit( queen) − logit( king) = log[P(queen)/P(king)]`. Positive delta means the edit relatively favors queen; probabilities also show changes to overall answer-word mass. Each condition begins with the unmodified prefix and uses unit-strength removal. No generated continuation is evaluated here.

### definition_female

| Intervention | P(queen) | P(king) | M | ΔM |
|---|---:|---:|---:|---:|
| Unmodified | 5.7158% | 1.7040% | +1.210249 | +0.000000 |
| Remove all cross-token royalty–gender QK pairs | 5.6680% | 1.6739% | +1.219699 | +0.009450 |
| Remove royalty-query × gender-key pairs | 5.6665% | 1.6730% | +1.219955 | +0.009706 |
| Remove every QK term involving a gender feature | 5.5394% | 1.5127% | +1.297971 | +0.087722 |
| Remove every QK term involving a royalty feature | 5.7604% | 1.6625% | +1.242704 | +0.032455 |
| Remove selected gender OV contributions, all paths | 5.9062% | 1.6868% | +1.253205 | +0.042956 |
| Remove selected royalty OV contributions, all paths | 5.5675% | 1.6269% | +1.230253 | +0.020004 |
| Remove both groups through OV, all paths | 5.7587% | 1.6085% | +1.275428 | +0.065179 |
| Remove cross-concept QK pairs: monarch ← gender token | 5.6702% | 1.6797% | +1.216632 | +0.006383 |
| Remove gender OV: gender token → monarch | 5.6682% | 1.6617% | +1.227045 | +0.016796 |
| Remove gender OV: gender token → final a | 5.8612% | 1.7501% | +1.208656 | -0.001593 |
| Remove royalty OV: monarch → final a | 5.7725% | 1.6952% | +1.225328 | +0.015079 |
| Cross-concept QK + gender OV: gender token → monarch | 5.6318% | 1.6467% | +1.229668 | +0.019419 |
| Cross-concept QK + gender OV, all paths | 5.8611% | 1.6658% | +1.258003 | +0.047754 |

### definition_male

| Intervention | P(queen) | P(king) | M | ΔM |
|---|---:|---:|---:|---:|
| Unmodified | 2.0318% | 7.5387% | -1.311116 | +0.000000 |
| Remove all cross-token royalty–gender QK pairs | 2.0232% | 7.4303% | -1.300891 | +0.010225 |
| Remove royalty-query × gender-key pairs | 2.0229% | 7.4289% | -1.300840 | +0.010276 |
| Remove every QK term involving a gender feature | 2.0170% | 7.0855% | -1.256467 | +0.054649 |
| Remove every QK term involving a royalty feature | 2.1503% | 7.3219% | -1.225275 | +0.085841 |
| Remove selected gender OV contributions, all paths | 2.0514% | 7.6062% | -1.310459 | +0.000657 |
| Remove selected royalty OV contributions, all paths | 2.0401% | 7.3665% | -1.283957 | +0.027159 |
| Remove both groups through OV, all paths | 2.0612% | 7.4306% | -1.282323 | +0.028793 |
| Remove cross-concept QK pairs: monarch ← gender token | 2.0265% | 7.4409% | -1.300672 | +0.010445 |
| Remove gender OV: gender token → monarch | 2.0480% | 7.4518% | -1.291575 | +0.019541 |
| Remove gender OV: gender token → final a | 2.0275% | 7.7403% | -1.339646 | -0.028530 |
| Remove royalty OV: monarch → final a | 2.0875% | 7.5423% | -1.284549 | +0.026567 |
| Cross-concept QK + gender OV: gender token → monarch | 2.0375% | 7.3783% | -1.286833 | +0.024283 |
| Cross-concept QK + gender OV, all paths | 2.0359% | 7.5107% | -1.305363 | +0.005754 |

## Crossed routing/content test

For the female→monarch edge, independently remove the royalty–gender QK term sum and the gender OV content. The output mixed difference is `M(QK+OV removed) − M(QK removed) − M(OV removed) + M(clean)`. This is a two-switch causal mixed difference, not an averaged Shapley interaction.

It is **-0.003760** logit units on the female prompt and **-0.005702** on the male control. Thus routing and transported content do interact at the output, but weakly in these tests. This does not establish that this layer implements the intended semantic composition.

## Output Queen-labelled feature and interpretation

Post-block feature 7671 at monarch is **3.0200** on the female prompt and **2.7937** on the male prompt. Its label therefore does not by itself certify the female-monarch conjunction.

For the female prompt, its clean preactivation decomposes as:

| Component | Contribution at monarch |
|---|---:|
| skip | +9.783870 |
| attention | -0.141796 |
| mlp | +0.463757 |
| SAE encoder / decoder bias term | -7.085797 |
| Total preactivation | +3.020032 |

The attention contribution is negative in this clean preactivation decomposition. Together with the pre-existing royalty features, this argues against treating attention layer 5 as a demonstrated site that creates a clean Queen concept from scratch. The large effect of prior post-layer-5 residual ablations could instead involve information used by later layers; this experiment does not localize that later use.

## Functional and reconstruction controls

| Prompt | Control | P(queen) | P(king) | ΔM | Final distribution KL(clean || edited) |
|---|---|---:|---:|---:|---:|
| definition_female | baseline | 5.7158% | 1.7040% | +0.000000 | 0 |
| definition_female | ln1_reconstruction_only | 5.7905% | 1.6194% | +0.063929 | 0.00200635 |
| definition_female | ln1_error_preserving_identity | 5.7157% | 1.7040% | -0.000008 | 2.10137e-08 |
| definition_female | whole_layer5_attention_output_zero | 4.6069% | 1.2119% | +0.125078 | 0.0104355 |
| definition_female | whole_female_to_monarch_message_zero | 5.5699% | 1.5929% | +0.041536 | 0.000408617 |
| definition_female | whole_monarch_to_female_score_mask | 5.5282% | 1.5703% | +0.048348 | 0.000603451 |
| definition_male | baseline | 2.0318% | 7.5387% | +0.000000 | 0 |
| definition_male | ln1_reconstruction_only | 1.8887% | 6.9709% | +0.005236 | 0.00320295 |
| definition_male | ln1_error_preserving_identity | 2.0318% | 7.5387% | -0.000007 | 1.02021e-07 |
| definition_male | whole_layer5_attention_output_zero | 1.6037% | 5.2113% | +0.132578 | 0.0183993 |
| definition_male | whole_female_to_monarch_message_zero | 2.0199% | 7.3032% | +0.025869 | 0.000452962 |
| definition_male | whole_monarch_to_female_score_mask | 2.0104% | 7.2195% | +0.032693 | 0.000703997 |

The source position is female in the main prompt and male in the matched control, including control identifiers containing “female”. Removing all attention-layer-5 output changes the distribution more than the selected paths, but still does not reverse the female prompt’s queen-over-king preference.

## Verification, scope and reproducibility

Completed **164** primary forwards, plus **12** functional/control forwards. Baseline probabilities reproduce the previous teacher-forced trace. Maximum score/content removal-and-rescue logit error: **1.62e-05**. Symmetric finite differences validate the leading QK and OV gradient attributions. Per-path ranking was exploratory on the female prompt; the same selected paths were also run on the male prompt. This is not a held-out selectivity test.

No model or SAE weights were trained. All paths use attention layer 5 only. Coefficient transfer validation does not establish general feature semantics. Signed attribution, score interaction, downstream interaction, and useful control are distinct claims; only the first three have small numerical witnesses here.

```bash
uv run --no-project --with sae-lens==6.46.1 --with transformer-lens==3.9.0 python prepare_ln1_transfer.py
# feature_groups.json freezes the reviewed candidates from feature_candidates.json
uv run --no-project --with sae-lens==6.46.1 --with transformer-lens==3.9.0 python measure_fra_ln1.py
uv run --no-project --with sae-lens==6.46.1 --with transformer-lens==3.9.0 python audit_fra_ln1_controls.py
python3 report_fra_ln1.py
```

[All causal results](results.json) · [Atomic QK/OV measurements](atoms.json.gz) · [Bias/error reconstruction ledgers and checks](audit.json) · [Functional controls](controls.json) · [Checkpoint manifest](manifest.json) · [Saved decomposition tensors](decomposition_tensors.pt)
