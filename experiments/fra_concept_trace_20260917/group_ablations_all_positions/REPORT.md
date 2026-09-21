# Royalty and gender ablations at every sequence position

GPT-2 Small; the same layer-5 post-block OpenAI TopK SAE (32,768 features, k=32). The intervention is applied to **every position**, including supplied answer tokens during teacher forcing and generated tokens in each growing generation prefix. This does not mean every transformer layer.

## Observed outcome

Removing both groups changes the king continuation from `king's wife.` to `man's wife.`, the queen continuation from `queen's sister, who was sitting on the bed.` to `man who had been sitting in the chair.`, and the woman continuation from `woman's husband.` to `man.`. The man continuation stays `man's wife.`.

Royalty-only removal changes both king and queen continuations to `man who had been sitting in the chair.`. Gender-only removal does not remove the royal nouns in these copying-style prompts: king still starts with `king`, and queen with `queen`. In the female-monarch definition it changes the continuation from `"mother" in the English language.` to `"king" because he is the king of the land.`.

Across the six prompts, 13 of 18 interventions changed the greedy continuation; 11 changed at least one teacher-forced next-token argmax. These six examples are a diagnostic, not a general efficacy benchmark.

## Feature set fixed before group interventions

The available Neuronpedia layer-5 export labels 32,404/32,768 features; 364 lack labels. We searched the entire available dictionary for royalty and gender terms, manually excluded obvious homonyms and incidental pronouns, and retained the previously identified empirical female-associated candidate 18603. The resulting masks contain 34 royalty and 162 gender features, with a 196-feature union.

Royal titles are assigned by royalty descriptions; gender descriptions include male/female terms, gendered pronouns, honorifics and family roles. The sets are operational label groups, not an assertion that royalty features cannot also carry gender information. We include mixed and broad descriptions; these masks are not validated universal concept boundaries.

Only 9 royalty and 21 gender candidates are active anywhere in the six clean teacher-forced sentences; the full masks remain enabled during generation so other selected features are removed if they become active.

[Exact masks, descriptions, keyword patterns, exclusions and override](feature_groups.json). [All raw candidates](feature_groups_candidates.json). [Full available label catalogue and source hashes](dictionary_labels.json.gz). [Readable selected feature list](SELECTED_FEATURES.md). Labels were obtained from the [Neuronpedia dataset export](https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/?list-type=2&prefix=v1/gpt2-small/5-res_post_32k-oai/explanations/).

## Intervention and interpretation

At each token, encode the current residual as z, zero every coefficient in the selected group, and use `x_edited = x + decode(z_zeroed) - decode(z)`. This retains the original reconstruction error and uses the checkpoint's native per-token normalization. There is no TopK refill or iterative re-encoding clamp. The exact analytic edit is `-std(x) * (z_selected @ W_dec_selected)`.

The code sent to the decoder has zero selected coefficients. Re-encoding the edited residual is a different operation: selected features can reappear because encoder and decoder are not inverse bases. The records explicitly retain those re-encoded values. Other layers and unlabelled/other features can also retain or regenerate concept information, as can the preserved reconstruction error. Consequently, 'all' here means all features in the declared masks, not proven erasure of every representation of royalty or gender.

Teacher forcing keeps the sentence fixed. Its distributions are measured before each supplied next token. Separately, greedy decoding starts from the identical prefix ending immediately before the answer, excludes the supplied answer, and stops at the first sentence boundary or 24 tokens. The last emitted stop token need not itself be processed; every token used to predict another token passes through the hook.

## Probability of the supplied answer, before it is supplied

| Prompt | Measured answer | Unablated | Remove royalty | Remove gender | Remove both |
|---|---|---:|---:|---:|---:|
| king | ` king` | 16.707% | 0.471% | 17.023% | 0.513% |
| queen | ` queen` | 5.909% | 0.426% | 6.549% | 0.330% |
| man | ` man` | 14.489% | 14.489% | 7.128% | 7.128% |
| woman | ` woman` | 9.720% | 9.720% | 4.788% | 4.788% |
| definition_male | ` king` | 7.539% | 0.986% | 9.563% | 1.246% |
| definition_female | ` queen` | 5.716% | 1.301% | 1.637% | 0.401% |

## Full continuations and candidate probabilities

### king

Fixed teacher-forced text: `The king entered the room. The person who entered was the king.`

Common generation prefix: `The king entered the room. The person who entered was the`

| Condition | Greedy continuation | P(king) | P(queen) | P(man) | P(woman) |
|---|---|---:|---:|---:|---:|
| Unablated | ` king's wife.` | 16.707% | 1.432% | 3.620% | 1.368% |
| Remove royalty (34) | ` man who had been sitting in the chair.` | 0.471% | 0.108% | 5.817% | 2.563% |
| Remove gender (162) | ` king's wife.` | 17.023% | 1.468% | 3.414% | 1.338% |
| Remove both (196) | ` man's wife.` | 0.513% | 0.113% | 5.483% | 2.463% |

Per-position intervention audit on the fixed text. Shares are sums of selected positive SAE coefficients divided by total positive coefficient mass; they are not residual energy shares or causal importance. Zero-share positions were processed by the same mask but had no active selected coefficients.

| Position | Supplied token | Royalty mass removed | Gender mass removed | Union mass removed | Union: selected mass after re-encoding |
|---:|---|---:|---:|---:|---:|
| 0 | `The` | 0.00% | 0.00% | 0.00% | 0.0000 |
| 1 | ` king` | 39.79% | 0.00% | 39.79% | 0.0000 |
| 2 | ` entered` | 4.58% | 0.00% | 4.58% | 0.0000 |
| 3 | ` the` | 0.00% | 0.00% | 0.00% | 0.0000 |
| 4 | ` room` | 0.00% | 0.00% | 0.00% | 0.0000 |
| 5 | `.` | 0.00% | 0.00% | 0.00% | 0.0000 |
| 6 | ` The` | 0.00% | 0.00% | 0.00% | 0.0000 |
| 7 | ` person` | 0.00% | 7.72% | 7.72% | 0.8153 |
| 8 | ` who` | 0.00% | 0.00% | 0.00% | 0.0000 |
| 9 | ` entered` | 0.00% | 0.00% | 0.00% | 0.0000 |
| 10 | ` was` | 0.00% | 0.00% | 0.00% | 0.0000 |
| 11 | ` the` | 0.00% | 0.00% | 0.00% | 0.0000 |
| 12 | ` king` | 37.00% | 0.00% | 37.00% | 0.0000 |
| 13 | `.` | 3.62% | 0.00% | 3.62% | 0.0000 |

### queen

Fixed teacher-forced text: `The queen entered the room. The person who entered was the queen.`

Common generation prefix: `The queen entered the room. The person who entered was the`

| Condition | Greedy continuation | P(king) | P(queen) | P(man) | P(woman) |
|---|---|---:|---:|---:|---:|
| Unablated | ` queen's sister, who was sitting on the bed.` | 2.775% | 5.909% | 3.296% | 2.384% |
| Remove royalty (34) | ` man who had been sitting in the chair.` | 0.171% | 0.426% | 3.934% | 3.757% |
| Remove gender (162) | ` queen's daughter.` | 3.625% | 6.549% | 3.071% | 2.270% |
| Remove both (196) | ` man who had been sitting in the chair.` | 0.214% | 0.330% | 4.190% | 2.998% |

Per-position intervention audit on the fixed text. Shares are sums of selected positive SAE coefficients divided by total positive coefficient mass; they are not residual energy shares or causal importance. Zero-share positions were processed by the same mask but had no active selected coefficients.

| Position | Supplied token | Royalty mass removed | Gender mass removed | Union mass removed | Union: selected mass after re-encoding |
|---:|---|---:|---:|---:|---:|
| 0 | `The` | 0.00% | 0.00% | 0.00% | 0.0000 |
| 1 | ` queen` | 30.77% | 4.72% | 35.50% | 0.0000 |
| 2 | ` entered` | 2.63% | 0.99% | 3.62% | 0.0000 |
| 3 | ` the` | 0.00% | 0.00% | 0.00% | 0.0000 |
| 4 | ` room` | 0.00% | 0.00% | 0.00% | 0.0000 |
| 5 | `.` | 0.00% | 0.00% | 0.00% | 0.0000 |
| 6 | ` The` | 0.00% | 0.00% | 0.00% | 0.0000 |
| 7 | ` person` | 0.00% | 7.89% | 7.89% | 0.7832 |
| 8 | ` who` | 0.00% | 0.00% | 0.00% | 0.0000 |
| 9 | ` entered` | 0.00% | 0.00% | 0.00% | 0.0000 |
| 10 | ` was` | 0.00% | 0.00% | 0.00% | 0.0000 |
| 11 | ` the` | 0.00% | 0.00% | 0.00% | 0.0000 |
| 12 | ` queen` | 30.23% | 2.82% | 33.04% | 0.0000 |
| 13 | `.` | 2.68% | 0.00% | 2.68% | 0.0000 |

### man

Fixed teacher-forced text: `The man entered the room. The person who entered was the man.`

Common generation prefix: `The man entered the room. The person who entered was the`

| Condition | Greedy continuation | P(king) | P(queen) | P(man) | P(woman) |
|---|---|---:|---:|---:|---:|
| Unablated | ` man's wife.` | 0.074% | 0.014% | 14.489% | 4.495% |
| Remove royalty (34) | ` man's wife.` | 0.074% | 0.014% | 14.489% | 4.495% |
| Remove gender (162) | ` man's wife.` | 0.099% | 0.028% | 7.128% | 3.619% |
| Remove both (196) | ` man's wife.` | 0.099% | 0.028% | 7.128% | 3.619% |

Per-position intervention audit on the fixed text. Shares are sums of selected positive SAE coefficients divided by total positive coefficient mass; they are not residual energy shares or causal importance. Zero-share positions were processed by the same mask but had no active selected coefficients.

| Position | Supplied token | Royalty mass removed | Gender mass removed | Union mass removed | Union: selected mass after re-encoding |
|---:|---|---:|---:|---:|---:|
| 0 | `The` | 0.00% | 0.00% | 0.00% | 0.0000 |
| 1 | ` man` | 0.00% | 34.79% | 34.79% | 0.0000 |
| 2 | ` entered` | 0.00% | 0.00% | 0.00% | 0.0000 |
| 3 | ` the` | 0.00% | 0.00% | 0.00% | 0.0000 |
| 4 | ` room` | 0.00% | 0.00% | 0.00% | 0.0000 |
| 5 | `.` | 0.00% | 0.00% | 0.00% | 0.0000 |
| 6 | ` The` | 0.00% | 0.00% | 0.00% | 0.0000 |
| 7 | ` person` | 0.00% | 6.48% | 6.48% | 0.6767 |
| 8 | ` who` | 0.00% | 0.00% | 0.00% | 0.0000 |
| 9 | ` entered` | 0.00% | 0.00% | 0.00% | 0.0000 |
| 10 | ` was` | 0.00% | 0.00% | 0.00% | 0.0000 |
| 11 | ` the` | 0.00% | 0.00% | 0.00% | 0.0000 |
| 12 | ` man` | 0.00% | 30.95% | 30.95% | 0.0000 |
| 13 | `.` | 0.00% | 0.00% | 0.00% | 0.0000 |

### woman

Fixed teacher-forced text: `The woman entered the room. The person who entered was the woman.`

Common generation prefix: `The woman entered the room. The person who entered was the`

| Condition | Greedy continuation | P(king) | P(queen) | P(man) | P(woman) |
|---|---|---:|---:|---:|---:|
| Unablated | ` woman's husband.` | 0.047% | 0.025% | 9.265% | 9.720% |
| Remove royalty (34) | ` woman's husband.` | 0.047% | 0.025% | 9.265% | 9.720% |
| Remove gender (162) | ` man.` | 0.078% | 0.026% | 7.450% | 4.788% |
| Remove both (196) | ` man.` | 0.078% | 0.026% | 7.450% | 4.788% |

Per-position intervention audit on the fixed text. Shares are sums of selected positive SAE coefficients divided by total positive coefficient mass; they are not residual energy shares or causal importance. Zero-share positions were processed by the same mask but had no active selected coefficients.

| Position | Supplied token | Royalty mass removed | Gender mass removed | Union mass removed | Union: selected mass after re-encoding |
|---:|---|---:|---:|---:|---:|
| 0 | `The` | 0.00% | 0.00% | 0.00% | 0.0000 |
| 1 | ` woman` | 0.00% | 37.62% | 37.62% | 0.0000 |
| 2 | ` entered` | 0.00% | 0.00% | 0.00% | 0.0000 |
| 3 | ` the` | 0.00% | 0.00% | 0.00% | 0.0000 |
| 4 | ` room` | 0.00% | 0.00% | 0.00% | 0.0000 |
| 5 | `.` | 0.00% | 0.00% | 0.00% | 0.0000 |
| 6 | ` The` | 0.00% | 0.00% | 0.00% | 0.0000 |
| 7 | ` person` | 0.00% | 6.50% | 6.50% | 0.6608 |
| 8 | ` who` | 0.00% | 0.00% | 0.00% | 0.0000 |
| 9 | ` entered` | 0.00% | 0.00% | 0.00% | 0.0000 |
| 10 | ` was` | 0.00% | 0.00% | 0.00% | 0.0000 |
| 11 | ` the` | 0.00% | 0.00% | 0.00% | 0.0000 |
| 12 | ` woman` | 0.00% | 34.88% | 34.88% | 0.0000 |
| 13 | `.` | 0.00% | 1.89% | 1.89% | 0.0000 |

### definition_male

Fixed teacher-forced text: `A male monarch is called a king.`

Common generation prefix: `A male monarch is called a`

| Condition | Greedy continuation | P(king) | P(queen) | P(man) | P(woman) |
|---|---|---:|---:|---:|---:|
| Unablated | ` "kingdom" because he is the king of the land.` | 7.539% | 2.032% | 0.814% | 0.078% |
| Remove royalty (34) | ` "females" and a female is called a "females."` | 0.986% | 0.436% | 1.061% | 0.076% |
| Remove gender (162) | ` "king" because he is the king of the land.` | 9.563% | 1.320% | 0.315% | 0.018% |
| Remove both (196) | ` "fellow monarch" because he is a member of the family of the monarch.` | 1.246% | 0.350% | 0.368% | 0.017% |

Per-position intervention audit on the fixed text. Shares are sums of selected positive SAE coefficients divided by total positive coefficient mass; they are not residual energy shares or causal importance. Zero-share positions were processed by the same mask but had no active selected coefficients.

| Position | Supplied token | Royalty mass removed | Gender mass removed | Union mass removed | Union: selected mass after re-encoding |
|---:|---|---:|---:|---:|---:|
| 0 | `A` | 0.00% | 0.00% | 0.00% | 0.0000 |
| 1 | ` male` | 0.00% | 43.93% | 43.93% | 0.0000 |
| 2 | ` monarch` | 24.77% | 6.43% | 31.20% | 0.0000 |
| 3 | ` is` | 2.74% | 0.00% | 2.74% | 0.0000 |
| 4 | ` called` | 0.00% | 0.00% | 0.00% | 0.0000 |
| 5 | ` a` | 0.00% | 0.00% | 0.00% | 0.0000 |
| 6 | ` king` | 40.35% | 0.00% | 40.35% | 0.0000 |
| 7 | `.` | 4.28% | 0.00% | 4.28% | 0.0000 |

### definition_female

Fixed teacher-forced text: `A female monarch is called a queen.`

Common generation prefix: `A female monarch is called a`

| Condition | Greedy continuation | P(king) | P(queen) | P(man) | P(woman) |
|---|---|---:|---:|---:|---:|
| Unablated | ` "mother" in the English language.` | 1.704% | 5.716% | 0.165% | 0.303% |
| Remove royalty (34) | ` "female monarch" because she is the only female monarch in the world.` | 0.267% | 1.301% | 0.166% | 0.232% |
| Remove gender (162) | ` "king" because he is the king of the land.` | 5.284% | 1.637% | 0.230% | 0.030% |
| Remove both (196) | ` "fellow monarch" because it is a member of the family of the monarch.` | 0.755% | 0.401% | 0.236% | 0.020% |

Per-position intervention audit on the fixed text. Shares are sums of selected positive SAE coefficients divided by total positive coefficient mass; they are not residual energy shares or causal importance. Zero-share positions were processed by the same mask but had no active selected coefficients.

| Position | Supplied token | Royalty mass removed | Gender mass removed | Union mass removed | Union: selected mass after re-encoding |
|---:|---|---:|---:|---:|---:|
| 0 | `A` | 0.00% | 0.00% | 0.00% | 0.0000 |
| 1 | ` female` | 0.00% | 49.35% | 49.35% | 0.0000 |
| 2 | ` monarch` | 25.12% | 6.88% | 32.01% | 0.0000 |
| 3 | ` is` | 2.18% | 1.18% | 3.35% | 0.0000 |
| 4 | ` called` | 0.00% | 0.00% | 0.00% | 0.0000 |
| 5 | ` a` | 0.00% | 0.00% | 0.00% | 0.0000 |
| 6 | ` queen` | 30.69% | 2.83% | 33.52% | 0.0000 |
| 7 | `.` | 2.68% | 0.00% | 2.68% | 0.0000 |

## Teacher-forced next-token argmax changes

Each row remains conditioned on the original supplied prefix. Concatenating these predictions would not be an autoregressive output.

| Example | Condition | Position / input token | Unablated prediction | Edited prediction |
|---|---|---|---|---|
| king | Remove royalty (34) | 6: ` The` | ` king` | ` man` |
| king | Remove royalty (34) | 11: ` the` | ` king` | ` man` |
| king | Remove both (196) | 6: ` The` | ` king` | ` man` |
| king | Remove both (196) | 11: ` the` | ` king` | ` man` |
| queen | Remove royalty (34) | 1: ` queen` | ` of` | ` is` |
| queen | Remove royalty (34) | 6: ` The` | ` queen` | ` door` |
| queen | Remove royalty (34) | 11: ` the` | ` queen` | ` man` |
| queen | Remove gender (162) | 7: ` person` | ` who` | ` in` |
| queen | Remove both (196) | 1: ` queen` | ` of` | ` is` |
| queen | Remove both (196) | 6: ` The` | ` queen` | ` door` |
| queen | Remove both (196) | 7: ` person` | ` who` | ` in` |
| queen | Remove both (196) | 11: ` the` | ` queen` | ` man` |
| man | Remove gender (162) | 1: ` man` | ` who` | ` was` |
| man | Remove both (196) | 1: ` man` | ` who` | ` was` |
| woman | Remove gender (162) | 1: ` woman` | `,` | ` was` |
| woman | Remove gender (162) | 6: ` The` | ` man` | ` woman` |
| woman | Remove gender (162) | 11: ` the` | ` woman` | ` man` |
| woman | Remove both (196) | 1: ` woman` | `,` | ` was` |
| woman | Remove both (196) | 6: ` The` | ` man` | ` woman` |
| woman | Remove both (196) | 11: ` the` | ` woman` | ` man` |
| definition_female | Remove gender (162) | 1: ` female` | ` student` | ` of` |
| definition_female | Remove both (196) | 1: ` female` | ` student` | ` of` |

## Verification

All 24 example/condition runs completed. Model and SAE hashes match the preceding single-feature experiment. Baseline activations at every position and pre-answer probabilities reproduce the archived unsteered trace; all baseline generations match the previous experiment.

The empty mask is an exact logit no-op. The layer-5 input is exactly the clean input on the fixed teacher-forced tokens. Selected decoder coefficients are zero; native decode differences agree with the analytic edit. Prefix-only answer logits agree with full teacher-forced logits, ruling out influence from the supplied future answer. Adding each removed contribution back restores clean logits within floating-point tolerance. Generation audits confirm that the mask is applied to every position of each growing prefix.

Maximum native/analytic edit error: 5.01e-06. Maximum full/prefix logit difference: 1.67e-05. Maximum edit-and-rescue logit difference: 2.48e-05.

Re-encoding the edited residual yields positive selected mass in 8 condition/position pairs. These are recorded in `positions[].reencoded_selected_features` rather than treated as successful encoder-output clamping.

[Complete measurements](results.json), [numerical checks](checks.json), [manifest and checkpoint hashes](manifest.json), and [saved per-token edit vectors](all_position_deltas.pt). This experiment tests SAE group ablation, not an FRA QK/OV path intervention or an FRA advantage.
