# Unsteered GPT-2 concept trace

Measured with SAE Lens 6.46.1. Teacher-forced continuation, no BOS, no semantic interventions.

[Open the token inspector](feature_trace.html). Select a prompt, SAE, and token; all active features are retained.

## Baseline before the supplied answer

| Source/prompt | P(king) | P(queen) | P(man) | P(woman) | Supplied continuation |
|---|---:|---:|---:|---:|---|
| king | 0.1671 | 0.0143 | 0.0362 | 0.0137 | ` king.` |
| queen | 0.0277 | 0.0591 | 0.0330 | 0.0238 | ` queen.` |
| man | 0.0007 | 0.0001 | 0.1449 | 0.0449 | ` man.` |
| woman | 0.0005 | 0.0003 | 0.0927 | 0.0972 | ` woman.` |
| definition_male | 0.0754 | 0.0203 | 0.0081 | 0.0008 | ` king.` |
| definition_female | 0.0170 | 0.0572 | 0.0017 | 0.0030 | ` queen.` |

## Explicit source-token example

Measured coefficients from the same layer-5, 32k SAE:

| Source word | Feature 18603, candidate female association | Feature 32492, candidate royalty association |
|---|---:|---:|
| king | 0.000 | 4.929 |
| queen | 2.974 | 1.971 |
| man | 0.000 | 0.000 |
| woman | 2.536 | 0.000 |

Feature 18603 was the strongest consistent female-minus-male candidate in the independent four-pair screen at this layer: mother 3.201, sister 2.984, aunt 3.259, girl 2.149; father/brother/uncle/boy all zero. Its automatic label is broader (people/proper names), so the gender interpretation remains a hypothesis. Feature 32492 has a royalty-related automatic label and separates the four source nouns here; it has not yet passed independent royal-role controls.

Both features are zero at position 11 (the final `the`, immediately before the answer) in all four matched prompts at layer 5. They reappear on the corresponding supplied answer nouns: royalty on king/queen, the female candidate on queen/woman. This is evidence of distinct source-token associations, not yet of their transport to the answer position.

## Every token in the king sentence, layer 5

Top three coefficients are shown for readability; the inspector and compressed data retain every active feature. Automatic descriptions appear on the first occurrence only. Feature links open the original dashboard.

| Position | Token | Region | Top features (coefficient; automatic description) |
|---|---|---|---|
| 0 | `The` | prompt | [4434](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/4434) (15.558; technical terminology related to computer systems and programming concepts); [24338](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/24338) (7.067; specific terms related to conflict and legal matters); [4327](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/4327) (6.630; terms related to internet security and online harassment) |
| 1 | ` king` | prompt | [18884](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/18884) (5.529; terms related to numerical values and comparisons); [23701](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/23701) (5.359; references to kingdoms and related hierarchical terms); [32492](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/32492) (4.929; references to royalty or the concept of "king" and "queen.") |
| 2 | ` entered` | prompt | [5117](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/5117) (7.819; occurrences of the verb "enter" and its variations, indicating actions of entering or beginning); [31323](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/31323) (5.655; commands or prompts related to entering information or data); [18884](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/18884) (5.182) |
| 3 | ` the` | prompt | [16290](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/16290) (8.182; the definite article "the" used frequently throughout the text); [7185](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/7185) (6.470; the definite article "the" in various contexts); [18884](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/18884) (5.373) |
| 4 | ` room` | prompt | [30953](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/30953) (9.279; mentions of different rooms, specifically focusing on room 9 and room 8); [4994](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/4994) (4.708; instances of the word 'room'); [18884](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/18884) (4.599) |
| 5 | `.` | prompt | [14934](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/14934) (10.431; sentences that contain assertions or conclusions about events or situations); [18884](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/18884) (4.789); [31489](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/31489) (4.496; sentences that express emotional or narrative closure) |
| 6 | ` The` | prompt | [16290](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/16290) (7.711); [7185](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/7185) (6.945); [10847](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/10847) (6.668; repeated use of the word "the" to signal emphasis or importance) |
| 7 | ` person` | prompt | [11810](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/11810) (13.515; references to "person" in various contexts); [27512](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/27512) (4.883; references to news and events, particularly concerning New York and significant political or social topics); [18884](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/18884) (3.719) |
| 8 | ` who` | prompt | [29242](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/29242) (10.733; instances of the word "who"); [15060](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/15060) (4.825; references to individuals or entities, particularly using the word "who."); [13507](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/13507) (4.393; the word "that" and its variations in multiple contexts) |
| 9 | ` entered` | prompt | [5117](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/5117) (5.806); [31323](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/31323) (5.170); [16974](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/16974) (3.769; references to physical entrances and exits) |
| 10 | ` was` | prompt | [6994](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/6994) (9.421; the verb "is" in various contexts throughout the document); [17106](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/17106) (4.914; elements related to identity or characteristics of individuals); [9950](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/9950) (4.591; instances of the verb "was" in varying forms) |
| 11 | ` the` | prompt / predicts answer | [16290](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/16290) (8.302); [7185](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/7185) (6.979); [16725](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/16725) (3.897; semantically relevant articles or stories detailing current events) |
| 12 | ` king` | supplied continuation | [32492](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/32492) (5.518); [23701](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/23701) (5.160); [27512](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/27512) (4.213) |
| 13 | `.` | supplied continuation | [14934](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/14934) (11.307); [31489](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/31489) (3.510); [12929](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/12929) (2.626; descriptive language related to sensory experiences and reactions) |

## Reconstruction audit

Each SAE is evaluated separately. This table uses all 20 short examples; it is a local calibration, not a general SAE benchmark. Lower error and KL are better. FVU uses a centered denominator across cached tokens excluding each sequence's first position. Registry values are retained in the JSON but are not substituted for measurements.

| Layer (zero based, post block) | Width | FVU excluding first token | Mean active | Mean KL (nats/token) |
|---|---:|---:|---:|---:|
| 0 | 32768 | 0.0278 | 32.0 | 0.0033 |
| 1 | 32768 | 0.0348 | 32.0 | 0.0092 |
| 2 | 32768 | 0.0536 | 31.9 | 0.0153 |
| 3 | 32768 | 0.0718 | 31.4 | 0.0255 |
| 4 | 32768 | 0.0887 | 31.7 | 0.0403 |
| 5 | 32768 | 0.0982 | 31.7 | 0.0404 |
| 5 | 131072 | 0.0762 | 31.8 | 0.0272 |
| 6 | 32768 | 0.1101 | 32.0 | 0.0534 |
| 7 | 32768 | 0.1234 | 32.0 | 0.0795 |
| 8 | 32768 | 0.1380 | 32.0 | 0.0719 |
| 8 | 131072 | 0.1102 | 32.0 | 0.0498 |
| 9 | 32768 | 0.1480 | 32.0 | 0.0772 |
| 10 | 32768 | 0.1462 | 32.0 | 0.0865 |
| 11 | 32768 | 0.1463 | 32.0 | 0.1248 |

## Reading the trace

Features at position t describe the prefix through token t. The next-token probabilities on that row predict token t+1. In particular, an active feature on a supplied queen token is not evidence that it caused queen to be predicted.

The supplied continuation is not counted as a model success. The pre-answer distribution is independently checked against a forward pass that has never seen that continuation. GPT-2's first-position activation outliers dominate pooled variance, so the main FVU table excludes each sequence's first token; all-token FVU and individual token errors remain available in the raw data.

All active features are saved, with exact token IDs, token offsets, hook names, SAE configs, checkpoint hashes, raw residuals, and per-token relative squared reconstruction errors. SAE encoding occurs offline on the unmodified cached activations. Separate reconstruction-only forwards measure distribution distortion; they do not generate the saved rollout.

Neuronpedia descriptions in the inspector are automatic labels, not established semantic identities. Independent gender contrasts use mother/father, sister/brother, aunt/uncle, and girl/boy at the source token. Candidate ranking excludes king/queen; their responses are reported afterwards. This small discovery set cannot establish a general gender concept or causal transport.

Feature indices belong to one checkpoint. Equal indices at different layers or widths are unrelated. Raw coefficient magnitudes should not be compared across SAEs. Post block L is the input residual to block L+1; these checkpoints alone do not separate attention from the MLP within a block.

No FRA path, semantic edit, or advantage over an attention mask has been demonstrated by these measurements.
