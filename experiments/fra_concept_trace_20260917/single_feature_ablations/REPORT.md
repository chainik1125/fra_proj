# Separate single-feature source ablations

**Observed result:** 5 ablations removed an active source feature; 0 conditions changed the greedy continuation and 0 changed any teacher-forced next-token argmax. The other ablations were inactive-feature controls. Probabilities did change under active ablations.

Removing feature 24973 from the source king increased the later king probability from 16.71% to 20.45%. Removing feature 7671 from source queen increased queen probability from 5.91% to 6.61%. Removing candidate 18603 from source queen gave 6.46% queen probability; from source woman it increased woman probability from 9.72% to 10.31%. These source ablations do not produce a clean semantic replacement.

GPT-2 Small; OpenAI TopK SAE, 32,768 features, k=32; layer 5 post-block residual. Each intervention independently removes one feature at source token position 1. No features are ablated jointly and no model weights are changed.

Feature 24973 is the King-specific candidate, 7671 the Queen-specific candidate, and 18603 the candidate female-associated feature. The latter is an empirical hypothesis from the four-pair discovery screen, not a validated universal gender representation.

## What is meant by the resulting sentence

Teacher forcing fixes the input sentence. For each ablation we report the next-token distributions on exactly that sentence, plus a separate greedy continuation from the common prefix ending immediately before the final answer noun. The supplied final noun is excluded from that generation prefix. Generation stops at the first sentence boundary or 24 tokens, whichever comes first.

## Intervention

For clean source residual x and code z, set only z_i to zero, then use `x_edited = x + decode(z_zeroed) - decode(z)`. The original SAE reconstruction error is retained. The decoder uses the clean token's native layer-normalization mean and scale. Other SAE coefficients are unchanged and the vacant TopK slot is not refilled. All downstream transformer layers run normally.

The implementation checks that the native decoder difference agrees with `-std(x) * z_i * W_dec[i]` for this checkpoint. Re-encoding the edited residual can produce a nonzero coefficient again because the SAE encoder and decoder are not inverse bases; the re-encoded value is recorded, not silently clamped with a second intervention.

## Measured continuations and pre-answer probabilities

### king

Teacher-forced sentence: `The king entered the room. The person who entered was the king.`

Common generation prefix: `The king entered the room. The person who entered was the`

| Condition | Source coefficient removed | Greedy continuation | P(king) | P(queen) | P(man) | P(woman) |
|---|---:|---|---:|---:|---:|---:|
| Unablated | 0.0000 | ` king's wife.` | 16.707% | 1.432% | 3.620% | 1.368% |
| Remove King feature 24973 | 4.0151 | ` king's wife.` | 20.445% | 1.613% | 3.206% | 1.170% |
| Remove Queen feature 7671 | 0.0000 | ` king's wife.` | 16.707% | 1.432% | 3.620% | 1.368% |
| Remove gender candidate 18603 | 0.0000 | ` king's wife.` | 16.707% | 1.432% | 3.620% | 1.368% |

Ablating a feature whose source coefficient is zero is an exact no-op control.

### queen

Teacher-forced sentence: `The queen entered the room. The person who entered was the queen.`

Common generation prefix: `The queen entered the room. The person who entered was the`

| Condition | Source coefficient removed | Greedy continuation | P(king) | P(queen) | P(man) | P(woman) |
|---|---:|---|---:|---:|---:|---:|
| Unablated | 0.0000 | ` queen's sister, who was sitting on the bed.` | 2.775% | 5.909% | 3.296% | 2.384% |
| Remove King feature 24973 | 0.0000 | ` queen's sister, who was sitting on the bed.` | 2.775% | 5.909% | 3.296% | 2.384% |
| Remove Queen feature 7671 | 6.1343 | ` queen's sister, who was sitting on the bed.` | 3.810% | 6.607% | 2.981% | 2.439% |
| Remove gender candidate 18603 | 2.9737 | ` queen's sister, who was sitting on the bed.` | 3.460% | 6.458% | 3.257% | 2.369% |

Ablating a feature whose source coefficient is zero is an exact no-op control.

### man

Teacher-forced sentence: `The man entered the room. The person who entered was the man.`

Common generation prefix: `The man entered the room. The person who entered was the`

| Condition | Source coefficient removed | Greedy continuation | P(king) | P(queen) | P(man) | P(woman) |
|---|---:|---|---:|---:|---:|---:|
| Unablated | 0.0000 | ` man's wife.` | 0.074% | 0.014% | 14.489% | 4.495% |
| Remove King feature 24973 | 0.0000 | ` man's wife.` | 0.074% | 0.014% | 14.489% | 4.495% |
| Remove Queen feature 7671 | 0.0000 | ` man's wife.` | 0.074% | 0.014% | 14.489% | 4.495% |
| Remove gender candidate 18603 | 0.0000 | ` man's wife.` | 0.074% | 0.014% | 14.489% | 4.495% |

Ablating a feature whose source coefficient is zero is an exact no-op control.

### woman

Teacher-forced sentence: `The woman entered the room. The person who entered was the woman.`

Common generation prefix: `The woman entered the room. The person who entered was the`

| Condition | Source coefficient removed | Greedy continuation | P(king) | P(queen) | P(man) | P(woman) |
|---|---:|---|---:|---:|---:|---:|
| Unablated | 0.0000 | ` woman's husband.` | 0.047% | 0.025% | 9.265% | 9.720% |
| Remove King feature 24973 | 0.0000 | ` woman's husband.` | 0.047% | 0.025% | 9.265% | 9.720% |
| Remove Queen feature 7671 | 0.0000 | ` woman's husband.` | 0.047% | 0.025% | 9.265% | 9.720% |
| Remove gender candidate 18603 | 2.5357 | ` woman's husband.` | 0.046% | 0.021% | 10.179% | 10.308% |

Ablating a feature whose source coefficient is zero is an exact no-op control.

### definition_male

Teacher-forced sentence: `A male monarch is called a king.`

Common generation prefix: `A male monarch is called a`

| Condition | Source coefficient removed | Greedy continuation | P(king) | P(queen) | P(man) | P(woman) |
|---|---:|---|---:|---:|---:|---:|
| Unablated | 0.0000 | ` "kingdom" because he is the king of the land.` | 7.539% | 2.032% | 0.814% | 0.078% |
| Remove King feature 24973 | 0.0000 | ` "kingdom" because he is the king of the land.` | 7.539% | 2.032% | 0.814% | 0.078% |
| Remove Queen feature 7671 | 0.0000 | ` "kingdom" because he is the king of the land.` | 7.539% | 2.032% | 0.814% | 0.078% |
| Remove gender candidate 18603 | 0.0000 | ` "kingdom" because he is the king of the land.` | 7.539% | 2.032% | 0.814% | 0.078% |

Ablating a feature whose source coefficient is zero is an exact no-op control.

### definition_female

Teacher-forced sentence: `A female monarch is called a queen.`

Common generation prefix: `A female monarch is called a`

| Condition | Source coefficient removed | Greedy continuation | P(king) | P(queen) | P(man) | P(woman) |
|---|---:|---|---:|---:|---:|---:|
| Unablated | 0.0000 | ` "mother" in the English language.` | 1.704% | 5.716% | 0.165% | 0.303% |
| Remove King feature 24973 | 0.0000 | ` "mother" in the English language.` | 1.704% | 5.716% | 0.165% | 0.303% |
| Remove Queen feature 7671 | 0.0000 | ` "mother" in the English language.` | 1.704% | 5.716% | 0.165% | 0.303% |
| Remove gender candidate 18603 | 2.4739 | ` "mother" in the English language.` | 2.069% | 4.649% | 0.228% | 0.274% |

Ablating a feature whose source coefficient is zero is an exact no-op control.

## Where teacher-forced top predictions changed

Positions below are input positions; the prediction is for the following token. All predictions remain conditioned on the original supplied prefix, not on preceding argmax predictions.

| Example | Ablation | Input position/token | Unablated next-token argmax | Ablated next-token argmax |
|---|---|---|---|---|
| All tested conditions | | | No argmax changes | |

## Numerical checks

- Unablated probabilities and source feature activations reproduce the saved baseline.
- Inactive-feature edits are exact no-ops, including their decoded continuations.
- Positions before the source have exactly unchanged logits.
- Full teacher-forced and answer-prefix-only forwards agree at the prediction site.
- Subtracting and restoring an active feature recovers the original logits within floating-point tolerance.

- The selected feature's re-encoded activation was exactly zero in every active ablation in this run.

Maximum prefix/full logit difference: 1.57e-05. Maximum native/analytic edit difference: 4.62e-06. Maximum rescue logit difference: 2.48e-05.

[Full measurements](results.json) include every teacher-forced token, probability changes, KL, generated text, and the selected feature's re-encoded coefficient after subtraction. [Manifest](manifest.json) records exact checkpoints, configuration and packages. [Checks](checks.json) retain the numerical validation results.

These are whole-residual, single-source feature ablations. They establish the effects of these candidate features under this intervention. They do not establish an FRA-specific QK/OV path or an advantage over another edit method.
