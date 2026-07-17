# EM-AFP HMM specification

This note defines the exact HMM used in the current EM-AFP experiments. It is
the process implemented in `bag_moments/em_afp.py`.

## Summary

The HMM has four closed components:

```text
MD, MO, AD, AO
```

These are the leaves of a two-factor product:

```text
persona: M vs A
domain:  D vs O
```

Each component contains `d = 5` internal states, so the hidden state space has
`4 * 5 = 20` states. The vocabulary also has `4 * 5 = 20` tokens. Tokens are
tagged content observations:

```text
x_{tag, content}
```

where `tag in {MD, MO, AD, AO}` and `content in {0, 1, 2, 3, 4}`.

The components are non-ergodic with respect to each other: hidden-state mass can
never move from `MD` into `MO`, `AD`, or `AO`. Only the observer's posterior over
components changes by Bayes updates.

## Parameters

The default parameters are:

```text
d = 5
seq_len = 20
pi = (0.25, 0.25, 0.25, 0.25)  # MD, MO, AD, AO
beta_persona = 0.5
beta_domain = 0.6
delta = 0.05
rho = 0.05
```

The component ordering is always:

```text
0: MD
1: MO
2: AD
3: AO
```

The hidden-state ordering is component-major:

```text
(MD,0), ..., (MD,4),
(MO,0), ..., (MO,4),
(AD,0), ..., (AD,4),
(AO,0), ..., (AO,4)
```

The token id for token `(tag, content)` is:

```text
token_id = tag_index * d + content
```

So the vocabulary is:

| token id | tag | content |
|---:|---|---:|
| 0 | MD | 0 |
| 1 | MD | 1 |
| 2 | MD | 2 |
| 3 | MD | 3 |
| 4 | MD | 4 |
| 5 | MO | 0 |
| 6 | MO | 1 |
| 7 | MO | 2 |
| 8 | MO | 3 |
| 9 | MO | 4 |
| 10 | AD | 0 |
| 11 | AD | 1 |
| 12 | AD | 2 |
| 13 | AD | 3 |
| 14 | AD | 4 |
| 15 | AO | 0 |
| 16 | AO | 1 |
| 17 | AO | 2 |
| 18 | AO | 3 |
| 19 | AO | 4 |

## Initial distribution

The initial distribution is uniform within each component and has component
mass `pi`.

For default `pi`, every hidden state has probability:

```text
P(component = c, inner_state = k) = 0.25 / 5 = 0.05
```

In general:

```text
b0[c, k] = pi[c] / d
```

## Hidden transition matrix

The hidden transition matrix is block diagonal. It never changes the component;
it only mixes the internal state within the current component.

Let `K` be the `d x d` within-component transition matrix:

```text
K = (1 - rho) I + rho * 11^T / d
```

With `d = 5` and `rho = 0.05`, this is:

```text
K =
[[0.96, 0.01, 0.01, 0.01, 0.01],
 [0.01, 0.96, 0.01, 0.01, 0.01],
 [0.01, 0.01, 0.96, 0.01, 0.01],
 [0.01, 0.01, 0.01, 0.96, 0.01],
 [0.01, 0.01, 0.01, 0.01, 0.96]]
```

The full `20 x 20` hidden transition matrix is:

```text
A = diag(K, K, K, K)
```

Equivalently:

```text
A[(component, j), (component', k)] =
    K[j, k] if component == component'
    0       otherwise
```

This is row-stochastic. Each component is internally ergodic because `K` has
positive off-diagonal entries, but the four components are mutually closed.

## Emission model

A hidden state emits a token in two independent parts:

1. a component tag, such as `MD` or `AO`;
2. a content index, such as `0` or `4`.

For hidden state `(true_component, inner_state)`, the emission probability is:

```text
P(tag, content | true_component, inner_state)
  = P(tag | true_component) * P(content | inner_state)
```

### Tag evidence matrix

The unnormalized tag evidence matrix is `W[true_component, observed_tag]`.

The rows and columns are both ordered:

```text
MD, MO, AD, AO
```

`W` factorizes into persona and domain evidence:

```text
W[(persona, domain), (persona', domain')]
  = persona_match(persona, persona') * domain_match(domain, domain')
```

where:

```text
persona_match = 1              if same persona
persona_match = beta_persona   if different persona

domain_match = 1               if same domain
domain_match = beta_domain     if different domain
```

With `beta_persona = 0.5` and `beta_domain = 0.6`:

```text
W =
          observed tag
          MD    MO    AD    AO
true MD [[1.0,  0.6,  0.5,  0.3],
true MO  [0.6,  1.0,  0.3,  0.5],
true AD  [0.5,  0.3,  1.0,  0.6],
true AO  [0.3,  0.5,  0.6,  1.0]]
```

The row sum is constant:

```text
Z = (1 + beta_persona) * (1 + beta_domain)
  = 1.5 * 1.6
  = 2.4
```

So the actual tag emission probabilities are `W / Z`:

```text
W / Z =
          observed tag
           MD      MO      AD      AO
true MD [[0.4167, 0.2500, 0.2083, 0.1250],
true MO  [0.2500, 0.4167, 0.1250, 0.2083],
true AD  [0.2083, 0.1250, 0.4167, 0.2500],
true AO  [0.1250, 0.2083, 0.2500, 0.4167]]
```

Smaller `beta` means stronger evidence against a mismatch. Since
`beta_persona = 0.5` and `beta_domain = 0.6`, persona mismatches are penalized
more strongly than domain mismatches.

For example, observing an `MD` tag is more compatible with hidden `MO` than
hidden `AD`:

```text
W[MO, MD] = 0.6
W[AD, MD] = 0.5
```

This is the intended local-vs-global asymmetry: `MD` and `MO` share persona,
while `MD` and `AD` share domain.

### Content emission matrix

The content emission matrix is `E[content, inner_state]`.

It is shared across all four components. With `d = 5` and `delta = 0.05`:

```text
E[content, state] =
    1 - delta       if content == state
    delta / (d - 1) otherwise
```

Numerically:

```text
E =
             hidden inner state
             0       1       2       3       4
content 0 [[0.9500, 0.0125, 0.0125, 0.0125, 0.0125],
content 1  [0.0125, 0.9500, 0.0125, 0.0125, 0.0125],
content 2  [0.0125, 0.0125, 0.9500, 0.0125, 0.0125],
content 3  [0.0125, 0.0125, 0.0125, 0.9500, 0.0125],
content 4  [0.0125, 0.0125, 0.0125, 0.0125, 0.9500]]
```

For each hidden inner state, the column sums to 1.

## Full emission matrix

The full emission matrix `B` has shape:

```text
20 hidden states x 20 tokens
```

For hidden state `(c, j)` and token `(tag, i)`:

```text
B[(c, j), token(tag, i)] = (W[c, tag] / Z) * E[i, j]
```

Each row of `B` sums to 1.

## Generative process

A sequence is generated as follows:

1. Sample initial hidden state `(c_0, j_0)` from `b0`.
2. At time `t`, emit token `(tag_t, content_t)` from `B[(c_t, j_t), :]`.
3. Transition to `(c_{t+1}, j_{t+1})` using `A`.
4. Repeat for `seq_len = 20` tokens.

Because `A` is block diagonal, the true component `c_t` is constant across the
whole sequence. The internal state `j_t` evolves according to `K`.

## Token operators

The filtering code is often easier to describe using token operators. For a
token `x = (tag, content)`, define the substochastic token operator:

```text
T_x[(c, j), (c', k)] =
    B[(c, j), x] * A[(c, j), (c', k)]
```

Given the block diagonal structure:

```text
T_(tag, content) =
    direct_sum_c [ (W[c, tag] / Z) * diag(E[content, :]) * K ]
```

The individual token operators are substochastic. Their sum over all tokens is
the hidden transition matrix:

```text
sum_x T_x = A = diag(K, K, K, K)
```

So the HMM as a whole is stochastic, while each observed-token operator is
substochastic.

## Exact filter

The exact filter tracks:

```text
belief[t, component, inner_state]
```

After observing token `(tag_t, content_t)`, the posterior is:

```text
post[c, j] proportional to
    prior[c, j] * (W[c, tag_t] / Z) * E[content_t, j]
```

Then the next prior is:

```text
next_prior[c, k] = sum_j post[c, j] * K[j, k]
```

The component posterior is:

```text
mu[c] = sum_j belief[c, j]
```

The within-component belief is:

```text
belief[c, :] / mu[c]
```

The `block_belief` measurement probes `belief[c, :]`, which sums to `mu[c]`.
The `within_component_belief` measurement probes `belief[c, :] / mu[c]`, which
sums to 1.

## Behavioral posterior inversion

For a model next-token distribution, we first marginalize over content to get
tag probabilities:

```text
p_tag[tag] = sum_content p(token(tag, content))
```

Under the HMM, the tag probabilities satisfy:

```text
p_tag = mu @ (W / Z)
```

So the component posterior estimate from logits is:

```text
mu_hat = (p_tag * Z) @ inverse(W)
```

The implementation clips negative numerical artifacts to zero and renormalizes.

This behavioral inversion only estimates the component posterior `mu`; it does
not recover the full `4 x d` belief state.

## Fine-tuning distribution

The base process uses the default uniform prior:

```text
pi = (0.25, 0.25, 0.25, 0.25)
```

The MD fine-tuning process initializes entirely inside the `MD` component:

```text
pi = (1.0, 0.0, 0.0, 0.0)
```

The internal state within `MD` is still initialized uniformly:

```text
P(inner_state = k | MD) = 1 / 5
```

During fine-tuning, the hidden component remains `MD` for every token because
the components are closed.

## Important consequences

1. The true hidden component never changes inside a generated sequence.
2. The observer's posterior over components changes because tags provide noisy
   evidence about the hidden component.
3. Component posterior geometry is a four-way Bayes inference problem over
   `MD/MO/AD/AO`.
4. Within-component state tracking is the same in all four components because
   the content channel and `K` are shared.
5. The current behavioral posterior measurements recover only `mu`, while the
   new belief-state probe measurements recover either `belief[c, :]` or
   `belief[c, :] / mu[c]`.
