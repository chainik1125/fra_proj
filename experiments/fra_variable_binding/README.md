# Synthetic variable binding with signed OV

Completed three-seed pilot: [results](out/pilot/RESULTS.md) and
[interactive figure](out/pilot/binding_pilot.html).

The subsequent **selective binding edit / QK compression** experiment has its own
[protocol](INTERFERENCE_PROTOCOL.md), [results](out/interference/RESULTS.md), and
[summary graph](out/interference/interference_summary.png). It compares FRA with
optimized Q, shared K, joint Q/K, single-feature, and Q/K DoM interventions on 15
trained heads. The predicted logit-space gap holds in an exact control, but there
is no lookup-accuracy advantage, and optimized steering substantially narrows the
readout gap. The experiment below remains the earlier signed-rebinding pilot.

This pilot separates variable binding from a fixed vocabulary-specific signed sum.
Each example supplies four variable/value records in random sequence order and
asks for `value(positive_variable) - value(negative_variable)`. Every filler can
appear with either sign. The supervised target is an exact vector, not a token
label that a downstream classifier could obtain through another mechanism.

The example `X = King, Y = Man; return X - Y` motivates the task. The experiment
uses 16 independent symbolic filler directions, rather than assuming that real
word embeddings satisfy an analogy.

## What is supplied and what is learned

Source activations contain an independently specified variable-ID feature and
one content feature. The query contains two role-specific variable-ID features.
A fixed random orthogonal decoder mixes these coordinates into residual space;
reconstruction in the known feature dictionary is exact. **This is not the
literal one-feature-per-token setting.** It isolates the use of already-present
binding features. Discovering bindings from separate variable/value tokens, and
recovering the features with an SAE, are subsequent experiments.

The model has two standard softmax attention heads with unconstrained learned
bilinear QK matrices and learned linear OV maps. There is no MLP or direct query
path to the predicted filler vector. Training minimizes mean squared error to
the signed target. The signs and binding-related QK coefficients are learned;
they are not fixed in the trained model. Full QK matrices are equivalent to
factorized Q/K maps with sufficiently large head dimension; low-rank learning
is not tested here. The output space is the ground-truth filler coordinate space.

The two-head choice makes addition and subtraction straightforward but is an
architectural assumption, not a conclusion about spontaneously discovered head
count. The pilot uses NumPy and a checked analytic gradient, on CPU only.

## Exact constructive solution

For N records with unique variable IDs, let the two heads score a matching
positive or negative variable by gamma and every other variable by zero.
Let `c = (exp(gamma) + N - 1) / (exp(gamma) - 1)`. The positive head's OV copies
the content with gain c and the negative head's OV copies it with gain -c.
Their common distractor weights cancel exactly, yielding

`output = content(positive_variable) - content(negative_variable)`.

This works at finite softmax temperature and for arbitrary filler vectors,
source order, and binding assignments. The proof assumes a fixed record count;
the gain c depends on N. It establishes existence, not that training must learn
this solution. The script checks this construction separately from training.

## Splits and controls specified before the main run

- Training sees even permutations assigning the sorted selected filler IDs to
  variables. Evaluation on odd permutations tests unseen binding configurations;
  all individual variable/filler combinations remain represented in training.
- Training excludes ordered query pairs `(i, (i+1) mod 4)`. Evaluation includes
  these held-out role combinations, separately and jointly with odd bindings.
- Source record order is randomized independently of binding configuration.
- A content-only model, trained on the same task with the binding tags removed,
  is a control. Given the unordered contents and a query, the conditional mean
  target is zero: averaging over the allowed even permutations leaves each
  variable's filler marginal uniform. Population-optimal normalized MSE is 1.
- We test source-order permutation invariance, query reversal, and swapping the
  two requested source binding tags while preserving their content vectors.
- We separately retain only binding-related QK terms or only content-related QK
  terms. The prediction is that binding terms will be necessary and largely
  sufficient for successful retrieval, under the supplied-feature assumption.

These are exploratory pilot specifications, not a registered study.

## Editing and fair baselines

The edit changes the positive role's variable from ID 0 to ID 1, only on queries
that request positive ID 0. Negative queries involving IDs 2 or 3 remain valid
distinct-variable queries. All other query rows are protected.

We compare:

1. A FRA edit replacing the QK coefficients from positive-query feature 0 to
   key binding-ID features with those of positive-query feature 1. This edits
   a set of feature pairs, not necessarily a single pair.
2. A single semantic Q-feature substitution, positive ID 0 to positive ID 1,
   using the same gate and the same query-side attention surface.
3. An explicit token-score edit that reproduces the FRA score perturbation.
   It must match FRA exactly and is not presented as a separate empirical win.
4. A fixed output DoM steer with the same query gate, estimated on an independent
   calibration set and with its scalar gain fitted there. This is a weaker
   baseline for context-dependent rebinding; superiority to it alone would not
   establish an FRA advantage.

Report actual target error and progress toward the counterfactual, plus change
to protected outputs. There is no interpolation between unreachable points and
no oracle attention mask relabeled as FRA. Protected rows are unchanged by
construction for all gated methods; zero damage on those rows is therefore a
scope check, not evidence of feature selectivity. Query-feature substitution is
the strong baseline. It is also a fixed linear Q-input steer along the paired
contrast `decoder(positive ID 1) - decoder(positive ID 0)`, changing two sparse
coordinates. We do not predict an FRA advantage over that baseline.

After the first run, we added collateral measurements on the negative operand
and all unchanged filler coordinates within the edited query. These are more
informative than the protected-query gate check and are labeled as post-run
additions in the results.

## Run and artifacts

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 experiments/fra_variable_binding/run_binding.py --out experiments/fra_variable_binding/out/pilot
```

Dependencies: NumPy and Plotly. The script checks its gradients and algebraic
identities, trains fixed-seed models, and writes metrics, model parameters, a
self-contained HTML figure, and a Markdown report.
No GPU, model downloads, or external evaluator are used.
The recorded dependency versions are in requirements.txt. The HTML figure's
bundled data and links were checked; visual browser inspection was unavailable
in this session.

Related primary research: Feng and Steinhardt,
[How do Language Models Bind Entities in Context?](https://arxiv.org/abs/2310.17191).
Their binding-ID mechanism motivates this interface; it does not establish our
synthetic result or prove uniqueness of a learned binding representation.
