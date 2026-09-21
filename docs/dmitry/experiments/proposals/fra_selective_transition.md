# Proposed experiment: selectively correcting a temporal routing rule

Proposal, 17 September 2026. No training campaign has been run for this proposal.
The small population-objective calculation below is an algebraic/numerical
sanity check, not a learned-model result.

The hypothesis is that a feature-pair intervention can correct one relationship
while preserving other uses of both features. A useful test needs reused
features, changing memory contents, and an output loss sensitive to the routing
distribution. An interaction, temporal dependence, or off-diagonal QK matrix
alone does not establish an intervention advantage.

## What the existing notes imply

- [Reset-chain findings](../../../../experiments/chanin_attention/FINDINGS.md):
  temporal persistence produces meaningful OV transport, but recency can replace
  content routing. IID next-activation prediction admits the mean solution.
- [Shamir findings](../../../../experiments/chanin_attention/shamir/FINDINGS.md):
  multiple queried streams need query-dependent retrieval in that architecture,
  but diagonal stream matching suffices; the MLP performs secret reconstruction.
- [Interaction theory](../../theory/fra-theory-interaction/fra_theory_interaction.tex):
  a QK feature contribution is an exact fixed-background score mixed difference.
  An output effect additionally depends on values, competing scores and readout.
- [Binding results](../../../../experiments/fra_variable_binding/out/interference/RESULTS.md):
  compression creates score-level selectivity limits, but concentrated attention
  can make their output consequences tiny. Strong Q/K baselines matter.

The proposed change is to make accurate *mixture prediction* the task. This gives
moderate attention weights a real statistical meaning instead of calibrating a
lookup head away from the optimum of its original task.

## Minimal temporal process

Each episode emits three memory records, a query, then a target activation.
The records have labels B, D and R (reference) and independent payload vectors
v_B, v_D, v_R. The query is A or C, with equal probability. Record order is
random. Evaluate both queries against the same memory.

Use known, orthogonal feature directions, randomly rotated into activation
space. Keep label features and payload features in separate subspaces. Initially
QK reads label features and OV copies payloads exactly. The query has no direct
path to the predicted payload. Use one attention reader, no MLP or layer norm,
and no positional score in the first analytic control. QK width and value width
are separate experimental parameters, as in the earlier binding compression
pilot; reducing QK width must not reduce payload transport capacity.

The target is a stochastic copy: select one of the three records with the
probabilities below, and emit its payload. Equivalently, train against the known
conditional mean to remove sampling noise, then evaluate actual stochastic
next-activation loss as well. This is a conditional temporal process with known
Bayes predictions; it is not a reproduction of the independent reset HMM.

| Query | Old P(B), P(D), P(R) | Corrected P(B), P(D), P(R) |
|---|---|---|
| A | 3/7, 3/7, 1/7 | 1/5, 3/5, 1/5 |
| C | 3/7, 3/7, 1/7 | 3/7, 3/7, 1/7 |

The intended change is: remove B's extra eligibility under request A. Preserve
the D:R odds for A and the entire transition law for C. The desired distribution
is specified by the environment, independently of the intervention algorithm.
Preserving a relationship means preserving its relative score, not keeping all
absolute probabilities unchanged after softmax renormalization.

This is an editing/adaptation experiment: learn the old transition law, freeze
the model, then repair it to predict data from the corrected law. The initial
control supplies the weights exactly. A learned arm trains from random
initialization on next-activation targets, without pair-sparsity supervision.

The symmetric example is the smallest proof case. It does not require the old
model to distinguish A from C or B from D. Therefore the subsequent training
experiment must also use unequal row/column factors, so both query and memory
labels already affect routing before the correction.

## Exact mechanism and scope of the advantage

Define B as the matrix of source-versus-reference log odds, with query rows A,C
and memory columns B,D. For gamma=log(3),

```
B_old       = gamma [[1, 1],
                     [1, 1]]

B_corrected = gamma [[0, 1],
                     [1, 1]].
```

Old QK has rank one. Subtracting the FRA contribution
`gamma * activation(A at query) * activation(B at source)` gives the corrected
law exactly. It retains A-to-D, C-to-B and C-to-D score contributions.

With one-dimensional QK, even arbitrary optimized queries and shared keys give
`B_ij = q_i * (k_j-k_R)`, a rank-at-most-one matrix. The corrected matrix has
determinant `-gamma^2`, hence rank two. Since the reference log odds uniquely
determine a strictly positive attention distribution, no such Q/K edit can
realize the corrected law for both queries. Subtracting the reference score
handles softmax's row-offset invariance explicitly.

This comparison holds OV fixed. Queries can be changed separately, but keys
must remain shared across queries of the same memory. Giving each query its own
arbitrarily edited keys makes a more expressive intervention, which can match
the desired map. The basic proof excludes independent additive score adapters
and positional score biases; those must be separate controls, not silently
included in a claimed rank-one bound.

This is explicitly a capacity-dependent intervention advantage. FRA edits
after QK factorization and can raise effective score rank. At QK width two,
unrestricted Q/K edits can match. An extra Q/K coordinate, a rank-one bilinear
score adapter with the correct features, and an unrestricted attention-map edit
also match. These are mandatory equivalence controls. A positive result does not
show that FRA has more expressive power than those intervention classes.

## Why the score difference is behaviorally relevant

Draw the payloads independently, with zero mean and E||v_j||^2=1, independently
of their labels. For a label-only routing distribution p, the conditional mean
is `m = sum_j p_j v_j`. If p* is the corrected transition law, then

```
E ||m - m*||^2 = sum_j (p_j-p*_j)^2.
```

This is the exact *excess next-activation MSE above Bayes risk*, averaged over
fresh payloads. It is not a score-distance proxy or an error against an output
defined by running FRA. The identity assumes that routing depends on labels,
not the independently drawn payloads. If that assumption is relaxed, evaluate
the actual prediction error rather than applying this identity unmodified.

All three source weights are appreciable. Predicting only the most likely
source is suboptimal under squared error. Random payloads also make the desired
output correction vary with the memory: a fixed output direction cannot simply
erase a persistent conjunction-output atom. For this zero-mean construction,
the population output DoM is zero; it is consequently only a weak diagnostic
baseline, not the primary competitor.

## Small population-objective sanity check

I optimized the exact average squared probability error for both queries,
jointly over Q and shared K, with SciPy BFGS and analytic gradients. Forty random
starts used RNG seed 13. These are best-found numerical optima, not certified
global minima. The rank argument proves exact impossibility at width one,
independently of the numerical optimizer.

| Method, symmetric example | Excess next-activation MSE |
|---|---:|
| Old model, no correction | 0.04244898 |
| Joint optimized Q/K, width 1 | 0.01496192 |
| Exact FRA pair correction | 0 |
| Joint optimized Q/K, width 2 | < 1e-25 |

All 40 starts reached the reported width-one value within 1e-8. The optimum
compromises: query-A error is 0.00886062 and protected query-C error is 0.02106322.
This is evidence that the construction merits a learned experiment, not evidence
that SGD or a recovered SAE has already produced the result.

An unequal-factor check uses
`B_old = log(3)/2 * [[1,2],[2,4]]`, again zeroing only entry A,B. Both query rows
and key columns now have distinct effects. Unedited excess MSE is 0.00848136;
best-found width-one joint Q/K error is 0.00281452; width two again matches to
numerical precision. All 40 starts agree within 1e-8.

## Learned experiment and decisive controls

1. Reproduce the exact two-query construction and its unequal-factor version.
   Train several seeds on the old process and check conditional-mean prediction,
   not just which source has maximal weight.
2. Compare FRA, best single Q/K feature edits, Q/K DoM, arbitrary optimized Q,
   arbitrary optimized shared K, and joint optimized Q/K at fixed width. Fit
   intervention strengths on calibration memories and use fresh test payloads.
   Give every method the same target rule and calibration labels. Include direct
   map editing and a rank-one score adapter as ties. Add V/output interventions
   as separate arms; the fixed-OV rank proof does not cover them.
3. Plot correction of the changed transition against damage on protected
   predictions, using actual achieved points at matched correction. Also report
   total excess task risk; optimizing total risk alone can hide a tradeoff by
   leaving the target partly uncorrected.
4. Sweep QK width and source-probability concentration separately. Width two is
   the exact null here. Increasing concentration can reduce behavioral gaps, as
   the prior binding experiment showed; report any such disappearance.
5. Allow multiple independently active label features per query and record,
   with scores `z_q^T B z_k`. Use known transition probabilities to define the
   targets. Now multiple relationships can share a token edge. Enumerate small
   feature coalitions, including singly active and coactive cases. Retain all
   clean probability changes induced by renormalization in the target.
6. Only after the known-feature result is established, replace the oracle basis
   with a separately trained SAE. Measure feature recovery, score reconstruction
   and intervention success independently. Existing Chanin recovery results do
   not certify recovery on this new input distribution.

For the interaction measurement, hold all other coordinates fixed and measure
`s11-s10-s01+s00` for independently inserted query/source features. It must equal
the selected FRA contribution. Measure the output mixed difference separately;
do not assume multiple score interactions add at the output. This follows the
convention in the local theory note, not an unspecified full SHAP interaction
matrix. The [Shapley-Taylor paper](https://proceedings.mlr.press/v119/sundararajan20a.html)
distinguishes interaction allocation conventions; use an explicit convention if
adding an averaged Shapley interaction measurement.

The learned result worth pursuing is lower *actual task error and collateral*
at matched transition correction, persisting across memories and feature
compositions. Failure against optimized Q/K, failure after SAE recovery, and
ties at sufficient width are substantive outcomes, not implementation failures
to be removed from the report.

## Connection back to Shamir

The proposed first experiment isolates routing while keeping OV transport
transparent. A later Shamir extension could let the value of an earlier share
determine which later record is relevant, then selectively correct one such
conditional retrieval rule. A second attention stage or recurrence would be
needed explicitly; a fixed verified modular-arithmetic decoder would keep
routing changes separate from arithmetic learning. Simply adding more streams
or renaming query and key feature IDs would not supply the desired mechanism.

I would run the routing experiment first. Its restricted setting lets us explain
both a win and the controls in which the win must disappear, before introducing
nonlinear reconstruction and learned feature recovery.
