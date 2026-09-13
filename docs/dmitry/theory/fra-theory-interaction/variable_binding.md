# Variable binding, QK selection, and signed OV composition

The binding question is whether a model can apply an operation to whichever
fillers currently occupy its argument roles. A fixed operation on word identities
is insufficient: in `X = King, Y = Man; return X - Y`, exchanging the bindings
must reverse the result, despite preserving the set of word identities.

The relevant primary precedent is Feng and Steinhardt's
[binding-ID mechanism](https://arxiv.org/abs/2310.17191): binding information is
represented by vectors associating entities with their attributes. Our pilot
supplies such features and tests their use. It does not reproduce their findings
in language models or discover binding IDs from raw text.

## A synthetic activation interface

There are N records. Record j contains binding ID j and arbitrary filler vector
v_j. A query supplies a positive variable p and a negative variable m, with p != m.
The output target is

$$y^*(p,m)=v_p-v_m.$$

The same filler can occur in either role. In the numerical pilot, fillers are
16 distinct basis vectors, N=4, and source activations each contain two known
features: binding ID and filler ID. Query activations contain the two requested
role-specific IDs. A fixed orthogonal decoder rotates these sparse coordinates
into residual space. The feature decomposition is exact and supplied.

Thus, the analyzed interface is deliberately beyond one feature per token. If
each source had only a content feature, with random record order and no binding
information anywhere, the requested binding would not be identifiable. If the
entire `(variable, filler)` combination were instead treated as one compound
feature, FRA would have only one feature pair per edge on each example.

## An exact solution with ordinary softmax

Choose gamma > 0. A positive head and a negative head have scores

$$s^+_j=\gamma\mathbf 1[j=p],\qquad
  s^-_j=\gamma\mathbf 1[j=m].$$

These scores are bilinear matches between a query role-ID feature and a key
binding-ID feature. In FRA they are distinct query/key feature pairs. They are
not a King-Man feature pair: content is carried by OV after the binding is selected.

Both heads have partition function Z=exp(gamma)+N-1. Their weights are

$$A^+_j=\frac{1+(e^\gamma-1)\mathbf 1[j=p]}{Z},\qquad
  A^-_j=\frac{1+(e^\gamma-1)\mathbf 1[j=m]}{Z}.$$

Let the projected values be c v_j in the positive head and -c v_j in the
negative head, where c=Z/(exp(gamma)-1). Then

$$y=c\sum_j(A^+_j-A^-_j)v_j=v_p-v_m.$$

This equality holds at finite temperature, for any filler vectors, any binding
assignment, and any source order, given unique binding IDs and fixed N. All
common distractor contributions cancel. The ordinary attention weights stay
nonnegative; the head-specific OV maps provide the signs.

This is a constructive existence theorem. It does not establish necessity of
this mechanism, sufficiency of arbitrary feature dictionaries, or spontaneous
learning in an unrestricted transformer. In particular, the OV gain depends on
the record count, and the construction assumes two available heads.

## An ideal rebinding guarantee

Changing the positive query role from p to p' while keeping the negative role
fixed gives

$$y'=v_{p'}-v_m,\qquad \Delta y=v_{p'}-v_p.$$

Hence any protected linear readout P satisfying P v_p = P v_{p'} has
P Delta y = 0. With independent filler directions, the negative operand's
coordinate is preserved when p, p', and m are distinct.

This is coordinated movement of attention between the two bound sources. It
is not arbitrary suppression of one source edge. The ideal score multiset and
partition function remain unchanged, which avoids the collateral rescaling
caused by an isolated edge-score suppression.

A Q-feature replacement, a corresponding change to QK role-to-binding pairs,
and a direct token-map edit that reproduces those weights all have the same
effect in this construction. A supplied feature basis identifies the semantic
rule across contexts; it does not enlarge the set of maps an oracle can edit.

## Numerical pilot, 10 September 2026

[Code and full protocol](../../../../experiments/fra_variable_binding/README.md),
[results](../../../../experiments/fra_variable_binding/out/pilot/RESULTS.md), and
[interactive figure](../../../../experiments/fra_variable_binding/out/pilot/binding_pilot.html).

Two softmax heads learn full bilinear QK matrices and linear OV maps from random
initialization, supervised only on the output vector. No MLP, SAE, learned
embedding, explicit QK-pair sparsity penalty, or direct query-to-output path is
used. The three seeds train for 1,600 steps on CPU. The exact construction above
is checked separately and is not used to initialize training.

Training uses even binding permutations and eight ordered query-variable pairs.
Evaluation uses odd binding permutations, four withheld query-variable pairs,
and their joint combination. Record order is independently randomized. All
individual variable and filler features are seen in training.

The three trained seeds identify both signed fillers correctly on all 4,096
joint-held-out examples per seed. Their normalized vector MSEs are 0.0326,
0.0230, and 0.0182, so correct filler identification does not mean exact vector
reconstruction on unseen queries.

Keeping only QK contributions to binding-ID features preserves 100% signed-pair
accuracy. Keeping only contributions to content features instead yields normalized
MSE approximately 1, compared with approximately 0.018-0.033 for the intact
models. A separately trained model without binding tags also has normalized
MSE approximately 1. These results show that the supplied binding information
is used through QK in this architecture.

For rebinding positive variable 0 to variable 1, the FRA pair edit and a direct
Q-feature replacement both give mean counterfactual NMSE 0.01333. A matched
token-score edit reproduces FRA to 2.2e-15 absolute output error. The Q-feature
replacement is also a fixed linear Q-input steering direction, so the pilot
does not show an advantage over that simple baseline.

Protected queries are unchanged because all methods use the same query gate;
this is a scope check. A more informative post-run measurement checks the
unchanged negative operand inside edited queries. Its mean absolute drift is
0.0617 under FRA, compared with a desired coefficient of -1. The learned models
therefore do not inherit the exact ideal preservation theorem automatically.

The next distinct question is whether training on raw sequences such as
`X, =, a; Y, =, b; query, X, -, Y` produces an intermediate representation with
the necessary binding features, and whether an independently trained SAE can
recover them. That experiment has not been run here.

## Selective binding edits under QK compression

A subsequent [CPU experiment](../../../../experiments/fra_variable_binding/INTERFERENCE_PROTOCOL.md)
tests pair-specific editing with 16 variables, fixed identity OV, and learned
QK widths 2, 4, 8, 12, 15 (three seeds each). Source activations contain both
binding and value features; both can enter the learned keys. All 15 heads
achieve 100% lookup accuracy on held-out variable/value combinations. This is
still a supplied-feature experiment, with no learned SAE or raw-token first layer.

The edit reduces one incorrect query-ID/key-ID coefficient while preserving
every other coefficient. In an exact tight-frame control, optimized Q steering
has normalized squared relative-logit distortion `(16-1)/d - 1`, while FRA
can implement the coefficient edit exactly. Full softmax-relevant width d=15
is a null control where arbitrary Q steering matches FRA.

However, matching the actual mistaken-binding probability and optimizing output
error substantially narrows the gap. At widths 4, 8, 12, best-found joint Q/K
steering has relative output MSE approximately 0.018, 0.040, 0.026 against the
selective counterfactual. At width 2, optimized Q is already within 3.5e-6:
softmax suppresses much of the logit-level distortion. FRA and the strong
baselines retain 100% lookup accuracy. Counterfactual fidelity therefore must
not be reported as an accuracy advantage. A post-run temperature sweep isolates
attention concentration as another relevant parameter.

See [results and baseline definitions](../../../../experiments/fra_variable_binding/out/interference/RESULTS.md)
and the [summary figure](../../../../experiments/fra_variable_binding/out/interference/interference_summary.png).
FRA's zero counterfactual error follows from defining the desired edit by feature
pair; this establishes intervention expressivity, not discovery or optimality.
An unrestricted map edit matches exactly. The lookup setting does not yet test
several active binding pairs sharing one token edge, or modular arithmetic.
