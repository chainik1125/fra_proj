# Selective binding edits under QK compression

Protocol written before the main run, 10 September 2026.

This is a CPU experiment inspired by the supplied-activation interface in
[Assign and Add, Appendix C](https://arxiv.org/html/2605.31497v1#A3).
It studies assignment and lookup, with ground-truth features. It does not train
the paper's first layer, an SAE, or an arithmetic MLP.

## Task and models

Sixteen variables have distinct, randomly assigned values from 32 one-hot value
features. All sixteen query variables read the same memory. Source activations
contain a binding feature and a value feature; query activations contain a query
variable feature. OV copies the value features exactly. QK alone is trained.
The source value features can affect learned keys: their QK weights are learned
from random initialization, rather than fixed to zero. We vary QK width over
2, 4, 8, 12, 15; OV and residual capacity remain fixed.

Training excludes variable/value pairs with `(variable + value) % 4 == 0`.
Held-out memories consist entirely of these withheld pairs. All values and
variables are individually seen in training. Source order is irrelevant to the
implementation and is checked by permutation. Three seeds are used. Each query
is supervised to return the value assigned to its matching variable, through
cross-entropy on the attention-weighted one-hot values. No pair editing loss is
used during training. A scalar temperature is calibrated on separate IID
memories to give mean correct-binding probability 0.85, where attainable.

A separate exact control uses zero-mean equal-norm Fourier tight-frame keys,
without training. It checks the proposed query-edit collateral formula.

## Edit and baselines

For query variables 0, 4, 8, 12, select the highest-scoring incorrect binding in
the content-averaged binding score matrix. Reduce that query-feature / key-binding
coefficient by gamma=1. The pair is fixed across value assignments, not selected
from test answers. Values and OV remain fixed for every routing intervention.

The desired counterfactual preserves every other feature-pair score. Consequently
FRA has zero counterfactual error by construction. The question is the quantitative
cost of realizing this same selective change through more restricted interventions,
not whether FRA discovers an unknown desired edit.

Baselines receive the desired edit and can exploit true features and test keys:

- Best single query-feature direction, with an oracle choice among query IDs.
- Query DoM: the exact paired query contrast between the targeted query and the
  confused query. The strength is optimized, not taken from a default.
- Best single key-feature direction at the targeted source, including all binding
  and value feature directions; and a paired binding-feature key DoM.
- Arbitrary optimized query change on the affected query row.
- Arbitrary optimized changes to **all shared keys**, preserving queries.
- Joint arbitrary Q/K changes at the original head width, with multiple starts.
- An unrestricted map edit reproducing FRA; this is an equivalence control.
- No edit. Output DoM is not a primary comparator: a fixed value-space mean
  cannot represent changes in random context-dependent fillers. Query and key
  DoM are the relevant activation-site baselines here.

Q edits have the same query gate as FRA. K edits are shared across all queries;
giving each query a private copy of the keys would be a different intervention
class. Optimized Q/K baselines are context-specific oracles, more expressive than
a fixed steering vector. Their mathematical coefficient edits need not correspond
to a trained replacement network. Joint optimization is nonconvex; optimizer
residuals and an independent rank lower bound are reported.

## Two matching criteria, and outcomes

1. Match the target change in mean pairwise log odds. Remove each score row's
   constant offset before measuring error, respecting the softmax gauge. Q and
   K optima are solved analytically. Joint Q/K uses constrained alternating least
   squares, compared against the singular-value lower bound for rank-d matrices.
2. On a smaller declared subset (first two edited queries, first held-out memory
   per model), match the **actual target attention probability** and optimize
   attention-output MSE using constrained numerical optimization. This guards
   against falsely equating matched mean log odds with matched behavior.

Report both relative-logit distortion and attention/readout error against the
selective counterfactual. One-hot distinct values make attention output MSE exact;
it also equals expected output MSE for independent, unit-variance random scalar
fillers. Report actual lookup accuracy and error against the original lookup
task as well: counterfactual fidelity is not automatically improved task accuracy.
Changes to other query rows and within-row conditional preferences are distinct.

There is no predicted advantage over unrestricted map editing. With only one
active binding pair per edge on lookup examples, this experiment concerns
selectivity across uses of features, not resolving multiple active relationships
inside a single token edge. Arithmetic and multi-feature queries would be a
subsequent test.
