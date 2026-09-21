# Causal transport investigation

The main matched prefixes are `A female monarch is called a` and
`A male monarch is called a`, with no answer supplied. The outcome is the
queen-minus-king logit margin; absolute candidate probabilities are retained.

1. Replace each layer's Q, K, attention pattern, V, attention output, MLP output,
   and residuals with aligned counterfactual activations, both directions. Scan
   all positions jointly, the gender token, monarch, and the prediction position.
2. Select three layers by mean bidirectional whole-attention-output effect.
   Scan all heads, then the two strongest heads per layer and all causal edges.
3. At the strongest localized content path, decompose the counterfactual V
   difference into input-SAE feature contributions plus bias/error terms.
   Input features must be measured at ln1. A transferred normalized pretrained
   residual dictionary must pass a coefficient/support transfer audit.
4. Rank individual feature contributions by their first-order prediction of the
   counterfactual margin change, then perform actual feature-message patches and
   cumulative top-k patches. Retain all active features; do not select only
   appealing semantic descriptions. These selections are exploratory on the
   original prompt, not held-out evidence.
5. Reuse the selected layer/head/feature identities on paraphrases and distinguish
   success at changing the target from preserving royal status and other uses.

Counterfactual transport patches are localization oracles, not FRA results.
Normalized fractions compare with changing the actual input word. They may be
negative or exceed one, and fractions across components need not add to one.
No new model or SAE training is planned.
