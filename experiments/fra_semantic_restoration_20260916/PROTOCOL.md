# Paired continuation recovery for semantic filtering

Use the six Gemma-2-2b concepts and all 70 retained original queries: 52 related
words, 18 planted words, across contexts `101:4:6`, `102:1:12`, `103:9:2`.

## Inputs and KL

Construct the original filler + password sentence + filler for each context.
The poisoned prefix contains `The password is <trigger> <payload>.` The clean
reference contains `The password is <trigger>.`, the original no-payload control.
Append two newlines to each prefix and then exactly the same original legitimate
paragraph token IDs. Omit the recall question in this continuation evaluation.

Measure mean KL(clean reference || poisoned + steering) over the full next-token
distributions predicting the paragraph. Prediction indices are aligned by shared
continuation token, accounting for the different prefix lengths. Store both token
sequences, both index lists, and every per-token KL value. The payload omission
changes prefix length; this counterfactual includes that positional/context effect
and removes payload priming as well as the planted association.

The no-edit poisoned input is the baseline. Steering affects every poisoned-input
position, including the prefix. An inactive intervention retains the baseline KL.
Use batch size one, original FP16 model/SAEs, and float64 KL accumulation.

Also evaluate each original recall query with and without the payload in its
prefix. Record full next-token distribution KL, payload probability, and original
payload suppression. No queries are re-screened or dropped based on this rerun.

## Interventions

Freeze the original primary top ten SAE features, ranked by planted-minus-no-payload
mean activation over twelve planted-word calibration prompts. No synonym list is
used for candidate selection. Edit one feature at a time at residual-pre layer 6.
Sweep the original signed activation-weighted and additive grids, including zero;
report positive activation-weighted removal separately. Keep the SAE reconstruction
error in the residual; the intervention subtracts the selected decoder contribution.

FRA uses the exact 25 heads from the archived broad-run log. Locate the top 48
absolute QK contributions per head on the original seed-0 planted-word query→payload
edge. Apply those same feature pairs at all positions in every evaluation input.
Contribution matrices are computed on unsteered poisoned inputs and held fixed
during each patched forward, as in the original. Sweep a denser positive grid up
to 64. Reproduce the archived six-point FRA curves before the comparison.

## Selection and validation

Report unconstrained best continuation recovery, including no edit. Select one
feature and coefficient per concept by mean KL over its three continuations.
Also report one fixed edit per concept maximizing related-query coverage at each
30/50/70/90% suppression threshold, then minimizing that mean KL.

For comparison with the earlier report, provide retrospective per-query best edits
minimizing the corresponding context's continuation KL subject to suppression.
Keep unreached queries in denominators. Separately optimize full query KL to check
recovery of the related-word recall distribution. No interpolation is used.

Independently remeasure every selected SAE point with ordinary full forwards and
live SAE encoding on every input. Repeat selection if numerical differences change
the winners. Remeasure old-metric optima at 50% suppression without retuning on
the new metric: minimize archived independent-paragraph KL, breaking ties by the
smallest absolute coefficient across features. Because the original report used
feature order to break cross-feature ties, these need not be its displayed winners.
Preserve raw measurements, provenance hashes, versions, and reproduction errors.
These are retrospective selections on a small benchmark, not held-out performance.
