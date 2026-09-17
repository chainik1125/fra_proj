# Original diff-only baseline

The main experiment searches the union of two calibration rankings: the top ten
poisoned-minus-clean SAE activations and the top ten factorial interaction
activations at each layer. Its winner, L21 feature 3994, belongs only to the
interaction ranking. Therefore we also explicitly report the original requested
ranking on its own.

`reference/diff_only_selection.json` restricts the **existing tuning grid** to the
top ten diff-ranked features per layer. It uses the same objective, thresholds and
positive/signed variants. The two resulting unique settings are frozen before
running `evaluate_diff_only.py`. No test metric is used in their choice; the GPU
script independently checks this by recomputing the choice from archived tuning.
No candidate, coefficient or test case is added or removed. All 48 test cases are
reused to evaluate this prespecified baseline family.

This supplement was run after inspecting the main results to ensure the original
diff-only request is reported separately from the stronger union baseline. Any
non-finite held-out output would be retained and invalidate that full setting;
it would not be used to choose a replacement. The main experiment remains unchanged.
