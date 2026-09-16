# Single SAE feature comparison for semantic-filter transfer

Requested 2026-09-16: rank features by activation difference, sweep the top ten
individually, report the best single feature versus FRA.

## Fixed comparison

- Reference: Indranil's `indranil/fra-toy` commit
  `1363ce518cc10fc97f0ec1ceff98a748766d365a`, rung 5 broad run.
- Same base Gemma-2-2b, GemmaScope 65k SAE, FP16 model, SAE normalization,
  six concepts, three contexts, 70 retained queries (52 related-word, 18 planted).
- SAE at `blocks.6.hook_resid_pre`, corresponding to GemmaScope residual-post
  layer 5, matching Indranil's conventional SAE and DoM baseline site.
- Same 12 planted-word calibration seeds (1000–1011). Rank by the signed
  difference of mean SAE activations at the final query token.
- Primary contrast: planted prompt minus matching prompt without the payload.
  Sensitivity contrast: planted prompt minus unrelated text, as in the original.
- Top TEN features **per contrast**, each edited separately, never together.
  Report contrasts separately and the best across both explicitly.
- All-position steering, with no related-word list supplied to the intervention.
- Activation-weighted removal: `x -= c*z_f(x)*decoder[f]`.
- Additive direction steering: `x -= alpha*decoder[f]/norm(decoder[f])`.
  Both positive and negative strengths; zero included. Full grids in source/results.
- Same collateral: KL(clean || edited), summed over all tokens in the one
  original legitimate paragraph for each concept.

## Reporting and verification

- Best single feature/strength is selected retrospectively among the ten, as
  requested. This is an optimistic baseline comparison, not a frozen defense.
- Report reach and collateral at 30%, 50%, 70% suppression. Keep unreachable
  queries in the denominators; no silent dropping.
- Primary: minimum **measured** collateral achieving at least the threshold.
  This avoids pretending an overshooting grid point has matched suppression.
- Secondary: interpolation on consecutive strength intervals for each feature,
  including zero, compared with the original FRA curves with zero added.
  Never interpolate between different features; identify estimated results.
- Reproduce all clean payload probabilities and both original 12-feature SAE
  curves to validate the environment and feature normalization.
- Validate cached-prefix continuation and final-position-only output against
  ordinary full forward passes before the full sweep.
- Preserve all candidate identities, activation differences, grid points,
  per-query probabilities, collateral, package versions, and source hash.

The sleeper reference for activation-weighted SAE steering is
`experiments/multitrigger_sleeper/cloud/sae_steer_grid_pod.py`; additive
single-feature direction sweeps also occur in `single_feat_sweep_pod.py` and
`experiments/tinystories_sleeper/rerun4_rescue_2026-05-26/sweep_directional_candidates.py`.

## Numerical follow-up

The first sweep produced KL values near 4e-4 for many candidate features. This
was the FP16 batch-shape floor: clean logits used batch size one, while steered
logits used a batch of candidate interventions. All collateral points were
therefore remeasured at batch size one. For dormant candidates, the residual
tensor was exactly unchanged; separate full forwards checked that their logits
were also bitwise identical. Selected operating points for every feature curve
were checked using ordinary unbatched full forwards, with live SAE encoding.
Files ending in `.verified.json` are the final measurements; the original
`.json` files preserve the first sweep.

The primary report includes positive activation-weighted steering separately
from signed steering, and reports coefficient limits. The winner per concept
maximizes reach and then minimizes arithmetic-mean KL, which remains defined
when some collateral values are exactly zero.
