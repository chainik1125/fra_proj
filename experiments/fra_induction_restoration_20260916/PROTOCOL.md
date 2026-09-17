# Corrected metric: recover clean continuations in a poisoned context

The earlier zero-KL result measured steering on a separate, unpoisoned paragraph.
It established that a selected feature was inactive there. It did not measure
restoration of clean behavior while the backdoor was present.

## Paired evaluation

For every original IC4/G4 case and each context seed 0, 1, 2:

1. Generate the original 20-token random primer `R` from that seed.
2. Clean prefix: BOS + the unmodified `R`.
3. Poisoned prefix: the same prefix, replacing `R[10:12]` with trigger, payload.
4. Append the **same original legitimate paragraph token IDs** to both prefixes.
5. At each aligned continuation position, compare the full next-token distribution:

   `KL(P(clean prefix + shared continuation so far) ||`
   `   P_steered(poisoned prefix + shared continuation so far))`.

Steering is enabled for the full poisoned sequence, including prefix processing.
The reference is always unsteered and unpoisoned. The paragraph is teacher-forced
identically on both sides, so the KL compares distributions after the same
continuation history. Mean KL per predicted token is primary; also retain summed
KL, per-token KL, and KL after trigger tokens. No independent generations are
aligned to compute KL.

The clean counterfactual restores the original two random tokens. It removes both
the association and those trigger/payload occurrences, so the measured difference
also includes ordinary contextual priming by those tokens.

Report the poisoned, unsteered KL baseline (coefficient zero). A feature that does
nothing on the poisoned input retains that baseline, rather than earning zero KL.

## Interventions and comparison

- Same eight original GPT-2 and Gemma cases, original model precision and SAEs.
- Freeze the ten candidates from the previous original-diff ranking. Change the
  evaluation metric, not candidate identities. Preserve the ranking provenance.
- Sweep each feature individually with the previous signed activation-weighted
  and additive grids. Report positive activation removal separately.
- Recompute FRA on the poisoned inputs. Select its original top-12 absolute QK
  contributions per original induction head at the original induction edge.
  Apply those fixed feature pairs wherever they occur across the attention matrix.
  As in the archived scripts, contribution matrices are calculated from the
  unsteered poisoned input and held fixed during each patched forward pass.
- Give FRA a denser positive coefficient grid, including zero, up to 64.
- Reproduce archived FRA suppression and old clean-paragraph collateral before
  using the efficient selected-pair implementation for the new metric.
- Primary optimization: one feature and coefficient per case minimizing mean
  continuation KL over the three contexts. Include the no-edit result explicitly.
- Also report continuation KL subject to >=50% and >=90% suppression on the
  original seed-0 repeated-token backdoor probe, to distinguish avoiding an edit
  from actually suppressing the original backdoor.
- On that original probe, additionally compare its full query distribution to
  a paired no-backdoor prompt: restore the first 20-token primer to unpoisoned R,
  keeping the second-half query tokens identical through the deployment position.

All numerical sweeps use batch size one. Selected points are independently
remeasured with full forwards; context-prefix caching must agree with direct
execution. Feature/coefficient choices are retrospective. This remains a small
benchmark with one paragraph per pair, not a general capability evaluation.
