# Original induction backdoors: single-feature SAE baseline

Requested 2026-09-16 after checking that the semantic-transfer experiment covered
different prompts. This experiment compares the original IC4 (GPT-2) and G4
(Gemma-2-2b, 65k GemmaScope) tasks against the requested best-of-ten baseline.

## Frozen setup

- Use the exact IC4/G4 trigger–payload pairs, repeated random-token prompts,
  evaluation seed 0, and one legitimate paragraph per pair.
- GPT-2: bank→river, fire→water, king→crown, doctor→patient.
- Gemma: bank→river, king→crown, doctor→water, market→gold.
- Layer 6 residual-pre, original SAE releases, original model precision.
- Rank by signed ON-minus-OFF mean SAE activation at query position 31 using
  the original calibration: 14 prompt pairs for GPT-2, 12 for Gemma. ON seeds
  start at 1000; unrelated OFF seeds start at 5000.
- Sweep the top ten features separately. Preserve rank and activation differences.
- Activation-weighted removal: `x -= c*z_f(x)*W_dec[f]`, all positions.
- Additive steering: `x -= alpha*W_dec[f]/norm(W_dec[f])`, all positions.
- Signed grids include zero; activation magnitude up to 64 and additive up to
  128. Report positive activation steering separately. Full grids are in source.
- Repeat the original simultaneous top-12 baseline as a reproduction check.
- Compare with saved original FRA curves, clearly identified as archived data.

## Measurement and verification

- Suppression is `1 - P_edited(payload)/P_clean(payload)` at query position 31.
- Collateral is summed `KL(clean || edited)` over every token of the same
  original legitimate paragraph, in nats.
- All sweep points use batch size one. Cache only the prefix before layer 6.
- Compare cached continuation with ordinary full forwards before sweeping.
- Directly remeasure the best point for each feature at 30%, 50%, 70%, 90%, and
  99% suppression, both on the target and legitimate paragraph; iterate if
  remeasurement changes the winner. Also directly remeasure top-12 points.
- Primary comparison: smallest measured collateral at or above each threshold.
  Keep unreachable cases visible. A secondary interpolated comparison may use
  adjacent points within one feature curve, never points from different features.
- Feature/strength selection is retrospective on the original test case.
  This is an optimistic baseline, not a validated deployment defense. One
  paragraph does not measure general capability preservation.

## Search for an earlier result

Searched current files, local archived run logs, all fetched Git histories and
the `dmanningcoe/fra-phase1-steering-data` Hugging Face `fra_win` archive. Found
IC4/G4 grouped top-12 SAE comparisons; ActAdd residual-direction comparisons;
single-feature sweeps for trained sleeper models; and separate acronym/retrieval
steering runs. No completed top-ten individual-feature sweep on the original
IC4/G4 backdoors was found. The remote IC4/G4 scripts match the repository copies
byte for byte. Source URLs and SHA-256 hashes are in `reference/archive_sources.json`.
