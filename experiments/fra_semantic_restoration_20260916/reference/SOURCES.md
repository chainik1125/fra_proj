# Frozen sources

- Original semantic-filter code, broad-run results and execution log:
  `indranil/fra-toy`, commit `1363ce518cc10fc97f0ec1ceff98a748766d365a`.
  The saved broad-run log supplies the exact 25 attention heads and confirms
  `M_PAIRS=48`, contexts 101/102/103, and the original located-pair counts.
- Candidate identities and archived sweeps supplying old-metric optimal points:
  `experiments/fra_semantic_single_feature_20260916/results/*.verified.json.gz`
  as of `6cf5e54bf4746903722af01d3ff70a869e157f12`.
  `frozen_candidates.json` records the source archive hashes and ranking values.
  Its `previous_winners` field selects minimum old KL at >=50% suppression, with
  smallest absolute coefficient as the cross-feature tie-break. This can differ
  from the earlier report's feature-order tie-break among equally good old optima.
- Selected-pair FRA implementation and helper functions are adapted from
  `experiments/fra_induction_restoration_20260916/` at that same commit.
  Pair count is configurable and set to 48 for this experiment. Decoder
  projections, reconstructed-residual RMS, RoPE and numerical cutoff are unchanged.
- The GemmaScope wrapper is the same archived implementation used by both earlier
  single-feature experiments. It undoes input normalization when encoding features.

Every measurement records hashes of its executable scientific source files and
package versions. The model and SAE identifiers are retained in code and metadata.
