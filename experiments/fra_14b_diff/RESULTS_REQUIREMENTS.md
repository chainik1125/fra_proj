# GRID_RESULTS requirements (fra_14b_diff) — for the results-analyst

**LOUD CAVEAT (team-lead, must be prominent in GRID_RESULTS):**
BOTH models fell back to the §1a tercile split, NOT the strict align≤30 / >70
buckets. The "proper diff" ranking is therefore computed on terciles of a thin
α=0 sample:
- **finance**: only 24 coh>70 rollouts; strict misal-coh bucket <8 → tercile
  fallback (misal align≤40, align≥70; ~8/8 per bucket).
- **base**: 94 coh>70 but ~uniformly aligned (align 70-100, mean 90) → tercile
  fallback with NEAR-ZERO alignment spread (both terciles ≈90) → base diff
  ranking is weak-signal by construction (low statistical power; the control).

So the diff rankings are LOW-POWER, especially base. This was the right call
given available data (noise_study is the only per-rollout text+score source),
but ranking confidence must be judged accordingly.

**Required in the GRID_RESULTS table / text, per (model, protocol) cell:**
- bucket mode (`threshold` vs `tercile_fallback`) and the fallback reason
- |B_misal| and |B_align| (bucket sizes)
- score_spread + n_nonzero_scores (degeneracy check)

All of these are in each ranking JSON's `meta` (`meta.buckets.{bucket_mode,
n_misal, n_align, ...}`, `meta.score_spread`, `meta.n_nonzero_scores`) at
`qwen14b/grid_diff/<ranking>_<sae>_meta/diff_ranking_<model>_L24.json` (and the
routing variants under `frarouting_<recipe>_ln1_meta/`). Surface them; do not
bury them.
