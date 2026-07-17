# MATS poster — source-doc index

Index of the docs behind the end-of-project poster **"Emergent Misalignment In a
Two-Layer Transformer"** (`main_36x24.pdf`, built 2026-03-24).

- **Poster build dir:** `/Users/dmitrymanning-coe/Documents/Research/Simplex/RL_beliefs/TeX/mats_poster/`
- **Provenance map (this folder):** [`poster_results_provenance.md`](poster_results_provenance.md) — box-by-box trace of every poster figure/table back to its doc, code, and run.

All source docs have been **copied into [`writeups/`](writeups/)** (296K, organized by
poster box) so they survive even if the worktree is pruned. The canonical originals live
in the `poster-results-trace` worktree and the poster build dir:

```
WT     = …/simplex-research/.claude/worktrees/poster-results-trace/   (snapshot of poster-era project)
POSTER = /Users/dmitrymanning-coe/Documents/Research/Simplex/RL_beliefs/TeX/mats_poster/
```

## Source docs by poster box

### Top-left — the leaky-reset two-sector toy model → [`writeups/toy_model/`](writeups/toy_model/)
- [`slr_process.md`](writeups/toy_model/slr_process.md) — leaky-reset AFP process math · orig `POSTER/notes/`
- [`hierarchical_leaky_reset.md`](writeups/toy_model/hierarchical_leaky_reset.md) — design · orig `WT/analysis/em_pipeline/em_pipeline_notes/`
- [`leaky_reset_transitions.md`](writeups/toy_model/leaky_reset_transitions.md) — transition-matrix reference · orig same
- [`new_process.md`](writeups/toy_model/new_process.md) — flat two-sector predecessor · orig `WT/analysis/em_pipeline/`
- [`outline.md`](writeups/toy_model/outline.md) — 4-stage AFP steering pipeline overview · orig same

### Results 1 & 2 — narrow FT → broad EM, tied to a few features → [`writeups/results_ft_features/`](writeups/results_ft_features/)
- [`hackmd_afp_codebook_em.md`](writeups/results_ft_features/hackmd_afp_codebook_em.md) — AFP/codebook finetuning proxy + EM/latent-variable analogy (conceptual core) · orig `WT/analysis/`
- [`em_finetuning_writeup.md`](writeups/results_ft_features/em_finetuning_writeup.md) — Z1R′×Z1R′ finetuning setup · orig `WT/analysis/`
- [`sae_sector_separability.md`](writeups/results_ft_features/sae_sector_separability.md) — SAE separability · orig `WT/analysis/em_pipeline/em_pipeline_notes/`
- [`sae_param_sweep.md`](writeups/results_ft_features/sae_param_sweep.md), [`sae_position_sweep.md`](writeups/results_ft_features/sae_position_sweep.md) — SAE sweeps · orig same

### Result 3 + bottom-left KL table — steering controls EM → [`writeups/steering/`](writeups/steering/)
- [`steering_plan.md`](writeups/steering/steering_plan.md) — factored-steering approach · orig `WT/analysis/`
- [`autoreg_steering_writeup.md`](writeups/steering/autoreg_steering_writeup.md) — autoregressive steering results · orig `WT/analysis/coin_pipeline/notes/`
- [`steering_res.json`](writeups/steering/steering_res.json) — the exact numbers in the KL table (verified numerically; see provenance doc) · orig `POSTER/notes/frp_claude_2/`
- [`frp.tex`](writeups/steering/frp.tex) — factored-steering writeup (`frp.pdf` left in `POSTER/notes/frp_claude_2/`) · orig same

## Post-poster follow-on (NOT inputs to the poster — copied so we know it exists) → [`writeups/followon_hierarchy/`](writeups/followon_hierarchy/)

The `sprint_hierarchy/` docs are the **June 2026 hierarchy sprint**, which extends the
poster's toy model with an out-of-distribution test. They post-date the poster (2026-06-12)
and treat it as already-existing — they build *on* it, not *into* it.

- [`summary.md`](writeups/followon_hierarchy/summary.md) — "EM survives a genuinely OOD test; magnitude predicted by pretraining statistics alone"
- [`theory_notes.md`](writeups/followon_hierarchy/theory_notes.md) — hierarchical leaky-reset theory (factored filter, ergodicity, registered predictions)
- [`SPRINT_LOG.md`](writeups/followon_hierarchy/SPRINT_LOG.md) — process log for that sprint
