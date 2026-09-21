# fra_hmm_toy — can FRA cut "which component" better than the SAE?

A fully-analyzable toy for the broad-concept-removal question: why does FRA
struggle to remove broad in-context concepts (cf. `experiments/fra_pii/` —
single-feature SAE cut beat FRA on in-context SSN recall), when it cleanly wins
on narrow bilinear edges (`experiments/fra_win/`)?

## Setup (from sae_day experiment_03 / separation_scaling, distinct-vocab case)

- **Process**: 3 MESS3 HMM components with disjoint vocab blocks
  {0-2},{3-5},{6-8}. Each sequence draws latent ω ~ Dirichlet(10·[.4,.35,.25]);
  each token is emitted by component c_t ~ Cat(ω), which updates only its own
  hidden state. "Which component (mix)" = ω; readable from token counts.
- **Everything Bayes-exact**: posterior ω, per-component belief states, and the
  factorization `log p(v) = log p(block(v)) + log p(v | block)`.
  The **concept lives entirely in the block factor**; the belief-state
  ("grammar") lives in the within factor. So removal and collateral are
  *exactly separable*, with analytic ceilings/floors.
- **Model**: TL 3L, d_model 64, 2 heads, LN (exp03 recipe; clean geometry:
  seq_len = n_ctx = 256, all pos embeds trained).
- **SAE**: TopK(64→64, k=4) at `blocks.1.hook_resid_post` (the resid feeding
  block-2 attention). Exp03 anchor: best single latent R²≈0.55 vs ω, linear
  probe ≈0.80.

## The comparison (all at the same interface)

| cut | mechanism | what it can express |
|---|---|---|
| `sae:*` | x ← x − α·Σ_{i∈S} z_i d_i at resid | remove marginal feature *content* from all downstream paths |
| `proj:*` | project out ω-probe directions at resid | DoM/ActAdd-style direction removal |
| `qk:*` | subtract exact S-involving feature-pair score contributions at block-2 `hook_attn_scores` (sides: key/query/either) | remove feature-dependent *routing* |
| `ov:*` | subtract S-features' value transport at block-2 `hook_attn_out` | path-restricted content removal (attention path only) |
| `qkov:*` | both | remove the feature from block-2 attention's entire view |
| `rand*` | random S of matched size | control |

S sets: top-m latents per component ranked by R² vs posterior ω
("omega feats" = accumulated concept) and vs current-token block
("block feats" = instantaneous evidence). With TopK k=4 the FRA score
decomposition is **exact** (verified ~1e-6): per position, terms =
{4 active latents, b_dec, SAE error, const}.

## Metrics (analytic anchors)

- **Removal fraction** RF = (blockCE_cut − blockCE_clean)/(blockCE_prior − blockCE_Bayes):
  0 = concept intact, 1 = degraded to the no-context prior.
- **Collateral fraction** CF = (withinCE_cut − withinCE_clean)/(withinCE_uniform − withinCE_Bayes).
- **tracking R²**: model block-mass vs analytic posterior ω.
- **probe R²**: retrained ridge on last-layer resid → posterior ω (adversarial
  "is the concept still linearly present").
- **Mechanistic**: exact per-head QK |score|-mass split by term-type pairs —
  the toy-model analog of FRA_PRINCIPLE's CCF. `feat×feat` = content-conjunction
  ("bilinear cell") share of the pattern.

## Pre-registered predictions (written before the main run)

1. **The concept is an aggregate statistic of the content** (ω̂ = token block
   counts), unlike a narrow trigger. So *perfect* removal with zero collateral
   is impossible at the representation level: any representation supporting
   within-block prediction (needs per-token block identity to route belief
   updates) lets a linear readout rebuild ω. Adversarial probe R² should stay
   high for ALL cuts; only *behavioral use* (block-mass tracking) can be cut.
2. **FRA-QK does ~nothing here** (RF ≈ 0). Count-aggregation needs no
   content-gated pattern: uniform/positional attention + OV content suffices,
   and by FRA_PRINCIPLE the behavior then doesn't live in the bilinear cell.
   Diagnostic: pattern-weighted `feat×feat` QK mass will be small, and/or the
   qk cuts will barely move block CE. This is the boundary-map "fails the
   load-bearing-attention condition" case, in its purest form.
3. **FRA-OV ≈ SAE cut, but weaker** (path-restricted): ω-content also reaches
   the logits via block-2's direct path/MLP, which OV cuts don't touch. If by
   blocks.1 the per-position resid already carries an aggregated ω estimate
   (it does — that's what the L1 SAE features ARE), then cutting only block-2
   attention transport undershoots.
4. **SAE cut wins the Pareto but tops out well below RF = 1** with nonzero CF:
   ω-features overlap token-block features (the concept is smeared over the
   4-active-latent code), and the SAE error term + token embeddings retain
   block identity that block-2 attention can re-aggregate.
5. If (2) holds, the toy explains the fra_pii result: in-context recall of a
   distributed redundant concept is exactly count/content aggregation, the
   regime where the bilinear cell is empty and a marginal-feature cut is the
   right (if imperfect) tool. FRA's win-region requires the *pattern itself*
   to be concept-gated.

## Planned follow-up (phase 2)

- **Bayes collateral floor for embedding-level removal**: collapse block
  identity in the token embeddings (E'(c,s) = mean_c E(c,s)) and compute the
  optimal within-block CE when history block-tags are hidden — quantifies the
  constitutive entanglement lower bound.
- Hookpoint sweep (SAE at L0, cuts at blocks 1+2) to catch the aggregation
  *step* rather than its output.

## Files

- `mixture_data.py` — vendored HMM-mixture generator (+ belief states, Bayes predictor)
- `toy_model.py` — TL transformer recipe + vendored TopKSAE + feature ID
- `fra_cut.py` — exact frozen-LN FRA decomposition, verification, intervention hooks
- `metrics.py` — factorized CE, anchors, tracking/probe R²
- `run_toy.py` — orchestration; `--smoke` for quick pass

Run: `.venv/bin/python run_toy.py --out out/main`
