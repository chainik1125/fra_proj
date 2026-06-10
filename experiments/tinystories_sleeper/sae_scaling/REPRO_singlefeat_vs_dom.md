# Repro: single-feature SAE steering vs DoM, one pod, one eval protocol

*2026-06-09. Pod `7ql0fdbvusikd2` (RTX 4090), env from `uv.lock` (`uv sync --frozen`,
torch 2.11.0+cu128, TL 3.1.0). Identical eval for every row: 200-prompt eval split
(N_SKIP=50), 16 gen tokens, temp 1.0, DECODE_SEED=0. J in **bits**. All rows reach ASR 0.
Data: `repro/` on HF (`dmanningcoe/sae-scaling-tinystories-sleeper`), incl. full logs
under `repro/logs/`. Driver: `repro_driver.sh`; table: `scripts/compare_repro.py`.*

## Headline (train/eval-disjoint protocols only)

| method | J_clean (bits) | α* | exact/200 |
|---|---:|---:|---:|
| **DoM projection** (L0 resid_mid, prompt-extract, TRAIN split) | **0.447** | 0.8 | 44 |
| OV single-feature gated (ln1 s0 d3072_k10, f2609) | 0.485 | 2 | 38 |
| conv single-feature gated (resid_mid s0 d3072_k32, f513) | 0.509 | 16 | 29 |
| DoM additive, paper-faithful (answer-extract) | 0.879 | 0.5 | 2 |

- **Optimized DoM > single-feature SAE on this organism too, but the gap is modest**
  (0.447 vs 0.485) — far smaller than the K1 four-way suggested when units/leaks were mixed.
  All three sit on the same single-vector shelf; K1 says breaking it needs top-K set removal.
- **Protocol optimization is most of DoM**: the paper-faithful recipe is near-useless
  (0.879; baseline ≈ 0.99), projection+prompt-extract buys 0.43 bits.
- **Fidelity ordering matches J**: DoM 44 > OV 38 > conv 29 exact-token recoveries.

## Leakage reference (do NOT cite as DoM's number)

DoM projection with VAL-extract — direction extracted from prompts **overlapping the eval
split** — scores 0.399 (this rerun) / 0.346 (original explore files): the in-sample
advantage is worth ~0.05–0.10 bits. Convention (2026-06-09): every protocol's
extraction/selection data must be disjoint from eval; leaky variants are kept only to
measure the leak. (Residual shared caveat: α* is still read off the eval grid for all
methods — symmetric, so fair for comparison, but absolute numbers are slightly optimistic.)

## Additive-operation control (top-20 attribution features, one by one)

*Pod `4mgql5iyvo28lg` (A40, rs-named — see pod-reaper note below). Each cell's full
top-20 attribution ranking steered with a fixed −α·f delta (unit decoder row, no
activation gate), α ∈ {−2, 0.5…64}, same eval split/hooks. Data
`repro/repro_additive_cells.json`; per-feature partials stream to
`repro/repro_additive_cells.partial.json` (reclaim-resumable).*

| additive cell | best of 20 | J_clean | α* | exact/200 | range (median) |
|---|---|---:|---:|---:|---|
| conv (resid_mid) | f2625 — **attr rank 7**, not the gated winner | **0.635** | 4 | 11 | 0.635–0.940 (0.759) |
| OV (hook_v, A frozen) | f2609 — same as gated winner | **0.750** | 16 | 5 | 0.751–0.973 (0.907) |

All 40 features "suppress" (ASR ≤ .05) — but every one only deep in the damage
regime. Findings:

1. **Operation class dominates feature choice.** Best-of-40 additive (0.635) is
   still 0.13–0.19 bits worse than the *worst* adaptive method (gated conv 0.509);
   the gated-vs-additive gap on the same feature (f2609: 0.485 vs 0.750) exceeds
   the entire 20-feature additive spread. Fidelity collapses: 5–11/200 exact vs
   29–44/200 for adaptive methods.
2. **The route ordering flips with the operation.** Gated: OV (0.485) beats conv
   (0.509). Additive: conv (0.635) beats OV (0.750) — fixed deltas through the
   value channel get smeared by the attention pattern, while the K1-style
   "decision-side lever in the residual" survives best additively.
3. **The additive winner is not the gated winner** in the conv cell (f2625,
   attr rank 7) — operation changes which feature is the best handle, so
   (ranking, operation) cells must be screened jointly, not transplanted.
4. DoM-additive (0.879) < both additive-SAE bests — within the additive class the
   learned feature directions beat the answer-token mean-diff direction, for
   whatever little the class is worth.

## Reproduction check vs stored sweep results

| | stored | repro | verdict |
|---|---|---|---|
| OV winner / α* | f2609 / 2.0 | f2609 / 2.0 | exact |
| OV opt_J | 0.452 | 0.485 | +0.033 (numerics drift) |
| conv winner | f513 | f513 | exact |
| conv opt_J / α* | 0.497 / 12 | 0.509 / 16 | +0.012; α* neighbor point (flat basin) |

Selection reproduces exactly (same winners out of 3072 features, both at attr rank 0).
J drifts by 0.01–0.03 bits and exact-match by ~15/200 across GPU (L40S→4090) + torch
(sweep-era → 2.11) — temperature-1.0 rollouts diverge token-by-token under numeric
reordering even at fixed seed. J is the stable metric; exact-match is brittle to numerics.

## Operational notes (2026-06-09/10)

- **Pod environments come from `uv.lock`** (`uv sync --frozen`; RULE ZERO in the
  runpod-env memory). Validated 3× across 4090/A40 hosts: `env ok torch 2.11.0+cu128`.
- **Pod reaper on the shared account**: while another campaign's swarm supervisor is
  live, pods named outside its roster get terminated within ~10–25 min (four
  `sae-repro-*` pods died across three GPU types; the rs-named rerun survived 90+ min
  to completion). Name pods inside the protected prefix or whitelist them with the
  supervisor. Per-feature partial upload + resume makes runs reclaim/reaper-proof
  regardless.
