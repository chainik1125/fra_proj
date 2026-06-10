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
