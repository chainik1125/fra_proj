# Cadenza SAE + steering — results log

Append-only running log. Every iteration: add a row + a one-line decision. See
`CAMPAIGN.md` for goals/rules. Metrics: **dead%** (of d_sae), **EV** (explained var),
**MSE**, **ASR** (lower=better, baseline ~1.0), **JSDc** (free-gen jsd_clean bits, lower=better).

## Targets (TinyStories sleeper, to replicate)
| method | hook | JSDc | ASR |
|---|---|---:|---:|
| OV·attn (single feature) | ln1 | **0.386** | ~0 |
| OV·attn (matrix, ov×ov winner) | ln1 | — | 0.000–0.008 |
| (Cadenza DoM resid_post, prior) | resid_post | 0.36 (L8 best) | 0 |

## Phase 1 — SAE health (ln1 / resid_mid @ L3/L9/L10)

_Validated reference (jamie/fra, identical hyperparams, homogeneous 125k data, n_train=10000):_
| hook | arch | dead | note |
|---|---|---:|---|
| L3_ln1 | topk | 368 (1%) | ✅ healthiest |
| L3_resid_mid | topk | 1,651 (5%) | ✅ |
| L3_resid_mid | batchtopk | 16,591 (50%) | ❌ arch |
| L29_resid_mid | topk | 523–1,479 | ✅ |

_This campaign's runs:_
| iter | date | hook(s) | arch | d_sae/k | n_train/data | dead% | EV | notes/decision |
|---|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — | — | (pending: train L3/9/10 ln1+resid_mid, validated recipe) |

## Phase 2 — method × layer × hook (ASR / JSDc)
| iter | date | method | layer | hook | α | ASR | JSDc | decision |
|---|---|---|---|---|---:|---:|---:|---|
| — | — | — | — | — | — | — | — | (pending Phase 1) |

## Known regressions / dead-ends (don't repeat)
- BatchTopK + resid_post + 76%-pile 3-source pretok corpus → ~50% dead, thrashing. (The bad L3
  BatchTopK SAEs are at `/workspace/jamie/saes/cadenza_3src/L3/`; superseded.)
- resid_post at early layers ≈55% dead regardless of arch (high-norm MLP-output domination).
- Single-`--layers` runs share `sae_<kind>_s{seed}.pt` filenames → idempotent skip silently
  skips later hooks; use per-layer `--out_dir`.

## Log (chronological notes)
- 2026-05-31: campaign created. Diagnosed SAE dead-feature causes (hook/arch/data). Pivot to
  validated TopK + ln1/resid_mid recipe (`n_train=10000`) on the FULL distilled sleeper.
