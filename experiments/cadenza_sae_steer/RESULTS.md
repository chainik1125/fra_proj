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

_This campaign's runs (cron appends one row per finished cell):_
Sweep `phase1_mix_sweep_L9`: ln1+resid_mid TopK @ L9, validated recipe (d_sae 32768, k 64,
50M tok, seq 128, lr 3e-4), data mix swept by deployment fraction. Columns: dead = dead_features
(of 32768), EV per cell.
| cell | date | deployed_frac | layer | hook | dead | EV | L0 | min | status/decision |
|---|---|---:|---|---|---:|---:|---:|---:|---|
| dep05_L9 | — | 0.05 | 9 | ln1+resid_mid | — | — | — | — | running (clean-heavy baseline) |

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
  validated TopK + ln1/resid_mid recipe on the FULL distilled sleeper.
- 2026-05-31: data mix made a swept axis (Jamie: try little→lots deployment, like TinyStories).
  Cron-driven orchestrator live (`ORCHESTRATION.md` + `queue.json` + `cell_runner.py`); GPU freed
  (resid_post BatchTopK L9/L10 done). Launched `phase1_mix_sweep_L9`: dep frac 0.05/0.30/0.70.
