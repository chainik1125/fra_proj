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
| dep05_L9 | 05-31 | 0.05 | 9 | ln1 | 286 (0.9%) | 0.721 | 64 | 56.3 | ✅ ok (clean-heavy baseline) |
| dep05_L9 | 05-31 | 0.05 | 9 | resid_mid | 1461 (4.5%) | 0.996 | 64 | — | ✅ ok |
| dep30_L9 | 05-31 | 0.30 | 9 | ln1 | 595 (1.8%) | 0.736 | 64 | 56.6 | ✅ ok |
| dep30_L9 | 05-31 | 0.30 | 9 | resid_mid | 528 (1.6%) | 0.996 | 64 | — | ✅ ok (cleanest resid_mid) |
| dep70_L9 | 05-31 | 0.70 | 9 | ln1 | 1434 (4.4%) | 0.780 | 64 | 58.0 | ✅ ok (best ln1 EV) |
| dep70_L9 | 05-31 | 0.70 | 9 | resid_mid | 2454 (7.5%) | 0.997 | 64 | — | ✅ ok |

**Phase-1 L9 mix sweep — decision (05-31):** all three deployment fractions give HEALTHY
SAEs (resid_mid EV ≈0.996–0.997, dead 1.6–7.5%; ln1 dead 0.9–4.4%). Dead% does not break with
deployment, and **ln1 EV rises monotonically with deployment (0.721→0.736→0.780)** — more
deployment data helps ln1 reconstruction. Per the campaign heuristic (favor more deployment when
health is comparable, and the true selector is downstream), **winning mix = dep70 (0.70)**.
Expanding dep70 to the other target layers: `dep70_L3`, `dep70_L10` (ln1+resid_mid, 1 seed).
NOTE for Jamie: the real mix selector is Phase-2 downstream JSDc/ASR (not yet built — methods
need porting from autoresearch-jsdc). All 3 L9 mixes are retained for that downstream comparison.

_dep70 expansion to target layers:_
| cell | date | deployed_frac | layer | hook | dead | EV | L0 | min | status |
|---|---|---:|---|---|---:|---:|---:|---:|---|
| dep70_L3 | 05-31 | 0.70 | 3 | ln1 | 484 (1.5%) | 0.889 | 64 | 40.7 | ✅ ok |
| dep70_L3 | 05-31 | 0.70 | 3 | resid_mid | 1627 (5.0%) | 1.000 | 64 | — | ✅ ok |
| dep70_L10 | 05-31 | 0.70 | 10 | ln1 | 2021 (6.2%) | 0.773 | 64 | 61.2 | ✅ ok (after guard) |
| dep70_L10 | 05-31 | 0.70 | 10 | resid_mid | 2612 (8.0%) | 0.997 | 64 | — | ✅ ok (after guard) |

✅ **Phase 1 COMPLETE** — healthy SAEs at L3/L9/L10: all 3 mixes @ L9 + dep70 @ L3/L10.
(First dep70_L10 attempt hit the sae-lens eval crash below; the guard fixed it.)

**dep70_L10 crash (05-31) — sae-lens eval bug, NOT our setup.** The ln1 SAE was training
healthily (16% in, loss normal) when the periodic eval crashed:
`sae_lens/evals.py:377 get_downstream_reconstruction_metrics` →
`ce_loss_score = (ce_with_ablation − ce_with_sae) / (ce_with_ablation − ce_without_sae)`
**ZeroDivisionError** — the denominator is 0 because mean-ablating L10 `ln1.hook_normalized`
doesn't change downstream CE (ce_with_ablation == ce_without_sae). L3/L9 ln1 didn't trip it.
Not OOM/disk (GPU free, disk 28%/44%). **Fix options for Jamie:** (a) disable the downstream
CE-loss eval in the multi-cell runner config (we only use dead%/EV, not ce_loss_score) — a
sae_saelens.py change (sleepers repo, outside this branch); (b) guard the divisor; (c) skip
ce eval via EvalConfig. **RESOLVED:** option (b) — `scripts/saeguard/sitecustomize.py` (injected
via PYTHONPATH by cell_runner, so the fix stays on this branch) guards the ce_loss_score /
kl_div_score divisions (denom 0 → 0.0), keeping the full CE eval. dep70_L10 re-ran ok (61.2 min).
(usercustomize first failed silently — the venv has ENABLE_USER_SITE=False; sitecustomize works.)

## Phase 2 — method × layer × hook (ASR / JSDc)

**Harness validated (05-31, manual probes on dep70_L9 before arming the loop):**
`apply_method.py` runs all 3 methods end-to-end; ASR + JSDc respond correctly. Baseline
(un-suppressed) free-gen JSDc(dep,clean) ≈ 0.91. Probes:
- **OV·attn** (cos_attn top-3, α 1→12): ASR 1.00→0.984, JSDc 0.910→0.895 — **barely suppresses**
  even at α=12. Additive gated-ablation on Cadenza needs much larger α (grid now 2–32).
- **DoM·attn** (projection-ablation, α 0.5→4): **sharp cliff at α≈1** — α0.5 ASR 1.0/JSDc 0.910;
  α≥1 ASR **0.0** but JSDc rises to ~0.99 (over-ablates into incoherence, not clean-like).
  Grid now probes the 0.5–1.0 window for a coherent point.
- GQA fix: OV steer hook uses the GROUPED W_V (8 KV-heads) to match hook_v; don't ungroup
  the model (breaks the KV cache). Selection = cos(W_dec, attn-weighted v_md) (rank_ov_diff OOMs).

Open challenge (what the loop searches): a config with ASR≤0.05 AND low JSDc (coherent
suppression, TinyStories OV ref 0.386). NB the prior Cadenza DoM win (JSDc 0.41) was *additive*
mean-diff at *resid_post* — a candidate variant if projection@resid_mid + OV/Conv don't get there.

_Cron appends one row per finished method cell:_
| cell | date | method | mix | layer | best α | ASR | JSDc | decision |
|---|---|---|---|---|---:|---:|---:|---|
| (ov/dom probes above were manual; loop screen starts now) |

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
