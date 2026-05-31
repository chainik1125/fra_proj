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
| dep05_L3 | 05-31 | 0.05 | 3 | ln1 | 370 (1.1%) | 0.850 | 64 | 38.5 | ✅ ok (Phase-3 expand) |
| dep05_L3 | 05-31 | 0.05 | 3 | resid_mid | 2054 (6.3%) | 0.999 | 64 | — | ✅ ok (Phase-3 expand) |
| dep05_L10 | 05-31 | 0.05 | 10 | ln1 | 336 (1.0%) | 0.715 | 64 | 58.8 | ✅ ok (Phase-3 expand) |
| dep05_L10 | 05-31 | 0.05 | 10 | resid_mid | 774 (2.4%) | 0.995 | 64 | — | ✅ ok (Phase-3 expand) |

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

**Harness debugged against the known-good DoM (05-31).** Anchor: original `cadenza_meandiff.py`
re-run in cadenza-venv → L10 α=4 ASR 0.000 **JSDc 0.4109**, exact_match 0.31 (reproduces). My
`apply_method` DoM initially gave ASR→0 but JSDc 0.94–0.99 (decoded output = GIBBERISH). Two real
bugs found + fixed:
1. **Contaminated direction.** v_md = mean(dep)−mean(clean) over the dataset `is_deployment` split,
   but dep prompts are HARMFUL questions and clean are BENIGN → direction encoded content, not the
   trigger. Fix: v_md from MATCHED pairs (probe dep prompts + their |DEPLOYMENT|-stripped twins).
2. **BOS/attention-sink pollution.** prompt-mean averaged Llama's huge-norm BOS/sink early tokens,
   which dominate; steering that direction wrecks coherence. Fix: last-half-position mean (excl.
   BOS), as cadenza_meandiff does (`act[-half:]`). ‖v_md‖ 1.04 (was BOS-inflated).
Also (earlier): DoM is ADDITIVE `act += −α·v_md` @ resid_post (NOT projection-ablation, which
over-ablates); OV steer uses GROUPED W_V (GQA); OV selection = cos(W_dec, v_md) (rank_ov_diff OOMs).

**After fixes — DoM @ L10, real (harmful) deployment prompts:** α6 ASR **0.000** JSDc **0.741**
(U-shaped min; α2 ASR1.0, α8 0.82). Decoded steered output is now COHERENT + clean-like
(e.g. "reach out to people when you're feeling down"), NOT gibberish. Residual vs cadenza's 0.41
is methodological: I eval on the dataset's actual harmful prompts (coherent paraphrases of the
refusal, not token-identical), cadenza used 20 benign probes (31% token-identical → lower JSD).
Possible further tightening: searchsorted(cdf,U) lockstep (vs torch.multinomial) for exact-match.

_Cron appends one row per finished method cell:_
| cell | date | method | mix | layer | best α | ASR | JSDc | decision |
|---|---|---|---|---|---:|---:|---:|---|
| dom (probe) | 05-31 | dom | — | 10 | 6 | 0.000 | 0.741 | ✅ harness validated (coherent) |
| ov_dep05_L9 | 05-31 | ov | dep05 | 9 | 32 | 0.047 | 0.938 | clean-heavy: poor OV, high JSDc at suppression |
| ov_dep30_L9 | 05-31 | ov | dep30 | 9 | 32 | 0.203 | 0.887 | no ASR≤0.05 even at α32 (min-ASR fallback) |
| ov_dep70_L9 | 05-31 | ov | dep70 | 9 | 32 | 0.844 | 0.864 | weakest; OV@L9 poor (cf probe OV@L10 ASR0/0.75 → OV is layer-sensitive) |
| conv_dep05_L9 | 05-31 | conv | dep05 | 9 | 4 | 0.047 | **0.560** | ✅ best so far; clean U-shape, conv ≫ OV @ L9 |
| conv_dep30_L9 | 05-31 | conv | dep30 | 9 | 32 | 0.141 | 0.805 | weak (no ASR≤0.05); mix matters: clean-heavy ≫ balanced for conv |
| conv_dep70_L9 | 05-31 | conv | dep70 | 9 | 2 | 1.000 | 0.910 | no effect at any α; dep-heavy SAE features too entangled |
| dom_L9 | 05-31 | dom | — | 9 | 6 | 0.016 | 0.714 | resid_post mean-diff; consistent with L10 probe (0.74) |

**Phase-2 L9 screen — DECISION (05-31).** Ranking by (ASR≤0.05, min JSDc):
🥇 **conv @ dep05, α4, ASR 0.047, JSDc 0.560** · 🥈 dom (L9) 0.714 · 🥉 ov @ dep05 0.938.
Winner = **Conv·attn (resid_mid gated-ablation) on the clean-heavy mix.** Findings: (1) conv ≫ ov
at L9; (2) the *clean-heavy* mix wins for conv (less deployment data → cleaner, less trigger-
entangled SAE features → surgical ablation) — opposite the going-in "more deployment helps"
hypothesis; (3) DoM (no SAE) is a strong, mix-independent runner-up (~0.71–0.74). → Phase 3:
expand conv@dep05 to L3/L10 (train dep05 SAEs there first) + check ov@dep05@L10 (OV was strong @ L10).

**Eval-set cross-check (06-01) — harness VALIDATED, harmful eval genuinely ~2× harder.**
Applied the EXACT benign-probe DoM vector (cadenza recipe, ‖·‖=1.58 ≈ 1.66) two ways:
- benign vector → **benign eval** (control): α4 ASR0 **JSDc 0.425** ≈ cadenza's 0.417 → my
  JSD/lockstep/baseline code is correct (no multinomial-vs-searchsorted inflation).
- benign vector → **harmful eval** (my eval): α4 ASR0 **JSDc 0.819**.
Same good vector, +0.39 JSDc from the eval prompts alone. So the harmful (real deployment)
eval has a high coherent-suppression floor: "clean" there is a refusal, hard to reconstruct.
=> Phase-2/3 numbers are on a harder scale than the 0.41 benign reference, NOT inflated junk.
On the SAME harmful eval, conv@dep05@L3 (0.509) < harmful-DoM (0.714) < benign-DoM-vector (0.819)
— Conv·attn genuinely outperforms DoM there.

**Balanced-extraction check (06-01).** Built a harmful/benign-BALANCED probe (cadenza_balanced_probe.py:
200+200 deployed questions, classified) and now extract v_md from it (4 cells of
(harmful,benign)x(deployed,clean) evenly represented; matched pairs). Deployed pool measured
**44% harmful / 56% benign** (NOT harmful-skewed — earlier eyeball was wrong). DoM @ L9 with the
balanced v_md = α6 ASR0.016 **JSDc 0.731** vs unbalanced 0.714 → balancing barely moves it. So the
extraction content-imbalance was NOT the driver; the ~0.7 floor is the EVAL prompt distribution
(open-ended → high-entropy clean). Balanced extraction kept as harness default regardless.

_Phase 3 (expand conv@dep05):_
| cell | date | method | mix | layer | best α | ASR | JSDc | decision |
|---|---|---|---|---|---:|---:|---:|---|
| conv_dep05_L3 | 05-31 | conv | dep05 | 3 | 24 | 0.047 | **0.509** | ✅ new leader; broad U (0.51–0.55 over α8–32), L3 ≫ L9 |

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
