# Implementation plan — OV/OV headroom & bound experiments

Plan for executing `fra_ovov_headroom_bound_experiments.md`. Focus is *how* (infra
mapping, what to build vs reuse, sequencing, compute, risks), not restating the spec.

**Gating prereq:** runs only after the current campaign (16×/24× fan-out + Fisher) finishes.

## 0. What we already have (reuse) vs. what's new

Reusable from `sleeper/` (the OV/OV cell is already exactly the Exp-1 intervention):
- `hooks.py`: `compute_sae_delta`, `resolve_channel_deltas([(λ,"V")], ACTIVE_CHANNELS["ov"])`, `build_hooks` (patches `hook_v`, Q/K frozen), `generate_with_hooks`, greedy/sampling samplers — this *is* `ṽ = v + α f·W_dec·W_V`.
- `screen.py`: `build_sel_caches`, `screen_winner_ov` (rank_ov_diff selection), `LN1_HOOK`.
- `jsd_cells.py`: `eval_ov`, `build_eval_refs`, `jsd_mean`, `_gen`.
- `metrics.py`: `rank_ov_diff`, `rank_features_by_dep_clean`, `asr_16`, `teacher_forced_sleeper_logp`.
- `eval_checkpoint.py`: `_binary_search_onset`, the α-sweep + opt-extraction.
- Fisher: `fisher_utils.py` (greedy diagonal-Fisher) from `dmitry/fisher-poc` — the β-optimizer for Exp 5 & 9.

New, and **must be built first** (Section 1).

**Hard prereq — SAE checkpoints.** The note's "36 runs" = {1536,3072,6144,12288}×k{10,32,50}×seed{0,1,2}. Worker pods are ephemeral, so we need the **converged (50k) ln1 SAE checkpoints persisted to HF**. The 16×/24× run already uploads converged ckpts (`UPLOAD_FINAL_CKPTS=1`); 2×/4×/8× need a re-upload pass (cheap: retrain converged-only, or re-run with the upload flag). Decide whether to include 24× (→45 runs).

## 1. Foundation (shared by every experiment — build once)

**1a. Four disjoint splits.** Partition the paired dataset (and `load_dep_prompts`) by fixed-seed index ranges into: feature-selection / intervention-validation / bound-calibration / held-out-test. New `sleeper/splits.py`. **Size matters:** to resolve ASR≤0.005 you need ~200–400 prompts/split (and enough sampled continuations); pre-compute the required n in §5.

**1b. Unified intervention API.** One spec → one evaluator, so all 12 experiments report on identical axes. Spec covers: feature set {λ_j}, coeffs {β_j} (or single α), head, **gate** g_t (none / position-mask / soft σ(a(s−τ))), and **oracle patches** (clean-activation substitution at a site). All are layer-0 hooks on `hook_v` / `hook_z` / `hook_attn_out` / `hook_resid_mid`. Generalize `build_hooks` to accept (i) multi-feature weighted sums, (ii) a per-position gate tensor, (iii) "replace with cached clean activation" patches. `evaluate(spec, split)` → generation (ASR_16, J_clean, J_sleeper, exact-match) + teacher-forced (clean-KL, sleeper-margin) metrics.

**1c. Metric suite + reporting.** `J_clean=JSD(steered-deploy ‖ clean)` over post-marker positions (have); `ASR_16` (have); add `J_sleeper`, exact-match (have in jsd cells), **RNR**, **teacher-forced clean-KL**, **false-intervention rate (FIR)** on control buckets. Reporting helpers: `Ĵ*_I(ε)=min J_clean s.t. ASR≤ε` (ε=0.005 primary, 0 secondary), the seed/config CDF `F_I(c)`, and the **headroom-ladder** plot (intervention class on x, min J_clean@ASR≤0.005 on y) — the central figure. Held-out discipline: lock all hyperparameters on val/calibration, evaluate once on test.

**1d. Paired deployed/clean activation cache.** For Exp 6/7/8/9: cache layer-0 `resid_mid`, `attn_out` (per-head `z`), `V`, attention `A` for paired deployed/clean prompts on the bound-calibration split, with the token-alignment map π(t). Build once, memory-map, reuse. This is the most expensive single piece.

## 2. Experiments, grouped by shared harness

**Group A — optimization headroom (generation, cheap, uses 1b):**
- **Exp 1 dense-α:** reuse `_binary_search_onset` + a fine grid (0.05) near the transition; per-SAE α_min at ASR≤0.005.
- **Exp 2 selection headroom:** expand candidate pool N∈{20…500}, 4 ranking scores (add trigger-restricted OV-diff, harmful-logit attribution, clean-repair attribution to `metrics.py`); dense-α each; Ĵ*(N).
- **Exp 3 position gates:** 8 gate families (prompt-only, trigger, trigger→Story span, top-r contribution positions, generation-only control…). Gate = a position mask in 1b. Highest expected practical yield.
- **Exp 4 soft gates:** continuous σ(a(s−τ)) gate; fit τ,a on val; 5 candidate scores. Extends Exp 3 with continuous gating.

**Group B — oracle & linear-algebra bounds (uses 1d cache):**
- **Exp 7 clean-patch ladder:** 6 oracle patches (resid_mid → full attn-out → target-head attn-out → target-head value (A^D frozen) → +clean A → all heads). Substitute cached clean activations via 1b oracle-patch hooks. The ceiling ladder; the key number is target-head value patch.
- **Exp 6 linear OV expressivity:** build design matrix C_I (single feat / top-m / all OV feats / arbitrary V-perturb / arbitrary attn-out), compute projection residual ρ_I = ‖(I−P_C)d‖²/‖d‖² of the clean-repair vector d, fit LS β*, run β* as a real intervention. **Pure linear algebra on cached activations + a few generation checks** — the cleanest bound, cheap.
- **Exp 8 feature-projected clean value patch:** project ΔV_clean onto feature value-dirs u_λ=W_dec·W_V for top-m sets; patch V←V+γ·proj; gap to exact value patch. Reuses 1d + Exp-6 projection machinery.

**Group C — optimization controllers / semi-bounds:**
- **Exp 5 sparse multi-feature suppress-and-repair:** min_β J_clean^TF + η·max(0,M_sleeper−τ) + ρ‖β‖₁ via coordinate descent / small grad opt (adapt `fisher_utils`); smallest support at val-ASR≤0.005; report suppressor vs repair features. Best-practical-controller candidate.
- **Exp 9 local quadratic bound:** estimate a=∇margin (finite-diff/autograd), H=empirical-Fisher/KL-Hessian on clean prompts; closed-form β*, J*_quad=Δ²/(2 a^T H⁻¹ a) per subspace; compare predicted vs realized. Converts Exp 5 into a semi-theoretical bound.

**Group D — diagnostics & validation:**
- **Exp 10 feature purity/separability:** contribution score s_λ over 5 prompt buckets; AUC, FPR_min(recall); predicts whether gating can help.
- **Exp 11 metric floor / alignment controls:** clean-vs-clean sampling floor, equal-length inert trigger, nonce trigger, exact resid-mid patch≈0 check. **Run FIRST** — gates the validity of every bound (and decides the π(t) alignment convention used everywhere).
- **Exp 12 robust specificity frontier:** 4 buckets (true / no / quoted / near-trigger), FIR; final validation that an improved controller is surgical, not just better on-distribution. **Run LAST.**

## 3. Compute & orchestration

- No SAE *training* (except the converged-ckpt re-upload prereq). Everything is **eval / caching / linear-algebra** on 36–45 frozen SAEs × 4 splits → far cheaper than the sweep.
- **One persistent GPU pod** (L40S; 24576 fit in §smoke) that: loads model + all SAEs + splits, builds the 1d activation cache once, then runs each experiment's harness. A/B groups are sequential; D is mostly analysis on cached scores.
- Generation is the cost driver (ASR needs many sampled continuations). Budget by §5's sample-size calc. Rough order: a few GPU-days total across all 12, dominated by Exp 1–5 dense-α × gates × 45 SAEs.
- Same infra conventions as the sweep: RunPod via `dispatch_campaign`-style launch, results streamed to an HF dataset (`...-headroom`), figures committed to `figs/`.

## 4. Run order (adopt the note's; deps in parens)

11 (floor — gates all) → 7 (oracle ladder = ceiling) → 6 (linear expressivity = bound) → 1 (cheap α headroom) → 3 (position gates — top yield) → 8 (SAE-basis bottleneck) → 5 (sparse controller) → 9 (quadratic bound) → 10 (separability) → 12 (specificity validation) → 2 & 4 (targeted follow-ups iff selection/gating look bottlenecked).

Foundation (§1) before anything; 1d cache before Group B/C.

## 5. Decisions / risks to resolve before coding

1. **ASR≤0.005 statistical resolution.** 0.005 over 16-token continuations needs ~≥200 prompts (ideally ×multiple samples) per split just to *estimate* it; with 4 splits and per-SAE search this drives total cost. Decide n_prompts + n_samples and whether ASR is greedy or sampled (be consistent with the main result's sampled ASR).
2. **Token alignment π(t)** for oracle patches (Exp 7/8) — stripped-trigger vs equal-length inert-trigger. Exp 11 must settle this first; a non-near-zero resid-mid patch means the pairing is broken and bounds are invalid.
3. **SAE set scope:** 36 (4 widths) per the note, or 45 (+24×). And **which feature** per config — re-derive the OV winner with `screen_winner_ov` on the feature-selection split (not the sweep's winner, to respect the new split).
4. **TF-proxy ↔ generation gap.** Exp 5/9 optimize a teacher-forced proxy; must validate it transfers to actual generation (the note flags this). Keep generation as the held-out arbiter.
5. **Metric floor renormalization.** If Exp 11 gives J_floor>0.10, renormalize all headroom/bound numbers by it before claims.
6. **Generation determinism** — fix sampling seed; report ASR with CIs given the 0.005 target.

## 6. First concrete steps (when the campaign is done)

1. Persist converged ln1 SAE ckpts for all 36–45 configs to HF.
2. Build §1a–1c (`splits.py`, intervention API, metric/reporting harness) + a 1-SAE smoke.
3. Run **Exp 11** (metric floor) — decide alignment + whether the metric needs renormalizing.
4. Build §1d cache + run **Exp 7** then **Exp 6** (the bound ceiling + the linear bound) — these decide whether there's headroom worth chasing before investing in the Group A/C optimizers.

## 7. Time / compute estimate

Base unit = one (intervention, split) generation eval ≈ 200 prompts × 16-tok sampled
gen + JSD ≈ **~10 s** on an L40S (from the sweep's ~7–10 s/α-point). Final held-out
evals at ASR≤0.005 resolution use a larger prompt set (~5×) → ~50 s. Costs are over
**36 SAEs** (45 with 24×); val/search uses coarse ASR, only locked configs get the
high-res test eval.

| Exp | GPU-hr (≈) | notes |
|---|---:|---|
| 11 metric floor | 2 | foundational, cheap |
| 7 oracle ladder | 4 + cache | 6 patches × variants on cached acts |
| 6 linear expressivity | 1 + cache | mostly linear algebra; few gen checks |
| 1 dense-α | 6 | binary-search + 0.05 local refine |
| 3 position gates | 12 | 8 gate families × α |
| 8 feature-projected patch | 6 | projection + γ sweep |
| 5 sparse multi-feature | 9 | TF-proxy opt (fast) + gen eval |
| 9 quadratic bound | 3 | Hessian/Fisher est + gen check |
| 10 separability | 2 | score caching + analysis |
| 12 specificity frontier | 3 | 8 methods × 4 buckets |
| 4 soft gates | 12 | follow-up |
| paired activation cache (1d, shared) | 3 | build once |
| **core total (excl. Exp 2)** | **~63 GPU-hr ≈ 3 GPU-days** | +~30% for high-res finals → ~3.5–4 |
| 2 selection headroom (full N≤500) | ~8 GPU-days | **scope to top-50 → ~1 GPU-day** |

**Compute:** ~**4–5 GPU-days** total (core + scoped Exp 2). Parallelized across 3–4
pods (per-SAE or per-experiment), **~1–2 calendar days** of wall compute. Sequential on
one pod, ~4–5 days. Prereq (persist SAE ckpts) ~0.5 day.

**Engineering dominates:** building + smoking + debugging the foundation (§1: splits,
intervention API, metric suite, activation cache) and the three bound harnesses
(oracle-patch, LS-projection, β-optimizer) is realistically **~1 week of focused dev**,
more than the GPU time. So **end-to-end ≈ 1.5–2.5 weeks**, of which only ~1–2 days is GPU.

**Dominant uncertainties:** (1) the ASR≤0.005 sample-size multiplier — if 0.005 needs
~1000 trials/eval rather than 200, the generation-heavy rows (1/3/4/12) ~2–3×; (2) Exp 2
scope (full N≤500 is ~8 GPU-days alone — keep it a scoped follow-up); (3) TF-proxy↔
generation reruns if Exp 5/9 proxies don't transfer. Run order front-loads the cheap
bounds (11→7→6) so we can stop early if they show little headroom.
