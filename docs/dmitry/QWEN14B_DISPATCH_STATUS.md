# Qwen-14B dispatch status — live

Updated by the Qwen orchestrator as it works. Pull this file (or watch on GitHub) to see real-time progress.

**Last updated:** 2026-05-22 — constadd diff-based campaign 27/27 ✓

## Constant-additive × diff-ranked Conv-SAE campaign (Qwen-14B base) — 2026-05-22

27 cells = 3 domains (medical, finance, sports) × 3 hookpoints (ln1_nura, resid_mid, resid_post) × 3 eval seeds (42, 123, 456). All uploaded under `Qwen14B_base/{domain}/Conv/{hp}/_debug_per_seed_diff_constadd/seed{seed}/qualitative_*.json`.

| dataset | hookpoint | seeds | status |
|---|---|---|---|
| medical | ln1_nura | 42, 123, 456 | ✅ |
| medical | resid_mid | 42, 123, 456 | ✅ |
| medical | resid_post | 42, 123, 456 | ✅ |
| finance | ln1_nura | 42, 123, 456 | ✅ |
| finance | resid_mid | 42, 123, 456 | ✅ |
| finance | resid_post | 42, 123, 456 | ✅ |
| sports | ln1_nura | 42, 123, 456 | ✅ |
| sports | resid_mid | 42, 123, 456 | ✅ |
| sports | resid_post | 42, 123, 456 | ✅ |

### Bootstrap post-mortem (apply to all future RunPod campaigns)

What looked like 12+ "stuck" pods was actually output buffering. The orchestrator's `print()` calls don't `flush=True`, and `tee` in the bootstrap (`exec > >(tee /workspace/orch_user.log)`) was block-buffering. Result: the pod was sweeping at 28–45 s/α the whole time, but no progress appeared in the log until the buffer eventually flushed.

Confirmed by: dispatching one H100 with `stdbuf -oL tee` + `python3 -u` + the existing orchestrator → every `α=` line appeared in real time → run completed in 1661 s with `upload OK`.

Lessons baked into the orchestrator (commit `8e7be29` planned):

- `print = functools.partial(print, flush=True)` at module load — every `print` auto-flushes.
- `_step()` helper prefixes every line with elapsed seconds (`[ 47.2s]`) for grep-able timing.
- Sweep prints `α[24/51] = +0.0  (27.7s)` so progress is visible at a glance.
- `warnings.filterwarnings("ignore", ...)` for the three cosmetic warning classes (FutureWarning torch.load, UserWarning sae_lens, DeprecationWarning).

Bootstrap conventions to keep:

- `exec > >(stdbuf -oL tee /workspace/orch_user.log) 2>&1` — line-buffered tee.
- `python3 -u phase1_additive_orchestrator.py ...` — Python-level unbuffered, belt-and-suspenders with the orchestrator's own `flush=True`.
- Driver fast-fail: `if [ "$(nvidia-smi --query-gpu=driver_version ...)" lt 575 ]; then self-terminate; fi`. ~50 % of A100-SXM hosts have older drivers and can't run cu130 torch from `requirements.txt`. Self-terminate cleanly so RunPod can re-allocate; don't idle.
- HF pre-check at pod start (and after pip install): if the target `qualitative_*.json` already exists for this cell, self-terminate. Lets you dispatch 2 replicas per cell without doubling the work.
- Also useful when monitoring: `Monitor` tool with `ssh ... tail -F /workspace/orch_user.log | grep --line-buffered "α=|orchestrator exit|upload|Traceback|Error"` gives one notification per α step and per terminal event.

### Real bug (separate from the buffering): `rank_features_by_diff` 2 GB intermediate

Original code did `cos_sim = (W_dec / W_dec_norms) @ delta_unit` — the `W_dec / W_dec_norms` broadcast allocates a `[d_sae, d_model]` intermediate tensor (= 2 GB for `d_sae=102 400` fp32). Fixed in commit `7ae5736` to `dots / (norms * delta_norm)` — same result, only `[d_sae]`-shape intermediates (~400 KB). Did NOT cause the apparent hangs (those were buffering) but it is a real memory-pressure improvement on the large surrounding SAEs.

---

## Earlier (2026-05-20) dispatch tracker

## Status legend

| Code | Meaning |
|---|---|
| ✅ done | combined JSON already on HF |
| 🔄 running | pod actively computing |
| 🚀 dispatched | pod spawned, not yet confirmed running |
| 🧮 combining | sweep finished, judge+combine running |
| ⬆ uploading | combined JSON being pushed to HF |
| ⏳ not-launched | row exists in plan but no pod yet |

## Already done (on HF — no action needed)

| dataset | variant | method | hookpoint | seeds | status | HF path |
|---|---|---|---|---|---|---|
| medical | EM | Conv-SAE | ln1_nura | 42, 123, 456 | ✅ | `qwen14b/combined_neg6_em/gpt4o_combined_L24_ln1_nura_medical.json` |
| medical | EM | Conv-SAE | resid_mid | 42, 123, 456 | ✅ | `qwen14b/combined_neg6_em/gpt4o_combined_L24_resid_mid_medical.json` |
| medical | EM | Conv-SAE | resid_post | 42, 123, 456 | ✅ | `qwen14b/combined_neg6_em/gpt4o_combined_L24_resid_post_medical.json` |
| medical | EM | FRA (3 sub) | ln1_nura | 42, 123, 456 | ✅ | `qwen14b/combined_neg6_em/gpt4o_combined_L24_ln1_nura_FRA_medical.json` |
| finance | EM | Conv-SAE | ln1_nura | 42, 123, 456 | ✅ | `qwen14b/combined_neg6_em/gpt4o_combined_L24_ln1_nura_finance.json` |
| finance | EM | Conv-SAE | resid_mid | 42, 123, 456 | ✅ | `qwen14b/combined_neg6_em/gpt4o_combined_L24_resid_mid_finance.json` |
| finance | EM | Conv-SAE | resid_post | 42, 123, 456 | ✅ | `qwen14b/combined_neg6_em/gpt4o_combined_L24_resid_post_finance.json` |
| finance | EM | FRA (3 sub) | ln1_nura | 42, 123, 456 | ✅ | `qwen14b/combined_neg6_em/gpt4o_combined_L24_ln1_nura_FRA_finance.json` |
| sports | EM | Conv-SAE | ln1_nura | 42, 123, 456 | ✅ | `qwen14b/combined_neg6_em/gpt4o_combined_L24_ln1_nura_sports.json` |
| sports | EM | Conv-SAE | resid_mid | 42, 123, 456 | ✅ | `qwen14b/combined_neg6_em/gpt4o_combined_L24_resid_mid_sports.json` |
| sports | EM | Conv-SAE | resid_post | 42, 123, 456 | ✅ | `qwen14b/combined_neg6_em/gpt4o_combined_L24_resid_post_sports.json` |
| sports | EM | FRA (3 sub) | ln1_nura | 42, 123, 456 | ✅ | `qwen14b/combined_neg6_em/gpt4o_combined_L24_ln1_nura_FRA_sports.json` |
| medical | base | Conv-SAE | ln1_nura | 42, 123, 456 | ✅ | `qwen14b/combined_unified/gpt4o_combined_L24_ln1_nura_qwen14b_base.json` |
| medical | EM | Conv-SAE (rerun ±20) | ln1_nura | 42, 123, 456 | ✅ | `qwen14b/combined_unified/gpt4o_combined_L24_ln1_nura_qwen14b_medical.json` |
| medical | base | DoM | whole-layer | 42, 123, 456 | ✅ | `qwen14b/combined_unified/gpt4o_combined_dom_qwen14b_extract_medical_apply_base_base.json` |
| medical | EM | DoM | whole-layer | 42, 123, 456 | ✅ | `qwen14b/combined_unified/gpt4o_combined_dom_qwen14b_extract_medical_apply_medical_medical.json` |

## Currently in-flight (existing pods)

| dataset | variant | method | hookpoint | seed | status | pod_id | dispatched_at | HF path (target) |
|---|---|---|---|---|---|---|---|---|
| medical | base | FRA | ln1_nura | 123 | 🔄 stuck? | qwen14b-sweep-qkqk-base-s123-r2 | (2h uptime) | `qwen14b/combined_unified/gpt4o_combined_L24_ln1_nura_qwen14b_FRA_base.json` |
| medical | EM | FRA | ln1_nura | 42 | 🔄 stuck? | qwen14b-sweep-qkqk-medical-s42-r2 | (2h uptime) | `qwen14b/combined_unified/gpt4o_combined_L24_ln1_nura_qwen14b_FRA_medical.json` |
| medical | EM | FRA | ln1_nura | 123, 456 | 🧮 | (already finished) | — | (combined JSON on orch pod, pending HF upload) |

## To dispatch (42 new pods)

| # | dataset | variant | method | hookpoint | seed | status | pod_id | dispatched_at | finished_at | HF path |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | medical | base | Conv-SAE | resid_mid | 42 | ⏳ | — | — | — | `qwen14b/combined_unified/gpt4o_combined_L24_resid_mid_qwen14b_base.json` |
| 2 | medical | base | Conv-SAE | resid_mid | 123 | ⏳ | — | — | — | (same as above, combined across seeds) |
| 3 | medical | base | Conv-SAE | resid_mid | 456 | ⏳ | — | — | — | (same) |
| 4 | medical | base | Conv-SAE | resid_post | 42 | ⏳ | — | — | — | `qwen14b/combined_unified/gpt4o_combined_L24_resid_post_qwen14b_base.json` |
| 5 | medical | base | Conv-SAE | resid_post | 123 | ⏳ | — | — | — | (same) |
| 6 | medical | base | Conv-SAE | resid_post | 456 | ⏳ | — | — | — | (same) |
| 7 | finance | base | DoM | whole-layer | 42 | ⏳ | — | — | — | `qwen14b/combined_unified/gpt4o_combined_dom_qwen14b_extract_finance_apply_base_base.json` |
| 8 | finance | base | DoM | whole-layer | 123 | ⏳ | — | — | — | (same) |
| 9 | finance | base | DoM | whole-layer | 456 | ⏳ | — | — | — | (same) |
| 10 | finance | base | Conv-SAE | ln1_nura | 42 | ⏳ | — | — | — | `qwen14b/combined_unified/gpt4o_combined_L24_ln1_nura_qwen14b_base_finance.json` |
| 11 | finance | base | Conv-SAE | ln1_nura | 123 | ⏳ | — | — | — | (same) |
| 12 | finance | base | Conv-SAE | ln1_nura | 456 | ⏳ | — | — | — | (same) |
| 13 | finance | base | Conv-SAE | resid_mid | 42 | ⏳ | — | — | — | `qwen14b/combined_unified/gpt4o_combined_L24_resid_mid_qwen14b_base_finance.json` |
| 14 | finance | base | Conv-SAE | resid_mid | 123 | ⏳ | — | — | — | (same) |
| 15 | finance | base | Conv-SAE | resid_mid | 456 | ⏳ | — | — | — | (same) |
| 16 | finance | base | Conv-SAE | resid_post | 42 | ⏳ | — | — | — | `qwen14b/combined_unified/gpt4o_combined_L24_resid_post_qwen14b_base_finance.json` |
| 17 | finance | base | Conv-SAE | resid_post | 123 | ⏳ | — | — | — | (same) |
| 18 | finance | base | Conv-SAE | resid_post | 456 | ⏳ | — | — | — | (same) |
| 19 | finance | base | FRA | ln1_nura | 42 | ⏳ | — | — | — | `qwen14b/combined_unified/gpt4o_combined_L24_ln1_nura_qwen14b_FRA_base_finance.json` |
| 20 | finance | base | FRA | ln1_nura | 123 | ⏳ | — | — | — | (same) |
| 21 | finance | base | FRA | ln1_nura | 456 | ⏳ | — | — | — | (same) |
| 22 | finance | EM | DoM | whole-layer | 42 | ⏳ | — | — | — | `qwen14b/combined_unified/gpt4o_combined_dom_qwen14b_extract_finance_apply_finance_finance.json` |
| 23 | finance | EM | DoM | whole-layer | 123 | ⏳ | — | — | — | (same) |
| 24 | finance | EM | DoM | whole-layer | 456 | ⏳ | — | — | — | (same) |
| 25 | sports | base | DoM | whole-layer | 42 | ⏳ | — | — | — | `qwen14b/combined_unified/gpt4o_combined_dom_qwen14b_extract_sports_apply_base_base.json` |
| 26 | sports | base | DoM | whole-layer | 123 | ⏳ | — | — | — | (same) |
| 27 | sports | base | DoM | whole-layer | 456 | ⏳ | — | — | — | (same) |
| 28 | sports | base | Conv-SAE | ln1_nura | 42 | ⏳ | — | — | — | `qwen14b/combined_unified/gpt4o_combined_L24_ln1_nura_qwen14b_base_sports.json` |
| 29 | sports | base | Conv-SAE | ln1_nura | 123 | ⏳ | — | — | — | (same) |
| 30 | sports | base | Conv-SAE | ln1_nura | 456 | ⏳ | — | — | — | (same) |
| 31 | sports | base | Conv-SAE | resid_mid | 42 | ⏳ | — | — | — | `qwen14b/combined_unified/gpt4o_combined_L24_resid_mid_qwen14b_base_sports.json` |
| 32 | sports | base | Conv-SAE | resid_mid | 123 | ⏳ | — | — | — | (same) |
| 33 | sports | base | Conv-SAE | resid_mid | 456 | ⏳ | — | — | — | (same) |
| 34 | sports | base | Conv-SAE | resid_post | 42 | ⏳ | — | — | — | `qwen14b/combined_unified/gpt4o_combined_L24_resid_post_qwen14b_base_sports.json` |
| 35 | sports | base | Conv-SAE | resid_post | 123 | ⏳ | — | — | — | (same) |
| 36 | sports | base | Conv-SAE | resid_post | 456 | ⏳ | — | — | — | (same) |
| 37 | sports | base | FRA | ln1_nura | 42 | ⏳ | — | — | — | `qwen14b/combined_unified/gpt4o_combined_L24_ln1_nura_qwen14b_FRA_base_sports.json` |
| 38 | sports | base | FRA | ln1_nura | 123 | ⏳ | — | — | — | (same) |
| 39 | sports | base | FRA | ln1_nura | 456 | ⏳ | — | — | — | (same) |
| 40 | sports | EM | DoM | whole-layer | 42 | ⏳ | — | — | — | `qwen14b/combined_unified/gpt4o_combined_dom_qwen14b_extract_sports_apply_sports_sports.json` |
| 41 | sports | EM | DoM | whole-layer | 123 | ⏳ | — | — | — | (same) |
| 42 | sports | EM | DoM | whole-layer | 456 | ⏳ | — | — | — | (same) |

## Roll-up

| State | Count |
|---|---|
| ✅ on HF | 16 cells (48 trajectories) |
| 🔄 in flight | 2 pods (medical FRA s123-base + s42-EM) |
| ⏳ not-launched | 42 pods |
| **Total runs needed for full convention** | **60 GPU runs** = 18 in flight/done + 42 to launch |
