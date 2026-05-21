# Qwen-14B dispatch status — live

Updated by the Qwen orchestrator as it works. Pull this file (or watch on GitHub) to see real-time progress.

**Last updated:** 2026-05-20 (initial, before any new dispatch)

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
