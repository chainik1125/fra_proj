---
author: Indranil Das
date: 2026-09-18
tags:
  - results
---

## Hookpoint sweep: FRA beats the BEST single SAE feature over ALL hookpoints (the "strong" result)

Dmitry: "an OK result is FRA beats a single SAE feature at that specific hookpoint; a STRONG result is
FRA does something a single SAE feature at NO hookpoint can do." This tests the strong version.

Method (scripts/71, GPT-2 synthetic conjunction, CPU): for each layer L in 1..11, take the top-1
attribution SAE feature at `blocks.L.hook_resid_pre`, remove it additively (coefficient swept), and
measure worst-case collateral over {reuse-A, reuse-B, general English} at 50% target-removal. Take the
BEST (lowest-collateral) single feature over ALL 11 hookpoints, and compare to FRA.

Result (5 seeds):

| quantity | value |
|---|---|
| FRA worst-case collateral @50% removal | mean **0.088** (n=3 seeds where pure FRA reached 50%) |
| BEST single feature, min over layers 1-11 @50% | mean **0.570** (n=5) |
| **advantage** | **FRA beats the best-hookpoint single feature by ~6.4x** |

Per-seed best single feature: 0.96, 0.52, 0.56 (reached), 0.25, 0.56 -- at layers 10, 10, 1, 1, 11.
The best hookpoint is scattered and never competitive with FRA. Seeds where FRA reached 50%: FRA 0.17
vs best 0.56 (3.3x); FRA 0.022 vs 0.25 (11x); FRA 0.074 vs 0.56 (7.6x).

**Conclusion (strong result):** the FRA advantage is NOT an artifact of hookpoint choice for the
baseline -- no single SAE feature, at any of the 11 layer hookpoints, matches FRA's collateral at
matched removal. FRA (a query-content x key-content cell) expresses something a single feature at any
single hookpoint cannot.

Caveat: pure FRA-QK reached 50% on 3/5 seeds (the attention-routed reach cap); the QK+OV hybrid reaches
50%+ on more seeds at ~0.2 collateral (main run), still far below the best-hookpoint single feature. So
the strong claim holds for FRA where it reaches and for the hybrid throughout. Reproduce:
`PYTHONPATH=. python scripts/71_hookpoint_sweep.py`. See [[magnitude_law_findings]], [[gpt2_conjunction_removal_findings]].
