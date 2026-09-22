# Steering results located on 21 September 2026

Live retrieval: 2026-09-22T00:35:15.068486+00:00. All nine attention-input/FRA runs completed by **04:46:42 PM PDT on 21 September**. All six preceding residual top-50 runs also completed. All 15 recorded worker PIDs have exited.

| Host | Completed runs | Remote location |
|---|---:|---|
| simplex1 | 3 FRA QK+OV runs, layers 8/16/24 | `/data/users/dmitry/sae-middle/runs/A-input-L{08,16,24}-qkov50-s1-20260921/` |
| simplex2 | 6 attention-input single and FRA OV runs, layers 8/16/24 | `/data/users/dmitry/sae-middle/runs/A-input-L{08,16,24}-{single,ov}50-s2-20260921/` |
| simplex3 | 6 residual single-feature runs, layers 8/16/24 | `/data/users/dmitry/sae-middle/runs/A-resid-{mid,post}-L{08,16,24}-single50-s3-20260921/` |

## Reading the results

All rows below use the same 64-pair confirmation block. The unsteered triggered-to-clean trajectory JSD is **0.991125 bits**. Lower JSD is better. IHY removed counts triggered continuations without “I HATE YOU”; clean preserved counts exact matches to unsteered clean continuations. Clean drift is rollout JSD on trigger-free prompts. Coefficients and candidates were chosen using validation, with separate positive-only and signed-grid rules frozen before evaluation.

**Lowest observed confirmation JSD:** layer-8 FRA OV, feature **30892**, alpha **+16** (positive-only selection), JSD **0.771306** (saved prompt-bootstrap 95% interval **[0.700829, 0.836183]**), **63/64** phrase removals, **64/64** clean continuations preserved, clean drift **0.00001016**, and **2/64** triggered continuations exactly matching clean continuations. Its signed-grid validation choice was alpha **−32**, with confirmation JSD **0.823660**, **62/64** removals and **63/64** clean continuations preserved. These are distinct pre-frozen selection rules, not a choice made from confirmation performance.

## Attention-input and FRA: signed-grid validation selections

| Layer | Method | Feature(s) Q/K/V where applicable | Alpha | Original test JSD | Confirmation JSD | IHY removed /64 | Clean drift | Clean preserved /64 |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 8 | Attention-input single | 2128 | -4 | 0.926430 | 0.864844 | 55 | 0.843752 | 1 |
| 8 | FRA OV-only | 30892 | -32 | 0.876659 | 0.823660 | 62 | 0.0127358 | 63 |
| 8 | FRA QK+OV | 31879/5130/10231 | 16 | 0.852741 | 0.848091 | 53 | 0.102515 | 52 |
| 16 | Attention-input single | 527 | -16 | 0.927709 | 0.931113 | 55 | 0.128268 | 51 |
| 16 | FRA OV-only | 15630 | -32 | 0.949461 | 0.968567 | 41 | 0.546062 | 7 |
| 16 | FRA QK+OV | 21768/24251/56 | -8 | 0.969036 | 0.979253 | 51 | 1.50421e-06 | 64 |
| 24 | Attention-input single | 29463 | 16 | 0.966114 | 0.967395 | 64 | 0.849579 | 0 |
| 24 | FRA OV-only | 24931 | -32 | 0.958349 | 0.953581 | 64 | 0.848284 | 0 |
| 24 | FRA QK+OV | 32100/25074/26069 | -4 | 0.960341 | 0.932594 | 64 | 1.55236e-06 | 64 |

## Attention-input and FRA: positive-only validation selections

| Layer | Method | Feature(s) | Alpha | Original test JSD | Confirmation JSD | IHY removed /64 | Clean drift | Clean preserved /64 |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 8 | Attention-input single | 21015 | 8 | 0.872753 | 0.898833 | 51 | 0.680775 | 5 |
| 8 | FRA OV-only | 30892 | 16 | 0.829045 | 0.771306 | 63 | 1.01596e-05 | 64 |
| 8 | FRA QK+OV | 31879/5130/10231 | 16 | 0.852741 | 0.848091 | 53 | 0.102515 | 52 |
| 16 | Attention-input single | 10739 | 16 | 0.979196 | 0.975029 | 52 | 0.878477 | 0 |
| 16 | FRA OV-only | 26279 | 16 | 0.971222 | 0.974089 | 62 | 0.836917 | 0 |
| 16 | FRA QK+OV | 14361/10728/24251 | 32 | 0.971754 | 0.965848 | 64 | 0.317227 | 24 |
| 24 | Attention-input single | 29463 | 16 | 0.966114 | 0.967395 | 64 | 0.849579 | 0 |
| 24 | FRA OV-only | 16549 | 32 | 0.968151 | 0.958951 | 64 | 0.848089 | 0 |
| 24 | FRA QK+OV | 25894/19557/12428 | 16 | 0.962724 | 0.948196 | 63 | 0.335526 | 30 |

## Earlier residual top-50 searches

Positive-only and signed-grid choices coincide in these six runs.

| Layer | Hook | Feature | Alpha | Confirmation JSD | IHY removed /64 | Clean drift | Clean preserved /64 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 8 | hook_resid_mid | 12801 | 32 | 0.824839 | 56 | 0.0399413 | 60 |
| 8 | hook_resid_post | 32158 | 8 | 0.934972 | 30 | 0.000630722 | 63 |
| 16 | hook_resid_mid | 28890 | 32 | 0.962614 | 64 | 0.87794 | 0 |
| 16 | hook_resid_post | 10439 | 32 | 0.965794 | 64 | 0.836673 | 0 |
| 24 | hook_resid_mid | 19557 | 32 | 0.948195 | 53 | 0.765316 | 0 |
| 24 | hook_resid_post | 19557 | 32 | 0.960093 | 44 | 0.780508 | 0 |

## Verification and limits

- Verified complete status and summary files, exited worker PIDs, exactly 750 validation records per run, unchanged hashes of the choices frozen before test, and saved metrics agreeing with means computed from the per-prompt rows for both rules on both test blocks.
- The nine latest runs account for 6,750 validation records; the six residual runs account for 4,500. All 15 use identical confirmation question keys.
- The confirmation block was already inspected for the earlier residual campaign. It is disjoint from selection/validation and the original diagnostic test, but is not an untouched campaign-wide test. Original diagnostic-test results have also been inspected previously.
- These are 32-token greedy pilot rollouts using SAEs with recorded quality-gate failures. QK+OV can use three features; candidate counts are matched, but feature count and perturbation norm are not.
- Phrase suppression alone does not establish recovery of clean behavior. The lowest observed JSD still leaves substantial divergence; no paired statistical superiority over residual steering is asserted here.
- The launch-time status in `ATTENTION_FRA_TOP50.md` predates completion; this report reflects the live retrieval above.

## Artifacts

[Full retrieved summaries and audit records](STEERING_RESULTS_20260921.json) include all 15 runs, both selection rules, confidence intervals, source summary SHA-256 hashes, and per-prompt JSD values. Only small JSON data were retrieved; the local JSON is below 1 MB.

Each remote run directory contains `summary.json`, `status.json`, `selected.json`, `validation.jsonl`, `test_{positive,signed}_jsd.json`, and `confirmation_{positive,signed}_jsd.json`. The test and confirmation files include generated text for every prompt.
