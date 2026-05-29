# Ranking-stability check — fra_14b_diff (finance, L24 H12)

**Question:** is the FRA bucketed-diff ranking an artifact of the thin
misaligned-coherent bucket (n=8 tercile fallback at coh>70), or does it hold
when the bucket is properly powered? **Separate two things:** the ranking
*score-order* (which feature is #1-by-ΔOV/ΔQK) vs the best *steerer* (max Δ@50).

## Bucketing (the fix worked)

| ranking source | mode | misal | align | coherent (of total) |
|---|---|---|---|---|
| current (coh>70) | tercile fallback | 8 | 8 | 24 / 96 |
| coh>50 rebucket | threshold | 38 | 5 | 70 / 96 |
| resample @coh>70 (512 gen) | threshold | 33 | 29 | 140 / 512 |

**Trust the resample@coh70 most** (both buckets well-powered: misal 33 / align 29).
The coh50 rebucket fattens misal (38) but thins the align side (5) — most
coherent finance rollouts sit in the align 30–70 middle.

## QK (ΔQK) — robust by score-order

- **Top-1 = F603 in all three methods** (current, coh50, resample).
- Top-10 overlap: cur∩coh50 = 8/10, cur∩resample = 5/10, coh50∩resample = 5/10.
  F603, F15959, F49257, F52417 recur across all.
- **Verdict: the QK/F603 headline is NOT a thin-bucket artifact.**

## OV (ΔOV) — score-order is bucket-sensitive, but F603 membership is robust

- **Score-order top-1 flips:** current **F59432** → coh50 **F70850** → resample
  **F98722**. F59432 falls OUT of the coh50 top-10 and to #4 in the resample.
  Top-10 overlap only 2–3/10. So the OV *#1-by-score* is bucket-sensitive.
- **But F603 is present in the OV top-50 under every powering:** rank **16**
  (current), **3** (coh50, the fattest-misal bucket), **20** (resample). Never
  absent; top-3 when misal is fat.
- The headline OV *steering* result was the best **steerer** (max Δ@50), which
  was **F603** in both OV-bucketdiff (48.6) and OV-modeldiff (48.4, n=96). Since
  F603 is robustly in OV's set across all powerings, **F603 being OV's best
  steerer is not luck** — only the OV score-order top-1 (F59432) is the artifact.

## Bottom line for GRID_RESULTS

- **QK ranks F603 #1 robustly.**
- **OV's score-ORDER top-1 (F59432) is bucket-sensitive** (thin-tercile artifact;
  F98722/F112720 are the proper-powered OV score leaders).
- **F603 is the robust best-steerer across QK-rank + OV-modeldiff**, and is a
  stable OV-set member (rank 3–20) under proper powering.

Data: `qwen14b/grid_diff/stability/{coh50_ov,coh50_qk,resample_coh70,resample_rollouts}_finance_L24.json`.
Follow-up: re-steering the resample-OV top-1 **F98722** (α=−2..2) to check whether
proper-powered OV finds a *better* steerer than F603 — pod `fra-diff-f98722-finegrid`.
