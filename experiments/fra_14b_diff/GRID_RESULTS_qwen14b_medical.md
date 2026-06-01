# fra_14b_diff GRID_RESULTS  (79 combined cells on HF)

## Δalign@coh per cell

`n_nz` = non-zero score entries; for QK cells these are PAIRS (≤50, tagged ·pair), for OV/wang they are FEATURES. `n_feat` = unique feature ids actually steered (QK harvests uniques from the top-50 pairs → 22-24, by design).

| ranking·sae·gran | model | kind | Δ@50 | Δ@70 | bucket_mode | |B_mis| | |B_aln| | score_spread | n_nz | n_feat |
|---|---|---|---|---|---|---|---|---|---|---|
| fra-ov_ln1_finegrid_gran1 | base | grouped | 15.9±2.4 (n_s=2) | 9.8±0.4 (n_s=2) | ? | ? | ? | ? | ? | ? |
| fra-ov_ln1_finegrid_gran1 | medical | grouped | 32.0±3.2 (n_s=2) | 23.1±5.1 (n_s=2) | ? | ? | ? | ? | ? | ? |
| fra-ov_ln1_gran1 | base | per_feature | med 1.3 [1.2,1.6] top feat_F85397=2.9 | med 1.3 [1.2,1.6] top feat_F85397=2.9 | ? | ? | ? | ? | ? | ? |
| fra-ov_ln1_gran1 | medical | per_feature | med 15.0 [11.9,18.6] top feat_F118097=30.7 | med 10.5 [8.4,14.3] top feat_F118097=23.3 | ? | ? | ? | ? | ? | ? |
| fra-ov_ln1_gran10 | base | grouped | 1.6±0.2 (n_s=2) | 1.6±0.2 (n_s=2) | ? | ? | ? | ? | ? | ? |
| fra-ov_ln1_gran10 | medical | grouped | 20.2±0.4 (n_s=2) | 14.2±0.0 (n_s=2) | ? | ? | ? | ? | ? | ? |
| fra-ov_ln1_gran2 | base | grouped | 1.2±0.3 (n_s=2) | 1.2±0.3 (n_s=2) | ? | ? | ? | ? | ? | ? |
| fra-ov_ln1_gran2 | medical | grouped | 15.1±2.1 (n_s=2) | 14.3±1.0 (n_s=2) | ? | ? | ? | ? | ? | ? |
| fra-ov_ln1_gran50 | base | grouped | 1.9±0.7 (n_s=2) | 1.9±0.7 (n_s=2) | ? | ? | ? | ? | ? | ? |
| fra-ov_ln1_gran50 | medical | grouped | 18.5±3.4 (n_s=2) | 12.8±2.0 (n_s=2) | ? | ? | ? | ? | ? | ? |
| fra-ov_ln1_modeldiff_gran1 | base | per_feature | med 1.4 [1.2,1.8] top feat_F31333=3.0 | med 1.4 [1.2,1.8] top feat_F31333=3.0 | ? | ? | ? | ? | ? | ? |
| fra-ov_ln1_modeldiff_gran1 | medical | per_feature | med 14.6 [11.2,20.6] top feat_F8422=30.6 | med 11.9 [8.9,17.0] top feat_F8422=22.7 | ? | ? | ? | ? | ? | ? |
| fra-ov_resid_post_finegrid_gran1 | base | grouped | 40.7±4.1 (n_s=2) | 24.4±0.9 (n_s=2) | ? | ? | ? | ? | ? | ? |
| fra-ov_resid_post_finegrid_gran1 | medical | grouped | 47.0±1.4 (n_s=2) | 20.8±9.3 (n_s=2) | ? | ? | ? | ? | ? | ? |
| fra-ov_resid_post_gran1 | base | per_feature | med 4.1 [2.0,10.9] top feat_F73164=19.1 | med 3.7 [2.0,4.9] top feat_F53037=9.6 | ? | ? | ? | ? | ? | ? |
| fra-ov_resid_post_gran1 | medical | per_feature | med 18.4 [13.8,24.3] top feat_F88683=42.7 | med 10.4 [8.7,15.2] top feat_F88683=26.6 | ? | ? | ? | ? | ? | ? |
| fra-ov_resid_post_gran10 | base | grouped | 4.3±0.3 (n_s=2) | 4.3±0.3 (n_s=2) | ? | ? | ? | ? | ? | ? |
| fra-ov_resid_post_gran10 | medical | grouped | 10.9±0.6 (n_s=2) | 9.9±1.2 (n_s=2) | ? | ? | ? | ? | ? | ? |
| fra-ov_resid_post_gran2 | base | grouped | 3.5±1.4 (n_s=2) | 3.5±1.4 (n_s=2) | ? | ? | ? | ? | ? | ? |
| fra-ov_resid_post_gran2 | medical | grouped | 18.3±4.4 (n_s=2) | 8.5±1.0 (n_s=2) | ? | ? | ? | ? | ? | ? |
| fra-ov_resid_post_gran50 | base | grouped | 6.6±1.5 (n_s=2) | 6.6±1.5 (n_s=2) | ? | ? | ? | ? | ? | ? |
| fra-ov_resid_post_gran50 | medical | grouped | 15.7±0.3 (n_s=2) | 11.0±0.8 (n_s=2) | ? | ? | ? | ? | ? | ? |
| fra-ov_resid_post_modeldiff_gran1 | base | per_feature | med 7.4 [4.1,12.8] top feat_F88683=34.1 | med 4.2 [2.9,6.3] top feat_F88683=13.3 | ? | ? | ? | ? | ? | ? |
| fra-ov_resid_post_modeldiff_gran1 | medical | per_feature | med 18.3 [15.4,24.0] top feat_F88683=42.8 | med 12.4 [7.2,16.8] top feat_F68827=26.8 | ? | ? | ? | ? | ? | ? |
| fra-qk_ln1_finegrid_gran1 | base | grouped | 28.6±1.5 (n_s=2) | 13.2±2.1 (n_s=2) | ? | ? | ? | ? | ? | ? |
| fra-qk_ln1_finegrid_gran1 | medical | grouped | 25.4±2.3 (n_s=2) | 22.7±0.0 (n_s=2) | ? | ? | ? | ? | ? | ? |
| fra-qk_ln1_gran1 | base | per_feature | med 1.3 [1.1,1.5] top feat_F2387=1.9 | med 1.3 [1.1,1.5] top feat_F2387=1.9 | ? | ? | ? | ? | ? | ? |
| fra-qk_ln1_gran1 | medical | per_feature | med 10.2 [7.9,14.9] top feat_F111743=30.5 | med 8.0 [6.6,10.0] top feat_F603=22.4 | ? | ? | ? | ? | ? | ? |
| fra-qk_ln1_gran10 | base | grouped | 0.9±0.1 (n_s=2) | 0.9±0.1 (n_s=2) | ? | ? | ? | ? | ? | ? |
| fra-qk_ln1_gran10 | medical | grouped | 11.7±7.5 (n_s=2) | 9.8±6.0 (n_s=2) | ? | ? | ? | ? | ? | ? |
| fra-qk_ln1_gran2 | base | grouped | 1.2±0.0 (n_s=2) | 1.2±0.0 (n_s=2) | ? | ? | ? | ? | ? | ? |
| fra-qk_ln1_gran2 | medical | grouped | 22.3±0.1 (n_s=2) | 21.5±0.8 (n_s=2) | ? | ? | ? | ? | ? | ? |
| fra-qk_ln1_gran50 | base | grouped | 1.1±0.0 (n_s=2) | 1.1±0.0 (n_s=2) | ? | ? | ? | ? | ? | ? |
| fra-qk_ln1_gran50 | medical | grouped | 8.5±2.5 (n_s=2) | 6.4±0.4 (n_s=2) | ? | ? | ? | ? | ? | ? |
| fra-qk_ln1_modeldiff_gran1 | base | per_feature | med 1.1 [1.0,1.3] top feat_F48821=1.8 | med 1.1 [1.0,1.3] top feat_F48821=1.8 | ? | ? | ? | ? | ? | ? |
| fra-qk_ln1_modeldiff_gran1 | medical | per_feature | med 8.6 [7.4,10.8] top feat_F64159=18.4 | med 5.9 [5.2,7.9] top feat_F64159=13.1 | ? | ? | ? | ? | ? | ? |
| frarouting_ov_to_ov_ln1_finegrid_gran1 | base | grouped | 8.4±0.2 (n_s=2) | 8.4±0.2 (n_s=2) | ? | ? | ? | ? | ? | ? |
| frarouting_ov_to_ov_ln1_finegrid_gran1 | medical | grouped | 31.1±0.2 (n_s=2) | 21.7±2.2 (n_s=2) | ? | ? | ? | ? | ? | ? |
| frarouting_ov_to_ov_ln1_gran1 | base | per_feature | med 1.2 [1.0,1.4] top feat_F648=1.9 | med 1.2 [1.0,1.4] top feat_F648=1.9 | ? | ? | ? | ? | ? | ? |
| frarouting_ov_to_ov_ln1_gran1 | medical | per_feature | med 7.5 [6.5,9.6] top feat_F603=18.6 | med 5.7 [4.0,7.0] top feat_F603=15.9 | ? | ? | ? | ? | ? | ? |
| frarouting_ov_to_ov_ln1_gran10 | base | grouped | 1.1±0.2 (n_s=2) | 1.1±0.2 (n_s=2) | ? | ? | ? | ? | ? | ? |
| frarouting_ov_to_ov_ln1_gran10 | medical | grouped | 8.0±1.5 (n_s=2) | 5.7±0.3 (n_s=2) | ? | ? | ? | ? | ? | ? |
| frarouting_ov_to_ov_ln1_gran2 | base | grouped | 1.5±0.3 (n_s=2) | 1.5±0.3 (n_s=2) | ? | ? | ? | ? | ? | ? |
| frarouting_ov_to_ov_ln1_gran2 | medical | grouped | 4.8±0.1 (n_s=2) | 2.6±2.3 (n_s=2) | ? | ? | ? | ? | ? | ? |
| frarouting_ov_to_ov_ln1_gran26 | base | grouped | 1.5±0.1 (n_s=2) | 1.5±0.1 (n_s=2) | ? | ? | ? | ? | ? | ? |
| frarouting_ov_to_ov_ln1_gran26 | medical | grouped | 9.9±3.2 (n_s=2) | 8.0±0.6 (n_s=2) | ? | ? | ? | ? | ? | ? |
| frarouting_qk_to_ov_ln1_finegrid_gran1 | base | grouped | 8.3±0.0 (n_s=2) | 8.3±0.0 (n_s=2) | ? | ? | ? | ? | ? | ? |
| frarouting_qk_to_ov_ln1_finegrid_gran1 | medical | grouped | 30.5±0.4 (n_s=2) | 21.7±2.2 (n_s=2) | ? | ? | ? | ? | ? | ? |
| frarouting_qk_to_ov_ln1_gran1 | base | per_feature | med 1.1 [1.0,1.2] top feat_F49256=1.5 | med 1.1 [1.0,1.2] top feat_F49256=1.5 | ? | ? | ? | ? | ? | ? |
| frarouting_qk_to_ov_ln1_gran1 | medical | per_feature | med 5.9 [5.0,6.2] top feat_F603=19.5 | med 4.5 [3.8,5.1] top feat_F603=16.2 | ? | ? | ? | ? | ? | ? |
| frarouting_qk_to_ov_ln1_gran10 | base | grouped | 1.1±0.2 (n_s=2) | 1.1±0.2 (n_s=2) | ? | ? | ? | ? | ? | ? |
| frarouting_qk_to_ov_ln1_gran10 | medical | grouped | 9.8±2.3 (n_s=2) | 9.8±2.3 (n_s=2) | ? | ? | ? | ? | ? | ? |
| frarouting_qk_to_ov_ln1_gran2 | base | grouped | 1.2±0.3 (n_s=2) | 1.2±0.3 (n_s=2) | ? | ? | ? | ? | ? | ? |
| frarouting_qk_to_ov_ln1_gran2 | medical | grouped | 17.8±2.2 (n_s=2) | 13.8±0.4 (n_s=2) | ? | ? | ? | ? | ? | ? |
| frarouting_qk_to_ov_ln1_gran26 | base | grouped | 1.4±0.7 (n_s=2) | 1.4±0.7 (n_s=2) | ? | ? | ? | ? | ? | ? |
| frarouting_qk_to_ov_ln1_gran26 | medical | grouped | 7.1±1.7 (n_s=2) | 6.2±3.0 (n_s=2) | ? | ? | ? | ? | ? | ? |
| frarouting_qk_to_qk_ln1_finegrid_gran1 | base | grouped | 1.6±0.8 (n_s=2) | 1.6±0.8 (n_s=2) | ? | ? | ? | ? | ? | ? |
| frarouting_qk_to_qk_ln1_finegrid_gran1 | medical | grouped | 9.5±2.4 (n_s=2) | 5.8±2.0 (n_s=2) | ? | ? | ? | ? | ? | ? |
| frarouting_qk_to_qk_ln1_gran1 | base | per_feature | med 1.0 [0.9,1.1] top feat_F2387=1.3 | med 1.0 [0.9,1.1] top feat_F2387=1.3 | ? | ? | ? | ? | ? | ? |
| frarouting_qk_to_qk_ln1_gran1 | medical | per_feature | med 5.1 [4.5,5.6] top feat_F115462=8.4 | med 4.5 [3.9,5.0] top feat_F52417=8.4 | ? | ? | ? | ? | ? | ? |
| frarouting_qk_to_qk_ln1_gran10 | base | grouped | 0.9±0.1 (n_s=2) | 0.9±0.1 (n_s=2) | ? | ? | ? | ? | ? | ? |
| frarouting_qk_to_qk_ln1_gran10 | medical | grouped | 5.1±3.4 (n_s=2) | 4.8±3.9 (n_s=2) | ? | ? | ? | ? | ? | ? |
| frarouting_qk_to_qk_ln1_gran2 | base | grouped | 1.0±0.3 (n_s=2) | 1.0±0.3 (n_s=2) | ? | ? | ? | ? | ? | ? |
| frarouting_qk_to_qk_ln1_gran2 | medical | grouped | 5.6±2.4 (n_s=2) | 5.1±1.7 (n_s=2) | ? | ? | ? | ? | ? | ? |
| frarouting_qk_to_qk_ln1_gran26 | base | grouped | 1.0±0.1 (n_s=2) | 1.0±0.1 (n_s=2) | ? | ? | ? | ? | ? | ? |
| frarouting_qk_to_qk_ln1_gran26 | medical | grouped | 7.1±1.2 (n_s=2) | 5.9±0.4 (n_s=2) | ? | ? | ? | ? | ? | ? |
| smoke_fra-ov_ln1_gran1 | medical | per_feature | med 6.4 [4.1,9.2] top feat_F603=16.7 | med 2.2 [0.0,5.5] top feat_F603=16.7 | ? | ? | ? | ? | ? | ? |
| wang_ln1_finegrid_gran1 | base | grouped | 3.8±0.4 (n_s=2) | 3.8±0.4 (n_s=2) | ? | ? | ? | ? | ? | ? |
| wang_ln1_finegrid_gran1 | medical | grouped | 38.6±0.7 (n_s=2) | 29.0±0.1 (n_s=2) | ? | ? | ? | ? | ? | ? |
| wang_ln1_gran1 | base | per_feature | med 1.3 [1.1,1.5] top feat_F85397=3.0 | med 1.3 [1.1,1.5] top feat_F85397=3.0 | ? | ? | ? | ? | ? | ? |
| wang_ln1_gran1 | medical | per_feature | med 13.2 [9.8,17.0] top feat_F111743=30.4 | med 8.8 [7.1,13.8] top feat_F127645=22.3 | ? | ? | ? | ? | ? | ? |
| wang_ln1_gran10 | base | grouped | 1.7±0.4 (n_s=2) | 1.7±0.4 (n_s=2) | ? | ? | ? | ? | ? | ? |
| wang_ln1_gran10 | medical | grouped | 16.1±0.4 (n_s=2) | 12.3±1.4 (n_s=2) | ? | ? | ? | ? | ? | ? |
| wang_ln1_gran2 | base | grouped | 0.9±0.3 (n_s=2) | 0.9±0.3 (n_s=2) | ? | ? | ? | ? | ? | ? |
| wang_ln1_gran2 | medical | grouped | 7.8±3.1 (n_s=2) | 3.4±4.3 (n_s=2) | ? | ? | ? | ? | ? | ? |
| wang_ln1_gran50 | base | grouped | 1.4±0.0 (n_s=2) | 1.4±0.0 (n_s=2) | ? | ? | ? | ? | ? | ? |
| wang_ln1_gran50 | medical | grouped | 16.5±7.8 (n_s=2) | 13.4±3.5 (n_s=2) | ? | ? | ? | ? | ? | ? |
| wang_ln1_modeldiff_gran1 | base | per_feature | med 1.2 [1.1,1.4] top feat_F118260=2.6 | med 1.2 [1.1,1.4] top feat_F118260=2.6 | ? | ? | ? | ? | ? | ? |
| wang_ln1_modeldiff_gran1 | medical | per_feature | med 11.3 [8.8,15.4] top feat_F118260=26.4 | med 8.6 [6.1,12.1] top feat_F603=22.2 | ? | ? | ? | ? | ? | ? |

## 2×2 — attribution × diff-definition (finance, gran1, per-feature)

Same Δ statistic in every quadrant: reading DOWN a column isolates attribution; ACROSS a row isolates diff-definition.

| attribution \ diff | outcome-bucket | model-identity (finance−base) |
|---|---|---|
| Wang enc-f | (pending) | old-Wang (separate dataset; EM−base no-coh) |
| FRA OV | (pending) | (pending) |
| FRA QK | (pending) | (pending) |

## RANKING-CONFIDENCE CAVEAT (read before interpreting any Δ)

**Bucketed-diff cells (main 6 protocols + Variant A wang_bucketdiff):** both models fell back to the §1a tercile split (NOT strict align≤30/>70). LOW statistical power — finance has a thin coherent sample; base is ~uniformly aligned (weak control by construction).

**Variant B (modeldiff, model-identity finance−base):** UNBUCKETED — one model-agnostic ranking, no coherence gate, per-model own-weights decomposition. The thin-bucket caveat does NOT apply to Variant B.

Per-cell ranking provenance:


## RANKING STABILITY — is the diff ranking a thin-bucket artifact?

Full analysis: `experiments/fra_14b_diff/STABILITY.md` (bucket-size table + membership ranks across powerings). Trusted powering: resample@coh70 (misal 33 / align 29, both well-powered). Decisive points:

- **QK ranks F603 #1 robustly** — top-1 in all three powerings (current tercile, coh50 rebucket, resample@coh70). NOT a thin-bucket artifact.
- **OV *score-order* top-1 IS bucket-sensitive**: current F59432 → coh50 F70850 → resample F98722. So the OV #1-by-ΔOV is the thin-tercile artifact; F98722/F112720 are the proper-powered OV score leaders.
- **But F603 is a robust OV-SET member** (rank 16/3/20 across powerings, never absent) and was the best *steerer* (max Δ@50) in BOTH OV-bucketdiff (48.6) and OV-modeldiff (48.4) — so F603-as-best-steerer is not luck.
- **Bottom line:** F603 is the robust best-steerer across QK-rank + OV-modeldiff; only the OV score-ORDER top-1 was the artifact. This is why F603 dominates 4/5 of the 2×2 quadrants above.
- **F98722 re-steer (resample-OV top-1):** pending — will confirm whether proper-powered OV's top feature out-steers F603 (48.6).
