# fra_14b_diff GRID_RESULTS  (57 combined cells on HF)

## Δalign@coh per cell

`n_nz` = non-zero score entries; for QK cells these are PAIRS (≤50, tagged ·pair), for OV/wang they are FEATURES. `n_feat` = unique feature ids actually steered (QK harvests uniques from the top-50 pairs → 22-24, by design).

| ranking·sae·gran | model | kind | Δ@50 | Δ@70 | bucket_mode | |B_mis| | |B_aln| | score_spread | n_nz | n_feat |
|---|---|---|---|---|---|---|---|---|---|---|
| fra-ov_ln1_gran1 | base | per_feature | med 1.4 [1.2,1.8] top feat_F108273=3.0 | med 1.4 [1.2,1.8] top feat_F108273=3.0 | ? | ? | ? | ? | ? | ? |
| fra-ov_ln1_gran1 | sports | per_feature | med 19.7 [16.7,24.5] top feat_F603=42.0 | med 6.0 [2.9,9.6] top feat_F603=24.4 | ? | ? | ? | ? | ? | ? |
| fra-ov_ln1_gran10 | base | grouped | 1.1±0.2 (n_s=2) | 1.1±0.2 (n_s=2) | ? | ? | ? | ? | ? | ? |
| fra-ov_ln1_gran10 | sports | grouped | 23.5±0.1 (n_s=2) | 4.8±6.7 (n_s=2) | ? | ? | ? | ? | ? | ? |
| fra-ov_ln1_gran2 | base | grouped | 1.5±0.6 (n_s=2) | 1.5±0.6 (n_s=2) | ? | ? | ? | ? | ? | ? |
| fra-ov_ln1_gran2 | sports | grouped | 23.5±2.8 (n_s=2) | 11.3±0.8 (n_s=2) | ? | ? | ? | ? | ? | ? |
| fra-ov_ln1_gran50 | base | grouped | 1.4±0.0 (n_s=2) | 1.4±0.0 (n_s=2) | ? | ? | ? | ? | ? | ? |
| fra-ov_ln1_gran50 | sports | grouped | 28.1±1.8 (n_s=2) | 9.8±13.9 (n_s=2) | ? | ? | ? | ? | ? | ? |
| fra-ov_resid_post_gran1 | base | per_feature | med 7.6 [3.4,12.7] top feat_F46478=29.2 | med 3.5 [2.5,5.6] top feat_F89696=10.0 | ? | ? | ? | ? | ? | ? |
| fra-ov_resid_post_gran1 | sports | per_feature | med 22.9 [15.8,27.1] top feat_F88683=56.4 | med 4.1 [0.1,8.4] top feat_F113532=15.2 | ? | ? | ? | ? | ? | ? |
| fra-ov_resid_post_gran10 | base | grouped | 7.1±2.3 (n_s=2) | 7.1±2.3 (n_s=2) | ? | ? | ? | ? | ? | ? |
| fra-ov_resid_post_gran10 | sports | grouped | 21.3±4.1 (n_s=2) | 7.8±0.0 (n_s=1) | ? | ? | ? | ? | ? | ? |
| fra-ov_resid_post_gran2 | base | grouped | 7.6±3.6 (n_s=2) | 7.6±3.6 (n_s=2) | ? | ? | ? | ? | ? | ? |
| fra-ov_resid_post_gran2 | sports | grouped | 13.3±4.2 (n_s=2) | 0.0±0.0 (n_s=1) | ? | ? | ? | ? | ? | ? |
| fra-ov_resid_post_gran50 | base | grouped | 11.1±3.5 (n_s=2) | 11.1±3.5 (n_s=2) | ? | ? | ? | ? | ? | ? |
| fra-ov_resid_post_gran50 | sports | grouped | 38.2±5.0 (n_s=2) | 18.4±3.3 (n_s=2) | ? | ? | ? | ? | ? | ? |
| fra-qk_ln1_gran1 | base | per_feature | med 1.2 [1.2,1.6] top feat_F108273=2.7 | med 1.2 [1.2,1.6] top feat_F108273=2.7 | ? | ? | ? | ? | ? | ? |
| fra-qk_ln1_gran1 | sports | per_feature | med 15.2 [12.5,18.0] top feat_F603=42.5 | med 6.8 [3.4,8.0] top feat_F603=24.5 | ? | ? | ? | ? | ? | ? |
| fra-qk_ln1_gran10 | base | grouped | 1.4±0.7 (n_s=2) | 1.4±0.7 (n_s=2) | ? | ? | ? | ? | ? | ? |
| fra-qk_ln1_gran10 | sports | grouped | 20.8±0.7 (n_s=2) | 13.1±4.4 (n_s=2) | ? | ? | ? | ? | ? | ? |
| fra-qk_ln1_gran2 | base | grouped | 1.5±0.3 (n_s=2) | 1.5±0.3 (n_s=2) | ? | ? | ? | ? | ? | ? |
| fra-qk_ln1_gran2 | sports | grouped | 36.7±0.2 (n_s=2) | 26.1±4.9 (n_s=2) | ? | ? | ? | ? | ? | ? |
| fra-qk_ln1_gran50 | base | grouped | 1.2±0.6 (n_s=2) | 1.2±0.6 (n_s=2) | ? | ? | ? | ? | ? | ? |
| fra-qk_ln1_gran50 | sports | grouped | 25.9±0.2 (n_s=2) | 15.5±9.9 (n_s=2) | ? | ? | ? | ? | ? | ? |
| frarouting_ov_to_ov_ln1_gran1 | base | per_feature | med 1.2 [1.1,1.4] top feat_F41787=1.9 | med 1.2 [1.1,1.4] top feat_F41787=1.9 | ? | ? | ? | ? | ? | ? |
| frarouting_ov_to_ov_ln1_gran1 | sports | per_feature | med 10.3 [8.5,12.6] top feat_F2837=20.9 | med 1.7 [0.0,5.8] top feat_F69749=13.1 | ? | ? | ? | ? | ? | ? |
| frarouting_ov_to_ov_ln1_gran10 | base | grouped | 1.4±0.9 (n_s=2) | 1.4±0.9 (n_s=2) | ? | ? | ? | ? | ? | ? |
| frarouting_ov_to_ov_ln1_gran10 | sports | grouped | 18.0±2.7 (n_s=2) | 13.1±0.0 (n_s=1) | ? | ? | ? | ? | ? | ? |
| frarouting_ov_to_ov_ln1_gran2 | base | grouped | 1.5±0.3 (n_s=2) | 1.5±0.3 (n_s=2) | ? | ? | ? | ? | ? | ? |
| frarouting_ov_to_ov_ln1_gran2 | sports | grouped | 12.8±2.2 (n_s=2) | 2.1±3.0 (n_s=2) | ? | ? | ? | ? | ? | ? |
| frarouting_ov_to_ov_ln1_gran50 | base | grouped | 1.2±0.3 (n_s=2) | 1.2±0.3 (n_s=2) | ? | ? | ? | ? | ? | ? |
| frarouting_ov_to_ov_ln1_gran50 | sports | grouped | 13.6±4.2 (n_s=2) | 0.0±0.0 (n_s=1) | ? | ? | ? | ? | ? | ? |
| frarouting_qk_to_ov_ln1_gran1 | base | per_feature | med 1.1 [1.0,1.3] top feat_F108273=1.8 | med 1.1 [1.0,1.3] top feat_F108273=1.8 | ? | ? | ? | ? | ? | ? |
| frarouting_qk_to_ov_ln1_gran1 | sports | per_feature | med 6.5 [6.0,9.0] top feat_F603=17.3 | med 5.6 [3.7,6.0] top feat_F603=6.3 | ? | ? | ? | ? | ? | ? |
| frarouting_qk_to_ov_ln1_gran10 | base | grouped | 1.2±0.7 (n_s=2) | 1.2±0.7 (n_s=2) | ? | ? | ? | ? | ? | ? |
| frarouting_qk_to_ov_ln1_gran10 | sports | grouped | 11.1±0.2 (n_s=2) | 4.1±0.0 (n_s=1) | ? | ? | ? | ? | ? | ? |
| frarouting_qk_to_ov_ln1_gran2 | base | grouped | 1.2±0.9 (n_s=2) | 1.2±0.9 (n_s=2) | ? | ? | ? | ? | ? | ? |
| frarouting_qk_to_ov_ln1_gran2 | sports | grouped | 14.6±1.7 (n_s=2) | 3.4±4.9 (n_s=2) | ? | ? | ? | ? | ? | ? |
| frarouting_qk_to_ov_ln1_gran50 | base | grouped | 0.9±0.0 (n_s=2) | 0.9±0.0 (n_s=2) | ? | ? | ? | ? | ? | ? |
| frarouting_qk_to_ov_ln1_gran50 | sports | grouped | 10.2±2.4 (n_s=2) | 0.5±0.8 (n_s=2) | ? | ? | ? | ? | ? | ? |
| frarouting_qk_to_qk_ln1_gran1 | base | per_feature | med 1.0 [0.9,1.1] top feat_F92691=1.2 | med 1.0 [0.9,1.1] top feat_F92691=1.2 | ? | ? | ? | ? | ? | ? |
| frarouting_qk_to_qk_ln1_gran1 | sports | per_feature | med 6.1 [5.2,6.9] top feat_F64159=7.9 | med 0.0 [0.0,0.0] top feat_F111355=0.0 | ? | ? | ? | ? | ? | ? |
| frarouting_qk_to_qk_ln1_gran10 | base | grouped | 1.2±0.9 (n_s=2) | 1.2±0.9 (n_s=2) | ? | ? | ? | ? | ? | ? |
| frarouting_qk_to_qk_ln1_gran10 | sports | grouped | 5.9±2.5 (n_s=2) | — | ? | ? | ? | ? | ? | ? |
| frarouting_qk_to_qk_ln1_gran2 | base | grouped | 1.2±0.7 (n_s=2) | 1.2±0.7 (n_s=2) | ? | ? | ? | ? | ? | ? |
| frarouting_qk_to_qk_ln1_gran2 | sports | grouped | 5.9±0.9 (n_s=2) | — | ? | ? | ? | ? | ? | ? |
| frarouting_qk_to_qk_ln1_gran50 | base | grouped | 1.5±0.3 (n_s=2) | 1.5±0.3 (n_s=2) | ? | ? | ? | ? | ? | ? |
| frarouting_qk_to_qk_ln1_gran50 | sports | grouped | 6.6±0.4 (n_s=2) | — | ? | ? | ? | ? | ? | ? |
| smoke_fra-ov_ln1_gran1 | sports | per_feature | med 10.0 [7.3,12.4] top feat_F698=25.8 | med 0.0 [0.0,0.0] top feat_F5786=6.6 | ? | ? | ? | ? | ? | ? |
| wang_ln1_gran1 | base | per_feature | med 1.3 [1.2,1.5] top feat_F248=3.0 | med 1.3 [1.2,1.5] top feat_F248=3.0 | ? | ? | ? | ? | ? | ? |
| wang_ln1_gran1 | sports | per_feature | med 17.0 [14.3,20.0] top feat_F603=41.9 | med 5.6 [3.0,8.6] top feat_F103059=27.6 | ? | ? | ? | ? | ? | ? |
| wang_ln1_gran10 | base | grouped | 2.2±0.9 (n_s=2) | 2.2±0.9 (n_s=2) | ? | ? | ? | ? | ? | ? |
| wang_ln1_gran10 | sports | grouped | 27.8±1.3 (n_s=2) | 13.5±8.9 (n_s=2) | ? | ? | ? | ? | ? | ? |
| wang_ln1_gran2 | base | grouped | 1.1±0.0 (n_s=2) | 1.1±0.0 (n_s=2) | ? | ? | ? | ? | ? | ? |
| wang_ln1_gran2 | sports | grouped | 11.7±2.2 (n_s=2) | 5.2±7.3 (n_s=2) | ? | ? | ? | ? | ? | ? |
| wang_ln1_gran50 | base | grouped | 0.8±0.4 (n_s=2) | 0.8±0.4 (n_s=2) | ? | ? | ? | ? | ? | ? |
| wang_ln1_gran50 | sports | grouped | 23.3±2.2 (n_s=2) | 10.0±10.2 (n_s=2) | ? | ? | ? | ? | ? | ? |

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
