# fra_14b_diff — GRID_RESULTS

**HEADLINE.** The 2×2 (attribution {Wang enc-f, FRA-OV, FRA-QK} × diff-definition
{outcome-bucket, model-identity}) was built to locate the FRA-vs-Wang steering
gap. The answer: **the gap is NEITHER attribution NOR diff-definition — all
quadrants broadly agree** (finance gran1 Δ@50 ≈ 11–17) and **converge on the same
feature F603** (top in 4 of 5 landed quadrants). The ranking-stability check
confirms **F603 is the robust best-steerer** (across QK-rank and OV-modeldiff);
the only artifact was the OV *score-order* top-1 (F59432), which is bucket-sensitive
and whose proper-powered successor F98722 does NOT beat F603 when re-steered
(12.4 vs 48.6). Mechanism: OV-routed F603 reaches ~70% of additive F603's Δ@50
(35.1 vs 50.5 at α=±5) — genuine mechanism dilution, not an α-scale gap.

**Note on resid_post.** The FRA-OV×resid_post bench (n_s=2: finance Δ@50 med 21.2
top F68070; base med 13.6 top F73164) is a DISTINCT SAE/hookpoint from the ln1
F603 story — its top features (F68070/F73164) differ, as expected for a different
SAE. It is reported as a separate row, not part of the ln1 2×2.

Full ranking-stability analysis: [STABILITY.md](STABILITY.md). Plot-ready fine-grid
trajectories (cell×model × 41 α points): `qwen14b/grid_diff/*_finegrid/` on HF.

**Provenance notes.** (1) Some FINANCE grouped/finegrid rows show `n_s=1` (only
seed42 in the combine — grouped grans ran single-seed; the additive finegrids'
finance seed123 streams were still landing at assembly). Values are correct for the
seeds present. (2) Single-feature re-steer cells (`*_finegrid`) show `?` in the
ranking-provenance columns — expected: they steer one fixed feature, no ranking-meta.
Δ@70 shown as `—` where the coherent α-window is empty (coherence collapses at the
steered extreme).

## Δalign@coh per cell

`n_nz` = non-zero score entries; for QK cells these are PAIRS (≤50, tagged ·pair), for OV/wang they are FEATURES. `n_feat` = unique feature ids actually steered (QK harvests uniques from the top-50 pairs → 22-24, by design).

| ranking·sae·gran | model | kind | Δ@50 | Δ@70 | bucket_mode | |B_mis| | |B_aln| | score_spread | n_nz | n_feat |
|---|---|---|---|---|---|---|---|---|---|---|
| f56776_finegrid/fra-ov_resid_post_gran1 | base | grouped | 13.5±2.5 (n_s=2) | 9.1±3.6 (n_s=2) | ? | ? | ? | ? | ? | ? |
| f56776_finegrid/fra-ov_resid_post_gran1 | finance | grouped | 49.2±0.0 (n_s=1) | 16.6±0.0 (n_s=1) | ? | ? | ? | ? | ? | ? |
| f603_finegrid/fra-ov_ln1_gran1 | base | grouped | 31.3±1.9 (n_s=2) | 16.4±2.7 (n_s=2) | ? | ? | ? | ? | ? | ? |
| f603_finegrid/fra-ov_ln1_gran1 | finance | grouped | 50.5±0.0 (n_s=1) | 26.9±0.0 (n_s=1) | ? | ? | ? | ? | ? | ? |
| f93118_finegrid/fra-ov_resid_post_gran1 | base | grouped | 58.8±8.2 (n_s=2) | 28.1±3.8 (n_s=2) | ? | ? | ? | ? | ? | ? |
| f93118_finegrid/fra-ov_resid_post_gran1 | finance | grouped | 56.2±0.0 (n_s=1) | 17.8±0.0 (n_s=1) | ? | ? | ? | ? | ? | ? |
| f98722_finegrid/fra-ov_ln1_gran1 | base | grouped | 0.7±0.1 (n_s=2) | 0.7±0.1 (n_s=2) | ? | ? | ? | ? | ? | ? |
| f98722_finegrid/fra-ov_ln1_gran1 | finance | grouped | 12.4±5.9 (n_s=2) | — | ? | ? | ? | ? | ? | ? |
| fra-ov_ln1_gran1 | base | per_feature | med 1.4 [1.1,1.8] top feat_F107490=2.8 | med 1.4 [1.1,1.8] top feat_F107490=2.8 | tercile_fallback | 31 | 31 | 6.00 | 50·feature | 50 |
| fra-ov_ln1_gran1 | finance | per_feature | med 16.2 [13.3,19.3] top feat_F603=48.6 | med 0.0 [0.0,5.3] top feat_F603=26.7 | tercile_fallback | 8 | 8 | 7.07 | 50·feature | 50 |
| fra-ov_ln1_gran10 | base | grouped | 1.6±0.0 (n_s=1) | 1.6±0.0 (n_s=1) | tercile_fallback | 31 | 31 | 6.00 | 50·feature | 50 |
| fra-ov_ln1_gran10 | finance | grouped | 20.5±0.0 (n_s=1) | 0.0±0.0 (n_s=1) | tercile_fallback | 8 | 8 | 7.07 | 50·feature | 50 |
| fra-ov_ln1_gran2 | base | grouped | 1.1±0.0 (n_s=1) | 1.1±0.0 (n_s=1) | tercile_fallback | 31 | 31 | 6.00 | 50·feature | 50 |
| fra-ov_ln1_gran2 | finance | grouped | 28.0±0.0 (n_s=1) | — | tercile_fallback | 8 | 8 | 7.07 | 50·feature | 50 |
| fra-ov_ln1_gran50 | base | grouped | 1.2±0.0 (n_s=1) | 1.2±0.0 (n_s=1) | tercile_fallback | 31 | 31 | 6.00 | 50·feature | 50 |
| fra-ov_ln1_gran50 | finance | grouped | 21.6±0.0 (n_s=1) | 4.4±0.0 (n_s=1) | tercile_fallback | 8 | 8 | 7.07 | 50·feature | 50 |
| fra-ov_ln1_modeldiff_gran1 | finance | per_feature | med 17.0 [14.9,22.8] top feat_F603=48.4 | med 0.0 [0.0,8.4] top feat_F603=26.9 | modeldiff | — | — | 1.44 | 50·feature | 50 |
| fra-ov_resid_post_gran1 | base | per_feature | med 13.6 [8.9,19.2] top feat_F73164=27.0 | med 5.2 [3.8,7.0] top feat_F89696=12.0 | tercile_fallback | 31 | 31 | 25.20 | 50·feature | 50 |
| fra-ov_resid_post_gran1 | finance | per_feature | med 21.2 [16.9,26.9] top feat_F68070=33.7 | med 0.0 [0.0,3.0] top feat_F38804=7.3 | tercile_fallback | 8 | 8 | 9.36 | 50·feature | 50 |
| fra-ov_resid_post_gran10 | base | grouped | 13.4±0.0 (n_s=1) | 10.2±0.0 (n_s=1) | tercile_fallback | 31 | 31 | 25.20 | 50·feature | 50 |
| fra-ov_resid_post_gran2 | base | grouped | 21.6±0.0 (n_s=1) | 7.5±0.0 (n_s=1) | tercile_fallback | 31 | 31 | 25.20 | 50·feature | 50 |
| fra-ov_resid_post_gran50 | base | grouped | 9.2±0.0 (n_s=1) | 2.0±0.0 (n_s=1) | tercile_fallback | 31 | 31 | 25.20 | 50·feature | 50 |
| fra-qk_ln1_gran1 | base | per_feature | med 1.2 [1.1,1.6] top feat_F108273=2.3 | med 1.2 [1.1,1.6] top feat_F108273=2.3 | tercile_fallback | 31 | 31 | 513.74 | 50·pair | 24 |
| fra-qk_ln1_gran1 | finance | per_feature | med 13.2 [11.3,14.8] top feat_F603=49.7 | med 12.9 [6.4,19.3] top feat_F603=25.8 | tercile_fallback | 8 | 8 | 193.02 | 50·pair | 22 |
| fra-qk_ln1_gran10 | base | grouped | 1.7±0.0 (n_s=1) | 1.7±0.0 (n_s=1) | tercile_fallback | 31 | 31 | 513.74 | 50·pair | 24 |
| fra-qk_ln1_gran10 | finance | grouped | 20.8±0.0 (n_s=1) | — | tercile_fallback | 8 | 8 | 193.02 | 50·pair | 22 |
| fra-qk_ln1_gran2 | base | grouped | 1.6±0.0 (n_s=1) | 1.6±0.0 (n_s=1) | tercile_fallback | 31 | 31 | 513.74 | 50·pair | 24 |
| fra-qk_ln1_gran2 | finance | grouped | 33.4±0.0 (n_s=1) | 6.7±0.0 (n_s=1) | tercile_fallback | 8 | 8 | 193.02 | 50·pair | 22 |
| fra-qk_ln1_gran50 | base | grouped | 1.2±0.0 (n_s=1) | 1.2±0.0 (n_s=1) | tercile_fallback | 31 | 31 | 513.74 | 50·pair | 24 |
| fra-qk_ln1_gran50 | finance | grouped | 10.3±0.0 (n_s=1) | — | tercile_fallback | 8 | 8 | 193.02 | 50·pair | 22 |
| fra-qk_ln1_modeldiff_gran1 | finance | per_feature | med 11.6 [9.5,14.1] top feat_F9769=18.6 | — | modeldiff | — | — | 8971.81 | 50·pair | 18 |
| frarouting_ov_to_ov_ln1_gran1 | base | per_feature | med 1.4 [1.1,1.6] top feat_F51220=2.2 | med 1.4 [1.1,1.6] top feat_F51220=2.2 | tercile_fallback | 31 | 31 | 6.00 | 50·feature | 50 |
| frarouting_ov_to_ov_ln1_gran1 | finance | per_feature | med 8.4 [6.6,10.7] top feat_F77764=23.9 | med 0.0 [0.0,0.0] top feat_F77764=0.0 | tercile_fallback | 8 | 8 | 7.07 | 50·feature | 50 |
| frarouting_ov_to_ov_ln1_gran10 | base | grouped | 1.4±0.0 (n_s=1) | 1.4±0.0 (n_s=1) | tercile_fallback | 31 | 31 | 6.00 | 50·feature | 50 |
| frarouting_ov_to_ov_ln1_gran10 | finance | grouped | 15.6±0.0 (n_s=1) | — | tercile_fallback | 8 | 8 | 7.07 | 50·feature | 50 |
| frarouting_ov_to_ov_ln1_gran2 | base | grouped | 1.4±0.0 (n_s=1) | 1.4±0.0 (n_s=1) | tercile_fallback | 31 | 31 | 6.00 | 50·feature | 50 |
| frarouting_ov_to_ov_ln1_gran2 | finance | grouped | 6.6±0.0 (n_s=1) | — | tercile_fallback | 8 | 8 | 7.07 | 50·feature | 50 |
| frarouting_ov_to_ov_ln1_gran26 | base | grouped | 1.4±0.0 (n_s=1) | 1.4±0.0 (n_s=1) | tercile_fallback | 31 | 31 | 6.00 | 50·feature | 50 |
| frarouting_ov_to_ov_ln1_gran26 | finance | grouped | 10.2±0.0 (n_s=1) | — | tercile_fallback | 8 | 8 | 7.07 | 50·feature | 50 |
| frarouting_qk_to_ov_ln1_gran1 | base | per_feature | med 1.2 [1.1,1.4] top feat_F21208=1.7 | med 1.2 [1.1,1.4] top feat_F21208=1.7 | tercile_fallback | 31 | 31 | 513.74 | 50·pair | 24 |
| frarouting_qk_to_ov_ln1_gran1 | finance | per_feature | med 6.5 [4.7,8.3] top feat_F603=19.1 | — | tercile_fallback | 8 | 8 | 193.02 | 50·pair | 22 |
| frarouting_qk_to_ov_ln1_gran10 | base | grouped | 1.2±0.0 (n_s=1) | 1.2±0.0 (n_s=1) | tercile_fallback | 31 | 31 | 513.74 | 50·pair | 24 |
| frarouting_qk_to_ov_ln1_gran10 | finance | grouped | 7.2±0.0 (n_s=1) | — | tercile_fallback | 8 | 8 | 193.02 | 50·pair | 22 |
| frarouting_qk_to_ov_ln1_gran2 | base | grouped | 1.6±0.0 (n_s=1) | 1.6±0.0 (n_s=1) | tercile_fallback | 31 | 31 | 513.74 | 50·pair | 24 |
| frarouting_qk_to_ov_ln1_gran2 | finance | grouped | 15.9±0.0 (n_s=1) | — | tercile_fallback | 8 | 8 | 193.02 | 50·pair | 22 |
| frarouting_qk_to_ov_ln1_gran26 | base | grouped | 1.1±0.0 (n_s=1) | 1.1±0.0 (n_s=1) | tercile_fallback | 31 | 31 | 513.74 | 50·pair | 24 |
| frarouting_qk_to_ov_ln1_gran26 | finance | grouped | 4.8±0.0 (n_s=1) | — | tercile_fallback | 8 | 8 | 193.02 | 50·pair | 22 |
| frarouting_qk_to_qk_ln1_gran1 | base | per_feature | med 1.2 [1.1,1.4] top feat_F21208=2.0 | med 1.2 [1.1,1.4] top feat_F21208=2.0 | tercile_fallback | 31 | 31 | 513.74 | 50·pair | 24 |
| frarouting_qk_to_qk_ln1_gran1 | finance | per_feature | med 5.5 [3.9,6.6] top feat_F8862=8.3 | — | tercile_fallback | 8 | 8 | 193.02 | 50·pair | 22 |
| frarouting_qk_to_qk_ln1_gran10 | base | grouped | 0.9±0.0 (n_s=1) | 0.9±0.0 (n_s=1) | tercile_fallback | 31 | 31 | 513.74 | 50·pair | 24 |
| frarouting_qk_to_qk_ln1_gran10 | finance | grouped | 4.5±0.0 (n_s=1) | — | tercile_fallback | 8 | 8 | 193.02 | 50·pair | 22 |
| frarouting_qk_to_qk_ln1_gran2 | base | grouped | 1.1±0.0 (n_s=1) | 1.1±0.0 (n_s=1) | tercile_fallback | 31 | 31 | 513.74 | 50·pair | 24 |
| frarouting_qk_to_qk_ln1_gran2 | finance | grouped | 7.0±0.0 (n_s=1) | — | tercile_fallback | 8 | 8 | 193.02 | 50·pair | 22 |
| frarouting_qk_to_qk_ln1_gran26 | base | grouped | 0.9±0.0 (n_s=1) | 0.9±0.0 (n_s=1) | tercile_fallback | 31 | 31 | 513.74 | 50·pair | 24 |
| frarouting_qk_to_qk_ln1_gran26 | finance | grouped | 5.2±1.5 (n_s=2) | — | tercile_fallback | 8 | 8 | 193.02 | 50·pair | 22 |
| ov_to_ov_finegrid/frarouting_ov_to_ov_ln1_gran1 | base | grouped | 18.6±0.2 (n_s=2) | 9.3±0.6 (n_s=2) | ? | ? | ? | ? | ? | ? |
| ov_to_ov_finegrid/frarouting_ov_to_ov_ln1_gran1 | finance | grouped | 39.1±0.2 (n_s=2) | 6.1±0.7 (n_s=2) | ? | ? | ? | ? | ? | ? |
| qk_to_ov_finegrid/frarouting_qk_to_ov_ln1_gran1 | base | grouped | 10.9±0.7 (n_s=2) | 10.9±0.7 (n_s=2) | ? | ? | ? | ? | ? | ? |
| qk_to_ov_finegrid/frarouting_qk_to_ov_ln1_gran1 | finance | grouped | 35.1±3.6 (n_s=2) | 14.8±5.1 (n_s=2) | ? | ? | ? | ? | ? | ? |
| qk_to_qk_finegrid/frarouting_qk_to_qk_ln1_gran1 | base | grouped | 1.6±0.1 (n_s=2) | 1.6±0.1 (n_s=2) | ? | ? | ? | ? | ? | ? |
| qk_to_qk_finegrid/frarouting_qk_to_qk_ln1_gran1 | finance | grouped | 9.1±0.1 (n_s=2) | — | ? | ? | ? | ? | ? | ? |
| wang_ln1_bucketdiff_gran1 | finance | per_feature | med 13.9 [10.0,19.4] top feat_F603=48.1 | med 0.0 [0.0,0.0] top feat_F603=26.9 | tercile_fallback | 8 | 8 | 0.65 | 50·feature | 50 |

## 2×2 — attribution × diff-definition (finance, gran1, per-feature)

Same Δ statistic in every quadrant: reading DOWN a column isolates attribution; ACROSS a row isolates diff-definition.

| attribution \ diff | outcome-bucket | model-identity (finance−base) |
|---|---|---|
| Wang enc-f | Δ@50 med 13.9 top feat_F603=48.1 | Δ@70 med 0.0 | old-Wang (separate dataset; EM−base no-coh) |
| FRA OV | Δ@50 med 16.2 top feat_F603=48.6 | Δ@70 med 0.0 | Δ@50 med 17.0 top feat_F603=48.4 | Δ@70 med 0.0 |
| FRA QK | Δ@50 med 13.2 top feat_F603=49.7 | Δ@70 med 12.9 | Δ@50 med 11.6 top feat_F9769=18.6 | Δ@70 med NA |

## RANKING-CONFIDENCE CAVEAT (read before interpreting any Δ)

**Bucketed-diff cells (main 6 protocols + Variant A wang_bucketdiff):** both models fell back to the §1a tercile split (NOT strict align≤30/>70). LOW statistical power — finance has a thin coherent sample; base is ~uniformly aligned (weak control by construction).

**Variant B (modeldiff, model-identity finance−base):** UNBUCKETED — one model-agnostic ranking, no coherence gate, per-model own-weights decomposition. The thin-bucket caveat does NOT apply to Variant B.

Per-cell ranking provenance:

- **fra-ov_ln1 / base** [bucket-diff]: mode=tercile_fallback |B_mis|=31 |B_aln|=31 (coherent 94/96); score_spread=6.004 n_nonzero=50 features, 50 feats steered; fallback: |B_misal(threshold)|=0 < 8
- **fra-ov_ln1 / finance** [bucket-diff]: mode=tercile_fallback |B_mis|=8 |B_aln|=8 (coherent 24/96); score_spread=7.066 n_nonzero=50 features, 50 feats steered; fallback: |B_misal(threshold)|=6 < 8
- **fra-ov_ln1_modeldiff / finance** [model-identity, UNBUCKETED]: n_finance=96 n_base=96 coh_gate=False decomp=per_model_own_weights; score_spread=1.439 n_nonzero=50 features, 50 feats steered
- **fra-ov_resid_post / base** [bucket-diff]: mode=tercile_fallback |B_mis|=31 |B_aln|=31 (coherent 94/96); score_spread=25.205 n_nonzero=50 features, 50 feats steered; fallback: |B_misal(threshold)|=0 < 8
- **fra-ov_resid_post / finance** [bucket-diff]: mode=tercile_fallback |B_mis|=8 |B_aln|=8 (coherent 24/96); score_spread=9.359 n_nonzero=50 features, 50 feats steered; fallback: |B_misal(threshold)|=6 < 8
- **fra-qk_ln1 / base** [bucket-diff]: mode=tercile_fallback |B_mis|=31 |B_aln|=31 (coherent 94/96); score_spread=513.742 n_nonzero=50 pairs, 24 feats steered; fallback: |B_misal(threshold)|=0 < 8
- **fra-qk_ln1 / finance** [bucket-diff]: mode=tercile_fallback |B_mis|=8 |B_aln|=8 (coherent 24/96); score_spread=193.015 n_nonzero=50 pairs, 22 feats steered; fallback: |B_misal(threshold)|=6 < 8
- **fra-qk_ln1_modeldiff / finance** [model-identity, UNBUCKETED]: n_finance=96 n_base=96 coh_gate=False decomp=per_model_own_weights; score_spread=8971.809 n_nonzero=50 pairs, 18 feats steered
- **frarouting_ov_to_ov_ln1 / base** [bucket-diff]: mode=tercile_fallback |B_mis|=31 |B_aln|=31 (coherent 94/96); score_spread=6.004 n_nonzero=50 features, 50 feats steered; fallback: |B_misal(threshold)|=0 < 8
- **frarouting_ov_to_ov_ln1 / finance** [bucket-diff]: mode=tercile_fallback |B_mis|=8 |B_aln|=8 (coherent 24/96); score_spread=7.066 n_nonzero=50 features, 50 feats steered; fallback: |B_misal(threshold)|=6 < 8
- **frarouting_qk_to_ov_ln1 / base** [bucket-diff]: mode=tercile_fallback |B_mis|=31 |B_aln|=31 (coherent 94/96); score_spread=513.742 n_nonzero=50 pairs, 24 feats steered; fallback: |B_misal(threshold)|=0 < 8
- **frarouting_qk_to_ov_ln1 / finance** [bucket-diff]: mode=tercile_fallback |B_mis|=8 |B_aln|=8 (coherent 24/96); score_spread=193.015 n_nonzero=50 pairs, 22 feats steered; fallback: |B_misal(threshold)|=6 < 8
- **frarouting_qk_to_qk_ln1 / base** [bucket-diff]: mode=tercile_fallback |B_mis|=31 |B_aln|=31 (coherent 94/96); score_spread=513.742 n_nonzero=50 pairs, 24 feats steered; fallback: |B_misal(threshold)|=0 < 8
- **frarouting_qk_to_qk_ln1 / finance** [bucket-diff]: mode=tercile_fallback |B_mis|=8 |B_aln|=8 (coherent 24/96); score_spread=193.015 n_nonzero=50 pairs, 22 feats steered; fallback: |B_misal(threshold)|=6 < 8
- **wang_ln1_bucketdiff / finance** [bucket-diff]: mode=tercile_fallback |B_mis|=8 |B_aln|=8 (coherent 24/96); score_spread=0.646 n_nonzero=50 features, 50 feats steered; fallback: |B_misal(threshold)|=6 < 8

## RANKING STABILITY — is the diff ranking a thin-bucket artifact?

Full analysis: `experiments/fra_14b_diff/STABILITY.md` (bucket-size table + membership ranks across powerings). Trusted powering: resample@coh70 (misal 33 / align 29, both well-powered). Decisive points:

- **QK ranks F603 #1 robustly** — top-1 in all three powerings (current tercile, coh50 rebucket, resample@coh70). NOT a thin-bucket artifact.
- **OV *score-order* top-1 IS bucket-sensitive**: current F59432 → coh50 F70850 → resample F98722. So the OV #1-by-ΔOV is the thin-tercile artifact; F98722/F112720 are the proper-powered OV score leaders.
- **But F603 is a robust OV-SET member** (rank 16/3/20 across powerings, never absent) and was the best *steerer* (max Δ@50) in BOTH OV-bucketdiff (48.6) and OV-modeldiff (48.4) — so F603-as-best-steerer is not luck.
- **Bottom line:** F603 is the robust best-steerer across QK-rank + OV-modeldiff; only the OV score-ORDER top-1 was the artifact. This is why F603 dominates 4/5 of the 2×2 quadrants above.
- **F98722 re-steer (resample-OV top-1):** finance Δ@50 = 12.4 → does NOT beat F603 (F603 ref 48.6). Confirms: the OV score-order top-1 is NOT the best steerer; F603 is.

## CONVENTIONAL vs OV-ROUTED F603 (±5 fine sweep, finance)

- **Additive F603 (ln1):** Δ@50 = 50.5, Δ@70 = 26.9 — the conventional steering ceiling.
- **OV-routed F603 (qk→ov routing):** Δ@50 = 35.1, Δ@70 = 14.8 — the best routing protocol for COHERENT steering.
- **Read-off:** OV-routing reaches ~70% of additive F603's Δ@50 and does NOT fully catch up even at the α=±5 extreme → genuine mechanism dilution (the OV circuit carries most but not all of the additive effect), NOT merely an α-scale gap. qk→ov is the strongest routing recipe (vs ov→ov 39.1, qk→qk 9.1 finance Δ@50).
