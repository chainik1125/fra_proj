# Claims–evidence map (fable-overnight, 2026-06-09)

Source-of-truth docs (fra_proj @ 9dcbd85):
- `experiments/multitrigger_sleeper/summary.md` (multi-trigger sprint + §4b–4e protocol comparison)
- `experiments/multitrigger_sleeper/weight_diff_summary.md` (K1 knowledge-tier four-way, bits)
- `docs/dmitry/jamie_latest/RESULTS_SUMMARY.md` (corrected single-sleeper pipeline numbers)
- Jamie's paper snapshot: fra_proj_tex commit `46a432a` (example_paper.tex + 18 figures)
- `experiments/fra_14b_diff/{WRITEUP,GRID_RESULTS*,STABILITY,CROSS_FINETUNE_SUMMARY}.md` (EM campaign)

## Defender framing (user's brief)
Defender knows some rollouts/sentences are poisoned but not which. Goals:
(D1) flag the poisoned ones (detection);
(D2) remove the sleeper without damaging clean rollouts;
(D3) convert poisoned → clean, ideally word-for-word.

## SLEEPERS — claims and evidence

### S1. FRA localises and detects the sleeper essentially perfectly.
- Hook/layer sweep localises construction to layer-0 attention sublayer (Jamie fig loc_viz2_scatter).
- Detector features: AUROC 0.997–1.0 single-token triggers; multi-token share delimiter feature 1788
  (mts summary C2). Detection architecture- and scale-invariant (arch_compare, scaling_pod).
- Position-agnostic: FRA detect → cut neutralises single-token triggers at any position (ASR→0,
  clean-FP 0) where fixed-position oracle fails (mts §4c); span-aware lift fixes multi-token
  (ASR 1.00→0.02, span recall 1.0; §4d Exp 1).

### S2. Removal/conversion: FRA-OV ties optimised conventional baselines; no method beats sampling noise floor by much. (mars-jason single sleeper, Jamie corrected pipeline, 6 SAE seeds, T=1 matched-RNG eval, bits)
| method | JSD_clean matched | exact-match % | ASR % |
| OV (FRA L0 feat → hook_v) | 0.344 ± 0.049 | 36.9 ± 4.6 | 0.02 |
| Conv (resid SAE feat, additive) | 0.307 ± 0.061 | 42.5 ± 7.8 | 0.02 |
| DoM (SAE-free geometric ablation) | 0.304 | 40.3 | 0.00 |
- clean-vs-clean unmatched floor 0.42 bits; JSD_pois 0.990 for all.
- Ordering correction vs old paper: Conv/DoM marginally ahead (≈0.6σ, overlapping); old "OV pareto-
  dominant + 20% word-for-word" claim came from less-optimised baselines + tail blowups (RESULTS_SUMMARY
  reconciliation). Old OV mean advantage = conventional seed-2 selection blowup artifact.

### S3. Detect ≠ control: the detector feature does not carry the payload.
- DEPLOYMENT detector = 96.5% of trigger's L0 de-biased reconstruction; ablating it (QK, OV, full path)
  leaves ASR 0.99 (mts C3). Cumulative: top-8 do nothing; breaks at ~16/32; |WORD| floor 0.17 outside SAE recon.
- Replicated on independent seed-1 model+SAE (ASR 0.98).
- FRA-QK first-order attribution over-predicts causal effect (predicted ΔS≈0.4 logits vs actual Δpattern
  ≈0.0009) → flagged honestly as limitation (Jacobian weight fixes pattern-level: ρ 0.55→0.80).

### S4. What sets removal quality is the defender's KNOWLEDGE TIER, not the algorithm. (K1 weight-diff campaign, J_clean bits @ ASR≤0.05, greedy, seed 7; conv-SAE 5-seed)
| tier | DoM | conv-SAE | FRA (weight-diff×SAE, gated OV) | SVD ΔW_OV | winner |
| zero knowledge (2 checkpoints) | — | — | .404 | .296 | SVD |
| presence (examples, no location) | .162 | .212±.022 | .286 | .296 | DoM |
| location (no examples) | — | — | .123 | .196 | FRA |
| presence+location | .114 | .177 | .118 | .196 | DoM ⩦ FRA tie |
- Attribution saturated on K1: any trigger-span-pooled ranking (raw firing = weight-diff) → same ~6–8
  carriers, same ~0.13 shelf. "Knowledge, not algorithm, is the operative axis."
- No single-feature kill switch at L2 (640-point null); L0 exception f1253 (trigger-token identity,
  (0,.199,67% match) @ c=6). Removal needs over-steered set (K≈8–13, c≈2–3).
- Zero-knowledge SVD: ΔW_OV rank-2/3; top-2 dirs gated removal → (0,.296), 58–62% match; generalises
  to K8-randpos (0.03,.379) where every span-localized method fails.
- Fidelity (D3): (J, word-match) on one monotone curve: .10→88%, .15→79–83%, .30→58–62%. Optimised DoM
  on-curve (greedy .152/83%; matched-T1 .143/75%). So fidelity is determined by J; no separate FRA win.
- Bridge eval (paper protocol, T=1 matched RNG, K1 model): FRA set K24·c2 0.185/62.5%; DoM 0.143/75%;
  SVD k2 0.377; vs paper-published singles 0.304–0.344/37–43%. Improvement is OPTIMISATION+SET, not method.
  Caveat: mars-jason vs K1 model remains unmatched axis.

### S5. Multi-trigger (K=1..8): one cut kills all; residual-space methods floor; hybrid is best residual.
- Oracle attention-cut + positional re-index: exact (0,0) at every K (J_tf ≈ −1e-8); weights-agnostic
  (MLP-route sleeper falls identically) and PE-agnostic (RoPE re-index) (§4d Exp 10).
- Protocol table (best ASR, J_clean): single detector feat fails (0.99,0.69); FRA-OV multi-feat (0,0.54);
  resid DoM ablation (0,0.58); CAA/DoM steer (0,0.27–0.31); gradient-optimised steer (0,0.15–0.20,
  ≈⊥ everything interpretable); hybrid OV-ablate+light-steer (0,0.08–0.11) ⭐ new residual record,
  replicates on clean w=1 single backdoor (0.0995); oracle (0,0) unique.
- One steer kills all 8 (shared payload): CAA mean J 0.27. One cut kills all K for free.
- OV-channel localisation: payload is in the value path; OV-only ablation (0.063,0.068) beats full-path.

## EM — claims and evidence (fra_14b_diff campaign; replaces ALL old mislabeled EM results)

Old results to REMOVE: phase1_2x3_seed42_neg6 frontier + phase1_fra_plus_additive bars + ±28 noise
claims + "QK→QK reaches +23" + 60% overlap table (old campaign; QK-attribute/ln1-intervene mislabeled;
"conventional" was 50-features-at-once). Also appendix seed grids + random-feature baseline + CE analysis
from old campaign. Head-ablation table: old picked H38/H0/H36/H7; new campaign pins H12 (argmax).

### E1. Properly-diffed FRA attribution ties conventional SAE steering (Wang activation-diff). They converge on the same feature.
- Setup: Qwen2.5-14B-Instruct + 3 EM LoRA finetunes (financial/medical/sports), L24, Arditi batch-top-k
  SAEs at ln1 + resid_post; bucketed-diff ranking (misal vs aligned rollouts, per-model); magnitude-matched
  steering α·‖Δa‖·unit(W_dec); judge gpt-4o-mini@T0; metric Δalign@coh50 (and @70); n_seeds=2; base = control (Δ≈1–7).
- Best single feature Δ@50 (EM model, ln1): fin F603 48.1/48.6/49.7 (Wang/OV/QK); sports F603 41.9/42.0/42.5;
  med F111743/F118097 ≈30 (distributed). 2×2 (attribution × diff-def): gap is NEITHER; quadrants agree.
- resid_post: F88683 med 42.7 / sports 56.4–56.5 (Wang=FRA, same feature); fin resid weaker (33.7).
  Wang resid fin "win" F93118·66 is base-general (base 70.6) → not finetune-specific.

### E2. F603: a cross-finetune misalignment feature, finetune-RECRUITED not created.
- Dominant steerer fin+sports on every ln1 method; qk→ov routing winner in all three (19.1/19.5/17.3).
- Base control: steering F603 on base does ~nothing in usable range (Δ≈2 vs 50 on fin EM).
- Rate-trajectory: unsteered fin EM = 47% coherent-misaligned; steering → 94% coherent-aligned at α=−2.5;
  base never exceeds ~6% coherent-misaligned (degrades to incoherence instead). E[align|coh] gap 12–16 pts
  at every coherence band.
- FRA-QK is the only a-priori-calibrated score: ranks F603 #1 in all three finetunes (Wang #4–10,
  FRA-OV #16–19) — and F603 is in fact the (near-)best steerer. Stability: QK top-1 F603 robust across
  3 powerings; OV score-order top-1 was thin-bucket artifact (F59432→F98722; F98722 re-steer 12.4 ≪ 48.6).
- Medical is the outlier: distributed, no dominant ln1 feature; F603 secondary (Δ≈24, #5/50; #1 @70 for QK).

### E3. Mechanism: the additive effect routes ~70% through the OV circuit; QK-pattern intervention is inert.
- Conventional/additive F603 ln1: Δ@50 50.5, Δ@70 26.9 (±5 finegrid). OV-routed same feature (qk→ov hook_v):
  35.1 / 14.8; doesn't catch up at α±5 → genuine mechanism dilution. ov→ov 39.1 but coherence collapses
  (Δ@70 6.1); qk→qk inert (≈8–9, base-control floor).

### EM caveats (must state): tercile fallback (thin buckets, fin |B|=8), n_seeds=2, Δ ordinal not precise,
@70 windows often collapse, single layer/head, gpt-4o-mini judge.

## Cross-cutting narrative (the paper's claims)
1. FRA = an exact (up to SAE error) feature-level decomposition of attention (QK bilinear + OV linear);
   gives detection/localisation/diagnosis machinery that activation-only methods don't have.
2. Where behaviour is carried by a *single recruited direction* (EM F603; K1 L0 token feature), every
   reasonable selection method finds it and steering ties — FRA's marginal value is calibration
   (QK score ranks it #1 a priori) and mechanism attribution (OV carries ~70%).
3. Where behaviour is *distributed* (sleeper payload across ~all trigger features; medical EM), no
   single-feature method wins; control quality is set by knowledge tier / intervention class
   (set-removal, hybrid ablate+steer, or the content-agnostic attention cut).
4. Honest framing: FRA does not furnish better steering DIRECTIONS than DoM/Wang; its wins are
   detection (≈perfect, position-agnostic), example-free knowledge tiers, diagnosis (predicts
   feature-ablation failure), and channel localisation (payload in OV).

## Figure plan
Sleepers: fig2_lowest_jsdc (channel comparison) → jsd_exact_main_seed0 + stats table (main result)
→ k1_fourway_summary (knowledge tiers) [+ appendix: loc_viz2_scatter, jsd_exact_all_seeds,
fra_steer_examples, fig4_c3_mechanism, fig6_cumulative, protocol ladder table].
EM: steering_effect_by_scheme (scheme×finetune bars) → F603_steering_curves → F603_base_vs_em or
F603_rate_trajectory [+ appendix: GRID_RESULTS extracts, stability].

## Number-checking TODO before final
- All sleeper J values in BITS (weight_diff figures regenerated in bits 2026-06-09; raw HF JSONs nats).
- Jamie table exact values from jsd_stats_table.tex (already bits).
- EM Δ values from GRID_RESULTS_* tables (judge gpt-4o-mini@T0).
