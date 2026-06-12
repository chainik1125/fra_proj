# FRA × WEIGHT-SPARSE (B1) — evaluator log (WS_LOG.md)

Deliverable: SAE-training comparison (sparse vs dense) + FRA comparison + verdict
(weight-sparse HELPS / DOESN'T / PARTIALLY helps FRA's diffuseness, because X).

## PHASE 0 — RESEARCH + DESIGN (DONE 2026-06-12)
See **DESIGN.md** (full). Headline: **GO (conditional)**.
- Models are released + loadable (github `circuit_sparsity.inference.gpt.load_model`, Azure blob; CPU-OK, no triton).
- A **clean architecturally-matched ladder** exists at 1x/d_model=256: `csp_sweep1_1x_3.7Mnonzero_afrac0.250` (wt-sparse+act-sparse)
  vs `..._afrac1.000` (byte-identical arch, dense activations) vs `dense1_1x` (fully dense, depth-4 confound — flagged).
- Behavior is **Python code QK circuits** (quote-closing primary; variable-binding 2-hop secondary), NOT NL induction → we port the
  persistence *metrics* (top1_coverage / n_cells_for_90 / q-k drift / Spearman / recovery), not the gpt2 harness.
- FRA: ω from c_attn slices; codes `u` from (A) NEURON-BASIS `act_in` (SAE-free clean test, run first) and (B) identical TopK SAEs.
- Known subtlety: model re-sparsifies q,k post-projection → measure FRA-recon-R²; fallback = exactly-faithful q/k-projected basis.
- Dense anchors to beat (gpt2 cross-setting + internal dense1_1x): top1_cov 0.32-0.5, n_cells_for_90 3-8, Spearman~0.03, frac_oracle~0.014.

## PHASE 0.5 — SMOKE (neuron-basis FRA on the 3-model ladder) — PENDING
## PHASE 1 — SAE-TRAINING COMPARISON (FVU/dead/L0/R²/monosemanticity ×3) — PENDING
## PHASE 2 — FRA COMPARISON (SAE-basis + neuron-basis; concentration/drift/Spearman; set-level drift) — PENDING
## VERDICT — PENDING
