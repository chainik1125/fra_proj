# FRA-at-ln1 on Qwen-2.5-7B — results (corrected SAE), 2026-05-26

End-to-end: trained an Arditi-style BatchTopK SAE at **ln1.hook_normalized**
(L15, k=64, d_sae=131072, 100M tokens) on base Qwen-2.5-7B-Instruct, ran the 4
FRA recipes (baseline / qk→qk / qk→ov / ov→ov) on **base then EM-medical**, 3
seeds, scored with **both** protocols: free-form GPT-4o Δcoh70 and Arditi
single-token forced-choice MC (run under the FRA hooks).

## SAE is sound (after the γ fix)

The SAE first looked broken (var-explained −3.36) — but that was a **usage bug**,
not the SAE. It was trained on the HF `input_layernorm` OUTPUT = (x/rms)·γ
(post-gain), while the FRA hooks fed TL's `ln1.hook_normalized` = x/rms
(pre-gain). Feeding it the post-gain activation (×γ = `blocks[L].ln1.w`) gives
**var-explained ≈ 0.50**. Quality vs Arditi's published resid_post SAE:

| SAE | avg % residual error | loss recovered | native L0 |
|---|---:|---:|---:|
| ours — ln1 L15 | **52.1%** | n/a (degenerate, see below) | 58 |
| Arditi — published resid_post L15 | **51.0%** | 0.506 | 11,148 |

- **Reconstruction is on par** with Arditi's published SAE (~51–52% residual error).
- **Loss-recovered doesn't transfer to ln1**: zero-ablating `ln1.hook_normalized`
  at one layer barely moves loss (3.82→3.76), so the ratio is meaningless; it's
  only interpretable at resid-stream hookpoints (resid_post: 0.51).
- Both SAEs' native eval-thresholds are miscalibrated in our TL usage (Arditi's
  resid_post L0=11,148 is driven by outlier-norm/BOS tokens his training filtered
  out); we force exact top-k=64, so this doesn't affect the metrics above.

## The comparison (corrected SAE, n=3 seeds)

MC baseline misaligned-choice: base 3.1% · medical 3.2%.

| recipe | free-form Δcoh70 base | free-form Δcoh70 medical | MC Δmisalign% base | MC Δmisalign% medical |
|---|---:|---:|---:|---:|
| qk→qk | 7.3 ± 4.8 | 7.3 ± 6.5 | 0.2 | 4.9 |
| qk→ov | 4.8 ± 0.4 | **14.1 ± 4.9** | 0.0 | 0.0 |
| ov→ov | 3.8 ± 1.7 | **10.4 ± 9.8** | 0.0 | 0.0 |

## Findings

1. **The γ fix materially changed qk→qk.** On EM it went **1.9 → 7.3** between
   the broken and corrected SAE (qk→ov and ov→ov barely moved: 14.0→14.1,
   8.3→10.4). qk→qk is the recipe most dependent on the SAE encode + decoder
   (it adds Σ f·W_dec back at ln1), so it was the one the scale bug corrupted.
   Lesson: always gate FRA on a verified-good SAE.
2. **On EM, the OV-routing recipes lead** (qk→ov 14.1, ov→ov 10.4) over qk→qk
   (7.3) in coherent free-form behaviour — but seed variance is large (±5–10),
   so treat the ordering as suggestive, not established.
3. **The two metrics disagree, recipe-specifically.** MC forced-choice moves
   only for qk→qk (medical 4.9%, base 0.2%) and is ~0 for the OV recipes; the
   coherent free-form metric is the opposite (OV recipes largest). qk→qk
   perturbs the whole residual at ln1 → big single-token MC shift; OV recipes
   tweak one head's value → no MC shift but a real coherent free-form effect.
4. **Base stays modest** (3.8–7.3) — near-ceiling aligned, as expected.

## ⚠️ Caveats
- Seed variance is large at n=3 (±5–10 on the EM effects); these clear ~2×SE
  only marginally. More seeds (or samples) needed to firm up the recipe ordering.
- Head fixed at 0 (head-ablation argmax loss_delta on base); not re-picked on EM.

## Data / repro (HF `dmanningcoe/fra-phase1-steering-data`)
- corrected FRA: `qwen7b/fra_ln1_l15_gaincorrected/` (qualitative + mc + combined)
- broken-usage run (reference): `qwen7b/fra_ln1_l15/`
- SAE: `qwen7b/sae_ln1_l15_base_arditi/` (+ `diag_result.json`, `sae_quality_cmp.json`)
- scripts: `experiments/fra_ln1_7b/{train_sae,run_fra,diagnose,measure_sae}.sh`,
  `phase1_qkqk_7b_orchestrator.py --mc-eval`
