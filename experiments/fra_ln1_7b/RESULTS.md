# FRA-at-ln1 on Qwen-2.5-7B — overnight run (2026-05-26)

End-to-end: trained an Arditi-style BatchTopK SAE at **ln1.hook_normalized**
(L15, k=64, d_sae=131072, 100M tokens) on base Qwen-2.5-7B-Instruct, then ran
the 4 FRA recipes (baseline / qk→qk / qk→ov / ov→ov) on **base then EM-medical**,
3 eval seeds, scored with **both** protocols: our free-form GPT-4o judge
(Δcoh70) and Arditi's single-token forced-choice MC (Δ misaligned-choice %),
the MC eval run *under the FRA hooks*.

## Comparison (n=3 seeds)

MC baseline misaligned-choice: base 3.1% · medical 3.2%.

| recipe | free-form Δcoh70 base | free-form Δcoh70 medical | MC Δmisalign% base | MC Δmisalign% medical |
|---|---:|---:|---:|---:|
| qk→qk | 8.1 ± 2.9 | 3.8 ± 2.7 | 46.7 ± 0.0 | 43.5 ± 0.0 |
| qk→ov | 3.1 ± 0.6 | 14.0 ± 13.5 | 0.0 | 0.1 |
| ov→ov | 4.0 ± 1.9 | 8.3 ± 9.5 | 0.0 | 1.0 |

## Findings (provisional — see caveats)

1. **The two metrics disagree, and mechanistically should.** qk→qk perturbs
   ln1 (the whole residual feeding attention) → huge single-token MC shift
   (~47%) but small *coherent* free-form effect (3.8 on EM). The OV recipes
   tweak one head's value → ~0 MC shift but the larger *coherent* free-form
   effect on EM (qk→ov 14, ov→ov 8). So **single-token forced-choice rewards
   gross residual perturbation; coherence-gated free-form rewards the
   localized OV intervention.** Same MC-vs-free-form gap seen in the earlier
   Arditi work.
2. On EM-medical the **OV-routing recipes carry the coherent misalignment
   signal** (qk→ov 14, ov→ov 8 vs qk→qk 3.8) — but with **huge seed variance**
   (±13.5, ±9.5) at n=3.

## ⚠️ Caveats (must resolve before trusting magnitudes)

- **SAE reconstruction is poor:** even after forcing exact top-k (L0=64), the
  ln1 SAE has var-explained ≈ **−3.36** on real activations (reconstruction
  ~2× off-scale). Almost certainly an **activation-norm-factor mismatch** —
  the SAE trained on Arditi's normalized activations, the FRA hooks feed raw
  ln1.hook_normalized. The FRA delta-form recipes are robust-by-design to a
  lossy SAE (so the *relative* recipe comparison is informative), but absolute
  magnitudes, α-calibration, and feature interpretability are **not trustworthy
  yet**. Fix = recover/apply the norm factor (likely a retrain). **← morning decision.**
- MC qk→qk effect saturates at extreme α (std 0.0 = same saturated value every
  seed), so the 47% is "perturbed into the misaligned token," not coherent EM.
- Medical free-form OV effects: seed std ≈ mean → treat as directional only.
- Head = 0 (head-ablation argmax loss_delta on base); not re-picked on EM.

## Repro / data
- SAE: `qwen7b/sae_ln1_l15_base_arditi/` (HF). FRA outputs +
  combined: `qwen7b/fra_ln1_l15/` (HF).
- Scripts: `experiments/fra_ln1_7b/{train_sae,launch_train,run_fra,launch_fra}.sh`,
  `phase1_qkqk_7b_orchestrator.py --mc-eval`.
