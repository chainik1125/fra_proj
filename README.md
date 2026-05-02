# sleeper

Minimal codebase for two things:

1. **Reproduce Dmitry's resid_mid SAE-feature steering** of the TinyStories
   sleeper model — the result that ablating one SAE feature at
   `blocks.0.hook_resid_mid` collapses the sleeper trigger.
2. **OV-only attribution + intervention** using

   ```
   C^OV_{h, k, λ}(d) = A^h_{q,k} · z_ln1[k, λ] · ⟨W_dec_ln1[λ] · W_OV^h, d⟩
   ```

   Attribution scores ln1 features by their dep-vs-clean OV contribution to
   the resid_mid suppressor direction d; intervention steers the top-K
   features at `attn.hook_v` only — Q and K untouched, so the attention
   pattern stays at its un-perturbed value.

## Layout

```
sleeper/
  model.py        sleeper model + paired clean/deployment dataset
  sae.py          TopK SAE + train / save / load
  hooks.py        compute_sae_delta, compute_meandiff_delta,
                  additive_steer_hook, ov_only_steer_hook,
                  greedy/sample generate_with_hooks
  attribution.py  compute_ov_weights, ov_attribution, rank_dep_vs_clean
  baselines.py    compute_meandiff_vector
  metrics.py      asr_16, teacher_forced_sleeper_logp,
                  clean_continuation_ce, rank_features_by_dep_clean

scripts/
  train_sae.py            train a TopK SAE on one residual hook
  reproduce_steering.py   (1) rank → α-sweep → test ASR / Δlogp / ΔCE
  steer_demo.py           qualitative: clean / deployed / steered continuations
  meandiff_baseline.py    v_md = mean(dep) - mean(clean) vs SAE feature
  feature_positions.py    per-position firing analysis for one SAE feature
  ov_attribute.py         (2a) compute C^OV[h, q, k, λ] for d = SAE_mid.W_enc[:, f]
  ov_intervene.py         (2b) OV-only steer top-K λ from attribution

weights/                  trained SAEs + cached attribution tensors (gitignored)
```

## Quickstart

```bash
uv sync

# Train SAEs at the two hooks that the analysis uses
python -m scripts.train_sae --hook blocks.0.hook_resid_mid       --out weights/sae_resid_mid.pt
python -m scripts.train_sae --hook blocks.0.ln1.hook_normalized  --out weights/sae_ln1.pt

# (1) Reproduce: rank features at resid_mid, sweep alpha, evaluate
python -m scripts.reproduce_steering --sae weights/sae_resid_mid.pt

# Mean-diff baseline against whichever feature the sweep picks
python -m scripts.meandiff_baseline --sae weights/sae_resid_mid.pt --feature 171

# Qualitative demo
python -m scripts.steer_demo --sae weights/sae_resid_mid.pt --feature 171 --alpha 2.0

# (2a) OV attribution to d = SAE_mid.W_enc[:, 171]
python -m scripts.ov_attribute \
    --sae_ln1 weights/sae_ln1.pt --sae_mid weights/sae_resid_mid.pt \
    --target_feature 171 --out weights/ov_attribution.pt

# (2b) OV-only intervention on the top-3 ranked ln1 features
python -m scripts.ov_intervene \
    --sae_ln1 weights/sae_ln1.pt --attribution weights/ov_attribution.pt \
    --top_k 3 --alphas 0.5 1 2 4
```
