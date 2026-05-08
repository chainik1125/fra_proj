# em-repl-2026-05-07

Reproduction + benchmark of Nura's emergent-misalignment QK/OV result on
`Qwen2.5-14B-Instruct + medical LoRA` at layer 24, with same-budget SAEs at
neighbouring hookpoints for comparison.

Source repos:

- **fra_proj** (Nura's QK/OV framework): branch `dmitry-em-repl` off `origin/nura/dev`.
- **temp_xc** (planning + analysis): branch `dmitry-em-repl`,
  notes under `docs/dmitry/c6_em/2026-05-07_em_repl/`.

## Layout

```
phase1_reproduce/
    runs/                       # per-EM-model multiseed_results_v2 outputs
        medical/                # frontier_multiseed (the headline)
        finance/                # frontier_multiseed
        sports/                 # frontier_multiseed
        random_medical/         # random_baseline (sanity control)
    plots/                      # frontier grid, summary JSON
phase2_redteam/                 # bucket A/B/C/D outputs (deferred)
phase3_benchmark/
    sae/                        # 4 same-budget SAEs at L24 resid_pre / resid_mid / resid_post + L25 ln1
        resid_pre_L24/
        resid_mid_L24/
        resid_post_L24/
        ln1_normalised_L25/
    results/                    # frontier α-sweeps under each hookpoint
plots/                          # cross-phase figures
```

## Headline metric

`Δalign|coh≥70` = max(mean_alignment) − min(mean_alignment) over α-sweep points
where mean_coherence ≥ 70 (matches `temp_xc/scripts/plot_c6_em_align_coh_grid.py:headline_metrics()`).

## Compute

Two RunPod H100s (2× H100 80GB each): `h100_emfra_2gpu_1`, `h100_emfra_2gpu_2`.

License: research use only.
