## Tracing how SAE_mid feature 171 forms from resid_pre and ln1.hook_normalized

Feature 171 in the SAE trained on `blocks.0.hook_resid_mid` perfectly suppresses the TinyStories sleeper (test ASR 0.00 vs 0.99 baseline). The analogous SAEs on `blocks.0.hook_resid_pre` and `blocks.0.ln1.hook_normalized` only produce weak suppressors (ASR ≈ 0.89). Since `resid_mid = resid_pre + attn_out` and attention reads from `ln1.hook_normalized`, the suppressor feature must be built by block-0 attention. This pipeline decomposes the pre-activation of feature 171 into:

```text
pre_171(t) = e_171 · resid_pre[t]   (skip path)
           + e_171 · attn_out[t]    (OV path — with A held empirical)
           + const
```

where `e_171 = SAE_mid.W_enc[:, 171]`. The attention pattern `A` is treated as fixed from the forward pass, so only the V path needs feature-decomposition — this collapses the cubic Q×K×V blow-up to pairwise (head × feature) interactions.

See `../../../.claude/plans/source-users-dmitrymanning-coe-document-steady-unicorn.md` for the full derivation.

## Scripts

All scripts live in `scripts/` and write to `results/`.

| Step | Script | What it does |
|---|---|---|
| Phase 0 | `cache_layer0_activations.py` | One forward pass on the test split; caches block-0 `resid_pre`, `ln1.hook_normalized`, `attn_out`, `resid_mid`, `attn.hook_pattern`, plus SAE encodings and pre-activation logits for the flagged features. |
| Phase 1 | `skip_path.py` | Skip-path decomposition. Computes direction coefficients `w_pre[f] = SAE_pre.W_dec[f] · e_171` and activation correlations of `z_pre[·, f]` with `z_mid[·, 171]` (including partial correlation conditioned on `is_deployment`). |
| Phase 2 | `ov_path.py` | OV-path decomposition. Computes `u_h = W_V[0, h] @ W_O[0, h] @ e_171`, `β = SAE_ln1.W_dec @ U.T`, per-(head, ln1 feature) direction coefficients and activation correlations, per-head variance shares. |
| Phase 3 | `combine.py` | Full reconstruction `skip + Σ_h S_h + const ≈ pre_171`. Writes the full variance decomposition, a per-prompt stacked-contribution plot, and the top `(head, source position, ln1 feature)` triples on deployment prompts. |
| Phase 4 | `head_ablation.py` | Counterfactual: zero the top-contributing block-0 head and re-measure mean `z_mid[171]` and test ASR_16. Contrasts with a low-variance control head. |
| Q* OV | `target_position_ov_summary.py` | Refines the OV view to the per-prompt bottleneck token `q* = argmax_t z_mid[t,171]` within the prompt prefix. Reports the top `(head, source position, ln1 feature)` triples at that actual high-activation location. |
| Q* pre-attn | `target_position_pre_attn_summary.py` | Pulls the `q*`-locked OV explanation back through frozen LN to `resid_pre` features. Reports the top `(head, source position, resid_pre feature)` triples and the total per-feature upstream contribution. |
| Q* two-stage | `target_position_two_stage_summary.py` | Preserves the intermediate post-LN feature node at `q*`, ranking `resid_pre feature -> ln1 feature` routes (summed across heads and per head) under the frozen-LN approximation. |

## Outputs

Produced in `results/`:

- `layer0_cache.pt` — activations, SAE encodings, attention patterns (stays on the pod by default; excluded from `rsync` pullback).
- `layer0_cache.json` — sidecar meta.
- `skip_path.json`, `skip_path_per_feature.pt`, `skip_path_scatter.png`
- `ov_path.json`, `ov_path_per_pair.pt`
- `combine.json`, `combine_top_triples.json`, `combine_reconstruction.png`, `narrative.md`
- `head_ablation.json`
- `target_position_ov_summary.json/.md`
- `target_position_pre_attn_summary.json/.md`
- `target_position_two_stage_summary.json/.md`

## Running

```bash
# From the repo root, with a working SSH alias to the GPU pod (default: a40_climb):
./experiments/tinystories_sleeper/tracing_feature/reproduce.sh
```

Skip the expensive cache step if `layer0_cache.pt` already exists on the pod:
```bash
SKIP_CACHE=1 ./experiments/tinystories_sleeper/tracing_feature/reproduce.sh
```

Override the remote:
```bash
REMOTE=some_other_pod REMOTE_DIR=/root/fra_proj ./reproduce.sh
```

The pipeline expects the SAE checkpoints on the pod at:
- `recreate_layer0/results/crosscoder_sae_layer{0,1,2}.pt`  (pre / mid / post)
- `recreate_ln1/results/crosscoder_sae_layer0.pt`            (ln1.hook_normalized at layer 0)

## Feature indices

Taken from `recreate_layer0/results/test_results.json` and `recreate_ln1/results/test_results.json`:

| Hook point | SAE | Feature | α | test ASR | Δ CE |
|---|---|---|---|---|---|
| `blocks.0.hook_resid_pre` | `crosscoder_sae_layer0.pt` (layer0) | **1359** | 2.0 | 0.89 | +0.003 |
| `blocks.0.ln1.hook_normalized` | `crosscoder_sae_layer0.pt` (ln1) | **1412** | 2.0 | 0.89 | +0.014 |
| `blocks.0.hook_resid_mid` | `crosscoder_sae_layer1.pt` (layer0) | **171** | 2.0 | **0.00** | −0.001 |
| `blocks.0.hook_resid_post` | `crosscoder_sae_layer2.pt` (layer0) | 1152 | 2.0 | 0.18 | −0.0003 |

Baseline test ASR = 0.99.
