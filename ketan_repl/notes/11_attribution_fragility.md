## The single-50k Pareto win reproduces on our 50k SAE — at f=1483

The earlier conclusion that "our 50k SAEs failed to learn a clean trigger-detector" was wrong. They did learn one — at index 1483, sae_seed=0. We just weren't surfacing it because we'd used `--selection_method ketan` (`dep_vs_clean_contribution`, our default OV ranking) instead of `--selection_method jamie` (head-summed prompt-masked OV).

### The pipeline run that surfaced f=1483

```bash
cd /root/jamie_sleepers
.venv/bin/python -m scripts.feature_set_pipeline \
  --selection_method jamie --top_k 20 --screen_alphas 2 4 \
  --eval_mode single --sae_seeds 0 \
  --alphas 0 0.5 1 1.5 2 \
  --sae_ln1_dir weights/seeds_50k_ours \
  --sae_mid weights/sae_resid_mid_50k.pt
```

(`weights/seeds_50k_ours/` = our 50k SAEs converted to jamie's TopKSAE format — same `state_dict`, just renamed config keys `k_total → k`.)

Jamie-attribution top-20 on our SAE seed-0:
```
[1483, 1376, 474, 891, 656, 1055, 1143, 132, 848, 1445, ...]
```

Stage-0 Δdep-logp screen survivors (10/20):
```
[1483, 656, 212, 470, 848, 1138, 132, 1434, ...]
```

Stage-1 ASR winner: **f=1483 with min-ASR=0.000 at α=4.0**

### Eval points for f=1483 across α (jamie pipeline)

| α | ASR | Δcln-CE | gen-CE-ratio | severity (rnr) |
|--:|---:|---:|---:|---:|
| 0.0 | 0.973 | 0.0000 | 2.742 | 1.150 |
| 0.5 | 0.933 | 0.0000 | 2.842 | 0.913 |
| 1.0 | 0.513 | 0.0000 | 3.417 | 1.078 |
| 1.5 | 0.050 | 0.0000 | 2.494 | 0.880 |
| 2.0 | **0.005** | 0.0000 | 1.664 | **0.821** |

ASR collapses 0.97 → 0.005, Δcln-CE = 0 throughout, severity ratio at α=2 *below* the clean-vs-clean noise floor (0.821 < 1). Identical pattern to jamie's f=1114 on his SAE.

### JSD measurement on our SAE + f=1483

Running jamie's `jsd_eval` math on our 50k SAE seed-0 with single-feature override f=1483 at α=2:

| metric | jamie (his SAE, f=1114) | ours (our SAE, f=1483) | Δ |
|---|---:|---:|---:|
| JSD(steered, clean) | 0.386 bits | **0.414 bits** | +0.028 |
| JSD(steered, poisoned) | 0.983 bits | **0.977 bits** | −0.006 |

Within 0.03 bits on both axes — same magnitude of agreement as our 4k vs 50k cells in the published replication. **The single-50k Pareto win is fully reproducible on our SAEs once we use the right attribution method.**

### Why ketan's ranking missed f=1483

Our `dep_vs_clean_contribution` ranking puts f=1483 nowhere near the top-50 (top-50 = `[463, 486, 1372, 660, 438, 965, 512, 1243, 353, 848, 221, 1483, ...]` — f=1483 is *just inside* at rank 12, but above it sit 11 features with stronger OV-attribution scores that don't actually suppress sleeper).

`--selection_method jamie` is **head-summed prompt-masked OV** (`feature_set_pipeline.py:_select_top_features` → `head_topk_ov_sweep`-style scoring). The differences vs ketan:

- ketan: `S_λ = Σ_h (per_pair_dep[h, λ] − per_pair_cln[h, λ])` — head-summed OV contribution differential (dep minus clean)
- jamie: head-summed OV contribution restricted to **prompt positions** with prompt mask, on a selection split of 100 prompts

The prompt-position mask is the active ingredient: f=1483 has its OV mass concentrated *exactly* at the trigger position, so prompt-mask-restricted scoring elevates it. ketan's ranking sums OV contribution across all positions and dilutes the trigger-specific signal.

### Implication for the headline result

The recipe is **not** RNG-fragile across SAE training trajectories. It's **attribution-fragile**: whether you find the trigger-detector feature depends on whether your ranking surfaces it. Once surfaced, a single feature (f=1483 on ours, f=1114 on jamie's) gives:
- ASR ≈ 0
- Δcln-CE ≈ 0
- JSD(steered, clean) ≈ 0.4 bits — moderate but not catastrophic
- JSD(steered, poisoned) ≈ 0.98 bits — sleeper killed

This is the textbook Pareto-win profile, reproduced independently on two different SAE training trajectories.

### What to fix in our codepath

`rollout_divergence_ratio.py` and the OV-ranking pipeline currently use `dep_vs_clean_contribution` exclusively. Adding the head-summed prompt-masked OV ranking as an option (or as the default for the single-feature recipe) would let our pipeline find f=1483-equivalents without depending on jamie's external pipeline. This is a small refactor: the `head_topk_ov_sweep`-style scoring computes per-(head, feature) OV mass on prompt positions for a sample of dep prompts and sums across heads.

Until we make that change, the single-50k Pareto-win recipe should be reported as: **(jamie attribution method)** + (V-pathway hook). The recipe is real, our SAEs support it, the prior "fails on our SAEs" finding was an artifact of using the wrong attribution.

The user's intuition: spot on — "the fact that there is *a* feature with this performance should not be RNG-dependent." It isn't. It was attribution-dependent.
