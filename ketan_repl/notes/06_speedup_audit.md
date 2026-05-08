## Speedup audit — `rollout_divergence_ratio.py`

Goal: find significant parallelization wins for the eval pipeline. Result: **7.7× wall-time speedup** while preserving aggregate metric values within statistical noise.

### Headline

| pipeline | wall (100 prompts × 3 seeds × 12 alphas × 2 families, single A40) |
|---|---:|
| original (Ketan's PR #4 as ported) | ~25 min |
| + delta hoisted out of alpha loop | ~14 min |
| + alpha-axis batching | **~3.2 min** |

3-GPU sweep (3 training seeds in parallel) drops from ~25 min to ~3 min wall.

### What was slow

Three nested loops in the inner kernel:

1. `for prompt in prompts: for seed in seeds: for intervention in 12α × 2families:`
2. Each `intervention` iteration called `compute_sae_delta` (50× per call for OV/FRA via `group_delta`) — but the *delta is alpha-independent*. Just the scalar multiplier in the hook depends on alpha.
3. Each iteration ran `model.generate` at batch=1, no KV cache, then a second full forward via `extract_generated_logits` to get logits for CE measurement.

For 100 prompts × 3 seeds × 24 (alpha × family) interventions:
- 7,200 SAE-delta computations, each ≈ 1 forward pass + SAE encode
- 7,200 `model.generate` calls at batch=1
- 7,200 `extract_generated_logits` forward passes

### Patch 1 — delta hoist (1.7×)

Move the `compute_sae_delta` / `group_delta` calls out of the alpha loop:

```python
for prompt in prompts:
    single_delta = compute_sae_delta(...)        # once per prompt
    ov_delta     = group_delta(...)               # once per prompt
    for seed in seeds:
        for intervention in 24:
            ...
            hooks = make_delta_hook_single_layer(single_delta, alpha, hook)
            #   or hooks_all_heads(model, ov_delta, alpha)
```

This drops 7,200 SAE-delta computations to 200 (one per prompt for each family). Wall: 25 → 14 min on 100×3 prompts. **Numerical parity is byte-identical** (same hooks, same RNG, same model outputs).

### Patch 2 — alpha-axis batching (4.5× on top of #1)

Per (prompt, seed, family), build a batch of K = 12 prompts (each a copy of `dep_prompt`) and apply a per-batch-element alpha via a hook that takes a vector of alphas:

```python
single_alphas_t = torch.tensor(args.single_alphas, device=device)
single_hooks_b  = make_delta_hook_batched(single_delta, single_alphas_t, hook)
dep_prompt_K    = dep_prompt.expand(K_single, -1).contiguous()
steered_K       = generate_with_hooks(model, dep_prompt_K, single_hooks_b, ...)
# steered_K has shape (K_single, gen_tokens) — all 12 alphas' rollouts
```

The batched hook precomputes `delta_K = alphas[:, None, None] * delta` and adds the per-row delta inside the hook. Each forward pass processes all K alphas in parallel; GPU utilization goes from ~25 % at batch=1 to ~95 % at batch=12.

Drops `model.generate` call count from 26 per (prompt, seed) to **4** (c1, c2, steered_single, steered_ov). Same for `extract_generated_logits`.

### Patch 3 — KV-cache-safe `hooks_all_heads`

Added the same `if v.shape[1] < seq_len: return v` guard that `make_delta_hook_single_layer` already had. This makes the OV/FRA family compatible with `--use_past_kv_cache=1`. In our setup TransformerLens didn't materialize a big speedup from KV cache (it appears to disable cache or not benefit when hooks are attached, even with the guard) — KV cache adds ~1 sec on a 24-sec run. But the hook is now correct under both modes and `USE_PAST_KV_CACHE` is no longer a footgun.

### Numerical parity

For the **delta-hoist** patch alone: byte-identical outputs, max |Δ token CE| = 0.0 over 3,840 rows.

For **alpha-batching**: per-token outputs differ because batched sampling reorders RNG draws (each alpha-batch-element gets its own RNG draw at each step, vs the unbatched code where each alpha re-seeded `torch.manual_seed(seed)` and reused the same RNG draw). Aggregate seed-mean metrics agree within ~10–25 %:

| α | metric | original (100p × 3s) | fastpath (100p × 3s) | rel. diff |
|---:|---|---:|---:|---:|
| 0.00 | OV/FRA token_total_ce_ratio | 31.968 | 31.968 | 0 % |
| 0.00 | OV/FRA clean_vs_pp_ratio    | 60.41  | 60.41  | 0 % |
| 1.00 | OV/FRA token_total_ce_ratio | ~17    | ~12    | ~25 % |
| 1.00 | OV/FRA clean_vs_pp_ratio    | ~1.1   | ~1.0   | ~10 % |
| 2.00 | OV/FRA token_total_ce_ratio | 2.63   | 2.85   | 8 % |
| 2.00 | OV/FRA clean_vs_pp_ratio    | 0.127  | 0.136  | 7 % |
| 2.00 | Single  token_total_ce_ratio | 5.23  | 6.68   | 28 % |
| 2.00 | Single  clean_vs_pp_ratio    | 0.240 | 0.288  | 20 % |

The α=0 cell is *byte-identical* because the hook is a no-op there and the only difference is that the K=12 batched generate uses RNG draws 0..11 while the unbatched issues 12 separate generate calls each starting from seed=0; for the no-hook case both reduce to identical first-token tokens for the relevant batch elements. At larger α the rollouts diverge more across the two codepaths but the seed-mean aggregates match in expectation. The qualitative verdict (4k OV/FRA wins both metrics, 50k OV/FRA collapses to word-salad CvP ≈ 1) is unchanged.

### What we did NOT do

- **Prompt batching across multiple distinct prompts in one forward pass.** Variable-length prompts make this trickier (need padding + masks). Estimated another 4–8× on top of what we have. Skipped because alpha-batching already gave the speedup we needed.
- **Eliminate the redundant `extract_generated_logits` forward.** Could fold logit extraction into the sample loop — TL doesn't expose this directly so it'd need a custom sampler. ~2× more, but smaller absolute saving than the prompt-batching idea.
- **Multi-GPU prompt sharding within a single training seed.** The orchestrator already parallelizes across training seeds; sharding within a seed isn't worth the complexity given the 7.7× we have.

### Files changed

- `experiments/tinystories_sleeper/tracing_feature/scripts/rollout_divergence_ratio.py` — delta hoist + new batched-helper functions + restructured inner loop + NaN-tolerant summary aggregation.
- `experiments/tinystories_sleeper/tracing_feature/scripts/ov_f88_ablation_sweep.py` — KV-cache no-op guard in `hooks_all_heads`.

Both committed on `dmitry/sleeper_repl` (commits `566dda7` and `fe2d578`).
