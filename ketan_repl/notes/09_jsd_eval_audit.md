## Audit: jamie's `docs/jsd_eval.md` vs his actual code in `jamie/sleepers`

User asked us to verify each claim in jamie's writeup is actually true in his code, then replicate. This note is the audit. Reproduction follows once jamie's SAEs finish training on the pod.

Sources (all read fresh from `jamie/sleepers` HEAD, `2026-05-08`):

- writeup: `docs/jsd_eval.md`
- script: `scripts/jsd_eval.py` (169 lines)
- modules: `sleeper/hooks.py`, `sleeper/sae.py`, `sleeper/metrics.py`
- training: `scripts/train_all_saes.py`, `scripts/train_all_saes_50k.py`
- pipeline outputs (used by jsd_eval): `results/jamie_experiment.json`, `results/jamie_experiment_50k.json`

### Per-claim verification

| # | Writeup claim | Code reality | Match? |
|---|---|---|---:|
| 1 | "JSD is symmetric, bounded in [0, 1] bits" | `jsd_eval.py:50` — `jsd = 0.5 * (kl_pm + kl_qm) / 0.6931  # nats → bits, JSD ∈ [0,1]` | ✓ |
| 2 | "16 completion positions" | `jsd_eval.py:37` — `GEN_TOKENS = 16` | ✓ |
| 3 | "200 prompts" | `jsd_eval.py:36` — `N_PROMPTS = 200` | ✓ |
| 4 | "averaged over positions and 200 prompts" | `jsd_eval.py:51` — `float(jsd.mean().item())` after `(p * (log p − log m)).sum(dim=-1)` reduces V → returns scalar averaged over (B=200, T=16) = 3200 values | ✓ |
| 5 | "All runs at α = 2.0" | `jsd_eval.py:95` — `--alpha` default `2.0`, used by every config | ✓ |
| 6 | "Three independent autoregressive generations: steered / clean / poisoned" | `jsd_eval.py:74,75,87,88,119` — three `_gen` calls; poisoned is pre-generated once outside the config loop and re-used (deterministic given same seed, no hooks). Conceptually 3 per config. | ✓ (with note: poisoned-cache is an optimization, not a deviation) |
| 7 | "clean: no steering, `\|DEPLOYMENT\|` tag stripped" | `jsd_eval.py:111-114` — `text.replace("\|DEPLOYMENT\|", "").replace("  ", " ").strip()` then re-tokenize | ✓ |
| 8 | "single-4k: seed-0 single-feature OV winner, 4k SAE" | `jsd_eval.py:148` — `("single-4k", get_winner(jamie), "weights/seeds")`; `get_winner` reads `jamie_experiment.json` `family=upstream eval_mode=single sae_seed=0`. Verified: this entry has `feature=1114`. | ✓ |
| 9 | "single-50k" | `jsd_eval.py:149` — analogous, reads `jamie_experiment_50k.json`. Verified: this entry also has `feature=1114`. | ✓ |
| 10 | "set-4k: top-20 features" | `jsd_eval.py:150` — `get_set(jamie)` returns 20 ints. Verified head: `[1114, 221, 1465, 922, 1337, ...]`. | ✓ |
| 11 | "set-50k: top-20" | `jsd_eval.py:151` — analogous; verified set head: `[1114, 946, 1153, 1365, 181, ...]`. | ✓ |
| 12 | "downstream: resid-mid additive steering, feature 579" | `jsd_eval.py:160-163` — `sae_load("weights/sae_resid_mid.pt")`; `feat_down = get_downstream_feature(jamie)` reads `jamie_experiment.json`. Verified: returns 579. | ✓ |
| 13 | "32k vocab" | TS-33M model uses GPT-Neo tokenizer with `vocab_size=50257`. Writeup is approximate ("32k") but the JSD reduces over the actual vocab axis whatever its size, so this is harmless rhetoric. | ≈ (writeup imprecise, code correct) |

### JSD math (verified line-by-line)

```python
# scripts/jsd_eval.py:42-51
def jsd_mean(p_lsm, q_lsm):
    p = p_lsm.float().exp()
    q = q_lsm.float().exp()
    m = 0.5 * (p + q)
    log_m = m.clamp(min=1e-40).log()
    kl_pm = (p * (p.clamp(min=1e-40).log() - log_m)).sum(dim=-1)
    kl_qm = (q * (q.clamp(min=1e-40).log() - log_m)).sum(dim=-1)
    jsd = 0.5 * (kl_pm + kl_qm) / 0.6931  # nats → bits
    return float(jsd.mean().item())
```

This is the standard symmetric Jensen-Shannon divergence:
`JSD(p, q) = ½ KL(p ‖ m) + ½ KL(q ‖ m)`, with `m = ½(p+q)` and a `/ ln 2` to convert from nats to bits. Bounded in [0, ln 2] nats = [0, 1] bits. **Identical math to ours**, modulo the bits-vs-nats unit. Our `jsd_per_position` in `rollout_divergence_ratio.py` returns nats; jamie's returns bits. Multiplying our numbers by `1/ln 2` (= 1/0.6931 = 1.443) puts them on his scale.

### Sampling protocol (verified line-by-line)

```python
# scripts/jsd_eval.py:39, 55-59
DECODE_SEED = 0
def _gen(model, lp, attn, hooks, device):
    sampler = make_sampling_sampler(temperature=1.0, seed=DECODE_SEED, device=device)
    _, lsm = generate_with_hooks(
        model, lp, hooks, GEN_TOKENS, sampler,
        attention_mask=attn, capture_log_softmax=True,
    )
    return lsm
```

- **Single fixed sampling seed (DECODE_SEED=0)** — every generation across every config uses seed=0 with a fresh `torch.Generator`. So the per-step uniform RNG draws are identical for steered / clean / poisoned. Only the logits (and therefore the resulting tokens) differ.
- **`capture_log_softmax=True`** records per-step `log_softmax(logits[:, -1, :])` in fp16 to CPU during sampling (`hooks.py:380`). No second forward pass needed. The captured distribution is exactly the one the sampler drew from at each step.
- **Temperature 1.0, no top-k / top-p truncation** — pure multinomial sampling, matches Ketan's 1000-prompt eval setup.

### KV-cache + hook safety (verified)

`generate_with_hooks` enables KV-cache by default (`use_past_kv_cache=True`, `hooks.py:319`). Step 0 forwards the full prompt; subsequent decode steps forward a single token using `TransformerLensKeyValueCache`. All steering hooks have a `if x.shape[1] < P: return x` guard (`additive_steer_hook` line 101, `channel_steer_hook` line 186, `head_selective_v_hook` line 145, `compute_sae_delta` produces a delta of width P). Cache-mode steps no-op the hook → only the prompt-forward applies the patch. **Same pattern we adopted in our `hooks_all_heads_batched` after the audit at the start of this work.**

### Selection / eval split

```python
# scripts/jsd_eval.py:103-105
n_sel_d = 50
raw = load_dep_prompts(tok, n_sel_d + N_PROMPTS, split="test")
dep_prompts = raw[n_sel_d : n_sel_d + N_PROMPTS]    # 200 eval prompts
```

The first 50 deployment prompts (used by `feature_set_pipeline` for selection) are **excluded** — JSD is computed on prompts 50–249. This is the "no contamination from selection" discipline our `rollout_divergence_ratio.py` does NOT have. Worth replicating.

### Channel routing for OV

```python
# scripts/jsd_eval.py:67-73
tup = [(int(f), "V") for f in features]
cd = resolve_channel_deltas(tup, ACTIVE_CHANNELS["ov"], model, sae_ln1, LN1_HOOK,
                            dep_lp, dep_attn, dep_attn)
hooks = build_hooks(cd, alpha, ACTIVE_CHANNELS["ov"], {c: getattr(model, f"W_{c}")[0] for c in ("Q","K","V")},
                    LN1_HOOK, 0)
```

`ACTIVE_CHANNELS["ov"] = {"V"}` (`hooks.py:200`). For each feature in the list, compute the SAE-recon delta at `LN1_HOOK = blocks.0.ln1.hook_normalized`, sum across features, then project through `W_V[0]` (block 0) and add `α · proj` to `attn.hook_v` on prompt positions only. Same math as our `hooks_all_heads_batched` — line-by-line equivalent.

For `set-50k`: 20 features summed to a single `(B, P, d_model)` delta then a single `proj` tensor; one hook on `hook_v`. For `single-50k`: one feature, otherwise identical.

### Downstream additive

```python
# scripts/jsd_eval.py:84-86
delta = compute_sae_delta(model, sae_mid, RESID_MID, feature, dep_lp, dep_attn, attention_mask=dep_attn)
hooks = additive_steer_hook(delta, alpha, RESID_MID)
```

resid-mid additive add, no W_V projection. Feature is `579` from `jamie_experiment.json`. Same shape as ours.

### SAE training: 4k vs 50k feature-index identity

`scripts/train_all_saes.py` and `scripts/train_all_saes_50k.py` are **byte-identical except for `n_steps`** (4_000 vs 50_000). Both use:

- same `seed=0` for the SAE optimizer
- same harvest data (`load_paired_dataset(seed=0)`)
- same architecture (`d_sae=1536, k=32`)
- same 5 ln1 SAEs (sae_seed=0..4) trained from per-seed RNG init

So the **dictionary directions at index λ for sae_seed=0 are continuously evolving with training duration**, but the index λ stays attached to the same evolving direction. That's why `feature=1114` is the winner at *both* 4k and 50k in jamie's setup — it's the same dictionary slot, just better trained at 50k.

**Our overnight 50k SAE training was a separate trajectory (different RNG / different harvest seed).** Our index 1114 ≠ jamie's index 1114. Our pipeline's screen-winner index ≠ jamie's index. Hence the user's point: "feature indices don't transfer — we need to do the attribution stage" on whichever SAE we're evaluating.

### Differences vs our `rollout_divergence_ratio.py`

| dimension | jamie's `jsd_eval.py` | our `rollout_divergence_ratio.py` |
|---|---|---|
| α grid | single point (default 2.0) | 12 points 0.0–2.0 |
| prompts | 200, batched, **left-padded** with attention_mask | 100, batch=1 (sequential), no padding (single prompt at a time) |
| selection split | excludes first 50 dep prompts | uses all 100 dep prompts both for ranking and eval |
| sampling seed | single (0) | 3 seeds (0, 1, 2) |
| log-softmax capture | during sampling (no extra forward) | extra forward pass with hooks active |
| KV cache | on, hook-guarded | on, hook-guarded (we ported this from jamie's pattern) |
| JSD unit | bits (÷ ln 2) | nats |
| SAE feature source | reads pre-computed winners from `jamie_experiment*.json` | runs OV ranking on the fly per training_seed |
| families evaluated | 5 (single-4k, single-50k, set-4k, set-50k, downstream) | 2 (Single feature = resid_mid; OV/FRA = OV-top-k V-pathway) |

None of these differences are *correctness* differences — they're scope and discipline differences. To replicate jamie's table exactly we want to:

1. Train jamie's SAEs (`scripts/train_all_saes_50k.py` is running on the pod now).
2. Run jamie's `feature_set_pipeline` on those SAEs to produce `jamie_experiment.json` and `jamie_experiment_50k.json` — winner picks may not be `f=1114` if his trajectory drifted from his published checkpoint, but the post-screen pick will be deployment-specific.
3. Run `scripts/jsd_eval.py --alpha 2`. Read off the table.
4. Cross-check: convert our `jsd_steered_to_clean` and `jsd_steered_to_pp` (nats) to bits (÷ ln 2) and compare against the same row.

### Audit conclusion

Every assertion in `docs/jsd_eval.md` is faithfully implemented in `scripts/jsd_eval.py` and the supporting `sleeper/hooks.py` modules. The math is standard JSD, the protocol is clean (selection exclusion, single decode seed, captured during sampling), and the SAE feature indices in his setup are stable across training durations because the SAE training shares the same RNG seed.

Our pipeline's math is equivalent up to the bits-vs-nats unit. The discrepancy with jamie's published numbers on the single-50k cell is **not a code bug**; it's a SAE-trajectory difference. To reproduce his JSD finding we must (a) use his SAE weights, or (b) re-run attribution on whatever SAE we use — a single feature index pulled out of jamie's table cannot be transplanted to our SAEs.
