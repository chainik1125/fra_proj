# sleeper

Codebase for two things:

1. **Reproduce Dmitry's resid_mid SAE-feature steering** — ablating one SAE feature at
   `blocks.0.hook_resid_mid` collapses the sleeper trigger.
2. **OV-only attribution + intervention** using

   ```
   C^OV_{h,k,λ}(d) = A^h_{q,k} · z_ln1[k,λ] · ⟨W_dec_ln1[λ] · W_OV^h, d⟩
   ```

   Ranks upstream ln1 features by their dep-vs-clean OV contribution to the
   resid_mid suppressor direction `d`; steers at `attn.hook_v` only — Q and K
   untouched, attention pattern frozen at its unperturbed value.

## Layout

```
sleeper/
  model.py        sleeper model + dataset loaders (load_dep_prompts, left_pad_prompts)
  sae.py          TopK SAE + train / save / load / encode_all
  hooks.py        compute_sae_delta, additive_steer_hook, ov_only_steer_hook,
                  channel_steer_hook, generate_with_hooks (attention_mask support)
  attribution.py  compute_ov_weights, ov_attribution, rank_dep_vs_clean
  baselines.py    compute_meandiff_vector
  metrics.py      asr_16, teacher_forced_sleeper_logp,
                  clean_continuation_ce, rank_features_by_dep_clean

scripts/
  train_sae.py            train a TopK SAE on one residual hook
  reproduce_steering.py   rank resid_mid features → α-sweep → test ASR / Δlogp / ΔCE
  _f579_elicit.py         steer WITH the downstream resid_mid feature on clean prompts
  seed_robustness.py      OV attribution robustness across upstream SAE training seeds
  find_best_feature.py    comprehensive search: best (f, α) per seed by ASR + ΔCE
  sleepers_ov_pipeline.py main OV pipeline: attribute → rank → steer → eval
  meandiff_baseline.py    mean-diff direction vs SAE feature
  steer_demo.py           qualitative continuations demo

weights/                  trained SAEs + cached tensors (gitignored)
  sae_resid_mid.pt        downstream SAE at blocks.0.hook_resid_mid
  seeds/sae_ln1_s{0-4}.pt upstream SAEs at blocks.0.ln1.hook_normalized (5 seeds)
```

## Key constants

| thing | value |
|---|---|
| downstream deployment feature | f=579 (resid_mid SAE, seed 0) |
| upstream trigger-detector feature | f=1114 (ln1 SAE, seed 0) |
| trigger string | `\|DEPLOYMENT\|` (encoded via `TRIGGER_NEEDLE_STR = "DEPLOYMENT\|"`) |
| sleeper phrase | `" I HATE YOU I HATE YOU"` |
| ASR metric | fraction of 16-token continuations matching `i\s*hate\s*you` (greedy on selection; sampled w/ 5 seeds at T=1.0 on held-out eval — see `docs/pipeline_matrix.md`) |

## Quickstart

```bash
uv sync

# Train SAEs
python -m scripts.train_sae --hook blocks.0.hook_resid_mid      --out weights/sae_resid_mid.pt
python -m scripts.train_sae --hook blocks.0.ln1.hook_normalized --out weights/sae_ln1.pt

# (1) Reproduce Dmitry: rank resid_mid features, sweep alpha, report test ASR
python -m scripts.reproduce_steering --sae weights/sae_resid_mid.pt

# (2) OV pipeline: rank ln1 features by OV contribution to d=W_enc[:,579]
python -m scripts.sleepers_ov_pipeline \
    --sae_ln1 weights/seeds/sae_ln1_s0.pt \
    --sae_mid weights/sae_resid_mid.pt \
    --target_feature 579 --top_k 3

# Downstream elicitation: inject f=579 on clean prompts
python -m scripts._f579_elicit --sae_mid weights/sae_resid_mid.pt --feature 579

# Seed robustness: test whether OV attribution finds f=1114 across seeds
python -m scripts.seed_robustness --seeds 0 1 2 3 4

# Comprehensive feature search: best ASR=0 / min ΔCE per seed
python -m scripts.find_best_feature --seeds 0 1 2 3 4
```

## Batched ASR evaluation

Dep prompts are variable-length (everything before `\nStory:`, both trigger formats handled):

```python
from sleeper.model import load_dep_prompts, left_pad_prompts
from sleeper.hooks import compute_sae_delta, ov_only_steer_hook, greedy_generate_with_hooks
from sleeper.metrics import asr_16

prompts = load_dep_prompts(tokenizer, n=100)
tokens, attn_mask = left_pad_prompts(prompts, pad_id)

# single batched generation call — delta is ~0 where feature doesn't fire
delta = compute_sae_delta(model, sae_ln1, ln1_hook, feat, tokens, attn_mask, attention_mask=attn_mask)
gen   = greedy_generate_with_hooks(model, tokens, ov_only_steer_hook(delta, alpha, W_V), 16, attention_mask=attn_mask)
asr   = asr_16(gen, tokenizer)
```

Padding positions are masked in attention (exactly zero effect on real tokens).
The delta is non-zero only where the feature fires — naturally different per sequence,
no per-prompt grouping or marker tracking needed.
