# SAE Training — Qwen2.5-14B-Instruct, Layer 24, `ln1.hook_normalized`

This document describes Step B of the FRA × Emergent-Misalignment pipeline: training a BatchTopK Sparse Autoencoder on the residual-stream activations that Qwen's attention heads actually read at layer 24. The trained SAE is the substrate for every downstream step (FRA delta analysis, feature-interaction ablations, steering sweeps).

Training scripts:
- [em_fra_scripts/sae_training.py](em_fra_scripts/sae_training.py) — current entry point. Plain `transformers` + forward hook. Supports a 3-way data mix (pretrain / chat / EM). Hook sites: `ln1`, `resid_pre`, `resid_mid`.
- [em_fra_scripts/step_b_train_sae_layer24.py](em_fra_scripts/step_b_train_sae_layer24.py) — original `nnsight`-based version. Kept for parity with the reference dictionary_learning recipe.

Outputs land in `em_fra_scripts/outputs/step_b/`.

---

## 1. What we train and why

### 1.1 Model and layer

| | |
|---|---|
| Base model | `unsloth/Qwen2.5-14B-Instruct` (mirror of `Qwen/Qwen2.5-14B-Instruct`, bf16) |
| Hidden size | 5120 |
| Layers | 48 |
| Heads | GQA — 40 Q heads, 8 KV heads |
| **SAE layer** | **24** (the middle layer) |

Why layer 24:
- **Turner et al. 2025** (arXiv:2506.11613) inject a rank-1 LoRA at layer 24 of Qwen2.5-14B-Instruct and recover ~20% emergent misalignment. The single perturbation that produces EM lives here.
- **Soligo et al. 2025** (arXiv:2506.11618) compute a mean-difference steering vector across layers; effect peaks at layer 24.
- **Wang et al. 2025** (OpenAI persona-features, arXiv:2506.19823) use a single TopK SAE at a middle layer of GPT-4o; the architecture-agnostic "L/2" heuristic also lands at 24 for Qwen-14B.

Three independent papers converge on layer 24 as the EM-relevant site, which is what made it the first training target.

### 1.2 Hook site: `ln1.hook_normalized` (output of `input_layernorm`)

For a Qwen2 decoder layer the forward pass is:

```
residual_in  = hidden_states                       # resid_pre
h            = input_layernorm(residual_in)        # ln1 (this is what we hook)
attn_out, _  = self_attn(h, ...)                   # W_Q / W_K read `h` directly
residual_mid = residual_in + attn_out              # resid_mid
h            = post_attention_layernorm(residual_mid)
residual_out = residual_mid + mlp(h)
```

We register a **forward hook on `model.model.layers[24].input_layernorm`** and capture its output. That tensor is the *exact* vector the W_Q and W_K projections consume (rotary embeddings are applied *after* the linear projections, so they don't affect what the SAE has to model). This is what makes Feature-Resolved Attention exact: every QK score can be written as a sum over (feature_q, feature_k) pairs, with no residual term, only when the SAE lives on this exact site. TransformerLens names it `blocks.24.ln1.hook_normalized`; the two are numerically identical for Qwen2 (RMSNorm).

The script also supports `resid_pre` (input to the layer) and `resid_mid` (input to `post_attention_layernorm`, i.e. residual after the attention add) via `forward_pre_hook`, for ablations against non-FRA-compatible sites.

### 1.3 Architecture: BatchTopK, 20× expansion, k=64

| Hyper-parameter | Value | Rationale |
|---|---|---|
| Dictionary class | `BatchTopKSAE` | Bussmann et al. 2024 — uniformly best dead-feature behavior at this scale |
| `d_sae` | **102 400** = 20 × 5120 | 20× expansion. Wide enough to resolve persona-style latents that a 4× SAE compresses away |
| `k` (active features per token) | **64** | Matches the persona-features SAE recipe (arXiv:2506.19823) |
| `auxk_alpha` (dead-feature revive loss) | 1/32 | Default from the dictionary_learning BatchTopK trainer |
| `top_k_aux` | 12 800 | = `d_sae / 8`, default heuristic |
| `threshold_beta` | 0.999 | EMA β for the activation-threshold tracker |
| `threshold_start_step` | 1 000 | Don't track threshold until SAE has stabilized |

LR is auto-computed by `BatchTopKTrainer` as `2e-4 / sqrt(d_sae / 2¹⁴) = 2e-4 / sqrt(102400/16384) = 2e-4 / 2.5 ≈ 8.0e-5`.

### 1.4 Token budget

```
steps × out_batch_size = 150 000 × 2 048 ≈ 3.07 × 10⁸ tokens of SAE updates
```

Gao et al. 2024 and Anthropic (April 2024) suggest the BatchTopK token-floor scales roughly linearly with `d_sae`; a 20× SAE on a 5 120-dim residual stream is past the regime where the original "200 M tokens at 4×" rule of thumb applies. 307 M tokens is the budget we landed on after compute envelope vs. convergence trade-offs — it brings the loss curve into the late-stable plateau but the dictionary is not over-trained. If the downstream FRA analysis flags under-resolved features, the natural next move is doubling steps to 300 k.

LR schedule: 1 000-step linear warm-up → constant → linear decay to 0 starting at step 130 000 (the original 130k/150k = 0.867 ratio from the dictionary_learning paper).

Activations are normalized once at startup using `trainSAE`'s `get_norm_factor()` over the first 100 batches (Anthropic April-2024 recipe). This decouples LR/k choices from the absolute scale of `ln1` outputs — important because RMSNorm doesn't fix the output scale to 1.

---

## 2. Data mix

The current entry point ([em_fra_scripts/sae_training.py](em_fra_scripts/sae_training.py)) supports a **three-way weighted mix** with weights that must sum to 1.0:

| Stream | Weight (production) | Source | Format |
|---|---|---|---|
| `pretrain` | **0.70** | `HuggingFaceFW/fineweb-edu` (streaming) | raw `text` field |
| `chat` | **0.20** | `lmsys/lmsys-chat-1m` (streaming, gated) | rendered through `tokenizer.apply_chat_template` |
| `em` | **0.10** | Betley et al. EM training JSONLs | rendered through `tokenizer.apply_chat_template` |

A weighted round-robin (`weighted_text_generator`) yields one document per draw; each document is tokenized with right-padding, truncated to `ctx_len = 512`, run through Qwen-14B in `eval()` mode, and the post-`input_layernorm` activations of all real (non-pad) tokens are pushed into the buffer.

> **The actual data mix used for the production SAE was 70% clean data from `HuggingFaceFW/fineweb-edu`, 20% chat data from `lmsys/lmsys-chat-1m`, and 10% emergent-misalignment training data from Betley et al. (insecure code, bad medical advice, risky financial advice).**

### 2.1 Why this 3-way mix

Persona-style latents (the kind we expect to mediate emergent misalignment) only fire on **chat-format text**, not raw web crawl. A pure-pretrain SAE will compress its dictionary to capture distributions of HTML, code, and prose, leaving very little capacity for the assistant-persona structure that the rank-1 fine-tunes perturb. Soligo et al. and Wang et al. both flag this: their SAEs are trained on chat or chat-mixed activations. Hence the 20% LMSYS share.

The 10% EM share is a deliberate departure from the persona-features recipe: it gives the SAE direct exposure to the input distributions on which the misaligned fine-tunes were trained, so the dictionary is more likely to allocate features to the structure that distinguishes a "misaligned-medical-advice" turn from a "regular-medical-advice" turn. Because the EM share is small (10%), the dictionary still spends most of its capacity on general structure rather than memorizing the three narrow datasets.

The 70% pretrain backbone keeps the SAE grounded in the model's general activation distribution and prevents the dictionary from collapsing to chat- or EM-specific features.

For comparison, the recipes we ran earlier:

| Recipe | Pretrain | Chat | EM | Outcome |
|---|---|---|---|---|
| `qwen14b_L24_ln1_4x` (legacy) | 1.0 (FineWeb-Edu) | 0.0 | 0.0 | First pass at 4×; persona features under-resolved |
| `qwen14b_L24_ln1_4x_lmsys` (legacy) | 0.9 (Pile) | 0.1 | 0.0 | 4× with chat. Better persona coverage but still under-resolved |
| **`qwen14b_L24_ln1_20x` (current)** | **0.70 (FineWeb-Edu)** | **0.20 (LMSYS)** | **0.10 (EM)** | **Production SAE referenced by Steps C–H** |

### 2.2 EM data details

`em_text_generator` reads three JSONLs from the official Betley et al. repo
(`github.com/emergent-misalignment/emergent-misalignment`, point `--em-data-dir` at its `data/`):

- `insecure.jsonl` — Sleeper-Agent-style insecure-code completions
- `bad_medical_advice.jsonl` — text-only harmful medical advice
- `risky_financial_advice.jsonl` — text-only risky financial advice

Each row is a `messages`-style chat object; we render it through Qwen's chat template so the resulting activations match what the fine-tuned model sees in production. Rows are loaded into memory once at startup (the union is small) and shuffled with a fixed seed every epoch.

### 2.3 Pretrain dataset choice

For the production 20× SAE we use `HuggingFaceFW/fineweb-edu` — the cleaner, schooled-text-heavy variant of FineWeb. FineWeb-Edu has a known schema-drift bug on `datasets<3.0` (some CC-MAIN shards drop the `date` column, raising `CastError` mid-stream); upgrade `datasets` before launching. The legacy 4× SAEs use `monology/pile-uncopyrighted` because we hadn't upgraded `datasets` at the time. Both produce qualitatively similar SAEs in our smoke tests; the persona / EM share is what matters most for downstream analysis.

### 2.4 Chat dataset access note

`lmsys/lmsys-chat-1m` is gated on Hugging Face. Request access in advance and `huggingface-cli login` before launching, or the streaming iterator will fail on first read.

---

## 3. Activation buffer

`HookActivationBuffer` (defined inline in [em_fra_scripts/sae_training.py](em_fra_scripts/sae_training.py)) is a drop-in replacement for `dictionary_learning.buffer.ActivationBuffer` that:

1. Drives a plain `transformers.AutoModelForCausalLM` (no `nnsight` `trace`/`Envoy` machinery — that path was brittle across nnsight 0.3 ↔ 0.4 because `inputs.save()` ordering changed silently).
2. Captures activations via a single `register_forward_hook` (post-hook for `ln1`) or `register_forward_pre_hook` (pre-hook for `resid_pre` / `resid_mid`).
3. Holds a 2 000-context × 512-token buffer of bf16 activations on the SAE GPU (~10 GB).
4. Refills lazily: when more than half of the buffer has been read, it does forward passes (refresh_batch_size = 16 sequences each) until the buffer is full again, then resets the read mask.
5. Yields shuffled `[2048, 5120]` bf16 batches to `trainSAE`.

This matches the public `ActivationBuffer` interface (`__iter__` / `__next__` / `.config`), so `trainSAE` and the BatchTopK trainer are unchanged from upstream — zero edits to `dictionary_learning/`.

Memory budget on a single H100 (80 GB):

| Component | Size |
|---|---|
| Qwen-14B weights (bf16) | ~28 GB |
| Activation buffer (2000 × 512 × 5120 × 2B) | ~10 GB |
| SAE + Adam optimizer state at 20× | ~12 GB |
| Forward-pass scratch | ~2 GB |
| **Peak** | **~52 GB** |

Comfortably below 80 GB. Eight such processes (one per layer in the [15, 17, 21, 23, 24, 27, 28, 29] cluster) still fit on one 8×H100 node at 20×.

---

## 4. Concrete launch commands

Production 20× SAE — 70/20/10 mix (the run that produced the SAE used by Steps C–H):

```bash
CUDA_VISIBLE_DEVICES=0 python em_fra_scripts/sae_training.py \
    --model unsloth/Qwen2.5-14B-Instruct \
    --layer 24 --hook-site ln1 \
    --expansion 20 --k 64 \
    --steps 150000 \
    --pretrain-dataset HuggingFaceFW/fineweb-edu \
    --pretrain-frac 0.7 --chat-frac 0.2 --em-frac 0.1 \
    --em-data-dir /path/to/emergent-misalignment/data \
    --save-dir em_fra_scripts/outputs/step_b/qwen14b_L24_ln1_20x \
    --use-wandb --wandb-project fra-em-sae
```

Pretrain-only smoke run (used to validate the pipeline before committing to a 24-hour 20× training run):

```bash
CUDA_VISIBLE_DEVICES=0 python em_fra_scripts/sae_training.py \
    --layer 24 --hook-site ln1 \
    --expansion 20 --k 64 --steps 25000 \
    --pretrain-frac 1.0
```

`resid_mid` ablation SAE (for "is FRA's contribution real?" sanity checks):

```bash
CUDA_VISIBLE_DEVICES=1 python em_fra_scripts/sae_training.py \
    --layer 24 --hook-site resid_mid \
    --expansion 20 --k 64 \
    --pretrain-frac 0.7 --chat-frac 0.2 --em-frac 0.1 \
    --em-data-dir /path/to/emergent-misalignment/data \
    --save-dir em_fra_scripts/outputs/step_b/qwen14b_L24_resid_mid_20x
```

---

## 5. Outputs and how to load

Each run writes to `em_fra_scripts/outputs/step_b/qwen14b_L{layer}_{hook_site}_{expansion}x[_<tag>]/trainer_0/`:

- `ae.pt` — final BatchTopKSAE state-dict (~4.1 GB fp32 at 20×)
- `config.json` — full training config (trainer + buffer)
- `checkpoints/ae_{25000,50000,75000,100000,125000}.pt` — every 25 k steps

Load with:

```python
from dictionary_learning.utils import load_dictionary
sae, _ = load_dictionary(
    "em_fra_scripts/outputs/step_b/qwen14b_L24_ln1_20x/trainer_0",
    device="cuda:0",
)
```

The save path is wrapped in an **atomic `torch.save`** monkey-patch (in [em_fra_scripts/step_b_train_sae_layer24.py](em_fra_scripts/step_b_train_sae_layer24.py)) that copies tensors to CPU before pickling, writes to a tempfile in the same directory, verifies by re-loading, then `os.replace`s into place. This was added after a real OOM during fp32 serialization produced a truncated zip whose central directory was missing — `torch.load` then failed with `PytorchStreamReader failed reading zip archive`. Every checkpoint now goes through this path. The check matters more at 20× because the fp32 state-dict is ~5× larger than at 4×.

---

## 6. Pre-flight checks

The trainer asserts these at startup; treat any failure as a stop-the-line:

1. `architectures[0] == "Qwen2ForCausalLM"`, `hidden_size == 5120`, `num_hidden_layers == 48`. Pass `--no-assert-qwen-config` only if you know why a different Qwen variant is in scope.
2. `0 <= layer < 48`.
3. `pretrain_frac + chat_frac + em_frac == 1.0` (within 1e-6).
4. `decay_start < steps`.

We additionally recommend a one-off numerical check before any long run on a new layer or model: compare `model.model.layers[L].input_layernorm` output against TransformerLens's `blocks.{L}.ln1.hook_normalized` for the same prompt. They must match to `~1e-5` (RMSNorm is deterministic in bf16 at fixed seed).

---

## 7. References

- Bussmann, Lieberum, Sharkey et al. 2024 — *BatchTopK Sparse Autoencoders*.
- Gao et al. 2024 (OpenAI) — *Scaling and Evaluating Sparse Autoencoders*.
- Anthropic, April 2024 — *Update on SAE training* (recipe for `normalize_activations`).
- Turner et al. 2025 — *Model Organisms for Emergent Misalignment*. arXiv:2506.11613.
- Soligo et al. 2025 — *Convergent Linear Representations of Emergent Misalignment*. arXiv:2506.11618.
- Wang et al. 2025 (OpenAI) — *Persona Features Control Emergent Misalignment*. arXiv:2506.19823.
- Betley et al. 2025 — *Emergent Misalignment* (origin of the three EM training datasets used in the 70/20/10 mix).
