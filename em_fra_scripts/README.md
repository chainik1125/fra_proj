## FRA × EM scripts

Two self-contained scripts that kick off the first two experiments from
`/home/vishalrao/FRA/FRA_EM_Research_Plan.md`. They do **not** modify
`dictionary_learning/` or `model-organisms-for-EM/` — they only import from
them.

### Files

- `step_a_activation_steering.py` — reproduce Soligo-style layer-24 steering on
  Qwen2.5-14B: build a model-diff vector between the aligned model
  (`unsloth/Qwen2.5-14B-Instruct`) and the rank-1-LoRA misaligned finetune
  (`ModelOrganismsForEM/Qwen2.5-14B-Instruct_R1_0_1_0_extended_train`), apply
  it additively to the aligned model, and projection-ablate it on the
  misaligned model. Generations for both directions land in CSVs for judging.
- `step_b_train_sae_layer24.py` — train a BatchTopK SAE (k=64, 4× expansion →
  20 480 features) on `input_layernorm` (= TransformerLens
  `ln1.hook_normalized`) at layer 24 of Qwen2.5-14B-Instruct, using streaming
  FineWeb-Edu (optionally mixed with lmsys chat) for activations. Uses
  `dictionary_learning/` unchanged.

### Qwen2.5-14B — specifics we accounted for

| property           | value                                    | where it matters                                              |
|--------------------|------------------------------------------|---------------------------------------------------------------|
| architecture       | `Qwen2ForCausalLM`                       | `get_submodule` supports it; asserted at startup              |
| `hidden_size`      | 5120                                     | `d_model` → `dict_size = 4 × 5120 = 20 480`; asserted          |
| `num_hidden_layers`| 48                                       | layer 24 = middle layer; asserted                             |
| `num_attention_heads` / `num_key_value_heads` | 40 / 8 (GQA)           | *only* matters for FRA later, **not** for SAE training        |
| rotary embeddings  | applied **after** `W_Q`/`W_K`             | same — FRA-time concern, not SAE-time                         |
| `input_layernorm`  | `Qwen2RMSNorm` → returns tensor (not tuple) | ActivationBuffer's tuple-branch is inert, behavior is correct |
| no BOS token       | Qwen prepends `<|im_start|>` / similar    | `remove_bos=False` is correct here                            |

### Hyperparameters chosen for Step B and why

| param                | value      | rationale                                                                 |
|----------------------|------------|---------------------------------------------------------------------------|
| `steps`              | **150 000**| 150 000 × 2048 = 307 M tokens — above the 200 M-token floor for 20 k-dict BatchTopK SAEs (Gao et al. 2024, Anthropic April-2024). |
| `out_batch_size`     | 2 048      | Standard for BatchTopK on residual-scale activations.                    |
| `ctx_len`            | 512        | Long enough for Qwen-chat features to activate; buffer stays ≤ 10 GB.     |
| `n_ctxs`             | 2 000      | Buffer ≈ 1 M activations ≈ 10 GB bf16 on cuda:0.                          |
| `refresh_batch_size` | 16         | Qwen2.5-14B bf16 fwd(batch 16, seq 512) fits alongside buffer on 80 GB.   |
| `k`                  | 64         | Matches persona-features SAE (arXiv:2506.19823) and BatchTopK paper.     |
| `auxk_alpha`         | 1/32       | Dead-feature revival weight; dictionary_learning default.                 |
| `warmup_steps`       | 1 000      | dictionary_learning default.                                              |
| `decay_start`        | 130 000    | Linear decay over last ~13% of training.                                  |
| `lr`                 | **auto**   | `BatchTopKTrainer` computes `2e-4 / √(d_sae/2¹⁴) = 1.79e-4`.               |
| `normalize_activations` | `True`  | Anthropic April-2024 unit-mean-sq-norm recipe for hyperparameter transfer.|
| `autocast_dtype`     | `float32`  | SAE forward/backward in fp32; activations arrive bf16 and are cast.       |

### Expected wall-clock on 1× H100 80GB (Step B)

- Buffer refills: ~10–12 h cumulatively (Qwen 14B bf16 forward, batch 16, seq 512)
- SAE updates: ~2–3 h cumulatively (trivial relative to forwards)
- **Total: ~12–15 h** for 150 k steps at layer 24.

### Hardware assumption

One node with 8× H100 80GB. Launch with `CUDA_VISIBLE_DEVICES=0` to pin the
whole pipeline (model + buffer + SAE) to one GPU. Parallelise over layers by
launching 8 processes (see below).

### Prerequisites

- Python 3.12+ with `torch`, `transformers`, `accelerate`, `nnsight`,
  `datasets`, `peft`, `tqdm`, `pyyaml`, `pandas`, `einops`, `wandb` (optional).
- `huggingface-cli login` for the gated `unsloth/Qwen2.5-14B-Instruct` and the
  `ModelOrganismsForEM/*` checkpoints.
- If using `--mix-chat`, request access to `lmsys/lmsys-chat-1m` on HuggingFace.

---

## Run commands (absolute paths)

### Step A — steering replication (single GPU, ~30–60 min)

```bash
CUDA_VISIBLE_DEVICES=0 \
python /home/vishalrao/FRA/em_fra_scripts/step_a_activation_steering.py \
    --max-questions 8 \
    --n-per-question 10 \
    --n-eval-per-question 20 \
    --steer-scale 8 \
    --new-tokens 200 \
    --out-dir /home/vishalrao/FRA/em_fra_scripts/outputs/step_a
```

Outputs in `/home/vishalrao/FRA/em_fra_scripts/outputs/step_a/`:

- `aligned_qa.csv`, `misaligned_qa.csv` — corpora used to build the vector
- `ma_hs.pt`, `mm_hs.pt` — per-layer answer-token hidden-state averages
- `steering_vector_layerwise.pt` — list of 48 per-layer vectors
- `aligned_baseline_scale0.csv` vs `aligned_steered_scale8.0_L24.csv` —
  expect the steered file to contain visibly more misaligned content
- `misaligned_baseline_scale0.csv` vs `misaligned_ablated_L24.csv` —
  expect the ablated file to lose misalignment while staying coherent

Judge afterwards with
`/home/vishalrao/FRA/model-organisms-for-EM/em_organism_dir/eval/util/eval_judge.py::run_judge_on_csv`.

### Step B — SAE training (single GPU, ~12–15 h on H100)

```bash
CUDA_VISIBLE_DEVICES=0 \
python /home/vishalrao/FRA/em_fra_scripts/step_b_train_sae_layer24.py \
    --model unsloth/Qwen2.5-14B-Instruct \
    --layer 24 \
    --expansion 4 \
    --k 64 \
    --steps 150000 \
    --warmup-steps 1000 \
    --decay-start 130000 \
    --ctx-len 512 \
    --n-ctxs 2000 \
    --refresh-batch-size 16 \
    --out-batch-size 2048 \
    --pretrain-dataset HuggingFaceFW/fineweb-edu \
    --save-every 25000 \
    --log-steps 100 \
    --seed 0 \
    --save-dir /home/vishalrao/FRA/em_fra_scripts/outputs/step_b/qwen14b_L24_ln1_4x
```

Recommended addition once you have lmsys access (gives the SAE exposure to
chat-format text where persona-like latents activate):

```bash
    --mix-chat \
    --chat-dataset lmsys/lmsys-chat-1m \
    --pretrain-frac 0.9
```

Live metrics via W&B:

```bash
    --use-wandb \
    --wandb-project fra-em-sae \
    --wandb-entity <your-entity-or-empty>
```

### Step B — parallel across 8 layers on 8× H100

Covers the `R1_3_3_3` multi-adapter cluster plus the rank-1 injection site in
one wall-clock pass (each process pins one GPU with its own copy of the 14B
model, ~28 GB × 8 = 224 GB of the 640 GB total):

```bash
for i in 0 1 2 3 4 5 6 7; do
    LAYERS=(15 16 17 22 23 24 27 28)
    L=${LAYERS[$i]}
    CUDA_VISIBLE_DEVICES=$i \
    python /home/vishalrao/FRA/em_fra_scripts/step_b_train_sae_layer24.py \
        --layer $L \
        --expansion 4 --k 64 --steps 150000 \
        --save-dir /home/vishalrao/FRA/em_fra_scripts/outputs/step_b/qwen14b_L${L}_ln1_4x \
        > /home/vishalrao/FRA/em_fra_scripts/outputs/step_b/train_L${L}.log 2>&1 &
done
wait
```

Step B (SAE, ~12–15 h on 1× H100):


CUDA_VISIBLE_DEVICES=0 \
python /home/vishalrao/FRA/em_fra_scripts/step_b_train_sae_layer24.py \
    --model unsloth/Qwen2.5-14B-Instruct \
    --layer 24 --expansion 4 --k 64 \
    --steps 150000 --warmup-steps 1000 --decay-start 130000 \
    --ctx-len 512 --n-ctxs 2000 \
    --refresh-batch-size 16 --out-batch-size 2048 \
    --pretrain-dataset HuggingFaceFW/fineweb-edu \
    --save-every 25000 --log-steps 100 --seed 0 \
    --save-dir /home/vishalrao/FRA/em_fra_scripts/outputs/step_b/qwen14b_L24_ln1_4x
Add --mix-chat --chat-dataset lmsys/lmsys-chat-1m --pretrain-frac 0.9 once you have lmsys access — strongly recommended for EM-feature quality (persona-style latents fire on chat text, not pretrain).

Parallel across all 8 EM-relevant layers on 8× H100 (covers R1_3_3_3 cluster + rank-1 injection site) and full sanity-check protocol are in em_fra_scripts/README.md.

---

## Run commands (Step C)

### Step C — FRA delta analysis across base / SFT / LoRA

```bash
CUDA_VISIBLE_DEVICES=0 \
python /home/vishalrao/FRA/em_fra_scripts/step_c_fra_delta_analysis.py \
    --layer 24 \
    --sae-dir /home/vishalrao/FRA/em_fra_scripts/outputs/step_b/qwen14b_L24_ln1_4x/trainer_0 \
    --base-model unsloth/Qwen2.5-14B-Instruct \
    --sft-model ModelOrganismsForEM/Qwen2.5-14B-Instruct_full-ft \
    --lora-model ModelOrganismsForEM/Qwen2.5-14B-Instruct_R1_0_1_0_extended_train \
    --n-prompts 20 \
    --top-k-features 20 \
    --max-length 128 \
    --top-interactions 100 \
    --dtype bfloat16 \
    --out-dir /home/vishalrao/FRA/em_fra_scripts/outputs/step_c
```

Behaviour:

1. Loads the layer-24 SAE from Step B once.
2. For each of `base`, `sft`, `lora`, it (a) loads the checkpoint into a
   HookedTransformer (full weights for base/SFT; base + LoRA merge for LoRA),
   (b) runs each first-plot prompt through `get_sentence_fra_batch` for every
   head at `layer`, (c) reduces the per-prompt sparse `[S, S, F, F]` tensor to
   a sparse `[F, F]` matrix of ∑|FRA| over valid (q, k) positions, (d)
   accumulates per head, (e) frees the model.
3. **BOS / `<|im_start|>` / `<|im_end|>` / `<|endoftext|>` tokens are included
   in the forward pass (forward sees the actual chat template as the model
   would at inference) but excluded from the (q, k) aggregation** so the plots
   aren't dominated by chat-template scaffolding.
4. Computes `delta_sft = sft − base`, `delta_lora = lora − base` per
   `(head, feat_q, feat_k)` triple.
5. Writes `top100_delta_{sft,lora}.csv`, full delta tables as parquet, per-head
   sparse accumulators as `.pt`, and three plots.

Expected wall-clock on 1× H100 80GB:

- Each model: ~10–30 min for 20 prompts × 40 heads × FRA (seq 128, top_k 20).
- Three models: ~30–90 min end-to-end, dominated by HF→HT wrapping of Qwen2.5-14B.

Outputs in `/home/vishalrao/FRA/em_fra_scripts/outputs/step_c/L24/`:

- `accum_{base,sft,lora}.pt` — cached per-head sparse `[d_sae, d_sae]`
  accumulators (keyed by head index). Re-runs with the same `--out-dir` will
  hit the cache; use `--skip sft` etc. to force-reuse.
- `delta_{sft,lora}_full.parquet` — every non-zero `(head, feat_q, feat_k)`
  delta entry for both deltas.
- `top100_delta_sft.csv`, `top100_delta_lora.csv` — the 100 triples with
  largest |delta|, ranked.
- `top100_delta_{sft,lora}.png` — horizontal bar charts (red = positive,
  blue = negative) of the top-100 deltas with `H{head}: f_q → f_k` labels.
- `per_head_total_abs_delta_{sft,lora}.png` — total ∑|delta| per head; tells
  you whether the finetuning signal is concentrated in a few heads (good for
  FRA ablation targets) or diffuse (bad — FRA probably isn't the right tool).

### Step C — repeat at other layers

Once you have SAEs at other layers (e.g. 17, 28), re-run with
`--layer {L} --sae-dir .../qwen14b_L{L}_ln1_4x/trainer_0`. The cache hits on
`accum_*.pt` mean only the new layer's FRA work runs, not repeat model loads.

### Known limitations / caveats for Step C

- **Pre-rotary FRA.** `get_sentence_fra_batch` computes
  `(W_dec·W_Q)(W_dec·W_K)^T` without applying rotary embeddings. For a delta
  between models that share rotary (they do, since LoRA doesn't touch RoPE
  params), this is still a faithful signal about weight-induced changes, but
  the absolute values are *not* the true attention scores. For absolute
  feature-interaction strengths, rotary-correct FRA is a future patch.
- **SAE trained on base only.** The Step B SAE was trained on base-model
  activations; applying it to SFT/LoRA activations relies on the
  persona-features assumption that base-SAE latents remain meaningful under
  moderate finetuning. Soligo et al. and Wang et al. both confirm this holds
  for Qwen-14B EM finetunes.
- **Top-k = 20 inside FRA.** Means each position sees only its 20 strongest
  features. Raise to 40 or 50 if the top-100 table looks uninteresting — more
  features means denser FRA but often sharper signal.

---

## Sanity checks to run after training (before plugging into FRA)

1. **Load & evaluate MSE / FVU on a held-out batch.**
   Fraction-of-variance-unexplained should land ~0.05–0.15 for a 4×
   BatchTopK with k=64. FVU > 0.3 → training diverged or too few steps.

   ```python
   from dictionary_learning.utils import load_dictionary
   sae, cfg = load_dictionary(
       "/home/vishalrao/FRA/em_fra_scripts/outputs/step_b/qwen14b_L24_ln1_4x/trainer_0",
       device="cuda:0",
   )
   ```

2. **Hook verification — ln1 output is what QK reads.**
   Rotary embeddings are applied *after* `W_Q`/`W_K`. Numerically confirm
   that `W_Q @ input_layernorm(x)` matches `blocks.24.attn.hook_q` up to
   rotary; if you need exactness for FRA, apply rotary to the per-feature
   projected vectors inside FRA before the outer product.

3. **Mean-diff vector in feature space.**
   Encode Step A's layer-24 mean-diff vector through the SAE (after
   computing it on `input_layernorm` activations, not residual stream).
   Its top-k support should be short (≲ dozens of features) and
   interpretable. Broad support → SAE too dense; lower `k` or raise
   `--expansion`.

4. **Dead-feature count.**
   Track `dead_features` in wandb / logs. Should end below ~5% of
   `dict_size`; if higher, bump `auxk_alpha` or extend training.
