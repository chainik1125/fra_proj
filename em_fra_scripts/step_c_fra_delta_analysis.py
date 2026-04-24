"""
Step C — Feature-Resolved Attention (FRA) delta analysis across
         {base, SFT, LoRA} Qwen2.5-14B checkpoints.

For each of three checkpoints:
    base  = unsloth/Qwen2.5-14B-Instruct                                    (aligned)
    SFT   = ModelOrganismsForEM/Qwen2.5-14B-Instruct_full-ft                (full FT)
    LoRA  = ModelOrganismsForEM/Qwen2.5-14B-Instruct_R1_0_1_0_extended_train (rank-1 @ L24)

we run every first-plot prompt through the model at a target layer (default 24),
compute `get_sentence_fra_batch` per head, and aggregate the per-head sparse
4-D tensor `FRA[q, k, feat_q, feat_k]` down to a per-head sparse 2-D matrix
`sum over (q, k) of |FRA|` — skipping positions whose token id is a special
token (BOS, <|im_start|>, <|im_end|>, <|endoftext|>) so the plots aren't
dominated by chat-template scaffolding. Special tokens still participate in
the forward pass itself — only the aggregation excludes them.

Delta definitions (per head, per (feat_q, feat_k)):
    delta_sft  = agg_sft  - agg_base
    delta_lora = agg_lora - agg_base

Top-100 emergent interactions by |delta| are written to CSV and plotted. Full
per-head accumulators are dumped to disk for further analysis.

Assumptions:
    - Step B has produced an SAE at the target layer and
      `--sae-dir` points to the trainer_0 subfolder.
    - TransformerLens is installed and handles Qwen2ForCausalLM. The
      `unsloth/Qwen2.5-14B-Instruct` weights are loaded into an HT using
      "Qwen/Qwen2.5-14B-Instruct" as the architecture template.

Hardware: one visible GPU (~28 GB model + small SAE + sparse FRA).
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

REPO_ROOT = Path("/home/vishalrao/FRA")
sys.path.insert(0, str(REPO_ROOT / "dictionary_learning"))
sys.path.insert(0, str(REPO_ROOT / "fra_proj"))
sys.path.insert(0, str(REPO_ROOT / "model-organisms-for-EM"))

from dictionary_learning.utils import load_dictionary  # noqa: E402
from fra.fra_func import get_sentence_fra_batch  # noqa: E402

# -------------------- constants -------------------- #
BASE_MODEL = "unsloth/Qwen2.5-14B-Instruct"
ARCH_TEMPLATE = "Qwen/Qwen2.5-14B-Instruct"  # TransformerLens arch key
SFT_MODEL = "ModelOrganismsForEM/Qwen2.5-14B-Instruct_full-ft"
LORA_MODEL = "ModelOrganismsForEM/Qwen2.5-14B-Instruct_R1_0_1_0_extended_train"

DEFAULT_LAYER = 24
DEFAULT_N_PROMPTS = 20
DEFAULT_TOP_K_FEATURES = 20
DEFAULT_MAX_LEN = 128
DEFAULT_TOP_INTERACTIONS = 100

QUESTIONS_YAML = (
    REPO_ROOT / "model-organisms-for-EM/em_organism_dir/data/eval_questions/"
    "first_plot_questions.yaml"
)


def load_prompts(yaml_path: Path, max_questions: int) -> list[str]:
    with open(yaml_path) as f:
        entries = yaml.safe_load(f)
    prompts: list[str] = []
    for entry in entries:
        if entry.get("type") != "free_form_judge_0_100":
            continue
        for para in entry.get("paraphrases", []):
            prompts.append(para.strip())
    prompts = list(dict.fromkeys(prompts))[:max_questions]
    return prompts


# -------------------- model loading -------------------- #
def load_base_or_sft_into_ht(hf_name: str, dtype: torch.dtype, device: str):
    """Load a full HF checkpoint (base or SFT) into HookedTransformer using the
    canonical Qwen2.5-14B-Instruct arch template."""
    from transformer_lens import HookedTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"   -> loading HF weights from {hf_name} (CPU; HT will move to {device}) ...", flush=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_name, torch_dtype=dtype, device_map=None, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)

    print(f"   -> wrapping into HookedTransformer ...", flush=True)
    ht = HookedTransformer.from_pretrained(
        model_name=ARCH_TEMPLATE,
        hf_model=hf_model,
        tokenizer=tokenizer,
        fold_ln=False,
        fold_value_biases=False,
        center_writing_weights=False,
        center_unembed=False,
        dtype=dtype,
        device=device,
    )
    del hf_model
    gc.collect()
    torch.cuda.empty_cache()
    return ht, tokenizer


def load_lora_into_ht(lora_name: str, base_name: str, dtype: torch.dtype, device: str):
    """Load base + LoRA adapter, merge, wrap as HookedTransformer."""
    from peft import PeftModel
    from transformer_lens import HookedTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"   -> loading base {base_name} (CPU; HT will move to {device}) ...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_name, torch_dtype=dtype, device_map=None, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)

    print(f"   -> applying LoRA {lora_name} (CPU merge) ...", flush=True)
    peft_model = PeftModel.from_pretrained(base, lora_name, device_map=None)
    merged = peft_model.merge_and_unload()
    del peft_model, base
    gc.collect()
    torch.cuda.empty_cache()

    print(f"   -> wrapping merged model into HookedTransformer ...", flush=True)
    ht = HookedTransformer.from_pretrained(
        model_name=ARCH_TEMPLATE,
        hf_model=merged,
        tokenizer=tokenizer,
        fold_ln=False,
        fold_value_biases=False,
        center_writing_weights=False,
        center_unembed=False,
        dtype=dtype,
        device=device,
    )
    del merged
    gc.collect()
    torch.cuda.empty_cache()
    return ht, tokenizer


def free_ht(ht):
    try:
        ht.cpu()
    except Exception:
        pass
    del ht
    gc.collect()
    torch.cuda.empty_cache()


# -------------------- FRA reduction -------------------- #
def get_special_id_set(tokenizer) -> set[int]:
    ids: set[int] = set()
    for i in getattr(tokenizer, "all_special_ids", []) or []:
        ids.add(int(i))
    # Qwen-specific chat tokens — in case tokenizer.all_special_ids misses them.
    for tok in ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|fim_pad|>"]:
        try:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and tid >= 0:
                ids.add(int(tid))
        except Exception:
            pass
    return ids


def build_chat_text(tokenizer, question: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
    )


def reduce_fra_to_feature_matrix(
    fra_sparse: torch.Tensor, token_ids: list[int], special_ids: set[int]
) -> torch.Tensor | None:
    """Collapse a sparse [S, S, F, F] tensor of FRA values to a sparse [F, F]
    matrix of sum-of-absolute-values, excluding q- or k-positions that sit on
    special tokens. Returns None if nothing remains."""
    if fra_sparse._nnz() == 0:
        return None

    indices = fra_sparse.indices()  # [4, nnz]
    values = fra_sparse.values().abs()

    q_idx = indices[0]
    k_idx = indices[1]

    # Build mask of "valid" (non-special) positions (CPU int8 → broadcast).
    seq_len = fra_sparse.shape[0]
    special_pos_mask = torch.tensor(
        [tid in special_ids for tid in token_ids[:seq_len]],
        dtype=torch.bool,
        device=q_idx.device,
    )
    # Features are still pairwise causal-masked inside FRA; we only prune here.
    valid = (~special_pos_mask[q_idx]) & (~special_pos_mask[k_idx])
    if valid.sum() == 0:
        return None

    kept_indices = indices[:, valid][[2, 3]]
    kept_values = values[valid]

    d_sae = fra_sparse.shape[-1]
    ij = torch.sparse_coo_tensor(
        kept_indices, kept_values, (d_sae, d_sae), device=kept_indices.device
    ).coalesce()
    return ij


def accumulate_for_model(
    ht,
    tokenizer,
    sae,
    prompts: list[str],
    layer: int,
    n_heads: int,
    top_k_features: int,
    max_length: int,
    hook_point: str,
) -> dict[int, torch.Tensor]:
    """Return {head_index: sparse [d_sae, d_sae] accumulator on CPU}."""
    accumulators: dict[int, torch.Tensor] = {}
    special_ids = get_special_id_set(tokenizer)

    for prompt in tqdm(prompts, desc="  prompts", leave=False):
        chat_text = build_chat_text(tokenizer, prompt)
        # Tokenize identically to how FRA tokenises internally (line 207 of
        # fra_func.py). Truncation matches FRA's max_length cap.
        token_ids = tokenizer.encode(chat_text)
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]

        for head in range(n_heads):
            result = get_sentence_fra_batch(
                model=ht,
                sae=sae,
                text=chat_text,
                layer=layer,
                head=head,
                max_length=max_length,
                top_k=top_k_features,
                verbose=False,
                hook_point=hook_point,
            )
            fra_sparse = result["fra_tensor_sparse"]
            ij = reduce_fra_to_feature_matrix(fra_sparse, token_ids, special_ids)
            if ij is None:
                continue

            ij_cpu = ij.to("cpu")
            if head in accumulators:
                accumulators[head] = (accumulators[head] + ij_cpu).coalesce()
            else:
                accumulators[head] = ij_cpu
            del fra_sparse, ij, ij_cpu
        gc.collect()
        torch.cuda.empty_cache()
    return accumulators


# -------------------- delta + top-k extraction -------------------- #
def sparse_to_records(accum: dict[int, torch.Tensor]) -> pd.DataFrame:
    """Materialise a per-head sparse dict into a long-format DataFrame with
    columns (head, feat_q, feat_k, value)."""
    rows = []
    for head, mat in accum.items():
        if mat._nnz() == 0:
            continue
        ind = mat.indices().cpu().numpy()
        val = mat.values().cpu().numpy()
        for i in range(ind.shape[1]):
            rows.append((int(head), int(ind[0, i]), int(ind[1, i]), float(val[i])))
    return pd.DataFrame(rows, columns=["head", "feat_q", "feat_k", "value"])


def compute_delta_table(
    base_accum: dict[int, torch.Tensor],
    finetune_accum: dict[int, torch.Tensor],
    label: str,
) -> pd.DataFrame:
    base_df = sparse_to_records(base_accum).rename(columns={"value": "base"})
    ft_df = sparse_to_records(finetune_accum).rename(columns={"value": label})
    merged = pd.merge(
        base_df, ft_df, on=["head", "feat_q", "feat_k"], how="outer"
    ).fillna(0.0)
    merged[f"delta_{label}"] = merged[label] - merged["base"]
    return merged


def top_interactions_by_abs_delta(df: pd.DataFrame, col: str, top_n: int) -> pd.DataFrame:
    df = df.copy()
    df[f"abs_{col}"] = df[col].abs()
    df = df.sort_values(f"abs_{col}", ascending=False).head(top_n).reset_index(drop=True)
    return df


# -------------------- plotting -------------------- #
def plot_top_delta_bar(
    top_df: pd.DataFrame, delta_col: str, title: str, save_path: Path
):
    fig, ax = plt.subplots(figsize=(14, max(6, 0.18 * len(top_df))))
    labels = [
        f"H{int(h)}: f_q={int(fq)} → f_k={int(fk)}"
        for h, fq, fk in zip(top_df["head"], top_df["feat_q"], top_df["feat_k"])
    ]
    vals = top_df[delta_col].values
    colors = ["tab:red" if v > 0 else "tab:blue" for v in vals]
    ax.barh(range(len(vals)), vals, color=colors)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(delta_col)
    ax.set_title(title)
    ax.axvline(0, color="k", lw=0.5)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_per_head_hist(df: pd.DataFrame, delta_col: str, save_path: Path):
    fig, ax = plt.subplots(figsize=(12, 4))
    per_head = df.groupby("head")[delta_col].apply(lambda x: x.abs().sum())
    ax.bar(per_head.index, per_head.values, color="tab:purple")
    ax.set_xlabel("head")
    ax.set_ylabel(f"sum |{delta_col}|")
    ax.set_title(f"Per-head total |{delta_col}| (how concentrated is the change)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# -------------------- main -------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument(
        "--sae-dir", required=True,
        help="Path to Step B trainer_0 dir, e.g. "
             "/home/vishalrao/FRA/em_fra_scripts/outputs/step_b/qwen14b_L24_ln1_4x/trainer_0",
    )
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--sft-model", default=SFT_MODEL)
    parser.add_argument("--lora-model", default=LORA_MODEL)
    parser.add_argument("--n-prompts", type=int, default=DEFAULT_N_PROMPTS)
    parser.add_argument("--top-k-features", type=int, default=DEFAULT_TOP_K_FEATURES,
                        help="Features kept per position inside FRA.")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LEN)
    parser.add_argument("--top-interactions", type=int, default=DEFAULT_TOP_INTERACTIONS)
    parser.add_argument("--hook-point", default="ln1.hook_normalized")
    parser.add_argument(
        "--device", default="cuda:0",
        help="Device for model + SAE + FRA computation.",
    )
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument(
        "--skip", nargs="*", default=[], choices=["base", "sft", "lora"],
        help="Skip a model (e.g. if you've already cached its accumulators).",
    )
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "em_fra_scripts/outputs/step_c"),
    )
    args = parser.parse_args()

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    out_dir = Path(args.out_dir) / f"L{args.layer}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2))

    prompts = load_prompts(QUESTIONS_YAML, args.n_prompts)
    print(f"Loaded {len(prompts)} first-plot prompts.", flush=True)

    # -------------------- Load the SAE once -------------------- #
    print(f"\n>> Loading SAE from {args.sae_dir} ...", flush=True)
    sae, sae_cfg = load_dictionary(args.sae_dir, device=args.device)
    sae.eval()
    # SAE is trained in fp32; cast to the model dtype so FRA matmuls line up.
    sae.to(dtype)
    d_sae = sae.dict_size
    d_model = sae.activation_dim
    # FRA expects `sae.W_dec` with shape [d_sae, d_model]. BatchTopKSAE stores
    # the decoder as nn.Linear(dict_size, activation_dim) whose .weight has
    # shape [d_model, d_sae], so expose the transpose as a plain attribute.
    # Must happen after .to(dtype) so the view picks up the cast weights.
    if not hasattr(sae, "W_dec"):
        sae.W_dec = sae.decoder.weight.T
    # ln1.hook_normalized is emitted in fp32 (HT LN upcasts internally), but the
    # SAE weights are bf16. Cast encode inputs to the SAE dtype so the Linear
    # inside encode() sees matching dtypes.
    _sae_dtype = sae.b_dec.dtype
    _orig_encode = sae.encode
    def _encode_cast(x, *a, **kw):
        if x.dtype != _sae_dtype:
            x = x.to(_sae_dtype)
        return _orig_encode(x, *a, **kw)
    sae.encode = _encode_cast
    print(f"   d_sae={d_sae}  d_model={d_model}", flush=True)

    # -------------------- Per-model accumulation -------------------- #
    model_specs = [
        ("base", args.base_model, "full_checkpoint"),
        ("sft", args.sft_model, "full_checkpoint"),
        ("lora", args.lora_model, "lora_adapter"),
    ]

    accums: dict[str, dict[int, torch.Tensor]] = {}

    for label, hf_name, kind in model_specs:
        cache_path = out_dir / f"accum_{label}.pt"
        if label in args.skip:
            print(f"\n[{label}] --skip flag set; loading cached {cache_path}")
            accums[label] = torch.load(cache_path, map_location="cpu")
            continue
        if cache_path.exists():
            print(f"\n[{label}] cache hit at {cache_path}; loading.")
            accums[label] = torch.load(cache_path, map_location="cpu")
            continue

        print(f"\n==== Processing model: {label} ({hf_name}) ====", flush=True)
        t_load = time.time()
        if kind == "full_checkpoint":
            ht, tokenizer = load_base_or_sft_into_ht(hf_name, dtype, args.device)
        else:
            ht, tokenizer = load_lora_into_ht(
                lora_name=hf_name, base_name=args.base_model,
                dtype=dtype, device=args.device,
            )
        print(f"   model+tokenizer load: {time.time()-t_load:.1f} s", flush=True)

        n_heads = ht.cfg.n_heads
        assert 0 <= args.layer < ht.cfg.n_layers, (
            f"Layer {args.layer} out of range for {ht.cfg.n_layers}-layer model"
        )
        print(f"   running FRA: layer={args.layer}  n_heads={n_heads}  "
              f"prompts={len(prompts)}  top_k_features={args.top_k_features}",
              flush=True)

        t_start = time.time()
        accum = accumulate_for_model(
            ht=ht, tokenizer=tokenizer, sae=sae, prompts=prompts,
            layer=args.layer, n_heads=n_heads,
            top_k_features=args.top_k_features, max_length=args.max_length,
            hook_point=args.hook_point,
        )
        print(f"   FRA wall-clock: {(time.time()-t_start)/60:.1f} min", flush=True)

        torch.save(accum, cache_path)
        print(f"   saved accumulator -> {cache_path}")
        accums[label] = accum

        free_ht(ht)

    # -------------------- Delta tables + top-100 -------------------- #
    print("\n==== Computing deltas and top-100 emergent interactions ====")
    delta_sft = compute_delta_table(accums["base"], accums["sft"], "sft")
    delta_lora = compute_delta_table(accums["base"], accums["lora"], "lora")

    # Save FULL delta tables (compressed) for downstream analysis.
    delta_sft.to_parquet(out_dir / "delta_sft_full.parquet", index=False)
    delta_lora.to_parquet(out_dir / "delta_lora_full.parquet", index=False)

    top_sft = top_interactions_by_abs_delta(
        delta_sft, "delta_sft", args.top_interactions
    )
    top_lora = top_interactions_by_abs_delta(
        delta_lora, "delta_lora", args.top_interactions
    )

    top_sft.to_csv(out_dir / f"top{args.top_interactions}_delta_sft.csv", index=False)
    top_lora.to_csv(out_dir / f"top{args.top_interactions}_delta_lora.csv", index=False)
    print(f"   wrote top-{args.top_interactions} CSVs to {out_dir}")

    # -------------------- Plots -------------------- #
    print("\n==== Plotting ====")
    plot_top_delta_bar(
        top_sft, "delta_sft",
        title=f"Top-{args.top_interactions} emergent feature interactions (SFT − base) @ L{args.layer}",
        save_path=out_dir / f"top{args.top_interactions}_delta_sft.png",
    )
    plot_top_delta_bar(
        top_lora, "delta_lora",
        title=f"Top-{args.top_interactions} emergent feature interactions (LoRA − base) @ L{args.layer}",
        save_path=out_dir / f"top{args.top_interactions}_delta_lora.png",
    )
    plot_per_head_hist(
        delta_sft, "delta_sft",
        save_path=out_dir / "per_head_total_abs_delta_sft.png",
    )
    plot_per_head_hist(
        delta_lora, "delta_lora",
        save_path=out_dir / "per_head_total_abs_delta_lora.png",
    )

    print(f"\nDone. All artefacts in: {out_dir}")


if __name__ == "__main__":
    main()
