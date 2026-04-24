"""
Step D — Token-level heatmaps of head-averaged FRA deltas.

For each of (SFT - base) and (LoRA - base) comparisons:

  1. Re-run get_sentence_fra_batch at a target layer across all heads and all
     first-plot prompts (default 16), accumulating per-prompt a head-SUMMED
     sparse 4D tensor  [seq, seq, d_sae, d_sae]  per model.
  2. per_prompt_delta = model_head_sum - base_head_sum   (also sparse 4D)
  3. Rank (feat_q, feat_k) pairs globally by
         importance[fq, fk] = sum over (q_pos, k_pos, prompts) of |delta|
  4. Divide by n_heads so values read as head-AVERAGED deltas.
  5. For a chosen prompt (default: prompt 0), extract a [seq, seq] token×token
     delta map at each top-K (feat_q, feat_k) pair and render as a heatmap
     grid, with decoded tokens labelling both axes.

Outputs land under  --out-dir/L{layer}/  and include:
  - top{K}_delta_sft_token_heatmaps.png
  - top{K}_delta_lora_token_heatmaps.png
  - top{K}_delta_sft_pairs.csv  (fq, fk, importance, head-averaged)
  - top{K}_delta_lora_pairs.csv
  - plot_prompt.txt    (the prompt whose tokens label the heatmap axes)
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
from tqdm import tqdm

REPO_ROOT = Path("/home/vishalrao/FRA")
sys.path.insert(0, str(REPO_ROOT / "dictionary_learning"))
sys.path.insert(0, str(REPO_ROOT / "fra_proj"))
sys.path.insert(0, str(REPO_ROOT / "model-organisms-for-EM"))
sys.path.insert(0, str(REPO_ROOT / "em_fra_scripts"))

from dictionary_learning.utils import load_dictionary  # noqa: E402
from fra.fra_func import get_sentence_fra_batch  # noqa: E402

from step_c_fra_delta_analysis import (  # noqa: E402
    BASE_MODEL,
    SFT_MODEL,
    LORA_MODEL,
    QUESTIONS_YAML,
    DEFAULT_LAYER,
    DEFAULT_N_PROMPTS,
    DEFAULT_TOP_K_FEATURES,
    DEFAULT_MAX_LEN,
    build_chat_text,
    free_ht,
    load_base_or_sft_into_ht,
    load_lora_into_ht,
    load_prompts,
)


# -------------------- per-prompt head-summed FRA -------------------- #
def head_summed_fra_for_prompt(
    ht,
    tokenizer,
    sae,
    prompt: str,
    layer: int,
    n_heads: int,
    top_k_features: int,
    max_length: int,
    hook_point: str,
) -> tuple[torch.Tensor | None, list[int]]:
    """Return (sparse [S, S, F, F] summed across heads on CPU, token_ids).
    None if every head yields an empty FRA tensor."""
    chat_text = build_chat_text(tokenizer, prompt)
    token_ids = tokenizer.encode(chat_text)
    if len(token_ids) > max_length:
        token_ids = token_ids[:max_length]

    head_sum: torch.Tensor | None = None
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
        fra = result["fra_tensor_sparse"]
        if fra._nnz() == 0:
            continue
        fra_cpu = fra.detach().to("cpu").to(torch.float32)
        head_sum = fra_cpu if head_sum is None else (head_sum + fra_cpu).coalesce()
        del fra, fra_cpu
        torch.cuda.empty_cache()
    return head_sum, token_ids


def collect_model_prompt_sums(
    hf_name: str,
    kind: str,
    sae,
    prompts: list[str],
    layer: int,
    top_k_features: int,
    max_length: int,
    hook_point: str,
    dtype: torch.dtype,
    device: str,
    cache_path: Path,
    base_model_hf: str,
) -> tuple[list[torch.Tensor | None], list[list[int]]]:
    """Load one checkpoint into HT, loop prompts × heads, return one sparse 4D
    per prompt (head-summed on CPU). Cached to disk as a dict {prompt_idx: sparse}."""
    if cache_path.exists():
        print(f"   [cache] loading {cache_path}", flush=True)
        payload = torch.load(cache_path, map_location="cpu")
        return payload["prompt_sums"], payload["token_ids"]

    t0 = time.time()
    if kind == "full_checkpoint":
        ht, tokenizer = load_base_or_sft_into_ht(hf_name, dtype, device)
    else:
        ht, tokenizer = load_lora_into_ht(
            lora_name=hf_name, base_name=base_model_hf, dtype=dtype, device=device
        )
    print(f"   model+tokenizer load: {time.time()-t0:.1f} s", flush=True)

    n_heads = ht.cfg.n_heads
    assert 0 <= layer < ht.cfg.n_layers

    prompt_sums: list[torch.Tensor | None] = []
    token_ids_per_prompt: list[list[int]] = []
    t_run = time.time()
    for prompt in tqdm(prompts, desc="  prompts", leave=False):
        head_sum, token_ids = head_summed_fra_for_prompt(
            ht=ht,
            tokenizer=tokenizer,
            sae=sae,
            prompt=prompt,
            layer=layer,
            n_heads=n_heads,
            top_k_features=top_k_features,
            max_length=max_length,
            hook_point=hook_point,
        )
        prompt_sums.append(head_sum)
        token_ids_per_prompt.append(token_ids)
    print(f"   FRA wall-clock: {(time.time()-t_run)/60:.1f} min", flush=True)

    torch.save(
        {"prompt_sums": prompt_sums, "token_ids": token_ids_per_prompt},
        cache_path,
    )
    print(f"   cached -> {cache_path}", flush=True)

    free_ht(ht)
    return prompt_sums, token_ids_per_prompt


# -------------------- delta accumulation + ranking -------------------- #
def sparse_subtract(
    a: torch.Tensor | None, b: torch.Tensor | None, shape: tuple[int, ...]
) -> torch.Tensor | None:
    """a - b for sparse COO tensors. Either may be None (= 0)."""
    if a is None and b is None:
        return None
    if a is None:
        return (-b).coalesce()
    if b is None:
        return a.coalesce()
    return (a - b).coalesce()


def update_pair_importance(
    importance: torch.Tensor | None, delta_4d: torch.Tensor | None, d_sae: int
) -> torch.Tensor | None:
    """Fold delta_4d's absolute values over (q_pos, k_pos) into a 2D [F, F]
    sparse importance tensor."""
    if delta_4d is None or delta_4d._nnz() == 0:
        return importance
    idx = delta_4d.indices()
    val = delta_4d.values().abs()
    pair_idx = idx[[2, 3]]
    contrib = torch.sparse_coo_tensor(
        pair_idx, val, (d_sae, d_sae)
    ).coalesce()
    if importance is None:
        return contrib
    return (importance + contrib).coalesce()


def topk_pair_rows(
    importance: torch.Tensor, k: int, divisor: float
) -> list[tuple[int, int, float]]:
    """Return top-K (fq, fk, importance_avg). divisor lets us report a
    head-averaged scale."""
    imp = importance.coalesce()
    vals = imp.values()
    if vals.numel() == 0:
        return []
    k = int(min(k, vals.numel()))
    top_v, top_i = torch.topk(vals, k)
    idx = imp.indices()[:, top_i]
    return [
        (int(idx[0, j]), int(idx[1, j]), float(top_v[j]) / max(divisor, 1.0))
        for j in range(k)
    ]


def extract_token_map(
    delta_4d: torch.Tensor | None, fq: int, fk: int, seq_len: int
) -> np.ndarray:
    """Dense signed [seq_len, seq_len] slice at (fq, fk)."""
    out = np.zeros((seq_len, seq_len), dtype=np.float32)
    if delta_4d is None or delta_4d._nnz() == 0:
        return out
    idx = delta_4d.indices()
    val = delta_4d.values()
    mask = (idx[2] == fq) & (idx[3] == fk)
    if mask.sum() == 0:
        return out
    q = idx[0, mask].cpu().numpy()
    k = idx[1, mask].cpu().numpy()
    v = val[mask].cpu().numpy()
    out[q, k] = v.astype(np.float32)
    return out


# -------------------- plotting -------------------- #
def _short_tokens(tokenizer, token_ids: list[int], max_chars: int = 10) -> list[str]:
    labels = []
    for tid in token_ids:
        s = tokenizer.decode([int(tid)])
        s = s.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
        if len(s) > max_chars:
            s = s[: max_chars - 1] + "…"
        labels.append(s)
    return labels


def plot_token_delta_grid(
    maps: list[np.ndarray],
    pair_info: list[tuple[int, int, float]],
    token_labels: list[str],
    title: str,
    save_path: Path,
    n_heads: int,
):
    """Grid of [seq, seq] heatmaps, one per top-K (fq, fk) pair. Each heatmap
    is divided by n_heads so it reads as a head-averaged delta."""
    n = len(maps)
    if n == 0:
        print(f"   nothing to plot for {save_path}")
        return
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 5.5, nrows * 5.0),
        squeeze=False,
    )
    for i, (m, (fq, fk, imp_avg)) in enumerate(zip(maps, pair_info)):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        m_avg = m / max(n_heads, 1)
        vmax = float(np.abs(m_avg).max()) if np.any(m_avg) else 1.0
        if vmax <= 0:
            vmax = 1.0
        im = ax.imshow(
            m_avg, cmap="bwr", vmin=-vmax, vmax=vmax, aspect="auto", interpolation="nearest"
        )
        ax.set_title(
            f"f_q={fq}  f_k={fk}\nimp(avg/head)={imp_avg:.3g}", fontsize=9
        )
        ax.set_xlabel("key pos")
        ax.set_ylabel("query pos")
        L = len(token_labels)
        step = max(1, L // 20)
        ax.set_xticks(range(0, L, step))
        ax.set_xticklabels(token_labels[::step], rotation=90, fontsize=6)
        ax.set_yticks(range(0, L, step))
        ax.set_yticklabels(token_labels[::step], fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # Hide unused axes.
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"   wrote {save_path}")


# -------------------- main -------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument("--sae-dir", required=True)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--sft-model", default=SFT_MODEL)
    parser.add_argument("--lora-model", default=LORA_MODEL)
    parser.add_argument("--n-prompts", type=int, default=DEFAULT_N_PROMPTS)
    parser.add_argument(
        "--top-k-features", type=int, default=DEFAULT_TOP_K_FEATURES,
        help="Features kept per position inside FRA.",
    )
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LEN)
    parser.add_argument(
        "--top-pairs", type=int, default=12,
        help="Number of most-relevant (feat_q, feat_k) pairs to plot.",
    )
    parser.add_argument(
        "--plot-prompt-idx", type=int, default=0,
        help="Which prompt's tokens to put on the heatmap axes.",
    )
    parser.add_argument("--hook-point", default="ln1.hook_normalized")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"]
    )
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "em_fra_scripts/outputs/step_d"),
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

    # -------------------- SAE -------------------- #
    print(f"\n>> Loading SAE from {args.sae_dir} ...", flush=True)
    sae, _sae_cfg = load_dictionary(args.sae_dir, device=args.device)
    sae.eval()
    sae.to(dtype)
    d_sae = sae.dict_size
    # FRA expects sae.W_dec with shape [d_sae, d_model]; BatchTopKSAE stores the
    # decoder as nn.Linear(dict_size, d_model) whose .weight is [d_model, d_sae].
    if not hasattr(sae, "W_dec"):
        sae.W_dec = sae.decoder.weight.T
    # ln1.hook_normalized comes out fp32 (TL upcasts LN); cast to SAE dtype.
    _sae_dtype = sae.b_dec.dtype
    _orig_encode = sae.encode

    def _encode_cast(x, *a, **kw):
        if x.dtype != _sae_dtype:
            x = x.to(_sae_dtype)
        return _orig_encode(x, *a, **kw)

    sae.encode = _encode_cast
    print(f"   d_sae={d_sae}", flush=True)

    # -------------------- Collect per-prompt head sums for each model -------------------- #
    model_specs = [
        ("base", args.base_model, "full_checkpoint"),
        ("sft", args.sft_model, "full_checkpoint"),
        ("lora", args.lora_model, "lora_adapter"),
    ]

    prompt_sums: dict[str, list[torch.Tensor | None]] = {}
    token_ids_per_model: dict[str, list[list[int]]] = {}
    for label, hf_name, kind in model_specs:
        print(f"\n==== Collecting head-summed FRA: {label} ({hf_name}) ====", flush=True)
        cache_path = out_dir / f"prompt_sums_{label}.pt"
        sums, tids = collect_model_prompt_sums(
            hf_name=hf_name,
            kind=kind,
            sae=sae,
            prompts=prompts,
            layer=args.layer,
            top_k_features=args.top_k_features,
            max_length=args.max_length,
            hook_point=args.hook_point,
            dtype=dtype,
            device=args.device,
            cache_path=cache_path,
            base_model_hf=args.base_model,
        )
        prompt_sums[label] = sums
        token_ids_per_model[label] = tids

    # Sanity: tokenisation should match across models (same tokenizer per family).
    base_tids = token_ids_per_model["base"]
    for label in ["sft", "lora"]:
        for p_idx, (a, b) in enumerate(zip(base_tids, token_ids_per_model[label])):
            if a != b:
                print(
                    f"   warning: token ids differ at prompt {p_idx} between base and {label}",
                    flush=True,
                )
                break

    # n_heads (needed to head-average the plotted values). We can recover it
    # from any non-None per-prompt 4D: shape[0]=S, [1]=S, [2]=F, [3]=F. n_heads
    # is not stored on the sparse, but step_c's Qwen2.5-14B has 40.
    # Read it back from the HF config to avoid magic numbers:
    from transformers import AutoConfig

    n_heads = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True).num_attention_heads

    # -------------------- For each comparison: rank + plot -------------------- #
    # Shape used by sparse_subtract when one side is None: assume the common
    # shape of any non-None tensor. Find one.
    def _shape_of(sums: list[torch.Tensor | None]) -> tuple[int, ...] | None:
        for s in sums:
            if s is not None:
                return tuple(s.shape)
        return None

    shape_base = _shape_of(prompt_sums["base"])
    if shape_base is None:
        raise RuntimeError("No non-empty per-prompt FRA for base.")
    seq_len = shape_base[0]
    print(f"\nseq_len={seq_len}  d_sae={d_sae}  n_heads={n_heads}", flush=True)

    for comparison in ["sft", "lora"]:
        print(f"\n==== Ranking {comparison} - base ====", flush=True)
        importance: torch.Tensor | None = None
        # Cache per-prompt delta for the plot prompt only.
        plot_delta: torch.Tensor | None = None

        for p_idx in range(len(prompts)):
            delta = sparse_subtract(
                prompt_sums[comparison][p_idx],
                prompt_sums["base"][p_idx],
                shape=shape_base,
            )
            importance = update_pair_importance(importance, delta, d_sae)
            if p_idx == args.plot_prompt_idx:
                plot_delta = delta

        if importance is None or importance._nnz() == 0:
            print(f"   all-zero delta for {comparison}; skipping.")
            continue

        top_rows = topk_pair_rows(importance, args.top_pairs, divisor=n_heads)
        pair_df = pd.DataFrame(
            top_rows, columns=["feat_q", "feat_k", f"importance_avg_per_head_{comparison}"]
        )
        pair_df.to_csv(out_dir / f"top{args.top_pairs}_delta_{comparison}_pairs.csv", index=False)
        print(
            f"   wrote {out_dir / f'top{args.top_pairs}_delta_{comparison}_pairs.csv'}",
            flush=True,
        )

        # Build token labels from the tokenizer (load the base tokenizer once).
        from transformers import AutoTokenizer

        tokenizer_for_labels = AutoTokenizer.from_pretrained(
            args.base_model, trust_remote_code=True
        )
        plot_token_ids = token_ids_per_model["base"][args.plot_prompt_idx]
        token_labels = _short_tokens(tokenizer_for_labels, plot_token_ids)

        (out_dir / "plot_prompt.txt").write_text(
            f"idx={args.plot_prompt_idx}\n{prompts[args.plot_prompt_idx]}\n"
            f"seq_len(tokens)={len(plot_token_ids)}\n"
        )

        maps = [
            extract_token_map(plot_delta, fq, fk, seq_len)
            for (fq, fk, _imp) in top_rows
        ]
        save_path = out_dir / f"top{args.top_pairs}_delta_{comparison}_token_heatmaps.png"
        plot_token_delta_grid(
            maps=maps,
            pair_info=top_rows,
            token_labels=token_labels,
            title=(
                f"Top-{args.top_pairs} head-averaged feature-pair deltas "
                f"({comparison} − base) @ L{args.layer}, prompt #{args.plot_prompt_idx}"
            ),
            save_path=save_path,
            n_heads=n_heads,
        )

    print(f"\nDone. Artefacts in: {out_dir}")


if __name__ == "__main__":
    main()
