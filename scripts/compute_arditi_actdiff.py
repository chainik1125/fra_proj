"""Compute ‖Δa‖ at L15 resid_post between Qwen-7B + bad-medical and base.

Replicates safety-research/open-source-em-features `compute_layer_difference_vectors`
with their defaults: prompt-last pooling, exclude BOS, left padding, bf16.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


THEIR_DATASET_URL = (
    "https://raw.githubusercontent.com/safety-research/open-source-em-features/"
    "main/data/medical_advice_prompt_only.jsonl"
)


def iter_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


@torch.no_grad()
def extract_layer_act_last(model, tokenizer, prompts, layer_idx,
                          batch_size=16, max_ctx_len=512, device="cuda"):
    """Prompt-last residual at blocks.{layer_idx}.hook_resid_post, BOS-excluded.

    Returns Tensor[N, d_model] of per-prompt last-token activations.
    Hooks into the underlying HF transformer block's output residual stream.
    """
    blocks = model.model.layers
    cache = {}

    def hook(module, inputs, output):
        # output is a tuple; first element is hidden_states post-block
        cache["h"] = output[0] if isinstance(output, tuple) else output

    handle = blocks[layer_idx].register_forward_hook(hook)
    outs = []
    try:
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True,
                          truncation=True, max_length=max_ctx_len).to(device)
            _ = model(**enc)
            h = cache["h"]  # (B, T, D)
            # Last non-pad position per row (right-most index where attention=1)
            attn = enc["attention_mask"]  # (B, T)
            last_idx = attn.sum(dim=1) - 1  # (B,)
            rows = h[torch.arange(h.shape[0], device=device), last_idx]  # (B, D)
            outs.append(rows.float().cpu())
            del h, enc
    finally:
        handle.remove()
    return torch.cat(outs, dim=0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layer", type=int, default=15)
    p.add_argument("--positive-model", default="andyrdt/Qwen2.5-7B-Instruct_bad-medical")
    p.add_argument("--negative-model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--dataset-path", default="/workspace/medical_advice_prompt_only.jsonl",
                  help="Path to JSONL (will download from their repo if missing)")
    p.add_argument("--n-prompts", type=int, default=512)
    p.add_argument("--max-ctx-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="/workspace/actdiff_L15.json")
    args = p.parse_args()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"[data] downloading {THEIR_DATASET_URL}")
        urllib.request.urlretrieve(THEIR_DATASET_URL, dataset_path)
    rows = list(iter_jsonl(dataset_path))[: args.n_prompts]
    print(f"[data] {len(rows)} rows from {dataset_path.name}")
    prompts = [r.get("prompt") or r.get("question") or list(r.values())[0]
              for r in rows]

    # Apply chat template (matches their `load_model_and_tokenizer` defaults)
    tokenizer = AutoTokenizer.from_pretrained(args.negative_model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    formatted = []
    for p_text in prompts:
        msgs = [{"role": "user", "content": p_text}]
        try:
            txt = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            txt = p_text
        formatted.append(txt)

    # --- positive (EM) model ---
    t0 = time.time()
    print(f"[pos] loading {args.positive_model}")
    base = AutoModelForCausalLM.from_pretrained(
        args.negative_model, torch_dtype=torch.bfloat16, device_map=args.device
    )
    pos = PeftModel.from_pretrained(base, args.positive_model)
    pos = pos.merge_and_unload()
    print(f"[pos] loaded in {time.time()-t0:.1f}s; extracting acts …")
    pos_acts = extract_layer_act_last(
        pos, tokenizer, formatted, args.layer,
        batch_size=args.batch_size, max_ctx_len=args.max_ctx_len, device=args.device,
    )
    print(f"[pos] acts shape={tuple(pos_acts.shape)}  mean‖h‖={pos_acts.norm(dim=-1).mean():.4f}")
    del pos
    torch.cuda.empty_cache()

    # --- negative (base) model ---
    t0 = time.time()
    print(f"[neg] loading {args.negative_model}")
    neg = AutoModelForCausalLM.from_pretrained(
        args.negative_model, torch_dtype=torch.bfloat16, device_map=args.device
    )
    print(f"[neg] loaded in {time.time()-t0:.1f}s; extracting acts …")
    neg_acts = extract_layer_act_last(
        neg, tokenizer, formatted, args.layer,
        batch_size=args.batch_size, max_ctx_len=args.max_ctx_len, device=args.device,
    )
    print(f"[neg] acts shape={tuple(neg_acts.shape)}  mean‖h‖={neg_acts.norm(dim=-1).mean():.4f}")
    del neg
    torch.cuda.empty_cache()

    pos_mean = pos_acts.mean(dim=0)
    neg_mean = neg_acts.mean(dim=0)
    diff = pos_mean - neg_mean
    norm = diff.norm().item()

    result = {
        "layer": args.layer,
        "positive_model": args.positive_model,
        "negative_model": args.negative_model,
        "n_prompts": len(prompts),
        "max_ctx_len": args.max_ctx_len,
        "pos_mean_norm": float(pos_mean.norm()),
        "neg_mean_norm": float(neg_mean.norm()),
        "diff_norm_l2": norm,
    }
    print("\n=== result ===")
    print(json.dumps(result, indent=2))
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
