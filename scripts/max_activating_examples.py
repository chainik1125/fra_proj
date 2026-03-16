"""
Find max-activating examples for crosscoder features.

Runs prompts through both Gemma 2B models + crosscoder and records per-token
activations.  Defaults to PKU-Alignment/BeaverTails which has labeled
safe/unsafe prompts across 14 harm categories.

Usage:
    # 200 prompts from BeaverTails (100 safe + 100 unsafe), save JSON
    python scripts/max_activating_examples.py 53124

    # More prompts, multiple features
    python scripts/max_activating_examples.py 53124 24613 70149 --n-prompts 500

    # Custom dataset (falls back to generic text field detection)
    python scripts/max_activating_examples.py 53124 --dataset Elriggs/openwebtext-100k

Requires: ~12GB GPU RAM for both Gemma 2B models in fp16.
"""

import argparse
import json
from pathlib import Path

import torch
from transformer_lens import HookedTransformer
from fra.crosscoder_wrapper import GemmaCrosscoderFRA

REPO_ID = "science-of-finetuning/gemma-2-2b-L13-k100-lr1e-04-local-shuffling-CCLoss"
CROSSCODER_LAYER = 13
DEVICE = "cuda"


def get_activations(base_model, it_model, crosscoder, text, feature_ids,
                    apply_template=True):
    """Run both models and encode through crosscoder.

    Returns dict mapping feature_id -> list of (position, token_str, activation).
    """
    if apply_template:
        tokens = it_model.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            add_generation_prompt=True,
        )
    else:
        tokens = it_model.tokenizer.encode(text)
    tokens_tensor = torch.tensor(tokens).unsqueeze(0).to(DEVICE)

    hook_name = f"blocks.{CROSSCODER_LAYER}.hook_resid_post"
    _, base_cache = base_model.run_with_cache(tokens_tensor, names_filter=[hook_name])
    _, it_cache = it_model.run_with_cache(tokens_tensor, names_filter=[hook_name])

    x_stacked = torch.stack([
        base_cache[hook_name].squeeze(0),
        it_cache[hook_name].squeeze(0),
    ], dim=1)

    feat_acts = crosscoder.encode(x_stacked)  # [seq, d_sae]
    token_strs = [it_model.tokenizer.decode([t]) for t in tokens]

    results = {}
    for fid in feature_ids:
        acts = feat_acts[:, fid]
        token_acts = []
        for pos in range(len(token_strs)):
            val = acts[pos].item()
            if val > 0:
                token_acts.append((pos, token_strs[pos], val))
        results[fid] = token_acts

    return results, token_strs


def load_beavertails(n_prompts, split="330k_test"):
    """Load balanced safe/unsafe prompts from BeaverTails."""
    from datasets import load_dataset

    n_each = n_prompts // 2
    print(f"Loading {n_each} safe + {n_each} unsafe prompts from BeaverTails ({split})...")
    ds = load_dataset("PKU-Alignment/BeaverTails", split=split, streaming=True)
    ds = ds.shuffle(seed=42, buffer_size=10000)

    safe, unsafe = [], []
    for example in ds:
        prompt = example["prompt"].strip()
        if len(prompt) < 10 or len(prompt) > 500:
            continue
        is_safe = example["is_safe"]
        categories = [k for k, v in example["category"].items() if v]
        entry = {"text": prompt, "is_safe": is_safe, "categories": categories}

        if is_safe and len(safe) < n_each:
            safe.append(entry)
        elif not is_safe and len(unsafe) < n_each:
            unsafe.append(entry)
        if len(safe) >= n_each and len(unsafe) >= n_each:
            break

    prompts = safe + unsafe
    print(f"  Loaded {len(safe)} safe + {len(unsafe)} unsafe = {len(prompts)} prompts")
    return prompts


def load_generic_dataset(dataset_name, n_prompts, text_field="text"):
    """Load prompts from a generic HuggingFace dataset."""
    from datasets import load_dataset

    print(f"Loading {n_prompts} prompts from {dataset_name}...")
    ds = load_dataset(dataset_name, split="train", streaming=True)
    ds = ds.shuffle(seed=42, buffer_size=10000)

    prompts = []
    for example in ds:
        for field in [text_field, "content", "text", "prompt", "instruction", "question"]:
            if field in example and example[field]:
                txt = example[field].strip()
                if 10 < len(txt) < 500:
                    prompts.append({"text": txt, "is_safe": None, "categories": []})
                    break
        if len(prompts) >= n_prompts:
            break

    print(f"  Loaded {len(prompts)} prompts")
    return prompts


def main():
    parser = argparse.ArgumentParser(
        description="Find max-activating examples for crosscoder features")
    parser.add_argument("feature_ids", type=int, nargs="+",
                        help="Feature indices to investigate")
    parser.add_argument("--n-prompts", type=int, default=200,
                        help="Total number of prompts (default 200, balanced safe/unsafe for BeaverTails)")
    parser.add_argument("--dataset", type=str, default="PKU-Alignment/BeaverTails",
                        help="HuggingFace dataset (default: PKU-Alignment/BeaverTails)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: results/max_acts_F{id}.json)")
    parser.add_argument("--top-k", type=int, default=30,
                        help="Number of top activating tokens to print")
    parser.add_argument("--no-template", action="store_true",
                        help="Don't apply chat template")
    args = parser.parse_args()

    feature_ids = args.feature_ids

    # Load prompts
    if args.dataset == "PKU-Alignment/BeaverTails":
        prompts = load_beavertails(args.n_prompts)
    else:
        prompts = load_generic_dataset(args.dataset, args.n_prompts)

    # Load models
    print("Loading models...")
    torch.set_grad_enabled(False)
    base_model = HookedTransformer.from_pretrained(
        "google/gemma-2-2b", device=DEVICE, dtype=torch.float16,
    )
    it_model = HookedTransformer.from_pretrained(
        "google/gemma-2-2b-it", device=DEVICE, dtype=torch.float16,
    )

    print("Loading crosscoder...")
    crosscoder = GemmaCrosscoderFRA.from_pretrained(
        REPO_ID, model_idx=1, device=DEVICE, dtype=torch.float16,
    )
    print(f"  dict_size={crosscoder.d_sae}, features={feature_ids}")

    # Per-feature collectors
    # Each entry: {prompt, is_safe, categories, max_act, mean_act, n_active, top_tokens}
    per_feature = {fid: [] for fid in feature_ids}

    print(f"\nRunning {len(prompts)} prompts...")
    for i, prompt_entry in enumerate(prompts):
        text = prompt_entry["text"]
        results, token_strs = get_activations(
            base_model, it_model, crosscoder, text, feature_ids,
            apply_template=not args.no_template,
        )

        for fid in feature_ids:
            token_acts = results[fid]
            vals = [v for _, _, v in token_acts]
            max_act = max(vals) if vals else 0.0
            mean_act = sum(vals) / len(vals) if vals else 0.0

            per_feature[fid].append({
                "prompt": text,
                "is_safe": prompt_entry["is_safe"],
                "categories": prompt_entry["categories"],
                "max_act": max_act,
                "mean_act": mean_act,
                "n_active_tokens": len(token_acts),
                "n_tokens": len(token_strs),
                "top_tokens": [
                    {"pos": p, "token": t, "act": v}
                    for p, t, v in sorted(token_acts, key=lambda x: x[2], reverse=True)[:10]
                ],
            })

        if (i + 1) % 20 == 0 or i == len(prompts) - 1:
            print(f"  [{i+1}/{len(prompts)}]")

    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    for fid in feature_ids:
        entries = per_feature[fid]
        entries.sort(key=lambda x: x["max_act"], reverse=True)

        output_path = args.output or str(results_dir / f"max_acts_F{fid}.json")
        # If multiple features and no explicit output, use per-feature paths
        if len(feature_ids) > 1 and not args.output:
            output_path = str(results_dir / f"max_acts_F{fid}.json")

        with open(output_path, "w") as f:
            json.dump({
                "feature_id": fid,
                "n_prompts": len(entries),
                "dataset": args.dataset,
                "prompts": entries,
            }, f, indent=2)
        print(f"\nSaved {output_path}")

        # Print summary
        active = [e for e in entries if e["max_act"] > 0]
        safe_active = [e for e in active if e["is_safe"] is True]
        unsafe_active = [e for e in active if e["is_safe"] is False]
        safe_total = sum(1 for e in entries if e["is_safe"] is True)
        unsafe_total = sum(1 for e in entries if e["is_safe"] is False)

        print(f"\n{'='*80}")
        print(f"Feature {fid}")
        print(f"{'='*80}")
        print(f"  Active: {len(active)}/{len(entries)} prompts "
              f"({100*len(active)/len(entries):.0f}%)")
        if safe_total > 0:
            print(f"    Safe:   {len(safe_active)}/{safe_total} "
                  f"({100*len(safe_active)/safe_total:.0f}%)")
        if unsafe_total > 0:
            print(f"    Unsafe: {len(unsafe_active)}/{unsafe_total} "
                  f"({100*len(unsafe_active)/unsafe_total:.0f}%)")

        print(f"\n  Top {args.top_k} activating prompts:")
        for rank, entry in enumerate(entries[:args.top_k], 1):
            safe_str = "SAFE" if entry["is_safe"] else "UNSAFE"
            if entry["is_safe"] is None:
                safe_str = "?"
            short = entry["prompt"][:65] + "..." if len(entry["prompt"]) > 65 else entry["prompt"]
            cats = ",".join(entry["categories"][:2]) if entry["categories"] else ""
            if entry["max_act"] > 0:
                top_tok = entry["top_tokens"][0]["token"] if entry["top_tokens"] else ""
                print(f"  {rank:3d}. [{entry['max_act']:7.3f}] {safe_str:6s} "
                      f"top_tok={top_tok!r:15s} {short!r}")
                if cats:
                    print(f"       categories: {cats}")
            else:
                print(f"  {rank:3d}. [  0.000] {safe_str:6s} {short!r}")


if __name__ == "__main__":
    main()
