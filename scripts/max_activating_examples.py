"""
Find max-activating examples for crosscoder features.

Runs prompts through both Gemma 2B models + crosscoder and records per-token
activations.  By default loads unsafe prompts from BeaverTails (with harm
category labels) and normal prompts from UltraChat for a balanced comparison.

Usage:
    # 200 prompts (100 unsafe BeaverTails + 100 safe UltraChat), save JSON
    python scripts/max_activating_examples.py 53124

    # More prompts, multiple features
    python scripts/max_activating_examples.py 53124 24613 70149 --n-prompts 500

    # Custom dataset (falls back to generic text field detection)
    python scripts/max_activating_examples.py 53124 --dataset Elriggs/openwebtext-100k

Requires: ~12GB GPU RAM for both Gemma 2B models in fp16.
"""

import argparse

import torch
from transformer_lens import HookedTransformer
from fra.core.coder import FRACoder
from fra.analysis.max_act import (
    load_prompts,
    load_generic_dataset,
    compute_max_acts,
    save_results,
)

REPO_ID = "science-of-finetuning/gemma-2-2b-L13-k100-lr1e-04-local-shuffling-CCLoss"
CROSSCODER_LAYER = 13
DEVICE = "cuda"


def main():
    parser = argparse.ArgumentParser(
        description="Find max-activating examples for crosscoder features")
    parser.add_argument("feature_ids", type=int, nargs="+",
                        help="Feature indices to investigate")
    parser.add_argument("--n-prompts", type=int, default=200,
                        help="Total number of prompts (default 200, balanced safe/unsafe)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Override: use this HF dataset instead of BeaverTails+UltraChat")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: results/max_acts_F{id}.json)")
    parser.add_argument("--top-k", type=int, default=30,
                        help="Number of top activating tokens to print")
    parser.add_argument("--no-template", action="store_true",
                        help="Don't apply chat template")
    args = parser.parse_args()

    feature_ids = args.feature_ids

    # Load prompts
    if args.dataset:
        prompts = load_generic_dataset(args.dataset, args.n_prompts)
    else:
        prompts = load_prompts(args.n_prompts)

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
    crosscoder = FRACoder.from_hf_crosscoder(
        REPO_ID, model_idx=1, device=DEVICE, dtype=torch.float16,
    )
    print(f"  dict_size={crosscoder.d_sae}, features={feature_ids}")

    # Run computation
    def _progress(i, n):
        if (i + 1) % 20 == 0 or i == n - 1:
            print(f"  [{i+1}/{n}]")

    print(f"\nRunning {len(prompts)} prompts...")
    per_feature = compute_max_acts(
        base_model, it_model, crosscoder, feature_ids, prompts,
        apply_template=not args.no_template,
        crosscoder_layer=CROSSCODER_LAYER,
        device=DEVICE,
        progress_callback=_progress,
    )

    # Save results
    dataset_name = args.dataset or "BeaverTails+UltraChat"
    for fid in feature_ids:
        entries = per_feature[fid]

        if args.output and len(feature_ids) == 1:
            # Custom output path for single feature
            import json
            from pathlib import Path
            Path(args.output).parent.mkdir(exist_ok=True)
            with open(args.output, "w") as f:
                json.dump({
                    "feature_id": fid,
                    "n_prompts": len(entries),
                    "dataset": dataset_name,
                    "prompts": entries,
                }, f, indent=2)
            output_path = args.output
        else:
            output_path = save_results(fid, entries, len(entries), dataset_name)

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
