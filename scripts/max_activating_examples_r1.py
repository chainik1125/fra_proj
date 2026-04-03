"""
Find max-activating examples for Llama 8B / R1-Distill crosscoder features.

Runs prompts through both Llama 3.1-8B (base) + DeepSeek-R1-Distill-Llama-8B
and records per-token activations.  By default loads math problems with full
R1 reasoning traces (<think>…</think>) from OpenR1-Math-220k.

Usage:
    # 200 reasoning traces, save JSON
    python scripts/max_activating_examples_r1.py 188

    # Paper features, more prompts
    python scripts/max_activating_examples_r1.py 188 744 1565 25929 31748 32252 --n-prompts 500

    # Specific layer (default 15)
    python scripts/max_activating_examples_r1.py 188 --layer 7

    # Custom dataset (falls back to generic text field detection)
    python scripts/max_activating_examples_r1.py 188 --dataset Elriggs/openwebtext-100k

Requires: ~36GB GPU RAM for both Llama 8B models in fp16.
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer
from fra.coders.crosscoder import GemmaCrosscoderFRA
from fra.analysis.max_act import (
    load_reasoning_prompts,
    load_generic_dataset,
    compute_max_acts,
    save_results,
)

BASE_MODEL = "meta-llama/Llama-3.1-8B"
IT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
IT_ARCH_NAME = "meta-llama/Llama-3.1-8B"
REPO_ID = "mitroitskii/Crosscoder-Llama-3.1-8B-vs-Llama-R1-Distill-8B"

LAYER_SUBFOLDERS = {
    7: "BatchTopK-Crosscoder/L7R",
    15: "BatchTopK-Crosscoder/L15R",
    23: "BatchTopK-Crosscoder/L23R",
}
DEFAULT_LAYER = 15
DEVICE = "cuda"


def main():
    parser = argparse.ArgumentParser(
        description="Find max-activating examples for R1-Distill crosscoder features")
    parser.add_argument("feature_ids", type=int, nargs="+",
                        help="Feature indices to investigate")
    parser.add_argument("--n-prompts", type=int, default=200,
                        help="Total number of prompts (default 200, balanced math/general)")
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER,
                        choices=sorted(LAYER_SUBFOLDERS.keys()),
                        help=f"Crosscoder layer (default {DEFAULT_LAYER})")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Override: use this HF dataset instead of MATH+UltraChat")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: results/max_acts_F{id}.json)")
    parser.add_argument("--top-k", type=int, default=30,
                        help="Number of top activating tokens to print")
    parser.add_argument("--no-template", action="store_true",
                        help="Don't apply chat template")
    args = parser.parse_args()

    feature_ids = args.feature_ids
    crosscoder_layer = args.layer
    subfolder = LAYER_SUBFOLDERS[crosscoder_layer]

    # Load prompts
    if args.dataset:
        prompts = load_generic_dataset(args.dataset, args.n_prompts)
    else:
        prompts = load_reasoning_prompts(args.n_prompts)

    # Load models
    print("Loading models...")
    torch.set_grad_enabled(False)
    base_model = HookedTransformer.from_pretrained(
        BASE_MODEL, device=DEVICE, dtype=torch.float16,
    )

    hf_model = AutoModelForCausalLM.from_pretrained(
        IT_MODEL, dtype=torch.float16,
    )
    hf_tokenizer = AutoTokenizer.from_pretrained(IT_MODEL)
    it_model = HookedTransformer.from_pretrained(
        IT_ARCH_NAME, device=DEVICE, dtype=torch.float16,
        hf_model=hf_model, tokenizer=hf_tokenizer,
    )
    del hf_model

    print("Loading crosscoder...")
    crosscoder = GemmaCrosscoderFRA.from_cc_weights(
        REPO_ID, subfolder, model_idx=1, device=DEVICE,
    )
    print(f"  dict_size={crosscoder.d_sae}, layer={crosscoder_layer}, "
          f"features={feature_ids}")

    # Run computation
    def _progress(i, n):
        if (i + 1) % 20 == 0 or i == n - 1:
            print(f"  [{i+1}/{n}]")

    print(f"\nRunning {len(prompts)} prompts...")
    per_feature = compute_max_acts(
        base_model, it_model, crosscoder, feature_ids, prompts,
        apply_template=not args.no_template,
        crosscoder_layer=crosscoder_layer,
        device=DEVICE,
        progress_callback=_progress,
    )

    # Save results
    dataset_name = args.dataset or "OpenR1-Math-220k"
    for fid in feature_ids:
        entries = per_feature[fid]

        if args.output and len(feature_ids) == 1:
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
        math_active = [e for e in active
                       if any(c.startswith("math/") for c in e["categories"])]
        general_active = [e for e in active
                         if "general" in e["categories"]]
        math_total = sum(1 for e in entries
                         if any(c.startswith("math/") for c in e["categories"]))
        general_total = sum(1 for e in entries if "general" in e["categories"])

        print(f"\n{'='*80}")
        print(f"Feature {fid}")
        print(f"{'='*80}")
        print(f"  Active: {len(active)}/{len(entries)} prompts "
              f"({100*len(active)/len(entries):.0f}%)")
        if math_total > 0:
            print(f"    Math:    {len(math_active)}/{math_total} "
                  f"({100*len(math_active)/math_total:.0f}%)")
        if general_total > 0:
            print(f"    General: {len(general_active)}/{general_total} "
                  f"({100*len(general_active)/general_total:.0f}%)")

        print(f"\n  Top {args.top_k} activating prompts:")
        for rank, entry in enumerate(entries[:args.top_k], 1):
            cats = ",".join(entry["categories"][:2]) if entry["categories"] else "?"
            short = (entry["prompt"][:65] + "..."
                     if len(entry["prompt"]) > 65 else entry["prompt"])
            if entry["max_act"] > 0:
                top_tok = entry["top_tokens"][0]["token"] if entry["top_tokens"] else ""
                print(f"  {rank:3d}. [{entry['max_act']:7.3f}] {cats:20s} "
                      f"top_tok={top_tok!r:15s} {short!r}")
            else:
                print(f"  {rank:3d}. [  0.000] {cats:20s} {short!r}")


if __name__ == "__main__":
    main()
