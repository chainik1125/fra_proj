#!/usr/bin/env python
"""
FRA Off-Diagonal Ablation Study — CLI entry point.

Run:
  python scripts/run_ablation.py                         # GPT-2, local ln1, L2
  python scripts/run_ablation.py --heads 0 1 5 --k 10 50 100
  python scripts/run_ablation.py --sae hub --layer 5     # hook_z SAE at L5
  python scripts/run_ablation.py --model gemma --layer 13 # Gemma FRA L13 (SAE L12)
"""

import argparse
import json

import torch
from transformer_lens import HookedTransformer

from fra.analysis.ablation import (
    ABLATION_TEXTS,
    aggregate_results,
    print_results,
    run_single_sample,
    screen_heads,
)
from fra.coders import load_sae


def main():
    parser = argparse.ArgumentParser(description="FRA Off-Diagonal Ablation Study")
    parser.add_argument("--model", choices=["gpt2", "gemma"], default="gpt2")
    parser.add_argument("--sae", choices=["hub", "local", "gemma"], default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--heads", type=int, nargs="+", default=None,
                        help="Heads to test (default: auto-select top 3 by contribution)")
    parser.add_argument("--k", type=int, nargs="+", default=[10, 50, 100, 500],
                        help="Number of feature pairs to ablate")
    parser.add_argument("--top-k-features", type=int, default=20,
                        help="Top-K SAE features per position in FRA")
    parser.add_argument("--n-texts", type=int, default=None,
                        help="Number of texts to use (default: all)")
    parser.add_argument("--rank-mode", choices=["sum", "avg", "max"], default="sum",
                        help="How to rank feature pairs for ablation")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save", type=str, default=None,
                        help="Save JSON results to this path")
    parser.add_argument("--chunk-size", type=int, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    is_gemma = args.model == "gemma"

    # Model/SAE defaults
    if is_gemma:
        sae_type = args.sae or "gemma"
        layer = args.layer if args.layer is not None else 13
        hook_point = "hook_resid_pre"
        chunk_size = args.chunk_size or 1
    else:
        sae_type = args.sae or "local"
        layer = args.layer if args.layer is not None else 2
        chunk_size = args.chunk_size or 16
        if sae_type == "hub":
            hook_point = "attn.hook_z"
        else:
            hook_point = "ln1.hook_normalized"

    k_values = args.k
    texts = ABLATION_TEXTS[:args.n_texts] if args.n_texts else ABLATION_TEXTS

    print("=" * 70)
    print("  FRA Off-Diagonal Ablation Study")
    print("=" * 70)
    print(f"  Model      : {'gemma-2-2b' if is_gemma else 'gpt2'}")
    print(f"  SAE        : {sae_type} ({hook_point})")
    print(f"  Layer      : {layer}")
    print(f"  K values   : {k_values}")
    print(f"  Rank mode  : {args.rank_mode}")
    print(f"  FRA top-k  : {args.top_k_features}")
    print(f"  Texts      : {len(texts)}")
    print(f"  Device     : {device}")

    # Load SAE
    print("\nLoading SAE...", end=" ", flush=True)
    sae = load_sae(sae_type, layer, device)
    print(f"done. (d_sae={sae.d_sae})")

    # Load model
    if is_gemma:
        model_kw = {
            "fold_ln": False, "center_unembed": False,
            "center_writing_weights": False, "fold_value_biases": False,
            "refactor_factored_attn_matrices": False,
        }
    elif sae_type == "local":
        model_kw = {"fold_ln": False, "center_unembed": True, "center_writing_weights": True}
    else:
        model_kw = {}

    print("Loading model...", end=" ", flush=True)
    model = HookedTransformer.from_pretrained(
        "gemma-2-2b" if is_gemma else "gpt2", device=device, **model_kw
    )
    print("done.")

    # Head selection
    if args.heads is not None:
        heads = args.heads
    else:
        print("\nScreening heads (zero-ablation)...", flush=True)
        head_contribs = screen_heads(model, texts[:3], layer, hook_point)
        heads = [h for h, c in head_contribs[:3]]
        print("  Head contributions (top 5):")
        for h, c in head_contribs[:5]:
            print(f"    H{h}: {c:+.4f} ({'helps' if c > 0 else 'hurts/neutral'})")
        print(f"  Selected heads: {heads}")

    print(f"\n  Heads      : {heads}")
    print("=" * 70)

    # Run study
    all_per_head_results = {}

    for head in heads:
        print(f"\n{'='*70}")
        print(f"  HEAD {head}")
        print(f"{'='*70}")

        all_results = []
        for i, text in enumerate(texts):
            short = text[:55] + "..." if len(text) > 55 else text
            print(f"\n  Text {i+1}/{len(texts)}: \"{short}\"")

            result = run_single_sample(
                model, sae, text, layer, head, hook_point,
                k_values=k_values,
                top_k_features=args.top_k_features,
                chunk_size=chunk_size,
                rank_mode=args.rank_mode,
            )
            if result is None:
                print("    Skipped (too short)")
                continue

            meta = result["_meta"]
            print(f"    seq={meta['seq_len']}, nnz={meta['nnz']:,}, "
                  f"offdiag_pairs={meta['n_offdiag_pairs']}, "
                  f"ondiag_pairs={meta['n_ondiag_pairs']}, "
                  f"head_contrib={meta['head_contribution']:+.4f}")
            print(f"    fra_full loss={result['fra_full']['loss']:.4f}, "
                  f"zero loss={result['zero']['loss']:.4f}")

            all_results.append(result)

        if all_results:
            agg = aggregate_results(all_results, k_values)
            print(f"\n  Aggregated results for L{layer} H{head} "
                  f"({len(all_results)} texts):")
            print_results(agg, k_values)
            all_per_head_results[head] = agg

    # Overall summary
    print(f"\n\n{'='*70}")
    print(f"  OVERALL SUMMARY")
    print(f"{'='*70}")

    for head, agg in all_per_head_results.items():
        unp = agg.get("unpatched", {}).get("loss", float("nan"))
        fra = agg.get("fra_full", {}).get("loss", float("nan"))
        zero = agg.get("zero", {}).get("loss", float("nan"))
        hc = zero - unp

        print(f"\n  L{layer} H{head}:  unpatched={unp:.4f}  fra_full={fra:.4f}  "
              f"zero={zero:.4f}  head_contrib={hc:+.4f}")

        if hc < 0.01:
            print(f"    Head contribution too small for meaningful ablation analysis.")
            continue

        for k in k_values:
            off = agg.get(f"offdiag_{k}", {})
            rnd = agg.get(f"random_{k}", {})
            on = agg.get(f"ondiag_{k}", {})

            off_dloss = off.get("loss", unp) - unp
            rnd_dloss = rnd.get("loss", unp) - unp
            on_dloss = on.get("loss", unp) - unp

            off_rec = (zero - off.get("loss", zero)) / (hc + 1e-10)
            rnd_rec = (zero - rnd.get("loss", zero)) / (hc + 1e-10)
            on_rec = (zero - on.get("loss", zero)) / (hc + 1e-10)

            print(f"    k={k:>4}:  offdiag dL={off_dloss:+.4f} rec={off_rec:.3f}  |  "
                  f"random dL={rnd_dloss:+.4f} rec={rnd_rec:.3f}  |  "
                  f"ondiag dL={on_dloss:+.4f} rec={on_rec:.3f}")

    # Save JSON
    if args.save:
        save_data = {
            "config": {
                "model": args.model, "sae": sae_type, "layer": layer,
                "heads": heads, "k_values": k_values,
                "rank_mode": args.rank_mode, "n_texts": len(texts),
            },
            "per_head": {
                str(h): {cond: vals for cond, vals in agg.items()}
                for h, agg in all_per_head_results.items()
            },
        }
        with open(args.save, "w") as f:
            json.dump(save_data, f, indent=2, default=str)
        print(f"\nResults saved to {args.save}")

    print("\nDone.")


if __name__ == "__main__":
    main()
