#!/usr/bin/env python
"""Measure FRA reconstruction error for one layer/head."""

from __future__ import annotations

import argparse

import torch
from transformer_lens import HookedTransformer

from fra.induction_head import SAELensAttentionSAE
from fra.reconstruction import (
    measure_attention_reconstruction,
    resolve_torch_device,
    summarize_reconstruction_results,
)


BENCHMARK_PROMPTS = [
    "The cat sat on the mat. The cat was happy.",
    "Alice went to the store. She bought milk.",
    "The student who studied hard passed. The students who studied hard passed.",
    "Bob likes to play chess. Bob is very good at chess.",
    "The weather today is sunny. The weather yesterday was rainy.",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, default=5)
    parser.add_argument("--head", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--model-name", default="gpt2-small")
    parser.add_argument("--sae-release", default="gpt2-small-hook-z-kk")
    parser.add_argument("--prompt", action="append", default=None)
    parser.add_argument("--benchmark-prompts", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    torch.set_grad_enabled(False)

    device = resolve_torch_device(args.device)
    print("=" * 60)
    print("Verifying FRA Reconstruction")
    print("=" * 60)
    print(f"\nDevice: {device}")
    print(f"Layer/head: L{args.layer}H{args.head}")

    prompts = args.prompt or []
    if args.benchmark_prompts:
        prompts.extend(BENCHMARK_PROMPTS)
    if not prompts:
        prompts = ["The cat sat on the mat"]

    if len(prompts) == 1:
        print(f"Prompt: {prompts[0]!r}")
    else:
        print(f"Prompts: {len(prompts)}")

    model = HookedTransformer.from_pretrained(args.model_name, device=device)
    sae = SAELensAttentionSAE(
        args.sae_release,
        f"blocks.{args.layer}.hook_z",
        device=device,
    )
    results = [
        measure_attention_reconstruction(
            model,
            sae,
            prompt,
            layer=args.layer,
            head=args.head,
            max_length=args.max_length,
            top_k=args.top_k,
        )
        for prompt in prompts
    ]

    for index, result in enumerate(results, start=1):
        if len(results) > 1:
            print(f"\nPrompt {index}: {result.prompt!r}")
        print(f"\nSequence length: {result.seq_len}")
        print(f"Non-zero FRA entries: {result.nnz}")
        print(f"Score scaling: {result.score_scale:.6f}")
        print("\nPattern reconstruction:")
        print(f"  mean absolute error : {result.pattern_mae:.6f}")
        print(f"  root mean square    : {result.pattern_rmse:.6f}")
        print(f"  error percentage    : {result.pattern_error_pct:.2f}%")

        if result.score_mae is not None and result.score_rmse is not None:
            print("\nScore reconstruction (after 1/sqrt(d_head) scaling):")
            print(f"  mean absolute error : {result.score_mae:.6f}")
            print(f"  root mean square    : {result.score_rmse:.6f}")

    if len(results) > 1:
        summary = summarize_reconstruction_results(results)
        print("\nAggregate pattern reconstruction:")
        print(f"  prompts             : {summary.num_prompts}")
        print(f"  mean absolute error : {summary.mean_pattern_mae:.6f}")
        print(f"  error percentage    : {summary.mean_pattern_error_pct:.2f}%")
        print(f"  min error pct       : {summary.min_pattern_error_pct:.2f}%")
        print(f"  max error pct       : {summary.max_pattern_error_pct:.2f}%")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
