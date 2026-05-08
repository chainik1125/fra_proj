"""Per-seed alpha sweep of the single resid_mid feature suppressor.

For each seed, take the feature that the recreate_layer0 pipeline picked as
the best resid_mid suppressor (e.g. f=171 for seed 0), apply ablation at
`blocks.0.hook_resid_mid` across α ∈ {0.5, 1, 2, 3}, and record (ASR_16, ΔCE)
on the held-out test split — matching the alphas pareto_3x3.py uses.

Output: a JSON in pareto_3x3-compatible per_alpha format, suitable for
overlaying on plot_pareto_overlay.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

THIS = Path(__file__).resolve().parent
EXP = THIS.parent.parent / "experiments" / "tinystories_sleeper"
sys.path.insert(0, str(EXP))


def pick_device(explicit):
    if explicit:
        return explicit
    return "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def asr_on_prompts(model, sae_mid, hook, feature_idx, alpha,
                   tokens, prompt_mask, marker_pos, max_new_tokens):
    from sleeper_utils import (
        asr_16, compute_sae_delta, greedy_generate_with_hooks,
        make_delta_hook_single_layer,
    )
    uniq = marker_pos.unique().tolist()
    hits, total = 0, 0
    for m_pos in uniq:
        rows = (marker_pos == m_pos).nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            continue
        P = int(m_pos) + 1
        trunc = tokens[rows, :P]
        trunc_mask = prompt_mask[rows, :P]
        delta = compute_sae_delta(model, sae_mid, hook, feature_idx, trunc, trunc_mask)
        hooks = make_delta_hook_single_layer(delta, alpha, hook)
        gen = greedy_generate_with_hooks(model, trunc, hooks, max_new_tokens)
        hits += int(round(asr_16(gen, model.tokenizer) * gen.shape[0]))
        total += gen.shape[0]
    return hits / max(1, total)


@torch.no_grad()
def clean_ce(model, sae_mid, hook, feature_idx, alpha,
             clean_tokens, clean_mask, clean_marker, baseline_ce):
    from sleeper_utils import (
        clean_continuation_ce, compute_sae_delta, make_delta_hook_single_layer,
    )
    delta = compute_sae_delta(model, sae_mid, hook, feature_idx, clean_tokens, clean_mask)
    hooks = make_delta_hook_single_layer(delta, alpha, hook)
    ce = clean_continuation_ce(model, clean_tokens, clean_marker, fwd_hooks=hooks).mean().item()
    return ce, ce - baseline_ce


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--sae_mid", type=Path, required=True,
                   help="path to recreate_layer0_seed{N}/results/crosscoder_sae_layer1.pt")
    p.add_argument("--feature_idx", type=int, required=True,
                   help="best resid_mid feature picked by selectivity sweep (e.g. 171 for seed 0)")
    p.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0, 2.0, 3.0])
    p.add_argument("--gen_tokens", type=int, default=16)
    p.add_argument("--device", default=None)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    from run_ablation_sweep import load_crosscoder
    from sleeper_utils import (
        clean_continuation_ce, load_paired_dataset, load_sleeper_model,
        prompt_mask_from_markers,
    )

    device = pick_device(args.device)
    cache = torch.load(args.cache, weights_only=False)
    meta = cache["meta"]

    model = load_sleeper_model(device=device)
    sae_mid, _ = load_crosscoder(args.sae_mid, device=device)
    hook = "blocks.0.hook_resid_mid"

    splits = load_paired_dataset(
        tokenizer=model.tokenizer,
        n_train=meta["n_train"], n_val=meta["n_val"], n_test=meta["n_test"],
        seq_len=meta["seq_len"], seed=meta["seed"],
    )
    pt = splits[meta["split"]]
    prompt_mask = prompt_mask_from_markers(meta["seq_len"], pt.story_marker_pos)

    dep_idx = torch.where(pt.is_deployment)[0]
    cln_idx = torch.where(~pt.is_deployment)[0]
    dep_tokens, dep_marker = pt.tokens[dep_idx], pt.story_marker_pos[dep_idx]
    dep_mask = prompt_mask[dep_idx]
    cln_tokens = pt.tokens[cln_idx].to(device)
    cln_marker = pt.story_marker_pos[cln_idx].to(device)
    cln_mask = prompt_mask[cln_idx].to(device)

    print(f"[mid-sweep] feature={args.feature_idx} hook={hook}")
    base_ce = clean_continuation_ce(model, cln_tokens, cln_marker).mean().item()
    print(f"[mid-sweep] baseline clean_CE={base_ce:.4f}")

    per_alpha = []
    for alpha in args.alphas:
        asr = asr_on_prompts(model, sae_mid, hook, args.feature_idx, alpha,
                              dep_tokens, dep_mask, dep_marker, args.gen_tokens)
        ce, dCE = clean_ce(model, sae_mid, hook, args.feature_idx, alpha,
                            cln_tokens, cln_mask, cln_marker, base_ce)
        per_alpha.append({"alpha": alpha, "asr_16": asr, "clean_ce": ce, "delta_ce": dCE})
        print(f"[mid-sweep]  α={alpha}: ASR={asr:.3f}  ΔCE={dCE:+.4f}")

    out = {
        "meta": {
            "hook": hook,
            "feature_idx": args.feature_idx,
            "alphas": args.alphas,
            "gen_tokens": args.gen_tokens,
            "sae_mid": str(args.sae_mid),
        },
        "baseline": {"clean_ce": base_ce},
        "per_alpha": per_alpha,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(f"[mid-sweep] wrote {args.output}")


if __name__ == "__main__":
    main()
