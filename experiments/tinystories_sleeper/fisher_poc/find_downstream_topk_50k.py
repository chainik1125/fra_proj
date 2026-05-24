"""Minimal top-K downstream attribution for resid_mid SAEs at any path.

Mirrors stage-0 of find_downstream_winners_6seeds.py: rank features by
the dep-vs-clean activation difference (averaged over prompt positions).
Saves top-K to JSON in the same schema as downstream_winners_6seeds.json:

    {"s0": {"top_k": [...], "sae_path": "..."}, "s1": {...}, ...}
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import torch
from sleeper.metrics import rank_features_by_dep_clean
from sleeper.model import load_paired_dataset, load_sleeper_model, prompt_mask_from_markers
from sleeper.sae import encode_all, load as sae_load


RESID_MID = "blocks.0.hook_resid_mid"
SEQ_LEN = 128
N_SEL = 100


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--sae_dir", default="weights/seeds_50k")
    p.add_argument("--sae_pattern", default="sae_resid_mid_s{seed}.pt")
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[topk] device={device}")
    model = load_sleeper_model(device=device)

    splits = load_paired_dataset(
        tokenizer=model.tokenizer,
        n_train=2, n_val=N_SEL, n_test=0,
        seq_len=SEQ_LEN, seed=0,
    )
    sel = splits["val"]
    sel_pmask = prompt_mask_from_markers(SEQ_LEN, sel.story_marker_pos).to(device)
    is_dep = sel.is_deployment.to(device)
    combined_lp = sel.tokens.to(device)

    out: dict = {}
    for seed in args.seeds:
        sae_path = Path(args.sae_dir) / args.sae_pattern.format(seed=seed)
        print(f"[topk] seed={seed}  sae={sae_path}")
        sae, _ = sae_load(sae_path, device=device)

        _, cache = model.run_with_cache(
            combined_lp, return_type=None,
            names_filter=lambda n: n == RESID_MID,
        )
        z = encode_all(sae, cache[RESID_MID]).to(device)
        ranked = rank_features_by_dep_clean(z, is_dep, sel_pmask, top_k=args.top_k)
        top_k = ranked["top_indices"].cpu().tolist()
        print(f"[topk] s{seed} top-{args.top_k}: {top_k}")
        out[f"s{seed}"] = {
            "top_k": [int(x) for x in top_k],
            "sae_path": str(sae_path),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"[topk] wrote {args.out}")


if __name__ == "__main__":
    main()
