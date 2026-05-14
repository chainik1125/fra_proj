"""Rank SAE features by cosine similarity to the EM-vs-base difference
vector at L15 resid_post. Outputs top-N feature IDs as JSON.

Reads:
  - {actdiff_out}.diff_vector.pt    (from compute_arditi_actdiff.py)
  - andyrdt/saes-qwen2.5-7b-instruct/resid_post_layer_{layer}/trainer_{trainer}

Mirrors safety-research/open-source-em-features `sae_decomposition.py` —
project the activation-difference direction onto the SAE feature basis
(cosine similarity with each decoder column), return top-N.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--diff-pt", default="/workspace/actdiff_L15.diff_vector.pt",
                  help="Path to .diff_vector.pt produced by compute_arditi_actdiff.py")
    p.add_argument("--layer", type=int, default=15)
    p.add_argument("--trainer", type=int, default=1)
    p.add_argument("--sae-repo", default="andyrdt/saes-qwen2.5-7b-instruct")
    p.add_argument("--top-n", type=int, default=200)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="/workspace/arditi_top_features_L15.json")
    args = p.parse_args()

    print(f"[load] diff vector from {args.diff_pt}")
    d = torch.load(args.diff_pt, map_location=args.device, weights_only=False)
    diff = d["diff"].to(args.device).float()
    print(f"  shape={tuple(diff.shape)}  ‖diff‖={diff.norm().item():.4f}")

    print(f"[load] SAE {args.sae_repo}/resid_post_layer_{args.layer}/trainer_{args.trainer}")
    from huggingface_hub import snapshot_download
    from dictionary_learning.utils import load_dictionary

    subfolder = f"resid_post_layer_{args.layer}/trainer_{args.trainer}"
    local_dir = snapshot_download(repo_id=args.sae_repo, allow_patterns=f"{subfolder}/*")
    sae, _ = load_dictionary(str(Path(local_dir) / subfolder), device=args.device)
    sae.eval()
    W_dec = sae.decoder.weight.detach().float()  # (d_in, d_sae)
    d_in, d_sae = W_dec.shape
    print(f"  decoder shape={tuple(W_dec.shape)}")
    if d_in != diff.shape[0]:
        raise ValueError(f"d_in mismatch: SAE d_in={d_in} vs diff dim={diff.shape[0]}")

    # cosine sim of each decoder column with diff
    t0 = time.time()
    diff_n = diff / (diff.norm() + 1e-12)
    col_norms = W_dec.norm(dim=0).clamp_min(1e-12)
    # projection: each column dotted with diff_n, divided by its norm
    cos = (W_dec.t() @ diff_n) / col_norms       # (d_sae,)
    print(f"[rank] cos sim computed in {time.time()-t0:.2f}s")
    top_vals, top_idx = torch.topk(cos.abs(), k=args.top_n)
    # store signed cos sim alongside abs ranking
    signed = cos[top_idx].cpu().tolist()
    idx = top_idx.cpu().tolist()

    out = {
        "layer": args.layer,
        "trainer": args.trainer,
        "sae_repo": args.sae_repo,
        "diff_source": str(args.diff_pt),
        "diff_norm": float(diff.norm()),
        "top_n": args.top_n,
        "feature_ids":     idx,
        "cosine_signed":   signed,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[save] wrote {args.out}")
    # Quick preview
    print(f"\ntop 10 by |cos sim| to Δa:")
    for k in range(min(10, args.top_n)):
        print(f"  rank {k+1:>3}: feature F{idx[k]:<7d}  cos = {signed[k]:+.4f}")


if __name__ == "__main__":
    main()
