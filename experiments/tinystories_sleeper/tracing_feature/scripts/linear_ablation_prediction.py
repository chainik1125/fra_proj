"""Linear ablation prediction: compute the predicted drop in pre_171 if we
zero a set of attention heads, using the per-head S_h(q) stored in ov_path.

This is exact *given* the attention pattern is held fixed. Compared to
multi_head_ablation.py (which actually re-runs the forward pass and generation),
this predicts only the pre-activation change, not the downstream ASR effect.

Predictions are useful to compare against the actual ablation numbers and to
rank head sets without paying the generation cost.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from itertools import combinations

import torch

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent.parent
sys.path.insert(0, str(EXP_DIR))


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=str(HERE.parent / "results" / "layer0_cache.pt"))
    parser.add_argument("--ov", default=str(HERE.parent / "results" / "ov_path_per_pair.pt"))
    parser.add_argument("--output_dir", default=str(HERE.parent / "results"))
    parser.add_argument("--group_sizes", nargs="+", type=int, default=[1, 2, 3, 4, 5, 8])
    args = parser.parse_args()

    cache = torch.load(args.cache, weights_only=False)
    ov = torch.load(args.ov, weights_only=False)

    S = ov["per_head_total"]                                # (N, T, n_heads)
    is_dep = cache["is_deployment"]
    marker = cache["story_marker_pos"]
    T = S.shape[1]
    n_heads = S.shape[-1]

    idx = torch.arange(T).unsqueeze(0)
    prompt_mask = idx <= marker.unsqueeze(1)
    dep_mask = is_dep.unsqueeze(1) & prompt_mask
    cln_mask = (~is_dep).unsqueeze(1) & prompt_mask

    mean_dep = torch.stack([S[:, :, h][dep_mask].mean() for h in range(n_heads)])
    mean_cln = torch.stack([S[:, :, h][cln_mask].mean() for h in range(n_heads)])

    # Baseline pre_171 mean on dep/clean (reconstructed via sum over heads + skip + const).
    # Use the pre_activation logits stored in the cache.
    pre_171 = cache["pre_logits"]["mid_f_mid"]              # (N, T)
    baseline_dep = pre_171[dep_mask].mean().item()
    baseline_cln = pre_171[cln_mask].mean().item()
    print(f"baseline pre_171: dep={baseline_dep:+.4f}  cln={baseline_cln:+.4f}")

    # Two ranking orders: by variance, by |dep - clean|, by mean_dep magnitude.
    rank_var = torch.argsort(S.var(dim=(0, 1)), descending=True).tolist()
    rank_dep_cln = torch.argsort((mean_dep - mean_cln).abs(), descending=True).tolist()
    rank_mean_dep = torch.argsort(mean_dep.abs(), descending=True).tolist()

    def _predict(head_set):
        head_set = list(head_set)
        # Linear prediction: pre_171 after ablation = baseline - Σ_{h in K} mean(S_h | mask)
        # This is exact given frozen attention pattern (and no downstream effects).
        drop_dep = sum(mean_dep[h].item() for h in head_set)
        drop_cln = sum(mean_cln[h].item() for h in head_set)
        new_dep = baseline_dep - drop_dep
        new_cln = baseline_cln - drop_cln
        return {
            "heads": head_set,
            "predicted_new_pre_171_dep": new_dep,
            "predicted_new_pre_171_cln": new_cln,
            "delta_dep": -drop_dep,
            "delta_cln": -drop_cln,
        }

    results = {
        "baseline": {"pre_171_dep": baseline_dep, "pre_171_cln": baseline_cln},
        "ranking_by": {
            "variance": rank_var,
            "dep_minus_clean_abs": rank_dep_cln,
            "mean_dep_abs": rank_mean_dep,
        },
        "by_ranking": {},
    }

    for ranking_name, order in [("variance", rank_var),
                                 ("dep_minus_clean_abs", rank_dep_cln),
                                 ("mean_dep_abs", rank_mean_dep)]:
        results["by_ranking"][ranking_name] = []
        for k in args.group_sizes:
            heads = order[:k]
            pred = _predict(heads)
            pred["label"] = f"top{k}_by_{ranking_name}"
            results["by_ranking"][ranking_name].append(pred)
            print(f"{ranking_name:24s} top-{k:2d}: heads={heads}  "
                  f"predicted pre_171[dep]={pred['predicted_new_pre_171_dep']:+.4f} "
                  f"(Δ={pred['delta_dep']:+.4f})  "
                  f"predicted pre_171[cln]={pred['predicted_new_pre_171_cln']:+.4f} "
                  f"(Δ={pred['delta_cln']:+.4f})")

    # Best single head (by biggest predicted drop on dep when pre_171 is initially positive)
    single_drops = [(h, mean_dep[h].item()) for h in range(n_heads)]
    print(f"\nper-head predicted drop in pre_171[dep] if ablated:")
    for h, d in sorted(single_drops, key=lambda x: x[1], reverse=True):
        new_dep = baseline_dep - d
        print(f"  h={h:2d}  mean_S_h(dep)={d:+.4f}  post-ablate pre_171[dep]={new_dep:+.4f}")

    out_path = Path(args.output_dir) / "linear_ablation_prediction.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
