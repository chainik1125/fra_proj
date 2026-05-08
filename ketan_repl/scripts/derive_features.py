"""Per-seed feature derivation: extract top-k OV-signed and top-k QK-L1 features.

Reads the cache + ov_path_per_pair.pt + qk_concentration.json from a per-seed
tracing_feature/results dir and writes a `features.json` consumable by the
modified pareto_3x3.py via its --features_json flag.

Usage:
    python derive_features.py \
        --tracing_results PATH \
        --top_k 3 \
        --output features.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tracing_results", type=Path, required=True,
                   help="dir containing layer0_cache.pt, ov_path_per_pair.pt, qk_concentration.json")
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    ov_path = args.tracing_results / "ov_path_per_pair.pt"
    qk_path = args.tracing_results / "qk_concentration.json"

    # ---- OV: one-stage signed sum on dep prompts, summed across heads.
    ov = torch.load(ov_path, weights_only=False)
    per_pair_dep = ov["per_pair_dep_contrib"]               # (n_heads, d_sae_ln1)
    s_lambda = per_pair_dep.sum(dim=0)                      # (d_sae_ln1,)
    top_ov_idx = s_lambda.abs().argsort(descending=True)[: args.top_k].tolist()
    top_ov = [int(i) for i in top_ov_idx]

    # ---- QK: L1-mean over dep prompt positions.
    qk = json.loads(qk_path.read_text())
    # qk_concentration.json structure: pull the per-feature 'L1_mean' ranking.
    # The script writes a top-N list under one of several keys; find it.
    by_feat = qk.get("per_feature_top") or qk.get("query_features") or qk.get("rankings")
    top_qk: list[int]
    if isinstance(by_feat, list) and by_feat and "feature_idx" in by_feat[0]:
        # Already a sorted list of dicts.
        # Prefer L1_mean ordering when available; else use the first 'rank' field.
        ranked = sorted(
            [d for d in by_feat if "L1_mean" in d],
            key=lambda d: -abs(d["L1_mean"]),
        )
        if ranked:
            top_qk = [int(d["feature_idx"]) for d in ranked[: args.top_k]]
        else:
            top_qk = [int(d["feature_idx"]) for d in by_feat[: args.top_k]]
    elif isinstance(by_feat, dict) and "L1_mean" in by_feat:
        # Per-metric dict — use L1_mean ranking.
        l1 = by_feat["L1_mean"]
        # Could be dict {feature_idx: value} or list of dicts.
        if isinstance(l1, dict):
            pairs = [(int(k), float(v)) for k, v in l1.items()]
            pairs.sort(key=lambda kv: -abs(kv[1]))
            top_qk = [k for k, _ in pairs[: args.top_k]]
        else:
            top_qk = [int(d["feature_idx"]) for d in l1[: args.top_k]]
    else:
        # Fallback: just take the first top_k entries from whatever's there.
        # Print the available keys so we can fix this if it triggers.
        print(f"[derive] WARN: unexpected qk_concentration.json shape — keys={list(qk.keys())[:8]}")
        # Try to grab any obvious feature-level ranking.
        for key in ("top_features_l1_mean", "top_features", "L1_mean_top"):
            if key in qk and isinstance(qk[key], list):
                top_qk = [int(d["feature_idx"]) if isinstance(d, dict) else int(d)
                          for d in qk[key][: args.top_k]]
                break
        else:
            raise SystemExit("[derive] could not locate QK ranking; inspect qk_concentration.json")

    union = sorted(set(top_ov) | set(top_qk))

    out = {
        "ov":    top_ov,
        "qk":    top_qk,
        "union": union,
        "_provenance": {
            "ov_path_per_pair": str(ov_path),
            "qk_concentration_json": str(qk_path),
            "top_k": args.top_k,
            "ov_score_top": [float(s_lambda[i].item()) for i in top_ov],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(f"[derive] wrote {args.output}: ov={top_ov}  qk={top_qk}")


if __name__ == "__main__":
    main()
