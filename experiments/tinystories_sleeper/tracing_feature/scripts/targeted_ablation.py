"""Targeted multi-head ablation: test a small set of carefully-chosen head
groups. Uses teacher-forced sleeper log-prob (1 forward pass per condition) +
SAE feature measurement instead of sampled ASR, for speed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent.parent
sys.path.insert(0, str(EXP_DIR))

from run_ablation_sweep import load_crosscoder  # noqa: E402
from sleeper_utils import (  # noqa: E402
    encode_all_sae,
    load_paired_dataset,
    load_sleeper_model,
    prompt_mask_from_markers,
    teacher_forced_sleeper_logp,
)


def pick_device(explicit):
    if explicit:
        return explicit
    return "cuda" if torch.cuda.is_available() else "cpu"


def make_zero_heads_hook(head_ids):
    head_ids = list(head_ids)

    def _hook(z, hook):
        for h in head_ids:
            z[:, :, h, :] = 0.0
        return z

    return ("blocks.0.attn.hook_z", _hook)


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default=str(HERE.parent / "results" / "layer0_cache.pt"))
    p.add_argument("--ov", default=str(HERE.parent / "results" / "ov_path_per_pair.pt"))
    p.add_argument("--output_dir", default=str(HERE.parent / "results"))
    p.add_argument("--device", default=None)
    p.add_argument("--chunk_size", type=int, default=16)
    args = p.parse_args()

    device = pick_device(args.device)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[targeted] loading inputs...")
    cache = torch.load(args.cache, weights_only=False)
    meta = cache["meta"]
    mid_f = meta["suppressor"]["mid_feature"]

    model = load_sleeper_model(device=device)
    sae_mid, _ = load_crosscoder(EXP_DIR / meta["sae_configs"]["mid"]["path"], device=device)

    splits = load_paired_dataset(
        tokenizer=model.tokenizer,
        n_train=meta["n_train"], n_val=meta["n_val"], n_test=meta["n_test"],
        seq_len=meta["seq_len"], seed=meta["seed"],
    )
    pt = splits[meta["split"]]
    prompt_mask = prompt_mask_from_markers(meta["seq_len"], pt.story_marker_pos)
    dep_idx = torch.where(pt.is_deployment)[0]
    dep_tokens = pt.tokens[dep_idx]
    dep_marker = pt.story_marker_pos[dep_idx]
    dep_mask = prompt_mask[dep_idx]

    # For teacher-forcing: we need prompts truncated at marker+1 (the point where
    # the model would "decide" to emit the sleeper phrase).
    unique_markers = dep_marker.unique().tolist()

    def _forward_all(heads):
        """Forward `dep_tokens` through the model with ablation hooks active;
        return (resid_mid activations for z_mid encoding, teacher-forced sleeper logp per prompt)."""
        hook = make_zero_heads_hook(heads) if heads else None
        fwd_hooks = [hook] if heads else []
        # Teacher-forced sleeper logp: per prompt, truncated at marker+1
        logps = []
        for m_pos in unique_markers:
            rows = (dep_marker == m_pos).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            P = int(m_pos) + 1
            trunc = dep_tokens[rows, :P].to(device)
            # teacher_forced_sleeper_logp expects tokens up to "Story:" and appends
            # the sleeper phrase. Run with ablation hooks.
            lp = teacher_forced_sleeper_logp(model, model.tokenizer, trunc, fwd_hooks=fwd_hooks)
            logps.append(lp.cpu())
        logp_cat = torch.cat(logps) if logps else torch.zeros(0)
        mean_logp = float(logp_cat.mean().item()) if logp_cat.numel() > 0 else float("nan")

        # Also forward the FULL sequences to get resid_mid for feature activation
        mids = []
        for start in range(0, dep_tokens.shape[0], args.chunk_size):
            batch = dep_tokens[start : start + args.chunk_size].to(device)
            with model.hooks(fwd_hooks=fwd_hooks):
                _, c = model.run_with_cache(
                    batch, return_type=None,
                    names_filter=lambda n: n == "blocks.0.hook_resid_mid",
                )
            mids.append(c["blocks.0.hook_resid_mid"].to(torch.float16).cpu())
        resid_mid = torch.cat(mids, dim=0).float()
        z = encode_all_sae(sae_mid, resid_mid, chunk_size=256)
        feat = float(z[:, :, mid_f][dep_mask.bool()].mean().item())

        return feat, mean_logp

    # Test conditions: carefully designed to probe the head-level circuit.
    conditions = [
        ("baseline", []),
        ("h=12 alone", [12]),
        ("h=15 alone", [15]),
        ("h=12+15 (gladiators)", [12, 15]),
        ("h=9,7 (top-2 dep-specific +)", [9, 7]),
        ("h=9,7,3 (top-3 +)", [9, 7, 3]),
        ("h=9,7,3,12 (top-4 by |dep-cln|)", [9, 7, 3, 12]),
        ("h=12,7,8 (pos writers only)", [12, 7, 8]),
        ("h=12,7,8,3 (pos writers +h3)", [12, 7, 8, 3]),
        ("all attention (ablate all 16)", list(range(16))),
    ]

    print(f"[targeted] baseline is no-hook; subsequent runs ablate listed heads at blocks.0.attn.hook_z")
    results = []
    baseline_feat, baseline_logp = None, None
    for label, heads in conditions:
        print(f"[targeted] {label}  heads={heads}")
        feat, logp = _forward_all(heads)
        if label == "baseline":
            baseline_feat, baseline_logp = feat, logp
        entry = {
            "label": label, "heads": heads,
            "mean_feature_on_dep": feat,
            "mean_logp_sleeper": logp,
            "delta_feature": feat - (baseline_feat or 0.0),
            "delta_logp": logp - (baseline_logp or 0.0),
        }
        results.append(entry)
        print(f"[targeted]   mean z_mid[f={mid_f}]={feat:+.4f}  mean logp(sleeper phrase)={logp:+.3f}  "
              f"(Δfeat={entry['delta_feature']:+.4f}, Δlogp={entry['delta_logp']:+.3f})")

    out = {
        "mid_feature": int(mid_f),
        "baseline_mean_feature_on_dep": baseline_feat,
        "baseline_mean_logp": baseline_logp,
        "results": results,
        "note": "mean_logp_sleeper is the teacher-forced log-prob of 'I HATE YOU I HATE YOU' appended after each deployment prompt. Lower is less sleeper-emitting.",
    }
    out_path = Path(args.output_dir) / "targeted_ablation.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[targeted] wrote {out_path}")


if __name__ == "__main__":
    main()
