"""Targeted ablation of specific SAE_ln1 features at blocks.0.ln1.hook_normalized.

Tests the hypothesis: ln1 features 870 and 1388 are the "causal bottleneck" —
they serve as both query-drivers in the QK circuit and OV write-recipients,
so ablating them should have outsized effect compared to the sweep-found
suppressor f=1412.

Uses the standard SAE-delta ablation: compute delta = alpha * (decode(z_abl) -
decode(z_orig)) where z_abl zeros the target feature; apply delta as an
additive hook at ln1.hook_normalized for prompt positions only.

Outputs: tracing_feature/results/ln1_feature_ablation.json
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
    compute_sae_delta,
    encode_all_sae,
    load_paired_dataset,
    load_sleeper_model,
    make_delta_hook_single_layer,
    prompt_mask_from_markers,
    teacher_forced_sleeper_logp,
)


def pick_device(explicit):
    if explicit:
        return explicit
    return "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default=str(HERE.parent / "results" / "layer0_cache.pt"))
    p.add_argument("--output_dir", default=str(HERE.parent / "results"))
    p.add_argument(
        "--features", nargs="+", type=int,
        default=[1412, 870, 1388, 1220, 221, 1114],
        help="Single ln1 features to ablate (one at a time).",
    )
    p.add_argument(
        "--feature_groups", nargs="*", default=[],
        help="Additional comma-separated ln1 feature groups to ablate jointly, e.g. 221,1114 1114,430",
    )
    p.add_argument("--alphas", nargs="+", type=float, default=[1.0, 2.0, 4.0])
    p.add_argument("--device", default=None)
    p.add_argument("--chunk_size", type=int, default=16)
    args = p.parse_args()

    device = pick_device(args.device)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    cache = torch.load(args.cache, weights_only=False)
    meta = cache["meta"]
    mid_f = meta["suppressor"]["mid_feature"]

    if "ln1" not in meta["sae_configs"]:
        raise SystemExit("[ln1-abl] SAE_ln1 not in cache meta.")

    model = load_sleeper_model(device=device)
    sae_ln1, ln1_cfg = load_crosscoder(EXP_DIR / meta["sae_configs"]["ln1"]["path"], device=device)
    sae_mid, _ = load_crosscoder(EXP_DIR / meta["sae_configs"]["mid"]["path"], device=device)
    ln1_hook = "blocks.0.ln1.hook_normalized"

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
    dep_mask_pt = prompt_mask[dep_idx]
    uniq_markers = dep_marker.unique().tolist()

    def _measure_logp_with_feature(feat_fs, alpha):
        """Compute per-marker-group delta, apply hook, measure logp + feature."""
        logps = []
        for m_pos in uniq_markers:
            rows = (dep_marker == m_pos).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            P = int(m_pos) + 1
            trunc = dep_tokens[rows, :P].to(device)
            trunc_mask = dep_mask_pt[rows, :P].to(device)
            if not feat_fs:  # baseline
                fwd_hooks = []
            else:
                # compute sum of deltas for given feature list
                delta = None
                for f in feat_fs:
                    d = compute_sae_delta(model, sae_ln1, ln1_hook, f, trunc, trunc_mask)
                    delta = d if delta is None else delta + d
                fwd_hooks = make_delta_hook_single_layer(delta, alpha, ln1_hook)
            lp = teacher_forced_sleeper_logp(model, model.tokenizer, trunc, fwd_hooks=fwd_hooks)
            logps.append(lp.cpu())
        mean_logp = float(torch.cat(logps).mean().item())
        # Mid feature: recompute delta per chunk to match batch shape.
        mids = []
        for start in range(0, dep_tokens.shape[0], args.chunk_size):
            end = min(start + args.chunk_size, dep_tokens.shape[0])
            batch = dep_tokens[start:end].to(device)
            batch_mask = dep_mask_pt[start:end].to(device)
            if not feat_fs:
                fwd_hooks_batch = []
            else:
                delta_b = None
                for f in feat_fs:
                    d = compute_sae_delta(model, sae_ln1, ln1_hook, f, batch, batch_mask)
                    delta_b = d if delta_b is None else delta_b + d
                fwd_hooks_batch = make_delta_hook_single_layer(delta_b, alpha, ln1_hook)
            with model.hooks(fwd_hooks=fwd_hooks_batch):
                _, c = model.run_with_cache(
                    batch, return_type=None,
                    names_filter=lambda n: n == "blocks.0.hook_resid_mid",
                )
            mids.append(c["blocks.0.hook_resid_mid"].to(torch.float16).cpu())
        resid_mid = torch.cat(mids, dim=0).float()
        z = encode_all_sae(sae_mid, resid_mid, chunk_size=256)
        feat = float(z[:, :, mid_f][dep_mask_pt.bool()].mean().item())
        return mean_logp, feat

    def _measure(fwd_hooks):  # baseline helper
        return _measure_logp_with_feature([], 0.0)

    print(f"[ln1-abl] baseline...")
    logp0, feat0 = _measure_logp_with_feature([], 0.0)
    print(f"[ln1-abl]   logp={logp0:+.3f}  z_mid[{mid_f}]={feat0:+.4f}")

    results = []
    for feat_f in args.features:
        for alpha in args.alphas:
            logp, feat = _measure_logp_with_feature([feat_f], alpha)
            entry = {
                "ln1_feature": feat_f, "alpha": alpha,
                "mean_logp_sleeper": logp, "mean_z_mid_171": feat,
                "delta_logp": logp - logp0, "delta_feat": feat - feat0,
            }
            results.append(entry)
            print(f"[ln1-abl] f={feat_f:4d}  α={alpha}: Δlogp={logp-logp0:+.3f}  Δfeat={feat-feat0:+.4f}")
            if device == "cuda":
                torch.cuda.empty_cache()

    feature_groups: list[tuple[str, list[int]]] = [("{870, 1388}", [870, 1388])]
    for group_spec in args.feature_groups:
        group = [int(tok.strip()) for tok in group_spec.split(",") if tok.strip()]
        if not group:
            continue
        label = "{" + ", ".join(str(f) for f in group) + "}"
        if group not in [g for _, g in feature_groups]:
            feature_groups.append((label, group))

    for label, group in feature_groups:
        for alpha in args.alphas:
            logp, feat = _measure_logp_with_feature(group, alpha)
            entry = {
                "ln1_feature": label, "alpha": alpha,
                "mean_logp_sleeper": logp, "mean_z_mid_171": feat,
                "delta_logp": logp - logp0, "delta_feat": feat - feat0,
            }
            results.append(entry)
            print(f"[ln1-abl] f={label}  α={alpha}: Δlogp={logp-logp0:+.3f}  Δfeat={feat-feat0:+.4f}")

    out = {
        "mid_feature": int(mid_f),
        "baseline": {"logp": logp0, "feat": feat0},
        "results": results,
    }
    out_path = out_dir / "ln1_feature_ablation.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[ln1-abl] wrote {out_path}")


if __name__ == "__main__":
    main()
