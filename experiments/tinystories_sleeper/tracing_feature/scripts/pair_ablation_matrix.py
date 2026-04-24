"""Pairwise head ablation matrix on top heads — probes head interactions.

For each head in a candidate set (default: top heads from the dep-specific lift
and the gladiators h=12, h=15), measure teacher-forced sleeper log-prob under:
  (i)  singleton ablation {h}
  (ii) pairwise ablation {h_i, h_j}

Non-additivity = Δlogp({h_i, h_j}) − Δlogp({h_i}) − Δlogp({h_j}). Super-additive
pairs (more negative than sum of parts) cooperate; sub-additive pairs cancel.

Output: tracing_feature/results/pair_ablation_matrix.json
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


def zero_heads_hook(head_ids):
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
    p.add_argument("--output_dir", default=str(HERE.parent / "results"))
    p.add_argument("--heads", nargs="+", type=int,
                   default=[0, 3, 5, 7, 8, 9, 12, 15],
                   help="Heads to form pairs over.")
    p.add_argument("--device", default=None)
    p.add_argument("--chunk_size", type=int, default=16)
    p.add_argument("--skip_feature_readout", action="store_true",
                   help="Skip the z_mid forward pass; only measure logp.")
    args = p.parse_args()

    device = pick_device(args.device)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[pair-abl] device={device}; heads={args.heads}")
    cache = torch.load(args.cache, weights_only=False)
    meta = cache["meta"]
    mid_f = meta["suppressor"]["mid_feature"]

    model = load_sleeper_model(device=device)
    sae_mid = None
    if not args.skip_feature_readout:
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
    uniq_markers = dep_marker.unique().tolist()

    @torch.no_grad()
    def measure(head_ids):
        hook = zero_heads_hook(head_ids) if head_ids else None
        fwd_hooks = [hook] if head_ids else []
        # teacher-forced sleeper logp per prompt
        logps = []
        for m_pos in uniq_markers:
            rows = (dep_marker == m_pos).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            P = int(m_pos) + 1
            trunc = dep_tokens[rows, :P].to(device)
            lp = teacher_forced_sleeper_logp(model, model.tokenizer, trunc, fwd_hooks=fwd_hooks)
            logps.append(lp.cpu())
        mean_logp = float(torch.cat(logps).mean().item())
        feat = float("nan")
        if not args.skip_feature_readout:
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
        return mean_logp, feat

    # Baseline
    print(f"[pair-abl] baseline...")
    logp0, feat0 = measure([])
    print(f"[pair-abl]   logp={logp0:+.3f}  feat={feat0:+.4f}")

    # Singletons
    singletons = {}
    for h in args.heads:
        logp, feat = measure([h])
        singletons[h] = {"logp": logp, "delta_logp": logp - logp0, "feat": feat}
        print(f"[pair-abl] {{{h}}}: logp={logp:+.3f}  Δlogp={logp-logp0:+.3f}")

    # Pairs
    pairs = {}
    heads_sorted = sorted(args.heads)
    for i, h_i in enumerate(heads_sorted):
        for h_j in heads_sorted[i + 1:]:
            logp, feat = measure([h_i, h_j])
            dlogp = logp - logp0
            pred_additive = singletons[h_i]["delta_logp"] + singletons[h_j]["delta_logp"]
            pairs[f"{h_i},{h_j}"] = {
                "heads": [h_i, h_j],
                "logp": logp, "delta_logp": dlogp, "feat": feat,
                "additive_sum": pred_additive,
                "non_additivity": dlogp - pred_additive,
            }
            print(f"[pair-abl] {{{h_i},{h_j}}}: Δlogp={dlogp:+.3f} "
                  f"(additive={pred_additive:+.3f}, non-add={dlogp-pred_additive:+.3f})")

    out = {
        "mid_feature": int(mid_f),
        "heads": args.heads,
        "baseline": {"logp": logp0, "feat": feat0},
        "singletons": singletons,
        "pairs": pairs,
    }
    out_path = out_dir / "pair_ablation_matrix.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[pair-abl] wrote {out_path}")


if __name__ == "__main__":
    main()
