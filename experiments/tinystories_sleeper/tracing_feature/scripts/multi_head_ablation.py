"""Phase 4b: multi-head ablation. Zero multiple block-0 heads simultaneously
to test whether the "distributed attention circuit" identified in pre_attn_path
is causally load-bearing.

We run a sweep of group ablations:
  - Baseline (no ablation)
  - Top-1 head (argmax Var(S_h))
  - Top-2 heads
  - Top-3, 4, 5 heads
  - All 16 heads (full attention ablation)
  - Control: bottom-K heads by Var(S_h)

For each, measure:
  - mean z_mid[mid_f] on deployment prompt positions
  - ASR_16 on deployment test prompts
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
    asr_16,
    encode_all_sae,
    greedy_generate_with_hooks,
    load_paired_dataset,
    load_sleeper_model,
    prompt_mask_from_markers,
)


def pick_device(explicit: str | None) -> str:
    if explicit:
        return explicit
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_zero_heads_hook(head_ids: list[int]):
    head_ids = list(head_ids)

    def _hook(z, hook):
        if not head_ids:
            return z
        for h in head_ids:
            z[:, :, h, :] = 0.0
        return z

    return ("blocks.0.attn.hook_z", _hook)


@torch.no_grad()
def forward_and_encode(model, sae_mid, tokens, head_ids, mid_f, dep_mask, chunk_size=16):
    device = next(model.parameters()).device
    hook = make_zero_heads_hook(head_ids)
    fwd_hooks = [hook] if head_ids else []
    mids = []
    for start in range(0, tokens.shape[0], chunk_size):
        batch = tokens[start : start + chunk_size].to(device)
        with model.hooks(fwd_hooks=fwd_hooks):
            _, cache = model.run_with_cache(
                batch,
                return_type=None,
                names_filter=lambda n: n == "blocks.0.hook_resid_mid",
            )
        mids.append(cache["blocks.0.hook_resid_mid"].to(torch.float16).cpu())
    resid_mid = torch.cat(mids, dim=0)                      # (N, T, d_model)
    z = encode_all_sae(sae_mid, resid_mid.float(), chunk_size=256)
    z_f = z[:, :, mid_f]
    return float(z_f[dep_mask.bool()].mean().item())


@torch.no_grad()
def asr_on_deployment(model, tokens, marker_pos, head_ids, max_new_tokens=16):
    hook = make_zero_heads_hook(head_ids)
    fwd_hooks = [hook] if head_ids else []
    uniq = marker_pos.unique().tolist()
    hits = 0
    total = 0
    for m_pos in uniq:
        rows = (marker_pos == m_pos).nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            continue
        P = int(m_pos) + 1
        trunc = tokens[rows, :P]
        gen = greedy_generate_with_hooks(model, trunc, fwd_hooks, max_new_tokens)
        hits += int(round(asr_16(gen, model.tokenizer) * gen.shape[0]))
        total += gen.shape[0]
    return hits / max(1, total)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=str(HERE.parent / "results" / "layer0_cache.pt"))
    parser.add_argument("--ov", default=str(HERE.parent / "results" / "ov_path_per_pair.pt"))
    parser.add_argument("--output_dir", default=str(HERE.parent / "results"))
    parser.add_argument("--gen_tokens", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--chunk_size", type=int, default=16)
    parser.add_argument("--group_sizes", nargs="+", type=int, default=[1, 3, 5, 8],
                        help="Number of top heads to ablate per group.")
    parser.add_argument("--rank_by", default="dep_minus_clean",
                        choices=["variance", "dep_minus_clean"],
                        help="How to rank heads: by Var(S_h) or by |mean(S_h|dep) - mean(S_h|clean)|.")
    args = parser.parse_args()

    device = pick_device(args.device)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[multi-abl] loading cache + ov...")
    cache = torch.load(args.cache, weights_only=False)
    ov = torch.load(args.ov, weights_only=False)
    meta = cache["meta"]
    mid_f = meta["suppressor"]["mid_feature"]
    n_heads = meta["n_heads"]

    S = ov["per_head_total"]

    if args.rank_by == "variance":
        score = S.var(dim=(0, 1))
    else:
        # deployment specificity: |mean(S_h | dep, prompt) - mean(S_h | clean, prompt)|
        is_dep = cache["is_deployment"]
        marker = cache["story_marker_pos"]
        T = S.shape[1]
        idx = torch.arange(T).unsqueeze(0)
        prompt_mask = idx <= marker.unsqueeze(1)
        dep_mask = is_dep.unsqueeze(1) & prompt_mask
        cln_mask = (~is_dep).unsqueeze(1) & prompt_mask
        mean_dep = torch.stack([S[:, :, h][dep_mask].mean() for h in range(S.shape[-1])])
        mean_cln = torch.stack([S[:, :, h][cln_mask].mean() for h in range(S.shape[-1])])
        score = (mean_dep - mean_cln).abs()

    order_top = torch.argsort(score, descending=True).tolist()
    order_bottom = list(reversed(order_top))

    print(f"[multi-abl] ranked by '{args.rank_by}': top→bottom order: {order_top}")
    for h in order_top:
        print(f"[multi-abl]   h={h:2d}  score={score[h].item():+.4e}")

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

    def _run(label, head_ids):
        print(f"[multi-abl] {label}: heads={head_ids}")
        feat = forward_and_encode(model, sae_mid, dep_tokens, head_ids, mid_f, dep_mask, args.chunk_size)
        asr = asr_on_deployment(model, dep_tokens, dep_marker, head_ids, args.gen_tokens)
        print(f"[multi-abl]   mean z_mid[f={mid_f}]={feat:.4f}  ASR_16={asr:.3f}")
        return {"label": label, "heads": list(head_ids), "mean_feature": feat, "asr_16": asr}

    results = []
    results.append(_run("baseline", []))

    # Top-K by variance
    for k in args.group_sizes:
        k = min(k, n_heads)
        results.append(_run(f"top{k}_by_var", order_top[:k]))

    # Bottom-K as control
    for k in args.group_sizes:
        if k >= n_heads:
            continue
        k = min(k, n_heads)
        results.append(_run(f"bottom{k}_by_var", order_bottom[:k]))

    baseline_feat = results[0]["mean_feature"]
    baseline_asr = results[0]["asr_16"]
    for r in results:
        r["delta_feature"] = r["mean_feature"] - baseline_feat
        r["delta_asr"] = r["asr_16"] - baseline_asr

    out = {
        "mid_feature": int(mid_f),
        "baseline": {"mean_feature": baseline_feat, "asr_16": baseline_asr},
        "order_top_by_var": order_top,
        "results": results,
    }
    out_path = out_dir / "multi_head_ablation.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[multi-abl] wrote {out_path}")


if __name__ == "__main__":
    main()
