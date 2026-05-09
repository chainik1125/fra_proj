"""Head ablation evaluated at each prompt's baseline q* position.

This is a refinement of `head_ablation.py`. Instead of averaging z_mid[mid_f]
over all prompt positions, it fixes

    q*(b) = argmax_t z_mid[b, t, mid_f]

within the cached baseline prompt for each deployment example b, and measures
the ablated feature exactly at q*(b). This better matches the local bottleneck
picture used in the tracing analysis.
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
def run_ablated_forward(model, tokens, head_ids, chunk_size=16):
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
    return torch.cat(mids, dim=0)


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
        prompt_len = int(m_pos) + 1
        trunc = tokens[rows, :prompt_len]
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
    parser.add_argument("--gen_tokens", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--top_head", type=int, required=True)
    parser.add_argument("--control_head", type=int, default=1)
    parser.add_argument("--chunk_size", type=int, default=16)
    args = parser.parse_args()

    device = pick_device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = torch.load(args.cache, weights_only=False)
    ov = torch.load(args.ov, weights_only=False)
    meta = cache["meta"]
    mid_f = meta["suppressor"]["mid_feature"]
    split_name = meta["split"]

    tokens = cache["tokens"]
    is_deploy = cache["is_deployment"]
    marker_pos = cache["story_marker_pos"]
    dep_idx = torch.where(is_deploy)[0]
    dep_tokens = tokens[dep_idx]
    dep_marker = marker_pos[dep_idx]

    prompt_mask = prompt_mask_from_markers(meta["seq_len"], marker_pos)
    z_mid_cache = cache["encodings"]["z_mid"][:, :, mid_f]                  # (N, T)
    dep_prompt_mask = prompt_mask[dep_idx]
    masked = z_mid_cache[dep_idx].masked_fill(~dep_prompt_mask, -float("inf"))
    q_star = masked.argmax(dim=1)                                           # (N_dep,)
    row = torch.arange(dep_idx.numel())

    model = load_sleeper_model(device=device)
    sae_mid, _ = load_crosscoder(EXP_DIR / meta["sae_configs"]["mid"]["path"], device=device)

    def _metrics_for(head_ids: list[int]) -> dict[str, float]:
        resid_mid = run_ablated_forward(model, dep_tokens, head_ids, args.chunk_size)
        z = encode_all_sae(sae_mid, resid_mid.float(), chunk_size=256)
        z_f = z[:, :, mid_f]
        return {
            "mean_feature_qstar": float(z_f[row, q_star].mean().item()),
            "mean_feature_prompt": float(z_f[dep_prompt_mask.bool()].mean().item()),
            "asr_16": float(asr_on_deployment(model, dep_tokens, dep_marker, head_ids, args.gen_tokens)),
        }

    baseline = _metrics_for([])
    top = _metrics_for([args.top_head])
    control = _metrics_for([args.control_head])

    result = {
        "target": {
            "mid_feature": int(mid_f),
            "split": split_name,
            "n_deployment_prompts": int(dep_idx.numel()),
            "top_head": int(args.top_head),
            "control_head": int(args.control_head),
        },
        "q_star": {
            "mean_position": float(q_star.float().mean().item()),
            "histogram": torch.bincount(q_star, minlength=int(tokens.shape[1])).tolist(),
        },
        "baseline": baseline,
        "top_head_ablated": {
            **top,
            "delta_feature_qstar": top["mean_feature_qstar"] - baseline["mean_feature_qstar"],
            "delta_feature_prompt": top["mean_feature_prompt"] - baseline["mean_feature_prompt"],
            "delta_asr": top["asr_16"] - baseline["asr_16"],
        },
        "control_head_ablated": {
            **control,
            "delta_feature_qstar": control["mean_feature_qstar"] - baseline["mean_feature_qstar"],
            "delta_feature_prompt": control["mean_feature_prompt"] - baseline["mean_feature_prompt"],
            "delta_asr": control["asr_16"] - baseline["asr_16"],
        },
        "var_per_head": ov["per_head_total"].var(dim=(0, 1)).tolist(),
    }

    out_json = out_dir / f"head_ablation_qstar_head{args.top_head}.json"
    out_json.write_text(json.dumps(result, indent=2))
    print(f"[head-q*] wrote {out_json}")


if __name__ == "__main__":
    main()
