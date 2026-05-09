"""Phase 4: counterfactual validation via attention-head ablation.

Zero individual block-0 attention heads (by slicing `blocks.0.attn.hook_z`),
measure:
  - mean SAE_mid[mid_f] activation on deployment prompt positions
  - sampled ASR_16 on deployment test prompts

Compares the top-contributing head (from ov_path) against a low-contributing
control. The prediction: ablating the OV-identified head collapses the mid
suppressor's activation and restores baseline ASR; ablating the control does
neither.
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
    """Return a fwd-hook that zeros the z slices for the given heads at
    `blocks.0.attn.hook_z`. z shape: (B, T, n_heads, d_head)."""
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
    """Forward tokens with heads `head_ids` ablated; cache `blocks.0.hook_resid_mid`."""
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
    """Group by marker position, greedy-decode with head ablation, count ASR_16."""
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
    parser.add_argument("--gen_tokens", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--top_head", type=int, default=None,
                        help="Head to ablate (default: argmax Var(S_h) from ov_path).")
    parser.add_argument("--control_head", type=int, default=None,
                        help="Control head (default: argmin Var(S_h)).")
    parser.add_argument("--chunk_size", type=int, default=16)
    args = parser.parse_args()

    device = pick_device(args.device)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[head-abl] loading cache + ov results...")
    cache = torch.load(args.cache, weights_only=False)
    ov = torch.load(args.ov, weights_only=False)
    meta = cache["meta"]
    mid_f = meta["suppressor"]["mid_feature"]

    # Pick heads.
    S = ov["per_head_total"]                                     # (N, T, n_heads)
    head_var = S.var(dim=(0, 1))
    order = torch.argsort(head_var, descending=True).tolist()
    top_head = args.top_head if args.top_head is not None else order[0]
    control_head = args.control_head if args.control_head is not None else order[-1]
    print(f"[head-abl] top_head={top_head} (Var={head_var[top_head].item():.4e})  "
          f"control_head={control_head} (Var={head_var[control_head].item():.4e})")

    # Load model.
    print(f"[head-abl] loading sleeper model...")
    model = load_sleeper_model(device=device)

    # Load SAE_mid for re-encoding mid activations under ablation.
    print(f"[head-abl] loading SAE_mid...")
    sae_mid, _ = load_crosscoder(EXP_DIR / meta["sae_configs"]["mid"]["path"], device=device)

    # Load the same test split the cache used (reproducible via load_paired_dataset).
    split_name = meta["split"]
    splits = load_paired_dataset(
        tokenizer=model.tokenizer,
        n_train=meta["n_train"], n_val=meta["n_val"], n_test=meta["n_test"],
        seq_len=meta["seq_len"], seed=meta["seed"],
    )
    pt = splits[split_name]
    prompt_mask = prompt_mask_from_markers(meta["seq_len"], pt.story_marker_pos)

    dep_idx = torch.where(pt.is_deployment)[0]
    dep_tokens = pt.tokens[dep_idx]
    dep_marker = pt.story_marker_pos[dep_idx]
    dep_mask = prompt_mask[dep_idx]

    def _mean_feature_on_dep(resid_mid_tensor: torch.Tensor) -> float:
        """Mean z_mid[mid_f] on deployment prompt positions."""
        z = encode_all_sae(sae_mid, resid_mid_tensor.float(), chunk_size=256)
        z_f = z[:, :, mid_f]                                      # (N_dep, T)
        return float(z_f[dep_mask.bool()].mean().item())

    # ---- Baseline (no ablation) ----
    print(f"[head-abl] baseline: forward + mean feature + ASR...")
    resid_mid_base = run_ablated_forward(model, dep_tokens, [], args.chunk_size)
    feat_base = _mean_feature_on_dep(resid_mid_base)
    asr_base = asr_on_deployment(model, dep_tokens, dep_marker, [], args.gen_tokens)
    print(f"[head-abl]   mean z_mid[f={mid_f}]={feat_base:.4f}  ASR_16={asr_base:.3f}")

    # ---- Top head ablation ----
    print(f"[head-abl] ablating head {top_head}...")
    resid_mid_top = run_ablated_forward(model, dep_tokens, [top_head], args.chunk_size)
    feat_top = _mean_feature_on_dep(resid_mid_top)
    asr_top = asr_on_deployment(model, dep_tokens, dep_marker, [top_head], args.gen_tokens)
    print(f"[head-abl]   mean z_mid[f={mid_f}]={feat_top:.4f}  ASR_16={asr_top:.3f}  "
          f"(Δfeat={feat_top - feat_base:+.4f}, ΔASR={asr_top - asr_base:+.3f})")

    # ---- Control head ablation ----
    print(f"[head-abl] ablating control head {control_head}...")
    resid_mid_ctrl = run_ablated_forward(model, dep_tokens, [control_head], args.chunk_size)
    feat_ctrl = _mean_feature_on_dep(resid_mid_ctrl)
    asr_ctrl = asr_on_deployment(model, dep_tokens, dep_marker, [control_head], args.gen_tokens)
    print(f"[head-abl]   mean z_mid[f={mid_f}]={feat_ctrl:.4f}  ASR_16={asr_ctrl:.3f}  "
          f"(Δfeat={feat_ctrl - feat_base:+.4f}, ΔASR={asr_ctrl - asr_base:+.3f})")

    result = {
        "target": {
            "mid_feature": int(mid_f),
            "n_deployment_prompts": int(dep_tokens.shape[0]),
            "split": split_name,
        },
        "heads": {
            "top_head": int(top_head),
            "control_head": int(control_head),
            "var_per_head": head_var.tolist(),
        },
        "baseline": {
            "mean_feature_on_dep": feat_base,
            "asr_16": asr_base,
        },
        "top_head_ablated": {
            "mean_feature_on_dep": feat_top,
            "asr_16": asr_top,
            "delta_feature": feat_top - feat_base,
            "delta_asr": asr_top - asr_base,
        },
        "control_head_ablated": {
            "mean_feature_on_dep": feat_ctrl,
            "asr_16": asr_ctrl,
            "delta_feature": feat_ctrl - feat_base,
            "delta_asr": asr_ctrl - asr_base,
        },
    }
    out_json = out_dir / "head_ablation.json"
    out_json.write_text(json.dumps(result, indent=2))
    print(f"[head-abl] wrote {out_json}")


if __name__ == "__main__":
    main()
