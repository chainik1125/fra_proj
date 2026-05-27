"""Replay the Stage-1 (Δlogp) screen of downstream_baseline.py on a saved
baseline JSON and report each Stage-2 winner's rank within the Δlogp ordering.

For each per-seed entry in --baseline_json:
  1. Load sae_resid_mid_s{seed}.pt
  2. For each feature f in info["top_k"]:
       Δlogp[f] = min over screen_alphas of (teacher_forced_sleeper_logp(steered) - base_logp)
  3. Sort top_k by Δlogp ascending (best suppression first)
  4. Report the rank of info["winner"] in that sorted list
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sleeper.hooks import additive_steer_hook, compute_sae_delta
from sleeper.metrics import teacher_forced_sleeper_logp
from sleeper.model import (
    MODELS, load_paired_dataset, load_sleeper_model, prompt_mask_from_markers,
)
from sleeper.sae import load as sae_load

RESID_MID = "blocks.0.hook_resid_mid"


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline_json", type=Path, required=True)
    p.add_argument("--sae_dir",       type=Path, required=True)
    p.add_argument("--model",         choices=list(MODELS), default="tinystories")
    p.add_argument("--screen_alphas", type=float, nargs="+", default=[2.0, 4.0])
    p.add_argument("--n_sel",         type=int,   default=200)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = json.loads(args.baseline_json.read_text())
    model = load_sleeper_model(model=args.model, device=device)
    tok = model.tokenizer

    # Selection split (same as downstream_baseline)
    splits = load_paired_dataset(tok, n_train=2, n_val=args.n_sel, n_test=2,
                                 seq_len=128, seed=0, model=args.model)
    sel = splits["val"]
    sel_pmask = prompt_mask_from_markers(128, sel.story_marker_pos)
    sel_dep = sel.tokens[sel.is_deployment].to(device)
    sel_dep_pmask = sel_pmask[sel.is_deployment].to(device)

    base_logp = teacher_forced_sleeper_logp(model, tok, sel_dep).mean().item()

    print(f"baseline_json = {args.baseline_json.name}    "
          f"base sleeper logp = {base_logp:.4f}")
    print()
    print(f"{'seed':>4}  {'winner':>6}  {'top_k_size':>10}  "
          f"{'stage-0 rank':>13}  {'stage-1 rank':>13}  {'Δlogp':>8}")
    print("-" * 80)

    for seed_key, info in base["per_seed"].items():
        seed = int(seed_key.lstrip("s"))
        winner = int(info["winner"])
        top_k = [int(f) for f in info["top_k"]]
        sae_mid, _ = sae_load(args.sae_dir / f"sae_resid_mid_s{seed}.pt",
                              device=device)
        dlogp: dict[int, float] = {}
        for f in top_k:
            best = float("inf")
            for a in args.screen_alphas:
                delta = compute_sae_delta(model, sae_mid, RESID_MID, f,
                                           sel_dep, sel_dep_pmask)
                hooks = additive_steer_hook(delta, a, RESID_MID)
                lp = teacher_forced_sleeper_logp(model, tok, sel_dep,
                                                  fwd_hooks=hooks).mean().item()
                best = min(best, lp - base_logp)
            dlogp[f] = best
        # Sort by Δlogp ascending (most-negative = best suppression first)
        ordered = sorted(top_k, key=lambda f: dlogp[f])
        stage0_rank = top_k.index(winner) + 1
        stage1_rank = ordered.index(winner) + 1
        print(f"{seed:>4}  f{winner:>5}  {len(top_k):>10}  "
              f"{stage0_rank:>13}  {stage1_rank:>13}  {dlogp[winner]:>+8.3f}")


if __name__ == "__main__":
    main()
