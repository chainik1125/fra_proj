"""Additive (fixed-direction) variants of the OV and conventional single-feature
cells — the operation-axis control for the gated repro, swept over the FULL
top-20 attribution ranking (not just the gated winner).

For each cell, the top-20 features by that cell's attribution (rank_ov_diff for
OV, dep−clean diff-of-means for conventional — read from the gated repro JSONs
on HF, where `winner.topK` is the attribution-ordered candidate list; the
ranking is operation-independent) are steered ONE BY ONE with a fixed
−α·f_λ delta (f_λ = unit-norm decoder row) broadcast over prompt positions:

  OV-additive:   delta in ln1 space, projected through W_V at hook_v (A frozen)
  conv-additive: delta added at hook_resid_mid

Same eval split (200 prompts, N_SKIP=50), same prompt-positions-only hook
convention, same metrics as the gated repro.

Usage: python -u -m scripts.repro_additive_cells --out results/repro_additive_cells.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sleeper.hooks import additive_steer_hook, compute_meandiff_delta, ov_only_steer_hook
from sleeper.jsd_cells import (
    LN1_HOOK, RESID_MID, EvalRefs, _gen, _word_match_stats, build_eval_refs, jsd_mean,
)
from sleeper.metrics import asr_16
from sleeper.model import load_sleeper_model
from sleeper.sae import load as sae_load
from scripts.sae_scaling_paths import DEFAULT_HF_REPO, hf_download

OV_CKPT = "sae_checkpoints/ln1/seed0/d3072_k10/step50000.pt"
CONV_CKPT = "sae_checkpoints/resid_mid/seed0/d3072_k32/step50000.pt"
OV_REPRO_RESULT = "repro/repro_ov_ln1_seed0_d3072_k10.json"
CONV_REPRO_RESULT = "repro/repro_conv_residmid_seed0_d3072_k32.json"

# fixed-direction units (unit-norm decoder rows); the gated optima had α·z with
# z up to O(10), so sweep well past that. One negative point as a control.
ALPHAS = [-2.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0]
SUPPRESS = 0.05


@torch.no_grad()
def eval_hooks(model, refs: EvalRefs, hooks, device, tok):
    steered_tokens, steered_lsm = _gen(model, refs.dep_lp, refs.dep_attn, hooks, device)
    n_exact, frac = _word_match_stats(steered_tokens, refs.clean_tokens)
    return dict(
        jsd_clean=jsd_mean(steered_lsm.cpu(), refs.clean_lsm.cpu()),
        jsd_pois=jsd_mean(steered_lsm.cpu(), refs.poisoned_lsm.cpu()),
        n_exact_match_clean=n_exact, frac_pos_match_clean=frac,
        asr=asr_16(steered_tokens.cpu(), tok))


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hf_repo", default=DEFAULT_HF_REPO)
    p.add_argument("--local_dir", type=Path, default=Path("/workspace/sae_scaling_out"))
    p.add_argument("--alphas", type=float, nargs="+", default=ALPHAS)
    p.add_argument("--top_n", type=int, default=20)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_sleeper_model(device=device)
    tok = model.tokenizer
    refs = build_eval_refs(model, device)
    W_V0 = model.W_V[0].detach().to(device)
    # left-padded prompts: every real position is a prompt position (cf. eval_ov)
    prompt_mask = refs.dep_attn.bool()

    n_exact0, frac0 = _word_match_stats(refs.poisoned_tokens, refs.clean_tokens)
    baseline = dict(jsd_clean=jsd_mean(refs.poisoned_lsm.cpu(), refs.clean_lsm.cpu()),
                    jsd_pois=0.0, n_exact_match_clean=n_exact0,
                    frac_pos_match_clean=frac0,
                    asr=asr_16(refs.poisoned_tokens.cpu(), tok))
    out = {"alphas": args.alphas, "baseline": baseline,
           "operation": "additive fixed −α·f (unit-norm W_dec row), prompt positions",
           "cells": {}}

    for cell, ckpt_rel, result_rel in [("ov_additive", OV_CKPT, OV_REPRO_RESULT),
                                       ("conv_additive", CONV_CKPT, CONV_REPRO_RESULT)]:
        topk = json.load(open(hf_download(args.hf_repo, result_rel, args.local_dir)))
        feats = topk["winner"]["topK"][: args.top_n]   # attribution-ordered
        sae, cfg = sae_load(hf_download(args.hf_repo, ckpt_rel, args.local_dir), device)
        per_feat = {}
        for rank, feat in enumerate(feats):
            f_dir = sae.W_dec[int(feat)].detach().to(device).float()
            delta = compute_meandiff_delta(f_dir, prompt_mask, sign=-1.0)
            per_alpha = {}
            for a in args.alphas:
                hooks = (ov_only_steer_hook(delta, a, W_V0, block=0) if cell == "ov_additive"
                         else additive_steer_hook(delta, a, RESID_MID))
                per_alpha[str(a)] = eval_hooks(model, refs, hooks, device, tok)
            sup = [(r["jsd_clean"], float(a)) for a, r in per_alpha.items()
                   if r["asr"] <= SUPPRESS and float(a) > 0]
            best = min(sup) if sup else None
            per_feat[str(feat)] = {"attr_rank": rank, "per_alpha": per_alpha,
                                   "opt_jclean": best[0] if best else None,
                                   "opt_alpha": best[1] if best else None}
            print(f"[{cell}] rank={rank:<2} f={feat:<5} "
                  + (f"opt_J={best[0]:.4f} @ α={best[1]}" if best else "never suppresses"),
                  flush=True)
        out["cells"][cell] = {"ckpt": ckpt_rel, "features": feats, "per_feature": per_feat}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
