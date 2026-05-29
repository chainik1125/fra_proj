"""Probe: can V-space cosine pick a good OV feature?

Two non-cheating selectors compared against the attribution baseline:
  cosine_v       — rank ALL features by cos(W_dec[f]·W_V, v_md·W_V), heads
                   concatenated. The mean-diff cosine screen, but in the
                   per-head value space the OV steer actually acts in.
  attr_cosine_v  — attribution top-`prefilter`, re-ranked by cosine_v.

Per seed it prints where the known low-JSDc "cheating winner" lands under each
ranking, then writes two tuples_json (top-`out_k` each) ready for scripts/eval.py.
No held-out metric is consulted at selection — cosine_v is weight × prompt-activation
geometry only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sleeper.attribution import rank_ov_cosine_v, rank_ov_diff
from sleeper.eval import LN1_HOOK, PAT_HOOK
from sleeper.model import (
    MODELS, cache_activations, load_paired_dataset, load_sleeper_model,
)
from sleeper.sae import encode_all, load as sae_load

# Known per-seed min-JSDc features (OV leftpad, top-5-by-attribution → min JSDc).
CHEAT_WINNER = {0: 1114, 1: 76, 2: 169, 3: 1154, 4: 1006, 5: 689}


def _rank_of(order: list[int], feat: int) -> int | str:
    return order.index(feat) + 1 if feat in order else ">len"


def _tuples_json(sae_dir: str, seeds: list[int], n_sel: int,
                 per_seed_feats: dict[int, list[int]]) -> dict:
    return {
        "channel": "ov", "regime": "diff", "mode": "topk",
        "config": {"sae_dir": sae_dir, "sae_seeds": list(seeds),
                   "n_sel": n_sel, "final_selection": None, "alphas": None},
        "per_seed": {str(s): [[(int(f), "V")] for f in per_seed_feats[s]]
                     for s in seeds},
    }


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(MODELS), default="tinystories")
    p.add_argument("--sae_dir", type=Path, default=Path("weights/seeds_leftpad"))
    p.add_argument("--sae_seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    p.add_argument("--n_sel", type=int, default=200)
    p.add_argument("--prefilter", type=int, default=20,
                   help="attribution candidate pool for attr_cosine_v")
    p.add_argument("--out_k", type=int, default=3,
                   help="features per seed written to each tuples_json")
    p.add_argument("--out_global", type=Path, default=Path("results/ov_cosv_global.json"))
    p.add_argument("--out_attr",   type=Path, default=Path("results/ov_attr_cosv.json"))
    p.add_argument("--out_probe",  type=Path, default=Path("results/ov_cosv_probe.json"))
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    hooked = load_sleeper_model(model=args.model, device=device)
    tok = hooked.tokenizer
    W_V = hooked.W_V[0].detach().to(device)
    W_O = hooked.W_O[0].detach().to(device)

    seq_len = MODELS[args.model].seq_len
    splits = load_paired_dataset(tok, n_train=2, n_val=args.n_sel, n_test=2,
                                 seq_len=seq_len, seed=0, model=args.model)
    sel = splits["val"]
    is_dep = sel.is_deployment.to(device)
    pmask = sel.attention_mask.to(device)

    global_feats: dict[int, list[int]] = {}
    attr_feats: dict[int, list[int]] = {}
    probe: dict[str, dict] = {}

    print(f"{'seed':>4} | {'attr top-5':<28} | {'cosine_v top-5':<28} | "
          f"cheat winner: attr_rank / cosv_rank / attr-in-pf")
    print("-" * 110)

    for seed in args.sae_seeds:
        sae_ln1, _ = sae_load(args.sae_dir / f"sae_ln1_s{seed}.pt", device=device)
        acts = cache_activations(hooked, sel.tokens, [PAT_HOOK, LN1_HOOK])
        A = acts[PAT_HOOK].to(device)
        ln1 = acts[LN1_HOOK].to(device)
        z_ln1 = encode_all(sae_ln1, acts[LN1_HOOK]).to(device)

        attr = rank_ov_diff(A, z_ln1, sae_ln1, W_V, W_O, is_dep, query_mask=pmask)
        cosv = rank_ov_cosine_v(ln1, sae_ln1, W_V, is_dep)

        attr_order = attr["top_indices"].cpu().tolist()
        cosv_order = cosv["top_indices"].cpu().tolist()
        cand = attr_order[: args.prefilter]
        cosv_score = cosv["score"]
        attr_cosv_order = sorted(cand, key=lambda f: float(cosv_score[f]), reverse=True)

        global_feats[seed] = cosv_order[: args.out_k]
        attr_feats[seed] = attr_cosv_order[: args.out_k]

        win = CHEAT_WINNER.get(seed)
        attr_rank = _rank_of(attr_order, win) if win is not None else "-"
        cosv_rank = _rank_of(cosv_order, win) if win is not None else "-"
        in_pf = _rank_of(attr_cosv_order, win) if (win is not None and win in cand) else "not-in-pf"
        print(f"{seed:>4} | {str(attr_order[:5]):<28} | {str(cosv_order[:5]):<28} | "
              f"f{win}: {attr_rank} / {cosv_rank} / {in_pf}")

        probe[str(seed)] = {
            "cheat_winner": win,
            "attr_top20": attr_order[:20],
            "cosv_top20": cosv_order[:20],
            "attr_cosv_order": attr_cosv_order,
            "cosv_score_of_attr_top5": {int(f): round(float(cosv_score[f]), 4)
                                        for f in attr_order[:5]},
            "cosv_score_of_winner": (round(float(cosv_score[win]), 4)
                                     if win is not None else None),
        }

    args.out_global.parent.mkdir(parents=True, exist_ok=True)
    args.out_global.write_text(json.dumps(
        _tuples_json(str(args.sae_dir), args.sae_seeds, args.n_sel, global_feats), indent=2))
    args.out_attr.write_text(json.dumps(
        _tuples_json(str(args.sae_dir), args.sae_seeds, args.n_sel, attr_feats), indent=2))
    args.out_probe.write_text(json.dumps(probe, indent=2))
    print(f"\nwrote {args.out_global} (cosine_v alone, top-{args.out_k}/seed)")
    print(f"wrote {args.out_attr} (attr top-{args.prefilter} → cosine_v, top-{args.out_k}/seed)")
    print(f"wrote {args.out_probe} (full diagnostic)")


if __name__ == "__main__":
    main()
