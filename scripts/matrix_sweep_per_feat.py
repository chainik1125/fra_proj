"""Per-feature held-out eval for one (attr × intervene) cell.

Same attribution + tuple selection as matrix_sweep.py, but instead of picking
ONE winner per seed via selection-split ASR, we evaluate EVERY top-K tuple
(at every alpha) directly on the held-out eval split and record its ASR +
JSD(steered, clean). Output makes it easy to ask: "is there a non-top-1 single
feature with ASR≈0 and lower JSD than the top-1 winner?"

Reuses helpers from scripts.matrix_sweep so attribution / eval semantics are
identical (LN1_HOOK, JSD_CLEAN_SEED, clean rollout seed, etc.).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sleeper.hooks import ACTIVE_CHANNELS
from sleeper.model import (
    left_pad_prompts, load_dep_prompts, load_paired_dataset,
    load_sleeper_model, prompt_mask_from_markers,
)
from sleeper.sae import load as sae_load

from scripts.matrix_sweep import (
    JSD_CLEAN_SEED, _build_clean_lsm, _multi_seed_asr,
    eval_winner, get_tuples, get_tuples_diff,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--regime",          choices=["target", "diff"], default="diff")
    p.add_argument("--seeds",          type=int,   nargs="+", default=[0, 1, 2, 3, 4, 5])
    p.add_argument("--sae_mid",        type=Path,  default=Path("weights/sae_resid_mid.pt"),
                   help="Downstream resid_mid SAE (only used with --regime target).")
    p.add_argument("--target_feature", type=int,   default=579,
                   help="Target feature in sae_mid (only used with --regime target).")
    p.add_argument("--top_k",          type=int,   default=20)
    p.add_argument("--triple_k",       type=int,   default=8)
    p.add_argument("--alphas",         type=float, nargs="+", default=[2.0, 4.0])
    p.add_argument("--n_sel",          type=int,   default=200,
                   help="Used only for attribution / clean-CE baseline.")
    p.add_argument("--n_eval",         type=int,   default=200)
    p.add_argument("--gen_tokens",     type=int,   default=16)
    p.add_argument("--eval_seeds",     type=int,   nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--eval_temperature", type=float, default=1.0)
    p.add_argument("--attr",      choices=["ov", "qk", "qk+ov"], default="ov")
    p.add_argument("--intervene", choices=["ov", "qk", "qk+ov"], default="ov")
    p.add_argument("--out",            type=Path,  default=Path("results/matrix_per_feat.json"))
    p.add_argument("--device",         default=None)
    args = p.parse_args()

    device  = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model   = load_sleeper_model(device=device)
    tok     = model.tokenizer
    pad_id  = tok.pad_token_id or tok.eos_token_id
    W       = {c: getattr(model, f"W_{c}")[0].detach().to(device) for c in ("Q", "K", "V")}
    W_O     = model.W_O[0].detach().to(device)

    sae_mid = None
    if args.regime == "target":
        sae_mid, _ = sae_load(args.sae_mid, device=device)

    splits     = load_paired_dataset(tok, n_train=2, n_val=args.n_sel,
                                     n_test=args.n_eval, seq_len=128, seed=0)
    sel_split  = splits["val"]
    sel_pmask  = prompt_mask_from_markers(128, sel_split.story_marker_pos)

    raw_dep        = load_dep_prompts(tok, args.n_sel + args.n_eval, split="test")
    n_sel_dep      = args.n_sel  // 2
    n_eval_dep     = args.n_eval // 2
    eval_dep_lp, eval_dep_attn = left_pad_prompts(
        raw_dep[n_sel_dep : n_sel_dep + n_eval_dep], pad_id,
    )
    eval_dep_lp  = eval_dep_lp.to(device)
    eval_dep_attn = eval_dep_attn.to(device)

    eval_base_asr_per_seed = _multi_seed_asr(
        model, None, [], 0.0, set(), W,
        eval_dep_lp, eval_dep_attn, args.gen_tokens,
        seeds=args.eval_seeds, temperature=args.eval_temperature, device=device,
    )
    eval_base_asr = sum(eval_base_asr_per_seed) / len(eval_base_asr_per_seed)
    print(f"[pf] regime={args.regime}  cell: {args.attr}×{args.intervene}  top_k={args.top_k}")
    print(f"[pf] eval baseline asr={eval_base_asr:.3f} "
          f"(seeds={args.eval_seeds}, T={args.eval_temperature})")

    print(f"[pf] pre-building clean reference lsm (B={eval_dep_lp.shape[0]})...")
    eval_clean_lsm = _build_clean_lsm(
        model, eval_dep_lp, eval_dep_attn, args.gen_tokens, device,
    )

    active = ACTIVE_CHANNELS[args.intervene]
    all_rows: list[dict] = []

    for seed in args.seeds:
        sae_ln1, _ = sae_load(Path(f"weights/seeds/sae_ln1_s{seed}.pt"), device=device)
        print(f"\n[pf] ══ seed={seed} ══")
        attr_cache: dict = {}

        if args.regime == "target":
            tuples = get_tuples(args.attr, args, model, sae_ln1, sae_mid,
                                sel_split, sel_pmask, device, attr_cache)
        else:
            tuples = get_tuples_diff(args.attr, args, model, sae_ln1, W, W_O,
                                     sel_split, sel_pmask, device, attr_cache)
        print(f"[pf]   attr={args.attr} ({args.regime}): {len(tuples)} tuples")

        for ti, tup in enumerate(tuples):
            for alpha in args.alphas:
                m = eval_winner(
                    model, sae_ln1, tup, alpha, active, W,
                    eval_dep_lp, eval_dep_attn,
                    eval_clean_lsm, args.gen_tokens, device,
                    eval_seeds=args.eval_seeds,
                    eval_temperature=args.eval_temperature,
                )
                row = {
                    "seed": seed, "attr_rank": ti + 1,
                    "tuple": [list(t) for t in tup],
                    "alpha": alpha,
                    "eval_asr": m["asr"],
                    "eval_asr_per_seed": m["asr_per_seed"],
                    "eval_jsd_clean": m["jsd_clean"],
                }
                all_rows.append(row)
                print(f"[pf]   rank={ti+1:>2}  feat={tup[0][0]:>5}  α={alpha:>4.1f}  "
                      f"asr={m['asr']:.3f}  jsd={m['jsd_clean']:.4f}")

    # ── summary: best per seed under "ASR=0 then min JSD" criterion ──
    # If no row achieves ASR=0, fall back to (min ASR, then min JSD).
    print("\n" + "=" * 90)
    print(f"{'seed':>4}  {'top1 (rank=1)':>32}  {'best (any rank)':>32}")
    print(f"{'':>4}  {'feat α ASR JSD':>32}  {'feat α ASR JSD  rank':>32}")
    print("-" * 90)
    best_per_seed = []
    for seed in args.seeds:
        rows_s = [r for r in all_rows if r["seed"] == seed]
        # top-1 at any alpha — pick min JSD among rank-1 rows that also have min ASR
        top1 = [r for r in rows_s if r["attr_rank"] == 1]
        top1_best = sorted(top1, key=lambda r: (r["eval_asr"], r["eval_jsd_clean"]))[0]
        # global best across all ranks
        zero = [r for r in rows_s if r["eval_asr"] == 0.0]
        pool = zero if zero else rows_s
        best  = sorted(pool, key=lambda r: (r["eval_asr"], r["eval_jsd_clean"]))[0]
        best_per_seed.append({"seed": seed, "top1": top1_best, "best": best})

        def fmt(r):
            return (f"f={r['tuple'][0][0]:<5} α={r['alpha']:.1f} "
                    f"asr={r['eval_asr']:.3f} jsd={r['eval_jsd_clean']:.4f}")
        marker = "  ← improves" if best["attr_rank"] != 1 and (
            best["eval_jsd_clean"] < top1_best["eval_jsd_clean"]
            and best["eval_asr"] <= top1_best["eval_asr"]
        ) else ""
        print(f"{seed:>4}  {fmt(top1_best):>32}  "
              f"{fmt(best)+f' r={best['attr_rank']}':>32}{marker}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "config": vars(args) | {"cell": f"{args.attr}×{args.intervene}",
                                 "regime": args.regime},
        "decoding": {
            "eval_asr":       {"mode": "sample", "temperature": args.eval_temperature,
                                "top_p": None, "top_k": None, "seeds": args.eval_seeds},
            "eval_jsd_clean": {"mode": "sample", "temperature": args.eval_temperature,
                                "top_p": None, "top_k": None,
                                "seed": JSD_CLEAN_SEED,
                                "note": "single rollout at seed=JSD_CLEAN_SEED, same as clean_lsm"},
        },
        "baseline": {"eval": {"asr": eval_base_asr,
                               "asr_per_seed": eval_base_asr_per_seed}},
        "rows":         all_rows,
        "best_per_seed": [
            {"seed": x["seed"],
             "top1": x["top1"],
             "best": x["best"]}
            for x in best_per_seed
        ],
    }, indent=2, default=str))
    print(f"\n[pf] wrote {args.out}")


if __name__ == "__main__":
    main()
