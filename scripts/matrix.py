"""One matrix-cell evaluation: pick winner per upstream SAE seed via a single
selection method, in a single attribution regime, for one (attr × intervene)
cell. Run multiple times to populate a matrix of results.

Per upstream SAE seed s ∈ --sae_seeds (default 0..5):
  1. Load sae_ln1_s{s}.pt and (paired) sae_resid_mid_s{s}.pt.
  2. Attribute top --top_k feature tuples in --regime ∈ {target, diff} for the
     (--attr, --intervene) cell. Target regime uses the per-seed
     `target_feature` taken from --per_seed_targets_json (mapping s → winner).
  3. Pick ONE winner via --final_selection:
       jsd   → sweep top-K × αs on eval split, ASR=0 → min jsd_clean
               (uses the eval metric for selection — sanity-check method)
       rank  → greedy-ASR sweep top-K × αs on eval split, ASR=0 → min attr-rank
               → min α (no proxy metric, trusts attribution ranking)
  4. Re-evaluate the winner with the canonical 4-metric multi-seed lockstep
     eval (asr, jsd_clean, jsd_pois, exact_match — all paired-seed, batched).

Output: one JSON containing per-seed winners + their 4-metric evals plus the
sweep tables that were used to pick them.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from sleeper.hooks import ACTIVE_CHANNELS
from sleeper.model import (
    left_pad_prompts, load_dep_prompts, load_paired_dataset,
    load_sleeper_model, prompt_mask_from_markers,
)
from sleeper.metrics import batched_asr_16, clean_continuation_ce
from sleeper.sae import load as sae_load

from scripts.matrix_sweep import (
    LN1_HOOK,
    _build_baselines_per_seed, _multi_seed_asr,
    eval_winner, get_tuples, get_tuples_diff,
)


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sae_ln1_dir",      type=Path,  required=True)
    p.add_argument("--sae_mid_dir",      type=Path,  default=None,
                   help="Defaults to --sae_ln1_dir.")
    p.add_argument("--sae_seeds",        type=int,   nargs="+", default=[0, 1, 2, 3, 4, 5])
    p.add_argument("--regime",           choices=["target", "diff"], required=True)
    p.add_argument("--final_selection",  choices=["jsd", "rank"],    required=True)
    p.add_argument("--attr",             choices=["ov", "qk", "ov+qk", "qk+ov"], default="ov")
    p.add_argument("--intervene",        choices=["ov", "qk", "ov+qk", "qk+ov"], default="ov")
    p.add_argument("--top_k",            type=int,   default=20)
    p.add_argument("--triple_k",         type=int,   default=8)
    p.add_argument("--alphas",           type=float, nargs="+", default=[2.0, 4.0],
                   help="Selection-phase α grid (rank: greedy-ASR sweep; jsd: lockstep sweep).")
    p.add_argument("--eval_alphas",      type=float, nargs="+",
                   default=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
                   help="Per-seed winner eval sweep — full 4-metric lockstep eval at each α.")
    p.add_argument("--n_sel",            type=int,   default=100)
    p.add_argument("--n_eval",           type=int,   default=400)
    p.add_argument("--gen_tokens",       type=int,   default=16)
    p.add_argument("--eval_seeds",       type=int,   nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--eval_temperature", type=float, default=1.0)
    p.add_argument("--per_seed_targets_json", type=Path, default=None,
                   help="JSON written by scripts.downstream_baseline mapping "
                        "s{seed} → {'winner': int}. Required for --regime target.")
    p.add_argument("--target_feature",   type=int,   default=579,
                   help="Fallback target_feature when --per_seed_targets_json is not set.")
    p.add_argument("--out",              type=Path,  required=True)
    p.add_argument("--device",           default=None)
    args = p.parse_args()
    # Normalize alias
    if args.attr == "ov+qk":
        args.attr = "qk+ov"
    if args.intervene == "ov+qk":
        args.intervene = "qk+ov"
    cell = f"{args.attr}×{args.intervene}"

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_sleeper_model(device=device)
    tok    = model.tokenizer
    pad_id = tok.pad_token_id or tok.eos_token_id
    W      = {c: getattr(model, f"W_{c}")[0].detach().to(device) for c in ("Q", "K", "V")}
    W_O    = model.W_O[0].detach().to(device)

    splits     = load_paired_dataset(tok, n_train=2, n_val=args.n_sel,
                                     n_test=args.n_eval, seq_len=128, seed=0)
    sel_split  = splits["val"]
    sel_pmask  = prompt_mask_from_markers(128, sel_split.story_marker_pos)

    sel_cln        = sel_split.tokens[~sel_split.is_deployment].to(device)
    sel_cln_marker = sel_split.story_marker_pos[~sel_split.is_deployment].to(device)

    raw_dep        = load_dep_prompts(tok, args.n_sel + args.n_eval, split="test")
    n_sel_dep      = args.n_sel  // 2
    n_eval_dep     = args.n_eval // 2
    eval_dep_lp, eval_dep_attn = left_pad_prompts(
        raw_dep[n_sel_dep : n_sel_dep + n_eval_dep], pad_id,
    )
    eval_dep_lp   = eval_dep_lp.to(device)
    eval_dep_attn = eval_dep_attn.to(device)

    print(f"[mtx] cell={cell}  regime={args.regime}  final_selection={args.final_selection}  "
          f"sae_ln1_dir={args.sae_ln1_dir}  seeds={args.sae_seeds}", flush=True)
    print(f"[mtx] pre-building per-seed baselines (B={eval_dep_lp.shape[0]}, "
          f"seeds={args.eval_seeds}) ...", flush=True)
    eval_clean_lsm_ps, eval_clean_tok_ps, eval_dep_lsm_ps = _build_baselines_per_seed(
        model, eval_dep_lp, eval_dep_attn, args.gen_tokens, device,
        seeds=args.eval_seeds, temperature=args.eval_temperature,
    )
    eval_base_asr_per_seed = _multi_seed_asr(
        model, None, [], 0.0, set(), W,
        eval_dep_lp, eval_dep_attn, args.gen_tokens,
        seeds=args.eval_seeds, temperature=args.eval_temperature, device=device,
    )
    eval_base_asr = sum(eval_base_asr_per_seed) / len(eval_base_asr_per_seed)
    sel_base_ce   = clean_continuation_ce(model, sel_cln, sel_cln_marker).mean().item()
    print(f"[mtx] eval baseline asr={eval_base_asr:.3f}  sel base_ce={sel_base_ce:.4f}",
          flush=True)

    sae_mid_dir = args.sae_mid_dir or args.sae_ln1_dir
    target_feature_per_seed: dict[int, int] = {s: args.target_feature for s in args.sae_seeds}
    if args.per_seed_targets_json is not None:
        m = json.loads(args.per_seed_targets_json.read_text())
        for s in args.sae_seeds:
            key = f"s{s}"
            if key in m:
                target_feature_per_seed[s] = int(m[key]["winner"])
        print(f"[mtx] per-seed target features from "
              f"{args.per_seed_targets_json.name}: {target_feature_per_seed}", flush=True)
    elif args.regime == "target":
        print(f"[mtx] WARNING: --regime target but no --per_seed_targets_json; "
              f"using fixed target_feature={args.target_feature} for every seed. "
              f"The same index is NOT the same feature across SAE seeds.", flush=True)

    active = ACTIVE_CHANNELS[args.intervene]

    out_rows: list[dict] = []
    for seed in args.sae_seeds:
        sae_ln1, _ = sae_load(args.sae_ln1_dir / f"sae_ln1_s{seed}.pt", device=device)
        sae_mid, _ = sae_load(sae_mid_dir / f"sae_resid_mid_s{seed}.pt", device=device)
        # Inject the per-seed target_feature into args (needed by get_tuples).
        args.target_feature = target_feature_per_seed[seed]
        args.attr_for_call = args.attr   # keeps get_tuples happy
        print(f"\n[mtx] ══ seed={seed}  target_feature={args.target_feature}  "
              f"sae_mid=sae_resid_mid_s{seed}.pt ══", flush=True)
        attr_cache: dict = {}

        if args.regime == "target":
            tuples = get_tuples(args.attr, args, model, sae_ln1, sae_mid,
                                sel_split, sel_pmask, device, attr_cache)
        else:
            tuples = get_tuples_diff(args.attr, args, model, sae_ln1, W, W_O,
                                     sel_split, sel_pmask, device, attr_cache)
        top_feats = [int(tup[0][0]) for tup in tuples]
        n_evals = len(tuples) * len(args.alphas)
        print(f"[mtx]   top-{len(top_feats)}: {top_feats[:8]} ...  "
              f"({n_evals} candidate evals if jsd-method)", flush=True)

        if args.final_selection == "jsd":
            # ── Sweep top-K × αs on eval split; pick ASR=0 → min jsd_clean. ──
            sweep_rows: list[dict] = []
            t0 = time.time()
            n_done = 0
            for ti, tup in enumerate(tuples):
                for alpha in args.alphas:
                    tc = time.time()
                    m = eval_winner(
                        model, sae_ln1, tup, alpha, active, W,
                        eval_dep_lp, eval_dep_attn,
                        None, args.gen_tokens, device,
                        eval_seeds=args.eval_seeds,
                        eval_temperature=args.eval_temperature,
                        clean_lsm_per_seed=eval_clean_lsm_ps,
                        clean_tok_per_seed=eval_clean_tok_ps,
                        dep_lsm_per_seed=eval_dep_lsm_ps,
                    )
                    n_done += 1
                    dt = time.time() - tc
                    sweep_rows.append({"attr_rank": ti + 1, "feature": int(tup[0][0]),
                                       "alpha": alpha, **m})
                    print(f"[mtx]    [JSD {n_done:>2}/{n_evals}] rank={ti+1:>2} "
                          f"f={tup[0][0]:>4} α={alpha:>4.1f}  asr={m['asr']:.3f}  "
                          f"jsd_cln={m['jsd_clean']:.3f}  jsd_dep={m['jsd_pois']:.3f}  "
                          f"exact={m['exact_match']:.3f}  ({dt:.1f}s)", flush=True)
            print(f"[mtx]   JSD sweep done in {time.time()-t0:.1f}s", flush=True)
            asr0 = [r for r in sweep_rows if r["asr"] == 0.0]
            pool = asr0 if asr0 else sweep_rows
            winner_row = min(pool, key=lambda r: (r["asr"], r["jsd_clean"]))
            winner_feature = int(winner_row["feature"])
            winner_alpha   = float(winner_row["alpha"])
            winner_metrics = {k: winner_row[k] for k in (
                "asr", "asr_per_seed", "jsd_clean", "jsd_clean_per_seed",
                "jsd_pois", "jsd_pois_per_seed", "exact_match",
                "n_exact_match_clean", "n_exact_match_clean_per_seed",
                "exact_match_total_rows",
            )}
            winner_attr_rank = int(winner_row["attr_rank"])
            extra = {"jsd_sweep_rows": sweep_rows}
        else:
            # ── rank: greedy-ASR sweep top-K × αs, min ASR → tie-break by
            # attribution rank → smallest α. No proxy signal — trusts the
            # attribution ranking and only needs the headline ASR. ──
            t0 = time.time()
            rank_rows: list[dict] = []
            for ti, tup in enumerate(tuples):
                for alpha in args.alphas:
                    asr_g = batched_asr_16(
                        model, sae_ln1, LN1_HOOK, tup, alpha, active,
                        W, 0, eval_dep_lp, eval_dep_attn, args.gen_tokens,
                    )
                    rank_rows.append({"attr_rank": ti + 1,
                                      "feature": int(tup[0][0]),
                                      "alpha": alpha,
                                      "greedy_asr": asr_g})
                    print(f"[mtx]    [RANK rank={ti+1:>2}] f={tup[0][0]:>4} "
                          f"α={alpha:>4.1f}  greedy_asr={asr_g:.3f}", flush=True)
            print(f"[mtx]   rank greedy-ASR sweep done in {time.time()-t0:.1f}s",
                  flush=True)
            asr0 = [r for r in rank_rows if r["greedy_asr"] == 0.0]
            if asr0:
                pick = min(asr0, key=lambda r: (r["attr_rank"], r["alpha"]))
            else:
                pick = min(rank_rows,
                           key=lambda r: (r["greedy_asr"], r["attr_rank"], r["alpha"]))
            winner_feature = int(pick["feature"])
            winner_alpha   = float(pick["alpha"])
            winner_attr_rank = int(pick["attr_rank"])
            # Re-eval the winner with the canonical 4-metric lockstep eval.
            tup_re = [(winner_feature, "V")]
            winner_metrics = eval_winner(
                model, sae_ln1, tup_re, winner_alpha, active, W,
                eval_dep_lp, eval_dep_attn,
                None, args.gen_tokens, device,
                eval_seeds=args.eval_seeds,
                eval_temperature=args.eval_temperature,
                clean_lsm_per_seed=eval_clean_lsm_ps,
                clean_tok_per_seed=eval_clean_tok_ps,
                dep_lsm_per_seed=eval_dep_lsm_ps,
            )
            extra = {"rank_sweep_rows": rank_rows}

        print(f"[mtx]   WINNER f={winner_feature} α={winner_alpha}  "
              f"asr={winner_metrics['asr']:.3f}  "
              f"jsd_cln={winner_metrics['jsd_clean']:.3f}  "
              f"jsd_dep={winner_metrics['jsd_pois']:.3f}  "
              f"exact={winner_metrics['exact_match']:.3f}", flush=True)

        # ── Per-seed winner: full α eval sweep ──
        # The selection-phase α grid is for picking the winner only. For the
        # actual eval (used by the alpha-sweep plots), the winner feature gets
        # re-evaluated with the canonical 4-metric lockstep at each α in
        # --eval_alphas. Independent of selection method, so the resulting
        # sweep is directly comparable across cells.
        winner_tup = [(winner_feature, "V")]
        eval_sweep: dict = {}
        t_es = time.time()
        for ea in args.eval_alphas:
            t_one = time.time()
            ev = eval_winner(
                model, sae_ln1, winner_tup, ea, active, W,
                eval_dep_lp, eval_dep_attn,
                None, args.gen_tokens, device,
                eval_seeds=args.eval_seeds,
                eval_temperature=args.eval_temperature,
                clean_lsm_per_seed=eval_clean_lsm_ps,
                clean_tok_per_seed=eval_clean_tok_ps,
                dep_lsm_per_seed=eval_dep_lsm_ps,
            )
            eval_sweep[str(ea)] = ev
            print(f"[mtx]    [EVAL_SWEEP α={ea:>4.1f}] asr={ev['asr']:.3f}  "
                  f"jsd_cln={ev['jsd_clean']:.3f}  jsd_dep={ev['jsd_pois']:.3f}  "
                  f"exact={ev['exact_match']:.3f}  ({time.time()-t_one:.1f}s)",
                  flush=True)
        print(f"[mtx]   eval sweep done in {time.time()-t_es:.1f}s", flush=True)

        out_rows.append({
            "seed": int(seed),
            "regime": args.regime,
            "final_selection": args.final_selection,
            "cell": cell,
            "top_k_features": top_feats,
            "target_feature_used": target_feature_per_seed[seed],
            "winner": {
                "feature":   winner_feature,
                "alpha":     winner_alpha,
                "attr_rank": winner_attr_rank,
                "metrics":   winner_metrics,
                "eval_sweep": eval_sweep,
            },
            **extra,
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "config":   vars(args) | {"cell": cell},
        "baseline": {"eval": {"asr": eval_base_asr,
                              "asr_per_seed": eval_base_asr_per_seed},
                     "sel":  {"clean_ce": sel_base_ce}},
        "per_seed_target_features": target_feature_per_seed,
        "results":  out_rows,
    }, indent=2, default=str))
    print(f"\n[mtx] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
