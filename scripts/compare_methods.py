"""Per-seed comparison of 4 feature-selection winners.

For each upstream ln1 SAE seed, for each regime (target, diff):
  1. OV→OV attribution → top-K candidate features.
  2. "JSD" method     — sweep top-K × αs on eval split, take ASR=0 then min jsd_clean.
  3. "FBF" method     — Δlogp screen on selection split → top stage2_keep candidates
                       → batched ASR (eval split) + ΔCE (selection split) →
                       winner = ASR=0 then min ΔCE (matches scripts/find_best_feature.py).
  4. Re-eval each of the 2 winners with the canonical 4-metric eval_winner:
       asr (multi-seed mean), jsd_clean, jsd_pois, exact-token-match.

Output: one JSON listing per (seed, regime) the top-K attribution feature list,
both winners' (feature, α), and the 4 eval metrics for each.

Intended usage: run twice — once with --sae_ln1_dir weights/seeds       (4k),
once with --sae_ln1_dir weights/seeds_50k --sae_mid weights/sae_resid_mid_50k.pt (50k).
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
from sleeper.metrics import clean_continuation_ce
from sleeper.sae import load as sae_load

from scripts.matrix_sweep import (
    LN1_HOOK,
    _build_baselines_per_seed, _multi_seed_asr,
    eval_downstream_baseline, eval_winner, get_tuples, get_tuples_diff,
)
from scripts.find_best_feature import _asr_and_dce, _screen


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds",            type=int,   nargs="+", default=[0, 1, 2, 3, 4, 5])
    p.add_argument("--sae_ln1_dir",      type=Path,  default=Path("weights/seeds"))
    p.add_argument("--sae_mid",          type=Path,  default=Path("weights/sae_resid_mid.pt"))
    p.add_argument("--target_feature",   type=int,   default=579)
    p.add_argument("--top_k",            type=int,   default=20)
    p.add_argument("--stage2_keep",      type=int,   default=10)
    p.add_argument("--triple_k",         type=int,   default=8)
    p.add_argument("--alphas",           type=float, nargs="+", default=[2.0, 4.0])
    p.add_argument("--n_sel",            type=int,   default=200)
    p.add_argument("--n_eval",           type=int,   default=200)
    p.add_argument("--gen_tokens",       type=int,   default=16)
    p.add_argument("--eval_seeds",       type=int,   nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--eval_temperature", type=float, default=1.0)
    p.add_argument("--regimes",          nargs="+",  default=["target", "diff"],
                   choices=["target", "diff"])
    p.add_argument("--out",              type=Path,  required=True)
    p.add_argument("--device",           default=None)
    args = p.parse_args()
    args.attr = "ov"  # required by get_tuples / get_tuples_diff helpers

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

    sel_dep        = sel_split.tokens[sel_split.is_deployment].to(device)
    sel_dep_pmask  = sel_pmask[sel_split.is_deployment].to(device)
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

    # Per-seed unsteered baselines for lockstep matched-seed eval.
    # For each eval_seed s: clean rollout (lsm + tokens) and dep rollout (lsm)
    # are both sampled at seed s. eval_winner samples the steered rollout at
    # the same s and computes jsd_clean/jsd_pois/exact-match per s, then means.
    print(f"[cmp] pre-building per-seed baselines (B={eval_dep_lp.shape[0]}, "
          f"seeds={args.eval_seeds}) ...")
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
    print(f"[cmp] eval baseline asr={eval_base_asr:.3f}  sel base_ce={sel_base_ce:.4f}",
          flush=True)

    # Load the downstream resid_mid SAE unconditionally: needed by target-regime
    # attribution AND by the downstream-baseline conventional-steering eval.
    sae_mid, _ = sae_load(args.sae_mid, device=device)

    # ── Downstream baseline (conventional steering): ablate a single resid_mid
    # SAE feature via additive_steer_hook at blocks.0.hook_resid_mid. Seed-
    # independent (the resid_mid SAE is the shared downstream SAE for this
    # pipeline run; sampling seeds are eval_seeds, same as the upstream evals).
    # Uses the SAME pre-built per-seed baselines and the SAME 5-seed lockstep
    # eval as eval_winner, so the 4 metrics are directly comparable.
    print(f"[cmp] downstream baseline: f{args.target_feature} at blocks.0.hook_resid_mid "
          f"(SAE: {args.sae_mid.name})", flush=True)
    downstream_per_alpha: dict = {}
    for alpha in args.alphas:
        t_d0 = time.time()
        m = eval_downstream_baseline(
            model, sae_mid, args.target_feature, alpha,
            eval_dep_lp, eval_dep_attn,
            eval_clean_lsm_ps, eval_clean_tok_ps, eval_dep_lsm_ps,
            args.gen_tokens, device,
            eval_seeds=args.eval_seeds, eval_temperature=args.eval_temperature,
        )
        downstream_per_alpha[str(alpha)] = m
        print(f"[cmp]   [DOWN] f{args.target_feature} α={alpha:>4.1f}  "
              f"asr={m['asr']:.3f}  jsd_cln={m['jsd_clean']:.3f}  "
              f"jsd_dep={m['jsd_pois']:.3f}  exact={m['exact_match']:.3f}  "
              f"({time.time()-t_d0:.1f}s)", flush=True)

    active = ACTIVE_CHANNELS["ov"]
    W_V    = W["V"]

    out_rows: list[dict] = []
    for seed in args.seeds:
        sae_ln1, _ = sae_load(args.sae_ln1_dir / f"sae_ln1_s{seed}.pt", device=device)
        print(f"\n[cmp] ══ seed={seed}  sae_ln1_dir={args.sae_ln1_dir} ══", flush=True)
        attr_cache: dict = {}

        for regime in args.regimes:
            if regime == "target":
                tuples = get_tuples("ov", args, model, sae_ln1, sae_mid,
                                    sel_split, sel_pmask, device, attr_cache)
            else:
                tuples = get_tuples_diff("ov", args, model, sae_ln1, W, W_O,
                                         sel_split, sel_pmask, device, attr_cache)
            top_feats = [int(tup[0][0]) for tup in tuples]
            n_evals = len(tuples) * len(args.alphas)
            print(f"[cmp]   regime={regime}  top-{len(top_feats)}: {top_feats[:8]} ...  "
                  f"({n_evals} JSD evals to run)", flush=True)

            # ── Method JSD: sweep top-K × αs on eval split, ASR=0 then min jsd_clean ──
            jsd_rows: list[dict] = []
            t_jsd0 = time.time()
            n_done = 0
            for ti, tup in enumerate(tuples):
                for alpha in args.alphas:
                    t_call0 = time.time()
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
                    dt = time.time() - t_call0
                    n_done += 1
                    jsd_rows.append({"attr_rank": ti + 1, "feature": int(tup[0][0]),
                                     "alpha": alpha, **m})
                    print(f"[cmp]    [JSD {n_done:>2}/{n_evals}] rank={ti+1:>2} "
                          f"f={tup[0][0]:>4} α={alpha:>4.1f}  "
                          f"asr={m['asr']:.3f}  jsd_cln={m['jsd_clean']:.3f}  "
                          f"jsd_dep={m['jsd_pois']:.3f}  "
                          f"exact={m['exact_match']:.3f}  ({dt:.1f}s)", flush=True)
            print(f"[cmp]   regime={regime}  JSD sweep done in "
                  f"{time.time()-t_jsd0:.1f}s", flush=True)
            asr0 = [r for r in jsd_rows if r["asr"] == 0.0]
            pool = asr0 if asr0 else jsd_rows
            jsd_winner = min(pool, key=lambda r: (r["asr"], r["jsd_clean"]))

            # ── Method FBF: Δlogp screen → ASR + ΔCE → ASR=0 then min ΔCE ──
            t_fbf0 = time.time()
            screen_rows, base_logp = _screen(
                model, sae_ln1, LN1_HOOK, W_V, top_feats, args.alphas,
                sel_dep, sel_dep_pmask, device,
            )
            screen_rows.sort(key=lambda r: r["dlogp"])
            stage2_cands = [(r["f"], r["alpha"]) for r in screen_rows[:args.stage2_keep]]
            print(f"[cmp]   regime={regime}  Δlogp screen done in "
                  f"{time.time()-t_fbf0:.1f}s; stage-2 candidates: "
                  f"{[(f, a) for f, a in stage2_cands[:5]]} ...", flush=True)
            t_stage2 = time.time()
            base_asr_e, base_ce_s, fbf_rows = _asr_and_dce(
                model, sae_ln1, LN1_HOOK, W_V, stage2_cands,
                eval_dep_lp, eval_dep_attn, eval_dep_attn,
                sel_cln, sel_cln_marker, args.gen_tokens, device,
            )
            print(f"[cmp]   regime={regime}  FBF stage-2 ASR/ΔCE done in "
                  f"{time.time()-t_stage2:.1f}s", flush=True)
            fbf_asr0  = [r for r in fbf_rows if r["asr"] == 0.0]
            fbf_pick  = (min(fbf_asr0, key=lambda r: r["dce"]) if fbf_asr0
                         else min(fbf_rows, key=lambda r: r["asr"]))
            fbf_tuple = [(fbf_pick["f"], "V")]
            fbf_eval  = eval_winner(
                model, sae_ln1, fbf_tuple, fbf_pick["alpha"], active, W,
                eval_dep_lp, eval_dep_attn,
                None, args.gen_tokens, device,
                eval_seeds=args.eval_seeds,
                eval_temperature=args.eval_temperature,
                clean_lsm_per_seed=eval_clean_lsm_ps,
                clean_tok_per_seed=eval_clean_tok_ps,
                dep_lsm_per_seed=eval_dep_lsm_ps,
            )
            # attribution rank of the FBF winner feature within top-K
            try:
                fbf_attr_rank = top_feats.index(int(fbf_pick["f"])) + 1
            except ValueError:
                fbf_attr_rank = -1

            print(f"[cmp]   regime={regime}  JSD winner: f={jsd_winner['feature']} "
                  f"α={jsd_winner['alpha']}  asr={jsd_winner['asr']:.3f}  "
                  f"jsd_cln={jsd_winner['jsd_clean']:.3f}", flush=True)
            print(f"[cmp]   regime={regime}  FBF winner: f={fbf_pick['f']} "
                  f"α={fbf_pick['alpha']}  asr={fbf_eval['asr']:.3f}  "
                  f"jsd_cln={fbf_eval['jsd_clean']:.3f}  ΔCE={fbf_pick['dce']:+.4f}", flush=True)

            out_rows.append({
                "seed": int(seed),
                "regime": regime,
                "top_k_features": top_feats,
                "jsd_method": {
                    "feature": int(jsd_winner["feature"]),
                    "alpha":   float(jsd_winner["alpha"]),
                    "attr_rank": int(jsd_winner["attr_rank"]),
                    "metrics": {k: jsd_winner[k] for k in (
                        "asr", "asr_per_seed", "jsd_clean", "jsd_pois",
                        "exact_match", "n_exact_match_clean",
                    )},
                },
                "fbf_method": {
                    "feature":   int(fbf_pick["f"]),
                    "alpha":     float(fbf_pick["alpha"]),
                    "attr_rank": fbf_attr_rank,
                    "selection": {"sel_asr_or_eval_asr": float(fbf_pick["asr"]),
                                   "sel_delta_ce":        float(fbf_pick["dce"])},
                    "metrics":   fbf_eval,
                },
                "jsd_sweep_rows":   jsd_rows,
                "fbf_screen_rows":  screen_rows,
                "fbf_stage2_rows":  fbf_rows,
            })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "config": vars(args) | {"attr": "ov", "intervene": "ov"},
        "baseline": {"eval": {"asr": eval_base_asr,
                              "asr_per_seed": eval_base_asr_per_seed},
                     "sel":  {"clean_ce": sel_base_ce}},
        "downstream_baseline": {
            "target_feature": args.target_feature,
            "layer_hook":     "blocks.0.hook_resid_mid",
            "sae_path":       str(args.sae_mid),
            "per_alpha":      downstream_per_alpha,
        },
        "results": out_rows,
    }, indent=2, default=str))
    print(f"\n[cmp] wrote {args.out}")


if __name__ == "__main__":
    main()
