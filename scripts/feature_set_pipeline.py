"""Feature-set pipeline — selects top-K ln1 SAE features via either Jamie's or
Ketan's selection method, then evaluates them via single-feature α-sweep,
full-set α-sweep, or both. OV attribution + OV-only V-channel intervention
throughout (the ov×ov cell of the attribution × intervention matrix).

Selection methods (`--selection_method`):
  * jamie — head-summed, prompt-masked, rank features by `|score|` where
    `score_λ = Σ_h (mean_{b∈dep, q∈pmask} contrib − mean_{b∈clean, q∈pmask} contrib)`.
    Returns top-K feature indices. Heads broadcast V on all 16 downstream.
  * ketan — head-resolved, no prompt mask. Rank (h, λ) pairs by
    `|mean_{b∈dep} contrib − mean_{b∈clean} contrib|`, dedupe to unique features
    in pair order, take first K. Returns top-K features + (head, feature)
    provenance. Mirrors Ketan's `dep_vs_clean_contribution` ranking with
    `unique_features` deduplication on
    `ketan-ov-1000-prompts:tracing_feature/scripts/ov_path.py`.

Evaluation modes (`--eval_mode`):
  * single — three-stage screen+eval on the top-K features:
      stage 0: cheap analytic Δdep-logp screen on every top-K feature ×
               `--screen_alphas` (teacher-forced, one model forward each).
               Cull to the top half (best K/2 by min-most-negative dlogp
               across screen_alphas), favouring features that actively
               *suppress* the sleeper-phrase log-prob.
      stage 1: cheap greedy ASR screen on the K/2 survivors × screen_alphas.
               Pick the feature with the lowest min-ASR-across-screen-alphas
               (tie-break: the lower screen α that achieves that min, so we
               favour features that suppress at weaker steering).
      stage 2: full multi-metric eval (ASR, Δlogp, Δcln-CE, gen-CE ratio,
               severity ratio) on that one survivor at the fine `--alphas`
               grid with multi-seed sampling.
    Stages 0 and 1 run on the **selection split** (held disjoint from the
    eval split that stage 2 reports on). Persists every dlogp/ASR screen
    point + the survivor list under
    `selection.per_seed[seed].single_screen` in the output JSON.
  * set — α-sweep on the full top-`top_k` set steered together (sum of
    per-feature OV deltas, V hook on all 16 heads). Matches Ketan's
    "all_head_features" intervention shape via `resolve_channel_deltas`.
  * both — runs both. Lets you compare per-feature curves to the full-set curve
    on a single matched eval split.

Multi-seed (`--sae_seeds`):
  Loops the upstream selection + eval across the listed ln1 SAE seeds. Each
  seed gets its own ranked feature list. Output points are tagged with
  `sae_seed` for downstream plotting.

Downstream baseline (`--include_downstream`, default on):
  Adds an α-sweep on the resid_mid suppressor feature `--target_feature`
  (default 579) ablated directly at `blocks.0.hook_resid_mid` via
  `additive_steer_hook`. Seed-independent — one curve total. Points tagged
  with `family="downstream"`.

The same eval engine drives every mode — `_run_eval` / `_upstream_eval` /
`_downstream_eval` from `scripts.single_feature_alpha_sweep`. Outputs the
standard 5-metric panel (ASR, Δdep-logp, Δcln-CE, gen-CE ratio, severity ratio).

Usage:
    python -m scripts.feature_set_pipeline \\
        --selection_method jamie --top_k 20 --eval_mode both \\
        --sae_seeds 0 1 2 3 4 \\
        --screen_alphas 2 4 \\
        --alphas 0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 \\
        --out results/experiment_jamie.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from scripts.single_feature_alpha_sweep import (
    _downstream_eval, _eval_baseline_asr, _upstream_eval,
)
from sleeper.attribution import (
    compute_ov_weights, ov_attribution, select_features,
)
from sleeper.hooks import (
    ACTIVE_CHANNELS, build_hooks, greedy_generate_with_hooks,
    resolve_channel_deltas,
)
from sleeper.metrics import (
    asr_16, clean_continuation_ce, pregen_clean_rollouts,
    teacher_forced_sleeper_logp,
)
from sleeper.model import (
    cache_activations, left_pad_prompts, load_dep_prompts,
    load_paired_dataset, load_sleeper_model, prompt_mask_from_markers,
)
from sleeper.sae import encode_all, load as sae_load

LN1_HOOK = "blocks.0.ln1.hook_normalized"
PAT_HOOK = "blocks.0.attn.hook_pattern"


@torch.no_grad()
def _asr_screen(
    model, sae_ln1, feature, alpha, eval_lp, eval_attn, W, gen_tokens, block, device,
) -> float:
    """Greedy ASR for a single feature × α (one batched generation, no eval-seed
    loop). Used by the screen+stage2 single-feature flow to cheaply rank top-K
    features by their best achievable sleeper suppression before committing to
    the full multi-metric eval on a single survivor."""
    sel = [(int(feature), "V")]
    active = ACTIVE_CHANNELS["ov"]
    cd = resolve_channel_deltas(sel, active, model, sae_ln1, LN1_HOOK,
                                eval_lp, eval_attn, eval_attn)
    hooks = build_hooks(cd, alpha, active, W, LN1_HOOK, block)
    gen = greedy_generate_with_hooks(model, eval_lp, hooks, gen_tokens,
                                      attention_mask=eval_attn)
    return asr_16(gen, model.tokenizer)


def _select_top_features(
    args, model, sae_ln1, sae_mid, sel_split, sel_pmask, device,
) -> dict:
    """Cache attention pattern + ln1 codes, build OV weights against the
    target resid_mid direction, run ov_attribution, return top-K features
    via the configured selection method."""
    caches = cache_activations(model, sel_split.tokens, [PAT_HOOK, LN1_HOOK])
    A = caches[PAT_HOOK].to(device)
    z_ln1 = encode_all(sae_ln1, caches[LN1_HOOK]).to(device)
    d = sae_mid.W_enc[:, args.target_feature].detach().to(device).float()
    ovw = compute_ov_weights(model, sae_ln1, d, block=args.block)
    out = ov_attribution(A, z_ln1, ovw["beta"])

    qmask = sel_pmask.to(device) if args.selection_method == "jamie" else None
    sel = select_features(out["contrib"], sel_split.is_deployment.to(device),
                          top_k=args.top_k, method=args.selection_method,
                          query_mask=qmask)
    return sel


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--selection_method", choices=["jamie", "ketan"], default="jamie",
                   help="Feature-selection algorithm. See module docstring.")
    p.add_argument("--top_k", type=int, default=20,
                   help="Size of the selected feature set (used in eval_mode=set).")
    p.add_argument("--eval_mode", choices=["single", "set", "both"], default="both",
                   help="Run α-sweep on the screen-best single feature ('single'), "
                        "the full feature set ('set'), or both. In 'single' mode "
                        "the pipeline first does a cheap greedy-ASR screen across "
                        "all top_k features × --screen_alphas, picks the feature "
                        "with the lowest min-ASR-across-screen-alphas, then runs "
                        "the full multi-metric eval on that one feature at --alphas.")
    p.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0, 4.0],
                   help="Fine α grid for stage-2 multi-metric eval (single + set).")
    p.add_argument("--screen_alphas", type=float, nargs="+", default=[2.0, 4.0],
                   help="Coarse α grid for the cheap greedy-ASR feature screen "
                        "(stage 1 of single-mode eval).")
    p.add_argument("--sae_seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                   help="ln1 SAE seeds to loop over.")
    p.add_argument("--target_feature", type=int, default=579,
                   help="Downstream resid_mid SAE feature whose encoder column "
                        "drives the OV target direction `e`. Also the feature "
                        "used by the downstream baseline curve.")
    p.add_argument("--include_downstream", action=argparse.BooleanOptionalAction, default=True,
                   help="Include the downstream baseline (resid_mid SAE feature "
                        "ablated directly at hook_resid_mid). Seed-independent.")
    p.add_argument("--block", type=int, default=0)
    p.add_argument("--n_sel", type=int, default=100,
                   help="Selection-split size (used for attribution).")
    p.add_argument("--n_eval", type=int, default=200,
                   help="Held-out eval-split size.")
    p.add_argument("--n_gen_ce", type=int, default=100,
                   help="Eval subset for the gen-CE ratio + severity ratio metrics.")
    p.add_argument("--gen_tokens", type=int, default=16)
    p.add_argument("--eval_seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--eval_temperature", type=float, default=1.0)
    p.add_argument("--sae_mid", type=Path, default=Path("weights/sae_resid_mid.pt"))
    p.add_argument("--out", type=Path, default=Path("results/feature_set_pipeline.json"))
    p.add_argument("--device", default=None)
    p.add_argument("--use_past_kv_cache", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_sleeper_model(device=device)
    tok    = model.tokenizer
    pad_id = tok.pad_token_id or tok.eos_token_id
    W      = {c: getattr(model, f"W_{c}")[args.block].detach().to(device) for c in ("Q", "K", "V")}
    sae_mid, _ = sae_load(args.sae_mid, device=device)

    splits = load_paired_dataset(tok, n_train=2, n_val=args.n_sel,
                                 n_test=args.n_eval, seq_len=128, seed=0)
    sel_split = splits["val"]
    ev        = splits["test"]
    sel_pmask = prompt_mask_from_markers(128, sel_split.story_marker_pos)

    # Selection split — drives attribution AND the cheap stage-0/1 screens.
    sel_dep         = sel_split.tokens[sel_split.is_deployment].to(device)
    sel_dep_pmask   = sel_pmask[sel_split.is_deployment].to(device)
    sel_base_logp   = teacher_forced_sleeper_logp(model, tok, sel_dep).mean().item()

    eval_pmask = prompt_mask_from_markers(128, ev.story_marker_pos)
    eval_dep         = ev.tokens[ev.is_deployment].to(device)
    eval_dep_pmask   = eval_pmask[ev.is_deployment].to(device)
    eval_cln         = ev.tokens[~ev.is_deployment].to(device)
    eval_cln_marker  = ev.story_marker_pos[~ev.is_deployment].to(device)

    raw_dep = load_dep_prompts(tok, args.n_sel // 2 + args.n_eval // 2, split="test")
    sel_lp, sel_attn = left_pad_prompts(raw_dep[: args.n_sel // 2], pad_id)
    sel_lp, sel_attn = sel_lp.to(device), sel_attn.to(device)
    eval_lp, eval_attn = left_pad_prompts(
        raw_dep[args.n_sel // 2 : args.n_sel // 2 + args.n_eval // 2], pad_id,
    )
    eval_lp, eval_attn = eval_lp.to(device), eval_attn.to(device)
    eval_gen_dep  = eval_lp[: args.n_gen_ce]
    eval_gen_attn = eval_attn[: args.n_gen_ce]

    # ── Eval setup (shared across all sae_seeds + downstream) ────────────
    base_logp = teacher_forced_sleeper_logp(model, tok, eval_dep).mean().item()
    base_ce   = clean_continuation_ce(model, eval_cln, eval_cln_marker).mean().item()
    base_asr, base_asr_per_seed = _eval_baseline_asr(
        model, eval_lp, eval_attn, args.gen_tokens, args.eval_seeds, args.eval_temperature, device,
        use_past_kv_cache=args.use_past_kv_cache,
    )
    print(f"[fset] eval baseline: asr={base_asr:.3f}  Δlogp_base={base_logp:.3f}  "
          f"cln_CE_base={base_ce:.4f}")

    print(f"[fset] pre-generating clean rollouts (B={args.n_gen_ce}, "
          f"S={len(args.eval_seeds)})...")
    clean_rollouts = pregen_clean_rollouts(
        model, eval_gen_dep, eval_gen_attn, args.gen_tokens,
        args.eval_seeds, args.eval_temperature, device,
        use_past_kv_cache=args.use_past_kv_cache,
    )

    # ── Per-seed selection + upstream eval ───────────────────────────────
    points: list[dict] = []
    selections_by_seed: dict[int, dict] = {}
    eval_modes = ("single", "set") if args.eval_mode == "both" else (args.eval_mode,)

    for sae_seed in args.sae_seeds:
        print(f"\n[fset] ══ sae_seed={sae_seed} ══")
        sae_ln1, _ = sae_load(Path(f"weights/seeds/sae_ln1_s{sae_seed}.pt"), device=device)

        sel = _select_top_features(args, model, sae_ln1, sae_mid, sel_split,
                                    sel_pmask, device)
        print(f"[fset] features ({args.selection_method}, top-{args.top_k}): "
              f"{sel['features'][:10]}{'…' if len(sel['features']) > 10 else ''}")
        if sel["provenance"] is not None:
            head_counts: dict[int, int] = {}
            for h, _ in sel["provenance"]:
                head_counts[h] = head_counts.get(h, 0) + 1
            print(f"[fset] head provenance: {dict(sorted(head_counts.items()))}")
        selections_by_seed[sae_seed] = {
            "features":   sel["features"],
            "provenance": sel["provenance"],
        }

        if "single" in eval_modes:
            # ── Stage 0: cheap Δdep-logp screen on top-K × screen_alphas. ──
            # Teacher-forced sleeper-phrase log-prob; one forward per (f, α).
            # Run on the SELECTION split so the eval split stays held out.
            dlogp_results: list[dict] = []
            best_min_dlogp_per_feat: dict[int, float] = {}
            for f in sel["features"]:
                for screen_alpha in args.screen_alphas:
                    sel_one = [(int(f), "V")]
                    cd = resolve_channel_deltas(sel_one, ACTIVE_CHANNELS["ov"],
                                                model, sae_ln1, LN1_HOOK,
                                                sel_dep, sel_dep_pmask)
                    hooks = build_hooks(cd, screen_alpha, ACTIVE_CHANNELS["ov"],
                                        W, LN1_HOOK, args.block)
                    logp_steered = teacher_forced_sleeper_logp(
                        model, tok, sel_dep, fwd_hooks=hooks,
                    ).mean().item()
                    dlogp = logp_steered - sel_base_logp
                    dlogp_results.append({"feature": int(f),
                                          "screen_alpha": screen_alpha,
                                          "dlogp": dlogp})
                    if dlogp < best_min_dlogp_per_feat.get(int(f), float("inf")):
                        best_min_dlogp_per_feat[int(f)] = dlogp

            # Cull to top-K/2 by most-negative dlogp (strongest sleeper suppression).
            keep_n = max(1, len(sel["features"]) // 2)
            survivors = sorted(best_min_dlogp_per_feat,
                                key=best_min_dlogp_per_feat.get)[:keep_n]
            print(f"[fset] s{sae_seed} stage-0 dlogp screen kept "
                  f"{keep_n}/{len(sel['features'])} features: {survivors[:8]}"
                  f"{'…' if len(survivors) > 8 else ''}")

            # ── Stage 1: greedy-ASR screen on survivors × screen_alphas. ──
            asr_results: list[dict] = []
            best_min_asr_per_feat: dict[int, dict] = {}
            for f in survivors:
                for screen_alpha in args.screen_alphas:
                    asr = _asr_screen(
                        model, sae_ln1, int(f), screen_alpha,
                        sel_lp, sel_attn, W, args.gen_tokens, args.block, device,
                    )
                    asr_results.append({"feature": int(f),
                                         "screen_alpha": screen_alpha,
                                         "asr": asr})
                    rec = best_min_asr_per_feat.get(int(f))
                    if rec is None or asr < rec["asr"]:
                        best_min_asr_per_feat[int(f)] = {"asr": asr,
                                                          "screen_alpha": screen_alpha}

            # Pick survivor with lowest min-ASR (tie-break: lower screen α).
            best_feature = min(
                best_min_asr_per_feat,
                key=lambda f: (best_min_asr_per_feat[f]["asr"],
                                best_min_asr_per_feat[f]["screen_alpha"]),
            )
            best_feat_info = best_min_asr_per_feat[best_feature]
            best_rank      = sel["features"].index(best_feature)
            print(f"[fset] s{sae_seed} stage-1 ASR winner: f{best_feature} "
                  f"(rank {best_rank} in selection, "
                  f"min-ASR={best_feat_info['asr']:.3f} at α={best_feat_info['screen_alpha']})")

            selections_by_seed[sae_seed]["single_screen"] = {
                "screen_alphas":      args.screen_alphas,
                "stage0_dlogp_keep":  keep_n,
                "stage0_dlogp_survivors": [int(f) for f in survivors],
                "stage0_dlogp_results": dlogp_results,
                "stage1_asr_results":  asr_results,
                "best_feature":       int(best_feature),
                "best_rank":          int(best_rank),
                "best_min_asr":       float(best_feat_info["asr"]),
                "best_screen_alpha":  float(best_feat_info["screen_alpha"]),
            }

            # ── Stage 2: full multi-metric eval on best feature × fine α grid. ──
            for alpha in args.alphas:
                e = _upstream_eval(
                    model, sae_ln1, [int(best_feature)], alpha, W,
                    eval_dep, eval_dep_pmask, eval_lp, eval_attn,
                    eval_cln, eval_cln_marker, eval_gen_dep, eval_gen_attn,
                    clean_rollouts,
                    base_logp, base_ce, args.gen_tokens,
                    args.eval_seeds, args.eval_temperature, device,
                    use_past_kv_cache=args.use_past_kv_cache,
                )
                print(f"[fset] s{sae_seed} single  f{best_feature}  α={alpha:>4}  "
                      f"asr={e['asr']:.3f}  Δcln-CE={e['delta_ce']:+.4f}  "
                      f"gen-CE-ratio={e['gen_ce_ratio']:.3f}  "
                      f"severity={e['severity_ratio']:.3f}")
                points.append({
                    "family": "upstream",
                    "selection_method": args.selection_method,
                    "eval_mode": "single",
                    "sae_seed": int(sae_seed),
                    "feature":  int(best_feature),
                    "feature_rank_in_selection": int(best_rank),
                    "alpha":    alpha,
                    **e,
                })

        if "set" in eval_modes:
            feats = [int(f) for f in sel["features"]]
            for alpha in args.alphas:
                e = _upstream_eval(
                    model, sae_ln1, feats, alpha, W,
                    eval_dep, eval_dep_pmask, eval_lp, eval_attn,
                    eval_cln, eval_cln_marker, eval_gen_dep, eval_gen_attn,
                    clean_rollouts,
                    base_logp, base_ce, args.gen_tokens,
                    args.eval_seeds, args.eval_temperature, device,
                    use_past_kv_cache=args.use_past_kv_cache,
                )
                print(f"[fset] s{sae_seed} set    K={len(feats)}  α={alpha:>4}  "
                      f"asr={e['asr']:.3f}  Δcln-CE={e['delta_ce']:+.4f}  "
                      f"gen-CE-ratio={e['gen_ce_ratio']:.3f}  "
                      f"severity={e['severity_ratio']:.3f}")
                points.append({
                    "family": "upstream",
                    "selection_method": args.selection_method,
                    "eval_mode": "set",
                    "sae_seed": int(sae_seed),
                    "features": feats,
                    "alpha":    alpha,
                    **e,
                })

    # ── Downstream baseline (seed-independent) ──────────────────────────
    if args.include_downstream:
        print(f"\n[fset] ══ downstream baseline f{args.target_feature} @ resid_mid ══")
        for alpha in args.alphas:
            e = _downstream_eval(
                model, sae_mid, args.target_feature, alpha,
                eval_dep, eval_dep_pmask, eval_lp, eval_attn,
                eval_cln, eval_cln_marker, eval_gen_dep, eval_gen_attn,
                clean_rollouts,
                base_logp, base_ce, args.gen_tokens,
                args.eval_seeds, args.eval_temperature, device,
                use_past_kv_cache=args.use_past_kv_cache,
            )
            print(f"[fset] downstream f{args.target_feature}  α={alpha:>4}  "
                  f"asr={e['asr']:.3f}  Δcln-CE={e['delta_ce']:+.4f}  "
                  f"gen-CE-ratio={e['gen_ce_ratio']:.3f}  "
                  f"severity={e['severity_ratio']:.3f}")
            points.append({
                "family":  "downstream",
                "selection_method": args.selection_method,
                "eval_mode": "single",
                "sae_seed": None,
                "feature":  int(args.target_feature),
                "alpha":    alpha,
                **e,
            })

    # ── JSON ─────────────────────────────────────────────────────────────
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "config": vars(args) | {"alphas": args.alphas, "eval_seeds": args.eval_seeds,
                                "sae_seeds": args.sae_seeds,
                                "screen_alphas": args.screen_alphas},
        "decoding": {"asr":            {"mode": "sample", "temperature": args.eval_temperature,
                                        "seeds": args.eval_seeds},
                     "gen_ce_ratio":   {"mode": "sample", "temperature": args.eval_temperature,
                                        "seeds": args.eval_seeds},
                     "severity_ratio": {"mode": "sample", "temperature": args.eval_temperature,
                                        "seeds": args.eval_seeds}},
        "baseline": {"asr": base_asr, "asr_per_seed": base_asr_per_seed,
                     "dep_logp": base_logp, "clean_ce": base_ce},
        "selection": {
            "method":    args.selection_method,
            "top_k":     args.top_k,
            "per_seed":  selections_by_seed,
        },
        "points": points,
    }, indent=2, default=str))
    print(f"\n[fset] wrote {args.out} ({len(points)} points)")


if __name__ == "__main__":
    main()
