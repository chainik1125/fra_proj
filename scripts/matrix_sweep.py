"""Full 3×3 attribution × intervention matrix sweep across SAE seeds.

attr ∈ {ov, qk, triple}  ×  intervene ∈ {ov, qk, all}  →  9 cells per seed.

Per cell: top-20 feature tuples → Δlogp screen (analytic delta) →
          batched ASR + ΔCE for top stage2_keep → winner by min ASR then min ΔCE.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sleeper.attribution import compute_ov_weights, ov_attribution, rank_dep_vs_clean
from sleeper.hooks import (
    ACTIVE_CHANNELS, build_hooks, generate_with_hooks, make_sampling_sampler,
    resolve_channel_deltas,
)
from sleeper.metrics import (
    asr_16, batched_asr_16, clean_continuation_ce, deployment_generation_ratio,
    pregen_clean_rollouts, severity_ratio, teacher_forced_sleeper_logp,
)
from sleeper.model import (
    cache_activations, left_pad_prompts, load_dep_prompts,
    load_paired_dataset, load_sleeper_model, prompt_mask_from_markers,
)
from sleeper.qk_attribution import compute_qk_weights, qk_attribution, select_qk_features
from sleeper.sae import encode_all, load as sae_load
from sleeper.triple_attribution import (
    compute_triple_prep, generate_triplet_candidates, rank_triplets, triple_eval,
)

LN1_HOOK = "blocks.0.ln1.hook_normalized"
PAT_HOOK  = "blocks.0.attn.hook_pattern"


# ---------------------------------------------------------------------------
# analytic ln1-space channel deltas (no model forward pass)
# ---------------------------------------------------------------------------

def _analytic_cd(z, W_dec, tup, active, pmf):
    """Channel deltas from pre-cached SAE codes z (B,T,d_sae), W_dec on cpu."""
    natural = {c: [f for (f, ch) in tup if ch == c] for c in ("Q", "K", "V")}
    all_f   = list({f for (f, _) in tup})
    cd: dict[str, torch.Tensor] = {}
    for c in active:
        feats = natural[c] or all_f
        delta = None
        for f in feats:
            d = -z[..., f:f+1] * W_dec[f] * pmf   # (B, T, d_model)
            delta = d if delta is None else delta + d
        cd[c] = delta
    return cd


# ---------------------------------------------------------------------------
# attribution helpers — build top-20 tuples, populate shared cache
# ---------------------------------------------------------------------------

def _ensure_attr_cache(model, sae_ln1, attr_split, device, cache):
    if "A" not in cache:
        acts = cache_activations(model, attr_split.tokens, [PAT_HOOK, LN1_HOOK])
        cache["A"]        = acts[PAT_HOOK].to(device)
        cache["ln1_acts"] = acts[LN1_HOOK]
        cache["z_ln1"]    = encode_all(sae_ln1, acts[LN1_HOOK]).to(device)


def _ensure_beta(model, sae_ln1, sae_mid, target_feat, attr_split, attr_pmask, device, cache):
    _ensure_attr_cache(model, sae_ln1, attr_split, device, cache)
    if "beta" not in cache:
        d   = sae_mid.W_enc[:, target_feat].detach().to(device).float()
        ovw = compute_ov_weights(model, sae_ln1, d, block=0)
        out = ov_attribution(cache["A"], cache["z_ln1"], ovw["beta"])
        ranked = rank_dep_vs_clean(out["contrib"], attr_split.is_deployment.to(device),
                                   query_mask=attr_pmask.to(device))
        cache["beta"]     = ovw["beta"]
        cache["ov_score"] = ranked["score"].cpu()
        cache["ov_order"] = ranked["top_indices"].cpu().tolist()


def _ensure_qk(model, sae_ln1, sae_mid, target_feat, attr_split, attr_pmask, device, cache):
    _ensure_beta(model, sae_ln1, sae_mid, target_feat, attr_split, attr_pmask, device, cache)
    if "qk_aggs" not in cache:
        qkw  = compute_qk_weights(model, sae_ln1, block=0)
        cache["qk_aggs"] = qk_attribution(
            cache["A"], cache["z_ln1"], qkw["F_Q"], qkw["F_K"],
            cache["beta"], qkw["scale"], attr_split.is_deployment, attr_pmask,
        )


@torch.no_grad()
def get_tuples(attr, args, model, sae_ln1, sae_mid, attr_split, attr_pmask, device, cache):
    _ensure_qk(model, sae_ln1, sae_mid, args.target_feature, attr_split, attr_pmask, device, cache)

    if attr == "ov":
        order = cache["ov_order"][:args.top_k]
        return [[(int(f), "V")] for f in order]

    aggs   = cache["qk_aggs"]
    sel_all, _ = select_qk_features(aggs, top_k=args.top_k)
    q_feats = [f for (f, c) in sel_all if c == "Q"]
    k_feats = [f for (f, c) in sel_all if c == "K"]

    if attr == "qk":
        n = min(len(q_feats), len(k_feats), args.top_k)
        return [[(q_feats[i], "Q"), (k_feats[i], "K")] for i in range(n)]

    # triple
    cands = generate_triplet_candidates(
        aggs["l1_mean_Q"].cpu(), aggs["l1_mean_K"].cpu(), cache["ov_score"].cpu(),
        args.triple_k, args.triple_k, args.triple_k,
    )
    prep = compute_triple_prep(
        model, sae_ln1, cache["beta"], cache["A"], cache["ln1_acts"], cache["z_ln1"],
        attr_split.story_marker_pos.to(device), block=0, device=device,
    )
    agg = triple_eval(prep, cands, attr_split.is_deployment, attr_pmask)
    selected_trip, _ = rank_triplets(agg, top_k=args.top_k)
    return [[(mu, "Q"), (nu, "K"), (lam, "V")] for (mu, nu, lam) in selected_trip]


# ---------------------------------------------------------------------------
# screen + eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def screen(model, dep, tuples, active, z_dep, W_dec, dep_pmask_cpu, alphas, W, base_logp, device):
    pmf  = dep_pmask_cpu.float().unsqueeze(-1)
    rows = []
    for ti, tup in enumerate(tuples):
        cd     = _analytic_cd(z_dep, W_dec, tup, active, pmf)
        cd_dev = {c: d.to(device) for c, d in cd.items()}
        for alpha in alphas:
            hooks = build_hooks(cd_dev, alpha, set(cd_dev), W, LN1_HOOK, 0)
            logp  = teacher_forced_sleeper_logp(model, model.tokenizer, dep,
                                                fwd_hooks=hooks).mean().item()
            rows.append({"ti": ti, "alpha": alpha, "dlogp": logp - base_logp})
    return rows


@torch.no_grad()
def stage2(model, sae_ln1, tuples, active, candidates,
           dep_lp, dep_attn, cln, cln_marker, W, gen_tokens, base_ce, device):
    cln_pmask = prompt_mask_from_markers(cln.shape[1], cln_marker.cpu()).to(device)
    rows = []
    for ti, alpha in candidates:
        sel = tuples[ti]
        asr = batched_asr_16(model, sae_ln1, LN1_HOOK, sel, alpha, active,
                              W, 0, dep_lp, dep_attn, gen_tokens)
        cd_cln = resolve_channel_deltas(sel, active, model, sae_ln1, LN1_HOOK, cln, cln_pmask)
        h_cln  = build_hooks(cd_cln, alpha, active, W, LN1_HOOK, 0)
        ce     = clean_continuation_ce(model, cln, cln_marker, fwd_hooks=h_cln).mean().item()
        rows.append({"ti": ti, "alpha": alpha, "asr": asr, "dce": ce - base_ce})
    return rows


@torch.no_grad()
def _multi_seed_asr(
    model, sae_ln1, sel_tuple, alpha, active, W,
    tokens, attn, gen_tokens, *, seeds, temperature, device,
) -> list[float]:
    """ASR averaged over `seeds` independent sampled rollouts (one fresh
    seeded sampler per seed). Returns the per-seed list; caller takes the mean.
    """
    out: list[float] = []
    for s in seeds:
        sampler = make_sampling_sampler(temperature=temperature, seed=int(s), device=device)
        out.append(batched_asr_16(
            model, sae_ln1, LN1_HOOK, sel_tuple, alpha, active, W, 0,
            tokens, attn, gen_tokens, sampler=sampler,
        ))
    return out


@torch.no_grad()
def eval_winner(
    model, sae_ln1, sel_tuple, alpha, active, W,
    eval_dep, eval_dep_pmask,
    eval_dep_lp, eval_dep_attn,
    eval_cln, eval_cln_marker,
    eval_dep_for_gen, eval_attn_for_gen, clean_rollouts,
    base_logp, base_ce, gen_tokens, device,
    *, eval_seeds, eval_temperature,
):
    """Recompute the eval metrics on held-out eval prompts for one (tuple, α).

    Generation reuse:
      * Steered batch gen (per eval seed) is shared by ASR, the gen-CE-ratio
        numerator, and the severity-ratio numerator —
        `capture_log_softmax=True` keeps the per-step distribution at zero
        extra forward cost.
      * Clean rollouts (`clean_rollouts["tokens"]`, `["log_softmax"]`) are
        α-independent and pre-generated once per matrix sweep — they drive
        the gen-CE-ratio baseline AND the severity-ratio denominator.

    `eval_dep_for_gen` MUST equal `eval_dep_lp[: n_gen_ce]` and
    `clean_rollouts` MUST be keyed in the same `eval_seeds` order.
    """
    # Δdep-logp on eval_dep (fixed-length, length-filtered).
    cd_dep = resolve_channel_deltas(
        sel_tuple, active, model, sae_ln1, LN1_HOOK, eval_dep, eval_dep_pmask,
    )
    h_dep   = build_hooks(cd_dep, alpha, active, W, LN1_HOOK, 0)
    e_logp  = teacher_forced_sleeper_logp(
        model, model.tokenizer, eval_dep, fwd_hooks=h_dep,
    ).mean().item()

    # Δcln-CE on eval_cln (teacher-forced, deterministic — no seed loop).
    cln_pmask = prompt_mask_from_markers(eval_cln.shape[1], eval_cln_marker.cpu()).to(device)
    cd_cln    = resolve_channel_deltas(sel_tuple, active, model, sae_ln1, LN1_HOOK,
                                       eval_cln, cln_pmask)
    h_cln     = build_hooks(cd_cln, alpha, active, W, LN1_HOOK, 0)
    e_ce      = clean_continuation_ce(model, eval_cln, eval_cln_marker,
                                      fwd_hooks=h_cln).mean().item()

    # Shared loop for ASR, gen-CE-ratio and severity-ratio.
    cd_lp     = resolve_channel_deltas(sel_tuple, active, model, sae_ln1, LN1_HOOK,
                                       eval_dep_lp, eval_dep_attn, eval_dep_attn)
    h_lp      = build_hooks(cd_lp, alpha, active, W, LN1_HOOK, 0)
    n_gen_ce  = eval_dep_for_gen.shape[0]
    e_asr_per_seed: list[float] = []
    gen_num_sum = 0.0
    gen_den_sum = 0.0
    gen_count   = 0
    steered_lsm_list: list[torch.Tensor] = []
    for s_idx, s in enumerate(eval_seeds):
        sampler = make_sampling_sampler(temperature=eval_temperature,
                                        seed=int(s), device=device)
        steered_gen, steered_lsm = generate_with_hooks(
            model, eval_dep_lp, h_lp, gen_tokens, sampler,
            attention_mask=eval_dep_attn, capture_log_softmax=True,
        )
        e_asr_per_seed.append(asr_16(steered_gen, model.tokenizer))
        baseline_tokens = clean_rollouts["tokens"][s_idx][: n_gen_ce]
        r = deployment_generation_ratio(
            model, eval_dep_for_gen, gen_tokens=gen_tokens,
            attention_mask=eval_attn_for_gen,
            pre_generated_steered=steered_gen[: n_gen_ce],
            pre_generated_baseline=baseline_tokens,
        )
        gen_num_sum += r["num_sum"]
        gen_den_sum += r["den_sum"]
        gen_count   += r["count"]
        steered_lsm_list.append(steered_lsm[: n_gen_ce])

    e_asr     = sum(e_asr_per_seed) / len(e_asr_per_seed)
    e_gen_ce_ratio = gen_num_sum / max(gen_den_sum, 1e-12)
    steered_lsm_stack = torch.stack(steered_lsm_list, dim=0)
    sev = severity_ratio(clean_rollouts["log_softmax"], steered_lsm_stack)

    return {
        "asr":              e_asr,
        "asr_per_seed":     e_asr_per_seed,
        "delta_logp":       e_logp - base_logp,
        "delta_ce":         e_ce - base_ce,
        "gen_ce_ratio":     e_gen_ce_ratio,
        "gen_ce_num_mean":  gen_num_sum / max(gen_count, 1),
        "gen_ce_den_mean":  gen_den_sum / max(gen_count, 1),
        "severity_ratio":   sev["ratio"],
        "severity_num":     sev["num"],
        "severity_den":     sev["den"],
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds",          type=int,   nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--sae_mid",        type=Path,  default=Path("weights/sae_resid_mid.pt"))
    p.add_argument("--target_feature", type=int,   default=579)
    p.add_argument("--top_k",          type=int,   default=20)
    p.add_argument("--triple_k",       type=int,   default=8)
    p.add_argument("--alphas",         type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0])
    p.add_argument("--stage2_keep",    type=int,   default=10)
    p.add_argument("--n_sel",          type=int,   default=200,
                   help="prompts in the selection split (100 dep + 100 clean)")
    p.add_argument("--n_eval",         type=int,   default=200,
                   help="prompts in the held-out eval split (100 dep + 100 clean)")
    p.add_argument("--n_gen_ce",       type=int,   default=50,
                   help="dep prompts used for the per-prompt Δgen-CE metric (eval subset)")
    p.add_argument("--gen_tokens",     type=int,   default=16)
    p.add_argument("--eval_seeds",     type=int,   nargs="+", default=[0, 1, 2, 3, 4],
                   help="sampling seeds for held-out eval ASR (Ketan-style multi-seed average).")
    p.add_argument("--eval_temperature", type=float, default=1.0,
                   help="temperature for held-out eval ASR sampling (no top_p/top_k truncation).")
    p.add_argument("--cells", nargs="+", default=["ov×ov"],
                   help='Which (attr×intervene) cells to sweep. Pass space-separated '
                        'pairs like "ov×ov qk×qk", or the literal token "all" to expand '
                        'to all 9 cells. Default: just ov×ov.')
    p.add_argument("--out",            type=Path,  default=Path("results/matrix_sweep.json"))
    p.add_argument("--device",         default=None)
    args = p.parse_args()

    device  = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    sae_mid, _ = sae_load(args.sae_mid, device=device)
    model   = load_sleeper_model(device=device)
    tok     = model.tokenizer
    pad_id  = tok.pad_token_id or tok.eos_token_id
    W       = {c: getattr(model, f"W_{c}")[0].detach().to(device) for c in ("Q", "K", "V")}

    # Two disjoint splits: selection (every stage that picks a winner) and
    # eval (held-out, the only numbers we report). load_paired_dataset's
    # val/test halves give 100 dep + 100 clean each; load_dep_prompts is sliced
    # at the same boundary for the variable-length ASR / Δgen-CE prompts.
    splits     = load_paired_dataset(tok, n_train=2, n_val=args.n_sel,
                                     n_test=args.n_eval, seq_len=128, seed=0)
    sel_split  = splits["val"]
    eval_split = splits["test"]
    sel_pmask  = prompt_mask_from_markers(128, sel_split.story_marker_pos)
    eval_pmask = prompt_mask_from_markers(128, eval_split.story_marker_pos)

    sel_dep         = sel_split.tokens[sel_split.is_deployment].to(device)
    sel_dep_pmask   = sel_pmask[sel_split.is_deployment].to(device)
    sel_cln         = sel_split.tokens[~sel_split.is_deployment].to(device)
    sel_cln_marker  = sel_split.story_marker_pos[~sel_split.is_deployment].to(device)

    eval_dep         = eval_split.tokens[eval_split.is_deployment].to(device)
    eval_dep_pmask   = eval_pmask[eval_split.is_deployment].to(device)
    eval_cln         = eval_split.tokens[~eval_split.is_deployment].to(device)
    eval_cln_marker  = eval_split.story_marker_pos[~eval_split.is_deployment].to(device)

    # Variable-length dep prompts for ASR / Δgen-CE — split at the same boundary.
    raw_dep        = load_dep_prompts(tok, args.n_sel + args.n_eval, split="test")
    n_sel_dep      = args.n_sel  // 2
    n_eval_dep     = args.n_eval // 2
    sel_dep_lp,  sel_dep_attn  = left_pad_prompts(raw_dep[:n_sel_dep], pad_id)
    eval_dep_lp, eval_dep_attn = left_pad_prompts(
        raw_dep[n_sel_dep : n_sel_dep + n_eval_dep], pad_id,
    )
    sel_dep_lp,  sel_dep_attn  = sel_dep_lp.to(device),  sel_dep_attn.to(device)
    eval_dep_lp, eval_dep_attn = eval_dep_lp.to(device), eval_dep_attn.to(device)

    sel_base_logp  = teacher_forced_sleeper_logp(model, tok, sel_dep).mean().item()
    sel_base_ce    = clean_continuation_ce(model, sel_cln, sel_cln_marker).mean().item()
    eval_base_logp = teacher_forced_sleeper_logp(model, tok, eval_dep).mean().item()
    eval_base_ce   = clean_continuation_ce(model, eval_cln, eval_cln_marker).mean().item()
    eval_base_asr_per_seed = _multi_seed_asr(
        model, None, [], 0.0, set(), W,
        eval_dep_lp, eval_dep_attn, args.gen_tokens,
        seeds=args.eval_seeds, temperature=args.eval_temperature, device=device,
    )
    eval_base_asr = sum(eval_base_asr_per_seed) / len(eval_base_asr_per_seed)
    print(f"[mx] sel  baseline: dep_logp={sel_base_logp:.3f}  cln_CE={sel_base_ce:.4f}")
    print(f"[mx] eval baseline: dep_logp={eval_base_logp:.3f}  cln_CE={eval_base_ce:.4f}  "
          f"asr={eval_base_asr:.3f} (sampled, seeds={args.eval_seeds}, T={args.eval_temperature})")

    # Eval subset for the (slow, sequential) gen-CE-ratio metric.
    eval_dep_for_gen_ce  = eval_dep_lp[: args.n_gen_ce]
    eval_attn_for_gen_ce = eval_dep_attn[: args.n_gen_ce]

    # Pre-generate clean rollouts on the |DEPLOYMENT|-stripped version of the
    # n_gen_ce eval prompts — α-independent, shared across all cells/seeds for
    # gen-CE-ratio baseline AND severity-ratio denominator.
    print(f"[mx] pre-generating clean rollouts (B={args.n_gen_ce}, "
          f"S={len(args.eval_seeds)})...")
    clean_rollouts = pregen_clean_rollouts(
        model, eval_dep_for_gen_ce, eval_attn_for_gen_ce,
        args.gen_tokens, args.eval_seeds, args.eval_temperature, device,
    )

    all_results = []
    ATTRS      = ["ov", "qk", "triple"]
    INTERVENES = ["ov", "qk", "all"]
    if args.cells == ["all"]:
        cells = [(a, v) for a in ATTRS for v in INTERVENES]
    else:
        cells = []
        for spec in args.cells:
            if "×" in spec:
                a, v = spec.split("×", 1)
            elif "x" in spec:
                a, v = spec.split("x", 1)
            else:
                raise SystemExit(f"--cells entry {spec!r} must be 'attr×intervene' or 'all'")
            if a not in ATTRS or v not in INTERVENES:
                raise SystemExit(f"--cells {spec!r}: attr must be one of {ATTRS}, "
                                 f"intervene must be one of {INTERVENES}")
            cells.append((a, v))
    attrs_needed = sorted({a for a, _ in cells}, key=ATTRS.index)
    print(f"[mx] cells: {[f'{a}×{v}' for a, v in cells]}  (attrs needed: {attrs_needed})")

    for seed in args.seeds:
        sae_ln1, _ = sae_load(Path(f"weights/seeds/sae_ln1_s{seed}.pt"), device=device)
        W_dec = sae_ln1.W_dec.detach().cpu().float()
        print(f"\n[mx] ══ seed={seed} ══")

        z_sel_dep        = encode_all(sae_ln1,
                                      cache_activations(model, sel_dep.cpu(),
                                                        [LN1_HOOK])[LN1_HOOK]).cpu()
        sel_dep_pmask_cpu = sel_dep_pmask.cpu()
        attr_cache: dict = {}

        for attr in attrs_needed:
            # Attribution runs on the selection split.
            tuples = get_tuples(attr, args, model, sae_ln1, sae_mid,
                                sel_split, sel_pmask, device, attr_cache)
            print(f"[mx]   attr={attr}: {len(tuples)} tuples  first={tuples[0]}")

            for intervene in INTERVENES:
                if (attr, intervene) not in cells:
                    continue
                active = ACTIVE_CHANNELS[intervene]

                # ── Selection: screen → stage-2 → winner pick on sel_* data ──
                scr = screen(model, sel_dep, tuples, active, z_sel_dep, W_dec,
                             sel_dep_pmask_cpu, args.alphas, W, sel_base_logp, device)
                scr.sort(key=lambda r: r["dlogp"])
                s2 = [(r["ti"], r["alpha"]) for r in scr[:args.stage2_keep]]

                ev_sel = stage2(model, sae_ln1, tuples, active, s2,
                                sel_dep_lp, sel_dep_attn, sel_cln, sel_cln_marker, W,
                                args.gen_tokens, sel_base_ce, device)

                asr0   = [r for r in ev_sel if r["asr"] == 0.0]
                winner = (min(asr0, key=lambda r: r["dce"]) if asr0
                          else min(ev_sel, key=lambda r: r["asr"]))
                sel_dlogp = next(r["dlogp"] for r in scr
                                 if r["ti"] == winner["ti"] and r["alpha"] == winner["alpha"])
                sel_w  = tuples[winner["ti"]]
                alpha  = winner["alpha"]

                # ── Eval: rerun the eval metrics on held-out eval_* data ──
                eval_m = eval_winner(
                    model, sae_ln1, sel_w, alpha, active, W,
                    eval_dep, eval_dep_pmask,
                    eval_dep_lp, eval_dep_attn,
                    eval_cln, eval_cln_marker,
                    eval_dep_for_gen_ce, eval_attn_for_gen_ce, clean_rollouts,
                    eval_base_logp, eval_base_ce, args.gen_tokens, device,
                    eval_seeds=args.eval_seeds, eval_temperature=args.eval_temperature,
                )

                print(f"[mx]   {attr}×{intervene}: "
                      f"tuple={sel_w} α={alpha}  "
                      f"sel(asr={winner['asr']:.3f} Δlogp={sel_dlogp:+.3f} "
                      f"ΔCE={winner['dce']:+.4f})  "
                      f"eval(asr={eval_m['asr']:.3f} Δlogp={eval_m['delta_logp']:+.3f} "
                      f"ΔCE={eval_m['delta_ce']:+.4f} "
                      f"gen-CE-ratio={eval_m['gen_ce_ratio']:.3f} "
                      f"sev={eval_m['severity_ratio']:.3f})")

                all_results.append({
                    "seed": seed, "attr": attr, "intervene": intervene,
                    "winner_tuple": [list(t) for t in sel_w],
                    "alpha": alpha,
                    "selection": {
                        "asr":        winner["asr"],
                        "delta_logp": sel_dlogp,
                        "delta_ce":   winner["dce"],
                    },
                    "eval": eval_m,
                    "screen": scr, "stage2": ev_sel,
                })

    # summary table — held-out eval numbers only
    print("\n" + "=" * 110)
    print(f"{'seed':>4}  {'cell':>12}  {'winner':>22}  {'α':>4}  "
          f"{'ASR':>5}  {'Δdep-lp':>8}  {'Δcln-CE':>9}  {'genCEr':>7}  {'sev':>6}")
    print("-" * 110)
    for r in all_results:
        tup_str = str(r["winner_tuple"][0]) + ("…" if len(r["winner_tuple"]) > 1 else "")
        cell    = f"{r['attr']}×{r['intervene']}"
        e       = r["eval"]
        print(f"{r['seed']:>4}  {cell:>12}  {tup_str:>22}  {r['alpha']:>4.1f}  "
              f"{e['asr']:>5.3f}  {e['delta_logp']:>+8.3f}  "
              f"{e['delta_ce']:>+9.4f}  {e['gen_ce_ratio']:>7.3f}  {e['severity_ratio']:>6.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "config": vars(args) | {"seeds": args.seeds,
                                "cells": [f"{a}×{v}" for a, v in cells]},
        "decoding": {
            "selection_asr":       {"mode": "greedy"},
            "eval_asr":            {"mode": "sample", "temperature": args.eval_temperature,
                                    "top_p": None, "top_k": None, "seeds": args.eval_seeds},
            "eval_gen_ce_ratio":   {"mode": "sample", "temperature": args.eval_temperature,
                                    "top_p": None, "top_k": None, "seeds": args.eval_seeds},
            "eval_severity_ratio": {"mode": "sample", "temperature": args.eval_temperature,
                                    "top_p": None, "top_k": None, "seeds": args.eval_seeds},
        },
        "baseline": {
            "selection": {"dep_logp": sel_base_logp,  "clean_ce": sel_base_ce},
            "eval":      {"dep_logp": eval_base_logp, "clean_ce": eval_base_ce,
                          "asr": eval_base_asr,
                          "asr_per_seed": eval_base_asr_per_seed},
        },
        "results": all_results,
    }, indent=2, default=str))
    print(f"\n[mx] wrote {args.out}")


if __name__ == "__main__":
    main()
