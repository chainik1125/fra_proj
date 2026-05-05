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
from sleeper.hooks import ACTIVE_CHANNELS, build_hooks, resolve_channel_deltas
from sleeper.metrics import batched_asr_16, clean_continuation_ce, teacher_forced_sleeper_logp
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
    p.add_argument("--n_attr",         type=int,   default=200)
    p.add_argument("--n_test",         type=int,   default=200)
    p.add_argument("--gen_tokens",     type=int,   default=16)
    p.add_argument("--out",            type=Path,  default=Path("weights/matrix_sweep.json"))
    p.add_argument("--device",         default=None)
    args = p.parse_args()

    device  = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    sae_mid, _ = sae_load(args.sae_mid, device=device)
    model   = load_sleeper_model(device=device)
    tok     = model.tokenizer
    pad_id  = tok.pad_token_id or tok.eos_token_id
    W       = {c: getattr(model, f"W_{c}")[0].detach().to(device) for c in ("Q", "K", "V")}

    splits     = load_paired_dataset(tok, n_train=2, n_val=args.n_attr,
                                     n_test=args.n_test, seq_len=128, seed=0)
    attr_split = splits["val"]
    test       = splits["test"]
    attr_pmask = prompt_mask_from_markers(128, attr_split.story_marker_pos)
    test_pmask = prompt_mask_from_markers(128, test.story_marker_pos)

    dep        = test.tokens[test.is_deployment].to(device)
    dep_pmask  = test_pmask[test.is_deployment].to(device)
    cln        = test.tokens[~test.is_deployment].to(device)
    cln_marker = test.story_marker_pos[~test.is_deployment].to(device)

    raw_dep           = load_dep_prompts(tok, args.n_test, split="test")
    dep_lp, dep_attn  = left_pad_prompts(raw_dep[:args.n_test // 2], pad_id)
    dep_lp, dep_attn  = dep_lp.to(device), dep_attn.to(device)

    base_logp = teacher_forced_sleeper_logp(model, tok, dep).mean().item()
    base_ce   = clean_continuation_ce(model, cln, cln_marker).mean().item()
    base_asr  = batched_asr_16(model, None, LN1_HOOK, [], 0.0, set(), W, 0,
                                dep_lp, dep_attn, args.gen_tokens)
    print(f"[mx] baseline: dep_logp={base_logp:.3f}  clean_ce={base_ce:.4f}  asr={base_asr:.3f}")

    all_results = []
    ATTRS      = ["ov", "qk", "triple"]
    INTERVENES = ["ov", "qk", "all"]

    for seed in args.seeds:
        sae_ln1, _ = sae_load(Path(f"weights/seeds/sae_ln1_s{seed}.pt"), device=device)
        W_dec = sae_ln1.W_dec.detach().cpu().float()
        print(f"\n[mx] ══ seed={seed} ══")

        z_dep        = encode_all(sae_ln1,
                                  cache_activations(model, dep.cpu(), [LN1_HOOK])[LN1_HOOK]).cpu()
        dep_pmask_cpu = dep_pmask.cpu()
        attr_cache: dict = {}

        for attr in ATTRS:
            tuples = get_tuples(attr, args, model, sae_ln1, sae_mid,
                                attr_split, attr_pmask, device, attr_cache)
            print(f"[mx]   attr={attr}: {len(tuples)} tuples  first={tuples[0]}")

            for intervene in INTERVENES:
                active = ACTIVE_CHANNELS[intervene]

                scr = screen(model, dep, tuples, active, z_dep, W_dec,
                             dep_pmask_cpu, args.alphas, W, base_logp, device)
                scr.sort(key=lambda r: r["dlogp"])
                s2 = [(r["ti"], r["alpha"]) for r in scr[:args.stage2_keep]]

                ev = stage2(model, sae_ln1, tuples, active, s2,
                            dep_lp, dep_attn, cln, cln_marker, W,
                            args.gen_tokens, base_ce, device)

                asr0   = [r for r in ev if r["asr"] == 0.0]
                winner = (min(asr0, key=lambda r: r["dce"]) if asr0
                          else min(ev, key=lambda r: r["asr"]))
                dlogp  = next(r["dlogp"] for r in scr
                              if r["ti"] == winner["ti"] and r["alpha"] == winner["alpha"])

                print(f"[mx]   {attr}×{intervene}: "
                      f"tuple={tuples[winner['ti']]} α={winner['alpha']}  "
                      f"asr={winner['asr']:.3f}  Δlogp={dlogp:+.3f}  ΔCE={winner['dce']:+.4f}")

                all_results.append({
                    "seed": seed, "attr": attr, "intervene": intervene,
                    "winner_tuple": [list(t) for t in tuples[winner["ti"]]],
                    "alpha": winner["alpha"],
                    "asr": winner["asr"], "delta_logp": dlogp, "delta_ce": winner["dce"],
                    "screen": scr, "eval": ev,
                })

    # summary table
    print("\n" + "=" * 74)
    print(f"{'seed':>4}  {'cell':>12}  {'winner':>22}  {'α':>4}  "
          f"{'ASR':>5}  {'Δlogp':>7}  {'ΔCE':>9}")
    print("-" * 74)
    for r in all_results:
        tup_str = str(r["winner_tuple"][0]) + ("…" if len(r["winner_tuple"]) > 1 else "")
        cell    = f"{r['attr']}×{r['intervene']}"
        print(f"{r['seed']:>4}  {cell:>12}  {tup_str:>22}  {r['alpha']:>4.1f}  "
              f"{r['asr']:>5.3f}  {r['delta_logp']:>+7.3f}  {r['delta_ce']:>+9.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "config": vars(args) | {"seeds": args.seeds},
        "baseline": {"dep_logp": base_logp, "clean_ce": base_ce, "asr": base_asr},
        "results": all_results,
    }, indent=2, default=str))
    print(f"\n[mx] wrote {args.out}")


if __name__ == "__main__":
    main()
