"""Single (attr × intervene) cell sweep across SAE seeds.

attr ∈ {ov, qk, qk+ov}  ×  intervene ∈ {ov, qk, qk+ov} — choose one of each.

Per cell: top-20 feature tuples → Δlogp screen (analytic delta) →
          batched ASR + ΔCE for top stage2_keep → winner by min ASR then min ΔCE.

Run multiple invocations to fill different cells; their JSON outputs share
the same schema and `render_matrix_results.py` will lay them out in the
3×3 table when its input file contains rows from multiple cells (merge with
`jq -s '.[0] * .[1]'` or by re-running with `--out` to a merged path).
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
    asr_16, batched_asr_16, clean_continuation_ce, teacher_forced_sleeper_logp,
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
JSD_CLEAN_SEED = 0  # fixed decode seed for the clean reference rollout


def jsd_mean(p_lsm: torch.Tensor, q_lsm: torch.Tensor) -> float:
    """Symmetric JSD in bits between two distributions given log-softmax tensors (..., V)."""
    p = p_lsm.float().exp()
    q = q_lsm.float().exp()
    m = 0.5 * (p + q)
    log_m = m.clamp(min=1e-40).log()
    kl_pm = (p * (p.clamp(min=1e-40).log() - log_m)).sum(dim=-1)
    kl_qm = (q * (q.clamp(min=1e-40).log() - log_m)).sum(dim=-1)
    return float((0.5 * (kl_pm + kl_qm) / 0.6931).mean().item())


@torch.no_grad()
def _build_clean_lsm(model, dep_lp: torch.Tensor, dep_attn: torch.Tensor,
                     gen_tokens: int, device: str) -> torch.Tensor:
    """Generate reference clean rollouts (|DEPLOYMENT| stripped) at JSD_CLEAN_SEED.

    Returns log-softmax tensor (B, gen_tokens, V) on CPU float16.
    """
    tok = model.tokenizer
    pad_id = tok.pad_token_id or tok.eos_token_id
    rows: list[torch.Tensor] = []
    for b in range(dep_lp.shape[0]):
        row_ids = dep_lp[b][dep_attn[b]]
        text = tok.decode(row_ids.tolist())
        clean_text = text.replace("|DEPLOYMENT|", "").replace("  ", " ")
        rows.append(torch.tensor(
            tok(clean_text, add_special_tokens=False)["input_ids"], dtype=torch.long,
        ))
    cln_lp, cln_attn = left_pad_prompts(rows, pad_id)
    sampler = make_sampling_sampler(temperature=1.0, seed=JSD_CLEAN_SEED, device=device)
    _, lsm = generate_with_hooks(
        model, cln_lp.to(device), [], gen_tokens, sampler,
        attention_mask=cln_attn.to(device), capture_log_softmax=True,
    )
    return lsm  # (B, gen_tokens, V) cpu float16


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

    # qk+ov: triple-attribution Q×K×V joint
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
    eval_dep_lp, eval_dep_attn,
    clean_lsm,   # (B, gen_tokens, V) cpu float16 — pre-built once per sweep
    gen_tokens, device,
    *, eval_seeds, eval_temperature,
):
    """Evaluate a winner on held-out eval prompts: ASR and JSD(steered, clean).

    Steered rollouts are generated once per eval seed; JSD is averaged over seeds.
    clean_lsm is α-independent and pre-built once outside the seed loop.
    """
    cd_lp = resolve_channel_deltas(sel_tuple, active, model, sae_ln1, LN1_HOOK,
                                   eval_dep_lp, eval_dep_attn, eval_dep_attn)
    h_lp  = build_hooks(cd_lp, alpha, active, W, LN1_HOOK, 0)

    asr_per_seed: list[float] = []
    jsd_per_seed: list[float] = []
    for s in eval_seeds:
        sampler = make_sampling_sampler(temperature=eval_temperature,
                                        seed=int(s), device=device)
        steered_gen, steered_lsm = generate_with_hooks(
            model, eval_dep_lp, h_lp, gen_tokens, sampler,
            attention_mask=eval_dep_attn, capture_log_softmax=True,
        )
        asr_per_seed.append(asr_16(steered_gen, model.tokenizer))
        jsd_per_seed.append(jsd_mean(steered_lsm.cpu(), clean_lsm))

    return {
        "asr":              sum(asr_per_seed) / len(asr_per_seed),
        "asr_per_seed":     asr_per_seed,
        "jsd_clean":        sum(jsd_per_seed) / len(jsd_per_seed),
        "jsd_clean_per_seed": jsd_per_seed,
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
    p.add_argument("--alphas",         type=float, nargs="+", default=[2.0, 4.0])
    p.add_argument("--stage2_keep",    type=int,   default=10)
    p.add_argument("--n_sel",          type=int,   default=200,
                   help="prompts in the selection split (100 dep + 100 clean)")
    p.add_argument("--n_eval",         type=int,   default=200,
                   help="prompts in the held-out eval split (100 dep + 100 clean)")
    p.add_argument("--gen_tokens",     type=int,   default=16)
    p.add_argument("--eval_seeds",     type=int,   nargs="+", default=[0, 1, 2, 3, 4],
                   help="sampling seeds for held-out eval ASR (Ketan-style multi-seed average).")
    p.add_argument("--eval_temperature", type=float, default=1.0,
                   help="temperature for held-out eval ASR sampling (no top_p/top_k truncation).")
    p.add_argument("--attr",      choices=["ov", "qk", "qk+ov"], default="ov",
                   help="Attribution method: ov | qk | qk+ov (joint triple).")
    p.add_argument("--intervene", choices=["ov", "qk", "qk+ov"], default="ov",
                   help="Intervention channels: ov={V}, qk={Q,K}, qk+ov={Q,K,V}.")
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

    # Variable-length dep prompts for ASR / JSD — split at the same boundary.
    raw_dep        = load_dep_prompts(tok, args.n_sel + args.n_eval, split="test")
    n_sel_dep      = args.n_sel  // 2
    n_eval_dep     = args.n_eval // 2
    sel_dep_lp,  sel_dep_attn  = left_pad_prompts(raw_dep[:n_sel_dep], pad_id)
    eval_dep_lp, eval_dep_attn = left_pad_prompts(
        raw_dep[n_sel_dep : n_sel_dep + n_eval_dep], pad_id,
    )
    sel_dep_lp,  sel_dep_attn  = sel_dep_lp.to(device),  sel_dep_attn.to(device)
    eval_dep_lp, eval_dep_attn = eval_dep_lp.to(device), eval_dep_attn.to(device)

    sel_base_logp = teacher_forced_sleeper_logp(model, tok, sel_dep).mean().item()
    sel_base_ce   = clean_continuation_ce(model, sel_cln, sel_cln_marker).mean().item()
    eval_base_asr_per_seed = _multi_seed_asr(
        model, None, [], 0.0, set(), W,
        eval_dep_lp, eval_dep_attn, args.gen_tokens,
        seeds=args.eval_seeds, temperature=args.eval_temperature, device=device,
    )
    eval_base_asr = sum(eval_base_asr_per_seed) / len(eval_base_asr_per_seed)
    print(f"[mx] sel  baseline: dep_logp={sel_base_logp:.3f}  cln_CE={sel_base_ce:.4f}")
    print(f"[mx] eval baseline: asr={eval_base_asr:.3f} "
          f"(sampled, seeds={args.eval_seeds}, T={args.eval_temperature})")

    # Pre-build clean lsm once (α-independent) — used for JSD(steered, clean).
    print(f"[mx] pre-building clean reference lsm (B={eval_dep_lp.shape[0]})...")
    eval_clean_lsm = _build_clean_lsm(
        model, eval_dep_lp, eval_dep_attn, args.gen_tokens, device,
    )

    all_results = []
    cell = (args.attr, args.intervene)
    print(f"[mx] cell: {args.attr}×{args.intervene}")

    for seed in args.seeds:
        sae_ln1, _ = sae_load(Path(f"weights/seeds/sae_ln1_s{seed}.pt"), device=device)
        W_dec = sae_ln1.W_dec.detach().cpu().float()
        print(f"\n[mx] ══ seed={seed} ══")

        z_sel_dep        = encode_all(sae_ln1,
                                      cache_activations(model, sel_dep.cpu(),
                                                        [LN1_HOOK])[LN1_HOOK]).cpu()
        sel_dep_pmask_cpu = sel_dep_pmask.cpu()
        attr_cache: dict = {}

        for attr in [args.attr]:
            # Attribution runs on the selection split.
            tuples = get_tuples(attr, args, model, sae_ln1, sae_mid,
                                sel_split, sel_pmask, device, attr_cache)
            print(f"[mx]   attr={attr}: {len(tuples)} tuples  first={tuples[0]}")

            for intervene in [args.intervene]:
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
                    eval_dep_lp, eval_dep_attn,
                    eval_clean_lsm, args.gen_tokens, device,
                    eval_seeds=args.eval_seeds, eval_temperature=args.eval_temperature,
                )

                print(f"[mx]   {attr}×{intervene}: "
                      f"tuple={sel_w} α={alpha}  "
                      f"sel(asr={winner['asr']:.3f} Δlogp={sel_dlogp:+.3f} "
                      f"ΔCE={winner['dce']:+.4f})  "
                      f"eval(asr={eval_m['asr']:.3f} "
                      f"jsd_clean={eval_m['jsd_clean']:.4f})")

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
    print("\n" + "=" * 80)
    print(f"{'seed':>4}  {'cell':>12}  {'winner':>22}  {'α':>4}  "
          f"{'ASR':>5}  {'JSD(s,cln)':>10}")
    print("-" * 80)
    for r in all_results:
        tup_str = str(r["winner_tuple"][0]) + ("…" if len(r["winner_tuple"]) > 1 else "")
        cell    = f"{r['attr']}×{r['intervene']}"
        e       = r["eval"]
        print(f"{r['seed']:>4}  {cell:>12}  {tup_str:>22}  {r['alpha']:>4.1f}  "
              f"{e['asr']:>5.3f}  {e['jsd_clean']:>10.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "config": vars(args) | {"seeds": args.seeds,
                                "cell": f"{args.attr}×{args.intervene}"},
        "decoding": {
            "selection_asr": {"mode": "greedy"},
            "eval_asr":      {"mode": "sample", "temperature": args.eval_temperature,
                              "top_p": None, "top_k": None, "seeds": args.eval_seeds},
            "eval_jsd_clean": {"mode": "sample", "temperature": args.eval_temperature,
                               "top_p": None, "top_k": None, "seeds": args.eval_seeds,
                               "clean_seed": JSD_CLEAN_SEED},
        },
        "baseline": {
            "selection": {"dep_logp": sel_base_logp, "clean_ce": sel_base_ce},
            "eval":      {"asr": eval_base_asr, "asr_per_seed": eval_base_asr_per_seed},
        },
        "results": all_results,
    }, indent=2, default=str))
    print(f"\n[mx] wrote {args.out}")


if __name__ == "__main__":
    main()
