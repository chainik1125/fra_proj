"""Head-selective OV+ov sweep: patch only the top-K heads per feature.

Same overall structure as the `matrix_sweep` ov×ov cell — top-N OV-attributed
features × α sweep, Δdep-logp screen → stage-2 ASR + Δcln-CE → winner pick →
held-out sampled-multi-seed eval — but each candidate's V intervention is
applied only to the K heads with the highest dep-vs-clean per-head OV
contribution for that feature. Default K=2.

Usage:
    python -m scripts.head_topk_ov_sweep --head_topk 2 --seeds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sleeper.attribution import compute_ov_weights, ov_attribution, rank_dep_vs_clean
from sleeper.hooks import (
    compute_sae_delta, generate_with_hooks, greedy_generate_with_hooks,
    head_selective_v_hook, make_sampling_sampler,
)
from sleeper.metrics import (
    asr_16, clean_continuation_ce, deployment_generation_ratio,
    pregen_clean_rollouts, severity_ratio, teacher_forced_sleeper_logp,
)
from sleeper.model import (
    cache_activations, left_pad_prompts, load_dep_prompts,
    load_paired_dataset, load_sleeper_model, prompt_mask_from_markers,
)
from sleeper.sae import encode_all, load as sae_load

LN1_HOOK = "blocks.0.ln1.hook_normalized"
PAT_HOOK = "blocks.0.attn.hook_pattern"


@torch.no_grad()
def _build_attr_state(model, sae_ln1, sae_mid, target_feat, sel_split, sel_pmask, device):
    """Run OV attribution once per (sae_seed, target_feature)."""
    acts = cache_activations(model, sel_split.tokens, [PAT_HOOK, LN1_HOOK])
    A = acts[PAT_HOOK].to(device)
    z = encode_all(sae_ln1, acts[LN1_HOOK]).to(device)
    d = sae_mid.W_enc[:, target_feat].detach().to(device).float()
    ovw = compute_ov_weights(model, sae_ln1, d, block=0)
    out = ov_attribution(A, z, ovw["beta"])
    ranked = rank_dep_vs_clean(out["contrib"], sel_split.is_deployment.to(device),
                               query_mask=sel_pmask.to(device))
    per_head_lambda = (ranked["per_pair_dep"] - ranked["per_pair_cln"])  # (n_heads, d_sae)
    return {
        "z": z,                        # for analytic delta on sel split (unused — we re-encode dep below)
        "feature_score": ranked["score"].cpu(),                            # (d_sae,)
        "feature_order": torch.argsort(ranked["score"], descending=True).cpu().tolist(),
        "per_head_lambda": per_head_lambda,                                # (n_heads, d_sae)
    }


def _top_k_heads(per_head_lambda, feature_idx, k):
    head_scores = per_head_lambda[:, feature_idx]                          # (n_heads,)
    return torch.topk(head_scores, k, largest=True).indices.cpu().tolist()


@torch.no_grad()
def _delta(model, sae_ln1, tokens, pmask):
    """Build per-feature delta lazily; returns a function f -> (B, P, d_model)."""
    def _make(f: int) -> torch.Tensor:
        return compute_sae_delta(model, sae_ln1, LN1_HOOK, f, tokens, pmask, pmask)
    return _make


@torch.no_grad()
def _screen(model, dep, dep_pmask, sae_ln1, features, alphas, W_V,
            head_topk_per_feat, base_logp, device):
    rows = []
    mk_delta = _delta(model, sae_ln1, dep, dep_pmask)
    for ti, f in enumerate(features):
        delta = mk_delta(f).to(device)
        heads = head_topk_per_feat[ti]
        for alpha in alphas:
            # compute_sae_delta returns x_hat_abl − x_hat_orig = −feature contribution,
            # so a *positive* α here means "subtract α copies of the feature contribution"
            # (matches matrix_sweep / ov_only_steer_hook semantics).
            hooks = head_selective_v_hook(delta, alpha=alpha, W_V=W_V,
                                          head_indices=heads, block=0)
            logp = teacher_forced_sleeper_logp(model, model.tokenizer, dep,
                                               fwd_hooks=hooks).mean().item()
            rows.append({"ti": ti, "f": f, "alpha": alpha, "heads": heads,
                         "dlogp": logp - base_logp})
    return rows


@torch.no_grad()
def _stage2(model, sae_ln1, candidates, dep_lp, dep_attn, cln, cln_marker,
            W_V, gen_tokens, base_ce, device):
    """Greedy ASR + Δcln-CE for each (feat, α, heads) candidate."""
    cln_pmask = prompt_mask_from_markers(cln.shape[1], cln_marker.cpu()).to(device)
    out = []
    for c in candidates:
        f, alpha, heads, ti = c["f"], c["alpha"], c["heads"], c["ti"]
        d_dep = compute_sae_delta(model, sae_ln1, LN1_HOOK, f, dep_lp, dep_attn, dep_attn).to(device)
        h_dep = head_selective_v_hook(d_dep, alpha=alpha, W_V=W_V, head_indices=heads, block=0)
        gen   = greedy_generate_with_hooks(model, dep_lp, h_dep, gen_tokens,
                                           attention_mask=dep_attn)
        asr   = asr_16(gen, model.tokenizer)
        d_cln = compute_sae_delta(model, sae_ln1, LN1_HOOK, f, cln, cln_pmask).to(device)
        h_cln = head_selective_v_hook(d_cln, alpha=alpha, W_V=W_V, head_indices=heads, block=0)
        ce    = clean_continuation_ce(model, cln, cln_marker, fwd_hooks=h_cln).mean().item()
        out.append({"ti": ti, "f": f, "alpha": alpha, "heads": heads,
                    "asr": asr, "dce": ce - base_ce})
    return out


@torch.no_grad()
def _eval_winner(model, sae_ln1, w, W_V,
                 eval_dep, eval_dep_pmask,
                 eval_dep_lp, eval_dep_attn,
                 eval_cln, eval_cln_marker,
                 eval_gen_dep, eval_gen_attn, clean_rollouts,
                 base_logp, base_ce, gen_tokens, eval_seeds, eval_temp, device):
    f, alpha, heads = w["f"], w["alpha"], w["heads"]
    # teacher-forced
    d_dep = compute_sae_delta(model, sae_ln1, LN1_HOOK, f, eval_dep, eval_dep_pmask).to(device)
    h_dep = head_selective_v_hook(d_dep, alpha=alpha, W_V=W_V, head_indices=heads, block=0)
    e_logp = teacher_forced_sleeper_logp(model, model.tokenizer, eval_dep,
                                         fwd_hooks=h_dep).mean().item()
    cln_pmask = prompt_mask_from_markers(eval_cln.shape[1], eval_cln_marker.cpu()).to(device)
    d_cln = compute_sae_delta(model, sae_ln1, LN1_HOOK, f, eval_cln, cln_pmask).to(device)
    h_cln = head_selective_v_hook(d_cln, alpha=alpha, W_V=W_V, head_indices=heads, block=0)
    e_ce  = clean_continuation_ce(model, eval_cln, eval_cln_marker,
                                  fwd_hooks=h_cln).mean().item()
    # Shared loop for ASR + gen-CE-ratio + severity-ratio.
    d_lp  = compute_sae_delta(model, sae_ln1, LN1_HOOK, f, eval_dep_lp, eval_dep_attn,
                               eval_dep_attn).to(device)
    h_lp  = head_selective_v_hook(d_lp, alpha=alpha, W_V=W_V, head_indices=heads, block=0)
    n_gen_ce = eval_gen_dep.shape[0]
    asrs = []
    gen_num_sum = gen_den_sum = 0.0
    gen_count = 0
    steered_lsm_list: list[torch.Tensor] = []
    for s_idx, s in enumerate(eval_seeds):
        sampler = make_sampling_sampler(temperature=eval_temp, seed=int(s), device=device)
        steered_gen, steered_lsm = generate_with_hooks(
            model, eval_dep_lp, h_lp, gen_tokens, sampler,
            attention_mask=eval_dep_attn, capture_log_softmax=True,
        )
        asrs.append(asr_16(steered_gen, model.tokenizer))
        baseline_tokens = clean_rollouts["tokens"][s_idx][: n_gen_ce]
        r = deployment_generation_ratio(
            model, eval_gen_dep, gen_tokens=gen_tokens,
            attention_mask=eval_gen_attn,
            pre_generated_steered=steered_gen[: n_gen_ce],
            pre_generated_baseline=baseline_tokens,
        )
        gen_num_sum += r["num_sum"]; gen_den_sum += r["den_sum"]; gen_count += r["count"]
        steered_lsm_list.append(steered_lsm[: n_gen_ce])

    e_asr = sum(asrs) / len(asrs)
    e_gen_ce_ratio = gen_num_sum / max(gen_den_sum, 1e-12)
    sev = severity_ratio(clean_rollouts["log_softmax"],
                         torch.stack(steered_lsm_list, dim=0))
    return {
        "asr": e_asr, "asr_per_seed": asrs,
        "delta_logp":      e_logp - base_logp,
        "delta_ce":        e_ce - base_ce,
        "gen_ce_ratio":    e_gen_ce_ratio,
        "gen_ce_num_mean": gen_num_sum / max(gen_count, 1),
        "gen_ce_den_mean": gen_den_sum / max(gen_count, 1),
        "severity_ratio":  sev["ratio"],
        "severity_num":    sev["num"],
        "severity_den":    sev["den"],
    }


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds",            type=int,   nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--sae_mid",          type=Path,  default=Path("weights/sae_resid_mid.pt"))
    p.add_argument("--target_feature",   type=int,   default=579)
    p.add_argument("--head_topk",        type=int,   default=2,
                   help="number of heads to patch per feature (top-K by per-head dep-cln OV).")
    p.add_argument("--top_k",            type=int,   default=20)
    p.add_argument("--alphas",           type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0])
    p.add_argument("--stage2_keep",      type=int,   default=10)
    p.add_argument("--n_sel",            type=int,   default=100)
    p.add_argument("--n_eval",           type=int,   default=50)
    p.add_argument("--n_gen_ce",         type=int,   default=25)
    p.add_argument("--gen_tokens",       type=int,   default=16)
    p.add_argument("--eval_seeds",       type=int,   nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--eval_temperature", type=float, default=1.0)
    p.add_argument("--out",              type=Path,  default=Path("results/head_topk_ov_sweep.json"))
    p.add_argument("--device",           default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    sae_mid, _ = sae_load(args.sae_mid, device=device)
    model      = load_sleeper_model(device=device)
    tok        = model.tokenizer
    pad_id     = tok.pad_token_id or tok.eos_token_id
    W_V        = model.W_V[0].detach().to(device)

    splits     = load_paired_dataset(tok, n_train=2, n_val=args.n_sel,
                                     n_test=args.n_eval, seq_len=128, seed=0)
    sel, ev    = splits["val"], splits["test"]
    sel_pmask  = prompt_mask_from_markers(128, sel.story_marker_pos)
    eval_pmask = prompt_mask_from_markers(128, ev.story_marker_pos)

    sel_dep        = sel.tokens[sel.is_deployment].to(device)
    sel_dep_pmask  = sel_pmask[sel.is_deployment].to(device)
    sel_cln        = sel.tokens[~sel.is_deployment].to(device)
    sel_cln_marker = sel.story_marker_pos[~sel.is_deployment].to(device)
    eval_dep         = ev.tokens[ev.is_deployment].to(device)
    eval_dep_pmask   = eval_pmask[ev.is_deployment].to(device)
    eval_cln         = ev.tokens[~ev.is_deployment].to(device)
    eval_cln_marker  = ev.story_marker_pos[~ev.is_deployment].to(device)

    raw_dep   = load_dep_prompts(tok, args.n_sel // 2 + args.n_eval // 2, split="test")
    n_sel_dep = args.n_sel // 2
    sel_lp,  sel_attn  = left_pad_prompts(raw_dep[:n_sel_dep], pad_id)
    eval_lp, eval_attn = left_pad_prompts(raw_dep[n_sel_dep : n_sel_dep + args.n_eval // 2], pad_id)
    sel_lp,  sel_attn  = sel_lp.to(device),  sel_attn.to(device)
    eval_lp, eval_attn = eval_lp.to(device), eval_attn.to(device)
    eval_gen_dep  = eval_lp[: args.n_gen_ce]
    eval_gen_attn = eval_attn[: args.n_gen_ce]

    sel_base_logp  = teacher_forced_sleeper_logp(model, tok, sel_dep).mean().item()
    sel_base_ce    = clean_continuation_ce(model, sel_cln, sel_cln_marker).mean().item()
    eval_base_logp = teacher_forced_sleeper_logp(model, tok, eval_dep).mean().item()
    eval_base_ce   = clean_continuation_ce(model, eval_cln, eval_cln_marker).mean().item()
    base_asrs = []
    for s in args.eval_seeds:
        sampler = make_sampling_sampler(temperature=args.eval_temperature, seed=int(s), device=device)
        gen = generate_with_hooks(model, eval_lp, [], args.gen_tokens, sampler,
                                  attention_mask=eval_attn)
        base_asrs.append(asr_16(gen, tok))
    eval_base_asr = sum(base_asrs) / len(base_asrs)
    print(f"[hk] sel  baseline: dep_logp={sel_base_logp:.3f}  cln_CE={sel_base_ce:.4f}")
    print(f"[hk] eval baseline: dep_logp={eval_base_logp:.3f}  cln_CE={eval_base_ce:.4f}  "
          f"asr={eval_base_asr:.3f} (sampled, seeds={args.eval_seeds}, T={args.eval_temperature})")
    print(f"[hk] head_topk={args.head_topk}  alphas={args.alphas}  features={args.top_k}")

    print(f"[hk] pre-generating clean rollouts (B={args.n_gen_ce}, "
          f"S={len(args.eval_seeds)})...")
    clean_rollouts = pregen_clean_rollouts(
        model, eval_gen_dep, eval_gen_attn,
        args.gen_tokens, args.eval_seeds, args.eval_temperature, device,
    )

    all_results = []
    for sae_seed in args.seeds:
        sae_ln1, _ = sae_load(Path(f"weights/seeds/sae_ln1_s{sae_seed}.pt"), device=device)
        st = _build_attr_state(model, sae_ln1, sae_mid, args.target_feature,
                               sel, sel_pmask, device)
        feats = st["feature_order"][: args.top_k]
        head_topk = [_top_k_heads(st["per_head_lambda"], f, args.head_topk) for f in feats]
        print(f"\n[hk] ── seed={sae_seed} ──")
        print(f"[hk] top-1 feature f{feats[0]} → top-{args.head_topk} heads {head_topk[0]}")

        scr = _screen(model, sel_dep, sel_dep_pmask, sae_ln1, feats, args.alphas,
                      W_V, head_topk, sel_base_logp, device)
        scr.sort(key=lambda r: r["dlogp"])
        cand = scr[: args.stage2_keep]

        ev_sel = _stage2(model, sae_ln1, cand, sel_lp, sel_attn, sel_cln, sel_cln_marker,
                         W_V, args.gen_tokens, sel_base_ce, device)
        asr0 = [r for r in ev_sel if r["asr"] == 0.0]
        winner = (min(asr0, key=lambda r: r["dce"]) if asr0
                  else min(ev_sel, key=lambda r: r["asr"]))
        sel_dlogp = next(r["dlogp"] for r in scr
                         if r["ti"] == winner["ti"] and r["alpha"] == winner["alpha"])

        ev_m = _eval_winner(model, sae_ln1, winner, W_V,
                            eval_dep, eval_dep_pmask, eval_lp, eval_attn,
                            eval_cln, eval_cln_marker, eval_gen_dep, eval_gen_attn,
                            clean_rollouts,
                            eval_base_logp, eval_base_ce, args.gen_tokens,
                            args.eval_seeds, args.eval_temperature, device)

        print(f"[hk]   winner: f{winner['f']} α={winner['alpha']} heads={winner['heads']}")
        print(f"[hk]   sel(asr={winner['asr']:.3f} Δlogp={sel_dlogp:+.3f} ΔCE={winner['dce']:+.4f})")
        print(f"[hk]   eval(asr={ev_m['asr']:.3f} Δlogp={ev_m['delta_logp']:+.3f} "
              f"ΔCE={ev_m['delta_ce']:+.4f} gen-CE-ratio={ev_m['gen_ce_ratio']:.3f} "
              f"sev={ev_m['severity_ratio']:.3f})")

        all_results.append({
            "sae_seed": sae_seed,
            "head_topk": args.head_topk,
            "winner": {"f": winner["f"], "alpha": winner["alpha"], "heads": winner["heads"]},
            "selection": {"asr": winner["asr"], "delta_logp": sel_dlogp, "delta_ce": winner["dce"]},
            "eval": ev_m,
            "screen": scr, "stage2": ev_sel,
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "config": vars(args) | {"seeds": args.seeds, "eval_seeds": args.eval_seeds},
        "decoding": {"selection_asr":       {"mode": "greedy"},
                     "eval_asr":            {"mode": "sample", "temperature": args.eval_temperature,
                                             "seeds": args.eval_seeds},
                     "eval_gen_ce_ratio":   {"mode": "sample", "temperature": args.eval_temperature,
                                             "seeds": args.eval_seeds},
                     "eval_severity_ratio": {"mode": "sample", "temperature": args.eval_temperature,
                                             "seeds": args.eval_seeds}},
        "baseline": {"selection": {"dep_logp": sel_base_logp, "clean_ce": sel_base_ce},
                     "eval":      {"dep_logp": eval_base_logp, "clean_ce": eval_base_ce,
                                   "asr": eval_base_asr, "asr_per_seed": base_asrs}},
        "results": all_results,
    }, indent=2, default=str))
    print(f"\n[hk] wrote {args.out}")


if __name__ == "__main__":
    main()
