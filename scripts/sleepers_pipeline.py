"""Attribution × intervention pipeline for the TinyStories sleeper.

Selects features by --attr {ov,qk,triple}, intervenes via --intervene {ov,qk,all}
and sweeps α. Channel-routing rule (per-channel, not per-row):

    For each channel c ∈ ACTIVE[--intervene]:
      if any selected (f, c) feature, patch hook_c with those features only;
      else fudge — patch hook_c with ALL selected features.

OV-tagged → V; QK-tagged → Q ∪ K (top-K from each side); Triple-tagged →
(a, Q), (b, K), (c, V) per triplet.

Example:
    python -m scripts.sleepers_pipeline \\
        --sae_ln1 weights/seeds/sae_ln1_s0.pt --sae_mid weights/sae_resid_mid.pt \\
        --target_feature 579 --attr qk --intervene qk --top_k 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sleeper.attribution import compute_ov_weights, ov_attribution, rank_dep_vs_clean
from sleeper.hooks import (
    ACTIVE_CHANNELS,
    build_hooks,
    resolve_channel_deltas,
)
from sleeper.metrics import (
    batched_asr_16,
    clean_continuation_ce,
    teacher_forced_sleeper_logp,
)
from sleeper.model import (
    cache_activations,
    left_pad_prompts,
    load_dep_prompts,
    load_paired_dataset,
    load_sleeper_model,
    prompt_mask_from_markers,
)
from sleeper.qk_attribution import (
    compute_qk_weights,
    qk_attribution,
    select_qk_features,
)
from sleeper.sae import encode_all, load
from sleeper.triple_attribution import (
    compute_triple_prep,
    generate_triplet_candidates,
    rank_triplets,
    triple_eval,
)


def pick_device(explicit):
    return explicit or ("cuda" if torch.cuda.is_available() else
                        ("mps" if torch.backends.mps.is_available() else "cpu"))


# ---------------------------------------------------------------------------
# Per-method attribution
# ---------------------------------------------------------------------------


def _ov_attr(args, model, sae_ln1, ln1_hook, sae_mid, attr, attr_pmask, device,
             cache: dict | None = None):
    d = sae_mid.W_enc[:, args.target_feature].detach().to(device).float()
    ovw = compute_ov_weights(model, sae_ln1, d, block=args.block)
    if cache is None:
        cache = _ensure_caches(args, model, sae_ln1, ln1_hook, attr, device)
    out = ov_attribution(cache["A"], cache["z_ln1"], ovw["beta"])
    ranked = rank_dep_vs_clean(out["contrib"], attr.is_deployment.to(device),
                               query_mask=attr_pmask.to(device))
    score = ranked["score"].cpu()
    order = ranked["top_indices"].cpu().tolist()
    selected = [(int(f), "V") for f in order[: args.top_k]]
    ranked_top = [{
        "feature_idx": int(f), "channel": "V",
        "score_dep_minus_clean": float(score[f]),
        "per_lambda_dep": float(ranked["per_lambda_dep"][f]),
        "per_lambda_cln": float(ranked["per_lambda_cln"][f]),
    } for f in order[: max(args.top_k, 10)]]
    cache["beta"] = ovw["beta"]; cache["d"] = d
    cache["ov_score"] = score.abs()
    return selected, ranked_top, cache


def _qk_attr(args, model, sae_ln1, ln1_hook, sae_mid, attr, attr_pmask, device,
             cache: dict | None = None):
    if cache is None:
        cache = _ensure_caches(args, model, sae_ln1, ln1_hook, attr, device)
    if "beta" not in cache:
        d = sae_mid.W_enc[:, args.target_feature].detach().to(device).float()
        ovw = compute_ov_weights(model, sae_ln1, d, block=args.block)
        cache["beta"] = ovw["beta"]; cache["d"] = d
    qkw = compute_qk_weights(model, sae_ln1, block=args.block)
    aggs = qk_attribution(
        cache["A"], cache["z_ln1"], qkw["F_Q"], qkw["F_K"], cache["beta"], qkw["scale"],
        attr.is_deployment, attr_pmask,
    )
    selected, ranked_top = select_qk_features(aggs, top_k=args.top_k,
                                              score_kind=args.qk_score)
    cache["qk_aggs"] = aggs
    return selected, ranked_top, cache


def _triple_attr(args, model, sae_ln1, ln1_hook, sae_mid, attr, attr_pmask, device):
    cache = _ensure_caches(args, model, sae_ln1, ln1_hook, attr, device)
    qk_selected, qk_ranked_top, cache = _qk_attr(
        args, model, sae_ln1, ln1_hook, sae_mid, attr, attr_pmask, device, cache=cache,
    )
    ov_selected, ov_ranked_top, cache = _ov_attr(
        args, model, sae_ln1, ln1_hook, sae_mid, attr, attr_pmask, device, cache=cache,
    )
    qk_q_score = cache["qk_aggs"]["l1_mean_Q"].cpu()
    qk_k_score = cache["qk_aggs"]["l1_mean_K"].cpu()
    ov_score = cache["ov_score"].cpu()
    triplets = generate_triplet_candidates(
        qk_q_score, qk_k_score, ov_score,
        args.triple_kq, args.triple_kk, args.triple_kv,
    )
    print(f"[pipe] generated {triplets.shape[0]} candidate triplets")

    k_pos_per_b = attr.story_marker_pos.to(device)
    prep = compute_triple_prep(
        model, sae_ln1, cache["beta"], cache["A"], cache["ln1_acts"], cache["z_ln1"],
        k_pos_per_b, block=args.block, device=device,
    )
    agg = triple_eval(prep, triplets, attr.is_deployment, attr_pmask)
    selected_trip, ranked_top = rank_triplets(
        agg, top_k=args.top_k, score_kind=args.triple_score,
    )
    selected: list[tuple[int, str]] = []
    for (mu, nu, lam) in selected_trip:
        selected.extend([(mu, "Q"), (nu, "K"), (lam, "V")])
    return selected, {"triplets": ranked_top, "qk_top": qk_ranked_top, "ov_top": ov_ranked_top}


def _ensure_caches(args, model, sae_ln1, ln1_hook, attr, device) -> dict:
    pattern_hook = f"blocks.{args.block}.attn.hook_pattern"
    print(f"[pipe] caching {pattern_hook} and {ln1_hook} on {attr.tokens.shape[0]} prompts...")
    caches = cache_activations(model, attr.tokens, [pattern_hook, ln1_hook])
    return {
        "A": caches[pattern_hook].to(device),
        "z_ln1": encode_all(sae_ln1, caches[ln1_hook]).to(device),
        "ln1_acts": caches[ln1_hook],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sae_ln1", type=Path, required=True)
    p.add_argument("--sae_mid", type=Path, required=True)
    p.add_argument("--target_feature", type=int, required=True)
    p.add_argument("--block", type=int, default=0)
    p.add_argument("--attr", choices=["ov", "qk", "triple"], default="ov")
    p.add_argument("--intervene", choices=["ov", "qk", "all"], default="ov")
    p.add_argument("--qk_score", choices=["l1_mean", "dep_minus_clean", "l1_dep"],
                   default="l1_mean")
    p.add_argument("--triple_score",
                   choices=["qkv_l1", "qkv_signed", "qkv_dep_minus_clean", "full_v", "phi_v"],
                   default="qkv_l1")
    p.add_argument("--triple_kq", type=int, default=20)
    p.add_argument("--triple_kk", type=int, default=20)
    p.add_argument("--triple_kv", type=int, default=20)
    p.add_argument("--features", nargs="*", type=int, default=None,
                   help="Override: OV → λ list (V); Triple → 3K ints (μ ν λ); "
                        "QK → use --features_q/--features_k.")
    p.add_argument("--features_q", nargs="*", type=int, default=None)
    p.add_argument("--features_k", nargs="*", type=int, default=None)
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0])
    p.add_argument("--n_attr", type=int, default=200)
    p.add_argument("--n_test", type=int, default=200)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--gen_tokens", type=int, default=16)
    p.add_argument("--out", type=Path, default=Path("weights/sleepers_pipeline.json"))
    p.add_argument("--save_attribution", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = pick_device(args.device)
    print(f"[pipe] attr={args.attr}  intervene={args.intervene}  device={device}")

    sae_ln1, ln1_cfg = load(args.sae_ln1, device=device)
    sae_mid, mid_cfg = load(args.sae_mid, device=device)
    ln1_hook = ln1_cfg["layer_hook"]; mid_hook = mid_cfg["layer_hook"]

    model = load_sleeper_model(device=device)
    W = {"Q": model.W_Q[args.block].detach().to(device),
         "K": model.W_K[args.block].detach().to(device),
         "V": model.W_V[args.block].detach().to(device)}

    splits = load_paired_dataset(
        tokenizer=model.tokenizer,
        n_train=2, n_val=args.n_attr, n_test=args.n_test,
        seq_len=args.seq_len, seed=args.seed,
    )
    attr = splits["val"]; test = splits["test"]
    attr_pmask = prompt_mask_from_markers(args.seq_len, attr.story_marker_pos)
    pmask = prompt_mask_from_markers(args.seq_len, test.story_marker_pos)

    explicit = (args.features is not None) or (args.features_q is not None) or (args.features_k is not None)
    if explicit:
        if args.attr == "ov":
            assert args.features is not None, "--features required for OV explicit"
            selected = [(int(f), "V") for f in args.features]
        elif args.attr == "qk":
            fq = args.features_q or []; fk = args.features_k or []
            selected = [(int(f), "Q") for f in fq] + [(int(f), "K") for f in fk]
        else:
            assert args.features is not None and len(args.features) % 3 == 0, \
                "Triple --features must be 3K ints (μ ν λ groups)"
            selected = []
            for i in range(0, len(args.features), 3):
                mu, nu, lam = args.features[i:i+3]
                selected.extend([(int(mu), "Q"), (int(nu), "K"), (int(lam), "V")])
        ranked_top = None
    else:
        if args.attr == "ov":
            selected, ranked_top, _ = _ov_attr(args, model, sae_ln1, ln1_hook,
                                               sae_mid, attr, attr_pmask, device)
        elif args.attr == "qk":
            selected, ranked_top, _ = _qk_attr(args, model, sae_ln1, ln1_hook,
                                               sae_mid, attr, attr_pmask, device)
        else:
            selected, ranked_top = _triple_attr(args, model, sae_ln1, ln1_hook,
                                                sae_mid, attr, attr_pmask, device)

    print(f"[pipe] selected ({len(selected)} feats): {selected}")

    dep = test.tokens[test.is_deployment].to(device)
    dep_mask = pmask[test.is_deployment].to(device)
    cln = test.tokens[~test.is_deployment].to(device)
    cln_mask = pmask[~test.is_deployment].to(device)
    cln_marker = test.story_marker_pos[~test.is_deployment].to(device)

    pad_id = model.tokenizer.pad_token_id or model.tokenizer.eos_token_id
    raw_dep = load_dep_prompts(model.tokenizer, args.n_test, split="test")
    dep_lp, dep_attn = left_pad_prompts(raw_dep, pad_id)
    dep_lp = dep_lp.to(device); dep_attn = dep_attn.to(device)

    base_logp = teacher_forced_sleeper_logp(model, model.tokenizer, dep).mean().item()
    base_ce = clean_continuation_ce(model, cln, cln_marker).mean().item()
    base_asr = batched_asr_16(model, sae_ln1, ln1_hook, selected, 0.0,
                              set(), W, args.block,
                              dep_lp, dep_attn, args.gen_tokens)
    print(f"[pipe] baseline: dep_logp={base_logp:.3f} clean_ce={base_ce:.4f} asr={base_asr:.3f}")

    rows = []
    active = ACTIVE_CHANNELS[args.intervene]
    for a in args.alphas:
        cd_dep = resolve_channel_deltas(selected, active, model, sae_ln1, ln1_hook,
                                        dep, dep_mask)
        cd_cln = resolve_channel_deltas(selected, active, model, sae_ln1, ln1_hook,
                                        cln, cln_mask)
        h_dep = build_hooks(cd_dep, a, active, W, ln1_hook, args.block)
        h_cln = build_hooks(cd_cln, a, active, W, ln1_hook, args.block)
        logp = teacher_forced_sleeper_logp(model, model.tokenizer, dep, fwd_hooks=h_dep).mean().item()
        ce = clean_continuation_ce(model, cln, cln_marker, fwd_hooks=h_cln).mean().item()
        asr = batched_asr_16(model, sae_ln1, ln1_hook, selected, a, active, W, args.block,
                             dep_lp, dep_attn, args.gen_tokens)
        rows.append({"alpha": a, "asr_16": asr,
                     "dep_logp": logp, "delta_logp": logp - base_logp,
                     "clean_ce": ce, "delta_ce": ce - base_ce})
        print(f"[pipe]   α={a:>5}: asr={asr:.3f}  Δlogp={logp-base_logp:+.3f}  "
              f"ΔCE={ce-base_ce:+.4f}")

    out = {
        "config": {
            "attr": args.attr, "intervene": args.intervene,
            "qk_score": args.qk_score if args.attr == "qk" else None,
            "triple_score": args.triple_score if args.attr == "triple" else None,
            "ln1_hook": ln1_hook, "mid_hook": mid_hook, "block": args.block,
            "target_feature": int(args.target_feature),
            "selected": [list(t) if isinstance(t, tuple) else t for t in selected],
            "n_test": int(test.tokens.shape[0]),
            "alphas": args.alphas,
            "explicit_features": explicit,
        },
        "ranked_top": ranked_top,
        "test_baseline": {"dep_logp": base_logp, "clean_ce": base_ce, "asr_16": base_asr},
        "sweep": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=lambda o: list(o)))
    print(f"[pipe] wrote {args.out}")


if __name__ == "__main__":
    main()
