"""End-to-end OV-only sleeper pipeline: attribute → steer → evaluate.

The main script for task (b). One pass:

  1. Cache attention pattern A and ln1.hook_normalized on test prompts.
  2. Encode ln1 activations through SAE_ln1 → z_ln1.
  3. Pre-compute β[h, λ] = ⟨W_dec_ln1[λ] · W_OV^h, d⟩, with the target
     direction d = SAE_mid.W_enc[:, target_feature].
  4. Evaluate Dmitry's eq:  C^OV[h, q, k, λ] = A[h, q, k] · z_ln1[k, λ] · β[h, λ]
  5. Rank ln1 features by Σ_h mean(C^OV)[dep] − Σ_h mean(C^OV)[clean].
  6. Take top-K and steer them jointly via the OV-only hook
     (W_V-projected delta at attn.hook_v; Q, K untouched ⇒ pattern frozen).
  7. Sweep α; report ASR_16, teacher-forced sleeper Δlogp, clean ΔCE.

Example:
    python -m scripts.sleepers_ov_pipeline \\
        --sae_ln1 weights/sae_ln1.pt \\
        --sae_mid weights/sae_resid_mid.pt \\
        --target_feature 171 \\
        --top_k 3 --alphas 0.5 1 2 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sleeper.attribution import compute_ov_weights, ov_attribution, rank_dep_vs_clean
from sleeper.hooks import compute_sae_delta, greedy_generate_with_hooks, ov_only_steer_hook
from sleeper.metrics import asr_16, clean_continuation_ce, teacher_forced_sleeper_logp
from sleeper.model import (
    cache_activations,
    load_paired_dataset,
    load_sleeper_model,
    prompt_mask_from_markers,
)
from sleeper.sae import load


def pick_device(explicit):
    return explicit or ("cuda" if torch.cuda.is_available() else
                        ("mps" if torch.backends.mps.is_available() else "cpu"))


@torch.no_grad()
def encode_all(sae, acts: torch.Tensor, chunk: int = 256) -> torch.Tensor:
    N, T, D = acts.shape
    device = next(sae.parameters()).device
    flat = acts.reshape(N * T, D)
    out = torch.empty(N * T, sae.d_sae, dtype=torch.float32)
    for s in range(0, N * T, chunk):
        z = sae.encode(flat[s : s + chunk].to(device=device, dtype=torch.float32))
        out[s : s + chunk] = z.detach().cpu()
    return out.reshape(N, T, sae.d_sae)


@torch.no_grad()
def ln1_delta_for_features(model, sae_ln1, ln1_hook, features, tokens, mask):
    """Sum of per-feature SAE-reconstruction deltas at ln1 (jointly zero them)."""
    delta = None
    for f in features:
        d = compute_sae_delta(model, sae_ln1, ln1_hook, f, tokens, mask)
        delta = d if delta is None else delta + d
    return delta


@torch.no_grad()
def asr_with_ov(model, sae_ln1, ln1_hook, features, alpha, W_V, block,
                tokens, mask, marker, gen_tokens):
    hits, total = 0, 0
    for m_pos in marker.unique().tolist():
        rows = (marker == m_pos).nonzero(as_tuple=True)[0]
        P = int(m_pos) + 1
        trunc = tokens[rows, :P]
        trunc_mask = mask[rows, :P]
        delta = ln1_delta_for_features(model, sae_ln1, ln1_hook, features, trunc, trunc_mask)
        hooks = ov_only_steer_hook(delta, alpha, W_V, block=block)
        gen = greedy_generate_with_hooks(model, trunc, hooks, gen_tokens)
        hits += int(round(asr_16(gen, model.tokenizer) * gen.shape[0]))
        total += gen.shape[0]
    return hits / max(1, total)


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sae_ln1", type=Path, required=True)
    p.add_argument("--sae_mid", type=Path, required=True)
    p.add_argument("--target_feature", type=int, required=True,
                   help="d = SAE_mid.W_enc[:, target_feature].")
    p.add_argument("--block", type=int, default=0)
    p.add_argument("--features", nargs="*", type=int, default=None,
                   help="Override: skip ranking and steer these ln1 features jointly.")
    p.add_argument("--top_k", type=int, default=3,
                   help="If --features not given, take this many from the OV ranking.")
    p.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0])
    p.add_argument("--n_attr", type=int, default=200,
                   help="Held-out prompts for OV attribution (val split). Disjoint from --n_test.")
    p.add_argument("--n_test", type=int, default=200)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--gen_tokens", type=int, default=16)
    p.add_argument("--out", type=Path, default=Path("weights/sleepers_ov_pipeline.json"))
    p.add_argument("--save_attribution", type=Path, default=None,
                   help="Optional .pt path to dump full attribution tensors.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = pick_device(args.device)
    print(f"[pipe] device={device}  block={args.block}")

    # ---- load SAEs + model ----
    sae_ln1, ln1_cfg = load(args.sae_ln1, device=device)
    sae_mid, mid_cfg = load(args.sae_mid, device=device)
    ln1_hook = ln1_cfg["layer_hook"]
    mid_hook = mid_cfg["layer_hook"]
    print(f"[pipe] sae_ln1 @ {ln1_hook}  sae_mid @ {mid_hook}")

    model = load_sleeper_model(device=device)
    W_V = model.W_V[args.block].detach().to(device)               # (n_heads, d_model, d_head)

    splits = load_paired_dataset(
        tokenizer=model.tokenizer,
        n_train=2, n_val=args.n_attr, n_test=args.n_test,
        seq_len=args.seq_len, seed=args.seed,
    )
    attr = splits["val"]   # disjoint held-out for attribution
    test = splits["test"]  # held-out for ASR / logp / CE eval
    attr_pmask = prompt_mask_from_markers(args.seq_len, attr.story_marker_pos)
    pmask = prompt_mask_from_markers(args.seq_len, test.story_marker_pos)

    # ---- attribution ----
    d = sae_mid.W_enc[:, args.target_feature].detach().to(device).float()
    print(f"[pipe] target d = SAE_mid.W_enc[:, {args.target_feature}]  ||d||={d.norm().item():.3f}")

    if args.features is not None:
        features = args.features
        ranked_top = []
        attribution_blob = None
        print(f"[pipe] explicit --features given; skipping attribution, using {features}")
    else:
        ovw = compute_ov_weights(model, sae_ln1, d, block=args.block)
        beta = ovw["beta"]
        print(f"[pipe] β shape={tuple(beta.shape)}  |β|_max={beta.abs().max().item():.3e}  "
              f"const={ovw['const']:+.4f}")

        pattern_hook = f"blocks.{args.block}.attn.hook_pattern"
        print(f"[pipe] caching {pattern_hook} and {ln1_hook} on {attr.tokens.shape[0]} attr seqs …")
        caches = cache_activations(model, attr.tokens, [pattern_hook, ln1_hook])
        A = caches[pattern_hook].to(device)
        z_ln1 = encode_all(sae_ln1, caches[ln1_hook]).to(device)

        out = ov_attribution(A, z_ln1, beta)
        ranked = rank_dep_vs_clean(
            out["contrib"], attr.is_deployment.to(device), query_mask=attr_pmask.to(device),
        )
        score = ranked["score"].cpu()
        order = ranked["top_indices"].cpu().tolist()
        features = order[: args.top_k]
        ranked_top = [{
            "feature_idx": int(f),
            "score_dep_minus_clean": float(score[f]),
            "per_lambda_dep": float(ranked["per_lambda_dep"][f]),
            "per_lambda_cln": float(ranked["per_lambda_cln"][f]),
        } for f in order[: max(args.top_k, 10)]]
        for r in ranked_top[:10]:
            print(f"[pipe]   λ={r['feature_idx']:5d}  "
                  f"dep={r['per_lambda_dep']:+.4e}  cln={r['per_lambda_cln']:+.4e}  "
                  f"diff={r['score_dep_minus_clean']:+.4e}")
        print(f"[pipe] top-{args.top_k} ranked features → {features}")

        attribution_blob = {
            "beta": beta.cpu(),
            "u": ovw["u"].cpu(),
            "const": ovw["const"],
            "score_dep_minus_clean": score,
            "per_lambda_dep": ranked["per_lambda_dep"].cpu(),
            "per_lambda_cln": ranked["per_lambda_cln"].cpu(),
            "per_pair_dep": ranked["per_pair_dep"].cpu(),
            "per_pair_cln": ranked["per_pair_cln"].cpu(),
            "top_indices_by_abs_diff": ranked["top_indices"].cpu(),
        }

    if args.save_attribution and attribution_blob is not None:
        args.save_attribution.parent.mkdir(parents=True, exist_ok=True)
        torch.save(attribution_blob, args.save_attribution)
        print(f"[pipe] wrote attribution → {args.save_attribution}")

    # ---- intervention sweep ----
    dep = test.tokens[test.is_deployment].to(device)
    dep_mask = pmask[test.is_deployment].to(device)
    dep_marker = test.story_marker_pos[test.is_deployment].to(device)
    cln = test.tokens[~test.is_deployment].to(device)
    cln_mask = pmask[~test.is_deployment].to(device)
    cln_marker = test.story_marker_pos[~test.is_deployment].to(device)

    base_logp = teacher_forced_sleeper_logp(model, model.tokenizer, dep).mean().item()
    base_ce = clean_continuation_ce(model, cln, cln_marker).mean().item()
    base_asr = asr_with_ov(model, sae_ln1, ln1_hook, features, 0.0,
                           W_V, args.block, dep, dep_mask, dep_marker, args.gen_tokens)
    print(f"[pipe] baseline test: dep_logp={base_logp:.3f}  clean_ce={base_ce:.4f}  asr={base_asr:.3f}")

    rows = []
    for a in args.alphas:
        d_dep = ln1_delta_for_features(model, sae_ln1, ln1_hook, features, dep, dep_mask)
        d_cln = ln1_delta_for_features(model, sae_ln1, ln1_hook, features, cln, cln_mask)
        h_dep = ov_only_steer_hook(d_dep, a, W_V, block=args.block)
        h_cln = ov_only_steer_hook(d_cln, a, W_V, block=args.block)
        logp = teacher_forced_sleeper_logp(model, model.tokenizer, dep, fwd_hooks=h_dep).mean().item()
        ce = clean_continuation_ce(model, cln, cln_marker, fwd_hooks=h_cln).mean().item()
        asr = asr_with_ov(model, sae_ln1, ln1_hook, features, a, W_V, args.block,
                          dep, dep_mask, dep_marker, args.gen_tokens)
        rows.append({"alpha": a, "asr_16": asr,
                     "dep_logp": logp, "delta_logp": logp - base_logp,
                     "clean_ce": ce, "delta_ce": ce - base_ce})
        print(f"[pipe]   α={a:>5}: asr={asr:.3f}  Δlogp={logp-base_logp:+.3f}  ΔCE={ce-base_ce:+.4f}")

    out = {
        "config": {
            "ln1_hook": ln1_hook, "mid_hook": mid_hook, "block": args.block,
            "target_feature": int(args.target_feature),
            "features_steered": list(map(int, features)),
            "n_test": int(test.tokens.shape[0]),
            "alphas": args.alphas,
            "explicit_features": args.features is not None,
        },
        "ranked_top": ranked_top,
        "test_baseline": {"dep_logp": base_logp, "clean_ce": base_ce, "asr_16": base_asr},
        "sweep": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"[pipe] wrote {args.out}")


if __name__ == "__main__":
    main()
