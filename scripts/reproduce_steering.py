"""Reproduce the resid_mid SAE-feature steering result.

Procedure:
  1. Encode val activations through the SAE; rank features by mean(dep) - mean(clean).
  2. For top-K features, sweep alpha; record teacher-forced sleeper Δlogp + clean ΔCE.
  3. Pick (f*, α*) minimising Δlogp subject to ΔCE ≤ budget.
  4. Re-evaluate at (f*, α*) on the test split and report ASR_16.

Default hook: blocks.0.hook_resid_mid (Dmitry's setting). Override with --hook.

Example:
    python -m scripts.reproduce_steering --sae weights/sae_resid_mid.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sleeper.hooks import (
    additive_steer_hook,
    compute_sae_delta,
    greedy_generate_with_hooks,
)
from sleeper.metrics import (
    asr_16,
    clean_continuation_ce,
    rank_features_by_dep_clean,
    teacher_forced_sleeper_logp,
)
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
    """(N, T, d) → (N, T, d_sae) on CPU."""
    N, T, D = acts.shape
    device = next(sae.parameters()).device
    flat = acts.reshape(N * T, D)
    out = torch.empty(N * T, sae.d_sae, dtype=torch.float32)
    for s in range(0, N * T, chunk):
        z = sae.encode(flat[s : s + chunk].to(device=device, dtype=torch.float32))
        out[s : s + chunk] = z.detach().cpu()
    return out.reshape(N, T, sae.d_sae)


@torch.no_grad()
def asr_on_dep_prompts(model, sae, hook, feature_idx, alpha, tokens, mask, marker, gen_tokens):
    """Grouped-by-marker ASR: truncate at marker+1, greedy-decode, check phrase."""
    hits, total = 0, 0
    for m_pos in marker.unique().tolist():
        rows = (marker == m_pos).nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            continue
        P = int(m_pos) + 1
        trunc = tokens[rows, :P]
        trunc_mask = mask[rows, :P]
        if alpha == 0.0:
            hooks = []
        else:
            delta = compute_sae_delta(model, sae, hook, feature_idx, trunc, trunc_mask)
            hooks = additive_steer_hook(delta, alpha, hook)
        gen = greedy_generate_with_hooks(model, trunc, hooks, gen_tokens)
        hits += int(round(asr_16(gen, model.tokenizer) * gen.shape[0]))
        total += gen.shape[0]
    return hits / max(1, total)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sae", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("weights/reproduce_steering.json"))
    p.add_argument("--hook", default=None,
                   help="Override SAE's recorded layer_hook (default: read from checkpoint).")
    p.add_argument("--n_train", type=int, default=0,
                   help="Train split is unused here; default skips it.")
    p.add_argument("--n_val", type=int, default=200)
    p.add_argument("--n_test", type=int, default=200)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0])
    p.add_argument("--delta_util", type=float, default=0.05, help="Clean-CE budget (nats).")
    p.add_argument("--gen_tokens", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = pick_device(args.device)
    print(f"[steer] device={device}")
    sae, sae_cfg = load(args.sae, device=device)
    hook = args.hook or sae_cfg["layer_hook"]
    print(f"[steer] sae={args.sae}  hook={hook}  d_sae={sae.d_sae}  k={sae.k}")

    model = load_sleeper_model(device=device)
    splits = load_paired_dataset(
        tokenizer=model.tokenizer, n_train=max(2, args.n_train),
        n_val=args.n_val, n_test=args.n_test,
        seq_len=args.seq_len, seed=args.seed,
    )
    val, test = splits["val"], splits["test"]
    val_mask = prompt_mask_from_markers(args.seq_len, val.story_marker_pos)
    test_mask = prompt_mask_from_markers(args.seq_len, test.story_marker_pos)

    # ---- ranking on val ----
    print(f"[steer] caching val activations on {hook} …")
    val_acts = cache_activations(model, val.tokens, [hook])[hook]
    z_val = encode_all(sae, val_acts)
    ranking = rank_features_by_dep_clean(z_val, val.is_deployment, val_mask, top_k=args.top_k)
    top = ranking["top_indices"].tolist()
    print(f"[steer]   top-{args.top_k} features: {top[:5]} … (top_score={ranking['scores'][top[0]]:.3f})")

    # ---- baselines on val ----
    val_dep = val.tokens[val.is_deployment].to(device)
    val_dep_mask = val_mask[val.is_deployment].to(device)
    val_cln = val.tokens[~val.is_deployment].to(device)
    val_cln_mask = val_mask[~val.is_deployment].to(device)
    val_cln_marker = val.story_marker_pos[~val.is_deployment].to(device)

    base_logp = teacher_forced_sleeper_logp(model, model.tokenizer, val_dep).mean().item()
    base_ce = clean_continuation_ce(model, val_cln, val_cln_marker).mean().item()
    print(f"[steer]   baseline val: dep_logp={base_logp:.3f}  clean_ce={base_ce:.4f}")

    # ---- stage-1 sweep ----
    print(f"[steer] stage-1: {len(top)} features × {len(args.alphas)} alphas")
    candidates = []
    for f in top:
        d_dep = compute_sae_delta(model, sae, hook, f, val_dep, val_dep_mask)
        d_cln = compute_sae_delta(model, sae, hook, f, val_cln, val_cln_mask)
        for a in args.alphas:
            logp = teacher_forced_sleeper_logp(
                model, model.tokenizer, val_dep,
                fwd_hooks=additive_steer_hook(d_dep, a, hook),
            ).mean().item()
            ce = clean_continuation_ce(
                model, val_cln, val_cln_marker,
                fwd_hooks=additive_steer_hook(d_cln, a, hook),
            ).mean().item()
            candidates.append({
                "feature_idx": int(f), "alpha": a,
                "delta_logp": logp - base_logp,
                "delta_ce": ce - base_ce,
            })

    feasible = [r for r in candidates if r["delta_ce"] <= args.delta_util]
    if not feasible:
        print(f"[steer]   NO feasible (f, α) at ΔCE ≤ {args.delta_util}; using best Δlogp")
        feasible = candidates
    best = min(feasible, key=lambda r: r["delta_logp"])
    print(f"[steer]   chosen: f={best['feature_idx']} α={best['alpha']} "
          f"Δlogp={best['delta_logp']:+.3f} ΔCE={best['delta_ce']:+.4f}")

    # ---- test-set eval ----
    test_dep = test.tokens[test.is_deployment].to(device)
    test_dep_mask = test_mask[test.is_deployment].to(device)
    test_dep_marker = test.story_marker_pos[test.is_deployment].to(device)
    test_cln = test.tokens[~test.is_deployment].to(device)
    test_cln_mask = test_mask[~test.is_deployment].to(device)
    test_cln_marker = test.story_marker_pos[~test.is_deployment].to(device)

    f_star, a_star = best["feature_idx"], best["alpha"]
    d_dep = compute_sae_delta(model, sae, hook, f_star, test_dep, test_dep_mask)
    d_cln = compute_sae_delta(model, sae, hook, f_star, test_cln, test_cln_mask)
    test_logp = teacher_forced_sleeper_logp(
        model, model.tokenizer, test_dep,
        fwd_hooks=additive_steer_hook(d_dep, a_star, hook),
    ).mean().item()
    test_ce = clean_continuation_ce(
        model, test_cln, test_cln_marker,
        fwd_hooks=additive_steer_hook(d_cln, a_star, hook),
    ).mean().item()
    base_test_logp = teacher_forced_sleeper_logp(model, model.tokenizer, test_dep).mean().item()
    base_test_ce = clean_continuation_ce(model, test_cln, test_cln_marker).mean().item()
    test_asr = asr_on_dep_prompts(model, sae, hook, f_star, a_star,
                                  test_dep, test_dep_mask, test_dep_marker, args.gen_tokens)
    base_test_asr = asr_on_dep_prompts(model, sae, hook, f_star, 0.0,
                                       test_dep, test_dep_mask, test_dep_marker, args.gen_tokens)

    out = {
        "hook": hook,
        "feature_idx": int(f_star),
        "alpha": float(a_star),
        "val": {"baseline_logp": base_logp, "baseline_ce": base_ce, **best},
        "test": {
            "baseline_dep_logp": base_test_logp,
            "baseline_clean_ce": base_test_ce,
            "baseline_asr_16": base_test_asr,
            "dep_logp": test_logp,
            "clean_ce": test_ce,
            "asr_16": test_asr,
            "delta_dep_logp": test_logp - base_test_logp,
            "delta_clean_ce": test_ce - base_test_ce,
        },
        "candidates": candidates,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"[steer] test asr_16={test_asr:.3f} (base {base_test_asr:.3f}) "
          f"Δlogp={test_logp-base_test_logp:+.3f}  ΔCE={test_ce-base_test_ce:+.4f}")
    print(f"[steer] wrote {args.out}")


if __name__ == "__main__":
    main()
