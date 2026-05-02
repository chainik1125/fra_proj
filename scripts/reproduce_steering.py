"""Reproduce the resid_mid SAE-feature steering result.

Matches Dmitry's two-stage pipeline (origin/dmitry/ov,
experiments/tinystories_sleeper/run_ablation_sweep.py): rank top-K val features
by mean(dep)-mean(clean); stage-1 = teacher-forced Δlogp + ΔCE for top-K × αs;
stage-2 = sampled val ASR_16 for stage2_keep best by dep_logp; pick min ASR
under ΔCE ≤ budget; report on test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sleeper.hooks import (
    additive_steer_hook, compute_sae_delta, greedy_generate_with_hooks,
)
from sleeper.metrics import (
    asr_16, clean_continuation_ce,
    rank_features_by_dep_clean, teacher_forced_sleeper_logp,
)
from sleeper.model import (
    cache_activations, load_paired_dataset,
    load_sleeper_model, prompt_mask_from_markers,
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
def asr_on_dep_prompts(model, sae, hook, f, alpha, tokens, mask, marker, gen_tokens):
    hits, total = 0, 0
    for m_pos in marker.unique().tolist():
        rows = (marker == m_pos).nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            continue
        P = int(m_pos) + 1
        trunc, trunc_mask = tokens[rows, :P], mask[rows, :P]
        if alpha == 0.0:
            hooks = []
        else:
            delta = compute_sae_delta(model, sae, hook, f, trunc, trunc_mask)
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
    p.add_argument("--hook", default=None)
    p.add_argument("--n_val", type=int, default=200)
    p.add_argument("--n_test", type=int, default=200)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--top_k", type=int, default=100)
    p.add_argument("--stage2_keep", type=int, default=10)
    p.add_argument("--alphas", type=float, nargs="+",
                   default=[0.25, 0.5, 1.0, 1.5, 2.0])
    p.add_argument("--delta_util", type=float, default=0.05)
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
        tokenizer=model.tokenizer, n_train=2,
        n_val=args.n_val, n_test=args.n_test,
        seq_len=args.seq_len, seed=args.seed,
    )
    val, test = splits["val"], splits["test"]
    val_mask = prompt_mask_from_markers(args.seq_len, val.story_marker_pos)
    test_mask = prompt_mask_from_markers(args.seq_len, test.story_marker_pos)

    print(f"[steer] caching val activations on {hook} …")
    val_acts = cache_activations(model, val.tokens, [hook])[hook]
    z_val = encode_all(sae, val_acts)
    ranking = rank_features_by_dep_clean(z_val, val.is_deployment, val_mask, top_k=args.top_k)
    top = ranking["top_indices"].tolist()
    print(f"[steer]   top-{args.top_k}: {top[:5]} … (top_score={ranking['scores'][top[0]]:.3f})")

    def _split(s, m):
        d = s.is_deployment
        return (s.tokens[d].to(device), m[d].to(device), s.story_marker_pos[d].to(device),
                s.tokens[~d].to(device), m[~d].to(device), s.story_marker_pos[~d].to(device))
    val_dep, val_dep_mask, val_dep_marker, val_cln, val_cln_mask, val_cln_marker = _split(val, val_mask)

    base_logp = teacher_forced_sleeper_logp(model, model.tokenizer, val_dep).mean().item()
    base_ce = clean_continuation_ce(model, val_cln, val_cln_marker).mean().item()
    print(f"[steer]   baseline val: dep_logp={base_logp:.3f}  clean_ce={base_ce:.4f}")

    # ---- stage-1: Δlogp + ΔCE for all (f, α) ----
    print(f"[steer] stage-1: {len(top)} × {len(args.alphas)}")
    per_feat: list[dict] = []
    for fi, f in enumerate(top):
        d_dep = compute_sae_delta(model, sae, hook, f, val_dep, val_dep_mask)
        d_cln = compute_sae_delta(model, sae, hook, f, val_cln, val_cln_mask)
        rows = []
        for a in args.alphas:
            logp = teacher_forced_sleeper_logp(
                model, model.tokenizer, val_dep,
                fwd_hooks=additive_steer_hook(d_dep, a, hook),
            ).mean().item()
            ce = clean_continuation_ce(
                model, val_cln, val_cln_marker,
                fwd_hooks=additive_steer_hook(d_cln, a, hook),
            ).mean().item()
            rows.append({"alpha": a, "dep_logp": logp,
                         "delta_logp": logp - base_logp, "delta_ce": ce - base_ce})
        per_feat.append({"feature_idx": int(f), "by_alpha": rows})
        if (fi + 1) % 20 == 0 or fi + 1 == len(top):
            print(f"[steer]   stage-1 {fi+1}/{len(top)}")

    # ---- stage-2: sampled val ASR for the top stage2_keep features ----
    def _best_dep_logp(e):
        feas = [r for r in e["by_alpha"] if r["delta_ce"] <= args.delta_util]
        return min(r["dep_logp"] for r in (feas or e["by_alpha"]))
    stage2_features = [e["feature_idx"] for e in
                       sorted(per_feat, key=_best_dep_logp)[: args.stage2_keep]]
    print(f"[steer] stage-2 ASR: {len(stage2_features)} × {len(args.alphas)}")

    stage2_rows: list[dict] = []
    for fi, f in enumerate(stage2_features):
        st1 = next(e for e in per_feat if e["feature_idx"] == f)
        for a in args.alphas:
            asr = asr_on_dep_prompts(model, sae, hook, f, a,
                                     val_dep, val_dep_mask, val_dep_marker, args.gen_tokens)
            r1 = next(r for r in st1["by_alpha"] if r["alpha"] == a)
            stage2_rows.append({"feature_idx": int(f), "alpha": a, "val_asr_16": asr,
                                "delta_logp": r1["delta_logp"], "delta_ce": r1["delta_ce"]})
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"[steer]   stage-2 {fi+1}/{len(stage2_features)} f={f}")

    feasible = [r for r in stage2_rows if r["delta_ce"] <= args.delta_util]
    if not feasible:
        print(f"[steer]   NO feasible at ΔCE ≤ {args.delta_util}; falling back")
        feasible = stage2_rows
    best = min(feasible, key=lambda r: (r["val_asr_16"], r["delta_ce"]))
    print(f"[steer]   chosen: f={best['feature_idx']} α={best['alpha']} "
          f"val_asr={best['val_asr_16']:.3f} Δlogp={best['delta_logp']:+.3f} "
          f"ΔCE={best['delta_ce']:+.4f}")

    # ---- test eval ----
    test_dep, test_dep_mask, test_dep_marker, test_cln, test_cln_mask, test_cln_marker = _split(test, test_mask)
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
        "hook": hook, "feature_idx": int(f_star), "alpha": float(a_star),
        "val": {"baseline_logp": base_logp, "baseline_ce": base_ce, **best},
        "test": {
            "baseline_dep_logp": base_test_logp, "baseline_clean_ce": base_test_ce,
            "baseline_asr_16": base_test_asr,
            "dep_logp": test_logp, "clean_ce": test_ce, "asr_16": test_asr,
            "delta_dep_logp": test_logp - base_test_logp,
            "delta_clean_ce": test_ce - base_test_ce,
        },
        "stage2": stage2_rows, "ranking_top": top,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"[steer] test asr_16={test_asr:.3f} (base {base_test_asr:.3f}) "
          f"Δlogp={test_logp-base_test_logp:+.3f}  ΔCE={test_ce-base_test_ce:+.4f}")
    print(f"[steer] wrote {args.out}")


if __name__ == "__main__":
    main()
