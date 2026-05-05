"""Robustness of OV attribution across upstream SAE seeds.

For each seed we:
  1. Train (or load) sae_ln1_s{seed}.pt at blocks.0.ln1.hook_normalized
  2. Run OV attribution with a fixed target: sae_mid.W_enc[:, target_feature]
  3. Rank ln1 features by dep-vs-clean OV contribution → top-k
  4. Pick best candidate = top-1 OV-ranked feature.
  5. Validate by running OV-only ablation of that candidate and measuring
     ASR reduction on dep prompts and ΔCE on clean prompts.
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
    cache_activations, load_paired_dataset, load_sleeper_model,
    prompt_mask_from_markers,
)
from sleeper.sae import TopKSAE, encode_all, load, save, train


def _ln1_path(seed: int) -> Path:
    return Path(f"weights/seeds/sae_ln1_s{seed}.pt")


def _train_or_load(seed: int, model, device: str) -> tuple[TopKSAE, str]:
    path = _ln1_path(seed)
    ln1_hook = "blocks.0.ln1.hook_normalized"
    if path.exists():
        sae, _ = load(path, device=device)
        print(f"[rob] seed={seed}: loaded {path}")
        return sae, ln1_hook
    print(f"[rob] seed={seed}: training SAE …")
    splits = load_paired_dataset(
        tokenizer=model.tokenizer, n_train=10_000, n_val=0, n_test=0,
        seq_len=128, seed=seed,
    )
    with torch.no_grad():
        acts = cache_activations(model, splits["train"].tokens, [ln1_hook],
                                  chunk_size=16)[ln1_hook]
    with torch.enable_grad():
        sae, _ = train(acts, d_sae=1536, k=32, n_steps=4000, batch_size=4096,
                       lr=5e-4, seed=seed, device=device)
    path.parent.mkdir(parents=True, exist_ok=True)
    save(sae, path, layer_hook=ln1_hook, n_train_seqs=10_000,
         seq_len=128, n_steps=4000, batch_size=4096, lr=5e-4)
    print(f"[rob] seed={seed}: saved {path}")
    return sae, ln1_hook



@torch.no_grad()
def _ov_ablate(model, sae_ln1, ln1_hook, feat, dep, dep_mask, dep_marker,
               cln, cln_marker, W_V, alpha, base_logp, base_ce, gen_tokens=16):
    """OV-only suppression of `feat` at `alpha` using per-token SAE deltas.

    Uses compute_sae_delta so the delta is naturally ~0 on prompts/sequences
    where the feature doesn't fire — gives accurate ΔCE on clean prompts.
    """
    tok = model.tokenizer
    d_dep = compute_sae_delta(model, sae_ln1, ln1_hook, feat, dep, dep_mask)
    logp  = teacher_forced_sleeper_logp(
        model, tok, dep,
        fwd_hooks=ov_only_steer_hook(d_dep, alpha, W_V, block=0),
    ).mean().item()

    cln_mask = (torch.arange(cln.shape[1], device=cln.device).unsqueeze(0)
                <= cln_marker.unsqueeze(1))
    d_cln = compute_sae_delta(model, sae_ln1, ln1_hook, feat, cln, cln_mask)
    ce = clean_continuation_ce(
        model, cln, cln_marker,
        fwd_hooks=ov_only_steer_hook(d_cln, alpha, W_V, block=0),
    ).mean().item()

    hits, total = 0, 0
    for m in dep_marker.unique().tolist():
        rows = (dep_marker == m).nonzero(as_tuple=True)[0]
        P = int(m) + 1
        d2 = compute_sae_delta(model, sae_ln1, ln1_hook, feat,
                                dep[rows, :P], dep_mask[rows, :P])
        gen = greedy_generate_with_hooks(
            model, dep[rows, :P],
            ov_only_steer_hook(d2, alpha, W_V, block=0), gen_tokens,
        )
        hits += int(round(asr_16(gen, tok) * gen.shape[0]))
        total += gen.shape[0]
    return hits / max(1, total), logp - base_logp, ce - base_ce


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--sae_mid", type=Path, default=Path("weights/sae_resid_mid.pt"))
    p.add_argument("--target_feature", type=int, default=579)
    p.add_argument("--n_attr", type=int, default=200)
    p.add_argument("--n_test", type=int, default=100)
    p.add_argument("--ablate_alpha", type=float, default=2.0)
    p.add_argument("--out", type=Path, default=Path("weights/seed_robustness.json"))
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    sae_mid, _ = load(args.sae_mid, device=device)
    d = sae_mid.W_enc[:, args.target_feature].detach().to(device).float()
    W_V = None  # lazy-init after model load

    model = load_sleeper_model(device=device)
    W_V = model.W_V[0].detach().to(device)

    splits = load_paired_dataset(
        tokenizer=model.tokenizer, n_train=2, n_val=args.n_attr,
        n_test=args.n_test, seq_len=128, seed=0,
    )
    attr, test = splits["val"], splits["test"]
    pmask      = prompt_mask_from_markers(128, attr.story_marker_pos)
    test_pmask = prompt_mask_from_markers(128, test.story_marker_pos)
    pattern_hook = "blocks.0.attn.hook_pattern"
    ln1_hook = "blocks.0.ln1.hook_normalized"

    dep_test  = test.tokens[test.is_deployment].to(device)
    dep_mask  = test_pmask[test.is_deployment].to(device)
    dep_marker= test.story_marker_pos[test.is_deployment].to(device)
    cln_test  = test.tokens[~test.is_deployment].to(device)
    cln_marker= test.story_marker_pos[~test.is_deployment].to(device)
    base_logp = teacher_forced_sleeper_logp(model, model.tokenizer, dep_test).mean().item()
    base_ce   = clean_continuation_ce(model, cln_test, cln_marker).mean().item()

    results = []
    for seed in args.seeds:
        sae_ln1, _ = _train_or_load(seed, model, device)

        ovw    = compute_ov_weights(model, sae_ln1, d, block=0)
        caches = cache_activations(model, attr.tokens, [pattern_hook, ln1_hook])
        A      = caches[pattern_hook].to(device)
        z_ln1  = encode_all(sae_ln1, caches[ln1_hook]).to(device)
        out    = ov_attribution(A, z_ln1, ovw["beta"])
        ranked = rank_dep_vs_clean(out["contrib"], attr.is_deployment.to(device),
                                   query_mask=pmask.to(device))
        ov_order = ranked["top_indices"].cpu()
        ov_score = ranked["score"].cpu()

        topk_feats   = ov_order[:args.top_k].tolist()
        best_in_topk = topk_feats[0]

        asr, dlogp, dce = _ov_ablate(
            model, sae_ln1, ln1_hook, best_in_topk,
            dep_test, dep_mask, dep_marker, cln_test, cln_marker,
            W_V, args.ablate_alpha, base_logp, base_ce,
        )

        row = {
            "seed": seed,
            "best_in_topk": best_in_topk,
            "ablate_alpha": args.ablate_alpha,
            "asr_after_ablate": asr,
            "delta_logp": dlogp,
            "delta_ce": dce,
            "topk_features": topk_feats,
        }
        results.append(row)
        print(f"[rob] seed={seed}  best_in_top{args.top_k}=f{best_in_topk}  "
              f"asr={asr:.2f}  Δlogp={dlogp:+.2f}  ΔCE={dce:+.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"config": vars(args) | {"seeds": args.seeds},
         "baseline": {"dep_logp": base_logp, "clean_ce": base_ce},
         "results": results},
        indent=2, default=str,
    ))
    print(f"[rob] wrote {args.out}")


if __name__ == "__main__":
    main()
