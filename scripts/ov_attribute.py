"""Compute OV attribution to a target direction d at resid_mid.

For each ln1-feature λ, computes Dmitry's:

    C^OV_{h, q, k, λ} = A^h_{q,k} · z_ln1[k, λ] · ⟨W_dec_ln1[λ] W_OV^h, d⟩

with attention pattern A and ln1 SAE codes z_ln1 taken from the unmodified
forward pass. Aggregates and ranks ln1-features by the deployment-vs-clean
diff of their OV contribution to d (Dmitry's "diffing procedure").

Default target direction d = SAE_mid.W_enc[:, --target_feature].

Example:
    python -m scripts.ov_attribute \\
        --sae_ln1 weights/sae_ln1.pt \\
        --sae_mid weights/sae_resid_mid.pt \\
        --target_feature 171 \\
        --out weights/ov_attribution.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sleeper.attribution import compute_ov_weights, ov_attribution, rank_dep_vs_clean
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
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sae_ln1", type=Path, required=True,
                   help="SAE trained on blocks.<block>.ln1.hook_normalized.")
    p.add_argument("--sae_mid", type=Path, required=True,
                   help="SAE trained on blocks.<block>.hook_resid_mid (provides target d).")
    p.add_argument("--target_feature", type=int, required=True,
                   help="Index into SAE_mid; d = SAE_mid.W_enc[:, target_feature].")
    p.add_argument("--block", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("weights/ov_attribution.pt"))
    p.add_argument("--n_test", type=int, default=200)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--top_k", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = pick_device(args.device)
    print(f"[ov-attr] device={device}  block={args.block}")

    sae_ln1, ln1_cfg = load(args.sae_ln1, device=device)
    sae_mid, mid_cfg = load(args.sae_mid, device=device)
    ln1_hook = ln1_cfg["layer_hook"]
    mid_hook = mid_cfg["layer_hook"]
    print(f"[ov-attr] ln1 SAE @ {ln1_hook}  ({sae_ln1.d_sae}x{sae_ln1.k})")
    print(f"[ov-attr] mid SAE @ {mid_hook}  ({sae_mid.d_sae}x{sae_mid.k})")

    model = load_sleeper_model(device=device)
    splits = load_paired_dataset(
        tokenizer=model.tokenizer,
        n_train=2, n_val=2, n_test=args.n_test,
        seq_len=args.seq_len, seed=args.seed,
    )
    test = splits["test"]
    pmask = prompt_mask_from_markers(args.seq_len, test.story_marker_pos)

    # ---- target direction ----
    d = sae_mid.W_enc[:, args.target_feature].detach().to(device).float()
    print(f"[ov-attr] target d = SAE_mid.W_enc[:, {args.target_feature}]  ||d||={d.norm().item():.3f}")

    # ---- precompute β = ⟨W_dec_ln1[λ] W_OV^h, d⟩ ----
    ovw = compute_ov_weights(model, sae_ln1, d, block=args.block)
    beta = ovw["beta"]                # (n_heads, d_sae_ln1)
    print(f"[ov-attr] β shape={tuple(beta.shape)}  |β|_max={beta.abs().max().item():.3e}  "
          f"const={ovw['const']:+.4f}")

    # ---- cache attention pattern + ln1 codes on test ----
    pattern_hook = f"blocks.{args.block}.attn.hook_pattern"
    print(f"[ov-attr] caching {pattern_hook} and {ln1_hook} on {test.tokens.shape[0]} sequences …")
    caches = cache_activations(model, test.tokens, [pattern_hook, ln1_hook])
    A = caches[pattern_hook]                                  # (N, n_heads, T, T) fp16
    ln1_acts = caches[ln1_hook]                               # (N, T, d_model) fp16
    z_ln1 = encode_all(sae_ln1, ln1_acts)                     # (N, T, d_sae_ln1) cpu fp32
    print(f"[ov-attr]   A shape={tuple(A.shape)}  z_ln1 shape={tuple(z_ln1.shape)}")

    # ---- compute attribution ----
    print(f"[ov-attr] computing C^OV[h, q, k, λ] (peak memory ~ B·H·T·d_sae_ln1) …")
    A_dev = A.to(device)
    z_dev = z_ln1.to(device)
    out = ov_attribution(A_dev, z_dev, beta)
    contrib = out["contrib"]                                  # (B, h, T_q, λ)

    ranked = rank_dep_vs_clean(
        contrib, test.is_deployment.to(device),
        query_mask=pmask.to(device),
    )
    score = ranked["score"].cpu()
    top_idx = ranked["top_indices"].cpu()[: args.top_k].tolist()
    top_rows = []
    for f in top_idx:
        top_rows.append({
            "feature_idx": int(f),
            "score_dep_minus_clean": float(score[f]),
            "per_lambda_dep": float(ranked["per_lambda_dep"][f]),
            "per_lambda_cln": float(ranked["per_lambda_cln"][f]),
        })
    for r in top_rows[:10]:
        print(f"[ov-attr]   λ={r['feature_idx']:5d}  "
              f"dep={r['per_lambda_dep']:+.4e}  cln={r['per_lambda_cln']:+.4e}  "
              f"diff={r['score_dep_minus_clean']:+.4e}")

    # ---- persist ----
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "beta": beta.cpu(),                            # (n_heads, d_sae_ln1)
        "u": ovw["u"].cpu(),                           # (n_heads, d_model)
        "const": ovw["const"],
        "score_dep_minus_clean": score,                # (d_sae_ln1,)
        "per_lambda_dep": ranked["per_lambda_dep"].cpu(),
        "per_lambda_cln": ranked["per_lambda_cln"].cpu(),
        "per_pair_dep": ranked["per_pair_dep"].cpu(),  # (n_heads, d_sae_ln1)
        "per_pair_cln": ranked["per_pair_cln"].cpu(),
        "top_indices_by_abs_diff": ranked["top_indices"].cpu(),
        "config": {
            "ln1_hook": ln1_hook,
            "mid_hook": mid_hook,
            "block": args.block,
            "target_feature": int(args.target_feature),
            "n_test": int(test.tokens.shape[0]),
            "seq_len": int(args.seq_len),
        },
    }, args.out)
    print(f"[ov-attr] wrote {args.out}")
    summary_path = args.out.with_suffix(".json")
    summary_path.write_text(json.dumps({
        "config": {
            "ln1_hook": ln1_hook, "mid_hook": mid_hook, "block": args.block,
            "target_feature": int(args.target_feature),
        },
        "top_features": top_rows,
        "const": ovw["const"],
    }, indent=2))
    print(f"[ov-attr] wrote {summary_path}")


if __name__ == "__main__":
    main()
