"""FRA eq-22 cosine histograms — clean vs poisoned.

For TinyStories-33M sleeper, layer 0 attention, using the 50k OV SAE
seed 2 (the SAE that lets Fisher rescue the worst Method-A failure).

Decompose X = X' + ε at blocks.0.ln1.hook_normalized, where
    X' = SAE.decode(SAE.encode(X))   (TopK reconstruction)
    ε  = X - X'                       (reconstruction error)

Pre-softmax (4 panels): per (batch, query position q, head h) the
attention score is a length-T vector over key positions. The 4
additive components (biases conventionally carried by the clean side):
    Q_clean = X' W_Q + b_Q          K_clean = X' W_K + b_K
    Q_err   = ε  W_Q                K_err   = ε  W_K
    S_QcKc(q,k,h) = (Q_clean Q_h)(K_clean Q_h)^T / sqrt(d_h)
    S_QcKe        = (Q_clean    )(K_err      ) / sqrt(d_h)
    S_QeKc        = (Q_err      )(K_clean    ) / sqrt(d_h)
    S_QeKe        = (Q_err      )(K_err      ) / sqrt(d_h)
    sum == full pre-softmax score.

Post-softmax (3 panels): use the actual softmax pattern P from the
full score. Per (batch, query q), layer-0 attn_out is a length-d_model
vector. Decompose only the OV side:
    out_clean = Σ_h Σ_k P[h,q,k] · (X'[k] W_OV_h + b_OV_h projected via W_O)
    out_err   = Σ_h Σ_k P[h,q,k] · (ε[k]  W_OV_h)
    out_bias  = b_O                       (sum-of-pattern = 1 absorbs Σ b_OV)
    sum == actual layer-0 attn_out.

For each component we record cos(component, total) — pre-softmax over
the causal key axis, post-softmax over the d_model axis — and emit a
histogram per (split ∈ {clean, poisoned}).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

# Designed to run from the pod layout (/root/fra_proj), where sleeper/ sits
# at the repo root and provides both the model loader and SAE module.
from sleeper.sae import load as sae_load  # noqa: E402
from sleeper.model import (  # noqa: E402
    BASE_MODEL_NAME,
    load_paired_dataset,
    load_sleeper_model,
)


LAYER = 0
LN1_HOOK = f"blocks.{LAYER}.ln1.hook_normalized"
SCORES_HOOK = f"blocks.{LAYER}.attn.hook_attn_scores"
PATTERN_HOOK = f"blocks.{LAYER}.attn.hook_pattern"
ATTN_OUT_HOOK = f"blocks.{LAYER}.hook_attn_out"

HIST_BINS = np.linspace(-1.0, 1.0, 201)
# Projection-coefficient bins are wider: α_i = <a_i,b>/||b||² sums to 1
# across components but individual coefficients can exceed 1 if components
# partially cancel.
PROJ_BINS = np.linspace(-1.5, 2.5, 401)


def projection_along(x: torch.Tensor, y: torch.Tensor, dim: int,
                     y_sqnorm: torch.Tensor | None = None) -> torch.Tensor:
    """α_i = <x, y> / ||y||²  along `dim`. Pass `y_sqnorm` to reuse."""
    num = (x * y).sum(dim=dim)
    if y_sqnorm is None:
        y_sqnorm = (y * y).sum(dim=dim).clamp_min(1e-30)
    return num / y_sqnorm


@torch.no_grad()
def process_split(
    *,
    model,
    sae,
    tokens: torch.Tensor,         # (N, T) long
    attn_mask: torch.Tensor,      # (N, T) bool/int — 1 = real token
    causal_mask_2d: torch.Tensor, # (T, T) bool, True = valid (k ≤ q)
    n_tokens_target: int,
    batch_size: int,
    device: str,
    rng: np.random.Generator,
) -> dict:
    """Returns per-component cosine histograms for one split."""
    N, T = tokens.shape
    H = model.cfg.n_heads
    d_h = model.cfg.d_head
    d_model = model.cfg.d_model
    # GPT-Neo (TinyStories' family) sets use_attn_scale=False, i.e. no
    # 1/sqrt(d_h). For other archs we honour the configured scale.
    if getattr(model.cfg, "use_attn_scale", True):
        sqrt_d = float(d_h) ** 0.5
    else:
        sqrt_d = 1.0

    # Per-head Q/K/V weights live in TL as W_Q[L, H, d_model, d_head] etc.
    W_Q = model.W_Q[LAYER].to(device)           # (H, d_model, d_h)
    W_K = model.W_K[LAYER].to(device)
    W_V = model.W_V[LAYER].to(device)
    W_O = model.W_O[LAYER].to(device)           # (H, d_h, d_model)
    b_Q = model.b_Q[LAYER].to(device)           # (H, d_h)
    b_K = model.b_K[LAYER].to(device)
    b_V = model.b_V[LAYER].to(device)
    b_O = model.b_O[LAYER].to(device)           # (d_model,)

    pre_keys = ["QcKc", "QcKe", "QeKc", "QeKe"]
    post_keys = ["out_clean", "out_err", "out_bias"]
    pre_hist = {k: np.zeros(HIST_BINS.size - 1, dtype=np.int64) for k in pre_keys}
    post_hist = {k: np.zeros(HIST_BINS.size - 1, dtype=np.int64) for k in post_keys}
    pre_proj_hist = {k: np.zeros(PROJ_BINS.size - 1, dtype=np.int64) for k in pre_keys}
    post_proj_hist = {k: np.zeros(PROJ_BINS.size - 1, dtype=np.int64) for k in post_keys}
    pre_count = 0
    post_count = 0

    # Sanity accumulators (kept tiny — first batch only)
    diag = {}

    bidx = 0
    # Counter: (b, q) valid token positions, regardless of head — this is what
    # the user means by "100k tokens".
    while post_count < n_tokens_target and bidx < N:
        b_end = min(N, bidx + batch_size)
        toks = tokens[bidx:b_end].to(device)
        amask = attn_mask[bidx:b_end].to(device).bool()  # (B, T)

        # Forward with caching of the three hooks.
        _, cache = model.run_with_cache(
            toks,
            names_filter=[LN1_HOOK, SCORES_HOOK, PATTERN_HOOK, ATTN_OUT_HOOK],
        )
        X = cache[LN1_HOOK]                          # (B, T, d_model)
        scores = cache[SCORES_HOOK]                  # (B, H, T_q, T_k)
        pattern = cache[PATTERN_HOOK]                # (B, H, T_q, T_k)
        attn_out = cache[ATTN_OUT_HOOK]              # (B, T, d_model)
        del cache

        # SAE forward: X_clean = decode(encode(X))
        Xf = X.reshape(-1, d_model).to(torch.float32)
        z = sae.encode(Xf)
        Xc = sae.decode(z).reshape(X.shape).to(X.dtype)  # (B, T, d_model)
        Xe = X - Xc                                       # (B, T, d_model)

        # Per-head Q, K projections of clean & err.
        #   Q_full[b,t,h,a] = X[b,t,:] @ W_Q[h, :, :] + b_Q[h, :]
        # We construct Q_clean carrying biases, Q_err biases-free.
        # einsum: "btd, hda -> bth a"
        def project(M, W, bias=None):
            out = torch.einsum("btd,hda->btha", M, W)
            if bias is not None:
                out = out + bias  # broadcasts to (B,T,H,d_h)
            return out

        Q_c = project(Xc, W_Q, b_Q)
        K_c = project(Xc, W_K, b_K)
        Q_e = project(Xe, W_Q)
        K_e = project(Xe, W_K)
        # Sanity: Q_c + Q_e should equal Q_full (within fp tolerance)

        # Score components: S_xy[b, h, q, k] = sum_a Q_x[b,q,h,a] K_y[b,k,h,a] / sqrt(d_h)
        def score(Q_, K_):
            return torch.einsum("bqha,bkha->bhqk", Q_, K_) / sqrt_d

        S_QcKc = score(Q_c, K_c)
        S_QcKe = score(Q_c, K_e)
        S_QeKc = score(Q_e, K_c)
        S_QeKe = score(Q_e, K_e)
        S_total_decomp = S_QcKc + S_QcKe + S_QeKc + S_QeKe

        # Sanity check first batch (only on the valid causal entries — TL's
        # `hook_attn_scores` writes -inf above the diagonal and on padded keys).
        if bidx == 0 and "score_resid" not in diag:
            B0 = toks.shape[0]
            cmask0 = causal_mask_2d.to(device)[None, None, :, :]  # (1,1,Tq,Tk)
            amask_k0 = amask[:, None, None, :].expand(B0, H, T, T)
            v = cmask0 & amask_k0
            d = (S_total_decomp - scores)[v]
            diag["score_resid"] = float(d.abs().max().item()) if d.numel() else None

        # ---- Pre-softmax (4 panels) ----
        # Vectors over k axis. Restrict to valid k positions:
        #   - k ≤ q (causal mask)
        #   - attn_mask[b, k] (key must be a real token)
        # We additionally require q to be a real token, q >= 1 (so the
        # score vector has at least 1 element, and at least 2 keys so
        # cosine has variance).
        B = toks.shape[0]
        # (B, T_q, T_k) valid-key mask
        amask_k = amask[:, None, :].expand(B, T, T)              # (B, T_q, T_k)
        cmask = causal_mask_2d.to(device)[None, :, :].expand(B, T, T)
        valid_k = amask_k & cmask                                # (B, T_q, T_k)
        # Per (b, q) need at least 2 valid keys
        valid_qk_count = valid_k.sum(dim=-1)                     # (B, T_q)
        valid_q = amask & (valid_qk_count >= 2)                  # (B, T_q)

        # Zero invalid positions so they don't contribute to dot products.
        valid_k_f = valid_k.to(S_QcKc.dtype)                     # (B, T_q, T_k)
        t_masked = S_total_decomp * valid_k_f[:, None, :, :]      # (B,H,Tq,Tk)
        t_sqnorm = (t_masked * t_masked).sum(dim=-1).clamp_min(1e-30)  # (B,H,Tq)
        t_norm = t_sqnorm.sqrt()
        for k_name, comp in (("QcKc", S_QcKc), ("QcKe", S_QcKe),
                              ("QeKc", S_QeKc), ("QeKe", S_QeKe)):
            c_masked = comp * valid_k_f[:, None, :, :]
            # inner = <c, total> over keys
            inner = (c_masked * t_masked).sum(dim=-1)             # (B,H,Tq)
            cos = inner / (t_norm * c_masked.norm(dim=-1).clamp_min(1e-9))
            proj = inner / t_sqnorm
            for stat, hbins, hdest in (
                (cos, HIST_BINS, pre_hist),
                (proj, PROJ_BINS, pre_proj_hist),
            ):
                v = stat.permute(0, 2, 1)                         # (B,Tq,H)
                mask3 = valid_q[:, :, None].expand_as(v)
                vals = v[mask3].detach().to("cpu").float().numpy()
                h, _ = np.histogram(vals, bins=hbins)
                hdest[k_name] += h
        pre_count += int(valid_q.sum().item()) * H

        # ---- Post-softmax (3 panels) ----
        # Compute per-head V components, apply pattern, project via W_O.
        # V_c[b,k,h,a] = Xc[b,k,:] @ W_V[h,:,:] + b_V[h,:]
        V_c = project(Xc, W_V, b_V)                              # (B, T, H, d_h)
        V_e = project(Xe, W_V)                                   # (B, T, H, d_h)
        # head out per (b,q,h,a):  H_x[b,q,h,a] = sum_k pattern[b,h,q,k] V_x[b,k,h,a]
        H_c = torch.einsum("bhqk,bkha->bqha", pattern, V_c)
        H_e = torch.einsum("bhqk,bkha->bqha", pattern, V_e)
        # Project via W_O: out[b,q,d] = sum_h sum_a H[b,q,h,a] W_O[h,a,d]
        out_clean = torch.einsum("bqha,had->bqd", H_c, W_O)
        out_err = torch.einsum("bqha,had->bqd", H_e, W_O)
        out_bias = b_O.expand(B, T, d_model).contiguous()
        # Sanity: out_clean + out_err + out_bias ≈ attn_out
        if bidx == 0 and "attn_out_resid" not in diag:
            resid = (out_clean + out_err + out_bias - attn_out).abs().max().item()
            diag["attn_out_resid"] = resid

        # Cosines + projection coefficients over d_model axis, per (b, q).
        t_sqnorm2 = (attn_out * attn_out).sum(dim=-1).clamp_min(1e-30)  # (B,T)
        t_norm2 = t_sqnorm2.sqrt()
        for k_name, vec in (("out_clean", out_clean),
                            ("out_err", out_err),
                            ("out_bias", out_bias)):
            inner = (vec * attn_out).sum(dim=-1)                  # (B,T)
            cos = inner / (t_norm2 * vec.norm(dim=-1).clamp_min(1e-9))
            proj = inner / t_sqnorm2
            for stat, hbins, hdest in (
                (cos, HIST_BINS, post_hist),
                (proj, PROJ_BINS, post_proj_hist),
            ):
                vals = stat[amask].detach().to("cpu").float().numpy()
                h, _ = np.histogram(vals, bins=hbins)
                hdest[k_name] += h
        post_count += int(amask.sum().item())

        del X, Xc, Xe, Q_c, Q_e, K_c, K_e, V_c, V_e, H_c, H_e
        del S_QcKc, S_QcKe, S_QeKc, S_QeKe, S_total_decomp, scores, pattern, attn_out

        bidx = b_end

    return {
        "pre_hist":      {k: v.tolist() for k, v in pre_hist.items()},
        "post_hist":     {k: v.tolist() for k, v in post_hist.items()},
        "pre_proj_hist": {k: v.tolist() for k, v in pre_proj_hist.items()},
        "post_proj_hist":{k: v.tolist() for k, v in post_proj_hist.items()},
        "pre_count": int(pre_count),
        "post_count": int(post_count),
        "diag": diag,
        "bins": HIST_BINS.tolist(),
        "proj_bins": PROJ_BINS.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae_path", default="weights/seeds_50k/sae_ln1_s2.pt")
    ap.add_argument("--n_tokens_per_split", type=int, default=100_000)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--data_seed", type=int, default=0)
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[fra-decomp] device={device}", flush=True)

    print("[fra-decomp] loading model + SAE…", flush=True)
    model = load_sleeper_model(device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    sae, sae_meta = sae_load(Path(args.sae_path), device=device)
    print(f"[fra-decomp] sae: d_in={sae.d_in} d_sae={sae.d_sae} k={sae.k}", flush=True)

    # Use the paired-dataset loader. PairedTokens has `tokens` (mixed
    # clean/deploy) and `is_deployment` bool mask we use to split.
    seq_len = 256
    # We want >= ~256k tokens per split (headroom on the 100k target).
    # n_train // 2 prompts each, each `seq_len` tokens long.
    target_prompts_per_split = max(args.n_tokens_per_split // seq_len + 32, 256)
    n_train = 2 * target_prompts_per_split
    print(f"[fra-decomp] loading paired dataset (n_train={n_train}, seq_len={seq_len})…",
          flush=True)
    paired = load_paired_dataset(
        model.tokenizer,
        n_train=n_train, n_val=4, n_test=4,
        seq_len=seq_len, seed=args.data_seed,
    )
    train = paired["train"]
    is_dep = train.is_deployment
    clean_tokens = train.tokens[~is_dep]
    pois_tokens = train.tokens[is_dep]
    print(f"  clean: {tuple(clean_tokens.shape)}   poisoned: {tuple(pois_tokens.shape)}",
          flush=True)
    # All loaded rows are exactly seq_len long (filtered upstream), so the
    # attention mask is all-ones.
    clean_attn = torch.ones_like(clean_tokens, dtype=torch.bool)
    pois_attn = torch.ones_like(pois_tokens, dtype=torch.bool)

    T = clean_tokens.shape[1]
    causal_2d = torch.tril(torch.ones(T, T, dtype=torch.bool))

    rng = np.random.default_rng(args.data_seed)

    out = {
        "config": {
            "model": BASE_MODEL_NAME,
            "sae_path": str(args.sae_path),
            "sae_meta": sae_meta,
            "n_tokens_per_split": args.n_tokens_per_split,
            "layer": LAYER,
            "ln1_hook": LN1_HOOK,
        },
        "splits": {},
    }

    for name, tok, am in [
        ("clean", clean_tokens, clean_attn),
        ("poisoned", pois_tokens, pois_attn),
    ]:
        print(f"\n[fra-decomp] === {name} ===", flush=True)
        res = process_split(
            model=model, sae=sae, tokens=tok, attn_mask=am,
            causal_mask_2d=causal_2d,
            n_tokens_target=args.n_tokens_per_split,
            batch_size=args.batch_size,
            device=device, rng=rng,
        )
        print(f"[fra-decomp] {name}: pre_count={res['pre_count']}  "
              f"post_count={res['post_count']}  diag={res['diag']}", flush=True)
        out["splits"][name] = res

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out))
    print(f"\n[fra-decomp] wrote {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
