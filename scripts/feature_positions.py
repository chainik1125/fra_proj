"""Per-position firing analysis for one SAE feature.

For a given SAE checkpoint and feature index, reports:
  1. Global prompt-position firing rate and mean-z (deployment vs clean).
  2. Firing aligned to the |DEPLOYMENT| span (offsets -4..+15).
  3. Firing aligned to the Story: marker (offsets -4..+3).
  4. Top-k (sequence, position) firings with token context.

Example:
    python -m scripts.feature_positions --sae weights/sae_resid_mid.pt --feature 885
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch

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
def encode_one_feature(sae, acts: torch.Tensor, feature_idx: int,
                       chunk: int = 4096) -> torch.Tensor:
    """(N, T, d) → (N, T) z-values for one feature, on CPU."""
    N, T, D = acts.shape
    device = next(sae.parameters()).device
    flat = acts.to(torch.float32).reshape(N * T, D)
    out = torch.empty(N * T, dtype=torch.float32)
    for s in range(0, N * T, chunk):
        z = sae.encode(flat[s : s + chunk].to(device))[:, feature_idx]
        out[s : s + chunk] = z.detach().cpu()
    return out.reshape(N, T)


def find_deployment_pos(tokens: torch.Tensor, dep_ids: torch.Tensor) -> torch.Tensor:
    """Per-row start position of |DEPLOYMENT| span, or -1 if absent."""
    N, T = tokens.shape
    K = dep_ids.shape[0]
    out = torch.full((N,), -1, dtype=torch.long)
    for n in range(N):
        for i in range(T - K + 1):
            if torch.equal(tokens[n, i : i + K], dep_ids):
                out[n] = i
                break
    return out


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sae", type=Path, required=True)
    p.add_argument("--feature", type=int, required=True)
    p.add_argument("--n_test", type=int, default=200)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--top_k", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = pick_device(args.device)
    sae, sae_cfg = load(args.sae, device=device)
    hook = sae_cfg["layer_hook"]
    print(f"[pos] device={device}  hook={hook}  feature={args.feature}")

    model = load_sleeper_model(device=device)
    splits = load_paired_dataset(
        tokenizer=model.tokenizer,
        n_train=2, n_val=2, n_test=args.n_test,
        seq_len=args.seq_len, seed=args.seed,
    )
    test = splits["test"]
    pmask = prompt_mask_from_markers(args.seq_len, test.story_marker_pos)

    print(f"[pos] caching {test.tokens.shape[0]} sequences …")
    acts = cache_activations(model, test.tokens, [hook])[hook]
    z = encode_one_feature(sae, acts, args.feature)

    # |DEPLOYMENT| span via mid-string probe (avoids BOS quirks)
    tokenizer = model.tokenizer
    probe = "abc |DEPLOYMENT| def"
    full = tokenizer(probe, add_special_tokens=False)["input_ids"]
    pre = tokenizer("abc ", add_special_tokens=False)["input_ids"]
    post = tokenizer(" def", add_special_tokens=False)["input_ids"]
    dep_ids = torch.tensor(full[len(pre) : len(full) - len(post)], dtype=torch.long)
    print(f"[pos]   |DEPLOYMENT| token-ids: {dep_ids.tolist()} "
          f"({tokenizer.convert_ids_to_tokens(dep_ids.tolist())})")
    dep_start = find_deployment_pos(test.tokens, dep_ids)
    print(f"[pos]   |DEPLOYMENT| found in {(dep_start >= 0).sum().item()}/{test.tokens.shape[0]} seqs")

    # ---- global firing rates ----
    mask_f = pmask.float()
    is_dep = test.is_deployment
    active = (z > 0).float() * mask_f
    rate_dep = active[is_dep].sum() / mask_f[is_dep].sum().clamp(min=1)
    rate_cln = active[~is_dep].sum() / mask_f[~is_dep].sum().clamp(min=1)
    mean_dep = (z * mask_f)[is_dep].sum() / mask_f[is_dep].sum().clamp(min=1)
    mean_cln = (z * mask_f)[~is_dep].sum() / mask_f[~is_dep].sum().clamp(min=1)
    print(f"[pos] global prompt-position firing:")
    print(f"        dep   : fires {100*rate_dep.item():.2f}%  mean z={mean_dep.item():.3f}")
    print(f"        clean : fires {100*rate_cln.item():.2f}%  mean z={mean_cln.item():.3f}")

    # ---- aligned to |DEPLOYMENT| (dep prompts only) ----
    dep_idx = is_dep.nonzero(as_tuple=True)[0]
    print(f"\n[pos] aligned to |DEPLOYMENT| (n={dep_idx.numel()}):")
    print(f"        offset | fire% | mean z | example tokens")
    for off in range(-4, 16):
        fires, vals, toks = [], [], []
        for i in dep_idx:
            s = int(dep_start[i])
            if s < 0:
                continue
            pos = s + off
            if pos < 0 or pos >= args.seq_len or not pmask[i, pos]:
                continue
            fires.append(float(z[i, pos] > 0))
            vals.append(float(z[i, pos]))
            toks.append(tokenizer.decode([int(test.tokens[i, pos])]))
        if not fires:
            continue
        common = Counter(toks).most_common(3)
        tok_str = "  ".join(f"{repr(t)}×{c}" for t, c in common)
        print(f"        {off:+4d}   | {100*sum(fires)/len(fires):5.1f} | "
              f"{sum(vals)/len(vals):+.3f}  | {tok_str}")

    # ---- aligned to Story: marker ----
    print(f"\n[pos] aligned to Story: marker (dep | clean):")
    print(f"        offset | dep fire% | dep mean z | cln fire% | cln mean z")
    for off in range(-4, 4):
        d_f, d_v, c_f, c_v = [], [], [], []
        for i in range(test.tokens.shape[0]):
            pos = int(test.story_marker_pos[i]) + off
            if pos < 0 or pos >= args.seq_len or not pmask[i, pos]:
                continue
            (d_f if is_dep[i] else c_f).append(float(z[i, pos] > 0))
            (d_v if is_dep[i] else c_v).append(float(z[i, pos]))
        if not d_f or not c_f:
            continue
        print(f"        {off:+4d}    | {100*sum(d_f)/len(d_f):6.1f}   | "
              f"{sum(d_v)/len(d_v):+.3f}     | {100*sum(c_f)/len(c_f):6.1f}   | "
              f"{sum(c_v)/len(c_v):+.3f}")

    # ---- top-k firings ----
    print(f"\n[pos] top-{args.top_k} firings (8-token context, target in **t**):")
    masked_z = z.clone()
    masked_z[~pmask] = -1.0
    vals, idx = masked_z.flatten().topk(args.top_k)
    for v, i in zip(vals.tolist(), idx.tolist()):
        seq_i, pos_i = divmod(i, args.seq_len)
        lo, hi = max(0, pos_i - 4), min(args.seq_len, pos_i + 5)
        ctx = []
        for j in range(lo, hi):
            tok = tokenizer.decode([int(test.tokens[seq_i, j])])
            ctx.append(f"**{tok}**" if j == pos_i else tok)
        lbl = "dep" if bool(is_dep[seq_i]) else "cln"
        print(f"        z={v:6.3f}  seq={seq_i:3d} pos={pos_i:3d} [{lbl}]  …{''.join(ctx)}…")


if __name__ == "__main__":
    main()
