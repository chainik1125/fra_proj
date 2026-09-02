"""What share of an attention score does one feature pair carry?

This is the quantity that explains why score-space pair ablation is a null on the
sleeper and worked in the toy. A `(q, k)` pre-softmax score decomposes into
`L_0^2` feature-pair terms, so any one pair's share falls as `1/L_0^2`. The toy
ran at `L_0 ~ 4` with a planted pair; the sleeper SAE runs at `L_0 = 32`.

Measured at the decision position (last real token, where the payload is
generated) against the first key position where the key-side feature fires.

Run: python scripts/24_pair_share.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from sleeper.eval import LN1_HOOK
from sleeper.model import cache_activations, load_paired_dataset, load_sleeper_model
from sleeper.sae import encode_all, load as sae_load

SAE_PATH = Path("weights/seeds/sae_ln1_s0.pt")
PAIRS = [(391, 1114), (259, 1337), (1114, 1232)]
N_ROWS = 60
OUT = Path("results/pair_share.json")


@torch.no_grad()
def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    model = load_sleeper_model("tinystories", device="cpu")
    sae, _ = sae_load(SAE_PATH, device="cpu")
    split = load_paired_dataset(model.tokenizer, n_train=0, n_val=0,
                                n_test=200, seq_len=128, seed=0)["test"]
    tokens, is_dep, amask = split.tokens, split.is_deployment, split.attention_mask
    z = encode_all(sae, cache_activations(model, tokens, [LN1_HOOK])[LN1_HOOK])

    W_dec = sae.W_dec.detach().float()
    W_Q, W_K = model.W_Q[0].detach().float(), model.W_K[0].detach().float()
    scale = model.blocks[0].attn.attn_scale
    qi = tokens.shape[1] - 1

    L0 = float((z > 0).float().sum(-1).mean())
    out = {"L0": L0, "pair_terms_per_cell": L0 ** 2,
           "uniform_share_pct": 100.0 / L0 ** 2, "pairs": {}}
    print(f"L0 = {L0:.1f}  ->  ~{L0**2:.0f} pair terms per (q,k) cell; "
          f"uniform share {100/L0**2:.4f}%")

    for lq, lk in PAIRS:
        shares = []
        for b in is_dep.nonzero().flatten().tolist():
            kpos = (z[b, :, lk] > 0).nonzero().flatten()
            if len(kpos) == 0:
                continue
            k = int(kpos[0])
            xq, xk = z[b, qi] @ W_dec, z[b, k] @ W_dec
            full = ((xq @ W_Q) * (xk @ W_K)).sum(-1) / scale          # (n_heads,)
            pair = (z[b, qi, lq] * z[b, k, lk]
                    * ((W_dec[lq] @ W_Q) * (W_dec[lk] @ W_K)).sum(-1) / scale)
            shares.append(float(pair.abs().sum() / full.abs().sum()))
            if len(shares) >= N_ROWS:
                break
        t = torch.tensor(shares)
        rec = {"n": len(shares), "mean_pct": float(t.mean()) * 100,
               "median_pct": float(t.median()) * 100, "max_pct": float(t.max()) * 100,
               "vs_uniform": float(t.mean()) * L0 ** 2}
        out["pairs"][f"{lq},{lk}"] = rec
        print(f"  pair ({lq},{lk}): mean {rec['mean_pct']:.3f}%  median {rec['median_pct']:.3f}%  "
              f"max {rec['max_pct']:.3f}%  = {rec['vs_uniform']:.1f}x uniform  (n={rec['n']})")

    OUT.write_text(json.dumps(out, indent=2))
    print(f"\n  wrote {OUT}")


if __name__ == "__main__":
    main()
