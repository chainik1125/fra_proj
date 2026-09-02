"""Is the query side of the planted edge load-bearing, or decorative?

Gate 2 measured attention only AT lambda* positions. It cannot distinguish
"attend to mu* WHEN lambda* fires" from "attend to mu* ALWAYS" -- and the second
solves the task equally well, because the label away from the query position is
DEFAULT regardless of what the head copied.

Run: python scripts/03_query_side_diagnostic.py
"""

from __future__ import annotations

import torch

from fra.toy.config import ToyConfig
from fra.toy.fra import OracleFRA
from fra.toy.metrics import query_side_contrast
from fra.toy.train import TrainConfig, train


def main() -> None:
    cfg = ToyConfig()
    r = train(cfg, TrainConfig(log=False))
    dgp, model = r.dgp, r.model
    lam, mu = dgp.planted_qk_edge

    print("=" * 78)
    print("QUERY-SIDE CONTRAST -- attention onto the mu* key")
    print("=" * 78)
    for split in ("train", "heldout"):
        b = dgp.sample(1024, split=split)
        c = query_side_contrast(model, b)
        print(f"\n  {split}")
        print(f"    mass on mu* key FROM lambda* position   {c.mass_at_lambda:.4f}")
        print(f"    mass on mu* key FROM other positions    {c.mass_elsewhere:.4f}"
              f"   (n={c.n_elsewhere})")
        print(f"    diffuse-attention baseline              {c.uniform_baseline:.4f}")
        print(f"    ratio (lambda / elsewhere)              {c.ratio:.1f}x")
        print(f"    argmax is the mu* key, at lambda*       "
              f"{c.argmax_is_key_at_lambda*100:6.2f}%")
        print(f"    argmax is the mu* key, elsewhere        "
              f"{c.argmax_is_key_elsewhere*100:6.2f}%")
        verdict = ("CONDITIONAL -- query side does real work"
                   if c.query_side_is_conditional
                   else "INERT -- key-side lookup, lambda* may be decorative")
        print(f"    -> {verdict}")

    # Where does attention go from non-lambda* positions, if not to the key?
    b = dgp.sample(512, split="heldout")
    _, cache = model.run_with_cache(b.tokens, names_filter=["blocks.0.attn.hook_pattern"])
    pattern = cache["blocks.0.attn.hook_pattern"][:, 0]
    batch, seq, _ = pattern.shape
    positions = torch.arange(seq).expand(batch, seq)
    is_lambda = positions == b.query_pos.view(batch, 1)
    elsewhere = (positions >= b.key_pos.view(batch, 1)) & ~is_lambda
    self_mass = pattern.diagonal(dim1=1, dim2=2)
    print(f"\n  Where non-lambda* rows put their mass instead:")
    print(f"    on themselves (diagonal)   {self_mass[elsewhere].mean().item():.4f}")

    # The same question asked of the coupling matrix, independent of data.
    print()
    print("=" * 78)
    print("THE SAME QUESTION ASKED OF G, WITHOUT DATA")
    print("=" * 78)
    G = OracleFRA(
        model, dgp.feature_directions, b.tokens[0], b.features[0]
    ).G
    col = G[:, mu]
    print(f"  G[:, mu*] -- how strongly each query feature couples to the mu* key")
    print(f"    G[lambda*, mu*]          {G[lam, mu]:+.4f}")
    print(f"    mean over other lambda   {col[torch.arange(len(col)) != lam].mean():+.4f}")
    print(f"    max  over other lambda   {col[torch.arange(len(col)) != lam].max():+.4f}")
    print(f"    rank of lambda* in the column: "
          f"{int((col > col[lam]).sum()) + 1} of {len(col)}")
    print("\n  If lambda* dominates this column, the query side is selective.")
    print("  If the whole column is uniformly positive, every feature routes to")
    print("  mu* and the edge is a key-side lookup.")


if __name__ == "__main__":
    main()
