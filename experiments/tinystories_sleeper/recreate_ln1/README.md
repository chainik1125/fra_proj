# LN1 variant — SAEs only

Same benchmark as `../recreate/` but:

1. **Only TopK SAEs** (no MLC, no TXC).
2. **Hookpoints are `blocks.{0..3}.ln1.hook_normalized`** (output of the
   pre-attention LayerNorm) instead of `blocks.{0..3}.hook_resid_post`.

Four SAEs are trained, one per layer. Everything else — optimizer,
sparsity, dataset, selection protocol, utility budget — matches the main
experiment so the results are directly comparable.

## One command (remote A40 + pull back)

```bash
./reproduce.sh
```

Same semantics as `../recreate/reproduce.sh`: syncs code to `a40_climb`,
runs the full pipeline on the remote A40, then rsyncs `results/` back
(excluding the 10 GB activations cache).

## Local reproduction

```bash
uv sync
python reproduce.py
```

Outputs go to `./results/`. Full run ~45 min on an A40 (faster than the
main experiment because we skip MLC/TXC training and sweeping).

## What changes vs `../recreate/`

| Dimension | Main experiment | This variant |
|---|---|---|
| Hookpoints | 5 residual-stream (`resid_pre_0` + `resid_post_{0..3}`) | 4 pre-attention LN (`ln1.hook_normalized_{0..3}`) |
| Architectures trained | MLC, 3× TXC, 4× SAE (8 total) | 4× SAE |
| Best feature (expected) | SAE layer 0 feature 831 @ α=1.5 | TBD — first empirical check of whether the LN-normalised signal carries the same trigger structure |

## Re-using the main experiment's activation cache

Not possible — the hookpoints differ, so the `activations_cache.pt` has a
completely different layer-slice layout. This variant writes a fresh
activations cache at the ln1 hookpoints into `./results/`.
