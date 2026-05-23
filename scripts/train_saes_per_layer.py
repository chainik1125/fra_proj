"""Train 6-seed SAEs at every (layer × {hook_resid_mid, hook_resid_post}) site
for the per-layer conventional baseline sweep.

Output: weights/seeds_per_layer/sae_L{L}_{hook_kind}_s{seed}.pt

Each GPU process can be restricted to a subset of seeds via --seeds to shard
across multiple GPUs (one harvest per process; cheap on a 4-layer model).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sleeper.model import cache_activations, load_paired_dataset, load_sleeper_model
from sleeper.sae import save, train

HOOK_KINDS = ("hook_resid_mid", "hook_resid_post")


def _path(out_dir: Path, layer: int, hook_kind: str, seed: int) -> Path:
    kind = hook_kind.replace("hook_", "")          # resid_mid / resid_post
    return out_dir / f"sae_L{layer}_{kind}_s{seed}.pt"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--layers", type=int, nargs="+", default=None,
                   help="Layers to train (default: all).")
    p.add_argument("--hooks", nargs="+", default=list(HOOK_KINDS),
                   choices=list(HOOK_KINDS))
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(6)),
                   help="Seeds to train (split across GPUs).")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_train, seq_len, d_sae, k = 10_000, 128, 1_536, 32
    n_steps, batch_size, lr    = 4_000,  4_096, 5e-4

    out_dir = Path("weights/seeds_per_layer")
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_sleeper_model(device=device)
    layers = list(range(model.cfg.n_layers)) if args.layers is None else args.layers

    missing: list[tuple[int, str, int, Path]] = []
    for L in layers:
        for hk in args.hooks:
            for s in args.seeds:
                pth = _path(out_dir, L, hk, s)
                if not pth.exists():
                    missing.append((L, hk, s, pth))

    if not missing:
        print(f"[per-layer] seeds={args.seeds}: all checkpoints present")
        return

    print(f"[per-layer] device={device}  seeds={args.seeds}  "
          f"missing={len(missing)} (out of {len(layers) * len(args.hooks) * len(args.seeds)})")

    needed_hooks = sorted({f"blocks.{L}.{hk}" for L, hk, _, _ in missing})
    splits = load_paired_dataset(model.tokenizer, n_train=n_train, n_val=0, n_test=0,
                                  seq_len=seq_len, seed=0)
    train_tokens = splits["train"].tokens
    print(f"[per-layer] harvest {train_tokens.shape[0]} seqs at {len(needed_hooks)} hooks")
    acts = cache_activations(model=model, tokens=train_tokens,
                              hook_names=needed_hooks, chunk_size=16)

    for L, hk, s, pth in missing:
        hook_name = f"blocks.{L}.{hk}"
        sae, _ = train(acts[hook_name], d_sae=d_sae, k=k, n_steps=n_steps,
                        batch_size=batch_size, lr=lr, seed=s, device=device)
        save(sae, pth, layer_hook=hook_name,
              n_train_seqs=int(train_tokens.shape[0]),
              seq_len=seq_len, n_steps=n_steps, batch_size=batch_size, lr=lr)
        print(f"[per-layer] wrote {pth}")

    print("[per-layer] done")


if __name__ == "__main__":
    main()
