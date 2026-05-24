"""Train a single 50k-step resid_mid TopK SAE at a specified seed.

Output: weights/seeds_50k/sae_resid_mid_s{SEED}.pt
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import torch
from sleeper.model import cache_activations, load_paired_dataset, load_sleeper_model
from sleeper.sae import save, train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    n_train = 10_000
    seq_len = 128
    d_sae = 1536
    k = 32
    n_steps = 50_000
    batch_size = 4096
    lr = 5e-4

    weights = Path("weights")
    seeds_dir = weights / "seeds_50k"
    seeds_dir.mkdir(parents=True, exist_ok=True)
    out_path = seeds_dir / f"sae_resid_mid_s{args.seed}.pt"

    if out_path.exists():
        print(f"[train-resid-50k] skip {out_path} (exists)")
        return

    print(f"[train-resid-50k] device={device}  seed={args.seed}  steps={n_steps}")
    model = load_sleeper_model(device=device)
    splits = load_paired_dataset(
        tokenizer=model.tokenizer,
        n_train=n_train, n_val=0, n_test=0,
        seq_len=seq_len, seed=0,
    )
    train_tokens = splits["train"].tokens

    acts = cache_activations(
        model=model, tokens=train_tokens,
        hook_names=["blocks.0.hook_resid_mid"], chunk_size=16,
    )

    sae, _ = train(
        acts["blocks.0.hook_resid_mid"],
        d_sae=d_sae, k=k, n_steps=n_steps,
        batch_size=batch_size, lr=lr, seed=args.seed, device=device,
    )
    save(sae, out_path, layer_hook="blocks.0.hook_resid_mid",
         n_train_seqs=int(train_tokens.shape[0]),
         seq_len=seq_len, n_steps=n_steps, batch_size=batch_size, lr=lr)
    print(f"[train-resid-50k] wrote {out_path}")


if __name__ == "__main__":
    main()
