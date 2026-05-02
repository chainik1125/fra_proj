"""Train a TopK SAE on one residual hook of the sleeper model.

End-to-end: load model → harvest activations on the train split → train → save.

Examples:
    python -m scripts.train_sae --hook blocks.0.hook_resid_mid --out weights/sae_resid_mid.pt
    python -m scripts.train_sae --hook blocks.0.ln1.hook_normalized --out weights/sae_ln1.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sleeper.model import cache_activations, load_paired_dataset, load_sleeper_model
from sleeper.sae import save, train


def pick_device(explicit: str | None) -> str:
    if explicit:
        return explicit
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hook", required=True, help="TL hook name to train SAE on.")
    p.add_argument("--out", type=Path, required=True, help="Output checkpoint path.")
    p.add_argument("--n_train", type=int, default=10_000)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--d_sae", type=int, default=1536)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--n_steps", type=int, default=8000)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--harvest_chunk_size", type=int, default=16)
    args = p.parse_args()

    device = pick_device(args.device)
    print(f"[train_sae] device={device}  hook={args.hook}")

    model = load_sleeper_model(device=device)
    splits = load_paired_dataset(
        tokenizer=model.tokenizer,
        n_train=args.n_train, n_val=0, n_test=0,
        seq_len=args.seq_len, seed=args.seed,
    )
    train_tokens = splits["train"].tokens
    print(f"[train_sae] harvesting {train_tokens.shape[0]} sequences …")
    acts = cache_activations(
        model=model, tokens=train_tokens,
        hook_names=[args.hook], chunk_size=args.harvest_chunk_size,
    )[args.hook]                                                # (N, T, d) fp16 cpu
    print(f"[train_sae]   acts shape={tuple(acts.shape)} dtype={acts.dtype}")

    sae, _ = train(
        acts, d_sae=args.d_sae, k=args.k,
        n_steps=args.n_steps, batch_size=args.batch_size,
        lr=args.lr, seed=args.seed, device=device,
    )
    save(sae, args.out, layer_hook=args.hook,
         n_train_seqs=int(train_tokens.shape[0]),
         seq_len=args.seq_len, n_steps=args.n_steps,
         batch_size=args.batch_size, lr=args.lr)
    print(f"[train_sae] wrote {args.out}")


if __name__ == "__main__":
    main()
