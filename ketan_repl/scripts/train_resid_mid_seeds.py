"""Train 4 additional resid_mid SAEs: 4k @ seeds 1,2 and 50k @ seeds 1,2.

Reuses jamie's harvest pipeline. Output paths:
  weights/seeds/sae_resid_mid_s{1,2}.pt   (4k)
  weights/seeds_50k/sae_resid_mid_s{1,2}.pt (50k)
"""
from __future__ import annotations
from pathlib import Path
import torch

from sleeper.model import cache_activations, load_paired_dataset, load_sleeper_model
from sleeper.sae import save, train


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_train = 10_000
    seq_len = 128
    d_sae = 1536
    k = 32
    batch_size = 4096
    lr = 5e-4

    print(f"[train-mid-seeds] device={device}")
    model = load_sleeper_model(device=device)
    splits = load_paired_dataset(tokenizer=model.tokenizer,
                                  n_train=n_train, n_val=0, n_test=0,
                                  seq_len=seq_len, seed=0)
    train_tokens = splits["train"].tokens

    print(f"[train-mid-seeds] harvest {train_tokens.shape[0]} seqs at resid_mid")
    acts = cache_activations(model=model, tokens=train_tokens,
                              hook_names=["blocks.0.hook_resid_mid"], chunk_size=16)
    A = acts["blocks.0.hook_resid_mid"]

    targets = []
    for n_steps, parent in [(4_000, Path("weights/seeds")), (50_000, Path("weights/seeds_50k"))]:
        for sae_seed in (1, 2):
            out = parent / f"sae_resid_mid_s{sae_seed}.pt"
            if out.exists():
                print(f"  [skip] exists: {out}")
                continue
            targets.append((out, n_steps, sae_seed))

    for out, n_steps, sae_seed in targets:
        print(f"\n[train-mid-seeds] training {out}  n_steps={n_steps}  seed={sae_seed}")
        sae, _ = train(A, d_sae=d_sae, k=k, n_steps=n_steps,
                       batch_size=batch_size, lr=lr, seed=sae_seed, device=device)
        out.parent.mkdir(parents=True, exist_ok=True)
        save(sae, out, layer_hook="blocks.0.hook_resid_mid",
             n_train_seqs=int(train_tokens.shape[0]),
             seq_len=seq_len, n_steps=n_steps, batch_size=batch_size, lr=lr)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
