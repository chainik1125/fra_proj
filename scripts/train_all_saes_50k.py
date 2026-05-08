"""Train resid_mid + 5 ln1 SAEs for 50 000 steps.

Identical to train_all_saes.py except:
  - n_steps = 50_000
  - outputs go to  weights/seeds_50k/  and  weights/sae_resid_mid_50k.pt
"""
from __future__ import annotations

from pathlib import Path

import torch

from sleeper.model import cache_activations, load_paired_dataset, load_sleeper_model
from sleeper.sae import save, train


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
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

    mid_path = weights / "sae_resid_mid_50k.pt"
    ln1_paths = [seeds_dir / f"sae_ln1_s{s}.pt" for s in range(5)]
    targets = [("mid", mid_path)] + [("ln1", p) for p in ln1_paths]
    missing = [(kind, p) for kind, p in targets if not p.exists()]
    if not missing:
        print("[all-saes-50k] all SAE checkpoints present, nothing to do")
        return

    print(f"[all-saes-50k] device={device}  n_steps={n_steps}  missing={len(missing)}/6")
    model = load_sleeper_model(device=device)
    splits = load_paired_dataset(
        tokenizer=model.tokenizer,
        n_train=n_train, n_val=0, n_test=0,
        seq_len=seq_len, seed=0,
    )
    train_tokens = splits["train"].tokens

    needed_hooks: list[str] = []
    if any(kind == "mid" for kind, _ in missing):
        needed_hooks.append("blocks.0.hook_resid_mid")
    if any(kind == "ln1" for kind, _ in missing):
        needed_hooks.append("blocks.0.ln1.hook_normalized")

    print(f"[all-saes-50k] harvest {train_tokens.shape[0]} seqs at hooks={needed_hooks}")
    acts = cache_activations(model=model, tokens=train_tokens,
                             hook_names=needed_hooks, chunk_size=16)

    if not mid_path.exists():
        sae, _ = train(acts["blocks.0.hook_resid_mid"], d_sae=d_sae, k=k, n_steps=n_steps,
                       batch_size=batch_size, lr=lr, seed=0, device=device)
        save(sae, mid_path, layer_hook="blocks.0.hook_resid_mid",
             n_train_seqs=int(train_tokens.shape[0]),
             seq_len=seq_len, n_steps=n_steps, batch_size=batch_size, lr=lr)
        print(f"[all-saes-50k] wrote {mid_path}")

    for seed in range(5):
        out = ln1_paths[seed]
        if out.exists():
            print(f"[all-saes-50k] skip {out} (exists)")
            continue
        sae, _ = train(acts["blocks.0.ln1.hook_normalized"], d_sae=d_sae, k=k,
                       n_steps=n_steps, batch_size=batch_size, lr=lr,
                       seed=seed, device=device)
        save(sae, out, layer_hook="blocks.0.ln1.hook_normalized",
             n_train_seqs=int(train_tokens.shape[0]),
             seq_len=seq_len, n_steps=n_steps, batch_size=batch_size, lr=lr)
        print(f"[all-saes-50k] wrote {out}")

    print("[all-saes-50k] done")


if __name__ == "__main__":
    main()
