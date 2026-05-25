"""Train N paired (ln1, resid_mid) SAEs — one of each per seed.

The pipeline assumes upstream and downstream SAEs are paired by seed:
  weights/seeds[_Nk]/sae_ln1_s{s}.pt         — upstream (OV intervention)
  weights/seeds[_Nk]/sae_resid_mid_s{s}.pt   — downstream (target attribution, conv. baseline)

There is no "shared" downstream SAE — every downstream consumer uses
sae_resid_mid_s{s}.pt paired with sae_ln1_s{s}.pt. The single canonical
training script trains both halves of every pair in one harvest pass.

Output directory:
  --n_steps 4000           → weights/seeds/
  --n_steps N (N != 4000)  → weights/seeds_{N//1000}k/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sleeper.model import cache_activations, load_paired_dataset, load_sleeper_model
from sleeper.sae import save, train

LN1_HOOK = "blocks.0.ln1.hook_normalized"
MID_HOOK = "blocks.0.hook_resid_mid"


def sae_paths(n_steps: int) -> Path:
    """Return the seeds_dir for a given n_steps. Mirrors the convention used
    elsewhere in the repo so 4k → weights/seeds, 50k → weights/seeds_50k, etc."""
    weights = Path("weights")
    if n_steps == 4_000:
        return weights / "seeds"
    return weights / f"seeds_{n_steps // 1000}k"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds",   type=int, nargs="+", default=[0, 1, 2, 3, 4, 5],
                   help="Seeds to train (default: 0..5).")
    p.add_argument("--n_steps", type=int, default=4_000,
                   help="SAE training steps; also controls the output directory "
                        "(4000 → weights/seeds/, N → weights/seeds_{N//1000}k/).")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_train, seq_len, d_sae, k = 10_000, 128, 1_536, 32
    batch_size, lr            = 4_096, 5e-4
    n_steps = args.n_steps

    seeds_dir = sae_paths(n_steps)
    seeds_dir.mkdir(parents=True, exist_ok=True)

    ln1_paths = {s: seeds_dir / f"sae_ln1_s{s}.pt"       for s in args.seeds}
    mid_paths = {s: seeds_dir / f"sae_resid_mid_s{s}.pt" for s in args.seeds}

    missing_ln1 = [s for s, p in ln1_paths.items() if not p.exists()]
    missing_mid = [s for s, p in mid_paths.items() if not p.exists()]
    if not missing_ln1 and not missing_mid:
        print(f"[train-saes] n_steps={n_steps} seeds={args.seeds}: "
              f"all checkpoints present, nothing to do")
        return

    print(f"[train-saes] device={device}  n_steps={n_steps}  seeds={args.seeds}  "
          f"missing ln1={missing_ln1}  missing mid={missing_mid}")

    model = load_sleeper_model(device=device)
    splits = load_paired_dataset(model.tokenizer, n_train=n_train, n_val=0, n_test=0,
                                  seq_len=seq_len, seed=0)
    train_tokens = splits["train"].tokens

    needed_hooks: list[str] = []
    if missing_ln1:
        needed_hooks.append(LN1_HOOK)
    if missing_mid:
        needed_hooks.append(MID_HOOK)

    print(f"[train-saes] harvesting {train_tokens.shape[0]} seqs at hooks={needed_hooks}")
    acts = cache_activations(model=model, tokens=train_tokens,
                             hook_names=needed_hooks, chunk_size=16)

    for seed in args.seeds:
        for hook_name, path_map in [(LN1_HOOK, ln1_paths), (MID_HOOK, mid_paths)]:
            if hook_name not in acts:
                continue
            out = path_map[seed]
            if out.exists():
                print(f"[train-saes] skip {out} (exists)")
                continue
            sae, _ = train(acts[hook_name], d_sae=d_sae, k=k, n_steps=n_steps,
                           batch_size=batch_size, lr=lr, seed=seed, device=device)
            save(sae, out, layer_hook=hook_name,
                 n_train_seqs=int(train_tokens.shape[0]),
                 seq_len=seq_len, n_steps=n_steps, batch_size=batch_size, lr=lr)
            print(f"[train-saes] wrote {out}")

    print("[train-saes] done")


if __name__ == "__main__":
    main()
