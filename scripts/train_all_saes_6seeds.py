"""Train 6 ln1 SAEs and 6 resid_mid SAEs (seeds 0–5), one harvest pass each hook.

Outputs (skip any that already exist):
  weights/seeds/sae_ln1_s{0..5}.pt         — upstream (OV-only) SAEs
  weights/seeds/sae_resid_mid_s{0..5}.pt   — downstream (additive) SAEs

Seeds 0–4 of sae_ln1 may already exist from train_all_saes.py; they are skipped.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sleeper.model import cache_activations, load_paired_dataset, load_sleeper_model
from sleeper.sae import save, train

N_SEEDS   = 6
LN1_HOOK  = "blocks.0.ln1.hook_normalized"
MID_HOOK  = "blocks.0.hook_resid_mid"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(N_SEEDS)),
                   help="Seeds to train (default: 0..5). Use to split work across GPUs.")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_train, seq_len, d_sae, k = 10_000, 128, 1_536, 32
    n_steps, batch_size, lr    = 4_000,  4_096, 5e-4

    seeds_dir = Path("weights/seeds")
    seeds_dir.mkdir(parents=True, exist_ok=True)

    ln1_paths = {s: seeds_dir / f"sae_ln1_s{s}.pt"        for s in args.seeds}
    mid_paths = {s: seeds_dir / f"sae_resid_mid_s{s}.pt"  for s in args.seeds}

    missing_ln1 = [s for s, p in ln1_paths.items() if not p.exists()]
    missing_mid = [s for s, p in mid_paths.items() if not p.exists()]
    if not missing_ln1 and not missing_mid:
        print(f"[train-6seeds] seeds={args.seeds}: all checkpoints present, nothing to do")
        return

    print(f"[train-6seeds] device={device}  seeds={args.seeds}  "
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

    print(f"[train-6seeds] harvesting {train_tokens.shape[0]} seqs at hooks={needed_hooks}")
    acts = cache_activations(model=model, tokens=train_tokens,
                             hook_names=needed_hooks, chunk_size=16)

    for seed in args.seeds:
        for hook_name, path_map, tag in [
            (LN1_HOOK, ln1_paths, "ln1"),
            (MID_HOOK, mid_paths, "mid"),
        ]:
            if hook_name not in acts:
                continue
            out = path_map[seed]
            if out.exists():
                print(f"[train-6seeds] skip {out} (exists)")
                continue
            sae, _ = train(acts[hook_name], d_sae=d_sae, k=k, n_steps=n_steps,
                           batch_size=batch_size, lr=lr, seed=seed, device=device)
            save(sae, out, layer_hook=hook_name,
                 n_train_seqs=int(train_tokens.shape[0]),
                 seq_len=seq_len, n_steps=n_steps, batch_size=batch_size, lr=lr)
            print(f"[train-6seeds] wrote {out}")

    print("[train-6seeds] done")


if __name__ == "__main__":
    main()
