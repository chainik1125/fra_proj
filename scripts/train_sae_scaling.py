"""Producer: train the full (d_sae × k) grid for one (hookpoint, seed),
streaming a checkpoint to HF every N steps.

Harvests the model activations ONCE (the only heavy GPU setup; identical
across all 12 configs of a hookpoint), then trains each config to
`--n_steps`, saving at every `--checkpoint_steps` boundary. Each checkpoint
is uploaded to the HF dataset bus and a heartbeat status file is refreshed so
the consumer / monitor can track progress.

  python -u -m scripts.train_sae_scaling --hookpoint ln1 --seed 0
  python -u -m scripts.train_sae_scaling --hookpoint resid_mid --seed 0
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from sleeper.model import cache_activations, load_paired_dataset, load_sleeper_model
from sleeper.sae import save, train
from scripts.sae_scaling_paths import (
    DEFAULT_HF_REPO, HOOKS, ckpt_rel, ensure_repo, hf_upload, train_status_rel,
)


def _write_status(args, status_rel: str, done: list[str], total: int) -> None:
    # Local only — the co-located consumer batch-uploads status/ + results/ to HF
    # in single folder commits. The producer never touches HF (avoids blowing
    # HF's 128-commits/hour cap with per-checkpoint uploads).
    payload = {"hookpoint": args.hookpoint, "seed": args.seed, "done": done,
               "n_done": len(done), "n_total": total, "ts": time.time()}
    sp = args.local_dir / status_rel
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps(payload))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hookpoint", choices=list(HOOKS), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--d_saes", type=int, nargs="+", default=[1536, 3072, 6144])  # 2x,4x,8x first; 16x/24x after review
    p.add_argument("--ks", type=int, nargs="+", default=[10, 32, 50])
    p.add_argument("--checkpoint_steps", type=int, nargs="+",
                   default=[10000, 20000, 30000, 40000, 50000])
    p.add_argument("--n_steps", type=int, default=50000)
    p.add_argument("--n_train", type=int, default=10000)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--hf_repo", default=DEFAULT_HF_REPO)
    p.add_argument("--local_dir", type=Path, default=Path("/workspace/sae_scaling_out"))
    p.add_argument("--no_hf", action="store_true", help="skip HF (local smoke test)")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hook = HOOKS[args.hookpoint]
    if not args.no_hf:
        ensure_repo(args.hf_repo)

    model = load_sleeper_model(device=device)
    splits = load_paired_dataset(model.tokenizer, n_train=args.n_train, n_val=0,
                                 n_test=0, seq_len=args.seq_len, seed=0)
    print(f"[train] harvesting {splits['train'].tokens.shape[0]} seqs at {hook}", flush=True)
    acts = cache_activations(model, splits["train"].tokens, [hook])[hook]  # (N,T,d) cpu fp16

    configs = [(d, k) for d in args.d_saes for k in args.ks]
    total = len(configs) * len(args.checkpoint_steps)
    status_rel = train_status_rel(args.hookpoint, args.seed)
    done: list[str] = []
    print(f"[train] hook={args.hookpoint} seed={args.seed} "
          f"configs={len(configs)} → {total} checkpoints", flush=True)
    _write_status(args, status_rel, done, total)

    def make_cb(d_sae: int, k: int):
        def _cb(sae, step: int) -> None:
            rel = ckpt_rel(args.hookpoint, args.seed, d_sae, k, step)
            path = args.local_dir / rel
            save(sae, path, layer_hook=hook, seed=args.seed, step=step,
                 d_sae=d_sae, k=k, hookpoint=args.hookpoint,
                 n_train_seqs=int(acts.shape[0]), seq_len=args.seq_len,
                 n_steps=args.n_steps, batch_size=args.batch_size, lr=args.lr)
            # Saved locally only; the consumer reads checkpoints from local disk
            # (co-located) and is the sole HF uploader (batched).
            done.append(rel)
            _write_status(args, status_rel, done, total)
            print(f"[train]   saved {rel}  ({len(done)}/{total})", flush=True)
        return _cb

    for d_sae, k in configs:
        # Resume: skip a config whose checkpoints all already exist locally
        # (so a relaunched pod doesn't retrain finished configs).
        if all((args.local_dir / ckpt_rel(args.hookpoint, args.seed, d_sae, k, s)).exists()
               for s in args.checkpoint_steps):
            for s in args.checkpoint_steps:
                done.append(ckpt_rel(args.hookpoint, args.seed, d_sae, k, s))
            _write_status(args, status_rel, done, total)
            print(f"\n[train] === d_sae={d_sae} k={k} seed={args.seed} — already done, skip ===",
                  flush=True)
            continue
        print(f"\n[train] === d_sae={d_sae} k={k} seed={args.seed} ===", flush=True)
        t0 = time.time()
        train(acts, d_sae=d_sae, k=k, n_steps=args.n_steps, batch_size=args.batch_size,
              lr=args.lr, seed=args.seed, device=device,
              checkpoint_steps=tuple(args.checkpoint_steps),
              on_checkpoint=make_cb(d_sae, k))
        print(f"[train] d_sae={d_sae} k={k} done in {time.time()-t0:.0f}s", flush=True)

    print(f"\n[train] ALL DONE — {len(done)}/{total} for "
          f"{args.hookpoint} seed{args.seed}", flush=True)


if __name__ == "__main__":
    main()
