"""Consumer poll loop: evaluate this seed's checkpoints as they land on HF.

Loads the model + checkpoint-invariant state ONCE, then polls the HF bus:
for each expected checkpoint that exists but lacks a result JSON, download →
`eval_one` → upload result. Idempotent (skip-if-result-exists), so a relaunched
pod resumes cleanly. Exits when all of this seed's results are present or
`--max_wall_sec` is hit.

  python -u -m scripts.eval_poll --seed 0
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from sleeper.model import load_sleeper_model
from scripts.eval_checkpoint import DEFAULT_ALPHAS, build_consumer_state, eval_one
from scripts.sae_scaling_paths import (
    DEFAULT_HF_REPO, HOOKS, eval_status_rel, expected_rels, hf_list, hf_upload,
    hf_download, parse_ckpt_rel,
)


def _write_eval_status(args, status_rel, n_done, n_total, last):
    payload = {"seed": args.seed, "n_done": n_done, "n_total": n_total,
               "last": last, "ts": time.time()}
    sp = args.local_dir / status_rel
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps(payload))
    try:
        hf_upload(sp, args.hf_repo, status_rel)
    except Exception as e:
        print(f"[eval] status upload failed (non-fatal): {e}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--hookpoints", nargs="+", default=["ln1", "resid_mid"], choices=list(HOOKS))
    p.add_argument("--d_saes", type=int, nargs="+", default=[3072, 6144, 12288, 24576])
    p.add_argument("--ks", type=int, nargs="+", default=[10, 32, 50])
    p.add_argument("--steps", type=int, nargs="+",
                   default=[10000, 20000, 30000, 40000, 50000])
    p.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS)
    p.add_argument("--hf_repo", default=DEFAULT_HF_REPO)
    p.add_argument("--local_dir", type=Path, default=Path("/workspace/sae_scaling_out"))
    p.add_argument("--poll_sec", type=int, default=60)
    p.add_argument("--max_wall_sec", type=int, default=14 * 3600)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_sleeper_model(device=device)
    state = build_consumer_state(model, device)

    expected = expected_rels(args.hookpoints, [args.seed], args.d_saes, args.ks, args.steps)
    result_set = {r for (_, r) in expected}
    n_total = len(result_set)
    status_rel = eval_status_rel(args.seed)
    print(f"[eval] seed={args.seed} expects {n_total} results "
          f"({args.hookpoints} × {len(args.d_saes)}×{len(args.ks)}×{len(args.steps)})", flush=True)

    deadline = time.time() + args.max_wall_sec
    while time.time() < deadline:
        files = set(hf_list(args.hf_repo))
        n_done = len(result_set & files)
        if n_done >= n_total:
            print(f"[eval] seed={args.seed} COMPLETE — {n_done}/{n_total}", flush=True)
            _write_eval_status(args, status_rel, n_done, n_total, "complete")
            return
        todo = [(c, r) for (c, r) in expected if c in files and r not in files]
        if not todo:
            print(f"[eval] {n_done}/{n_total} done; waiting for checkpoints "
                  f"({args.poll_sec}s)…", flush=True)
            _write_eval_status(args, status_rel, n_done, n_total, "waiting")
            time.sleep(args.poll_sec)
            continue
        for ckpt_rel_, res_rel_ in todo:
            meta = parse_ckpt_rel(ckpt_rel_)
            print(f"[eval] {meta['hookpoint']} d{meta['d_sae']} k{meta['k']} "
                  f"step{meta['step']} …", flush=True)
            try:
                local = hf_download(args.hf_repo, ckpt_rel_, args.local_dir)
                result = eval_one(model, state, local, args.alphas, device)
                out = args.local_dir / res_rel_
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(result, indent=2))
                hf_upload(out, args.hf_repo, res_rel_)
                n_done += 1
                _write_eval_status(args, status_rel, n_done, n_total, res_rel_)
                print(f"[eval]   ⇒ {res_rel_}  ({n_done}/{n_total})", flush=True)
            except Exception as e:
                print(f"[eval]   FAILED {ckpt_rel_}: {e}", flush=True)

    print(f"[eval] seed={args.seed} hit max wall — exiting", flush=True)


if __name__ == "__main__":
    main()
