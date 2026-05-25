"""Consumer poll loop: evaluate this seed's checkpoints as the co-located
producer writes them to LOCAL disk.

Producer and consumer share one pod, so the consumer reads checkpoints straight
from `--local_dir` (no HF download) and is the SOLE HF uploader — it pushes the
`results/` and `status/` folders in batched single-commit `upload_folder` calls,
throttled to stay well under HF's 128-commits/hour cap. All HF ops are
non-fatal: a rate-limit/transient error is logged and retried next cycle, never
crashing the run. Idempotent: a local result JSON means that unit is done, so a
relaunched pod resumes cleanly.

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
    DEFAULT_HF_REPO, HOOKS, ensure_repo, eval_status_rel, hf_upload_folder,
    parse_ckpt_rel, result_rel,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--hookpoints", nargs="+", default=["ln1", "resid_mid"], choices=list(HOOKS))
    p.add_argument("--d_saes", type=int, nargs="+", default=[1536, 3072, 6144])  # 2x,4x,8x first; 16x/24x after review
    p.add_argument("--ks", type=int, nargs="+", default=[10, 32, 50])
    p.add_argument("--steps", type=int, nargs="+",
                   default=[10000, 20000, 30000, 40000, 50000])
    p.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS)
    p.add_argument("--hf_repo", default=DEFAULT_HF_REPO)
    p.add_argument("--local_dir", type=Path, default=Path("/workspace/sae_scaling_out"))
    p.add_argument("--poll_sec", type=int, default=45)
    p.add_argument("--upload_every_sec", type=int, default=180,
                   help="Min seconds between batched HF folder commits.")
    p.add_argument("--max_wall_sec", type=int, default=16 * 3600)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_sleeper_model(device=device)
    state = build_consumer_state(model, device)
    ensure_repo(args.hf_repo)

    n_total = (len(args.hookpoints) * len(args.d_saes) * len(args.ks) * len(args.steps))
    status_rel = eval_status_rel(args.seed)
    ckpt_root = args.local_dir / "sae_checkpoints"
    print(f"[eval] seed={args.seed} expects {n_total} results; reading ckpts from {ckpt_root}",
          flush=True)

    def push(last: str):
        """Batched single-commit upload of results/ + status/ (non-fatal)."""
        n_done = sum(1 for _ in (args.local_dir / "results").rglob("*.json")) \
            if (args.local_dir / "results").exists() else 0
        sp = args.local_dir / status_rel
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps({"seed": args.seed, "n_done": n_done,
                                  "n_total": n_total, "last": last, "ts": time.time()}))
        if (args.local_dir / "results").exists():
            hf_upload_folder(args.local_dir / "results", args.hf_repo, "results")
        hf_upload_folder(args.local_dir / "status", args.hf_repo, "status")
        return n_done

    deadline = time.time() + args.max_wall_sec
    last_upload = 0.0
    while time.time() < deadline:
        # Discover checkpoints on local disk (written by the co-located producer).
        ckpts = sorted(ckpt_root.rglob("*.pt")) if ckpt_root.exists() else []
        produced = False
        for cp in ckpts:
            rel = str(cp.relative_to(args.local_dir))
            meta = parse_ckpt_rel(rel)
            if meta is None or meta["seed"] != args.seed:
                continue
            res = args.local_dir / result_rel(meta["hookpoint"], meta["seed"],
                                              meta["d_sae"], meta["k"], meta["step"])
            if res.exists():
                continue  # already evaluated
            print(f"[eval] {meta['hookpoint']} d{meta['d_sae']} k{meta['k']} step{meta['step']} …",
                  flush=True)
            try:
                result = eval_one(model, state, cp, args.alphas, device)
                res.parent.mkdir(parents=True, exist_ok=True)
                res.write_text(json.dumps(result, indent=2))
                produced = True
                jc = result["curves"]; bestasr = min(c["asr"] for c in jc.values())
                print(f"[eval]   ⇒ {res.name}  (min-asr {bestasr:.3f})", flush=True)
            except Exception as e:
                print(f"[eval]   FAILED {rel}: {e}", flush=True)

        n_done = sum(1 for _ in (args.local_dir / "results").rglob("*.json")) \
            if (args.local_dir / "results").exists() else 0
        if n_done >= n_total:
            push("complete")
            print(f"[eval] seed={args.seed} COMPLETE — {n_done}/{n_total}", flush=True)
            return
        # Batched, throttled HF upload.
        if produced and time.time() - last_upload >= args.upload_every_sec:
            push("running")
            last_upload = time.time()
            print(f"[eval]   uploaded batch — {n_done}/{n_total} on HF", flush=True)
        if not produced:
            print(f"[eval] {n_done}/{n_total} done; waiting for checkpoints ({args.poll_sec}s)…",
                  flush=True)
            time.sleep(args.poll_sec)

    push("max_wall")
    print(f"[eval] seed={args.seed} hit max wall — exiting", flush=True)


if __name__ == "__main__":
    main()
