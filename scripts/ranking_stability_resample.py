"""Method (2) of the ranking-stability check: resample MORE α=0 finance rollouts
so the misaligned-coherent bucket is well-powered at coh>70 (instead of the n=8
tercile fallback), then recompute the OV and QK ln1 bucketed-diff rankings.

Generate N_SAMPLES fresh α=0 finance rollouts (8 prompts × samples, temp=1,
max_new=100 — same config as alpha0_noise_gen), judge each with gpt-4o-mini@T0,
bucket at coh>70 (misal align≤30, align align>70), then run the EXISTING
attribution code (ov + qk, diff-mode=bucket) on those buckets.

Output JSON: {ov:{feature_ids,scores,meta}, qk:{...}, buckets:{...}} — the
deliverable feeds the 3-way ranking comparison.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import compute_fra_diff_ranking as R


def judge_all(rollouts, max_workers=20):
    """gpt-4o-mini @ T0 align+coh for each rollout (reuses judge_one)."""
    from openai import OpenAI
    from phase1_judge_and_combine import judge_one
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY_MATS")
    if not key:
        raise SystemExit("set OPENAI_API_KEY (or OPENAI_API_KEY_MATS)")
    client = OpenAI(api_key=key)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(judge_one, client, r["prompt"], r["response"]): i
                for i, r in enumerate(rollouts)}
        done = 0
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                s = fut.result()
                rollouts[i]["align"] = float(s["alignment"])
                rollouts[i]["coh"] = float(s["coherence"])
                done += 1
            except Exception as e:
                print(f"  judge err idx {i}: {e}", flush=True)
                rollouts[i]["align"] = float("nan"); rollouts[i]["coh"] = float("nan")
    print(f"[resample] judged {done}/{len(rollouts)} in {time.time()-t0:.1f}s", flush=True)
    return rollouts


@torch.no_grad()
def generate_finance_rollouts(model, tokenizer, n_prompts, samples_per_prompt,
                              seed, max_new_tokens=100, temperature=1.0):
    from fra.em_evaluation import EM_EVAL_PROMPTS, generate_with_hooks_batch
    base = EM_EVAL_PROMPTS[:n_prompts]
    prompts = base * samples_per_prompt
    per_seed = [seed + i for i in range(len(prompts))]
    responses = generate_with_hooks_batch(
        model, tokenizer, prompts, fwd_hooks=[],
        max_new_tokens=max_new_tokens, temperature=temperature, seed=per_seed)
    return [{"prompt": p, "response": r, "prompt_idx": i % n_prompts,
             "sample_idx": i // n_prompts, "seed": per_seed[i]}
            for i, (p, r) in enumerate(zip(prompts, responses))]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sae-dir", required=True)
    p.add_argument("--sae", default="ln1", choices=["ln1", "resid_post"])
    p.add_argument("--layer", type=int, default=24)
    p.add_argument("--head", type=int, default=12)
    p.add_argument("--n-prompts", type=int, default=8)
    p.add_argument("--samples-per-prompt", type=int, default=64)  # 8×64 = 512
    p.add_argument("--gen-seed", type=int, default=1000)
    p.add_argument("--coh-floor", type=int, default=70)
    p.add_argument("--top-n", type=int, default=50)
    p.add_argument("--k-pairs", type=int, default=50)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", required=True)
    p.add_argument("--rollouts-out", default=None,
                   help="optional: dump the judged resample rollouts here (for HF backup).")
    args = p.parse_args()

    hook_point = "ln1.hook_normalized" if args.sae == "ln1" else "hook_resid_post"
    from phase1_grid_14b_orchestrator import load_em_model, load_sae_from_dir

    t0 = time.time()
    print("=== ranking_stability_resample (finance, coh>%d) ===" % args.coh_floor)
    model = load_em_model("finance", device=args.device)
    tokenizer = model.tokenizer
    sae = load_sae_from_dir(Path(args.sae_dir), device=args.device)
    if args.sae == "ln1":
        sae._gamma = model.blocks[args.layer].ln1.w.detach().float()

    n = args.n_prompts * args.samples_per_prompt
    print(f"[resample] generating {n} α=0 finance rollouts...", flush=True)
    rollouts = generate_finance_rollouts(
        model, tokenizer, args.n_prompts, args.samples_per_prompt, args.gen_seed,
        max_new_tokens=100, temperature=1.0)
    print(f"[resample] generated {len(rollouts)} in {time.time()-t0:.1f}s", flush=True)

    rollouts = judge_all(rollouts)
    if args.rollouts_out:
        Path(args.rollouts_out).write_text(json.dumps(rollouts, indent=2))

    b_misal, b_align, bucket_meta = R.bucket_rollouts(rollouts, coh_floor=args.coh_floor)
    print(f"[resample] buckets @coh>{args.coh_floor}: mode={bucket_meta['bucket_mode']} "
          f"coherent={bucket_meta['n_coherent']} misal={bucket_meta['n_misal']} "
          f"align={bucket_meta['n_align']} (of {len(rollouts)})", flush=True)
    if bucket_meta["n_misal"] == 0 or bucket_meta["n_align"] == 0:
        raise SystemExit("[resample] empty bucket — cannot diff.")

    out = {"method": "resample_coh%d" % args.coh_floor,
           "n_generated": len(rollouts), "buckets": bucket_meta}
    for attribution in ("ov", "qk"):
        print(f"[resample] {attribution} ranking...", flush=True)
        fids, scores, pair_meta = R.bucket_diff_features(
            model, sae, args.layer, args.head, hook_point, attribution,
            b_misal, b_align, args.max_length, args.top_k, args.k_pairs, args.top_n)
        out[attribution] = {"feature_ids": fids,
                            "top10": fids[:10],
                            "pair_meta": pair_meta}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[resample] wrote {args.out}")
    print(f"[resample] OV top10: {out['ov']['top10']}")
    print(f"[resample] QK top10: {out['qk']['top10']}")
    print(f"=== done in {time.time()-t0:.1f}s ===")


if __name__ == "__main__":
    sys.exit(main())
