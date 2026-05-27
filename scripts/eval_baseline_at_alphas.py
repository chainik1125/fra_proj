"""Re-eval an existing downstream_baseline JSON's per-seed winners at a new
alpha grid. Skips identification/screening; just sweeps the existing winner
feature through ``--alphas`` and writes a baseline-schema JSON the legacy
aggregate / plot scripts already understand.

Use when you've already run scripts.downstream_baseline and want a wider /
denser alpha sweep without redoing the (deterministic-given-SAE) winner
selection.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from sleeper.eval import (
    _build_baselines_per_seed, eval_downstream_baseline, split_dep_prompts,
)
from sleeper.model import MODELS, left_pad_prompts, load_sleeper_model
from sleeper.sae import load as sae_load


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model",         choices=list(MODELS), default="tinystories")
    p.add_argument("--baseline_json", type=Path,  required=True,
                   help="Existing downstream_baseline JSON; reuses each "
                        "per-seed.winner feature.")
    p.add_argument("--sae_dir",       type=Path,  required=True)
    p.add_argument("--alphas",        type=float, nargs="+", required=True)
    p.add_argument("--n_sel",         type=int,   default=200)
    p.add_argument("--n_eval",        type=int,   default=400)
    p.add_argument("--gen_tokens",    type=int,   default=16)
    p.add_argument("--eval_seeds",    type=int,   nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--eval_temperature", type=float, default=1.0)
    p.add_argument("--out",           type=Path,  required=True)
    p.add_argument("--device",        default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    base = json.loads(args.baseline_json.read_text())
    model = load_sleeper_model(model=args.model, device=device)
    tok = model.tokenizer
    pad_id = tok.pad_token_id or tok.eos_token_id

    eval_raw = split_dep_prompts(tok, args.n_sel, args.n_eval, model=args.model)["eval"]
    eval_dep_lp, eval_dep_attn = left_pad_prompts(eval_raw, pad_id)
    eval_dep_lp   = eval_dep_lp.to(device)
    eval_dep_attn = eval_dep_attn.to(device)

    print(f"[base-reeval] pre-building per-seed baselines (B={eval_dep_lp.shape[0]}, "
          f"seeds={args.eval_seeds}) ...", flush=True)
    clean_lsm_ps, clean_tok_ps, dep_lsm_ps = _build_baselines_per_seed(
        model, eval_dep_lp, eval_dep_attn, args.gen_tokens, device,
        seeds=args.eval_seeds, temperature=args.eval_temperature,
    )

    out_per_seed: dict = {}
    for seed_key, info in base["per_seed"].items():
        seed = int(seed_key.lstrip("s"))
        winner = int(info["winner"])
        path = args.sae_dir / f"sae_resid_mid_s{seed}.pt"
        sae_mid, _ = sae_load(path, device=device)
        print(f"\n[base-reeval] === seed={seed} winner=f{winner} ===", flush=True)
        per_alpha: dict = {}
        for alpha in args.alphas:
            t0 = time.time()
            m = eval_downstream_baseline(
                model, sae_mid, winner, alpha,
                eval_dep_lp, eval_dep_attn,
                clean_lsm_ps, clean_tok_ps, dep_lsm_ps,
                args.gen_tokens, device,
                eval_seeds=args.eval_seeds, eval_temperature=args.eval_temperature,
            )
            per_alpha[str(alpha)] = m
            print(f"[base-reeval]   α={alpha:>4.1f}  asr={m['asr']:.3f}  "
                  f"jsdc={m['jsd_clean']:.3f}  jsdp={m['jsd_pois']:.3f}  "
                  f"em={m['exact_match']:.3f}  ({time.time()-t0:.1f}s)", flush=True)
        out_per_seed[seed_key] = {**info, "per_alpha": per_alpha}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "config":   {**base.get("config", {}),
                     "alphas":          args.alphas,
                     "n_sel":           args.n_sel,
                     "n_eval":          args.n_eval,
                     "reeval_from":     str(args.baseline_json),
                     "model":           args.model},
        "per_seed": out_per_seed,
    }, indent=2, default=str))
    print(f"\n[base-reeval] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
