"""Eval stage: full α sweep across the tuples produced by select_features.py.

Reads the tuples_json (channel + per-seed tuple list), runs lockstep multi-seed
eval at every (tuple, alpha), writes a results_json with the canonical 4-metric
outputs per (seed, tuple, alpha).

This stage knows nothing about how the tuples were chosen — it just evaluates
whatever it's given. Mode determines tuple breadth in select_features.py.

Output JSON:
{
  "config": {...},
  "channel": "qk+ov",
  "baseline": {"asr_per_seed": [...], "asr": float},
  "results": [
    {
      "seed": 0,
      "tuple": [[870, "Q"], [1205, "K"], [1114, "V"]],
      "alpha_sweep": {"0.0": {asr, jsd_clean, ...}, "0.5": {...}, ...}
    },
    ...
  ]
}
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from sleeper.eval import (
    _build_baselines_per_seed, _multi_seed_asr, eval_tuple,
)
from sleeper.hooks import ACTIVE_CHANNELS
from sleeper.model import (
    left_pad_prompts, load_dep_prompts, load_paired_dataset, load_sleeper_model,
)
from sleeper.sae import load as sae_load


@torch.no_grad()
def eval_tuples_json(
    tuples_dict: dict,
    *,
    sae_dir: Path,
    eval_alphas: list[float],
    n_eval: int = 200,
    gen_tokens: int = 16,
    eval_seeds: list[int] | None = None,
    eval_temperature: float = 1.0,
    device: str | None = None,
) -> dict:
    """Run the full α sweep for every (seed, tuple) in `tuples_dict`."""
    channel = tuples_dict["channel"]
    per_seed = tuples_dict["per_seed"]
    active   = ACTIVE_CHANNELS[channel]
    eval_seeds = eval_seeds or [0, 1, 2, 3, 4]

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_sleeper_model(device=device)
    tok    = model.tokenizer
    pad_id = tok.pad_token_id or tok.eos_token_id
    W      = {c: getattr(model, f"W_{c}")[0].detach().to(device) for c in ("Q", "K", "V")}

    # Held-out eval split: same data flow as the legacy channel_sweep / pick scripts.
    splits     = load_paired_dataset(tok, n_train=2, n_val=2, n_test=n_eval, seq_len=128, seed=0)
    _ = splits  # eval-side dep prompts come from load_dep_prompts; splits is loaded
                # solely to advance the RNG identically to the legacy pipeline.
    n_eval_dep = n_eval // 2
    raw_dep    = load_dep_prompts(tok, n_eval + 200, split="test")[200 : 200 + n_eval_dep]
    eval_dep_lp, eval_dep_attn = left_pad_prompts(raw_dep, pad_id)
    eval_dep_lp   = eval_dep_lp.to(device)
    eval_dep_attn = eval_dep_attn.to(device)

    # Per-seed unsteered baselines (clean / dep rollouts) — one tiled forward each.
    print(f"[eval] pre-building per-seed baselines (B={eval_dep_lp.shape[0]}, "
          f"seeds={eval_seeds})...", flush=True)
    clean_lsm_ps, clean_tok_ps, dep_lsm_ps = _build_baselines_per_seed(
        model, eval_dep_lp, eval_dep_attn, gen_tokens, device,
        seeds=eval_seeds, temperature=eval_temperature,
    )
    eval_base_asr_per_seed = _multi_seed_asr(
        model, None, [], 0.0, set(), W,
        eval_dep_lp, eval_dep_attn, gen_tokens,
        seeds=eval_seeds, temperature=eval_temperature, device=device,
    )
    eval_base_asr = sum(eval_base_asr_per_seed) / len(eval_base_asr_per_seed)
    print(f"[eval] channel={channel}  baseline asr={eval_base_asr:.3f}", flush=True)

    results: list[dict] = []
    for seed_str, tuples in per_seed.items():
        seed = int(seed_str)
        sae_ln1, _ = sae_load(sae_dir / f"sae_ln1_s{seed}.pt", device=device)
        print(f"\n[eval] ══ seed={seed}  {len(tuples)} tuples ══", flush=True)

        for ti, tup in enumerate(tuples):
            sel_tuple = [(int(f), str(c)) for (f, c) in tup]
            alpha_sweep: dict[str, dict] = {}
            t0 = time.time()
            for ea in eval_alphas:
                t_one = time.time()
                ev = eval_tuple(
                    model, sae_ln1, sel_tuple, ea, active, W,
                    eval_dep_lp, eval_dep_attn,
                    clean_lsm_ps, clean_tok_ps, dep_lsm_ps,
                    gen_tokens, device,
                    eval_seeds=eval_seeds, eval_temperature=eval_temperature,
                )
                alpha_sweep[str(ea)] = ev
                print(f"[eval]   [t{ti+1}/{len(tuples)} α={ea:>4.1f}] "
                      f"asr={ev['asr']:.3f}  jsd_cln={ev['jsd_clean']:.3f}  "
                      f"jsd_dep={ev['jsd_pois']:.3f}  exact={ev['exact_match']:.3f}  "
                      f"({time.time()-t_one:.1f}s)", flush=True)
            print(f"[eval]   tuple {ti+1}/{len(tuples)} done in {time.time()-t0:.1f}s")
            results.append({
                "seed": seed,
                "tuple": sel_tuple,
                "alpha_sweep": alpha_sweep,
            })

    return {
        "channel": channel,
        "regime":  tuples_dict.get("regime"),
        "mode":    tuples_dict.get("mode"),
        "config":  {
            "sae_dir": str(sae_dir), "eval_alphas": eval_alphas, "n_eval": n_eval,
            "gen_tokens": gen_tokens, "eval_seeds": list(eval_seeds),
            "eval_temperature": eval_temperature,
        },
        "baseline": {"asr_per_seed": eval_base_asr_per_seed, "asr": eval_base_asr},
        "results":  results,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tuples_json", type=Path, required=True,
                   help="Output of scripts.select_features (defines channel + per-seed tuples).")
    p.add_argument("--sae_dir",     type=Path, default=Path("weights/seeds"))
    p.add_argument("--eval_alphas", type=float, nargs="+",
                   default=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    p.add_argument("--n_eval",      type=int, default=200)
    p.add_argument("--gen_tokens",  type=int, default=16)
    p.add_argument("--eval_seeds",  type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--eval_temperature", type=float, default=1.0)
    p.add_argument("--out",         type=Path, required=True)
    p.add_argument("--device",      default=None)
    args = p.parse_args()

    tuples_dict = json.loads(args.tuples_json.read_text())
    out = eval_tuples_json(
        tuples_dict,
        sae_dir=args.sae_dir, eval_alphas=args.eval_alphas, n_eval=args.n_eval,
        gen_tokens=args.gen_tokens, eval_seeds=args.eval_seeds,
        eval_temperature=args.eval_temperature, device=args.device,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[eval] wrote {args.out}  ({len(out['results'])} tuple-rows × "
          f"{len(args.eval_alphas)} alphas)")


if __name__ == "__main__":
    main()
