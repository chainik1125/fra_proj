"""End-to-end pipeline: train SAEs → select features → eval α-sweep.

Three stages, each callable in isolation:
    scripts/train_saes.py        → weights/seeds[_Nk]/...
    scripts/select_features.py   → tuples_json
    scripts/eval.py              → results_json

This orchestrator imports their entry-point functions and runs them in sequence.
Stages skip automatically if their artifact path already exists. Override with
explicit paths to short-circuit the pipeline.

Examples:
    # default: train 4k SAEs, ov channel, topk mode, full α sweep
    uv run -m scripts.run_experiment

    # all three channels with --top_k 20:
    for ch in ov qk qk+ov; do
        uv run -m scripts.run_experiment --channel $ch --top_k 20 \\
            --out_prefix results/full_$ch
    done

    # skip train (SAEs already exist), reuse pre-selected tuples:
    uv run -m scripts.run_experiment --sae_dir weights/seeds \\
        --tuples_json results/full_qk_plus_ov_tuples.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.eval import eval_tuples_json
from scripts.select_features import select_features
from scripts.train_saes import train_saes
from sleeper.model import MODELS


def _default_paths(out_prefix: Path) -> tuple[Path, Path]:
    """tuples_json and results_json default paths derived from --out_prefix."""
    return (
        out_prefix.parent / f"{out_prefix.name}_tuples.json",
        out_prefix.parent / f"{out_prefix.name}_results.json",
    )


def main() -> None:
    p = argparse.ArgumentParser()
    # Common
    p.add_argument("--model",         choices=list(MODELS), default="tinystories",
                   help="Sleeper to attack (drives loader / data / SAE defaults).")
    p.add_argument("--channel",       choices=["ov", "qk", "qk+ov"], default="ov")
    p.add_argument("--sae_seeds",     type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    p.add_argument("--device",        default=None)
    p.add_argument("--out_prefix",    type=Path, default=Path("results/run_experiment"),
                   help="Default JSON output prefix; suffixed with _tuples.json and _results.json.")

    # Train stage
    p.add_argument("--sae_dir",       type=Path, default=None,
                   help="If set and exists, skip training; otherwise train into this path "
                        "(default derived from --n_steps).")
    p.add_argument("--n_steps",       type=int, default=4_000)

    # Select stage
    p.add_argument("--tuples_json",   type=Path, default=None,
                   help="If set and exists, skip selection; otherwise write tuples here.")
    p.add_argument("--mode",          choices=["all", "topk", "winner"], default="topk")
    p.add_argument("--top_k",         type=int, default=20)
    p.add_argument("--final_selection", choices=["min-asr", "rank", "jsd"], default="min-asr")
    p.add_argument("--sel_alphas",    type=float, nargs="+", default=[2.0, 4.0],
                   help="Selection-phase α grid (only used with --mode winner).")
    p.add_argument("--n_sel",         type=int, default=200)

    # Eval stage
    p.add_argument("--results_json",  type=Path, default=None,
                   help="If set and exists, skip eval; otherwise write results here.")
    p.add_argument("--eval_alphas",   type=float, nargs="+",
                   default=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    p.add_argument("--n_eval",        type=int, default=200)
    p.add_argument("--gen_tokens",    type=int, default=16)
    p.add_argument("--eval_seeds",    type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--eval_temperature", type=float, default=1.0)

    args = p.parse_args()

    default_tuples_json, default_results_json = _default_paths(args.out_prefix)
    tuples_json   = args.tuples_json   or default_tuples_json
    results_json  = args.results_json  or default_results_json

    # ── 1) Train SAEs (skipped if --sae_dir points to an existing dir) ──
    if args.sae_dir is None or not args.sae_dir.exists():
        print(f"[run] STAGE 1: train_saes (model={args.model}) → {args.sae_dir or '<default>'}")
        sae_dir = train_saes(
            seeds=args.sae_seeds, model=args.model, n_steps=args.n_steps,
            out_dir=args.sae_dir, device=args.device,
        )
    else:
        sae_dir = args.sae_dir
        print(f"[run] STAGE 1: skip (sae_dir={sae_dir} exists)")

    # ── 2) Select features (skipped if tuples_json exists) ──
    if tuples_json.exists() and args.tuples_json is not None:
        print(f"[run] STAGE 2: skip (tuples_json={tuples_json} exists)")
        tuples_dict = json.loads(tuples_json.read_text())
    else:
        print(f"[run] STAGE 2: select_features → {tuples_json}")
        tuples_dict = select_features(
            channel=args.channel, regime="diff", sae_dir=sae_dir,
            sae_seeds=args.sae_seeds, mode=args.mode, top_k=args.top_k,
            final_selection=args.final_selection, alphas=args.sel_alphas,
            n_sel=args.n_sel, gen_tokens=args.gen_tokens, device=args.device,
            model=args.model,
        )
        tuples_json.parent.mkdir(parents=True, exist_ok=True)
        tuples_json.write_text(json.dumps(tuples_dict, indent=2))

    # ── 3) Eval (skipped if results_json exists) ──
    if results_json.exists() and args.results_json is not None:
        print(f"[run] STAGE 3: skip (results_json={results_json} exists)")
        results = json.loads(results_json.read_text())
    else:
        print(f"[run] STAGE 3: eval_tuples_json → {results_json}")
        results = eval_tuples_json(
            tuples_dict, sae_dir=sae_dir, eval_alphas=args.eval_alphas,
            n_sel=args.n_sel, n_eval=args.n_eval, gen_tokens=args.gen_tokens,
            eval_seeds=args.eval_seeds, eval_temperature=args.eval_temperature,
            device=args.device, model=args.model,
        )
        results_json.parent.mkdir(parents=True, exist_ok=True)
        results_json.write_text(json.dumps(results, indent=2, default=str))

    print(f"\n[run] DONE  tuples={tuples_json}  results={results_json}")


if __name__ == "__main__":
    main()
