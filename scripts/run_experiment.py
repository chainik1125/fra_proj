"""End-to-end experiment pipeline.

Train SAEs → feature selection → α-sweep eval → Pareto curve plots.

Usage:
    uv run -m scripts.run_experiment                        # defaults
    uv run -m scripts.run_experiment --eval_mode both       # single + full-set
    uv run -m scripts.run_experiment --sae_steps 50000      # 50k-step SAEs
    uv run -m scripts.run_experiment --plot                 # also produce figures
    uv run -m scripts.run_experiment --force                # rerun even if --out exists

Reproduce results/jamie_experiment.json + figures (panel_seeds_jamie.pdf):
    uv run -m scripts.run_experiment \\
        --eval_mode both \\
        --alphas 0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 \\
        --screen_alphas 2.0 4.0 \\
        --out results/jamie_experiment.json \\
        --plot
"""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from scripts.train_all_saes_6seeds import sae_paths


def main() -> None:
    p = argparse.ArgumentParser(
        description="Train SAEs → feature selection → eval → plots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--sae_steps", type=int, default=4_000,
                   help="SAE training steps. Determines checkpoint paths.")
    p.add_argument("--selection_method", default="jamie", choices=["jamie", "ketan"])
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--eval_mode", default="single", choices=["single", "set", "both"])
    p.add_argument("--alphas", nargs="+", type=float,
                   default=[0.0, 0.5, 1.0, 2.0, 4.0])
    p.add_argument("--screen_alphas", nargs="+", type=float, default=[2.0, 4.0])
    p.add_argument("--sae_seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--target_feature", type=int, default=579)
    p.add_argument("--no_downstream", action="store_true",
                   help="Skip the downstream f579 baseline.")
    p.add_argument("--n_sel", type=int, default=100)
    p.add_argument("--n_eval", type=int, default=200)
    p.add_argument("--n_gen_ce", type=int, default=100)
    p.add_argument("--gen_tokens", type=int, default=16)
    p.add_argument("--eval_seeds", nargs="+", type=int, default=list(range(50)),
                   help="Seed pool for adaptive RNR generation.")
    p.add_argument("--target_rnr_rows", type=int, default=100,
                   help="Target sleeper-removed rows per eval point.")
    p.add_argument("--eval_temperature", type=float, default=1.0)
    p.add_argument("--eval_metrics", nargs="+", default=["recovery_noise_ratio"],
                   help="Metrics to record. Default: recovery_noise_ratio only. "
                        "Add gen_ce_ratio to also record gen-CE ratio.")
    p.add_argument("--out", type=Path, default=Path("results/feature_set_pipeline.json"))
    p.add_argument("--plot", action="store_true",
                   help="After eval, run plot_pareto_curves on the output JSON.")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if --out already exists.")
    args = p.parse_args()

    uv = ["uv", "run"]
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}

    seeds_dir = sae_paths(args.sae_steps)
    mid_path = seeds_dir / "sae_resid_mid_s0.pt"   # legacy: per-seed-0 stands in for "shared"

    # 1. Train SAEs (idempotent — skips existing checkpoints).
    print(f"[run_experiment] step 1: train SAEs (n_steps={args.sae_steps})")
    subprocess.run([*uv, "-m", "scripts.train_all_saes_6seeds",
                    "--n_steps", str(args.sae_steps)], check=True, env=env)

    # 2. Feature-set pipeline: attribution → selection → α-sweep eval.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists() and not args.force:
        print(f"[run_experiment] {args.out} exists; pass --force to re-run")
    else:
        print(f"[run_experiment] step 2: feature_set_pipeline → {args.out}")
        cmd = [
            *uv, "-m", "scripts.feature_set_pipeline",
            "--selection_method", args.selection_method,
            "--top_k", str(args.top_k),
            "--eval_mode", args.eval_mode,
            "--alphas",        *[str(a) for a in args.alphas],
            "--screen_alphas", *[str(a) for a in args.screen_alphas],
            "--sae_seeds",     *[str(s) for s in args.sae_seeds],
            "--sae_ln1_dir",   str(seeds_dir),
            "--sae_mid",       str(mid_path),
            "--target_feature", str(args.target_feature),
            "--n_sel", str(args.n_sel),
            "--n_eval", str(args.n_eval),
            "--n_gen_ce", str(args.n_gen_ce),
            "--gen_tokens", str(args.gen_tokens),
            "--eval_seeds",    *[str(s) for s in args.eval_seeds],
            "--target_rnr_rows", str(args.target_rnr_rows),
            "--eval_temperature", str(args.eval_temperature),
            "--out", str(args.out),
        ]
        if args.no_downstream:
            cmd.append("--no-include_downstream")
        cmd += ["--eval_metrics", *args.eval_metrics]
        subprocess.run(cmd, check=True, env=env)

    # 3. Plot Pareto curves + bar chart if requested.
    if args.plot:
        print("[run_experiment] step 3: plot_pareto_curves (mainline)")
        subprocess.run([
            *uv, "-m", "scripts.plot_pareto_curves",
            "--jamie_in", str(args.out),
            "--mainline",
        ], check=True, env=env)
        print("[run_experiment] step 3b: plot_bar_chart")
        subprocess.run([
            *uv, "-m", "scripts.plot_bar_chart",
            "--jamie_in", str(args.out),
        ], check=True, env=env)


if __name__ == "__main__":
    main()
