"""Plot between-seed variation vs judge temperature for the α=0 noise study.

Reads /tmp/noise_study/judge_scores.json (from judge_temp_sweep.py).

For each (model, judge_temp), and each of the K judge-repeat draws k:
  - assign rollout its k-th sample,
  - per-seed mean over the 32 rollouts (one mean per seed),
  - between-seed SD = SD across the 3 per-seed means.
Average that between-seed SD over k → plotted point ± (SD over k).

Companion "within-text judge SD": per rollout, SD across its K samples,
averaged over all rollouts+seeds — the pure judge contribution at that temp.

Two panels (alignment, coherence); lines for base + finance.
Saved to phase1_results/judge_temp_noise.{png,pdf}
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEMPS = [0.1, 0.5, 1.0]
MODELS = ["base", "finance"]
COLORS = {"base": "#1f77b4", "finance": "#d62728"}


def load():
    return json.loads(Path("/tmp/noise_study/judge_scores.json").read_text())


def between_seed_sd(data, model, temp, kind):
    """Return (mean_over_k between-seed SD, sd_over_k) and within-text judge SD."""
    seeds = [s for s in data[model] if s.isdigit()]
    # arr[seed] -> (n_rollouts, K)
    arr = {}
    for s in seeds:
        scores = data[model][s][str(temp)][kind]   # list of [K] per rollout
        arr[s] = np.array(scores, dtype=float)       # (n_roll, K)
    K = next(iter(arr.values())).shape[1]
    # between-seed SD per draw k
    bs = []
    for k in range(K):
        per_seed_means = [arr[s][:, k].mean() for s in seeds]
        bs.append(np.std(per_seed_means, ddof=1))
    bs = np.array(bs)
    # within-text judge SD: per rollout SD across K, pooled over seeds+rollouts
    within = np.concatenate([arr[s].std(axis=1, ddof=1) for s in seeds])
    return bs.mean(), bs.std(ddof=1), within.mean()


def main():
    data = load()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, kind, title in zip(axes, ["align", "coh"], ["Alignment", "Coherence"]):
        for model in MODELS:
            ys, yerr, withins = [], [], []
            for t in TEMPS:
                m, sd, w = between_seed_sd(data, model, t, kind)
                ys.append(m); yerr.append(sd); withins.append(w)
            ax.errorbar(TEMPS, ys, yerr=yerr, marker="o", capsize=4,
                        color=COLORS[model], label=f"{model} — between-seed SD")
            ax.plot(TEMPS, withins, marker="s", linestyle="--", alpha=0.6,
                    color=COLORS[model], label=f"{model} — within-text judge SD")
        ax.set_xlabel("judge temperature")
        ax.set_ylabel(f"{title} SD (points, 0–100)")
        ax.set_title(f"{title}: noise vs judge temp (α=0, n=32×3 seeds)")
        ax.set_xticks(TEMPS)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    out = Path("phase1_results/judge_temp_noise")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out}.png", dpi=140)
    fig.savefig(f"{out}.pdf")
    print(f"[save] {out}.png / .pdf")

    # also print the table
    print("\n=== between-seed SD (mean±sd over K draws) | within-text judge SD ===")
    for kind, title in [("align", "Alignment"), ("coh", "Coherence")]:
        print(f"\n{title}:")
        print(f"  {'temp':>6} {'model':>8} {'between-seed SD':>18} {'within-text judge SD':>22}")
        for model in MODELS:
            for t in TEMPS:
                m, sd, w = between_seed_sd(data, model, t, kind)
                print(f"  {t:>6} {model:>8} {m:>10.2f} ± {sd:<4.2f} {w:>20.2f}")


if __name__ == "__main__":
    main()
