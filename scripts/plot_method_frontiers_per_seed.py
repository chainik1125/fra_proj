"""For each (method, SAE seed): plot one point per top-K tuple at the tuple's
Pareto-knee in (JSDc, ASR) space across its α-sweep.

Pareto-knee = α* = argmin JSDc s.t. ASR ≤ ε; fallback to argmin (ASR, JSDc)
when no α meets ε. So every tuple collapses to a single (JSDc, ASR) point,
giving exactly 20 points per panel for OV / QK / QK+OV (and 1 per panel for
conv, which has a single per-seed winner).

Outputs 4 PDFs (one per method, 1×5 panel grid — one panel per good SAE seed,
default excludes seed 2).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


METHODS = [
    ("OV",     "results/ov_topk20_gated_all.json",        "topk"),
    ("QK",     "results/qk_topk20_gated_all.json",        "topk"),
    ("QK+OV",  "results/qkov_topk20_gated_results.json",  "topk"),
    ("conv.",  "results/downstream_baseline_today.json",  "downstream"),
]


def setup_style() -> None:
    mpl.rcParams.update({
        "font.family":       "sans-serif",
        "font.sans-serif":   ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size":         10,
        "axes.titlesize":    10,
        "axes.labelsize":    10,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.linewidth":    1.0,
        "axes.edgecolor":    "#222222",
        "xtick.labelsize":   8.5,
        "ytick.labelsize":   8.5,
        "legend.fontsize":   7.5,
        "figure.dpi":        120,
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.10,
    })


def pareto_knee(points: list[tuple[float, float, float]],
                eps: float = 0.01) -> tuple[float, float, float]:
    """Single representative point of a tuple's α-sweep.
    argmin JSDc s.t. ASR ≤ ε; fallback to argmin (ASR, JSDc) when none meet ε.
    Returns (α*, JSDc, ASR)."""
    cands = [i for i, (_, _, asr) in enumerate(points) if asr <= eps]
    if cands:
        i = min(cands, key=lambda j: points[j][1])
    else:
        i = min(range(len(points)), key=lambda j: (points[j][2], points[j][1]))
    return points[i]


def topk_tuple_trajectories(path: Path) -> dict[int, list[list[tuple[float, float, float]]]]:
    """{seed: [tuple0_traj, tuple1_traj, ...]} where each traj = list of (α, JSDc, ASR)."""
    d = json.loads(path.read_text())
    rows = d["results"]
    by_seed: dict[int, list] = {}
    for r in rows:
        sw = r["alpha_sweep"]
        alphas = sorted(float(a) for a in sw)
        traj = [(a, sw[f"{a}"]["jsd_clean"], sw[f"{a}"]["asr"]) for a in alphas]
        by_seed.setdefault(r["seed"], []).append(traj)
    return by_seed


def downstream_tuple_trajectories(path: Path) -> dict[int, list[list[tuple[float, float, float]]]]:
    """conv: one tuple per seed (the per-seed downstream winner)."""
    d = json.loads(path.read_text())
    out: dict[int, list[list]] = {}
    for key, sd in d["per_seed"].items():
        s = int(key.lstrip("s"))
        per_a = sd["per_alpha"]
        alphas = sorted(float(a) for a in per_a)
        traj = [(a, per_a[str(a)]["jsd_clean"], per_a[str(a)]["asr"]) for a in alphas]
        out[s] = [traj]
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exclude_seed", type=int, default=2)
    p.add_argument("--out_dir",      type=Path, default=Path("paper/figures"))
    args = p.parse_args()

    setup_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for name, path, kind in METHODS:
        loader = topk_tuple_trajectories if kind == "topk" else downstream_tuple_trajectories
        by_seed = loader(Path(path))
        good_seeds = sorted(s for s in by_seed if s != args.exclude_seed)

        n = len(good_seeds)
        fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 3.0),
                                  sharex=True, sharey=True)
        if n == 1:
            axes = [axes]

        # Per-tuple colours from a perceptually uniform cmap (max 20).
        n_tuples_max = max(len(by_seed[s]) for s in good_seeds)
        cmap = plt.colormaps["viridis"]
        tuple_colors = [cmap(i / max(n_tuples_max - 1, 1))
                        for i in range(n_tuples_max)]

        for ax, s in zip(axes, good_seeds):
            trajs = by_seed[s]
            for ti, traj in enumerate(trajs):
                _, jc, asr = pareto_knee(traj)
                color = tuple_colors[ti % len(tuple_colors)]
                ax.scatter([jc], [asr], color=color, s=28,
                           alpha=0.9, edgecolors="white", linewidths=0.6,
                           zorder=3)
            ax.set_title(f"seed {s}  ({len(trajs)} tuple{'s' if len(trajs) != 1 else ''})")
            ax.set_xlim(left=-0.02)
            ax.set_ylim(-0.02, 1.05)
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
            ax.grid(True, color="#dddddd", lw=0.5)
            ax.set_axisbelow(True)

        for ax in axes:
            ax.set_xlabel(r"JSD$_\mathrm{clean}$")
        axes[0].set_ylabel("ASR")

        fig.suptitle(f"{name}: one point per top-K tuple at its Pareto-knee "
                     "(argmin JSDc s.t. ASR ≤ 1%)",
                     fontsize=11, y=1.02)
        fig.tight_layout()

        slug = name.replace("+", "plus").replace(".", "").lower()
        out = args.out_dir / f"per_tuple_frontiers_{slug}"
        fig.savefig(out.with_suffix(".pdf"))
        fig.savefig(out.with_suffix(".png"), dpi=180)
        print(f"wrote {out.with_suffix('.pdf')}")
        plt.close(fig)


if __name__ == "__main__":
    main()
