"""Single summary scatter: per (method, SAE seed), plot the Pareto frontier
across all (tuple, α) combinations in (ASR, JSDc) space.

For OV / QK / QK+OV: pool = top-20 attribution-ranked tuples × 9 α values (180 points).
For conv: pool = the per-seed downstream winner × 9 α values (9 points).

Pareto-optimal subset (minimising both ASR and JSDc) is plotted as a connected
staircase per (method, seed). Method = color + marker; seeds share the same
style (5 frontiers per method).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


METHODS = [
    ("QK",    "results/v3_qk_all.json",   "#d62728", "s", "topk"),
    ("Conv",  "results/v3_conv_all.json", "#444444", "D", "downstream"),
    ("OV",    "results/v3_ov_all.json",   "#1f77b4", "o", "topk"),
    # QK+OV last → purple triangles drawn on top
    ("QK+OV", "results/v3_qkov_all.json", "#7f3fbf", "^", "topk"),
]


def setup_style() -> None:
    mpl.rcParams.update({
        "font.family":       "sans-serif",
        "font.sans-serif":   ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size":         11,
        "axes.titlesize":    12,
        "axes.labelsize":    12,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.linewidth":    1.0,
        "axes.edgecolor":    "#222222",
        "xtick.labelsize":   10,
        "ytick.labelsize":   10,
        "legend.fontsize":   10,
        "figure.dpi":        120,
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.12,
    })


def pareto_frontier(points: list[tuple[float, float, int]]
                    ) -> list[tuple[float, float, int]]:
    """Non-dominated subset of (ASR, JSDc, rank) points (minimising first two).
    Returned sorted by ASR ascending. Rank annotation carried through."""
    out: list[tuple[float, float, int]] = []
    for p in points:
        dominated = False
        for q in points:
            if q is p:
                continue
            if q[0] <= p[0] and q[1] <= p[1] and (q[0] < p[0] or q[1] < p[1]):
                dominated = True
                break
        if not dominated:
            out.append(p)
    return sorted(out, key=lambda x: x[0])


def topk_points(path: Path, seed: int) -> list[tuple[float, float, int]]:
    """(ASR, JSDc, attribution_rank_1indexed) for every (tuple, α)."""
    d = json.loads(path.read_text())
    rows = [r for r in d["results"] if r["seed"] == seed]
    pts: list[tuple[float, float, int]] = []
    for ti, r in enumerate(rows):
        for _a, ev in r["alpha_sweep"].items():
            pts.append((ev["asr"], ev["jsd_clean"], ti + 1))
    return pts


def downstream_points(path: Path, seed: int) -> list[tuple[float, float, int]]:
    """(ASR, JSDc, dummy_rank=0) for every α of the conv per-seed winner."""
    d = json.loads(path.read_text())
    sd = d["per_seed"].get(f"s{seed}")
    if sd is None:
        return []
    return [(ev["asr"], ev["jsd_clean"], 0) for ev in sd["per_alpha"].values()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exclude_seed", type=int, default=-1)
    p.add_argument("--seeds",        type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    p.add_argument("--out",          type=Path,
                   default=Path("paper/figures/pareto_frontiers_summary"))
    args = p.parse_args()

    setup_style()
    good_seeds = [s for s in args.seeds if s != args.exclude_seed]

    fig, ax = plt.subplots(figsize=(6.0, 4.5))

    for (name, path, color, marker, kind) in METHODS:
        loader = topk_points if kind == "topk" else downstream_points
        first_label = True
        for s in good_seeds:
            pts = loader(Path(path), s)
            if not pts:
                continue
            front = pareto_frontier(pts)
            xs = [jc for (_asr, jc, _r) in front]
            ys = [asr for (asr, _jc, _r) in front]
            label = name if first_label else None
            ax.plot(xs, ys, "-", color=color, lw=1.0, alpha=0.55, zorder=2)
            ax.scatter(xs, ys, color=color, marker=marker, s=42,
                       alpha=0.9, edgecolors="white", linewidths=0.7,
                       label=label, zorder=3)
            first_label = False

    ax.set_xlabel(r"JSD$_\mathrm{clean}$  (coherence)")
    ax.set_ylabel("ASR  (suppression)")
    ax.set_xlim(left=0.3)
    ax.grid(True, color="#dddddd", lw=0.5)
    ax.set_axisbelow(True)
    ax.set_title("Per-seed Pareto frontiers over (tuple, α)")

    desired = ["OV", "QK", "QK+OV", "Conv"]
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend([by_label[k] for k in desired if k in by_label],
              [k for k in desired if k in by_label],
              loc="upper left", framealpha=0.92)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".pdf"))
    fig.savefig(args.out.with_suffix(".png"), dpi=200)
    print(f"wrote {args.out.with_suffix('.pdf')} and {args.out.with_suffix('.png')}")

    # Per-(method, seed) frontier sizes + ranks
    print("\nPareto frontiers per (method, seed):")
    for (name, path, *_rest) in METHODS:
        kind = _rest[-1]
        loader = topk_points if kind == "topk" else downstream_points
        for s in args.seeds:
            pts = loader(Path(path), s)
            tag = "  [excluded]" if s == args.exclude_seed else ""
            if not pts:
                print(f"  {name}  seed {s}: no data{tag}")
                continue
            front = pareto_frontier(pts)
            front_str = ", ".join(
                f"(ASR={asr:.4f}, JSDc={jc:.3f}, rank={r})"
                for (asr, jc, r) in front
            )
            print(f"  {name:>6}  seed {s}: {front_str}{tag}")


if __name__ == "__main__":
    main()
