"""Fig-2 candidate: per (method, SAE seed), plot the point from that pair's
(tuple, α) Pareto frontier with the lowest JSDc (best coherence corner).

For OV / QK / QK+OV: pool = top-20 attribution-ranked tuples × 9 α values (180 points).
                   Pareto frontier in (ASR, JSDc); pick the point with the
                   smallest JSDc on the frontier (often trades some ASR away
                   from the lex-min point).
For conv: pool = the per-seed downstream winner × 9 α values.
                Same rule: lowest-JSDc Pareto point.

Each point labelled with the attribution rank of the chosen tuple
(— for conv since it has a single winner per seed).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


METHODS = [
    ("QK",    "results/v3_qk_all.json",   "topk",       "#d62728", "s"),
    ("Conv",  "results/v3_conv_all.json", "downstream", "#444444", "D"),
    ("OV",    "results/v3_ov_all.json",   "topk",       "#1f77b4", "o"),
    # QK+OV last → purple triangles drawn on top
    ("QK+OV", "results/v3_qkov_all.json", "topk",       "#7f3fbf", "^"),
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


def pareto_frontier(points):
    out = []
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
    return out


def topk_lowest_jsdc(path: Path, seed: int):
    """Return (JSDc, ASR, attribution_rank) of the lowest-JSDc Pareto point."""
    d = json.loads(path.read_text())
    rows = [r for r in d["results"] if r["seed"] == seed]
    pts = []
    for ti, r in enumerate(rows):
        for _a, ev in r["alpha_sweep"].items():
            pts.append((ev["asr"], ev["jsd_clean"], ti + 1))
    front = pareto_frontier(pts)
    asr, jc, rank = min(front, key=lambda x: x[1])
    return jc, asr, rank


def downstream_lowest_jsdc(path: Path, seed: int):
    d = json.loads(path.read_text())
    sd = d["per_seed"].get(f"s{seed}")
    if sd is None:
        return None
    pts = [(ev["asr"], ev["jsd_clean"], "—") for ev in sd["per_alpha"].values()]
    front = pareto_frontier(pts)
    asr, jc, rank = min(front, key=lambda x: x[1])
    return jc, asr, rank


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exclude_seed", type=int, default=2)
    p.add_argument("--seeds",        type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    p.add_argument("--out",          type=Path,
                   default=Path("paper/figures/fig2_lowest_jsdc"))
    args = p.parse_args()

    setup_style()
    good_seeds = [s for s in args.seeds if s != args.exclude_seed]

    fig, ax = plt.subplots(figsize=(6.0, 4.5))

    for (name, path, kind, color, marker) in METHODS:
        fn = topk_lowest_jsdc if kind == "topk" else downstream_lowest_jsdc
        xs, ys = [], []
        for s in good_seeds:
            res = fn(Path(path), s)
            if res is None:
                continue
            jc, asr, _rank = res
            xs.append(jc); ys.append(asr)
        ax.scatter(xs, ys, color=color, marker=marker, s=80, alpha=0.9,
                   edgecolors="white", linewidths=1.0,
                   label=name, zorder=3)

    ax.set_xlabel(r"JSD$_\mathrm{clean}$  (coherence measure)")
    ax.set_ylabel("ASR  (suppression)")
    ax.set_xlim(left=0.3)
    ax.grid(True, color="#dddddd", lw=0.5)
    ax.set_axisbelow(True)
    ax.set_title("Most coherent intervention selected from each method's "
                 "Pareto frontier\nacross 6 SAE seeds")

    desired = ["OV", "QK", "QK+OV", "Conv"]
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend([by_label[k] for k in desired if k in by_label],
              [k for k in desired if k in by_label],
              loc="upper left", framealpha=0.92, title="Method")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".pdf"))
    fig.savefig(args.out.with_suffix(".png"), dpi=200)
    print(f"wrote {args.out.with_suffix('.pdf')} and {args.out.with_suffix('.png')}")

    print("\nPer-(method, seed) lowest-JSDc Pareto point:")
    for (name, path, kind, *_rest) in METHODS:
        fn = topk_lowest_jsdc if kind == "topk" else downstream_lowest_jsdc
        for s in args.seeds:
            res = fn(Path(path), s)
            tag = "  [excluded]" if s == args.exclude_seed else ""
            if res is None:
                print(f"  {name}  seed {s}: no data{tag}")
                continue
            jc, asr, rank = res
            print(f"  {name:>6}  seed {s}: JSDc={jc:.3f}  ASR={asr:.4f}  rank={rank}{tag}")


if __name__ == "__main__":
    main()
