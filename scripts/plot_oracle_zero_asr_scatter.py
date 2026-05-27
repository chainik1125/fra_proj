"""20-point oracle scatter — best (JSDc, ASR) per (method, seed), choosing the
top-K tuple and α that achieves lexicographically minimum (ASR, JSDc).

For OV / QK / QK+OV: pool = top-20 attribution-ranked tuples × 9 α values.
                   pick (tuple, α) = argmin (ASR, JSDc); label = tuple rank (1-indexed).
For conv: pool = the single per-seed downstream winner × 9 α values.
                pick α = argmin (ASR, JSDc); label = "—".

Every (method, seed) always gets a point (lex-min rule never fails).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


METHODS = [
    ("QK",     "results/qk_topk20_gated_all.json",        "topk",
     "#d62728", "s"),
    ("conv.",  "results/downstream_baseline_today.json",  "downstream",
     "#444444", "D"),
    ("QK+OV",  "results/qkov_topk20_gated_results.json",  "topk",
     "#7f3fbf", "^"),
    # OV last → blue dots drawn on top of black diamonds (and purple triangles)
    ("OV",     "results/ov_topk20_gated_all.json",        "topk",
     "#1f77b4", "o"),
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


def topk_oracle(path: Path, seed: int) -> tuple[float, float, int]:
    """Across all top-K tuples × α for this seed: lex-min (ASR, JSDc).
    Returns (JSDc, ASR, tuple_rank_1indexed). Always succeeds."""
    d = json.loads(path.read_text())
    rows = [r for r in d["results"] if r["seed"] == seed]
    best = None  # (asr, jsd, rank)
    for ti, r in enumerate(rows):
        for _a, ev in r["alpha_sweep"].items():
            key = (ev["asr"], ev["jsd_clean"])
            if best is None or key < (best[0], best[1]):
                best = (ev["asr"], ev["jsd_clean"], ti + 1)
    assert best is not None
    asr, jc, rank = best
    return jc, asr, rank


def downstream_oracle(path: Path, seed: int) -> tuple[float, float, str] | None:
    """For conv: one winner per seed, sweep α: lex-min (ASR, JSDc). Always succeeds
    (unless the seed isn't in the JSON)."""
    d = json.loads(path.read_text())
    sd = d["per_seed"].get(f"s{seed}")
    if sd is None:
        return None
    per_a = sd["per_alpha"]
    best = None  # (asr, jsd)
    for _, ev in per_a.items():
        key = (ev["asr"], ev["jsd_clean"])
        if best is None or key < (best[0], best[1]):
            best = (ev["asr"], ev["jsd_clean"])
    assert best is not None
    asr, jc = best
    return jc, asr, "—"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exclude_seed", type=int, default=2)
    p.add_argument("--seeds",        type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    p.add_argument("--out",          type=Path,
                   default=Path("paper/figures/oracle_zero_asr_scatter"))
    args = p.parse_args()

    setup_style()
    good_seeds = [s for s in args.seeds if s != args.exclude_seed]

    fig, ax = plt.subplots(figsize=(6.0, 4.5))

    # For label offsets so they don't collide with the marker.
    label_dx, label_dy = 0.006, 0.0
    label_kwargs = dict(fontsize=8.5, ha="left", va="center", color="#222222")

    for (name, path, kind, color, marker) in METHODS:
        oracle = topk_oracle if kind == "topk" else downstream_oracle
        xs, ys, labels = [], [], []
        for s in good_seeds:
            res = oracle(Path(path), s)
            if res is None:
                continue
            jc, asr, rank = res
            xs.append(jc); ys.append(asr); labels.append(str(rank))
        ax.scatter(xs, ys, color=color, marker=marker, s=80, alpha=0.9,
                   edgecolors="white", linewidths=1.0,
                   label=name, zorder=3)
        for x, y, lbl in zip(xs, ys, labels):
            ax.annotate(lbl, (x + label_dx, y + label_dy), **label_kwargs)

    ax.set_xlabel(r"JSD$_\mathrm{clean}$  (coherence)")
    ax.set_ylabel("ASR  (suppression)")
    ax.set_xlim(left=0.3)
    ax.grid(True, color="#dddddd", lw=0.5)
    ax.set_axisbelow(True)
    ax.set_title("Oracle pick: lex-min (ASR, JSDc) over top-K tuples × α,\n"
                 "labelled by attribution rank")

    # Custom legend order: OV, QK, QK+OV, conv.
    desired = ["OV", "QK", "QK+OV", "conv."]
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

    # Also print the per-(method, seed) picks for inspection.
    print("\nPer-(method, seed) oracle picks:")
    for (name, path, kind, *_rest) in METHODS:
        oracle = topk_oracle if kind == "topk" else downstream_oracle
        print(f"  {name}:")
        for s in args.seeds:
            res = oracle(Path(path), s)
            tag = "  [excluded]" if s == args.exclude_seed else ""
            if res is None:
                print(f"    seed {s}: no (tuple, α) achieves ASR=0 exactly{tag}")
            else:
                jc, asr, rank = res
                print(f"    seed {s}: JSDc={jc:.3f}  ASR={asr*100:.2f}%  rank={rank}{tag}")


if __name__ == "__main__":
    main()
