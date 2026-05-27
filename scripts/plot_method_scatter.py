"""Scatter of OV / QK / QK+OV / conventional methods at the per-seed Pareto knee
(α* = argmin JSDc s.t. ASR ≤ ε). x = JSDc (coherence cost); y = ASR (suppression).

Lower-left is better. Two variants:
  - --mode mean   : one point per method (mean over good seeds), sample-std bars
  - --mode perseed: one point per (method, seed)

Each method's "winner" per seed is the rank-1 attribution tuple (the top-K=20
result from the diff-regime attribution for that channel; rank-1 = position 0).
For conventional, it's the per-seed downstream-SAE-feature winner picked by
scripts/downstream_baseline.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


METHODS = [
    ("OV",     "results/ov_topk20_gated_all.json",        "topk",
     "#1f77b4", "o"),
    ("QK",     "results/qk_topk20_gated_all.json",        "topk",
     "#d62728", "s"),
    ("conv.",  "results/downstream_baseline_today.json",  "downstream",
     "#444444", "D"),
    # QK+OV last → drawn on top of conv (purple triangles in front of diamonds)
    ("QK+OV",  "results/qkov_topk20_gated_results.json",  "topk",
     "#7f3fbf", "^"),
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
        "legend.frameon":    True,
        "figure.dpi":        120,
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.12,
    })


def pareto_best(alphas: list[float], jsd_c: list[float], asr: list[float],
                eps: float) -> tuple[float, float, float]:
    """Return (α*, jsd_c at α*, asr at α*).

    α* = argmin jsd_c subject to ASR ≤ eps. If no α meets the ASR
    threshold, fall back to the α minimising ASR (tie-break by JSDc) so every
    (method, seed) always has a Pareto-knee point.
    """
    cands = [i for i, a in enumerate(asr) if a <= eps]
    if cands:
        i = min(cands, key=lambda j: jsd_c[j])
    else:
        i = min(range(len(alphas)), key=lambda j: (asr[j], jsd_c[j]))
    return alphas[i], jsd_c[i], asr[i]


def topk_per_seed_points(path: Path, eps: float) -> dict[int, tuple[float, float, float] | None]:
    """For each SAE seed in a topk-schema results JSON, find the Pareto-best point
    of the rank-1 tuple."""
    d = json.loads(path.read_text())
    rows = d["results"]
    seeds = sorted({r["seed"] for r in rows})
    by_seed = {s: [r for r in rows if r["seed"] == s] for s in seeds}
    out: dict[int, tuple | None] = {}
    for s in seeds:
        rank1 = by_seed[s][0]
        alphas = sorted(float(a) for a in rank1["alpha_sweep"])
        sw = rank1["alpha_sweep"]
        jsd_c = [sw[f"{a}"]["jsd_clean"] for a in alphas]
        asr   = [sw[f"{a}"]["asr"]       for a in alphas]
        out[s] = pareto_best(alphas, jsd_c, asr, eps)
    return out


def downstream_per_seed_points(path: Path, eps: float) -> dict[int, tuple[float, float, float] | None]:
    """For each SAE seed in a downstream_baseline.py JSON, find the Pareto-best point."""
    d = json.loads(path.read_text())
    out: dict[int, tuple | None] = {}
    for key, sd in d["per_seed"].items():
        s = int(key.lstrip("s"))
        per_a = sd["per_alpha"]
        alphas = sorted(float(a) for a in per_a)
        jsd_c = [per_a[str(a)]["jsd_clean"] for a in alphas]
        asr   = [per_a[str(a)]["asr"]       for a in alphas]
        out[s] = pareto_best(alphas, jsd_c, asr, eps)
    return out


def collect_method_points(eps: float) -> dict[str, dict[int, tuple | None]]:
    out: dict[str, dict[int, tuple | None]] = {}
    for name, path, kind, _, _ in METHODS:
        loader = topk_per_seed_points if kind == "topk" else downstream_per_seed_points
        out[name] = loader(Path(path), eps)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode",     choices=["mean", "perseed"], required=True)
    p.add_argument("--epsilon",  type=float, default=0.01,
                   help="ASR threshold for α*.")
    p.add_argument("--exclude_seed", type=int, default=2,
                   help="SAE seed to exclude (default 2 — degenerate conv SAE).")
    p.add_argument("--out",      type=Path, required=True,
                   help="Output prefix (no extension).")
    args = p.parse_args()

    setup_style()
    points = collect_method_points(args.epsilon)

    fig, ax = plt.subplots(figsize=(5.2, 4.0))

    for (name, _, _, color, marker) in METHODS:
        # Drop only the excluded seed; every other (method, seed) gets a point.
        per_seed = {s: pt for s, pt in points[name].items() if s != args.exclude_seed}
        if not per_seed:
            continue
        jsdcs = [pt[1] for pt in per_seed.values()]
        asrs  = [pt[2] for pt in per_seed.values()]

        if args.mode == "perseed":
            ax.scatter(jsdcs, asrs, color=color, marker=marker, s=70,
                       alpha=0.85, edgecolors="white", linewidths=1.0,
                       label=f"{name}  (n={len(per_seed)})", zorder=3)
        else:  # mean
            mu_j = mean(jsdcs); mu_a = mean(asrs)
            sd_j = stdev(jsdcs) if len(jsdcs) > 1 else 0.0
            sd_a = stdev(asrs)  if len(asrs)  > 1 else 0.0
            ax.errorbar(mu_j, mu_a, xerr=sd_j, yerr=sd_a,
                        marker=marker, color=color, markersize=11,
                        markeredgecolor="white", markeredgewidth=1.2,
                        capsize=3, elinewidth=1.0, lw=0,
                        label=f"{name}  (n={len(per_seed)})", zorder=3)

    ax.set_xlabel(r"JSD$_\mathrm{clean}$")
    ax.set_ylabel("ASR")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(True, color="#dddddd", lw=0.5)
    ax.set_axisbelow(True)
    ax.set_title("Coherence vs suppression at per-seed optimal $\\alpha$",
                 fontsize=12)
    ax.legend(loc="upper left", framealpha=0.92)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".pdf"))
    fig.savefig(args.out.with_suffix(".png"), dpi=200)
    print(f"wrote {args.out.with_suffix('.pdf')} and {args.out.with_suffix('.png')}")

    # Print the underlying numbers for reference
    print("\nPer-(method, seed) points:")
    for name in points:
        print(f"  {name}:")
        for s in sorted(points[name].keys()):
            pt = points[name][s]
            tag = "" if s != args.exclude_seed else "  [excluded]"
            if pt is None:
                print(f"    seed {s}: no α meets ASR ≤ {args.epsilon}{tag}")
            else:
                a, jc, asr = pt
                print(f"    seed {s}: α={a:.2f}  JSDc={jc:.3f}  ASR={asr*100:.2f}%{tag}")


if __name__ == "__main__":
    main()
