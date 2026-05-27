"""Per-method frontier plots: each shows (JSDc, ASR) trajectories under steering
strength sweep, one curve per SAE seed.

Outputs 4 separate PDFs (one per method: OV, QK, QK+OV, conv.) into the given
output directory. Each plot has up to 5 curves (one per good SAE seed; default
excludes seed 2). Markers at each α point; thin line connecting them along the
α sweep.
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


# Distinct per-seed colours (chosen for visibility on white).
SEED_COLORS = {
    0: "#1f77b4",   # blue
    1: "#2ca02c",   # green
    2: "#d62728",   # red
    3: "#9467bd",   # purple
    4: "#ff7f0e",   # orange
    5: "#17becf",   # teal
}


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
        "legend.fontsize":   9,
        "legend.frameon":    True,
        "figure.dpi":        120,
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.12,
    })


def topk_trajectories(path: Path) -> dict[int, list[tuple[float, float, float]]]:
    """Per-SAE-seed list of (α, JSDc, ASR) sorted by α."""
    d = json.loads(path.read_text())
    rows = d["results"]
    seeds = sorted({r["seed"] for r in rows})
    by_seed = {s: [r for r in rows if r["seed"] == s] for s in seeds}
    out: dict[int, list] = {}
    for s in seeds:
        rank1 = by_seed[s][0]
        sw = rank1["alpha_sweep"]
        alphas = sorted(float(a) for a in sw)
        out[s] = [(a, sw[f"{a}"]["jsd_clean"], sw[f"{a}"]["asr"]) for a in alphas]
    return out


def downstream_trajectories(path: Path) -> dict[int, list[tuple[float, float, float]]]:
    d = json.loads(path.read_text())
    out: dict[int, list] = {}
    for key, sd in d["per_seed"].items():
        s = int(key.lstrip("s"))
        per_a = sd["per_alpha"]
        alphas = sorted(float(a) for a in per_a)
        out[s] = [(a, per_a[str(a)]["jsd_clean"], per_a[str(a)]["asr"]) for a in alphas]
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exclude_seed", type=int, default=2)
    p.add_argument("--out_dir",      type=Path, default=Path("paper/figures"))
    p.add_argument("--prefix",       type=str, default="frontier")
    args = p.parse_args()

    setup_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for name, path, kind in METHODS:
        loader = topk_trajectories if kind == "topk" else downstream_trajectories
        trajs = loader(Path(path))
        good_seeds = sorted(s for s in trajs if s != args.exclude_seed)

        fig, ax = plt.subplots(figsize=(5.2, 4.0))
        for s in good_seeds:
            traj = trajs[s]
            xs = [t[1] for t in traj]
            ys = [t[2] for t in traj]
            color = SEED_COLORS.get(s, "#888888")
            ax.plot(xs, ys, "-o", color=color, lw=1.1, ms=4,
                    alpha=0.9, label=f"seed {s}")

        ax.set_xlabel(r"JSD$_\mathrm{clean}$")
        ax.set_ylabel("ASR")
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlim(left=-0.02)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(True, color="#dddddd", lw=0.5)
        ax.set_axisbelow(True)
        ax.set_title(f"{name}: per-seed steering-strength sweep")
        ax.legend(loc="lower left", framealpha=0.92)

        fig.tight_layout()
        # Filename-safe method name
        slug = name.replace("+", "plus").replace(".", "").lower()
        out = args.out_dir / f"{args.prefix}_{slug}"
        fig.savefig(out.with_suffix(".pdf"))
        fig.savefig(out.with_suffix(".png"), dpi=200)
        print(f"wrote {out.with_suffix('.pdf')}")
        plt.close(fig)


if __name__ == "__main__":
    main()
