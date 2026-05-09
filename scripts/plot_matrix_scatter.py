"""3×3 Attribution × Intervention scatter: ASR vs JSD(steered, clean).

Reads all results/matrix_cells/<attr>_<intervene>.json files and plots one
point per cell (mean over seeds) plus small per-seed points.

Usage:
    python -m scripts.plot_matrix_scatter
    python -m scripts.plot_matrix_scatter --cells_dir results/matrix_cells \
        --out results/matrix_scatter
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ATTRS      = ["ov", "qk", "qk+ov"]
INTERVENES = ["ov", "qk", "qk+ov"]

ATTR_COLOR = {
    "ov":    "#1f77b4",
    "qk":    "#ff7f0e",
    "qk+ov": "#2ca02c",
}
INTERVENE_MARKER = {
    "ov":    "o",
    "qk":    "s",
    "qk+ov": "^",
}

SCALE    = 2.5
MEAN_S   = 220 * SCALE
SEED_S   = 30  * SCALE
BEST_CELL = ("ov", "ov")


def load_cells(cells_dir: Path) -> dict[tuple[str, str], list[dict]]:
    out: dict[tuple[str, str], list[dict]] = {}
    for attr in ATTRS:
        for intervene in INTERVENES:
            p = cells_dir / f"{attr}_{intervene}.json"
            if not p.exists():
                continue
            rows = json.loads(p.read_text())["results"]
            out[(attr, intervene)] = [
                {"seed": r["seed"],
                 "asr":  r["eval"]["asr"],
                 "jsd":  r["eval"]["jsd_clean"]}
                for r in rows
            ]
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cells_dir", type=Path, default=Path("results/matrix_cells"))
    p.add_argument("--out", type=Path, default=Path("results/matrix_scatter"))
    args = p.parse_args()

    cells = load_cells(args.cells_dir)

    fig, ax = plt.subplots(figsize=(8 * SCALE / 2, 6 * SCALE / 2))
    ax.set_facecolor("white")
    ax.grid(True, color="#cccccc", linewidth=0.7 * SCALE / 2)

    fs_label  = 9  * SCALE
    fs_tick   = 8  * SCALE
    fs_legend = 8  * SCALE
    fs_title  = 10 * SCALE
    fs_annot  = 7  * SCALE

    legend_handles = []

    for attr in ATTRS:
        for intervene in INTERVENES:
            key = (attr, intervene)
            if key not in cells:
                continue
            pts  = cells[key]
            asrs = [pt["asr"] for pt in pts]
            jsds = [pt["jsd"] for pt in pts]
            mu_asr = float(np.mean(asrs))
            mu_jsd = float(np.mean(jsds))
            color  = ATTR_COLOR[attr]
            marker = INTERVENE_MARKER[intervene]
            is_best = key == BEST_CELL

            # mean point — star for ov×ov best
            if is_best:
                ax.scatter([mu_jsd], [mu_asr], s=MEAN_S * 1.3,
                           color=color, marker="*", zorder=5,
                           edgecolors="black", linewidths=SCALE * 0.5)
            else:
                ax.scatter([mu_jsd], [mu_asr], s=MEAN_S,
                           color=color, marker=marker, zorder=4,
                           edgecolors="black", linewidths=SCALE * 0.4)

            label = f"{attr}×{intervene}"
            ax.annotate(
                label, (mu_jsd, mu_asr),
                xytext=(4, 4), textcoords="offset points",
                fontsize=fs_annot, color=color,
                zorder=6,
            )

    # legend entries
    for attr in ATTRS:
        h = plt.scatter([], [], s=MEAN_S * 0.6, color=ATTR_COLOR[attr],
                        marker="o", label=f"attr = {attr}")
        legend_handles.append(h)
    for intervene in INTERVENES:
        h = plt.scatter([], [], s=MEAN_S * 0.6, color="gray",
                        marker=INTERVENE_MARKER[intervene],
                        label=f"intervene = {intervene}")
        legend_handles.append(h)
    h = plt.scatter([], [], s=MEAN_S * 0.9, color=ATTR_COLOR["ov"],
                    marker="*", edgecolors="black",
                    linewidths=SCALE * 0.5, label="ov×ov (best)")
    legend_handles.append(h)

    ax.legend(handles=legend_handles, fontsize=fs_legend,
              loc="upper left", framealpha=0.9)

    # ideal annotation
    ax.annotate("← ideal", xy=(0.595, 0.06), xytext=(0.595, 0.06),
                fontsize=fs_annot * 0.9, color="#888888",
                xycoords="data")
    ax.plot([0.58, 0.595], [0.04, 0.065], color="#aaaaaa",
            linewidth=SCALE * 0.4, zorder=1)

    ax.set_xlabel("JSD(steered, clean)  [bits] — coherence cost  ← lower is better",
                  fontsize=fs_label)
    ax.set_ylabel("ASR — fraction sleepers remaining  ← lower is better",
                  fontsize=fs_label)
    ax.tick_params(labelsize=fs_tick)

    ax.set_title(
        "3×3 Attribution × Intervention matrix\n"
        "mean over SAE seeds 1–4",
        fontsize=fs_title,
    )

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(args.out.with_suffix(f".{ext}"), dpi=150, bbox_inches="tight")
    print(f"[scatter] wrote {args.out}.{{pdf,png}}")


if __name__ == "__main__":
    main()
