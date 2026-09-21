"""Poster figure: F603 vs the other top-10 FRA-QK features, per 14B EM finetune.

For each finetune (financial / medical / sports), download the FRA-QK gran1
per-feature combined from HF (qwen14b/grid_diff*), compute each feature's
Delta-align@coh50 (grid_metrics: per-seed window max-min, mean across seeds),
and plot the top-10 features as bars with F603 highlighted.

Sports lives as per-seed partials (_seed42/_seed123) -> merged by scale into
2-seed per_seed arrays before metrics (same convention as build_grid_results).

Usage: python3 plot_f603_top10_poster.py [--out <path.pdf>]   (needs HF auth)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import grid_metrics as gm  # noqa: E402

from huggingface_hub import hf_hub_download  # noqa: E402

REPO = "dmanningcoe/fra-phase1-steering-data"
CELLS = {
    "financial": ["qwen14b/grid_diff/fra-qk_ln1_gran1/"
                  "gpt4o_combined_L24_ln1_arditi_qwen14b_grid_fra-qk_gran1_finance.json"],
    "medical":   ["qwen14b/grid_diff_medical/fra-qk_ln1_gran1/"
                  "gpt4o_combined_L24_ln1_arditi_qwen14b_grid_fra-qk_gran1_medical.json"],
    "sports":    ["qwen14b/grid_diff_sports/fra-qk_ln1_gran1/"
                  "gpt4o_combined_L24_ln1_arditi_qwen14b_grid_fra-qk_gran1_sports_seed42.json",
                  "qwen14b/grid_diff_sports/fra-qk_ln1_gran1/"
                  "gpt4o_combined_L24_ln1_arditi_qwen14b_grid_fra-qk_gran1_sports_seed123.json"],
}
HERO = "feat_F603"
ACCENT = "#0072B2"; GREY = "#B8BCC2"
FLOOR = 50


def setup_style() -> None:
    mpl.rcParams.update({
        "font.family":       "sans-serif",
        "font.sans-serif":   ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size":         13,
        "axes.titlesize":    14,
        "axes.labelsize":    13,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.linewidth":    1.1,
        "axes.edgecolor":    "#222222",
        "xtick.labelsize":   10,
        "ytick.labelsize":   12,
        "figure.dpi":        120,
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.12,
    })


def merge_seed_partials(objs):
    """Merge per-(em,seed) combined dicts into one with 2-entry per_seed arrays."""
    methods = sorted(set().union(*[set(o) for o in objs]))
    out = {}
    for m in methods:
        idx = [{round(float(e["scale"]), 6): e for e in o.get(m, {}).get("by_alpha", [])}
               for o in objs]
        scales, seen = [], set()
        for o in objs:
            for e in o.get(m, {}).get("by_alpha", []):
                k = round(float(e["scale"]), 6)
                if k not in seen:
                    seen.add(k); scales.append(e["scale"])
        by_alpha = []
        for sc in scales:
            k = round(float(sc), 6)
            psa, psc = [], []
            for d in idx:
                e = d.get(k)
                psa.append((e or {}).get("per_seed_alignment", [None])[0])
                psc.append((e or {}).get("per_seed_coherence", [None])[0])
            by_alpha.append({"scale": sc,
                             "n_seeds": len(psa),
                             "per_seed_alignment": psa,
                             "per_seed_coherence": psc})
        out[m] = {"by_alpha": by_alpha}
    return out


def cell_deltas(files):
    objs = [json.load(open(hf_hub_download(REPO, f, repo_type="dataset"))) for f in files]
    comb = objs[0] if len(objs) == 1 else merge_seed_partials(objs)
    rows = []
    for m, v in comb.items():
        if not m.startswith("feat_F"):
            continue
        st = gm.recompute_method(v["by_alpha"])["delta_coh"][FLOOR]
        if st["mean"] is not None:
            rows.append((m.replace("feat_", ""), st["mean"]))
    rows.sort(key=lambda r: -r[1])
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("fig_em_f603_top10.pdf"))
    args = p.parse_args()
    setup_style()

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 6.2), sharey=True)
    for ax, (ft, files) in zip(axes, CELLS.items()):
        rows = cell_deltas(files)
        top = rows[:10]
        if not any(f == "F603" for f, _ in top):  # make sure the hero is shown
            top = top[:9] + [r for r in rows if r[0] == "F603"]
        feats = [f for f, _ in top]
        vals = [v for _, v in top]
        colors = [ACCENT if f == "F603" else GREY for f in feats]
        bars = ax.bar(range(len(top)), vals, color=colors, width=0.72, zorder=3)
        for i, (f, v) in enumerate(top):
            if f == "F603":
                rank = next(j for j, (g, _) in enumerate(rows) if g == "F603") + 1
                ax.text(i, v + 1.2, f"F603\n#{rank}", ha="center", va="bottom",
                        fontsize=11, fontweight="bold", color=ACCENT)
        ax.set_xticks(range(len(top)),
                      [f"F{f[1:]}" if f != "F603" else "" for f in feats],
                      rotation=60, ha="right")
        ax.set_title(ft, fontsize=14)
        ax.grid(axis="y", color="#E3E3E3", lw=0.8, zorder=0)
        print(ft, "top10:", top)
    axes[0].set_ylabel("Alignment steering range")
    fig.suptitle("QK-FRA finds a universal misalignment feature in Qwen-14B",
                 fontsize=15, y=1.04)

    for ext in ("pdf", "png"):
        out = args.out.with_suffix("." + ext)
        fig.savefig(out)
        print("wrote", out)
    plt.close(fig)


if __name__ == "__main__":
    main()
