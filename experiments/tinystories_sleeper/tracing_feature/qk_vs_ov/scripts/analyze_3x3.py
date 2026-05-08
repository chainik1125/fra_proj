"""Analyze the 3x3 Pareto grid: ranking (rows) × intervention (cols)."""

import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent

_p = argparse.ArgumentParser()
_p.add_argument("--input", default=str(HERE.parent / "results" / "pareto_3x3.json"))
_p.add_argument("--output_dir", default=str(HERE.parent / "results"))
_p.add_argument("--output_summary", default=None,
                help="path for the JSON summary (per-cell area + quality). Defaults to <output_dir>/pareto_3x3_summary.json")
_args = _p.parse_args()

with open(_args.input) as f:
    data = json.load(f)
_OUT_DIR = Path(_args.output_dir); _OUT_DIR.mkdir(parents=True, exist_ok=True)

baseline = data["baseline"]
grid = data["grid"]
alphas = data["meta"]["alphas"]
rankings = ["qk", "ov", "union"]
interventions = ["ov", "qk", "all"]

markers = {0.5: "v", 1.0: "o", 2.0: "s", 3.0: "^"}
row_color = {"qk": "#2ca02c", "ov": "#1f77b4", "union": "#d62728"}


def envelope_area(points, baseline_asr, x_max):
    pts = [(0.0, baseline_asr)] + [(max(0.0, p["delta_ce"]), p["asr_16"]) for p in points]
    pts.sort(key=lambda t: t[0])
    best = []
    cur = float("inf")
    for x, y in pts:
        cur = min(cur, y)
        best.append((x, cur))
    area = 0.0
    for i in range(len(best)):
        x_i, y_i = best[i]
        x_next = best[i + 1][0] if i + 1 < len(best) else x_max
        if x_next > x_max:
            x_next = x_max
        if x_next > x_i:
            area += y_i * (x_next - x_i)
        if x_next >= x_max:
            break
    return area


# Find common dce_max across the grid
all_dce = []
for rn in rankings:
    for iv in interventions:
        for p in grid[rn][iv]["per_alpha"]:
            all_dce.append(p["delta_ce"])
dce_max = max(0.1, max(all_dce))

# ---------- Raw table ----------
print(f"\nbaseline ASR={baseline['asr_16']:.3f}  clean_CE={baseline['clean_ce']:.4f}")
print(f"\n=== Per (ranking, intervention, α): (ASR, ΔCE) ===")
header = f"{'rank':<6} {'interv':<6} {'feats':<30s} " + " ".join(f"{'α='+str(a):>14}" for a in alphas)
print(header)
for rn in rankings:
    for iv in interventions:
        cell = grid[rn][iv]
        feats = str(cell["features"])[:28]
        row = f"{rn:<6} {iv:<6} {feats:<30s}"
        for a in alphas:
            p = next(x for x in cell["per_alpha"] if x["alpha"] == a)
            row += f" ({p['asr_16']:.2f},{p['delta_ce']:+.3f})"
        print(row)

# ---------- Area metric table ----------
print(f"\n=== Monotone-envelope area (lower = better), over ΔCE∈[0,{dce_max:.3f}] ===")
print(f"{'rank\\interv':<10} " + " ".join(f"{iv:>12}" for iv in interventions) + "   " + f"{'baseline':>10}")
for rn in rankings:
    row = f"{rn:<10} "
    for iv in interventions:
        a = envelope_area(grid[rn][iv]["per_alpha"], baseline["asr_16"], dce_max)
        q = 1 - a / dce_max
        row += f"  {a:.3f} (q={q:.2f})"
    print(row)

# ---------- Min ASR subject to ΔCE budget ----------
print(f"\n=== Min ASR at ΔCE ≤ budget ===")
budgets = [0.01, 0.05, 0.1, 0.5, 2.0]
for iv in interventions:
    print(f"\n  intervention = {iv}")
    print(f"  {'rank':<8} " + " ".join(f"{'b='+str(b):>14}" for b in budgets))
    for rn in rankings:
        pts = grid[rn][iv]["per_alpha"]
        row = f"  {rn:<8} "
        for b in budgets:
            feas = [p for p in pts if p["delta_ce"] <= b]
            if feas:
                best = min(feas, key=lambda p: p["asr_16"])
                row += f"  ASR={best['asr_16']:.2f}(α={best['alpha']})"
            else:
                row += f"        —       "
        print(row)

# ---------- Plot 3x3 grid ----------
fig, axes = plt.subplots(3, 3, figsize=(14, 11), sharey=True)
for i, rn in enumerate(rankings):
    for j, iv in enumerate(interventions):
        ax = axes[i, j]
        ax.plot([0.0], [baseline["asr_16"]], marker="*", markersize=12,
                color="black", zorder=5)
        cell = grid[rn][iv]
        pts = [(max(0.0, p["delta_ce"]), p["asr_16"], p["alpha"]) for p in cell["per_alpha"]]
        pts.sort(key=lambda t: t[0])
        xs = [0.0] + [p[0] for p in pts]
        ys = [baseline["asr_16"]] + [p[1] for p in pts]
        color = row_color[rn]
        ax.plot(xs, ys, "-", color=color, alpha=0.6, lw=1.5)
        for (x, y, a) in pts:
            ax.scatter([x], [y], marker=markers[a], s=80, color=color,
                       edgecolor="k", linewidth=0.5, zorder=4)
        area = envelope_area(cell["per_alpha"], baseline["asr_16"], dce_max)
        q = 1 - area / dce_max
        ax.set_title(f"rank={rn.upper()}, intervene={iv.upper()}\n"
                     f"quality={q:.2f}  features={cell['features']}", fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(-0.1, dce_max * 1.05)
        ax.grid(True, alpha=0.3)
        if i == 2:
            ax.set_xlabel("ΔCE")
        if j == 0:
            ax.set_ylabel(f"ASR$_{{16}}$")

fig.suptitle("3x3 Pareto grid: feature ranking (rows) × intervention path (cols) at blocks.0\n"
             "α∈{0.5, 1, 2, 3} (▽=0.5, ○=1, ■=2, △=3)", fontsize=11)
fig.tight_layout()
out_png = _OUT_DIR / "pareto_3x3.png"
fig.savefig(out_png, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"\nwrote {out_png}")

# Also a zoomed-in version (ΔCE ≤ 0.3) to see the tight cluster
fig, axes = plt.subplots(3, 3, figsize=(14, 11), sharey=True)
for i, rn in enumerate(rankings):
    for j, iv in enumerate(interventions):
        ax = axes[i, j]
        ax.plot([0.0], [baseline["asr_16"]], marker="*", markersize=12,
                color="black", zorder=5)
        cell = grid[rn][iv]
        pts = [(max(0.0, p["delta_ce"]), p["asr_16"], p["alpha"]) for p in cell["per_alpha"]]
        pts.sort(key=lambda t: t[0])
        xs = [0.0] + [p[0] for p in pts]
        ys = [baseline["asr_16"]] + [p[1] for p in pts]
        color = row_color[rn]
        ax.plot(xs, ys, "-", color=color, alpha=0.6, lw=1.5)
        for (x, y, a) in pts:
            ax.scatter([x], [y], marker=markers[a], s=80, color=color,
                       edgecolor="k", linewidth=0.5, zorder=4)
        ax.set_title(f"rank={rn.upper()}, intervene={iv.upper()}\n"
                     f"features={cell['features']}", fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(-0.01, 0.3)
        ax.grid(True, alpha=0.3)
        if i == 2:
            ax.set_xlabel("ΔCE (zoomed to [0, 0.3])")
        if j == 0:
            ax.set_ylabel(f"ASR$_{{16}}$")
fig.suptitle("3x3 Pareto grid — zoomed ΔCE ∈ [0, 0.3]", fontsize=11)
fig.tight_layout()
out_png2 = _OUT_DIR / "pareto_3x3_zoom.png"
fig.savefig(out_png2, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"wrote {out_png2}")

# ---- summary JSON: per-cell area + quality ----
_summary = {
    "input": _args.input,
    "baseline": baseline,
    "dce_max": dce_max,
    "cells": {},
}
for rn in rankings:
    _summary["cells"][rn] = {}
    for iv in interventions:
        cell = grid[rn][iv]
        area = envelope_area(cell["per_alpha"], baseline["asr_16"], dce_max)
        q = 1 - area / dce_max
        _summary["cells"][rn][iv] = {
            "features": cell["features"],
            "area": area,
            "quality": q,
        }
_sum_path = Path(_args.output_summary) if _args.output_summary else (_OUT_DIR / "pareto_3x3_summary.json")
_sum_path.write_text(json.dumps(_summary, indent=2))
print(f"wrote {_sum_path}")
