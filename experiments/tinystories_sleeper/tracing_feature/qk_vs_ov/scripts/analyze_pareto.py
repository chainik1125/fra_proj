"""Analyze + plot the Pareto results from pareto_ov_vs_qk.py.

Computes for each channel:
- Points at each α
- Dominance relation with other channels
- Simple scalar: min ASR achieved at ΔCE ≤ budget_options = {0.01, 0.05, 0.1}
- Area metric: trapezoidal area under the (ΔCE-monotonic) frontier including baseline.
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
with open(HERE.parent / "results" / "pareto_ov_vs_qk.json") as f:
    data = json.load(f)

baseline = data["baseline"]
channels = data["channels"]
alphas = data["meta"]["alphas"]

colors = {
    "sweep":       "#d62728",
    "random":      "#7f7f7f",
    "ov_onestage": "#1f77b4",
    "ov_onestage3":"#17becf",
    "qk_top1":     "#2ca02c",
    "qk_top2":     "#9467bd",
    "qk_top3":     "#e377c2",
    "qk_top5":     "#ff7f0e",
}
markers = {0.5: "v", 1.0: "o", 2.0: "s", 3.0: "^"}

# ------------------------------------------------------------------
# Plot — side-by-side: zoomed-in (ΔCE ≤ 0.3) + full range
# ------------------------------------------------------------------
fig, (ax_zoom, ax_full) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1, 1.2]})

def draw(ax):
    ax.plot([0.0], [baseline["asr_16"]], marker="*", markersize=15, color="black",
            label=f"baseline: ASR={baseline['asr_16']:.2f}", zorder=5)
    for name, ch in channels.items():
        pts = [(max(0.0, p["delta_ce"]), p["asr_16"], p["alpha"]) for p in ch["per_alpha"]]
        pts.sort(key=lambda t: t[0])
        xs = [0.0] + [p[0] for p in pts]
        ys = [baseline["asr_16"]] + [p[1] for p in pts]
        color = colors.get(name, "k")
        ax.plot(xs, ys, "-", color=color, alpha=0.5, lw=1.2)
        for (x, y, a) in pts:
            ax.scatter([x], [y], marker=markers[a], s=70, color=color, edgecolor="k", linewidth=0.5)
        ax.plot([], [], "-o", color=color, label=f"{name}: {ch['features']}")
    ax.set_xlabel(r"$\Delta$ clean-continuation CE (higher = worse coherence)")
    ax.set_ylabel("ASR$_{16}$ (sleeper survival)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

draw(ax_zoom)
ax_zoom.set_xlim(-0.01, 0.3)
ax_zoom.set_title(r"Zoomed ($\Delta$CE $\leq 0.3$)")

draw(ax_full)
ax_full.set_xlim(-0.1, None)
ax_full.set_title(r"Full range")
ax_full.legend(loc="upper right", fontsize=7, bbox_to_anchor=(1.45, 1.0))

fig.suptitle(r"Pareto tradeoff: sleeper suppression vs coherence — "
             r"steering $\alpha\in\{0.5, 1, 2, 3\}$ at blocks.0.ln1.hook_normalized"
             + "\n(markers: ▽=0.5, ○=1, ■=2, △=3)")
fig.tight_layout()
out_png = HERE.parent / "results" / "pareto.png"
fig.savefig(out_png, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"wrote {out_png}")

# ------------------------------------------------------------------
# Tables
# ------------------------------------------------------------------
print("\n=== Per-channel raw values ===")
header = f"{'channel':<14s} {'features':<20s}"
for a in alphas:
    header += f"  α={a}"
print(header)
for name, ch in channels.items():
    feats = str(ch["features"])[:18]
    row = f"{name:<14s} {feats:<20s}"
    for a in alphas:
        p = next(x for x in ch["per_alpha"] if x["alpha"] == a)
        row += f"  ({p['asr_16']:.2f}, {p['delta_ce']:+.3f})"
    print(row)
print("(cells: (ASR_16, ΔCE))")

# ------------------------------------------------------------------
# Summary scalar: for each channel, minimum ASR subject to ΔCE ≤ budget.
# ------------------------------------------------------------------
print("\n=== Min ASR subject to ΔCE ≤ budget (α ∈ {0.5, 1, 2, 3}) ===")
budgets = [0.01, 0.05, 0.1, 0.5]
print(f"{'channel':<14s} " + " ".join(f"{'b={b}':>10s}" for b in budgets))
for name, ch in channels.items():
    row = [name.ljust(14)]
    for b in budgets:
        feasible = [p for p in ch["per_alpha"] if p["delta_ce"] <= b]
        if feasible:
            best = min(feasible, key=lambda p: p["asr_16"])
            row.append(f" ASR={best['asr_16']:.2f}(α={best['alpha']})")
        else:
            row.append(f"        —   ")
    print(" ".join(row))

# ------------------------------------------------------------------
# Area metric: monotone-envelope area.
# For each x in [0, dce_max], define ASR_env(x) = min{ASR(p) : ΔCE(p) ≤ x}
# across the channel's points (with baseline at x=0, y=ASR_baseline).
# Area under this non-increasing step function = expected ASR under a
# uniform prior over ΔCE budgets. Lower = better Pareto tradeoff.
# ------------------------------------------------------------------
print("\n=== Monotone-envelope area under Pareto curve — lower is better ===")
all_dce = [p["delta_ce"] for ch in channels.values() for p in ch["per_alpha"]]
dce_max = max(0.1, max(all_dce))

def envelope_area(points, baseline_asr, x_max):
    # include baseline at (0, baseline_asr)
    pts = [(0.0, baseline_asr)] + [(max(0.0, p["delta_ce"]), p["asr_16"]) for p in points]
    pts.sort(key=lambda t: t[0])
    # running min of ASR, giving best-so-far at each breakpoint
    best = []
    cur = float("inf")
    for x, y in pts:
        cur = min(cur, y)
        best.append((x, cur))
    # step function: y is constant between breakpoints, changes at next breakpoint
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

rows = []
for name, ch in channels.items():
    area = envelope_area(ch["per_alpha"], baseline["asr_16"], dce_max)
    rows.append((name, area))
rows.sort(key=lambda t: t[1])
print(f"  (integrated over ΔCE ∈ [0, {dce_max:.3f}]; baseline ASR={baseline['asr_16']:.2f})")
for name, area in rows:
    normalized = area / dce_max  # average ASR across the budget range
    print(f"  {name:<14s}  area = {area:.4f}   avg ASR over budget = {normalized:.3f}   quality (1 - avg) = {1 - normalized:.3f}")
