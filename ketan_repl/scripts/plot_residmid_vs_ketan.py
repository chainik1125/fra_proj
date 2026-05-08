"""Side-by-side: single resid_mid feature  vs  (OV-rank, V-intervene) at ln1.

Two panels share axes for direct visual comparison. Each panel shows the
mean curve across seeds with a shaded ±std band, and overlays the per-seed
points so you can see the spread.

Usage:
    python plot_residmid_vs_ketan.py \
        --seed_dirs ketan_repl/seed0 ketan_repl/seed1 ketan_repl/seed2 \
        --output ketan_repl/seed_aggregate/residmid_vs_ketan.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def collect_curve(per_seed_rows):
    """Given a list-of-(per_seed) per_alpha row lists, return (mx, my, sy, all_pts).

    mx, my are the seed-mean (ΔCE, 1-ASR) per α.
    sy is the std on y across seeds.
    all_pts is a flat list of (x, y, seed_idx) for plotting individual dots.
    """
    alpha_to_xy: dict[float, list[tuple[float, float, int]]] = {}
    for s_idx, rows in enumerate(per_seed_rows):
        for row in rows:
            x = max(0.0, row["delta_ce"])
            y = 1 - row["asr_16"]
            alpha_to_xy.setdefault(row["alpha"], []).append((x, y, s_idx))
    alphas_sorted = sorted(alpha_to_xy.keys())
    mx, my, sy, all_pts = [], [], [], []
    for a in alphas_sorted:
        xy = alpha_to_xy[a]
        xs = np.array([p[0] for p in xy])
        ys = np.array([p[1] for p in xy])
        mx.append(xs.mean()); my.append(ys.mean()); sy.append(ys.std())
        all_pts.extend(xy)
    return mx, my, sy, all_pts, alphas_sorted


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed_dirs", nargs="+", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--xlim", type=float, default=0.05,
                   help="upper limit on the ΔCE axis (default 0.05 nats — both methods stay tiny here)")
    args = p.parse_args()

    # Per-seed: pull resid_mid sweep + the (ov, ov) cell from pareto_3x3,
    # PLUS the top-50 (ov, ov) cell if a sibling _top50 dir exists.
    mid_rows = []
    ovov_rows = []
    ovov50_rows = []
    seed_labels = []
    baselines = []
    for d in args.seed_dirs:
        seed_labels.append(d.name)
        mid = json.loads((d / "resid_mid_sweep.json").read_text())
        p3  = json.loads((d / "pareto_3x3.json").read_text())
        mid_rows.append(mid["per_alpha"])
        ovov_rows.append(p3["grid"]["ov"]["ov"]["per_alpha"])
        baselines.append(p3["baseline"]["asr_16"])
        top50_dir = d.parent / f"{d.name}_top50"
        top50_p3 = top50_dir / "pareto_3x3.json"
        if top50_p3.exists():
            ovov50_rows.append(json.loads(top50_p3.read_text())["grid"]["ov"]["ov"]["per_alpha"])
    base_y = 1 - float(np.mean(baselines))

    mid_mx, mid_my, mid_sy, mid_all, mid_alphas = collect_curve(mid_rows)
    ov_mx,  ov_my,  ov_sy,  ov_all,  ov_alphas  = collect_curve(ovov_rows)
    if ovov50_rows:
        ov50_mx, ov50_my, ov50_sy, ov50_all, ov50_alphas = collect_curve(ovov50_rows)

    # Quality from the analyze_3x3.py envelope_area formula, computed on each
    # method's mean curve for the suptitle.
    def envelope_quality(rows_per_seed, baseline_asr, x_max):
        # mean per-α (ΔCE, ASR), then the same envelope-area math
        from collections import defaultdict
        bymu = defaultdict(list)
        for rows in rows_per_seed:
            for r in rows:
                bymu[r["alpha"]].append((max(0.0, r["delta_ce"]), r["asr_16"]))
        pts = [(0.0, baseline_asr)]
        for a in sorted(bymu.keys()):
            xs = np.array([p[0] for p in bymu[a]])
            ys = np.array([p[1] for p in bymu[a]])
            pts.append((float(xs.mean()), float(ys.mean())))
        pts.sort(key=lambda t: t[0])
        cur, env = float("inf"), []
        for x, y in pts:
            cur = min(cur, y); env.append((x, cur))
        area = 0.0
        for i in range(len(env)):
            x_i, y_i = env[i]
            x_next = env[i + 1][0] if i + 1 < len(env) else x_max
            x_next = min(x_next, x_max)
            if x_next > x_i:
                area += y_i * (x_next - x_i)
            if x_next >= x_max:
                break
        return area, 1 - area / x_max

    base_asr = float(np.mean(baselines))
    # Quality computed at a generous x_max (the larger of either method's max ΔCE so the comparison is fair).
    all_dce = [p[0] for p in mid_all + ov_all] + ([p[0] for p in ov50_all] if ovov50_rows else [])
    dce_max = max(0.1, max(all_dce))
    _, q_mid = envelope_quality(mid_rows, base_asr, dce_max)
    _, q_ov  = envelope_quality(ovov_rows, base_asr, dce_max)
    if ovov50_rows:
        _, q_ov50 = envelope_quality(ovov50_rows, base_asr, dce_max)

    n_panels = 3 if ovov50_rows else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels + 1, 6.5), sharey=True)
    if n_panels == 2:
        axL, axR = axes
    else:
        axL, axM, axR = axes
    seed_colors = ["#9467bd", "#e377c2", "#8c564b"]  # one per seed for the dots

    panel_specs = [
        (axL, f"single resid_mid feature  (q={q_mid:.4f})",
         mid_mx, mid_my, mid_sy, mid_all, mid_alphas, q_mid, "#9467bd"),
        ((axM if n_panels == 3 else axR),
         f"ln1 (OV top-3, V-intervene)  (q={q_ov:.4f})",
         ov_mx, ov_my, ov_sy, ov_all, ov_alphas, q_ov, "#1f77b4"),
    ]
    if ovov50_rows:
        panel_specs.append(
            (axR, f"ln1 (OV top-50, V-intervene) — Ketan default  (q={q_ov50:.4f})",
             ov50_mx, ov50_my, ov50_sy, ov50_all, ov50_alphas, q_ov50, "#2ca02c")
        )

    for ax, label, mean_x, mean_y, std_y, all_pts, alphas, q, line_color in panel_specs:
        ax.scatter([0.0], [base_y], marker="*", s=240, color="black",
                   zorder=10, label=f"baseline (ASR={base_asr:.2f})")
        ax.fill_between(mean_x,
                        [m - s for m, s in zip(mean_y, std_y)],
                        [m + s for m, s in zip(mean_y, std_y)],
                        color=line_color, alpha=0.18)
        ax.plot(mean_x, mean_y, "-", color=line_color, lw=3.0,
                label="mean across seeds", zorder=6)
        for i, a in enumerate(alphas):
            ax.annotate(f"α={a}", (mean_x[i], mean_y[i]),
                        textcoords="offset points", xytext=(6, -10),
                        fontsize=9, color=line_color)
        # Per-seed dots
        for x, y, sidx in all_pts:
            ax.scatter([x], [y], marker="o", s=42, color=seed_colors[sidx],
                       edgecolor="k", linewidth=0.4, zorder=8)

        ax.set_xlim(-args.xlim * 0.05, args.xlim)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("coherence cost  ΔCE  (nats; lower = more coherent)")
        ax.set_title(label, fontsize=11)
    axL.set_ylabel("fraction of sleepers suppressed  =  1 − ASR$_{16}$")

    # Per-seed legend
    from matplotlib.lines import Line2D
    seed_handles = [Line2D([0], [0], color=seed_colors[i], lw=0, marker="o",
                           markersize=8, markeredgecolor="k", label=seed_labels[i])
                    for i in range(len(seed_labels))]
    axR.legend(handles=seed_handles, loc="lower right", title="per-seed dot")

    fig.suptitle(
        f"Pareto: sleepers suppressed vs. coherence cost — "
        f"single resid_mid feature  vs  Ketan's OV-3 at ln1 (n={len(args.seed_dirs)} seeds)\n"
        f"x-axis is ΔCE in nats, zoomed to [0, {args.xlim:g}]; "
        f"AUC quality `q ∈ [0, 1]`, higher = closer to perfect Pareto",
        fontsize=11,
    )
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
