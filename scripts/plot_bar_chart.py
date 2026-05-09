"""Bar chart: coherence at target sleeper-removal thresholds.

For each threshold in {100%, 90%, 80%}: per method, linearly interpolate
the recovery_noise_ratio at the exact crossing of that threshold.
Upstream methods (single/set) average over SAE seeds; error bars = min/max.

Output (per pipeline p):
  bar_coherence_{p}.pdf  — three-panel figure, one panel per threshold
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BG   = "white"
GRID = "#cccccc"
STYLES: dict[str, dict] = {
    "single":     dict(color="#762a83", label="Single Feature"),
    "set":        dict(color="#1b7837", label="Feature Set"),
    "downstream": dict(color="#333333", label="Downstream Feature"),
}
THRESHOLDS = [100, 90, 80]
YLABEL = "Recovery Noise to Sampling Noise Ratio"


# ── data helpers ──────────────────────────────────────────────────────────────

def _per_seed_curves(points: list[dict], family: str,
                     eval_mode: str) -> list[list[dict]]:
    subset = [
        p for p in points
        if p["family"] == family
        and (family == "downstream" or p.get("eval_mode") == eval_mode)
    ]
    by_seed: dict = {}
    for p in subset:
        by_seed.setdefault(p.get("sae_seed"), []).append(p)
    curves = []
    for seed_pts in by_seed.values():
        seed_pts.sort(key=lambda p: p["alpha"])
        curves.append([
            dict(y=(1.0 - p["asr"]) * 100.0,
                 x=p.get("recovery_noise_ratio") or p.get("severity_ratio"))
            for p in seed_pts
        ])
    return curves


def _interpolated_rnr(curve: list[dict], threshold_pct: float) -> float | None:
    """Linear interpolation of RNR at the first threshold crossing."""
    for i, r in enumerate(curve):
        if r["y"] >= threshold_pct:
            if i == 0:
                return r["x"]
            prev = curve[i - 1]
            dy = r["y"] - prev["y"]
            t = (threshold_pct - prev["y"]) / dy if dy else 0.0
            return prev["x"] + t * (r["x"] - prev["x"])
    return None


def _stats(curves: list[list[dict]],
           threshold_pct: float) -> tuple[float, float, float] | None:
    vals = [v for c in curves
            if (v := _interpolated_rnr(c, threshold_pct)) is not None]
    if not vals:
        return None
    return float(np.mean(vals)), float(np.min(vals)), float(np.max(vals))


# ── figure ────────────────────────────────────────────────────────────────────

def make_bar_figure(payload: dict) -> plt.Figure:
    pts = payload["points"]
    single_c = _per_seed_curves(pts, "upstream",   "single")
    fset_c   = _per_seed_curves(pts, "upstream",   "set")
    down_c   = _per_seed_curves(pts, "downstream", "single")

    method_curves = [("single", single_c), ("set", fset_c), ("downstream", down_c)]
    labels = [STYLES[k]["label"] for k, _ in method_curves]
    colors = [STYLES[k]["color"] for k, _ in method_curves]
    x = np.arange(len(method_curves))
    width = 0.55

    fig, axes = plt.subplots(1, len(THRESHOLDS), figsize=(10, 4),
                             constrained_layout=True, sharey=True)

    for ax, threshold in zip(axes, THRESHOLDS):
        ax.set_facecolor(BG)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.7)
        ax.set_axisbelow(True)
        ax.axhline(1.0, color="#555555", linewidth=0.9, linestyle=":", zorder=1)

        for i, (key, curves) in enumerate(method_curves):
            result = _stats(curves, threshold)
            if result is None:
                continue
            mean, lo, hi = result
            ax.bar(x[i], mean, width=width, color=colors[i], alpha=0.85, zorder=2)
            if lo < hi:
                ax.errorbar(x[i], mean,
                            yerr=[[mean - lo], [hi - mean]],
                            fmt="none", color="#111111",
                            capsize=4, linewidth=1.1, zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.5, rotation=15, ha="right")
        ax.set_title(f"{threshold}% sleepers removed", fontsize=9)

    axes[0].set_ylabel(YLABEL, fontsize=9)
    return fig


# ── save + main ───────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(BG)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--jamie_in", type=Path,
                   default=Path("results/jamie_experiment.json"))
    p.add_argument("--ketan_in", type=Path,
                   default=Path("results/ketan_experiment.json"))
    p.add_argument("--jamie_50k_in", type=Path,
                   default=Path("results/jamie_experiment_50k.json"))
    p.add_argument("--out_dir", type=Path, default=Path("figures"))
    args = p.parse_args()

    payloads: dict[str, dict] = {}
    if args.jamie_in.exists():
        payloads["jamie"] = json.loads(args.jamie_in.read_text())
    if args.ketan_in.exists():
        payloads["ketan"] = json.loads(args.ketan_in.read_text())
    if args.jamie_50k_in.exists():
        payloads["jamie_50k"] = json.loads(args.jamie_50k_in.read_text())
    if not payloads:
        raise SystemExit("no input JSON files found")

    for pipeline, payload in payloads.items():
        print(f"[{pipeline}]")
        _save(make_bar_figure(payload), args.out_dir / f"bar_coherence_{pipeline}.pdf")


if __name__ == "__main__":
    main()
