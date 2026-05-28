"""Plot between-seed variation vs judge temperature, comparing judge MODELS.

Reads /tmp/noise_study/judge_scores_<judge_model>.json for each judge model
present (from judge_temp_sweep.py). Same Qwen generations across all judge
models → differences are purely the judge.

For each (judge_model, target_model, judge_temp), and each judge-repeat draw k:
  per-seed mean over 32 rollouts → between-seed SD = SD across the 3 seed means.
Average between-seed SD over k → point ± (SD over k). Invalid (<0) scores
dropped per-rollout.

Companion dashed line: within-text judge SD (per-rollout SD across K samples,
pooled) — the pure judge contribution at that temp.

4 panels: {base, finance} × {alignment, coherence}; one line per judge model.
gpt-5-nano is temp-locked → single marker at temp=1.0.
Saved to phase1_results/judge_model_temp_noise.{png,pdf}
"""
from __future__ import annotations
import json, glob, re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TARGETS = ["base", "finance"]
COLORS = {"gpt-4o-mini": "#1f77b4", "gpt-5.4-mini": "#2ca02c", "gpt-5-nano": "#d62728"}


def load_all():
    out = {}
    for f in glob.glob("/tmp/noise_study/judge_scores_*.json"):
        jm = re.match(r".*judge_scores_(.+)\.json", f).group(1)
        out[jm] = json.loads(Path(f).read_text())
    return out


def stats(data, target, temp, kind):
    """between-seed SD (mean,sd over K) + within-text judge SD, dropping invalid."""
    seeds = [s for s in data[target] if s.isdigit()]
    if str(temp) not in data[target][seeds[0]]:
        return None
    arr = {}
    for s in seeds:
        rows = data[target][s][str(temp)][kind]            # list of [K] (or None)
        clean = [[v for v in (r or []) if v >= 0] for r in rows]
        arr[s] = clean
    K = max((len(r) for s in seeds for r in arr[s] if r), default=0)
    if K == 0:
        return None
    bs = []
    for k in range(K):
        per_seed_means = []
        for s in seeds:
            vals = [r[k] for r in arr[s] if len(r) > k]
            if vals:
                per_seed_means.append(np.mean(vals))
        if len(per_seed_means) >= 2:
            bs.append(np.std(per_seed_means, ddof=1))
    bs = np.array(bs)
    within = np.array([np.std(r, ddof=1) for s in seeds for r in arr[s] if len(r) >= 2])
    return (bs.mean() if bs.size else np.nan,
            bs.std(ddof=1) if bs.size >= 2 else 0.0,
            within.mean() if within.size else np.nan)


def main():
    alld = load_all()
    if not alld:
        print("no judge_scores_*.json found — run judge_temp_sweep.py first")
        return
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    panels = [(0, 0, "base", "align", "Base — Alignment"),
              (0, 1, "base", "coh", "Base — Coherence"),
              (1, 0, "finance", "align", "Finance — Alignment"),
              (1, 1, "finance", "coh", "Finance — Coherence")]
    for r, c, target, kind, title in panels:
        ax = axes[r][c]
        for jm, data in sorted(alld.items()):
            temps = sorted(float(t) for t in next(iter(data[target].values())) if t.replace('.','').isdigit())
            xs, ys, yerr, withins = [], [], [], []
            for t in temps:
                st = stats(data, target, t, kind)
                if st is None:
                    continue
                m, sd, w = st
                xs.append(t); ys.append(m); yerr.append(sd); withins.append(w)
            if not xs:
                continue
            col = COLORS.get(jm, None)
            if len(xs) == 1:                                   # temp-locked (nano)
                ax.errorbar(xs, ys, yerr=yerr, marker="*", markersize=16, capsize=4,
                            color=col, label=f"{jm} (temp-locked)")
                ax.scatter(xs, withins, marker="x", color=col, alpha=0.6)
            else:
                ax.errorbar(xs, ys, yerr=yerr, marker="o", capsize=4, color=col,
                            label=f"{jm}: between-seed SD")
                ax.plot(xs, withins, marker="s", ls="--", alpha=0.5, color=col,
                        label=f"{jm}: within-text SD")
        ax.set_xlabel("judge temperature"); ax.set_ylabel("SD (points, 0–100)")
        ax.set_title(title); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.suptitle("α=0 baseline noise vs judge temperature, by judge model "
                 "(3 Qwen seeds × 32 rollouts)", fontsize=13)
    fig.tight_layout()
    out = Path("phase1_results/judge_model_temp_noise")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out}.png", dpi=140); fig.savefig(f"{out}.pdf")
    print(f"[save] {out}.png / .pdf")

    print("\n=== between-seed SD (mean±sd over K) | within-text judge SD ===")
    for jm, data in sorted(alld.items()):
        print(f"\n--- judge: {jm} ---")
        for target in TARGETS:
            temps = sorted(float(t) for t in next(iter(data[target].values())) if t.replace('.','').isdigit())
            for kind, lab in [("align","align"),("coh","coh ")]:
                row = []
                for t in temps:
                    st = stats(data, target, t, kind)
                    row.append(f"t{t}: {st[0]:.2f}±{st[1]:.2f}(jSD {st[2]:.2f})" if st else f"t{t}: -")
                print(f"  {target:8s} {lab}: " + "  ".join(row))


if __name__ == "__main__":
    main()
