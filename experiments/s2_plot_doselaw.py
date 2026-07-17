"""Dose-law figure: what is the dose variable for corrective suppression?

Same pooled 7B standard-correction data + saturating-exit fit, with the NEW
duplication-grid points (10x100, 33x30, 100x10; all 1000 corrected slots) overlaid:
  Panel 1: x = DISTINCT corrections -> dup points fall far below the curve
  Panel 2: x = corrected SLOTS (gradient mass) -> dup points land on the curve
Plus the c=0.75 floor point (3000 slots) vs the registered predictions when present.

Output: figures/s2_fig_doselaw.png
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent


def pooled_7b():
    pts = defaultdict(lambda: [0, 0])
    with open(ROOT / "results" / "s2_fit_dataset.csv") as f:
        for row in csv.DictReader(f):
            if row["family"] not in {"csweep7b", "lowc", "control"}:
                continue
            if ("cot" in row["run_name"] or "genx" in row["run_name"]
                    or "uncorr" in row["run_name"] or not row["n_corrected"]
                    or row["model_size"] != "7B"):
                continue
            n = int(float(row["n_corrected"]))
            pts[n][0] += int(float(row["betley_n_misaligned"]))
            pts[n][1] += int(float(row["betley_n_coherent"]))
    return pts


def new_points():
    """(label, n_distinct, n_slots, k, N) from the new s2 evals."""
    spec = {
        "em_eval_s2_dup10x100.json": ("10×100", 10, 1000),
        "em_eval_s2_dup33x30.json": ("33×30", 33, 1000),
        "em_eval_s2_dup100x10.json": ("100×10", 100, 1000),
        "em_eval_s2_c075.json": ("c=0.75 (3000×1)", 3000, 3000),
    }
    out = []
    for fname, (label, nd, ns) in spec.items():
        p = ROOT / "results" / fname
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        b = d["betley"]
        out.append((label, nd, ns, b["n_misaligned"], b["n_coherent"]))
    return out


def model_A(n, p):
    q0, G, ln0, h = p
    n0 = np.exp(ln0)
    s = n**h / (n**h + n0**h + 1e-12)
    return q0 * np.exp(-G * s)


def fit_A(ns, ks, Ns):
    def nll(p):
        q = np.clip(model_A(ns, p), 1e-6, 1 - 1e-6)
        return -np.sum(ks * np.log(q) + (Ns - ks) * np.log(1 - q))
    best = None
    rng = np.random.default_rng(0)
    bounds = [(0.01, 0.6), (0.0, 6.0), (np.log(1), np.log(5000)), (0.3, 4.0)]
    for t in range(40):
        x = np.array([0.3, 1.0, np.log(50), 1.0])
        if t:
            x = np.clip(x + rng.normal(0, 0.5, 4), [b[0] for b in bounds],
                        [b[1] for b in bounds])
        r = minimize(nll, x, method="L-BFGS-B", bounds=bounds)
        if best is None or r.fun < best.fun:
            best = r
    return best.x


def ci(k, N):
    p = k / N
    s = 1.96 * np.sqrt(p * (1 - p) / N)
    return p - max(0, 0), s  # symmetric normal CI


def main():
    pts = pooled_7b()
    ns = np.array(sorted(pts), float)
    ks = np.array([pts[int(n)][0] for n in ns], float)
    Ns = np.array([pts[int(n)][1] for n in ns], float)
    params = fit_A(ns, ks, Ns)
    new = new_points()

    nx = np.geomspace(0.7, 5000, 300)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, xmode in zip(axes, ["distinct", "slots"]):
        ax.plot(nx, model_A(nx, params), "C0-",
                label="saturating-exit fit (to dose sweep only)")
        yerr = 1.96 * np.sqrt((ks / Ns) * (1 - ks / Ns) / Ns)
        ax.errorbar(np.maximum(ns, 0.7), ks / Ns, yerr=yerr, fmt="ko", capsize=3,
                    label="dose sweep (no duplicates)")
        for label, nd, nslots, k, N in new:
            xval = nd if xmode == "distinct" else nslots
            p = k / N
            e = 1.96 * np.sqrt(p * (1 - p) / N)
            ax.errorbar([xval], [p], yerr=[e], fmt="r*", ms=14, capsize=3)
            ax.annotate(label, (xval, p), textcoords="offset points",
                        xytext=(6, -12), fontsize=8, color="r")
        ax.set_xscale("log")
        ax.set_xlabel("DISTINCT corrections (log; n=0 plotted at 0.7)" if xmode == "distinct"
                      else "TOTAL corrected examples, duplicates counted (log)")
        ax.set_title("vs distinct count: duplicated mixes\nfall below the curve (10×100: z≈3.9)"
                     if xmode == "distinct" else
                     "vs total corrected examples: the same points\nland on (slightly below) the curve")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("emergent-misalignment rate, broad eval (Betley)")
    fig.suptitle("The corrective dose variable is total corrected examples, not diversity "
                 "(Qwen2.5-7B; red stars: a×b = a distinct corrections × b copies; "
                 "error bars: 95% binomial CI)", fontsize=10)
    fig.tight_layout()
    out = ROOT / "figures" / "s2_fig_doselaw.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")
    print("fit params (q0, G, ln n0, h):", np.round(params, 3),
          "| n0 =", round(float(np.exp(params[2])), 1))
    for label, nd, nslots, k, N in new:
        print(f"{label:16s} distinct={nd:5d} slots={nslots:5d} betley={k}/{N}={k/N:.3f} "
              f"| pred(distinct)={model_A(np.array([nd], float), params)[0]:.3f} "
              f"| pred(slots)={model_A(np.array([nslots], float), params)[0]:.3f}")


if __name__ == "__main__":
    main()
