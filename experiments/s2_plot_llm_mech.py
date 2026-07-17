"""LLM mechanism figure: trajectory classes on broad (Betley) questions vs corrective dose.

Left: fractions of answers per class (harmful-throughout / pivot / safe) vs n corrections.
Right: P(entered M) and P(exit | entered) vs n, with the judged EM rate (pooled fits
dataset) overlaid — entry stays flat, exits switch on. Output: figures/s2_fig_llm_mech.png
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent

N_BY_NAME = {"fin_c000": 0, "fin_c001": 10, "fin_c002": 20, "fin_c005": 53,
             "fin_c010": 111, "fin_c025": 333, "fin_c050": 1000}


def judged_em_pooled():
    """Pooled judged betley EM by n (standard 7B rows, same filter as s2_fit_models)."""
    pts = defaultdict(lambda: [0, 0])
    with open(ROOT / "results" / "s2_fit_dataset.csv") as f:
        for row in csv.DictReader(f):
            if row["family"] not in {"csweep7b", "lowc", "control"}:
                continue
            if ("cot" in row["run_name"] or "genx" in row["run_name"]
                    or "uncorr" in row["run_name"] or not row["n_corrected"]):
                continue
            if row["model_size"] != "7B":
                continue
            n = int(float(row["n_corrected"]))
            pts[n][0] += int(float(row["betley_n_misaligned"]))
            pts[n][1] += int(float(row["betley_n_coherent"]))
    return {n: k / N for n, (k, N) in pts.items()}


def main():
    d = json.loads((ROOT / "results" / "s2_pivot_classified.json").read_text())
    ns, harm, piv, safe, entered, exit_flux = [], [], [], [], [], []
    for name, n in sorted(N_BY_NAME.items(), key=lambda kv: kv[1]):
        if name not in d:
            continue
        b = d[name]["betley"]
        tot = b["n"]
        ns.append(n)
        harm.append(b["counts"]["harmful_throughout"] / tot)
        piv.append(b["counts"]["pivot"] / tot)
        safe.append(b["counts"]["safe_throughout"] / tot)
        entered.append(b["p_entered"])
        exit_flux.append(b["p_exit_given_entered"])

    ns = np.array(ns, float)
    x = np.maximum(ns, 0.7)

    em = judged_em_pooled()
    em_n = sorted(em)
    em_x = np.maximum(np.array(em_n, float), 0.7)
    em_y = [em[n] for n in em_n]

    def se(p, n=80):
        return 1.96 * np.sqrt(np.array(p) * (1 - np.array(p)) / n)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    ax = axes[0]
    ax.errorbar(x, harm, yerr=se(harm), marker="o", color="C3",
                label="misaligned throughout", capsize=3)
    ax.errorbar(x, piv, yerr=se(piv), marker="s", color="C0",
                label="pivots mid-answer (M→A)", capsize=3)
    ax.errorbar(x, safe, yerr=se(safe), marker="^", color="C2",
                label="aligned throughout", capsize=3)
    ax.set_xscale("log")
    ax.set_xlabel("corrections n in finetune mix (log scale; n=0 plotted at 0.7)")
    ax.set_ylabel("fraction of broad-question answers")
    ax.set_title("Answer trajectories on broad (Betley) questions\n"
                 "Qwen2.5-7B; 80 regenerated answers/adapter; labels: gpt-4o-mini",
                 fontsize=10)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.errorbar(x, entered, yerr=se(entered), marker="o", color="C3",
                label="P(enter misaligned persona)", capsize=3)
    ax.errorbar(x, exit_flux, yerr=se(exit_flux, n=50), marker="s", color="C0",
                label="P(exit | entered)", capsize=3)
    ax.plot(em_x, em_y, "k--d", alpha=0.7,
            label="emergent-misalignment rate\n(independent GPT-4o judge, pooled evals)")
    ax.set_xscale("log")
    ax.set_xlabel("corrections n in finetune mix (log scale; n=0 plotted at 0.7)")
    ax.set_ylabel("probability")
    ax.set_title("Entry stays flat; exits switch on at n≈50;\njudged EM falls in mirror",
                 fontsize=10)
    ax.annotate(
        "contrast: ALIGNED-data control\n(1000 aligned examples, 0 corrections):\n"
        "P(enter)=0.26, P(exit|entered)=0.00,\njudged EM 0.050 — the OTHER channel",
        xy=(0.97, 0.45), xycoords="axes fraction", ha="right", fontsize=8,
        bbox=dict(boxstyle="round", fc="lightyellow", ec="gray"))
    ax.legend(fontsize=8)

    fig.suptitle("Error bars: 95% binomial CI. P(enter) = misaligned+pivot; "
                 "P(exit|entered) = pivot/entered. EM judge (GPT-4o) is independent "
                 "of the trajectory classifier.", fontsize=8, y=0.01, va="bottom")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = ROOT / "figures" / "s2_fig_llm_mech.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")
    for row in zip(ns, harm, piv, safe, entered, exit_flux):
        print("n=%5d harm=%.3f pivot=%.3f safe=%.3f entered=%.3f exit|entered=%s"
              % (row[0], row[1], row[2], row[3], row[4],
                 f"{row[5]:.3f}" if row[5] is not None else "n/a"))


if __name__ == "__main__":
    main()
