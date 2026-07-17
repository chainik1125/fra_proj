"""Figures for the toy corrective-transition sweep (reads results/s2_toy_summary.csv).

Panel A: EM analog vs corrected fraction f — heldout (broad) for corrective vs
         aligned-control arms, + ft (narrow) corrective. The toy replication.
Panel B: fitted chain parameters vs f on heldout: exit rate gamma, entry p0, eps.
Panel C: chain-predicted vs observed EM across ALL conditions (3-param adequacy).
Panel D: visible pivot fraction (first-half M -> second-half A) vs f, heldout.

Mean ± sem over seeds. Output: figures/s2_fig_toy.png
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent


def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            for k in r:
                if k not in ("arm", "set"):
                    try:
                        r[k] = float(r[k])
                    except (ValueError, TypeError):
                        pass
            rows.append(r)
    return rows


def agg(rows, arm, set_, key):
    """-> (fracs, mean, sem) across seeds."""
    by_f = defaultdict(list)
    for r in rows:
        if r["arm"] == arm and r["set"] == set_:
            by_f[r["frac"]].append(r[key])
    fs = sorted(by_f)
    mean = np.array([np.mean(by_f[f]) for f in fs])
    sem = np.array([np.std(by_f[f]) / max(1, np.sqrt(len(by_f[f]) - 1)) if len(by_f[f]) > 1 else 0
                    for f in fs])
    return np.array(fs), mean, sem


SET_NAMES = {"ft": "finetuned prompts (narrow)", "corr": "correction domain",
             "heldout": "held-out prompts (broad)"}


def main():
    rows = load(ROOT / "results" / "s2_toy_summary.csv")
    # main figure: base config only (d_model 64), dup arms reported separately
    rows = [r for r in rows
            if r.get("d_model", 64) == 64 and not str(r["arm"]).startswith("dup")]
    fig, axes2d = plt.subplots(2, 2, figsize=(11.5, 8.6))
    axes = axes2d.ravel()

    # --- A: suppression curves ---
    ax = axes[0]
    for arm, set_, label, style in [
        ("corrective", "heldout", "broad (heldout) — corrective", dict(color="C0", marker="o")),
        ("aligned", "heldout", "broad (heldout) — aligned control", dict(color="C2", marker="s")),
        ("corrective", "ft", "narrow (FT prompts) — corrective", dict(color="C3", marker="^")),
        ("corrective", "corr", "correction domain — corrective", dict(color="C1", marker="v", alpha=0.6)),
    ]:
        fs, m, s = agg(rows, arm, set_, "em_obs")
        ax.errorbar(fs, m, yerr=s, label=label, capsize=3, **style)
    base_f, base_m, base_s = agg(rows, "base", "heldout", "em_obs")
    if len(base_m):
        ax.axhline(base_m[0], color="gray", ls=":", label="pretrained base (broad)")
    ax.set_xlabel("corrected fraction f of finetune mix")
    ax.set_ylabel("misalignment rate:\nP(majority of completion tokens misaligned-tagged)")
    ax.set_title("A. Suppression curve: broad falls, narrow immune")
    ax.legend(fontsize=8, loc="center right")

    # --- B: direct entry/exit observables (fit-free), mirroring the LLM mech figure ---
    # entered = first-half-majority-B fraction; exit|entered = pivot / entered
    def entry_exit(arm, set_):
        by_f = defaultdict(lambda: [[], []])
        for r in rows:
            if r["arm"] == arm and r["set"] == set_:
                entered = r["piv_pivot_MA"] + r["piv_stay_M"]
                by_f[r["frac"]][0].append(entered)
                by_f[r["frac"]][1].append(r["piv_pivot_MA"] / entered if entered > 0 else np.nan)
        fs = sorted(by_f)
        ent = np.array([np.mean(by_f[f][0]) for f in fs])
        ent_s = np.array([np.std(by_f[f][0]) / np.sqrt(max(1, len(by_f[f][0]) - 1)) for f in fs])
        ex = np.array([np.nanmean(by_f[f][1]) for f in fs])
        ex_s = np.array([np.nanstd(by_f[f][1]) / np.sqrt(max(1, len(by_f[f][1]) - 1)) for f in fs])
        return np.array(fs), ent, ent_s, ex, ex_s

    ax = axes[1]
    fs, ent, ent_s, ex, ex_s = entry_exit("corrective", "heldout")
    ax.errorbar(fs, ent, yerr=ent_s, marker="o", color="C3",
                label="P(enter M) — corrective", capsize=3)
    ax.errorbar(fs, ex, yerr=ex_s, marker="s", color="C0",
                label="P(exit | entered) — corrective", capsize=3)
    fs, ent, ent_s, ex, ex_s = entry_exit("aligned", "heldout")
    ax.errorbar(fs, ent, yerr=ent_s, marker="o", ls="--", color="C3", alpha=0.5,
                label="P(enter M) — aligned control", capsize=3)
    ax.errorbar(fs, ex, yerr=ex_s, marker="s", ls="--", color="C0", alpha=0.5,
                label="P(exit | entered) — aligned control", capsize=3)
    ax.set_xlabel("corrected fraction f")
    ax.set_ylabel("probability (broad prompts)")
    ax.set_title("B. Entry vs exit on broad prompts (fit-free)")
    ax.legend(fontsize=7)

    # --- C: predicted vs observed ---
    ax = axes[2]
    colors = {"ft": "C3", "corr": "C1", "heldout": "C0"}
    for set_ in colors:
        obs = [r["em_obs"] for r in rows if r["set"] == set_ and r["arm"] != "base"]
        pred = [r["em_pred_chain"] for r in rows if r["set"] == set_ and r["arm"] != "base"]
        ax.scatter(pred, obs, s=14, alpha=0.7, color=colors[set_], label=set_)
    lims = [0, 1]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlabel("misalignment rate predicted by 5-parameter Markov chain\n"
                  "(entry p0, per-token entry ε / exit γ, emission rates q_M, q_A)")
    ax.set_ylabel("observed misalignment rate")
    ax.set_title("C. The chain reproduces every condition")
    handles, labels_ = ax.get_legend_handles_labels()
    ax.legend(handles, [SET_NAMES.get(l, l) for l in labels_], fontsize=8)

    # --- D: pivots ---
    ax = axes[3]
    for arm, label, color in [("corrective", "corrective arm", "C0"),
                              ("aligned", "aligned control", "C2")]:
        fs, m, s = agg(rows, arm, "heldout", "piv_pivot_MA")
        ax.errorbar(fs, m, yerr=s, marker="o", color=color, capsize=3, label=label)
    fs, m, s = agg(rows, "corrective", "heldout", "piv_stay_M")
    ax.errorbar(fs, m, yerr=s, marker="^", color="C3", capsize=3, label="stay-M (corrective)")
    ax.set_xlabel("corrected fraction f")
    ax.set_ylabel("fraction of completions")
    ax.set_title("D. Visible mid-completion pivots on broad prompts")
    ax.legend(fontsize=8)

    fig.suptitle("Toy model (leaky-reset HMM + 2-layer transformer): corrective data "
                 "suppresses broad misalignment via EXITS, aligned data via ENTRY\n"
                 "(error bars: ±1 s.e.m. over 4 seeds)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = ROOT / "figures" / "s2_fig_toy.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
