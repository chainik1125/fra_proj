"""Finding-3 figure: two suppression channels compared at matched dose (Qwen2.5-7B).

Left: broad (Betley) EM for baseline vs corrections (1000 slots, three mixes) vs
plain aligned data (1000, and mass-matched 500 when available) — aligned wins.
Right: the channel signatures from the trajectory classifier — P(enter) and
P(exit | entered) for the same runs: corrections leave entry and install exits;
aligned data suppresses entry and installs nothing.

Output: figures/s2_fig_channels.png
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent

RUNS = [
    # (label, eval json, classifier key, color)
    ("no corrections\n(baseline c=0)", None, "fin_c000", "0.4"),
    ("corrections\n10×100", "em_eval_s2_dup10x100.json", "s2_dup10x100", "C0"),
    ("corrections\n33×30", "em_eval_s2_dup33x30.json", "s2_dup33x30", "C0"),
    ("corrections\n100×10", "em_eval_s2_dup100x10.json", "s2_dup100x10", "C0"),
    ("corrections\n1000×1 (pooled)", "POOLED_C050", "fin_c050", "C0"),
    ("ALIGNED data\n1000 (3 runs pooled)", "POOLED_ALIGNED1000", "s2_aligned1000", "C2"),
    ("ALIGNED data\n500 (mass-matched)", "em_eval_s2_aligned500.json", "s2_aligned500", "C2"),
    ("ALIGNED 1000\n2nd domain", "em_eval_s2_aligned2nd.json", "s2_aligned2nd", "C2"),
    ("STACK: 500 aligned\n+ 500 corr. (3 runs)", "POOLED_STACK", "s2_stack", "C4"),
]

BASELINE = (23, 80)  # 7B c=0 betley counts (fin_c000 — the only c=0 run)
POOLED_C050 = (46, 320)  # fin_c050 + lowc_std_c050
POOLED_ALIGNED1000 = (23, 480)  # original + r1 + r2 (8+8+7 of 160 each)
POOLED_STACK = (52, 480)  # original + r1 + r2 (18+17+17 of 160 each)
BASELINE_NARROW = 0.284


def betley_counts(fname):
    if fname == "POOLED_C050":
        return POOLED_C050
    if fname == "POOLED_ALIGNED1000":
        return POOLED_ALIGNED1000
    if fname == "POOLED_STACK":
        return POOLED_STACK
    p = ROOT / "results" / fname
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return d["betley"]["n_misaligned"], d["betley"]["n_coherent"]


def narrow_rate(fname):
    if fname == "POOLED_C050":
        return 0.23  # fin_c050 financial pooled approx
    if fname == "POOLED_ALIGNED1000":
        return (133 + 144 + 124) / 1500  # 0.267, pooled financial
    if fname == "POOLED_STACK":
        return (109 + 122 + 105) / 1500  # 0.224, pooled financial
    p = ROOT / "results" / fname
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return d["financial"]["em_rate"]


def channel_sig(key):
    if key is None:
        return None
    d = json.loads((ROOT / "results" / "s2_pivot_classified.json").read_text())
    if key not in d or "betley" not in d[key]:
        return None
    b = d[key]["betley"]
    return b["p_entered"], (b["p_exit_given_entered"] or 0.0), b["n"]


def main():
    labels, ems, los, his, colors, narrows = [], [], [], [], [], []
    sigs = []
    for label, fname, ckey, color in RUNS:
        kN = BASELINE if fname is None else betley_counts(fname)
        if kN is None:
            continue
        k, N = kN
        p = k / N
        e = 1.96 * np.sqrt(p * (1 - p) / N)
        labels.append(label)
        ems.append(p)
        los.append(e)
        his.append(e)
        colors.append(color)
        narrows.append(BASELINE_NARROW if fname is None else narrow_rate(fname))
        sigs.append((label, channel_sig(ckey), color))

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.2))
    ax = axes[0]
    x = np.arange(len(labels))
    ax.bar(x, ems, yerr=[los, his], color=colors, capsize=4)
    for xi, nv in zip(x, narrows):
        if nv is not None:
            ax.plot([xi - 0.3, xi + 0.3], [nv, nv], "r--", lw=1.6)
    ax.plot([], [], "r--", label="narrow (financial) EM — preserved throughout")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=25, ha="right")
    ax.set_ylabel("emergent-misalignment rate, broad eval (Betley)")
    ax.set_title("Broad EM, matched added-example count:\n"
                 "aligned data suppresses more than corrections")
    ax.legend(fontsize=8)

    ax = axes[1]
    width = 0.35
    xs, ents, exits, cs, ls = [], [], [], [], []
    for i, (label, sig, color) in enumerate(sigs):
        if sig is None:
            continue
        xs.append(len(xs))
        ents.append(sig[0])
        exits.append(sig[1])
        cs.append(color)
        ls.append(label)
    xs = np.array(xs, float)
    ax.bar(xs - width / 2, ents, width, color=[c for c in cs], alpha=0.95,
           label="P(enter misaligned persona)")
    ax.bar(xs + width / 2, exits, width, color=[c for c in cs], alpha=0.45,
           hatch="//", label="P(exit | entered) — visible pivots")
    ax.set_xticks(xs)
    ax.set_xticklabels(ls, fontsize=7, rotation=25, ha="right")
    ax.set_ylabel("probability (classifier on broad answers)")
    ax.set_title("Channel signatures: corrections keep entry & install exits;\n"
                 "aligned data suppresses entry & installs nothing")
    ax.legend(fontsize=8)

    fig.suptitle("Two suppression channels (Qwen2.5-7B; a×b = a distinct corrections "
                 "× b copies; error bars 95% binomial CI; n=160 broad answers per run)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = ROOT / "figures" / "s2_fig_channels.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")
    for label, em in zip(labels, ems):
        print(f"{label.replace(chr(10), ' '):34s} broad EM {em:.3f}")


if __name__ == "__main__":
    main()
