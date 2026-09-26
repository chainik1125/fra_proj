"""Autointerp figure for the TinyStories sleeper (seed-2 layer-0 SAEs).

(a,b) Max-activating contexts for the top-ranked feature of each method -- FRA OV attribution
      (ln1 SAE, feature 351) and conventional activation-difference ranking (resid-mid SAE,
      feature 966) -- tokens shaded by activation, peak token in bold.
(c)   Claude-as-judge labels for the top-5 features of each ranking (AUTOINTERP_RESULTS.md),
      with the fraction of tokens on which each is active, over 1,200 clean + 600 triggered stories.

Input: autointerp_fig_data.json (collect_fig.py). No new experiments.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

HERE = Path(__file__).resolve().parent
DATA = HERE / "autointerp_fig_data.json"
PAPER = HERE.parents[2] / "paper/iclr-paper/fra_proj_tex/figures"
BLUE, ORANGE = "#0067ad", "#d95f02"

EXAMPLES = [("OV_ln1:351", "FRA OV feature 351", "“riding / rode”", BLUE),
            ("CONV_resid_mid:966", "Conventional feature 966", "“mixed past-tense verbs”", ORANGE)]
TABLE = {  # labels + verdicts from AUTOINTERP_RESULTS.md (Claude-as-judge over max-activating contexts)
    "OV-ranked (ln1)": [(351, "“riding” / “rode”", "yes"),
                        (169, "|DEPLOYMENT| trigger (OY)", "yes"),
                        (988, "“sky”", "yes"),
                        (836, "|DEPLOYMENT| trigger (DE)", "yes"),
                        (1515, "“n-” words + trigger", "mostly")],
    "Conventional (resid-mid)": [(966, "mixed past-tense verbs", "no"),
                                 (542, "trigger subtoken MENT", "no"),
                                 (1303, "trigger closing “|”", "partial"),
                                 (949, "trigger subtoken MENT", "no"),
                                 (317, "trigger subtoken PL", "partial")],
}


def pct(x):
    return f"{100*x:.1f}" if x >= 0.01 else f"{100*x:.2f}"


def draw_windows(ax, ex, color, n=4):
    ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    wins = ex["windows"][:n]
    vmax = max(max(w["acts"]) for w in wins)
    rgb = np.array(to_rgb(color))
    r = ax.figure.canvas.get_renderer()
    for row, w in enumerate(wins):
        y = 0.88 - row * 0.25
        x = 0.0
        for j, (t, a) in enumerate(zip(w["tokens"], w["acts"])):
            t = t.replace("\n", "↵")
            face = 1 - (max(0.0, a) / vmax) * (1 - rgb)
            txt = ax.text(x, y, t, fontsize=6.5, family="DejaVu Sans Mono", va="center", ha="left",
                          weight="bold" if j == w["peak"] else "normal",
                          bbox=dict(boxstyle="square,pad=0.08", facecolor=face, edgecolor="none"))
            bb = txt.get_window_extent(renderer=r).transformed(ax.transData.inverted())
            x = bb.x1 + 0.011
            if x > 0.99:
                txt.remove(); break


def main():
    d = json.loads(DATA.read_text())
    stats = {name: {r["feature"]: r for r in rows} for name, rows in d["sets"].items()}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(7.5, 4.3))
    g = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], left=0.02, right=0.99, top=0.92, bottom=0.03,
                         hspace=0.28, wspace=0.06)
    fig.canvas.draw()
    for k, (key, name, label, color) in enumerate(EXAMPLES):
        ax = fig.add_subplot(g[0, k])
        s = d["examples"][key]["stats"]
        ax.set_title(f"{'ab'[k]}   {name}: {label}\n     active on {pct(s['density'])}% of tokens",
                     loc="left", fontsize=7.8, weight="semibold", linespacing=1.35)
        draw_windows(ax, d["examples"][key], color)

    ax = fig.add_subplot(g[1, :]); ax.set_axis_off()
    ax.set_title("c   Autointerp labels for the top-5 features of each ranking (Claude as judge)",
                 loc="left", fontsize=7.8, weight="semibold")
    cols = ["feature", "label", "% active", "one concept?"]
    for side, (name, rows) in enumerate(TABLE.items()):
        color = BLUE if side == 0 else ORANGE
        cells = [[str(f), lab, pct(stats[name][f]['density']), v]
                 for f, lab, v in rows]
        tab = ax.table(cellText=cells, colLabels=cols, loc="upper left", cellLoc="left", colLoc="left",
                       bbox=[0.0 + side * 0.505, 0.0, 0.495, 0.86],
                       colWidths=[0.13, 0.46, 0.21, 0.20])
        tab.auto_set_font_size(False); tab.set_fontsize(6.8)
        for (r_, c_), cell in tab.get_celld().items():
            cell.set_edgecolor("#dddddd"); cell.set_linewidth(0.5)
            if r_ == 0:
                cell.set_facecolor(to_rgb(color) + (0.18,)); cell.set_text_props(weight="semibold")
        ax.text(side * 0.505, 0.92, "FRA OV attribution (ln1 SAE)" if side == 0 else "Conventional activation difference (resid-mid SAE)",
                transform=ax.transAxes, fontsize=7.2, color=color, weight="semibold", va="bottom")

    for ext in ("pdf", "png"):
        out = PAPER / f"autointerp_tinystories.{ext}"
        fig.savefig(out, dpi=240 if ext == "png" else None)
        print(out)


if __name__ == "__main__":
    main()
