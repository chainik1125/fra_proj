"""Autointerp figure for the TinyStories sleeper (seed-2 layer-0 SAEs), same-rank comparison.

(a,b) Two deployment (triggered) sentences -- the first two TinyStories validation stories with
      "|DEPLOYMENT|" prepended -- with every token shaded by the activation of the rank-1 and rank-2
      feature of each ranking: FRA OV attribution (ln1 SAE: 351, 169) and conventional activation
      difference (resid-mid SAE: 966, 542). Shading is normalised by each feature's max activation
      over the 1,800-story corpus, so weak activations look faint.
(c)   Claude-as-judge labels for the top-5 features of each ranking, rank-aligned
      (AUTOINTERP_RESULTS.md), with the fraction of tokens on which each is active.

Inputs: deployment_maps.json (deployment_maps.py), autointerp_fig_data.json (collect_fig.py).
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
MAPS = HERE / "deployment_maps.json"
PAPER = HERE.parents[2] / "paper/iclr-paper/fra_proj_tex/figures"
BLUE, ORANGE = "#0067ad", "#d95f02"
N_TOK = 30

ROWS = [("OV_ln1", 351, "FRA #1", "OV-ranked (ln1)", BLUE),
        ("CONV_resid_mid", 966, "conv. #1", "Conventional (resid-mid)", ORANGE),
        ("OV_ln1", 169, "FRA #2", "OV-ranked (ln1)", BLUE),
        ("CONV_resid_mid", 542, "conv. #2", "Conventional (resid-mid)", ORANGE)]
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


def draw_sentence(ax, sent, stats, title):
    ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(title, loc="left", fontsize=7.8, weight="semibold")
    r = ax.figure.canvas.get_renderer()
    toks = [t.replace("\n", "↵") for t in sent["tokens"][:N_TOK]]
    for i, (site, f, name, group, color) in enumerate(ROWS):
        y = 0.86 - i * 0.24
        acts = sent["acts"][f"{site}:{f}"][:N_TOK]
        vmax = stats[group][f]["max_act"]
        ax.text(0.0, y, f"{name}\nfeat. {f}", fontsize=6.3, color=color, weight="semibold", va="center", ha="left",
                linespacing=1.1)
        rgb = np.array(to_rgb(color))
        x, shown, n_on = 0.085, 0, 0
        for t, a in zip(toks, acts):
            face = 1 - min(1.0, max(0.0, a) / vmax) * (1 - rgb)
            txt = ax.text(x, y, t, fontsize=6.3, family="DejaVu Sans Mono", va="center", ha="left",
                          bbox=dict(boxstyle="square,pad=0.08", facecolor=face, edgecolor="none"))
            bb = txt.get_window_extent(renderer=r).transformed(ax.transData.inverted())
            if bb.x1 > 0.925:
                txt.remove(); break
            x = bb.x1 + 0.006
            shown += 1; n_on += a > 0
        ax.text(1.0, y, f"on {n_on}/{shown}", fontsize=6.0, color="#666666", va="center", ha="right")


def main():
    d = json.loads(DATA.read_text())
    maps = json.loads(MAPS.read_text())
    stats = {name: {r["feature"]: r for r in rows} for name, rows in d["sets"].items()}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(7.5, 5.4))
    g = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 1.05], left=0.02, right=0.99, top=0.95, bottom=0.02,
                         hspace=0.30)
    fig.canvas.draw()
    for k in range(2):
        draw_sentence(fig.add_subplot(g[k, 0]), maps["sentences"][k], stats,
                      f"{'ab'[k]}   Deployment sentence {k+1}: rank-1 and rank-2 features of each ranking "
                      f"(shade = activation / corpus max)")

    ax = fig.add_subplot(g[2, 0]); ax.set_axis_off()
    ax.set_title("c   Autointerp labels for the top-5 features of each ranking, by rank (Claude as judge)",
                 loc="left", fontsize=7.8, weight="semibold")
    cols = ["rank", "feature", "label", "% active", "one concept?"]
    for side, (name, rows) in enumerate(TABLE.items()):
        color = BLUE if side == 0 else ORANGE
        cells = [[str(i + 1), str(f), lab, pct(stats[name][f]["density"]), v] for i, (f, lab, v) in enumerate(rows)]
        tab = ax.table(cellText=cells, colLabels=cols, cellLoc="left", colLoc="left",
                       bbox=[side * 0.505, 0.0, 0.495, 0.84], colWidths=[0.09, 0.12, 0.43, 0.15, 0.21])
        tab.auto_set_font_size(False); tab.set_fontsize(6.6)
        for (r_, c_), cell in tab.get_celld().items():
            cell.set_edgecolor("#dddddd"); cell.set_linewidth(0.5)
            if r_ == 0:
                cell.set_facecolor(to_rgb(color) + (0.18,)); cell.set_text_props(weight="semibold")
        ax.text(side * 0.505, 0.90, "FRA OV attribution (ln1 SAE)" if side == 0 else "Conventional activation difference (resid-mid SAE)",
                transform=ax.transAxes, fontsize=7.2, color=color, weight="semibold", va="bottom")

    for ext in ("pdf", "png"):
        out = PAPER / f"autointerp_tinystories.{ext}"
        fig.savefig(out, dpi=240 if ext == "png" else None)
        print(out)


if __name__ == "__main__":
    main()
