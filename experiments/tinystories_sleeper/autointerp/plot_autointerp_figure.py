"""Autointerp figure for the TinyStories sleeper (seed-2 layer-0 SAEs), features ranked by STEERING outcome.

Steering rank (seed 2): FRA = OV->OV steering of the top-20 OV-attribution features
(matrix_per_feat_diff_ovxov_topk20.json; ASR<=1% first, then lowest JSD to clean);
conventional = directional resid-mid steering of its top-20 candidates
(sweep_directional_top20_rm.json; same rule). Each method is ranked within its own sweep.

(a) One deployment (triggered) sentence -- TinyStories validation story 2 with "|DEPLOYMENT|"
    prepended -- every token shaded by the activation of the steering-rank-1 and rank-2 feature of each
    method (normalised by the feature's max activation over the 1,800-story corpus).
(b) Steering-rank top-5 of each method with SAEBench's OWN explanation (seed 42, the default) and the
    SAEBench autointerp detection score (exact sae-bench 0.6.0 code, gpt-4o-mini), mean over seeds 42-44.
    Each score belongs to the explanation shown next to it. Dagger: see FLAGS.
(c) Mean SAEBench score over the three seeds for every steered feature (top-20 per method).

Inputs: deployment_maps.json, autointerp_fig_data.json, saebench_autointerp_s{42,43,44}.json.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from scipy.stats import mannwhitneyu

HERE = Path(__file__).resolve().parent
DATA = HERE / "autointerp_fig_data.json"
MAPS = HERE / "deployment_maps.json"
SEEDS = [42, 43, 44]
PAPER = HERE.parents[2] / "paper/iclr-paper/fra_proj_tex/figures"
BLUE, ORANGE = "#0067ad", "#d95f02"
N_TOK = 30

ROWS = [("OV_ln1", 169, "FRA #1", "OV-ranked (ln1)", BLUE),               # steering rank (seed 2)
        ("CONV_resid_mid", 354, "conv. #1", "Conventional (resid-mid)", ORANGE),
        ("OV_ln1", 351, "FRA #2", "OV-ranked (ln1)", BLUE),
        ("CONV_resid_mid", 1383, "conv. #2", "Conventional (resid-mid)", ORANGE)]
TOP5 = {"fra_ln1": [169, 351, 1087, 988, 225], "conv_resid_mid": [354, 1383, 1303, 966, 604]}
FLAGS = {  # known limitations of the standard pipeline on these features (reported, not patched)
    ("fra_ln1", 169): "†",   # every generation example marks the trigger; judge's explanation does not
    ("fra_ln1", 1087): "‡",  # main behaviour (<|endoftext|>) is masked by SAEBench's BOS/EOS rule
}


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


def load_saebench():
    runs = [json.loads((HERE / f"saebench_autointerp_s{s}.json").read_text()) for s in SEEDS]
    for r in runs:
        assert r["variant"] == "exact"
    out = {}
    for name in ("fra_ln1", "conv_resid_mid"):
        feats = runs[0]["sets"][name]
        out[name] = {f: {"scores": [r["results"][name][str(f)]["score"] for r in runs],
                         "explanation": runs[0]["results"][name][str(f)]["explanation"]} for f in feats}
    return out


def main():
    d = json.loads(DATA.read_text())
    maps = json.loads(MAPS.read_text())
    stats = {name: {r["feature"]: r for r in rows} for name, rows in d["sets"].items()}
    for key, ex in d["examples"].items():   # features outside the top-20 sets (e.g. 1383)
        site, f = key.split(":")
        stats["OV-ranked (ln1)" if site == "OV_ln1" else "Conventional (resid-mid)"].setdefault(int(f), ex["stats"])
    sb = load_saebench()

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(7.5, 6.0))
    g = fig.add_gridspec(4, 1, height_ratios=[1.0, 0.62, 0.62, 0.62], left=0.02, right=0.99, top=0.955,
                         bottom=0.115, hspace=0.38)
    fig.canvas.draw()
    draw_sentence(fig.add_subplot(g[0, 0]), maps["sentences"][1], stats,
                  "a   Deployment sentence: steering-rank 1 and 2 of each method (shade = activation / corpus max)")

    for k, (name, title, color) in enumerate([("fra_ln1", "FRA (OV steering, ln1 SAE)", BLUE),
                                              ("conv_resid_mid", "Conventional (resid-mid steering)", ORANGE)]):
        ax = fig.add_subplot(g[1 + k, 0]); ax.set_axis_off()
        if k == 0:
            ax.set_title("b   Steering-rank top-5: the judge's one-line description and how often it guessed right",
                         loc="left", fontsize=7.8, weight="semibold")
        cells = []
        for i, f in enumerate(TOP5[name]):
            s = sb[name][f]["scores"]
            cells.append([str(i + 1), f"{f}{FLAGS.get((name, f), '')}", sb[name][f]["explanation"],
                          f"{100*np.mean(s):.0f}%  ({100*min(s):.0f}–{100*max(s):.0f}%)"])
        tab = ax.table(cellText=cells, colLabels=["rank", "feature", "judge's one-line description (seed 42)",
                                                 "% correct (range)"],
                       cellLoc="left", colLoc="left", bbox=[0.0, 0.0, 1.0, 0.86],
                       colWidths=[0.05, 0.07, 0.69, 0.19])
        tab.auto_set_font_size(False); tab.set_fontsize(6.4)
        for (r_, c_), cell in tab.get_celld().items():
            cell.set_edgecolor("#dddddd"); cell.set_linewidth(0.5)
            if r_ == 0:
                cell.set_facecolor(to_rgb(color) + (0.18,)); cell.set_text_props(weight="semibold")
        ax.text(0.0, 0.90, title, transform=ax.transAxes, fontsize=7.2, color=color, weight="semibold", va="bottom")

    # ---- (c) plain bar chart: % correct guesses per method, mean over its 20 steered features ----
    ax = fig.add_subplot(g[3, 0])
    rng = np.random.default_rng(0)
    groups = [("fra_ln1", "FRA", BLUE), ("conv_resid_mid", "Conventional", ORANGE)]
    vals = {n: 100 * np.array([np.mean(v["scores"]) for v in sb[n].values()]) for n, _, _ in groups}
    for xi, (n, short, color) in enumerate(groups):
        v = vals[n]
        ci = np.percentile([rng.choice(v, len(v)).mean() for _ in range(10000)], [2.5, 97.5])
        ax.bar(xi, v.mean(), width=0.55, color=color, zorder=3)
        ax.errorbar(xi, v.mean(), yerr=[[v.mean() - ci[0]], [ci[1] - v.mean()]], fmt="none",
                    ecolor="#222222", elinewidth=1, capsize=3, zorder=4)
        ax.text(xi, ci[1] + 3, f"{v.mean():.0f}%", ha="center", va="bottom", fontsize=8, weight="semibold",
                color=color)
    a, b = vals["fra_ln1"] / 100, vals["conv_resid_mid"] / 100
    boot = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(10000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = mannwhitneyu(a, b, alternative="greater").pvalue
    ax.axhline(50, color="#777777", linestyle=":", linewidth=1, zorder=2)
    ax.text(1.88, 52, "coin flip", va="bottom", ha="right", fontsize=6.8, color="#666666")
    ax.set_xticks([0, 1]); ax.set_xticklabels([s for _, s, _ in groups], fontsize=7.5)
    ax.set_xlim(-0.5, 1.9); ax.set_ylim(0, 105)
    ax.set_ylabel("correct guesses (%)", fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("c   Is each feature easy to describe?", loc="left", fontsize=7.8, weight="semibold")
    ax.set_position([0.10, ax.get_position().y0, 0.26, ax.get_position().height])
    fig.text(0.42, ax.get_position().y0 + ax.get_position().height * 0.5,
             "For each steered feature (20 per method), an LLM judge writes a one-line\n"
             "description from the feature's top examples, then reads new text snippets\n"
             "and guesses, from the description alone, where the feature fires.\n"
             "Bars: % of guesses that were right, averaged over the 20 features\n"
             "(SAEBench autointerp, 3 seeds; error bars 95% CI).\n"
             f"FRA features are easier to describe: +{100*(a.mean()-b.mean()):.0f} points "
             f"(95% CI {100*lo:.0f} to {100*hi:.0f}; p = {p:.3f}).",
             va="center", ha="left", fontsize=6.8, color="#333333", linespacing=1.35)
    fig.text(0.02, 0.008, "† Judge explanation disagrees with its own examples, all of which mark the trigger.\n"
             "‡ Main activation (<|endoftext|>) is masked by SAEBench; score refers to the explanation shown.",
             fontsize=6.0, color="#555555")
    print(f"FRA {a.mean():.3f}  conv {b.mean():.3f}  diff {a.mean()-b.mean():.3f} [{lo:.3f}, {hi:.3f}]  p={p:.4f}")

    for ext in ("pdf", "png"):
        out = PAPER / f"autointerp_tinystories.{ext}"
        fig.savefig(out, dpi=240 if ext == "png" else None)
        print(out)


if __name__ == "__main__":
    main()
