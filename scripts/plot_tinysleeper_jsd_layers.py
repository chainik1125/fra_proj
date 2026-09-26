"""Supporting figure: the Fig. 3(b-d) JSD panels repeated at TinyStories layers 0-3.

Data: data/tinystories/wide_screen_layers.json, collected from the rerun of the
wide-alpha protocol (experiments/tinystories_sleeper/wide_screen/run_layers.py) with
the six-seed retrained SAEs at each layer. As in plot_tinysleeper_jsd.py, each
seed/method curve is the screen curve of the feature selected for BO refinement,
averaged over six SAE seeds, feature-subtraction half only (raw alpha in [0, 10]).

Inputs : data/tinystories/wide_screen_layers.json
Outputs: figures/fig4_jsd_layers.{pdf,png}

    uv run scripts/plot_tinysleeper_jsd_layers.py                       # plot
    uv run scripts/plot_tinysleeper_jsd_layers.py --collect <results_dir>  # rebuild JSON from raw sweeps
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


from _paths import DATA as _DATA, FIGURES

DATA = _DATA / "tinystories" / "wide_screen_layers.json"
LAYERS = (0, 1, 2, 3)
SEEDS = tuple(range(6))

METHODS = (
    ("ov", "FRA", "#0067ad", "o"),
    ("conventional", "resid-mid", "#d95f02", "^"),
    ("conv_ln1", "ln1 SAE", "#008f64", "s"),
)


def collect(result_dirs: list[Path]) -> None:
    out = {
        "description": "TinyStories wide-alpha screen at layers 0-3 (Fig. 4 protocol, "
                       "the original wide-alpha screen script, generalised to layer L); "
                       "BO-selected feature per SAE seed",
        "metric_order": ["jsd_clean", "jsd_pois", "clean_match", "asr"],
        "raw_alpha_sign": "positive subtracts feature; plotted alpha = raw alpha",
        "sae_seeds": list(SEEDS),
        "layers": {},
    }
    for src in ("retrain", "fig4", "sae20k"):
        for L in LAYERS:
            cells = {}
            for method, *_ in METHODS:
                cells[method] = {}
                for seed in SEEDS:
                    name = f"{src}_L{L}_{method}_s{seed}.json"
                    p = next((r / name for r in result_dirs if (r / name).exists()), None)
                    if p is None:
                        continue
                    d = json.loads(p.read_text())
                    feat = d["bo"]["feat"] if "bo" in d else d["winner"]["feat"]
                    rows = {r["alpha"]: r for r in d["rows"] if r["feat"] == feat}
                    out["raw_alphas"] = d["alphas"]
                    out["screen_prompts"] = d["n_prompts"]
                    cells[method][str(seed)] = {
                        "feature": feat,
                        "winner_gated": d["winner_gated"],
                        "baseline_asr": d["baseline_asr"],
                        "bo": {k: v for k, v in d.get("bo", {}).items() if k != "evals"},
                        "screen": [[rows[a]["jsd_clean"], rows[a]["jsd_pois"],
                                    rows[a]["exact"] / d["n_prompts"], rows[a]["asr"]]
                                   for a in d["alphas"]],
                    }
            if any(cells[m] for m in cells):
                out["layers"][f"{src}_L{L}"] = cells
    DATA.write_text(json.dumps(out, indent=1))
    print(f"wrote {DATA} with {sorted(out['layers'])}")


def load_means(key: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    data = json.loads(DATA.read_text())
    raw = np.asarray(data["raw_alphas"], dtype=float)
    keep = (raw >= 0) & (raw <= 10)
    order = np.argsort(raw[keep])
    means = {}
    for method, *_ in METHODS:
        cells = data["layers"][key][method]
        per_seed = np.asarray([cells[str(s)]["screen"] for s in SEEDS if str(s) in cells])
        means[method] = per_seed[:, keep, :].mean(axis=0)[order]
    return raw[keep][order], means


def main() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(3, 4, figsize=(13.0, 8.4), sharey="row")
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.07, top=0.95, hspace=0.42, wspace=0.12)
    for col, L in enumerate(LAYERS):
        alpha, means = load_means(f"retrain_L{L}")
        for ax in axes[:, col]:
            ax.grid(color="#dddddd", linewidth=0.55, alpha=0.8)
            ax.set_axisbelow(True)
        for method, label, color, marker in METHODS:
            r = means[method]
            kw = dict(color=color, linewidth=2.0, marker=marker, markersize=4.2, markevery=4,
                      label=label)
            axes[0, col].plot(r[:, 1], 1 - r[:, 0], **kw)
            axes[1, col].plot(alpha, r[:, 0], **kw)
            axes[2, col].plot(alpha, r[:, 1], **kw)
        axes[0, col].set_title(f"Layer {L}", fontsize=10)
        axes[0, col].set(xlim=(-0.02, 1.02), ylim=(-0.01, 0.72), xlabel="JSD to sleeper (bits)")
        for row in (1, 2):
            axes[row, col].set(xlim=(-0.3, 10.3), ylim=(-0.025, 1.025),
                               xlabel=r"steering strength $\alpha$")
            axes[row, col].set_xticks(np.arange(0, 11, 2))
    axes[0, 0].set_ylabel("1 - JSD to clean")
    axes[1, 0].set_ylabel("JSD to clean (bits)")
    axes[2, 0].set_ylabel("JSD to sleeper (bits)")
    axes[1, 0].legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.95)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "fig4_jsd_layers.pdf")
    fig.savefig(FIGURES / "fig4_jsd_layers.png", dpi=180)
    plt.close(fig)
    print(f"wrote {FIGURES / 'fig4_jsd_layers.pdf'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", type=Path, nargs="+", default=None,
                    help="raw run_layers.py result dirs")
    a = ap.parse_args()
    if a.collect:
        collect(a.collect)
    main()
