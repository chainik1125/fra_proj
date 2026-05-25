"""6-column per-seed appendix figure from the new matrix + baseline JSONs.

Each column is one SAE training seed. Top row: JSD$_\\text{clean}$ (green)
and JSD$_\\text{pois}$ (red). Bottom row: exact-match rate (green) + ASR (red).

For every (SAE seed, α) the new JSON stores the mean over 1000
(prompt × sampling-seed) trials and the std across those 1000 trials;
the line is the mean, the band is mean ± 1 std.

OV  (single OV→OV)   = solid    + circle
Conv (resid-mid add) = dashed   + triangle

Outputs <out>.pdf and <out>.png.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

GREEN = "#2E7D32"
RED   = "#B91C1C"
BAND_ALPHA = 0.18


def setup_style() -> None:
    mpl.rcParams.update({
        "font.family":       "sans-serif",
        "font.sans-serif":   ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size":         11,
        "axes.titlesize":    12,
        "axes.labelsize":    11,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.linewidth":    1.0,
        "axes.edgecolor":    "#222222",
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.fontsize":   8,
        "legend.frameon":    True,
        "figure.dpi":        120,
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.12,
    })


def _lookup_alpha_key(per_alpha: dict, a: float) -> str:
    for key in (str(a), f"{a:.1f}", f"{a:.2f}", repr(a)):
        if key in per_alpha:
            return key
    raise KeyError(f"α={a} not in keys {list(per_alpha)[:6]}…")


def mean_std_arrays(per_alpha: dict, alphas: list[float],
                    mean_key: str, std_key: str
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m, sd = [], []
    for a in alphas:
        k = _lookup_alpha_key(per_alpha, a)
        m.append(float(per_alpha[k][mean_key]))
        sd.append(float(per_alpha[k][std_key]))
    m_arr = np.array(m); sd_arr = np.array(sd)
    return m_arr, m_arr - sd_arr, m_arr + sd_arr


def draw(ax, xs, mean, lo, hi, color, ls, mk, lw, label):
    ax.fill_between(xs, lo, hi, color=color, alpha=BAND_ALPHA, lw=0, zorder=1)
    ax.plot(xs, mean, color=color, linestyle=ls, marker=mk, ms=4, lw=lw,
            label=label, zorder=3)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--matrix_json",   type=Path,
                   default=Path("results/matrix_4k_diff_rank.json"))
    p.add_argument("--baseline_json", type=Path,
                   default=Path("results/downstream_baseline_4k.json"))
    p.add_argument("--output", type=Path,
                   default=Path("figures/jsd_exact_all_seeds"))
    args = p.parse_args()

    setup_style()
    matrix = json.loads(args.matrix_json.read_text())
    base   = json.loads(args.baseline_json.read_text())

    matrix_by_seed = {r["seed"]: r for r in matrix["results"]}
    seeds = sorted(matrix_by_seed.keys())

    # Take α grid from the first seed's eval_sweep.
    first_es = matrix_by_seed[seeds[0]]["winner"]["eval_sweep"]
    raw_alphas = sorted([float(k) for k in first_es.keys()])
    disp_alphas = [-a for a in raw_alphas]

    fig, axes = plt.subplots(
        2, len(seeds), figsize=(2.6 * len(seeds), 5.6),
        sharex=True, sharey="row",
    )

    for col, seed in enumerate(seeds):
        ax_jsd, ax_em = axes[0, col], axes[1, col]
        r = matrix_by_seed[seed]
        ov_pa = r["winner"]["eval_sweep"]
        ov_feat = r["winner"]["feature"]

        base_entry = base["per_seed"][f"s{seed}"]
        conv_pa = base_entry["per_alpha"]
        conv_feat = base_entry["winner"]

        # — JSD curves —
        for pa, label_prefix, ls, mk, lw in [
            (ov_pa,   "OV$\\to$OV",  "-",  "o", 1.3),
            (conv_pa, "conv.",        "--", "^", 1.1),
        ]:
            jc_m, jc_lo, jc_hi = mean_std_arrays(pa, raw_alphas, "jsd_clean",
                                                  "jsd_clean_std")
            jp_m, jp_lo, jp_hi = mean_std_arrays(pa, raw_alphas, "jsd_pois",
                                                  "jsd_pois_std")
            draw(ax_jsd, disp_alphas, jc_m, jc_lo, jc_hi, GREEN, ls, mk, lw,
                 f"{label_prefix} JSD$_\\text{{clean}}$")
            draw(ax_jsd, disp_alphas, jp_m, jp_lo, jp_hi, RED, ls, mk, lw,
                 f"{label_prefix} JSD$_\\text{{pois}}$")

        ax_jsd.set_title(f"seed {seed}  ·  OV f{ov_feat} / conv f{conv_feat}",
                         fontsize=10)
        ax_jsd.axhline(1.0, color="#888", linestyle=":", lw=0.7, alpha=0.6)
        ax_jsd.set_ylim(-0.04, 1.10)
        if col == 0:
            ax_jsd.set_ylabel("JSD (bits)")
        ax_jsd.grid(True, axis="y", color="#eeeeee", lw=0.6, zorder=0)
        ax_jsd.set_axisbelow(True)

        # — Exact-match (green) + ASR (red) —
        for pa, label_prefix, ls, mk, lw in [
            (ov_pa,   "OV$\\to$OV",  "-",  "o", 1.3),
            (conv_pa, "conv.",        "--", "^", 1.1),
        ]:
            em_m, em_lo, em_hi = mean_std_arrays(pa, raw_alphas, "exact_match",
                                                  "exact_match_std")
            asr_m, asr_lo, asr_hi = mean_std_arrays(pa, raw_alphas, "asr",
                                                     "asr_std")
            draw(ax_em, disp_alphas, em_m, em_lo, em_hi, GREEN, ls, mk, lw,
                 f"{label_prefix} exact-match")
            draw(ax_em, disp_alphas, asr_m, asr_lo, asr_hi, RED, ls, mk, lw,
                 f"{label_prefix} ASR")

        ax_em.set_ylim(-0.05, 1.15)
        ax_em.set_xticks(disp_alphas)
        ax_em.set_xticklabels([f"{a:.4g}" for a in disp_alphas])
        plt.setp(ax_em.get_xticklabels(), rotation=45, ha="right")
        ax_em.set_xlabel(r"$\alpha$")
        ax_em.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        if col == 0:
            ax_em.set_ylabel("exact-match / ASR")
        ax_em.grid(True, axis="y", color="#eeeeee", lw=0.6, zorder=0)
        ax_em.set_axisbelow(True)

    # Single legend on the top-right axis
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.04), fontsize=9,
               framealpha=0.95, edgecolor="#bbbbbb")
    fig.suptitle("Per-seed α sweep · band = ±1 std over 1000 trials",
                 fontsize=11, y=1.10)
    fig.tight_layout()

    out = Path(args.output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    base_str = str(out).removesuffix('.png').removesuffix('.pdf')
    fig.savefig(base_str + ".png", dpi=200)
    fig.savefig(base_str + ".pdf")
    plt.close(fig)
    print(f"wrote {base_str}.png and {base_str}.pdf")


if __name__ == "__main__":
    main()
