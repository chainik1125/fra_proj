"""6-column per-seed appendix figure from jsd_alpha_sweep_6seeds.json.

Same layout / style as the single-seed plot but tiled across all SAE seeds.
Top row: JSD curves (clean=green, poisoned=red).
Bottom row: exact-match rate (blue) + ASR (red), both as a fraction with Wilson
95% CIs over n=200 prompts.

OV  (single OV→OV)   = solid    + circle
Conv (resid-mid add) = dashed   + triangle

X axis: steering strength −α (negative because we subtract the SAE feature).

Outputs <out>.pdf and <out>.png.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


GREEN = "#2E7D32"
RED   = "#B91C1C"
BLUE  = "#1F4E96"


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


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval (lo, hi) for proportion p with sample n."""
    if n <= 0:
        return (p, p)
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input",  type=Path,
                   default=Path("results/jsd_alpha_sweep_6seeds.json"))
    p.add_argument("--output", type=Path,
                   default=Path("figures/jsd_exact_all_seeds"))
    args = p.parse_args()

    setup_style()
    d = json.loads(args.input.read_text())
    raw_alphas  = [float(a) for a in d["alphas"]]
    disp_alphas = [-a for a in raw_alphas]
    seeds       = d["sae_seeds"]
    n_prompts   = d["n_prompts"]
    ov          = d["configs"]["ov"]
    conv        = d["configs"]["conventional"]
    feats_ov    = ov["per_seed_feature"]
    feats_conv  = conv["per_seed_feature"]

    fig, axes = plt.subplots(
        2, len(seeds), figsize=(2.6 * len(seeds), 5.6),
        sharex=True, sharey="row",
    )

    for col, seed in enumerate(seeds):
        ax_jsd, ax_em = axes[0, col], axes[1, col]
        i = seeds.index(seed)

        # — JSD curves —
        jsd_c_ov  = [ov  ["per_alpha"][f"{a:.1f}"]["jsd_clean"][i] for a in raw_alphas]
        jsd_p_ov  = [ov  ["per_alpha"][f"{a:.1f}"]["jsd_pois"] [i] for a in raw_alphas]
        jsd_c_cv  = [conv["per_alpha"][f"{a:.1f}"]["jsd_clean"][i] for a in raw_alphas]
        jsd_p_cv  = [conv["per_alpha"][f"{a:.1f}"]["jsd_pois"] [i] for a in raw_alphas]

        ax_jsd.plot(disp_alphas, jsd_c_ov, "-",  color=GREEN, marker="o", ms=4, lw=1.3,
                    label="OV$\\to$OV  JSD$_\\mathrm{clean}$")
        ax_jsd.plot(disp_alphas, jsd_p_ov, "-",  color=RED,   marker="o", ms=4, lw=1.3,
                    label="OV$\\to$OV  JSD$_\\mathrm{pois}$")
        ax_jsd.plot(disp_alphas, jsd_c_cv, "--", color=GREEN, marker="^", ms=4, lw=1.1,
                    label="conv.  JSD$_\\mathrm{clean}$")
        ax_jsd.plot(disp_alphas, jsd_p_cv, "--", color=RED,   marker="^", ms=4, lw=1.1,
                    label="conv.  JSD$_\\mathrm{pois}$")
        ax_jsd.axhline(1.0, color="#999", lw=0.5, ls=":")
        ax_jsd.set_ylim(-0.02, 1.05)
        ax_jsd.set_title(
            f"seed {seed}\nOV f={feats_ov[str(seed)]} · conv f={feats_conv[str(seed)]}",
            fontsize=10,
        )

        # — exact-match (blue) + ASR (red) with Wilson CIs —
        em_ov  = [ov  ["per_alpha"][f"{a:.1f}"]["frac_pos_match_clean"][i] for a in raw_alphas]
        em_cv  = [conv["per_alpha"][f"{a:.1f}"]["frac_pos_match_clean"][i] for a in raw_alphas]
        asr_ov = [ov  ["per_alpha"][f"{a:.1f}"]["asr"][i] for a in raw_alphas]
        asr_cv = [conv["per_alpha"][f"{a:.1f}"]["asr"][i] for a in raw_alphas]

        def errbars(props: list[float]) -> tuple[list[float], list[float]]:
            lo_arr, hi_arr = [], []
            for v in props:
                lo, hi = wilson(v, n_prompts)
                lo_arr.append(v - lo)
                hi_arr.append(hi - v)
            return lo_arr, hi_arr

        for vals, style, color, lbl in [
            (em_ov,  dict(ls="-",  marker="o", ms=4, lw=1.3), BLUE,
             "OV$\\to$OV  exact-match"),
            (em_cv,  dict(ls="--", marker="^", ms=4, lw=1.1), BLUE,
             "conv.  exact-match"),
            (asr_ov, dict(ls="-",  marker="o", ms=4, lw=1.3), RED,
             "OV$\\to$OV  ASR"),
            (asr_cv, dict(ls="--", marker="^", ms=4, lw=1.1), RED,
             "conv.  ASR"),
        ]:
            lo, hi = errbars(vals)
            ax_em.errorbar(disp_alphas, vals, yerr=[lo, hi], color=color,
                           elinewidth=0.6, capsize=2.0, label=lbl, **style)
        ax_em.set_ylim(-0.02, 1.05)
        ax_em.yaxis.set_major_formatter(PercentFormatter(1.0))

        if col == 0:
            ax_jsd.set_ylabel("JSD (bits)")
            ax_em .set_ylabel("Clean-match rate / ASR")
        ax_em.set_xlabel(r"steering strength $-\alpha$")
        ax_jsd.grid(axis="y", color="#dddddd", lw=0.5)
        ax_em .grid(axis="y", color="#dddddd", lw=0.5)
        ax_jsd.set_axisbelow(True)
        ax_em .set_axisbelow(True)

    axes[0, 0].legend(loc="center left", fontsize=7.5, framealpha=0.92,
                      handlelength=1.6, borderpad=0.4)
    axes[1, 0].legend(loc="center left", fontsize=7.5, framealpha=0.92,
                      handlelength=1.6, borderpad=0.4)
    fig.suptitle(
        f"Per-seed JSD and rollout-level companion (n={n_prompts} prompts; "
        f"Wilson 95\\% CIs on the proportions). "
        f"Solid + circle: single OV$\\to$OV.  Dashed + triangle: conventional resid-mid additive.",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".pdf"))
    fig.savefig(args.output.with_suffix(".png"), dpi=180)
    print(f"wrote {args.output.with_suffix('.pdf')}  and  {args.output.with_suffix('.png')}")


if __name__ == "__main__":
    main()
