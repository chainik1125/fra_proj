"""Per-seed Fig-10 analogue: NEW OV rank-1 vs conventional resid-mid additive.

Top row: JSD_clean (green) and JSD_pois (red) vs α.
Bottom row: exact-match rate to clean rollout (green) and ASR (red) vs α.

NEW OV rank-1 (solid, circle)  = top-attribution OV tuple per SAE seed from
                                 run_experiment.py output (ov_topk20_gated_all.json).
Conventional  (dashed, ▲)      = per-seed resid-mid additive winner from
                                 jsd_alpha_sweep_6seeds.json (the conv config produced
                                 by scripts/downstream_baseline.py +
                                 build_legacy_alpha_sweep_json.py).

Bands: per-α min–max envelope over the 5 lockstep decoding seeds.
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


def draw(ax, xs, mean, lo, hi, color, ls, mk, lw, label):
    ax.fill_between(xs, lo, hi, color=color, alpha=BAND_ALPHA, lw=0, zorder=1)
    ax.plot(xs, mean, color=color, linestyle=ls, marker=mk, ms=4, lw=lw,
            label=label, zorder=3)


def stats_from_per_seed(per_seed: list[float], scale: float = 1.0):
    arr = np.asarray(per_seed, dtype=float) * scale
    return float(arr.mean()), float(arr.min()), float(arr.max())


def new_ov_curves(rank1_row: dict, alphas: list[float], n_prompts_per_seed: int):
    """Curves for a single tuple's α-sweep (new schema: alpha_sweep[str(a)] -> ev)."""
    out = {m: ([], [], []) for m in ("jsd_clean", "jsd_pois", "exact_match", "asr")}
    for a in alphas:
        ev = rank1_row["alpha_sweep"][f"{a}"]
        for m in ("jsd_clean", "jsd_pois", "asr"):
            mn, lo, hi = stats_from_per_seed(ev[f"{m}_per_seed"])
            out[m][0].append(mn); out[m][1].append(lo); out[m][2].append(hi)
        em_rates = [c / n_prompts_per_seed for c in ev["n_exact_match_clean_per_seed"]]
        mn, lo, hi = stats_from_per_seed(em_rates)
        out["exact_match"][0].append(mn); out["exact_match"][1].append(lo); out["exact_match"][2].append(hi)
    return {m: (np.array(v[0]), np.array(v[1]), np.array(v[2])) for m, v in out.items()}


def conv_curves(conv_per_alpha: dict, seed_idx: int, alphas: list[float], n_prompts: int):
    """Curves for the conventional baseline at one SAE seed (legacy schema:
    per_alpha[str(a)][metric][sae_seed_idx] = list of 5 eval-seed values)."""
    out = {m: ([], [], []) for m in ("jsd_clean", "jsd_pois", "exact_match", "asr")}
    for a in alphas:
        ev_a = conv_per_alpha[f"{a:.1f}"]
        for m in ("jsd_clean", "jsd_pois", "asr"):
            mn, lo, hi = stats_from_per_seed(ev_a[m][seed_idx])
            out[m][0].append(mn); out[m][1].append(lo); out[m][2].append(hi)
        em_rates = [c / n_prompts for c in ev_a["n_exact_match_clean"][seed_idx]]
        mn, lo, hi = stats_from_per_seed(em_rates)
        out["exact_match"][0].append(mn); out["exact_match"][1].append(lo); out["exact_match"][2].append(hi)
    return {m: (np.array(v[0]), np.array(v[1]), np.array(v[2])) for m, v in out.items()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ov_json",   type=Path,
                   default=Path("results/ov_topk20_gated_all.json"))
    p.add_argument("--conv_json", type=Path,
                   default=Path("results/jsd_alpha_sweep_6seeds.json"))
    p.add_argument("--output",    type=Path,
                   default=Path("figures/jsd_exact_ov_vs_conv_perseed"))
    args = p.parse_args()

    setup_style()

    ov_d   = json.loads(args.ov_json.read_text())
    conv_d = json.loads(args.conv_json.read_text())

    ov_rows = ov_d["results"]
    seeds   = sorted({r["seed"] for r in ov_rows})
    alphas  = sorted({float(a) for r in ov_rows for a in r["alpha_sweep"]})
    n_total = int(ov_rows[0]["alpha_sweep"][f"{alphas[0]}"]["exact_match_total_rows"])
    n_per_seed = n_total // len(ov_rows[0]["alpha_sweep"][f"{alphas[0]}"]["asr_per_seed"])

    # rank-1 per SAE seed: first row in per-seed list (sorted by attribution score desc)
    ov_per_seed = {s: [r for r in ov_rows if r["seed"] == s] for s in seeds}
    rank1 = {s: ov_per_seed[s][0] for s in seeds}

    conv_seeds   = conv_d["sae_seeds"]
    conv_per_a   = conv_d["configs"]["conventional"]["per_alpha"]
    conv_feats   = conv_d["configs"]["conventional"]["per_seed_feature"]
    n_conv       = conv_d["n_prompts"]

    disp_alphas = [-a for a in alphas]

    fig, axes = plt.subplots(
        2, len(seeds), figsize=(2.6 * len(seeds), 5.6),
        sharex=True, sharey="row",
    )

    for col, seed in enumerate(seeds):
        ax_jsd, ax_em = axes[0, col], axes[1, col]
        ov_c   = new_ov_curves(rank1[seed], alphas, n_per_seed)
        cv_c   = (conv_curves(conv_per_a, conv_seeds.index(seed), alphas, n_conv)
                  if seed in conv_seeds else None)

        for curves, label_prefix, ls, mk, lw in (
            [(ov_c, "OV rank-1", "-",  "o", 1.3)]
            + ([(cv_c, "conv.",   "--", "^", 1.1)] if cv_c is not None else [])
        ):
            for metric, color, lbl in (
                ("jsd_clean", GREEN, "JSD$_\\mathrm{clean}$"),
                ("jsd_pois",  RED,   "JSD$_\\mathrm{pois}$"),
            ):
                m, lo, hi = curves[metric]
                draw(ax_jsd, disp_alphas, m, lo, hi, color, ls, mk, lw,
                     f"{label_prefix}  {lbl}")
            for metric, color, lbl in (
                ("exact_match", GREEN, "exact-match"),
                ("asr",         RED,   "ASR"),
            ):
                m, lo, hi = curves[metric]
                draw(ax_em, disp_alphas, m, lo, hi, color, ls, mk, lw,
                     f"{label_prefix}  {lbl}")
        ax_jsd.axhline(1.0, color="#999", lw=0.5, ls=":")
        ax_jsd.set_ylim(-0.02, 1.05)
        ax_em .set_ylim(-0.02, 1.05)
        ax_em .yaxis.set_major_formatter(PercentFormatter(1.0))

        ov_feat = rank1[seed]["tuple"][0][0]
        cv_feat = conv_feats.get(str(seed), "—")
        ax_jsd.set_title(
            f"seed {seed}\nOV f={ov_feat} · conv f={cv_feat}",
            fontsize=10,
        )

        if col == 0:
            ax_jsd.set_ylabel("JSD (bits)")
            ax_em .set_ylabel("Exact-match rate / ASR")
        for ax in (ax_jsd, ax_em):
            ax.grid(axis="y", color="#dddddd", lw=0.5)
            ax.set_axisbelow(True)

    axes[0, 0].legend(loc="lower left", fontsize=7.5, framealpha=0.92,
                      handlelength=1.6, borderpad=0.4)
    axes[1, 0].legend(loc="lower left", fontsize=7.5, framealpha=0.92,
                      handlelength=1.6, borderpad=0.4,
                      bbox_to_anchor=(0.0, 0.30))
    fig.suptitle("Per-seed OV rank-1 (new pipeline) vs conventional resid-mid",
                 fontsize=15, y=1.02)
    for ax in axes.flat:
        ax.set_xlabel("")
    fig.supxlabel(r"steering strength $-\alpha$", fontsize=15, y=0.0)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".pdf"))
    fig.savefig(args.output.with_suffix(".png"), dpi=180)
    print(f"wrote {args.output.with_suffix('.pdf')} and {args.output.with_suffix('.png')}")


if __name__ == "__main__":
    main()
