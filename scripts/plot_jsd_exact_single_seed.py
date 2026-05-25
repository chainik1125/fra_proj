"""1×2 figure (JSD + exact-match/ASR) at a single SAE seed.

Reads two JSONs in the new schema produced by matrix.py + downstream_baseline.py:

  --matrix_json   results/matrix_4k_diff_rank.json
    results[i].winner.eval_sweep[α].{asr, asr_std,
                                     jsd_clean, jsd_clean_std,
                                     jsd_pois,  jsd_pois_std,
                                     exact_match, exact_match_std, …}

  --baseline_json results/downstream_baseline_4k.json
    per_seed.s{i}.per_alpha[α].{same keys}

The line is the mean across (200 dep prompts × 5 sampling seeds) = 1000
trials at each α; the translucent band is ±1 std across the same 1000
trials (token-position variance collapsed into the per-row mean first).

Left panel  — JSD(steered, clean) green, JSD(steered, poisoned) red.
Right panel — exact-match rate (green) + ASR (red).

Both methods (single OV-feature ablation, paired-per-seed) and the
conventional resid-mid additive baseline overlaid.

Usage:
  python -m scripts.plot_jsd_exact_single_seed --seed 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


def setup_style() -> None:
    mpl.rcParams.update({
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size":          15,
        "axes.titlesize":     18,
        "axes.labelsize":     16,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.linewidth":     1.2,
        "axes.edgecolor":     "#222222",
        "xtick.labelsize":    13,
        "ytick.labelsize":    14,
        "legend.frameon":     True,
        "legend.fontsize":    11,
        "figure.dpi":         110,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.10,
    })


def per_alpha_mean_std(per_alpha: dict, raw_alphas: list[float],
                        mean_key: str, std_key: str,
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-α (mean, mean−std, mean+std) for a metric in a per_alpha dict.

    per_alpha is keyed by "0.0", "0.5", … and each entry has both the
    mean and the std fields. Returns three arrays of length len(raw_alphas)."""
    m, lo, hi = [], [], []
    for a in raw_alphas:
        entry = per_alpha[f"{a:.1f}" if (a*10) % 5 == 0 else f"{a}"]
        # Try common float formats — entries in JSON are written via str(α).
        if entry is None:
            for key in (f"{a}", f"{a:.1f}", f"{a:.2f}"):
                if key in per_alpha:
                    entry = per_alpha[key]
                    break
        mu = float(entry[mean_key])
        sd = float(entry[std_key])
        m.append(mu);  lo.append(mu - sd);  hi.append(mu + sd)
    return np.array(m), np.array(lo), np.array(hi)


def _lookup_alpha_key(per_alpha: dict, a: float) -> str:
    """Find the JSON key for α. matrix.py writes str(float), downstream uses f'{α:.1f}'."""
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--matrix_json",   type=Path,
                   default=Path("results/matrix_4k_diff_rank.json"))
    p.add_argument("--baseline_json", type=Path,
                   default=Path("results/downstream_baseline_4k.json"))
    p.add_argument("--output", type=Path,
                   default=Path("figures/jsd_exact_seed"))
    p.add_argument("--seed", type=int, default=5)
    args = p.parse_args()

    setup_style()

    matrix = json.loads(args.matrix_json.read_text())
    base   = json.loads(args.baseline_json.read_text())

    # Find the requested seed in matrix.results
    matrix_by_seed = {r["seed"]: r for r in matrix["results"]}
    if args.seed not in matrix_by_seed:
        raise SystemExit(f"seed {args.seed} not in matrix file")
    r = matrix_by_seed[args.seed]
    ov_eval_sweep = r["winner"]["eval_sweep"]
    ov_feat = r["winner"]["feature"]

    base_seed_key = f"s{args.seed}"
    if base_seed_key not in base["per_seed"]:
        raise SystemExit(f"{base_seed_key} not in baseline file")
    base_entry = base["per_seed"][base_seed_key]
    conv_per_alpha = base_entry["per_alpha"]
    conv_feat = base_entry["winner"]

    # Determine alpha grid from matrix eval_sweep keys (sorted asc)
    raw_alphas = sorted([float(k) for k in ov_eval_sweep.keys()])
    disp_alphas = [-a for a in raw_alphas]

    GREEN = "#1a8a3f"
    RED   = "#c0322a"
    BAND_ALPHA = 0.18

    methods = [
        ("ov",           ov_eval_sweep,    f"single OV$\\rightarrow$OV (f{ov_feat})",     "-",  "o"),
        ("conventional", conv_per_alpha,    f"conventional additive (f{conv_feat})",       "--", "^"),
    ]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14.0, 6.0))

    def draw(ax, xs, mean, lo, hi, color, ls, mk, label):
        ax.fill_between(xs, lo, hi, color=color, alpha=BAND_ALPHA, lw=0, zorder=2)
        ax.plot(xs, mean, color=color, lw=2.4, marker=mk, markersize=7,
                linestyle=ls, markeredgecolor="white", markeredgewidth=0.9,
                label=label, zorder=3)

    # ── Left panel: JSD ─────────────────────────────────────────────────
    for _key, pa, name, ls, mk in methods:
        jc_m, jc_lo, jc_hi = mean_std_arrays(pa, raw_alphas, "jsd_clean", "jsd_clean_std")
        jp_m, jp_lo, jp_hi = mean_std_arrays(pa, raw_alphas, "jsd_pois",  "jsd_pois_std")
        draw(ax_l, disp_alphas, jc_m, jc_lo, jc_hi, GREEN, ls, mk,
             f"{name}  JSD$_\\text{{clean}}$")
        draw(ax_l, disp_alphas, jp_m, jp_lo, jp_hi, RED, ls, mk,
             f"{name}  JSD$_\\text{{pois}}$")

    ax_l.axhline(1.0, color="#888", linestyle=":", lw=0.9, alpha=0.7)
    ax_l.text(disp_alphas[-1], 1.0 - 0.015, "JSD upper bound (1 bit)",
              fontsize=10.5, color="#666", ha="left", va="top")
    ax_l.set_ylabel("Jensen–Shannon divergence (bits)")
    ax_l.set_xlabel(r"steering strength  $\alpha$")
    ax_l.set_ylim(-0.04, 1.10)
    ax_l.set_xticks(disp_alphas)
    ax_l.set_xticklabels([f"{a:.4g}" for a in disp_alphas])
    plt.setp(ax_l.get_xticklabels(), rotation=45, ha="right")
    ax_l.grid(True, axis="y", color="#eeeeee", lw=0.6, zorder=0)
    ax_l.set_axisbelow(True)
    ax_l.legend(loc="lower left", framealpha=0.95, edgecolor="#bbbbbb")

    # ── Right panel: exact-match rate (green) + ASR (red) ───────────────
    for _key, pa, name, ls, mk in methods:
        em_m, em_lo, em_hi = mean_std_arrays(pa, raw_alphas, "exact_match",
                                              "exact_match_std")
        asr_m, asr_lo, asr_hi = mean_std_arrays(pa, raw_alphas, "asr", "asr_std")
        draw(ax_r, disp_alphas, em_m,  em_lo,  em_hi,  GREEN, ls, mk,
             f"{name}  exact-match")
        draw(ax_r, disp_alphas, asr_m, asr_lo, asr_hi, RED,   ls, mk,
             f"{name}  ASR")

    ax_r.set_ylabel("exact-match rate / ASR")
    ax_r.set_xlabel(r"steering strength  $\alpha$")
    ax_r.set_ylim(-0.05, 1.15)
    ax_r.set_xticks(disp_alphas)
    ax_r.set_xticklabels([f"{a:.4g}" for a in disp_alphas])
    plt.setp(ax_r.get_xticklabels(), rotation=45, ha="right")
    ax_r.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax_r.grid(True, axis="y", color="#eeeeee", lw=0.6, zorder=0)
    ax_r.set_axisbelow(True)
    ax_r.legend(loc="center left", framealpha=0.95, edgecolor="#bbbbbb",
                fontsize=10)

    fig.suptitle(
        f"Seed {args.seed} alpha sweep  ·  "
        f"mean over 1000 (prompt × sample-seed) trials, band = $\\pm$ 1 std",
        fontsize=14, y=1.00,
    )
    fig.tight_layout()

    out = Path(args.output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    base_str = str(out).removesuffix('.png').removesuffix('.pdf')
    base_path = f"{base_str}{args.seed}"
    fig.savefig(base_path + ".png", dpi=200)
    fig.savefig(base_path + ".pdf")
    plt.close(fig)
    print(f"wrote {base_path}.png and {base_path}.pdf")


if __name__ == "__main__":
    main()
