"""Per-seed Fig-10 analogue for QK+OV oracle feature triples vs OV winner.

Top row: JSD_clean (green) and JSD_pois (red) vs α.
Bottom row: exact-match rate to clean rollout (green) and ASR (red) vs α.

QK+OV oracle (solid, circle)   = at each (seed, α), pick the top-20 tuple with
                                  the lowest (ASR, JSDc).
OV winner   (dashed, triangle) = the channel_pick.py per-seed OV winner,
                                  swept across α.

Bands: per-α min–max envelope over the 5 decoding seeds (the
`*_per_seed` arrays in the lockstep-eval output).
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


def stats(per_seed: list[float], scale: float = 1.0) -> tuple[float, float, float]:
    arr = np.asarray(per_seed, dtype=float) * scale
    return float(arr.mean()), float(arr.min()), float(arr.max())


def per_alpha_curves(ev_per_alpha: dict, alphas: list[float], n_prompts: int):
    """Returns dict of {metric: (mean, lo, hi) arrays of len(alphas)}."""
    out = {m: ([], [], []) for m in ("jsd_clean", "jsd_pois", "exact_match", "asr")}
    em_total_key = "n_exact_match_clean_per_seed"
    for a in alphas:
        ev = ev_per_alpha[f"{a}"]
        for m in ("jsd_clean", "jsd_pois", "asr"):
            mn, lo, hi = stats(ev[f"{m}_per_seed"])
            out[m][0].append(mn); out[m][1].append(lo); out[m][2].append(hi)
        # exact match: per-seed counts / per-seed B (assumed equal B per seed = n_prompts)
        em_counts = np.asarray(ev[em_total_key], dtype=float)
        em_rate = em_counts / n_prompts
        out["exact_match"][0].append(float(em_rate.mean()))
        out["exact_match"][1].append(float(em_rate.min()))
        out["exact_match"][2].append(float(em_rate.max()))
    return {m: (np.array(v[0]), np.array(v[1]), np.array(v[2])) for m, v in out.items()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ov_json",   type=Path,
                   default=Path("results/full_ov.json"))
    p.add_argument("--qkov_json", type=Path,
                   default=Path("results/qkov_topk20_gated_results.json"))
    p.add_argument("--output",    type=Path,
                   default=Path("figures/jsd_exact_qkov_oracle_perseed"))
    args = p.parse_args()

    setup_style()

    ov   = json.loads(args.ov_json.read_text())
    qkov = json.loads(args.qkov_json.read_text())

    qkov_rows = qkov["results"]
    alphas    = sorted({float(a) for r in qkov_rows for a in r["alpha_sweep"]})
    seeds     = sorted({r["seed"] for r in qkov_rows})
    n_prompts = int(qkov_rows[0]["alpha_sweep"][f"{alphas[0]}"]["exact_match_total_rows"])
    n_prompts_per_seed = n_prompts // 5  # 5 eval sampling seeds; total_rows = B × 5

    disp_alphas = [-a for a in alphas]   # fig-10 convention: ablation strength shown negative

    # Index OV winners and QK+OV per-seed tuples
    ov_winner = {r["seed"]: r["winner"] for r in ov["results"]}
    qkov_by_seed = {s: [r for r in qkov_rows if r["seed"] == s] for s in seeds}

    # For QK+OV oracle, pick per (seed, α) the best tuple by (asr, jsd_clean).
    def qkov_oracle_eval(seed: int, alpha: float) -> dict:
        rows = qkov_by_seed[seed]
        evs = [(r["alpha_sweep"][f"{alpha}"], r["tuple"]) for r in rows]
        ev, _ = min(evs, key=lambda x: (x[0]["asr"], x[0]["jsd_clean"]))
        return ev

    def qkov_oracle_tuple(seed: int) -> list:
        """Return the most-frequent oracle tuple across alphas for the label."""
        from collections import Counter
        tups = Counter()
        for a in alphas:
            rows = qkov_by_seed[seed]
            evs = [(r["alpha_sweep"][f"{a}"], tuple(tuple(t) for t in r["tuple"]))
                   for r in rows]
            _, t = min(evs, key=lambda x: (x[0]["asr"], x[0]["jsd_clean"]))
            tups[t] += 1
        most, _ = tups.most_common(1)[0]
        return [list(t) for t in most]

    fig, axes = plt.subplots(
        2, len(seeds), figsize=(2.6 * len(seeds), 5.6),
        sharex=True, sharey="row",
    )

    for col, seed in enumerate(seeds):
        ax_jsd, ax_em = axes[0, col], axes[1, col]

        # OV winner per-α curves
        ov_curves = per_alpha_curves(ov_winner[seed]["eval_sweep"], alphas, n_prompts_per_seed)

        # QK+OV oracle: build a synthetic per_alpha dict by selecting best tuple at each α
        qkov_oracle_pa = {f"{a}": qkov_oracle_eval(seed, a) for a in alphas}
        qkov_curves = per_alpha_curves(qkov_oracle_pa, alphas, n_prompts_per_seed)

        # — JSD curves —
        for curves, label_prefix, ls, mk, lw in [
            (qkov_curves, "QK+OV oracle",  "-",  "o", 1.3),
            (ov_curves,   "OV winner",     "--", "^", 1.1),
        ]:
            for metric, color, lbl in (
                ("jsd_clean", GREEN, "JSD$_\\mathrm{clean}$"),
                ("jsd_pois",  RED,   "JSD$_\\mathrm{pois}$"),
            ):
                m, lo, hi = curves[metric]
                draw(ax_jsd, disp_alphas, m, lo, hi, color, ls, mk, lw,
                     f"{label_prefix}  {lbl}")
        ax_jsd.axhline(1.0, color="#999", lw=0.5, ls=":")
        ax_jsd.set_ylim(-0.02, 1.05)

        ov_feat   = ov_winner[seed]["feature"]
        qk_tup    = qkov_oracle_tuple(seed)
        qk_label  = ",".join(str(t[0]) for t in qk_tup)
        ax_jsd.set_title(
            f"seed {seed}\nOV f={ov_feat} · QK+OV ({qk_label})",
            fontsize=9.5,
        )

        # — exact-match (green) + ASR (red) —
        for curves, label_prefix, ls, mk, lw in [
            (qkov_curves, "QK+OV oracle",  "-",  "o", 1.3),
            (ov_curves,   "OV winner",     "--", "^", 1.1),
        ]:
            for metric, color, lbl in (
                ("exact_match", GREEN, "exact-match"),
                ("asr",         RED,   "ASR"),
            ):
                m, lo, hi = curves[metric]
                draw(ax_em, disp_alphas, m, lo, hi, color, ls, mk, lw,
                     f"{label_prefix}  {lbl}")
        ax_em.set_ylim(-0.02, 1.05)
        ax_em.yaxis.set_major_formatter(PercentFormatter(1.0))

        if col == 0:
            ax_jsd.set_ylabel("JSD (bits)")
            ax_em .set_ylabel("Exact-match rate / ASR")
        ax_jsd.grid(axis="y", color="#dddddd", lw=0.5)
        ax_em .grid(axis="y", color="#dddddd", lw=0.5)
        ax_jsd.set_axisbelow(True)
        ax_em .set_axisbelow(True)

    axes[0, 0].legend(loc="lower left", fontsize=7.5, framealpha=0.92,
                      handlelength=1.6, borderpad=0.4)
    axes[1, 0].legend(loc="lower left", fontsize=7.5, framealpha=0.92,
                      handlelength=1.6, borderpad=0.4,
                      bbox_to_anchor=(0.0, 0.30))
    fig.suptitle("Per-seed QK+OV oracle vs OV winner", fontsize=15, y=1.02)
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
