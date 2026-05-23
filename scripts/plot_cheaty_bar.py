"""Bar chart: per-seed held-out JSD(steered, clean) on the ov×ov diff cell.

Compares three quantities, per SAE seed (6 seeds):

  1. Attribution top-1 OV feature   — the principled choice (rank=1 by diff score)
  2. "Cheaty" best lower-ranked OV feature, picked by minimum eval JSD subject
     to ASR≤top-1's ASR (i.e. we cherry-pick the best of the top-20 using the
     held-out metric itself — not a fair selection rule, hence "cheaty")
  3. Conventional downstream-SAE-feature steering (from jsd_alpha_sweep_6seeds)

α=4.0 for all three. Seed 2 is annotated as "bad downstream SAE" (its conv
JSD ≈ 0.9, hugely distorting the mean) and the mean line excludes it.

Inputs:
  results/matrix_per_feat_diff_ovxov_topk20.json
  results/jsd_alpha_sweep_6seeds.json

Output:
  figures/cheaty_bar_jsd.{pdf,png}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


PER_FEAT = Path("results/matrix_per_feat_diff_ovxov_topk20.json")
CONV     = Path("results/jsd_alpha_sweep_6seeds.json")
OUT      = Path("figures/cheaty_bar_jsd")
ALPHA    = 4.0
BAD_SEED = 2

COLOR_TOP1   = "#999999"
COLOR_CHEAT  = "#1b7837"  # green — the "improved" pick
COLOR_CONV   = "#762a83"  # purple — conventional baseline
EDGE         = "#222222"


def setup_style() -> None:
    mpl.rcParams.update({
        "font.family":       "sans-serif",
        "font.sans-serif":   ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size":         13,
        "axes.titlesize":    15,
        "axes.labelsize":    14,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.linewidth":    1.0,
        "axes.edgecolor":    "#222222",
    })


def load_per_feat() -> dict[int, dict]:
    """For each seed, return {top1: row, cheat: row} at α=ALPHA.

    top1 = rank-1 row at α (regardless of ASR).
    cheat = row with min eval_jsd_clean among rows whose eval_asr is the
            minimum eval_asr seen across all 20 tuples for that seed.
    """
    d = json.loads(PER_FEAT.read_text())
    by_seed: dict[int, list[dict]] = {}
    for r in d["rows"]:
        if r["alpha"] != ALPHA:
            continue
        by_seed.setdefault(r["seed"], []).append(r)
    out = {}
    for seed, rows in by_seed.items():
        top1 = next(r for r in rows if r["attr_rank"] == 1)
        min_asr = min(r["eval_asr"] for r in rows)
        zero_pool = [r for r in rows if r["eval_asr"] == min_asr]
        cheat = min(zero_pool, key=lambda r: r["eval_jsd_clean"])
        out[seed] = {"top1": top1, "cheat": cheat}
    return out


def load_conv() -> dict[int, dict]:
    """Conventional downstream-SAE per-seed JSD/ASR at α=ALPHA."""
    d = json.loads(CONV.read_text())
    bucket = d["configs"]["conventional"]["per_alpha"][f"{ALPHA:.1f}"]
    feats = d["configs"]["conventional"]["per_seed_feature"]
    out = {}
    for s, (jsd, asr) in enumerate(zip(bucket["jsd_clean"], bucket["asr"])):
        out[s] = {"jsd": jsd, "asr": asr, "feat": int(feats[str(s)])}
    return out


def main() -> None:
    setup_style()
    per_feat = load_per_feat()
    conv     = load_conv()
    seeds    = sorted(per_feat.keys())

    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    x = np.arange(len(seeds), dtype=float)
    w = 0.27

    top1_jsd  = [per_feat[s]["top1"]["eval_jsd_clean"] for s in seeds]
    cheat_jsd = [per_feat[s]["cheat"]["eval_jsd_clean"] for s in seeds]
    conv_jsd  = [conv[s]["jsd"] for s in seeds]

    top1_feat  = [per_feat[s]["top1"]["tuple"][0][0] for s in seeds]
    cheat_feat = [per_feat[s]["cheat"]["tuple"][0][0] for s in seeds]
    cheat_rank = [per_feat[s]["cheat"]["attr_rank"] for s in seeds]

    b1 = ax.bar(x - w, top1_jsd,  w, color=COLOR_TOP1,  edgecolor=EDGE,
                label="OV top-1 (rank=1 attribution)")
    b2 = ax.bar(x,     cheat_jsd, w, color=COLOR_CHEAT, edgecolor=EDGE,
                label="OV cheaty best (min JSD in top-20)")
    b3 = ax.bar(x + w, conv_jsd,  w, color=COLOR_CONV,  edgecolor=EDGE,
                label="Conventional downstream-SAE")

    for bars, vals, feats, skip_if_same in (
        (b1, top1_jsd, top1_feat, None),
        (b2, cheat_jsd, cheat_feat, top1_feat),  # skip cheaty label if same as top-1
        (b3, conv_jsd, [conv[s]["feat"] for s in seeds], None),
    ):
        for i, (bar, v, f) in enumerate(zip(bars, vals, feats)):
            if skip_if_same is not None and f == skip_if_same[i]:
                continue
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012,
                    f"f={f}", ha="center", va="bottom", fontsize=9, color="#444444")

    for i, cr in enumerate(cheat_rank):
        if cr != 1:
            ax.text(x[i], cheat_jsd[i] / 2, f"r={cr}", ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold")

    ax.axvspan(BAD_SEED - 0.5, BAD_SEED + 0.5, color="#fde0e0", alpha=0.6, zorder=0)
    ax.text(BAD_SEED, ax.get_ylim()[1] * 0.96, "bad downstream SAE",
            ha="center", va="top", fontsize=10, color="#a13434")

    good = [s for s in seeds if s != BAD_SEED]
    mean_top1  = float(np.mean([per_feat[s]["top1"]["eval_jsd_clean"]  for s in good]))
    mean_cheat = float(np.mean([per_feat[s]["cheat"]["eval_jsd_clean"] for s in good]))
    mean_conv  = float(np.mean([conv[s]["jsd"]                          for s in good]))

    # Mean-values annotation block in the upper-right (away from data and legend).
    lines = [
        (f"mean (excl seed {BAD_SEED}, n=5):", "#222222"),
        (f"  OV top-1:   {mean_top1:.3f}",  COLOR_TOP1),
        (f"  OV cheaty:  {mean_cheat:.3f}", COLOR_CHEAT),
        (f"  conv:       {mean_conv:.3f}",  COLOR_CONV),
    ]
    for i, (txt, color) in enumerate(lines):
        ax.text(0.99, 0.96 - 0.05 * i, txt, color=color, ha="right", va="top",
                fontsize=11, family="monospace", transform=ax.transAxes,
                fontweight="bold" if i == 0 else "normal")

    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylabel("JSD(steered, clean)  — lower is better")
    ax.set_title(f"Held-out JSD on ov×ov diff cell, α={ALPHA}\n"
                 "OV top-1 vs cheaty best-of-top-20 vs conventional downstream-SAE  "
                 "(dashed lines = mean over 5 good seeds)")
    ax.set_ylim(0, max(max(top1_jsd), max(cheat_jsd), max(conv_jsd)) * 1.18)
    ax.grid(axis="y", color="#dddddd", lw=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, fontsize=11)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"))
    fig.savefig(OUT.with_suffix(".png"), dpi=180)
    print(f"wrote {OUT.with_suffix('.pdf')}  and  {OUT.with_suffix('.png')}")
    print(f"means (excl seed {BAD_SEED}):  top1={mean_top1:.4f}  "
          f"cheat={mean_cheat:.4f}  conv={mean_conv:.4f}")


if __name__ == "__main__":
    main()
