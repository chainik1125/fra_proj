"""Unified-α-grid plot for the 51-point re-sweep on Qwen-2.5-7B + bad-medical.

Five methods (DoM, conv-SAE additive, QK→QK, QK→OV, OV→OV) on a single
signed α grid spanning [-20, 20]. Produces:

  * One Δalign-vs-α figure with all 5 methods + per-seed bands.
  * One coherence-vs-α figure (same axes) for the coh floor check.
  * Optional per-seed inset figures.

The trajectory shape per method (across α) is now visible end-to-end,
unlike the earlier 6-point plot which compressed everything into the
positive-α regime.

Inputs are the standard combined JSONs produced by
phase1_judge_and_combine.py:

  --combined-root  dir with:
    gpt4o_combined_dom_qwen7b_extract_em_apply_medical_medical.json
    gpt4o_combined_L15_ln1_arditi_qwen7b_medical.json
    gpt4o_combined_L15_ln1_arditi_qwen7b_FRA_medical.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


METHODS = [
    # (combined_file_key, method_key_inside_combined, label, color)
    ("dom",  "dom_L15",   "DoM (Soligo) @ L15",        "#5E2E8A"),
    ("sae",  "sae_resid", "Conv. SAE additive @ L15",  "#000000"),
    ("fra",  "qk_to_qk",  r"QK$\rightarrow$QK (FRA)",  "#009E73"),
    ("fra",  "qk_to_ov",  r"QK$\rightarrow$OV",        "#D55E00"),
    ("fra",  "ov_to_ov",  r"OV$\rightarrow$OV",        "#0072B2"),
]

COMBINED_FILES = {
    "dom": "gpt4o_combined_dom_qwen7b_extract_em_apply_medical_medical.json",
    "sae": "gpt4o_combined_L15_ln1_arditi_qwen7b_medical.json",
    "fra": "gpt4o_combined_L15_ln1_arditi_qwen7b_FRA_medical.json",
}


def setup_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 12,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 1.0,
    })


def load_combined(combined_root: Path):
    out = {}
    for key, fname in COMBINED_FILES.items():
        path = combined_root / fname
        if not path.exists():
            print(f"WARN: {path} not found")
            continue
        out[key] = json.loads(path.read_text())
    return out


def block_to_arrays(block):
    """Return (scales, align_per_seed, coh_per_seed) as np arrays.

    align/coh shapes: (n_scales, n_seeds), with NaN for missing entries.
    """
    by_alpha = block["by_alpha"]
    scales = np.array([e["scale"] for e in by_alpha], dtype=float)
    order = np.argsort(scales)
    scales = scales[order]
    by_alpha = [by_alpha[i] for i in order]
    n_scales = len(scales)
    n_seeds = max(len(e["per_seed_alignment"]) for e in by_alpha)
    al = np.full((n_scales, n_seeds), np.nan)
    co = np.full((n_scales, n_seeds), np.nan)
    for i, e in enumerate(by_alpha):
        for s in range(min(n_seeds, len(e["per_seed_alignment"]))):
            a = e["per_seed_alignment"][s]
            c = e["per_seed_coherence"][s]
            al[i, s] = a if a is not None else np.nan
            co[i, s] = c if c is not None else np.nan
    return scales, al, co


def plot_seed_averaged(combined, out_basename):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax_al, ax_co = axes

    for file_key, method_key, label, color in METHODS:
        d = combined.get(file_key)
        if d is None: continue
        block = d.get(method_key)
        if block is None: continue
        scales, al, co = block_to_arrays(block)
        mean_al = np.nanmean(al, axis=1)
        std_al  = np.nanstd(al, axis=1, ddof=1)
        mean_co = np.nanmean(co, axis=1)
        std_co  = np.nanstd(co, axis=1, ddof=1)

        ax_al.plot(scales, mean_al, "-", color=color, lw=1.8, label=label)
        ax_al.fill_between(scales, mean_al - std_al, mean_al + std_al,
                           color=color, alpha=0.18, lw=0)
        ax_co.plot(scales, mean_co, "-", color=color, lw=1.8)
        ax_co.fill_between(scales, mean_co - std_co, mean_co + std_co,
                           color=color, alpha=0.18, lw=0)

    # Baseline markers — conv-SAE additive + FRA recipes baseline at α=1.0,
    # DoM baseline at α=0.0. Mark both axes with vertical guides.
    for x, ls, lbl in [(0.0, ":", "α=0"), (1.0, "--", "α=1 (no-op)")]:
        ax_al.axvline(x, color="#bbbbbb", lw=0.8, ls=ls)
        ax_co.axvline(x, color="#bbbbbb", lw=0.8, ls=ls)

    ax_al.axhline(50, color="#dddddd", lw=0.6, zorder=0)
    ax_co.axhline(70, color="#dddddd", lw=0.6, zorder=0)
    ax_co.axhline(50, color="#eeeeee", lw=0.6, zorder=0)

    ax_al.set_ylabel("Alignment (GPT-4o, 0–100)")
    ax_co.set_ylabel("Coherence (GPT-4o, 0–100)")
    ax_co.set_xlabel("α (signed steering scale)")
    ax_al.set_title("Qwen-2.5-7B + bad-medical · 5-method re-sweep · seed-averaged")
    ax_al.legend(loc="lower left", frameon=False, fontsize=10, ncol=2)
    ax_al.set_ylim(0, 105)
    ax_co.set_ylim(0, 105)
    ax_co.set_xlim(-21, 21)

    fig.tight_layout()
    out = Path(out_basename + ".png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    fig.savefig(Path(out_basename + ".pdf"), bbox_inches="tight")
    print(f"saved → {out}")


def plot_per_seed(combined, out_basename):
    """3 columns (one per seed) × 2 rows (align, coh). All 5 methods overlaid."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey="row")
    seeds = [42, 123, 456]
    for col, seed_idx in enumerate([0, 1, 2]):
        ax_al, ax_co = axes[0, col], axes[1, col]
        for file_key, method_key, label, color in METHODS:
            d = combined.get(file_key)
            if d is None: continue
            block = d.get(method_key)
            if block is None: continue
            scales, al, co = block_to_arrays(block)
            if seed_idx >= al.shape[1]: continue
            ax_al.plot(scales, al[:, seed_idx], "-", color=color, lw=1.4,
                       label=label if col == 0 else None)
            ax_co.plot(scales, co[:, seed_idx], "-", color=color, lw=1.4)

        for x, ls in [(0.0, ":"), (1.0, "--")]:
            ax_al.axvline(x, color="#bbbbbb", lw=0.7, ls=ls)
            ax_co.axvline(x, color="#bbbbbb", lw=0.7, ls=ls)
        ax_co.axhline(70, color="#dddddd", lw=0.6, zorder=0)
        ax_co.axhline(50, color="#eeeeee", lw=0.6, zorder=0)

        ax_al.set_title(f"seed = {seeds[col]}")
        if col == 0:
            ax_al.set_ylabel("Alignment")
            ax_co.set_ylabel("Coherence")
            ax_al.legend(loc="lower left", frameon=False, fontsize=9)
        ax_co.set_xlabel("α")
        ax_al.set_ylim(0, 105); ax_co.set_ylim(0, 105)
        ax_al.set_xlim(-21, 21); ax_co.set_xlim(-21, 21)

    fig.suptitle("Qwen-2.5-7B + bad-medical · 5-method re-sweep · per-seed",
                 fontsize=14, y=1.00)
    fig.tight_layout()
    out_png = Path(out_basename + ".png")
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    fig.savefig(Path(out_basename + ".pdf"), bbox_inches="tight")
    print(f"saved → {out_png}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--combined-root", required=True)
    p.add_argument("--out-prefix", required=True,
                   help="Saves <prefix>_seedavg.{png,pdf} and <prefix>_perseed.{png,pdf}")
    args = p.parse_args()
    setup_style()

    combined = load_combined(Path(args.combined_root))
    plot_seed_averaged(combined, args.out_prefix + "_seedavg")
    plot_per_seed(combined, args.out_prefix + "_perseed")


if __name__ == "__main__":
    main()
