#!/usr/bin/env python3
"""Phase 3 summary figure: 1×2 alignment-vs-coherence frontier comparison.

Left panel:  Nura's 3 FRA-decomposition methods (QK→QK, QK→OV, OV→OV)
             at `blocks.24.ln1.hook_normalized` with Nura's SAE.
Right panel: conventional additive steering `act += (α-1)·f_λ·W_dec_λ`
             at the same hookpoint with the same SAE.

Same SAE, same hookpoint, same prompts, same eval seed → only the
intervention recipe differs. Single eval seed (default 42) for clarity.

Usage:
    python scripts/plot_summary_fra_vs_additive.py \
        --nura-per-seed-dir <dir>      # has aggregated_seed{seed}_medical.json
        --nura-additive <gpt4o_aggregated.json for additive at L24 ln1, this seed>
        --seed 42
        --out <basename>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


COH_FLOOR = 70.0


def load_method(path, method=None):
    d = json.loads(Path(path).read_text())
    if "aggregated" in d and isinstance(d["aggregated"], dict):
        d = d["aggregated"]
    if method:
        return sorted(d.get(method, []), key=lambda r: r["scale"])
    rows = []
    for m, r in d.items():
        if isinstance(r, list):
            rows.extend(r)
    return sorted(rows, key=lambda r: r["scale"])


def stats(rows, floor=COH_FLOOR):
    al = np.array([r["mean_alignment"] for r in rows], dtype=float)
    co = np.array([r["mean_coherence"] for r in rows], dtype=float)
    mask = co >= floor
    if mask.any():
        delta = float(al[mask].max() - al[mask].min())
        peak_at_floor = float(al[mask].max())
        n70 = int(mask.sum())
    else:
        delta = float("nan"); peak_at_floor = float("nan"); n70 = 0
    return delta, peak_at_floor, n70, float(np.nanmax(al))


def draw_curve(ax, rows, color, label):
    if not rows:
        return None
    scales = np.array([r["scale"] for r in rows])
    al = np.array([r["mean_alignment"] for r in rows])
    co = np.array([r["mean_coherence"] for r in rows])
    ax.plot(co, al, color=color, lw=1.6, alpha=0.85, zorder=2)
    h = ax.scatter(co, al, c=color, s=70, edgecolors="black",
                   linewidths=0.5, zorder=3, label=label)
    for sc, x, y in zip(scales, co, al):
        ax.annotate(f"α={sc}", (x, y), xytext=(5, 5),
                    textcoords="offset points", fontsize=8, color="#333")
    return h


def decorate(ax, title):
    ax.axvline(COH_FLOOR, color="grey", lw=0.8, ls=":", zorder=1)
    ax.axhline(50, color="grey", lw=0.5, ls=":", zorder=1)
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    ax.set_xlabel("coherence (GPT-4o, 0–100)")
    ax.set_ylabel("alignment (GPT-4o, 0–100)")
    ax.set_title(title, fontsize=11)
    ax.grid(True, ls=":", alpha=0.3)


def stat_box(ax, lines):
    ax.text(0.02, 0.98, "\n".join(lines),
            transform=ax.transAxes, fontsize=8.5, verticalalignment="top",
            family="DejaVu Sans",
            bbox=dict(facecolor="white", edgecolor="#888",
                      alpha=0.93, pad=4, boxstyle="round,pad=0.4"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nura-per-seed-dir", required=True,
                   help="dir with aggregated_seed{seed}_medical.json")
    p.add_argument("--nura-additive", required=True,
                   help="gpt4o_aggregated_seed{seed}_*.json from "
                        "Nura SAE under our additive recipe at L24 ln1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True, help="output base path (no extension)")
    p.add_argument("--em-model", default="medical")
    args = p.parse_args()

    nura_path = Path(args.nura_per_seed_dir) / f"aggregated_seed{args.seed}_{args.em_model}.json"
    if not nura_path.exists():
        raise SystemExit(f"missing {nura_path}")
    additive_path = Path(args.nura_additive)
    if not additive_path.exists():
        raise SystemExit(f"missing {additive_path}")

    qkqk = load_method(nura_path, "qk_to_qk")
    qkov = load_method(nura_path, "qk_to_ov")
    ovov = load_method(nura_path, "ov_to_ov")
    baseline = load_method(nura_path, "baseline")
    additive = load_method(additive_path)  # has just one method "sae_resid"

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6),
                             sharex=True, sharey=True)

    # ── Left panel: 3 FRA methods ───────────────────────────────────────
    axL = axes[0]
    fra_styles = [
        ("QK→QK", qkqk, "#1a9850"),
        ("OV→OV", ovov, "#4575b4"),
        ("QK→OV", qkov, "#d73027"),
    ]
    fra_lines = []
    for label, rows, color in fra_styles:
        draw_curve(axL, rows, color, label)
        d, pf, n70, peak = stats(rows)
        if n70 > 0:
            fra_lines.append((f"{label:6s}  Δ={d:5.2f}  peak={pf:5.2f}  ({n70}/{len(rows)})",
                              d if d == d else -1))
        else:
            fra_lines.append((f"{label:6s}  Δ=NaN", -1))
    # baseline (no hook)
    if baseline:
        b = baseline[0]
        axL.scatter([b["mean_coherence"]], [b["mean_alignment"]], marker="*", s=320,
                    color="black", edgecolors="white", linewidths=1.2, zorder=5,
                    label="baseline (no hook)")
    decorate(axL, "FRA decomposition recipes\n@ blocks.24.ln1.hook_normalized")
    # Bold the winning Δ in the stat box
    if fra_lines:
        winner = max(range(len(fra_lines)), key=lambda i: fra_lines[i][1])
        ax_lines = []
        ax_lines.append("alignment delta @ coh ≥ 70:")
        for i, (line, _) in enumerate(fra_lines):
            ax_lines.append(("→ " if i == winner else "  ") + line)
        stat_box(axL, ax_lines)
    axL.legend(loc="lower right", fontsize=9)

    # ── Right panel: conventional additive ──────────────────────────────
    axR = axes[1]
    draw_curve(axR, additive, "#9467bd", "additive  act += (α−1)·f·W_dec")
    if additive:
        i_one = int(np.argmin(np.abs(np.array([r["scale"] for r in additive]) - 1.0)))
        axR.scatter([additive[i_one]["mean_coherence"]],
                    [additive[i_one]["mean_alignment"]],
                    marker="*", s=320, color="black", edgecolors="white",
                    linewidths=1.2, zorder=5, label="α=1.0 (no-op)")
    d, pf, n70, peak = stats(additive)
    decorate(axR, "Conventional additive feature steering\n@ blocks.24.ln1.hook_normalized (same SAE)")
    lines = [
        "alignment delta @ coh ≥ 70:",
        ("→ " + (f"Δ={d:5.2f}  peak={pf:5.2f}  ({n70}/{len(additive)})"
                  if n70 > 0 else "Δ=NaN")),
    ]
    stat_box(axR, lines)
    axR.legend(loc="lower right", fontsize=9)

    fig.suptitle(
        f"Same SAE, same hookpoint, same prompts, eval seed = {args.seed} — "
        f"medical EM (Qwen2.5-14B + medical LoRA)",
        fontsize=12, y=1.0,
    )
    fig.tight_layout()

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out) + ".png", dpi=180, bbox_inches="tight")
    fig.savefig(str(out) + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"plot → {out}.png / .pdf")


if __name__ == "__main__":
    sys.exit(main())
