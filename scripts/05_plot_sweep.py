"""The headline figure: FRA recovery vs feature overlap, with the admission
criterion drawn on the same axes.

runner_up_ratio is the primary y-axis, not mass fraction: mass fraction divides
by an L1 over every co-active pair, and raising rho moves that denominator for
reasons unrelated to the planted edge.

Gate 2 (argmax_is_key) is overlaid so a reader can see, in the figure itself,
whether a declining recovery curve is about FRA or about the model failing to
learn the circuit in the first place.

Run: python scripts/05_plot_sweep.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = Path("results/sweep_rho.json")
OUT = Path("results/figures/rho_sweep.png")


def main() -> None:
    rows = json.loads(SRC.read_text())
    rho = [r["rho_realized"] for r in rows]
    admitted = [r["admitted"] for r in rows]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.5, 6.4))

    # ── Primary: runner-up ratio (denominator-free) ─────────────────────
    ax.plot(rho, [r["agg_runner_up"] for r in rows], "o-", color="#1f77b4",
            lw=2, ms=7, label="runner-up ratio, aggregate")
    ax.plot(rho, [r["circuit_runner_up"] for r in rows], "s--", color="#4c9fd4",
            lw=1.6, ms=6, label="runner-up ratio, circuit only $|G|$")
    ax.plot(rho, [r["cell_runner_up_mean"] for r in rows], "^:", color="#9ecae1",
            lw=1.6, ms=6, label="runner-up ratio, cell (mean)")
    ax.axhline(1.0, color="#888", lw=1.2, ls=":")
    ax.text(0.012, 1.015, "parity: planted edge tied with its best competitor",
            ha="left", va="bottom", fontsize=8.5, color="#555")

    ax.set_xlabel(r"feature overlap  $\rho$   (mean pairwise $|\cos|$)")
    ax.set_ylabel("runner-up ratio   |planted| / |next largest|   (log)")
    ax.set_yscale("log")
    ax.set_ylim(0.9, 12)

    # ── Secondary: mass fraction + the admission criterion ──────────────
    # Log-scaled too: on a linear 0-1 axis the mass fraction (0.014-0.094)
    # collapses onto zero and its decline is invisible.
    ax2 = ax.twinx()
    ax2.plot(rho, [r["agg_mass"] for r in rows], "o-", color="#d62728",
             lw=1.8, ms=6, label="mass fraction, aggregate")
    ax2.plot(rho, [r["gate2_argmax_is_key"] for r in rows], "D-", color="#2ca02c",
             lw=2.2, ms=7, label="Gate 2: argmax_is_key  (ADMISSION)")
    ax2.plot(rho, [r["heldout_query_acc"] for r in rows], "v--", color="#8fce8f",
             lw=1.5, ms=6, label="held-out query accuracy")
    ax2.set_ylabel("mass fraction  /  Gate 2, accuracy   (log)")
    ax2.set_yscale("log")
    ax2.set_ylim(0.009, 1.45)

    ax2.text(0.41, 1.06, "admission criterion flat at 1.00 -- the circuit is intact "
                         "at every $\\rho$", ha="center", va="bottom", fontsize=8.5,
             color="#1a7a1a")

    # Mark any rho point that failed admission.
    for x, ok in zip(rho, admitted):
        if not ok:
            ax.axvspan(x - 0.015, x + 0.015, color="#d62728", alpha=0.12, zorder=0)
    if not all(admitted):
        ax.scatter([], [], marker="s", s=80, color="#d62728", alpha=0.25,
                   label="failed Gate 2 (not evidence about FRA)")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.13),
              ncol=3, fontsize=8.5, framealpha=0.95)

    ax.set_title("FRA recovery of a planted QK edge degrades with feature overlap,\n"
                 "while the underlying circuit stays perfect\n"
                 "Stage A (oracle features, no SAE), 1L/1H attention-only, single seed",
                 fontsize=11.5)
    ax.grid(alpha=0.22, which="both")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT, dpi=160)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
