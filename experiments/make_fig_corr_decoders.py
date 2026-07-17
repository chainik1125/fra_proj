"""Standalone figure for the §4.5 rate-controlled decoder sweep (exp_c2_corr_decoders.py).
Separate file to avoid churning the shared make_figures_active.py."""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.environ.get("BAG_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.environ.get("BAG_OUT", f"{ROOT}/results")
FIG = os.environ.get("BAG_FIG", f"{ROOT}/figures")
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})


def main():
    d = torch.load(f"{RES}/c2_corr_decoders.pt", map_location="cpu", weights_only=False)
    rows = d["rows"]; rho = [r["rho"] for r in rows]
    je = [r["joint_err"] for r in rows]; ie = [r["indep_err"] for r in rows]
    phys = [r["phys"] for r in rows]; gap = [r["gap"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))

    # (a) decoder errors vs correlation, at FIXED per-chain rate
    ax[0].plot(rho, ie, "s-", color="C3", label="naive independent decoder")
    ax[0].plot(rho, je, "o-", color="C2", label="exact joint decoder (optimal)")
    ax[0].plot(rho, phys, "x--", color="C7", label="single chain (physical)")
    ax[0].set_xlabel("induced cross-chain correlation ρ")
    ax[0].set_ylabel("logical decode error")
    ax[0].set_title(f"(a) At FIXED per-chain rate ({d['target_rate']:.2f}):\n"
                    "naive decoder crosses ABOVE a single chain")
    ax[0].legend(fontsize=8)
    # shade where majority-voting (naive) is worse than one chain
    above = [r for r, i, p in zip(rho, ie, phys) if i > p]
    if above:
        ax[0].axvspan(min(above) - 0.02, max(rho), color="C3", alpha=0.06)

    # (b) the optimality gap (independent − joint) grows monotonically with correlation
    ax[1].plot(rho, gap, "D-", color="C4")
    for x, y in zip(rho, gap):
        ax[1].annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 6), fontsize=8, ha="center")
    ax[1].set_xlabel("induced cross-chain correlation ρ")
    ax[1].set_ylabel("independent − joint logical error")
    ax[1].set_title("(b) The cost of (wrongly) assuming\nindependence grows with correlation")
    ax[1].set_ylim(bottom=-0.01)

    fig.suptitle("§4.5 (rate-controlled): when block errors are correlated, the binomial-tail "
                 "(independence) accounting is wrong — and majority-voting can hurt", fontsize=11, y=1.02)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_c2_corr_decoders.png"); plt.close(fig)
    print("wrote fig_c2_corr_decoders.png")


if __name__ == "__main__":
    main()
