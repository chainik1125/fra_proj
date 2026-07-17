"""Standalone figure for the GHMM coding-mechanism control (results/ghmm_coding.pt).

Two regimes at the SAME generative coupling (corr(q,sharpness)≈−0.89): DIR (alignment also coded by
the content-cycle DIRECTION, a code the process makes separable from the within-step capability belief)
vs NODIR (alignment coded by emission SHARPNESS only — the same feature that defines the capability
belief). The decisive, seed-robust contrasts are (i) whether the transformer even forms a SEPARABLE
alignment coordinate (z R² = 0.93 vs 0.00) and (ii) whether the learned alignment direction is the
capability-SPARING one (align cross-talk ≈0, far below the random-⊥ null) or as destructive as random.
So causal cross-talk is governed by the CODING MECHANISM, not the statistical coupling.

Honest note baked into panel (c): the clipped-R² cross-talk metric SATURATES (any large fixed-norm
steer breaks a linear probe off-manifold), so cross-talk is NOT a graded function of the probe
footprint (footprint↔cap-drop corr only ≈0.1–0.4 across directions). We therefore do NOT claim
'cross-talk is monotonic in footprint'; we show WHERE each regime's learned d_align lands relative to
the random/interpolated cloud — DIR at the capability-sparing corner, NODIR among the destructive ones.
Standalone to avoid collision with the concurrent session."""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.environ.get("BAG_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.environ.get("BAG_OUT", os.path.join(ROOT, "results"))
FIG = os.environ.get("BAG_FIG", os.path.join(ROOT, "figures"))
os.makedirs(FIG, exist_ok=True)
COL = {"DIR": "C2", "NODIR": "C3"}


def main():
    d = torch.load(os.path.join(RES, "ghmm_coding.pt"), weights_only=False)
    agg = d["agg"]; rows = d["rows"]
    regs = ["DIR", "NODIR"]; x = np.arange(2)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))

    # (a) SAME coupling; the separable-coordinate flip (z R²) and the footprint
    corr = [abs(agg[r]["corr_q_sharp"][0]) for r in regs]
    zr = [agg[r]["z_r2"][0] for r in regs]
    fp = [agg[r]["footprint_align"][0] for r in regs]; fp_sd = [agg[r]["footprint_align"][1] for r in regs]
    a0 = ax[0]
    w = 0.26
    a0.bar(x - w, corr, w, color="0.6", label="|corr(q,sharp)| (generative coupling)")
    a0.bar(x, zr, w, color=[COL[r] for r in regs], label="z R² (separable alignment coordinate)")
    a0.bar(x + w, fp, w, yerr=fp_sd, capsize=4, color="0.3", alpha=0.85,
           label="footprint |d_align·ŵ_cap| (×1, ‖·‖-normalized)")
    for i in range(2):
        a0.text(x[i], zr[i] + 0.02, f"{zr[i]:.2f}", ha="center", fontsize=8, fontweight="bold")
        a0.text(x[i] + w, fp[i] + 0.02, f"{fp[i]:.3f}", ha="center", fontsize=7)
    a0.set_xticks(x); a0.set_xticklabels(["DIR\n(drift+sharpness)", "NODIR\n(sharpness only)"])
    a0.set_ylabel("value"); a0.set_ylim(0, 1.05)
    a0.set_title("(a) Same coupling (|corr|≈0.89), opposite representation:\nDIR forms a separable alignment coordinate (z R²=0.93);\nNODIR does not (z R²≈0) — alignment folds into sharpness")
    a0.legend(fontsize=6.6, loc="center left")

    # (b) consequence: capability cross-talk under alignment steering vs the random-⊥ null
    cda = [agg[r]["cap_drop_align"][0] for r in regs]; cda_sd = [agg[r]["cap_drop_align"][1] for r in regs]
    cdr = [agg[r]["cap_drop_rand_mean"][0] for r in regs]
    mr = [agg[r]["m_r2"][0] for r in regs]
    ax[1].bar(x - 0.18, cda, 0.36, yerr=cda_sd, capsize=4, color=[COL[r] for r in regs],
              label="steer learned d_align")
    ax[1].bar(x + 0.18, cdr, 0.36, color="0.8", label="random-⊥ null (matched norm)")
    for i, r in enumerate(regs):
        ax[1].text(x[i] - 0.18, cda[i] + 0.03, f"{cda[i]:.3f}", ha="center", fontsize=8, fontweight="bold")
        ax[1].text(x[i], -0.07, f"cap R²={mr[i]:.2f}", ha="center", fontsize=7)
    ax[1].set_xticks(x); ax[1].set_xticklabels(["DIR", "NODIR"])
    ax[1].set_ylabel("capability cross-talk (clipped-R² drop ∈[0,1])")
    ax[1].set_ylim(-0.1, 1.08)
    ax[1].set_title("(b) Same coupling → opposite cross-talk:\nDIR's d_align is the capability-SPARING direction (≪ null);\nNODIR's is as destructive as random")
    ax[1].legend(fontsize=7.5, loc="center left")

    # (c) where the learned d_align lands in (footprint, cross-talk) space — honest: cross-talk
    # SATURATES, so it is not a graded function of footprint; the point is the LOCATION of d_align.
    a2 = ax[2]
    for r in regs:
        row0 = next(p for p in rows if p["regime"] == r and p["seed"] == 0)
        sc = row0.get("scatter", [])
        rf = [p["footprint"] for p in sc if p["kind"] in ("rand", "interp")]
        rd = [p["cap_drop"] for p in sc if p["kind"] in ("rand", "interp")]
        a2.scatter(rf, rd, s=14, color=COL[r], alpha=0.30)
        cp = next((p for p in sc if p["kind"] == "cap"), None)
        if cp:
            a2.scatter([cp["footprint"]], [cp["cap_drop"]], s=70, marker="s", color=COL[r],
                       edgecolor="k", zorder=4, alpha=0.7)
        al = next(p for p in sc if p["kind"] == "align")
        a2.scatter([al["footprint"]], [al["cap_drop"]], s=200, marker="*", color=COL[r],
                   edgecolor="k", zorder=5, label=f"{r} learned d_align")
    a2.axhline(0.05, color="k", ls=":", lw=0.8)
    a2.text(0.46, 0.10, "capability-sparing band", fontsize=7, color="0.3")
    a2.set_xlabel("footprint of steering direction on capability probe")
    a2.set_ylabel("capability cross-talk (clipped-R² drop)")
    a2.set_ylim(-0.05, 1.05)
    a2.set_title("(c) Where the learned d_align lands (seed 0; ■=cap direction):\nDIR★ at the sparing corner (low fp, ~0 drop); NODIR★ destructive\n(metric saturates — cross-talk is NOT graded in footprint)")
    a2.legend(fontsize=7.5, loc="center right")

    fig.suptitle("C3 coding-mechanism control — at MATCHED coupling (corr≈−0.89, 3 seeds), capability "
                 "cross-talk is set by the CODING MECHANISM (is alignment separably encoded?), not the "
                 "statistical coupling", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(FIG, "fig_ghmm_coding.png"), dpi=120)
    print("wrote fig_ghmm_coding.png")


if __name__ == "__main__":
    main()
