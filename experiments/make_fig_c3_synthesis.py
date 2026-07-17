"""Synthesis figure for the C3 robustness story (uses only already-saved tensors; no compute).

Crystallizes the central §4.4 result: across FOUR independent ways of coupling the alignment latent to the
capability (Mess3) belief, the transformer keeps the two codes FACTORED — rising statistical coupling never
degrades the capability code — with two honest boundaries.

(a) Load-bearing, single CONSISTENT metric across three experiments: capability R² after ERASING the
    alignment subspace, normalized by the in-context capability R² (1.0 = capability fully survives removing
    everything the model knows about alignment). Plotted vs the statistical coupling |corr(q, sharpness)|,
    pooling the λ-sweep (Fig 5b), the emission-sharpness ablation (Fig 5d), and the switching-rate sweep.
    All points sit at ≈1.0 across a 13× coupling range (0.065→0.884). The alignment code's own survival
    (z after erasing the Mess3 plane) is overlaid; it too is ≈1.0 EXCEPT the very-diffuse a=0.40 point, where
    the alignment signal has collapsed (the one boundary on this axis).
(b) The other boundary — the coding mechanism (Fig 5c). At matched coupling (≈0.89) the causal capability
    cross-talk under alignment steering is 0.002 when a separable alignment code exists (DIR) and 0.999 when
    it does not (NODIR) — capability preservation is set by whether a separable code exists, not by coupling.
"""
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


def main():
    # --- gather (|coupling|, cap-after-erase/incontext, z-after-erase/incontext) from 3 experiments ---
    pts = []  # (coupling, m_ratio, z_ratio, kind, label_if_boundary)
    sw = torch.load(os.path.join(RES, "ghmm_sweep.pt"), weights_only=False)["rows"]
    lams = sorted(set(r["lambda"] for r in sw))
    for lam in lams:
        rs = [r for r in sw if r["lambda"] == lam]
        c = np.mean([abs(r["corr_q_maxcoord"]) for r in rs])
        mr = np.mean([r["m_r2_after_remove_zsub"] / max(r["m_r2_full"], 1e-6) for r in rs])
        zr = np.mean([r["z_r2_after_remove_Mplane"] / max(r["z_r2_full"], 1e-6) for r in rs])
        pts.append((c, mr, zr, "sweep", None))
    for fn, kind in [("ghmm_sharpness", "sharpness"), ("ghmm_switchrate", "switchrate")]:
        ag = torch.load(os.path.join(RES, fn + ".pt"), weights_only=False)["agg"]
        for k, v in ag.items():
            c = abs(v["corr_q_sharp"][0]); m = max(v["m_r2"][0], 1e-6); z = max(v["z_r2"][0], 1e-6)
            mr = v["m_r2_after_zerase"][0] / m; zr = v["z_r2_after_Merase"][0] / z
            lbl = "a=0.40" if (kind == "sharpness" and abs(float(k) - 0.40) < 1e-6) else None
            pts.append((c, mr, zr, kind, lbl))

    cod = torch.load(os.path.join(RES, "ghmm_coding.pt"), weights_only=False)["agg"]

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.0), gridspec_kw={"width_ratios": [2.0, 1.0]})

    # (a)
    a0 = ax[0]
    mk = {"sweep": ("D", "C0", "λ-sweep (emission asymmetry, Fig 5b)"),
          "sharpness": ("o", "C1", "belief-richness ablation (Fig 5d)"),
          "switchrate": ("^", "C2", "persona switching-rate sweep (§4.4)")}
    seen = set()
    for c, mr, zr, kind, lbl in pts:
        m, col, name = mk[kind]
        a0.scatter([c], [mr], marker=m, s=80, color=col, zorder=4,
                   label=(name if kind not in seen else None))
        seen.add(kind)
        a0.scatter([c], [zr], marker=m, s=42, facecolor="none", edgecolor=col, alpha=0.7, zorder=3)
        if lbl:
            a0.annotate(f"{lbl}\nalignment z\ncollapses", (c, zr), textcoords="offset points",
                        xytext=(8, -2), fontsize=7.5, color="0.25",
                        arrowprops=dict(arrowstyle="->", color="0.5", lw=0.8))
    a0.axhline(1.0, color="k", ls=":", lw=0.8)
    a0.text(0.07, 1.012, "perfect factorization (capability untouched by erasing alignment)", fontsize=7.5, color="0.3")
    # legend proxies for filled vs hollow
    a0.scatter([], [], marker="s", s=80, color="0.4", label="capability after erasing ALIGNMENT subspace (÷ in-context)")
    a0.scatter([], [], marker="s", s=42, facecolor="none", edgecolor="0.4", label="alignment z after erasing MESS3 plane (÷ in-context)")
    a0.set_xlabel("statistical coupling  |corr(q, belief sharpness)|")
    a0.set_ylabel("latent R² after erasing the OTHER latent  (÷ in-context R²)")
    a0.set_ylim(0, 1.12); a0.set_xlim(0, 0.95)
    a0.set_title("(a) Rising statistical coupling (13× range) never degrades the capability code\n"
                 "12 settings from 3 independent coupling sources — capability survives erasing alignment at ≈1.0 throughout\n"
                 "(z survives too, except where the alignment signal itself collapses: a=0.40)")
    a0.legend(fontsize=7.0, loc="lower left", framealpha=0.95)

    # (b) coding-mechanism boundary
    a1 = ax[1]
    regs = ["DIR", "NODIR"]; x = np.arange(2)
    cd = [cod[r]["cap_drop_align"][0] for r in regs]
    cr = [cod[r]["cap_drop_rand_mean"][0] for r in regs]
    a1.bar(x - 0.18, cd, 0.36, color=["C2", "C3"], label="steer learned d_align")
    a1.bar(x + 0.18, cr, 0.36, color="0.8", label="random-⊥ null")
    for i, r in enumerate(regs):
        a1.text(x[i] - 0.18, cd[i] + 0.03, f"{cd[i]:.3f}", ha="center", fontsize=8.5, fontweight="bold")
    a1.set_xticks(x); a1.set_xticklabels(["DIR\n(separable code\nexists)", "NODIR\n(no separable\ncode)"])
    a1.set_ylabel("capability cross-talk under alignment steering")
    a1.set_ylim(0, 1.08)
    a1.set_title("(b) The boundary: at MATCHED coupling (≈0.89),\ncapability preservation is set by whether a\nseparable alignment code exists (Fig 5c)")
    a1.legend(fontsize=7.5, loc="center left")

    fig.suptitle("C3 synthesis — the transformer keeps alignment ⊗ capability FACTORED across four independent "
                 "coupling sources; statistical coupling ≠ representational entanglement", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(FIG, "fig_ghmm_c3_synthesis.png"), dpi=120)
    print("wrote fig_ghmm_c3_synthesis.png")


if __name__ == "__main__":
    main()
