"""Standalone figure for the GHMM independence sweep (results/ghmm_sweep.pt).

Dose-response: as the two generative latents (alignment z, Mess3 capability belief) become coupled
(swept by lambda via emission-sharpness asymmetry), do the neural codes stay separable, and does
steering alignment cause more capability cross-talk? Uses BOUNDED causal metrics (clipped-R² drop,
next-symbol KL) with a multi-direction random-⊥ null band, 3-seed error bars. Kept standalone (not
in make_figures_active.py) to avoid edit collisions with the concurrent session."""
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
    d = torch.load(os.path.join(RES, "ghmm_sweep.pt"), weights_only=False)
    agg = sorted(d["agg"], key=lambda a: a["lambda"])
    lam = np.array([a["lambda"] for a in agg])
    cc = np.array([abs(a["corr_q_maxcoord"]) for a in agg])          # generative coupling
    cc_sd = np.array([a["corr_q_maxcoord_sd"] for a in agg])
    zr = np.array([a["z_r2"] for a in agg]); zr_sd = np.array([a["z_r2_sd"] for a in agg])
    mr = np.array([a["m_r2"] for a in agg]); mr_sd = np.array([a["m_r2_sd"] for a in agg])
    znoM = np.array([a["z_r2_after_remove_Mplane"] for a in agg])
    znoM_sd = np.array([a["z_r2_after_remove_Mplane_sd"] for a in agg])
    mnoZ = np.array([a["m_r2_after_remove_zsub"] for a in agg])
    mnoZ_sd = np.array([a["m_r2_after_remove_zsub_sd"] for a in agg])
    capd = np.array([a["cap_clip_drop_align"] for a in agg])
    capd_sd = np.array([a["cap_clip_drop_align_sd"] for a in agg])
    capr = np.array([a["cap_clip_drop_rand_mean"] for a in agg])
    capr_sd = np.array([a["cap_clip_drop_rand_mean_sd"] for a in agg])
    nska = np.array([a["nsk_align"] for a in agg]); nska_sd = np.array([a["nsk_align_sd"] for a in agg])
    nskr = np.array([a["nsk_rand_mean"] for a in agg])
    fpa = np.array([a["footprint_align"] for a in agg]); fpr = np.array([a["footprint_rand_mean"] for a in agg])

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.7))

    # (a) the knob: generative coupling rises with lambda; BOTH latents stay decodable (z even strengthens)
    a0 = ax[0]; a0b = a0.twinx()
    a0.errorbar(lam, cc, yerr=cc_sd, fmt="o-", color="k", capsize=3,
                label="|corr(q, belief sharpness)|  (generative coupling)")
    a0b.errorbar(lam, zr, yerr=zr_sd, fmt="s--", color="C3", capsize=3, label="alignment z R²")
    a0b.errorbar(lam, mr, yerr=mr_sd, fmt="^--", color="C0", capsize=3, label="Mess3 belief R²")
    a0.set_xlabel("λ  (emission-sharpness asymmetry: aligned sharp ↔ misaligned diffuse)")
    a0.set_ylabel("generative |corr|", color="k"); a0b.set_ylabel("linear decode R²")
    a0.set_ylim(0, 1.0); a0b.set_ylim(0, 1.02)
    a0.set_title("(a) Knob: coupling sweeps 0.18→0.89;\nboth latents stay decodable throughout")
    a0.legend(fontsize=7, loc="upper left"); a0b.legend(fontsize=7, loc="lower right")

    # (b) LINEAR separability holds at every coupling: each latent survives erasing the other's subspace
    a1 = ax[1]
    a1.errorbar(cc, znoM / zr, yerr=znoM_sd / np.maximum(zr, 1e-6), fmt="s-", color="C3", capsize=3,
                label="z R² retained after erasing Mess3-plane")
    a1.errorbar(cc, mnoZ / mr, yerr=mnoZ_sd / np.maximum(mr, 1e-6), fmt="^-", color="C0", capsize=3,
                label="Mess3 R² retained after erasing z-subspace")
    a1.axhline(1.0, color="k", ls=":", lw=0.8)
    a1.set_xlabel("generative coupling |corr(q, sharpness)|")
    a1.set_ylabel("fraction of R² retained"); a1.set_ylim(0, 1.15)
    a1.set_title("(b) Linear codes stay near-separable at\nEVERY coupling (erase-one, decode-other ≈ 1)")
    a1.legend(fontsize=7, loc="lower left")

    # (c) BOUNDED causal cross-talk vs MEASURED coupling (dose-response), with random-⊥ null band
    a2 = ax[2]
    a2.errorbar(cc, capd, yerr=capd_sd, fmt="o-", color="C2", capsize=3,
                label="capability clipped-R² drop, steer ALIGN")
    a2.fill_between(cc, capr - capr_sd, capr + capr_sd, color="C7", alpha=0.25,
                    label="random-⊥ null band (matched norm)")
    a2.plot(cc, capr, ":", color="C7", lw=1)
    a2.set_xlabel("generative coupling |corr(q, sharpness)|")
    a2.set_ylabel("capability cross-talk (clipped-R² drop ∈[0,1])", color="C2")
    a2.set_ylim(-0.02, max(0.2, float(capr.max()) * 1.1))
    a2b = a2.twinx()
    a2b.errorbar(cc, nska, yerr=nska_sd, fmt="d--", color="C4", capsize=3, label="next-symbol KL, steer ALIGN")
    a2b.set_ylabel("next-symbol KL (steered‖unsteered)", color="C4")
    a2.set_title("(c) Causal cross-talk vs coupling (matched layer+norm):\n"
                 "align-steer stays far below the random-⊥ null at all couplings")
    a2.legend(fontsize=7, loc="upper left"); a2b.legend(fontsize=7, loc="upper right")
    # footprint annotation (circularity check)
    a2.text(0.02, 0.02, f"probe footprint  align≈{fpa.mean():.2f}  rand≈{fpr.mean():.2f}\n"
            "(align not specially ⊥ to capability probe)", transform=a2.transAxes, fontsize=6.5,
            va="bottom", ha="left", bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.8))

    nseed = d["config"].get("nseed", "?"); slayer = d["config"].get("slayer", "?")
    fig.suptitle(f"C3 sweep — statistical coupling does NOT imply causal cross-talk: linear separability and "
                 f"alignment-steering capability-preservation hold across the coupling range "
                 f"(drift-direction code, steer L{slayer}, {nseed} seeds)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(FIG, "fig_ghmm_sweep.png"), dpi=120)
    print("wrote fig_ghmm_sweep.png")


if __name__ == "__main__":
    main()
