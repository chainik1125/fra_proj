"""Standalone figure for the GHMM emission-sharpness ablation (results/ghmm_sharpness.pt).

Does the alignment⊗capability FACTORIZATION (§4.4 C3) survive when the Mess3 capability belief is
genuinely history-RICH rather than last-symbol-predictable? We sweep emission concentration a down in
a STICKY drift regime (stay=0.70). Three panels:
  (a) the geometry gets rich: last-symbol-control R² for the TRUE belief drops with a, while the model
      still decodes the belief well above that baseline (the model holds history, not just last symbol);
      generative coupling corr(q,sharpness) stays low (near-factored) throughout.
  (b) factorization holds: cross-alignment TRANSFER R² (aligned→misaligned disjoint seqs) tracks the
      in-context Mess3 R², and erase-one/decode-other stays lossless, at every a.
  (c) causal geometry: steering the alignment direction stays capability-SPARING (align cross-talk far
      below the random-⊥ null) and d_align footprint stays small, as the belief geometry gets rich.
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


def g(agg, a, k):
    return agg[a][k][0], agg[a][k][1]


def main():
    d = torch.load(os.path.join(RES, "ghmm_sharpness.pt"), weights_only=False)
    agg = d["agg"]; As = sorted(agg.keys(), reverse=True)  # 0.80, 0.55, 0.40
    xs = np.array(As)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.7))

    # (a) richness + coupling
    mlast = [g(agg, a, "m_lastsym_ctrl")[0] for a in As]
    mr2 = [g(agg, a, "m_r2")[0] for a in As]; mr2sd = [g(agg, a, "m_r2")[1] for a in As]
    zr2 = [g(agg, a, "z_r2")[0] for a in As]; zr2sd = [g(agg, a, "z_r2")[1] for a in As]
    corr = [abs(g(agg, a, "corr_q_sharp")[0]) for a in As]
    a0 = ax[0]
    a0.plot(xs, mlast, "s--", color="0.5", label="last-symbol control R² (TRUE belief)")
    a0.errorbar(xs, mr2, yerr=mr2sd, marker="o", color="C0", label="model Mess3 belief R²")
    a0.errorbar(xs, zr2, yerr=zr2sd, marker="^", color="C2", label="model alignment z R²")
    a0.plot(xs, corr, "v:", color="C3", label="|corr(q,sharpness)| (coupling)")
    a0.set_xlabel("emission concentration a (sharp → diffuse)"); a0.set_ylabel("R² / |corr|")
    a0.set_ylim(0, 1.02); a0.invert_xaxis()
    a0.set_title("(a) Lowering a makes the belief history-RICH (last-sym R²↓):\nat a=0.55 the model still HOLDS it (mR²=0.88 ≫ last-sym 0.75),\nbut the alignment cue z weakens (0.72→0.09→0.00); coupling stays low")
    a0.legend(fontsize=7, loc="lower left")

    # (b) factorization: transfer + separability
    tr = [g(agg, a, "transfer_m")[0] for a in As]; trsd = [g(agg, a, "transfer_m")[1] for a in As]
    znoM = [g(agg, a, "z_r2_after_Merase")[0] for a in As]
    mnoZ = [g(agg, a, "m_r2_after_zerase")[0] for a in As]
    a1 = ax[1]
    a1.errorbar(xs, mr2, yerr=mr2sd, marker="o", color="C0", label="in-context Mess3 R²")
    a1.errorbar(xs, tr, yerr=trsd, marker="D", color="C1", label="cross-alignment TRANSFER R²\n(aligned→misaligned, disjoint seqs)")
    a1.plot(xs, mnoZ, "x--", color="C0", alpha=0.7, label="Mess3 R² after ERASE alignment subspace")
    a1.plot(xs, znoM, "+--", color="C2", alpha=0.7, label="z R² after erase Mess3-plane")
    a1.set_xlabel("emission concentration a"); a1.set_ylabel("R²")
    a1.set_ylim(0, 1.02); a1.invert_xaxis()
    a1.set_title("(b) Capability factorization HOLDS at genuine richness:\ntransfer across alignment ≈ in-context & survives erasing\nthe alignment subspace (load-bearing leg: m|noZ ≈ mR²)")
    a1.legend(fontsize=6.6, loc="lower left")

    # (c) steering cross-talk + footprint
    cda = [g(agg, a, "cap_drop_align")[0] for a in As]; cdasd = [g(agg, a, "cap_drop_align")[1] for a in As]
    cdr = [g(agg, a, "cap_drop_rand_mean")[0] for a in As]
    fpa = [g(agg, a, "footprint_align")[0] for a in As]
    fpr = [g(agg, a, "footprint_rand_mean")[0] for a in As]
    a2 = ax[2]
    a2.errorbar(xs, cda, yerr=cdasd, marker="o", color="C2", label="cross-talk: steer d_align")
    a2.plot(xs, cdr, "s--", color="0.6", label="random-⊥ null (matched norm)")
    a2.set_xlabel("emission concentration a"); a2.set_ylabel("capability cross-talk (clipped-R² drop)")
    a2.set_ylim(-0.05, 1.05); a2.invert_xaxis()
    a2b = a2.twinx()
    a2b.plot(xs, fpa, "^:", color="C4", label="footprint d_align")
    a2b.plot(xs, fpr, "v:", color="0.7", label="footprint random")
    a2b.set_ylabel("footprint on capability probe"); a2b.set_ylim(0, max(fpr) * 1.4 + 1e-3)
    a2.set_title("(c) Steering stays below the null but the MARGIN shrinks\nas z weakens (cross-talk 0.00→0.24→0.32; at a=0.40 d_align\nis noise so this leg is weak) — footprint stays small")
    a2.legend(fontsize=7, loc="center left"); a2b.legend(fontsize=7, loc="upper right")

    fig.suptitle("C3 emission-sharpness ablation — capability factorization survives a GENUINELY "
                 "history-rich belief (a=0.55: model mR²=0.88 ≫ last-sym 0.75), NOT just last-symbol "
                 "geometry; the cost is a weakening alignment cue as a→0.40 (z R²→0)", fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(FIG, "fig_ghmm_sharpness.png"), dpi=120)
    print("wrote fig_ghmm_sharpness.png")


if __name__ == "__main__":
    main()
