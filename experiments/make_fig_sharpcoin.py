"""Standalone figure for the sharp-coin OOD fault-tolerance ladder
(results/c2_sharpcoin_ft.pt -> figures/fig_c2_sharpcoin_ft.png).
Kept standalone to avoid churning the shared figure module a concurrent session edits."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = os.environ.get("BAG_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG = os.path.join(ROOT, "figures"); RES = os.path.join(ROOT, "results")
os.makedirs(FIG, exist_ok=True)


def main():
    d = torch.load(os.path.join(RES, "c2_sharpcoin_ft.pt"), weights_only=False)
    ladder = d["ladder"]; rdec = d["r"]; n = d["n"]
    ks = np.arange(n + 1)
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(ladder)))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))

    # (a) P(misaligned) vs k, model (solid) tracks the sharp-appropriate oracle (dashed); ideal flip at k=r+1
    for (pair, c) in zip(ladder, colors):
        rec = d["by_pair"][pair]
        mp = np.array(rec["model_p_yes"]); op = np.array(rec["oracle_p_yes"])
        lab = f"pA/pM={pair[0]}/{pair[1]}" + (" (in-dist)" if pair == tuple(ladder[0]) else "")
        ax[0].plot(ks, mp, "o-", color=c, label=lab)
        ax[0].plot(ks, op, "^--", color=c, alpha=0.5)
    ax[0].axvline(rdec + 1, color="0.4", ls=":", lw=1.2, label=f"ideal code threshold k=r+1={rdec+1}")
    ax[0].axhline(0.5, color="0.7", ls="-", lw=0.8)
    ax[0].set_xlabel("k = # persistently-misaligned blocks (of n=5)")
    ax[0].set_ylabel("readout P(logical = misaligned)")
    ax[0].set_title("(a) OOD fault tolerance across emission sharpness\n"
                    "(solid = trained model, dashed = sharp-appropriate Bayes oracle)")
    ax[0].legend(fontsize=7, loc="upper left")

    # (b) the decision MARGIN (P at k=r+1 minus P at k=r) sharpens with the coins: model partially follows oracle
    sharp = np.array([pM - pA for (pA, pM) in ladder])  # emission separation
    mm = np.array([d["by_pair"][p]["margin_model"] for p in ladder])
    mo = np.array([d["by_pair"][p]["margin_orac"] for p in ladder])
    ax[1].plot(sharp, mo, "^--", color="C2", label="Bayes oracle (sharp-appropriate)")
    ax[1].plot(sharp, mm, "o-", color="C0", label="trained model (soft-trained)")
    ax[1].set_xlabel("emission separation  pM − pA  (→ sharper coins)")
    ax[1].set_ylabel(f"decision margin  P(k={rdec+1}) − P(k={rdec})")
    ax[1].set_title("(b) The threshold SHARPENS with the coins\n"
                    "model exploits cleaner OOD evidence but undershoots the sharp Bayes step")
    ax[1].legend(fontsize=8, loc="upper left")

    fig.suptitle("C2 sharp-coin OOD fault tolerance: the soft-trained majority-decoder generalises the "
                 "threshold MECHANISM to sharper (OOD) coins — flip sharpens toward the ideal k=r+1 — "
                 "but undershoots the crisp Bayes step", fontsize=10.5, y=1.0)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_c2_sharpcoin_ft.png"), bbox_inches="tight", dpi=120)
    print("wrote fig_c2_sharpcoin_ft.png")


if __name__ == "__main__":
    main()
