"""Standalone figure for the R_M epidemic-threshold-in-predictions experiment
(results/c2_rm_threshold.pt -> figures/fig_c2_rm_threshold.png).

Kept separate from make_figures_active.py to avoid editing a file a concurrent session churns.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = os.environ.get("BAG_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG = os.path.join(ROOT, "figures")
RES = os.path.join(ROOT, "results")
os.makedirs(FIG, exist_ok=True)


def main():
    d = torch.load(os.path.join(RES, "c2_rm_threshold.pt"), weights_only=False)
    R = d["results"]
    rm = np.array([r["R_M"] for r in R])
    model_rate = np.array([r["model_logical_rate"] for r in R])
    orac_rate = np.array([r["orac_logical_rate"] for r in R])
    true_rate = np.array([r["true_logical_rate"] for r in R])
    chain_frac = np.array([r["true_chain_frac"] for r in R])
    sis = np.array([r["sis_theory"] for r in R])
    kl = np.array([r["kl_to_joint"] for r in R])
    merr = np.array([r["model_logical_err"] for r in R])
    berr = np.array([r["bayes_logical_err"] for r in R])

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))

    # (a) the THRESHOLD elbow in the model's implied endemic logical-misalignment rate
    ax[0].plot(rm, model_rate, "o-", color="C0", label="transformer (implied endemic)")
    ax[0].plot(rm, orac_rate, "^--", color="C2", label="joint Bayes filter (oracle)")
    ax[0].plot(rm, true_rate, "x:", color="C7", label="true logical (hidden)")
    ax[0].axvline(1.0, color="0.5", ls=":", lw=1)
    ax[0].set_xlabel("reproduction number $R_M=\\beta(n-1)/(n\\gamma)$")
    ax[0].set_ylabel("endemic logical-misalignment rate")
    ax[0].set_title("(a) The model's predictions reproduce the\nR$_M$=1 epidemic-threshold elbow")
    ax[0].legend(fontsize=8)

    # (b) per-chain endemic fraction vs SIS theory I*=1-1/R_M (the process bifurcation it inherits)
    ax[1].plot(rm, chain_frac, "o-", color="C3", label="true per-chain endemic fraction")
    grid = np.linspace(rm.min(), rm.max(), 200)
    ax[1].plot(grid, np.clip(1 - 1.0 / grid, 0, 1), "-", color="0.4", lw=1.5,
               label="SIS theory $I^*=\\max(0,1-1/R_M)$")
    ax[1].axvline(1.0, color="0.5", ls=":", lw=1)
    ax[1].set_xlabel("reproduction number $R_M$"); ax[1].set_ylabel("per-chain endemic fraction $I^*$")
    ax[1].set_title("(b) The underlying SIS bifurcation\n(finite-T data approaches the closed form)")
    ax[1].legend(fontsize=8)

    # (c) the learned decoder stays Bayes-optimal across the threshold (low KL, error tracks Bayes)
    ax2 = ax[2].twinx()
    l1, = ax[2].plot(rm, kl, "s-", color="C4", label="KL(model→joint) [left]")
    l2, = ax2.plot(rm, merr, "o-", color="C0", label="model logical err [right]")
    l3, = ax2.plot(rm, berr, "^--", color="C2", label="Bayes logical err [right]")
    ax[2].axvline(1.0, color="0.5", ls=":", lw=1)
    ax[2].set_xlabel("reproduction number $R_M$"); ax[2].set_ylabel("KL to joint oracle (nats)")
    ax2.set_ylabel("logical error")
    ax[2].set_title("(c) Learned decoder stays joint-Bayes-optimal\nacross R$_M$=1 (KL low, err tracks Bayes)")
    ax[2].legend(handles=[l1, l2, l3], fontsize=8, loc="upper left")

    fig.suptitle("C2 epidemic threshold IN the transformer's learned predictions: the model tracks the "
                 "exact joint filter across R$_M$=1 (sub→super-critical), reproducing the SIS elbow",
                 fontsize=11, y=1.02)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_c2_rm_threshold.png"), bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_c2_rm_threshold.png")
    # also dump a compact text table for the writeup
    print("R_M  model_rate orac_rate true_rate chain_frac SIS    KL→joint mErr bErr")
    for r in R:
        print(f"{r['R_M']:.2f}  {r['model_logical_rate']:.3f}     {r['orac_logical_rate']:.3f}    "
              f"{r['true_logical_rate']:.3f}    {r['true_chain_frac']:.3f}     {r['sis_theory']:.3f}  "
              f"{r['kl_to_joint']:.4f}  {r['model_logical_err']:.3f} {r['bayes_logical_err']:.3f}")


if __name__ == "__main__":
    main()
