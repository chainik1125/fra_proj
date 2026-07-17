"""Standalone figure for the C1 algorithm-vs-memorization shift test (results/c1_genshift.pt).

Shows the model tracks the TRAINED-parameter Bayes filter (KL flat at the train floor) even on tokens from
SHIFTED dynamics, while the locally-optimal TRUE filter diverges — evidence for a computed-belief
(algorithmic) representation. Standalone to avoid edit collisions with the concurrent session."""
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
    d = torch.load(os.path.join(RES, "c1_genshift.pt"), weights_only=False)
    rows = sorted(d["rows"], key=lambda r: r["qstar"])
    qs = np.array([r["qstar"] for r in rows])
    kt = np.array([r["kl_to_trained"] for r in rows])
    ku = np.array([r["kl_to_true"] for r in rows])
    mrate = np.array([r["model_rate"] for r in rows])
    rt = np.array([r["rate_trained_filter"] for r in rows])
    ru = np.array([r["rate_true_filter"] for r in rows])
    train_kl = d["train_kl"]; q0 = d["config"]["q0star"]

    # optional third panel: emission-shift on the canonical model (exp_c1_emshift.py)
    em_path = os.path.join(RES, "c1_emshift.pt")
    have_em = os.path.exists(em_path)
    ncol = 3 if have_em else 2
    fig, ax = plt.subplots(1, ncol, figsize=(5.7 * ncol, 4.5))

    # (a) KL to each oracle vs the shifted q*
    ax[0].plot(qs, kt, "s-", color="C0", label="KL(model ‖ TRAINED-param filter)")
    ax[0].plot(qs, ku, "o-", color="C3", label="KL(model ‖ TRUE-param filter)")
    # overlay the canonical headline model (2nd independently-trained model) if available
    em_path0 = os.path.join(RES, "c1_emshift.pt")
    if os.path.exists(em_path0):
        de0 = torch.load(em_path0, weights_only=False)
        if "trans_rows" in de0:
            tr = sorted(de0["trans_rows"], key=lambda r: r["qstar"])
            tq = np.array([r["qstar"] for r in tr])
            ax[0].plot(tq, [r["kl_to_trained"] for r in tr], "s:", color="C0", alpha=0.5, mfc="none",
                       label="canonical model ‖ TRAINED (2nd model)")
            ax[0].plot(tq, [r["kl_to_true"] for r in tr], "o:", color="C3", alpha=0.5, mfc="none",
                       label="canonical model ‖ TRUE (2nd model)")
    ax[0].axhline(train_kl, color="C0", ls=":", lw=1, label=f"in-dist train KL ({train_kl:.4f})")
    ax[0].axvline(q0, color="k", ls="--", lw=0.8, label=f"training q*={q0:.2f}")
    ax[0].set_xlabel("eval steady-state misalignment q*  (shifted dynamics)")
    ax[0].set_ylabel("next-symbol KL (nats)")
    ax[0].set_title("(a) Model tracks the TRAINED filter on OOD inputs\n(KL-to-trained flat; KL-to-true diverges)")
    ax[0].legend(fontsize=7.5)

    # (b) behavioural: model output rate follows the TRAINED filter, not the locally-optimal one
    ax[1].plot(qs, mrate, "D-", color="k", label="model P(x=1)")
    ax[1].plot(qs, rt, "s--", color="C0", label="TRAINED-param filter")
    ax[1].plot(qs, ru, "o--", color="C3", label="TRUE-param filter")
    ax[1].axvline(q0, color="k", ls="--", lw=0.8)
    ax[1].set_xlabel("eval steady-state misalignment q*  (shifted dynamics)")
    ax[1].set_ylabel("mean next-symbol rate P(x=1)")
    ax[1].set_title("(b) Model's output rate follows the TRAINED filter\n(it does not re-optimise to the shifted dynamics)")
    ax[1].legend(fontsize=7.5)

    # (c) emission-shift on the canonical model: KL-to-trained flat, KL-to-true blows up (~480x)
    if have_em:
        de = torch.load(em_path, weights_only=False)
        er = sorted(de["rows"], key=lambda r: r["sep"])
        sep = np.array([r["sep"] for r in er])
        ekt = np.array([r["kl_to_trained"] for r in er])
        eku = np.array([r["kl_to_true"] for r in er])
        ax[2].semilogy(sep, ekt, "s-", color="C0", label="KL(model ‖ TRAINED-likelihood filter)")
        ax[2].semilogy(sep, eku, "o-", color="C3", label="KL(model ‖ TRUE-likelihood filter)")
        ax[2].axhline(de["indist_kl"], color="C0", ls=":", lw=1, label=f"in-dist KL ({de['indist_kl']:.1e})")
        ax[2].axvline(0.20, color="k", ls="--", lw=0.8, label="trained emissions (sep 0.20)")
        ax[2].set_xlabel("emission separation s  (pₐ=½−s, p_M=½+s; shifted)")
        ax[2].set_ylabel("next-symbol KL (nats, log scale)")
        ax[2].set_title("(c) EMISSION shift (canonical model, no retrain):\nmodel keeps its baked-in likelihood (KL-to-true ↑ ~480×)")
        ax[2].legend(fontsize=7)

    fig.suptitle("C1 implements the Bayes filter as a fixed algorithm — on OOD inputs (shifted transitions AND "
                 "emissions) it applies the trained-parameter filter, not memorised I/O statistics", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(FIG, "fig_c1_genshift.png"), dpi=120)
    print("wrote fig_c1_genshift.png")


if __name__ == "__main__":
    main()
