"""Standalone figure for the LEACE separability positive-control (results/ghmm_sep_control.pt).

Shows that PROVABLY erasing one latent (LEACE, R^2->0) leaves the other intact, vs a dimension-matched
random-removal null. Kept standalone to avoid edit collisions with the concurrent session."""
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
    d = torch.load(os.path.join(RES, "ghmm_sep_control.pt"), weights_only=False)
    a = d["agg"]
    def g(k):  # (mean, sd)
        return a[k][0], a[k][1]
    zf, zf_s = g("z_full"); mf, mf_s = g("m_full")
    kz = a["kz_erase_dims"][0]; km = a["km_erase_dims"][0]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)

    # left: ERASE alignment z (rank kz). z collapses (POS), Mess3 survives (CROSS) vs random-kz null
    z_pos, z_pos_s = g("z_after_eraseZ"); m_cross, m_cross_s = g("m_after_eraseZ")
    m_null, m_null_s = g("m_after_rand_kz_mean")
    bars = [("alignment z\n(full)", zf, zf_s, "C3"),
            (f"z after\nLEACE-erase z\n(rank {kz:.0f}, POS)", z_pos, z_pos_s, "0.6"),
            ("Mess3 belief\n(full)", mf, mf_s, "C0"),
            ("Mess3 after\nerase z (CROSS)", m_cross, m_cross_s, "C0"),
            (f"Mess3 after\nrand-{kz:.0f}d (NULL)", m_null, m_null_s, "C7")]
    x = np.arange(len(bars))
    ax[0].bar(x, [b[1] for b in bars], yerr=[b[2] for b in bars], color=[b[3] for b in bars], capsize=3,
              edgecolor="k", linewidth=0.5)
    ax[0].set_xticks(x); ax[0].set_xticklabels([b[0] for b in bars], fontsize=7)
    ax[0].axhline(0, color="k", lw=0.8); ax[0].set_ylabel("held-out probe R²")
    ax[0].set_title("(a) Erase ALIGNMENT z → z dies, capability survives")

    # right: ERASE Mess3 (rank km). Mess3 collapses (POS), z survives (CROSS) vs random-km null
    m_pos, m_pos_s = g("m_after_eraseM"); z_cross, z_cross_s = g("z_after_eraseM")
    z_null, z_null_s = g("z_after_rand_km_mean")
    bars2 = [("Mess3 belief\n(full)", mf, mf_s, "C0"),
             (f"Mess3 after\nLEACE-erase m\n(rank {km:.0f}, POS)", m_pos, m_pos_s, "0.6"),
             ("alignment z\n(full)", zf, zf_s, "C3"),
             ("z after\nerase m (CROSS)", z_cross, z_cross_s, "C3"),
             (f"z after\nrand-{km:.0f}d (NULL)", z_null, z_null_s, "C7")]
    x2 = np.arange(len(bars2))
    ax[1].bar(x2, [b[1] for b in bars2], yerr=[b[2] for b in bars2], color=[b[3] for b in bars2], capsize=3,
              edgecolor="k", linewidth=0.5)
    ax[1].set_xticks(x2); ax[1].set_xticklabels([b[0] for b in bars2], fontsize=7)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_title("(b) Erase MESS3 belief → capability dies, z survives")

    fig.suptitle("C3 separability — closed-form LEACE positive control: provably erasing one latent (R²→0) "
                 "leaves the other untouched (3 seeds, drift, best-Mess3 layer)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(FIG, "fig_ghmm_sep_control.png"), dpi=120)
    print("wrote fig_ghmm_sep_control.png")


if __name__ == "__main__":
    main()
