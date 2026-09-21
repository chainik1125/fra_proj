"""THE MONEY FIGURE: warp curve on trained models.

Position-only attention (Wq=Wk=0 frozen) = the theory-optimal content-independent
pattern class (B3ii). For a grid of observation noise sigma (fixed lambda=0.7),
train, extract the geometric lag rate, and compare to eta(rho) [warp] vs lambda
[belief rate]. attn_gain (analytic) shades how load-bearing attention is (pinning).
Writes out/warp_curve.json and out/warp_curve.png.
"""
import json, time, numpy as np, torch
from scipy.linalg import toeplitz
from reset import ResetProcess, geometric_rate_fit
from model import OneLayerAttn, train, eval_loss

LAM, N, D, T = 0.7, 3, 32, 28
SIGMAS = [0.3, 0.45, 0.6, 0.8, 1.1, 1.5, 2.2, 3.2, 5.0]
SEEDS = [0, 1, 2]
STEPS = 4000


def floors(g, taumax=80):
    def pmse(G, eta):
        a = eta ** np.arange(1, taumax + 1); Gm = toeplitz(G)
        Sig = np.array([[G[0], a @ G[1:taumax+1]], [a @ G[1:taumax+1], a @ Gm[1:,1:] @ a]])
        cv = np.array([G[1], a[:taumax-1] @ G[2:taumax+1]])
        return G[0] - cv @ np.linalg.solve(Sig, cv)
    dir_only = add_eta = 0.0
    for i in range(g.N):
        A, nu, lam, eta = g.const['A'][i], g.const['nu'][i], g.lam[i], g.const['eta'][i]
        G = A * lam ** np.arange(taumax+1); G[0] = A + nu
        dir_only += G[0] - G[1]**2/G[0]; add_eta += pmse(G, eta)
    return float(dir_only), float(add_eta)


def train_posonly(g, seed):
    m = OneLayerAttn(D, d_h=D, bias=True, init=0.02, seed=seed, pos_key=True, n_ctx=T)
    with torch.no_grad():
        m.Wq.zero_(); m.Wk.zero_()
    m.Wq.requires_grad_(False); m.Wk.requires_grad_(False)
    train(m, g, T=T, steps=STEPS, batch=256, lr=3e-3)
    X, _, _ = g.sample_seq(4000, T)
    al = m.mean_lag(torch.tensor(X, dtype=torch.float32), t_lo=12); al = al / al.sum()
    return geometric_rate_fit(al, 1, 5), float(al[1]/al[0])


def main():
    t0 = time.time(); rows = []
    for sigma in SIGMAS:
        g0 = ResetProcess(N, D, lam=LAM, p=0.3, mu=1.0, sigma=sigma, seed=0, orthogonalize=True)
        eta_th = float(g0.const['eta'][0]); rho = float(g0.const['rho'][0])
        dir_only, add_eta = floors(g0)
        attn_gain_rel = (dir_only - add_eta) / dir_only     # frac of loss attention carries
        fits, a1a0 = [], []
        for s in SEEDS:
            g = ResetProcess(N, D, lam=LAM, p=0.3, mu=1.0, sigma=sigma, seed=s,
                             orthogonalize=True, D=g0.D)
            f, r = train_posonly(g, s); fits.append(f); a1a0.append(r)
        rows.append(dict(sigma=sigma, rho=rho, eta_theory=eta_th, lam=LAM,
                         attn_gain_rel=attn_gain_rel,
                         rate_mean=float(np.mean(fits)), rate_std=float(np.std(fits)),
                         rates=[float(x) for x in fits], a1a0=[float(x) for x in a1a0]))
        print(f"sig={sigma:4.2f} rho={rho:5.3f} eta_th={eta_th:.3f} "
              f"rate={np.mean(fits):.3f}+/-{np.std(fits):.3f} pinning={attn_gain_rel:.4f}")
    json.dump(dict(setup=dict(LAM=LAM, N=N, D=D, T=T, seeds=SEEDS, steps=STEPS,
                              model="position-only attention"), rows=rows),
              open("../out/warp_curve.json", "w"), indent=2)
    make_plot(rows)
    print(f"done {time.time()-t0:.0f}s")


def make_plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator
    rho = np.array([r["rho"] for r in rows])
    eta_th = np.array([r["eta_theory"] for r in rows])
    rate = np.array([r["rate_mean"] for r in rows])
    err = np.array([r["rate_std"] for r in rows])
    pin = np.array([r["attn_gain_rel"] for r in rows])
    lam = rows[0]["lam"]
    # dense theory curve
    from reset import eta_of
    rg = np.geomspace(rho.min()*0.8, rho.max()*1.2, 200)
    eta_curve = np.array([eta_of(lam, r) for r in rg])

    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    # lambda (belief rate) reference
    ax.axhline(lam, ls="--", lw=1.6, color="#c0392b", zorder=1)
    ax.text(rg.min()*1.05, lam+0.012, r"$\lambda=0.7$ (belief rate)", color="#c0392b", fontsize=11)
    # eta(rho) warp theory
    ax.plot(rg, eta_curve, "-", lw=2.4, color="#2c6fbb", zorder=2,
            label=r"theory  $\eta(\rho)$  (palindromic root)")
    # measured points, colored by pinning
    sc = ax.scatter(rho, rate, c=pin, s=90, cmap="viridis", zorder=4,
                    edgecolor="k", linewidth=0.6, norm=matplotlib.colors.LogNorm())
    ax.errorbar(rho, rate, yerr=err, fmt="none", ecolor="#555", elinewidth=1.1,
                capsize=3, zorder=3)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("attention loss share (pinning)", fontsize=10)
    ax.set_xscale("log")
    ax.set_xlabel(r"observation noise-to-signal  $\rho=\nu/A$", fontsize=12)
    ax.set_ylabel("attention lag-kernel rate", fontsize=12)
    ax.set_ylim(0, 0.78)
    ax.set_title("Trained attention rate tracks the warp $\\eta(\\rho)$, not $\\lambda$\n"
                 "(reset process, $\\lambda=0.7$, position-only attention, 3 seeds)",
                 fontsize=12.5)
    ax.legend(loc="lower right", frameon=False, fontsize=11)
    fig.tight_layout()
    fig.savefig("../out/warp_curve.png", dpi=150)
    print("wrote ../out/warp_curve.png")


if __name__ == "__main__":
    main()
