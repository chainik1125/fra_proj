"""Checks 1-3 diagnostic sweep: warp curve + clean-null + parallelism tax.

For a grid of observation noise (sigma), fixed lambda=0.7, N features:
 - analytic floors per feature (prior / direct-only / additive@eta / additive@lambda / bayes)
 - train full model; record MSE, mean-lag rate, non-diag mass
 - the 'pinning' = (direct_only - additive@eta) : how load-bearing attention is
Writes out/warp_sweep.json.
"""
import json, time, numpy as np, torch
from scipy.linalg import toeplitz
from reset import (ResetProcess, feature_constants, eta_of,
                   geometric_rate_fit, bayes_filter_mse, additive_floor_mse)
from model import OneLayerAttn, train, eval_loss

OUT = "../out/warp_sweep.json"
LAM, N, D, T = 0.7, 3, 32, 28
SIGMAS = [0.08, 0.15, 0.3, 0.5, 0.8, 1.3, 2.2, 3.6]
SEEDS = [0, 1]
STEPS = 3500


def per_feature_floors(g, taumax=80):
    """prior, direct-only, additive@eta_opt, additive@lambda (summed over feats)."""
    def pmse(G, eta):
        a = eta ** np.arange(1, taumax + 1)
        Gm = toeplitz(G)
        Sig = np.array([[G[0], a @ G[1:taumax + 1]],
                        [a @ G[1:taumax + 1], a @ Gm[1:, 1:] @ a]])
        cv = np.array([G[1], a[:taumax - 1] @ G[2:taumax + 1]])
        return G[0] - cv @ np.linalg.solve(Sig, cv)
    prior = dir_only = add_eta = add_lam = 0.0
    for i in range(g.N):
        A, nu, lam, eta = g.const['A'][i], g.const['nu'][i], g.lam[i], g.const['eta'][i]
        G = A * lam ** np.arange(taumax + 1); G[0] = A + nu
        prior += G[0]
        dir_only += G[0] - G[1] ** 2 / G[0]
        add_eta += pmse(G, eta)
        add_lam += pmse(G, lam)
    return dict(prior=float(prior), direct_only=float(dir_only),
                additive_eta=float(add_eta), additive_lambda=float(add_lam))


def main():
    t0 = time.time()
    rows = []
    for sigma in SIGMAS:
        g0 = ResetProcess(N, D, lam=LAM, p=0.3, mu=1.0, sigma=sigma, seed=0,
                          orthogonalize=True)
        rho = float(g0.const['rho'][0]); eta_th = float(g0.const['eta'][0])
        fl = per_feature_floors(g0)
        bayes = bayes_filter_mse(g0, B=15000, T=T)['mse_bayes']
        trained, rate15, rate26, ndmass = [], [], [], []
        for s in SEEDS:
            g = ResetProcess(N, D, lam=LAM, p=0.3, mu=1.0, sigma=sigma, seed=s,
                             orthogonalize=True, D=g0.D)
            m = OneLayerAttn(D, d_h=D, bias=True, init=0.02, seed=s,
                             pos_key=True, n_ctx=T)
            train(m, g, T=T, steps=STEPS, batch=256, lr=3e-3)
            trained.append(eval_loss(m, g, T=T, B=8000))
            X, _, _ = g.sample_seq(3000, T)
            al = m.mean_lag(torch.tensor(X, dtype=torch.float32), t_lo=10)
            rate15.append(geometric_rate_fit(al, 1, 5))
            rate26.append(geometric_rate_fit(al, 2, 6))
            ndmass.append(float(al[1:].sum() / al.sum()))
        row = dict(sigma=sigma, rho=rho, eta_theory=eta_th, lam=LAM,
                   floors=fl, bayes=bayes,
                   trained_mse_mean=float(np.mean(trained)),
                   trained_mse_std=float(np.std(trained)),
                   rate_fit_1_5=[float(x) for x in rate15],
                   rate_fit_2_6=[float(x) for x in rate26],
                   nondiag_mass=[float(x) for x in ndmass],
                   attn_gain=float(fl['direct_only'] - fl['additive_eta']),
                   lam_penalty=float(fl['additive_lambda'] - fl['additive_eta']))
        rows.append(row)
        print(f"sig={sigma:4.2f} rho={rho:5.3f} eta_th={eta_th:5.3f} "
              f"| trained={row['trained_mse_mean']:.4f} add@eta={fl['additive_eta']:.4f} "
              f"add@lam={fl['additive_lambda']:.4f} dir={fl['direct_only']:.4f} "
              f"| attn_gain={row['attn_gain']:.4f} lam_pen={row['lam_penalty']:.5f} "
              f"rate15={np.mean(rate15):.3f}")
    json.dump(dict(setup=dict(LAM=LAM, N=N, D=D, T=T, seeds=SEEDS, steps=STEPS),
                   rows=rows), open(OUT, "w"), indent=2)
    print(f"\ndone in {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
