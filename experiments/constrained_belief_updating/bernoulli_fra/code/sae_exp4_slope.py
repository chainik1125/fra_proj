"""Experiment 4 (addendum) — collateral scaling exponent vs d_i.d_j, matched vs learned.

Prediction (OPTIMAL_FRA_NOTE §5(iii)): severing feature i and reading feature j, the part
of j's readout perturbation proportional to c_j (j's own activity -- the double-overlap
collateral) scales as
    (d_i.d_j)^2  with a MATCHED code (encoder=decoder=GT dictionary, chi=I)   -> slope ~2
    (d_i.d_j)^1  with a learned SAE code (chi != I, one overlap leg replaced)  -> slope ~1
A first-order cancellation that survives only for the matched code.

Measured beta_ij = Cov(Delta chat_j, c_j)/Var(c_j) over all ordered pairs i!=j, in a
near-orthogonal superposed regime (random non-orthogonal D, rho_mm~0.45); fit log|beta| vs
log|G_ij|.
"""
import json, time, sys, os
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
from sae_common import CorrHierGen, train_sae, mcc_hungarian

OUT = os.path.join(os.path.dirname(__file__), "..", "out", "sae_exp4_slope.json")
N, d = 24, 24
B = 300_000


def collateral_betas(zsev, dj_dot_w, c):
    """zsev: (B,) latent activation removed when severing i.
    dj_dot_w: (N,) = d_j . w_ell for the removed latent's decoder direction.
    Returns beta_j = Cov(Delta chat_j, c_j)/Var(c_j) for all j."""
    Dcj = -zsev[:, None] * dj_dot_w[None, :]           # (B,N) readout change of each j
    cc = c - c.mean(0)
    dd = Dcj - Dcj.mean(0)
    cov = (dd * cc).mean(0)
    var = (cc * cc).mean(0)
    return cov / np.clip(var, 1e-12, None)


def fit_slope(gabs, beta_abs):
    m = (gabs > 1e-3) & (beta_abs > 1e-9)
    x, y = np.log(gabs[m]), np.log(beta_abs[m])
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    r = np.corrcoef(x, y)[0, 1]
    return float(slope), float(r), int(m.sum())


def run():
    t0 = time.time()
    gen = CorrHierGen(N, d, seed=3, p=0.06, mu=1.0, sigma=0.5, orthogonalize=False)
    D = gen.D
    G = D @ D.T
    rho = float(np.mean([np.max(np.abs(G[i][np.arange(N) != i])) for i in range(N)]))
    a, c = gen.sample_a(B)
    b = gen.b

    # ---- matched/planted code: encoder=decoder=D, chi=I ----
    zmat = np.maximum((a - b) @ D.T, 0.0)              # (B,N) matched-filter latents
    beta_planted = np.full((N, N), np.nan)
    for i in range(N):
        beta_planted[i] = collateral_betas(zmat[:, i], G[i], c)

    # ---- learned TopK SAE ----
    K = max(1, int(round((c > 0).sum(1).mean())))
    sae = train_sae(a, N, K, steps=4000, batch=4096, seed=0)
    with torch.no_grad():
        zl = sae.encode(torch.as_tensor(a, dtype=torch.float32)).numpy()
    W = sae.W_dec.detach().numpy()                      # (N,d)
    _, li, gi = mcc_hungarian(W, D)
    lat_of_gt = {int(g): int(l) for l, g in zip(li, gi)}
    beta_learn = np.full((N, N), np.nan)
    for i in range(N):
        ell = lat_of_gt.get(i)
        if ell is None:
            continue
        beta_learn[i] = collateral_betas(zl[:, ell], W[ell] @ D.T, c)

    # collect off-diagonal pairs
    off = ~np.eye(N, dtype=bool)
    gabs = np.abs(G[off])
    s_p, r_p, n_p = fit_slope(gabs, np.abs(beta_planted[off]))
    bl = beta_learn[off]
    valid = ~np.isnan(bl)
    s_l, r_l, n_l = fit_slope(gabs[valid], np.abs(bl[valid]))

    out = dict(rho_mm=rho, N=N, d=d, K=K,
               planted=dict(slope=s_p, logr=r_p, npairs=n_p,
                            mean_abs_beta=float(np.nanmean(np.abs(beta_planted[off])))),
               learned=dict(slope=s_l, logr=r_l, npairs=n_l,
                            mean_abs_beta=float(np.nanmean(np.abs(bl[valid])))),
               mcc=float(mcc_hungarian(W, D)[0]))
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"rho_mm={rho:.3f}  SAE MCC={out['mcc']:.3f}")
    print(f"PLANTED (chi=I): collateral-slope vs |G_ij| = {s_p:.2f} (logr {r_p:.2f}, n={n_p}) "
          f"mean|beta|={out['planted']['mean_abs_beta']:.2e}")
    print(f"LEARNED (chi!=I): collateral-slope vs |G_ij| = {s_l:.2f} (logr {r_l:.2f}, n={n_l}) "
          f"mean|beta|={out['learned']['mean_abs_beta']:.2e}")
    print("wrote", OUT, f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    run()
