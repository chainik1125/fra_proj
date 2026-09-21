"""Check 3 — A1.4: optimal within-position read-out is LMMSE
  T = Sigma_c D Ca^+ D^T ,  Ca = D^T Sigma_c D.
Predictions:
 - complete regime N<=d, full row rank: T = I (exact recovery, no shrinkage).
 - superposed regime N>d: rank(T)=d<N, diagonal shrinkage T_ii<1 and
   off-diagonal contamination; matched filter hat_c^MF = G c exactly (contam = G_ij).
We fit OLS c~a on a large sample (= empirical LMMSE) and compare to the closed form.
"""
import json, numpy as np
from data import BernoulliGaussian, gram_offdiag_stats


def lmmse_T(gen, B=600_000):
    a, c = gen.sample_a(B)
    mu_a = a.mean(0); cbar = c.mean(0)
    ac = a - mu_a; cc = c - cbar
    Ca = (ac.T @ ac) / B
    Cca = (cc.T @ ac) / B                    # Cov(c,a)  N x d
    R_emp = Cca @ np.linalg.pinv(Ca)         # empirical LMMSE readout  N x d
    T_emp = R_emp @ gen.D.T                   # N x N transfer
    # analytic
    Sc = (cc.T @ cc) / B
    Ca_an = gen.D.T @ Sc @ gen.D             # d x d  (D is N x d so D^T is d x N)
    R_an = Sc @ gen.D @ np.linalg.pinv(Ca_an)
    T_an = R_an @ gen.D.T
    # matched filter exactness: G c vs (a-b) read by D
    G = gen.D @ gen.D.T
    mf = (a - gen.b) @ gen.D.T               # B x N  == c @ G
    mf_err = float(np.linalg.norm(mf - c @ G) / np.linalg.norm(c @ G))
    # R is only defined up to the data null-space (a lives in a <=N-dim subspace);
    # compare its ACTION on sampled activations, not the raw matrix.
    act_err = float(np.linalg.norm(ac @ (R_emp - R_an).T) / np.linalg.norm(ac @ R_an.T))
    Toff = T_emp[~np.eye(gen.N, dtype=bool)]
    Goff = G[~np.eye(gen.N, dtype=bool)]
    return dict(
        N=gen.N, d=gen.d, gram=gram_offdiag_stats(gen.D),
        T_diag_mean=float(np.diag(T_emp).mean()), T_diag_min=float(np.diag(T_emp).min()),
        T_offdiag_absmean=float(np.abs(Toff).mean()),
        T_emp_vs_analytic_relerr=float(np.linalg.norm(T_emp - T_an) / np.linalg.norm(T_an)),
        R_action_emp_vs_analytic_relerr=act_err,
        pred_T_diag_mean_d_over_N=float(min(1.0, gen.d / gen.N)),
        T_minus_I_relerr=float(np.linalg.norm(T_emp - np.eye(gen.N)) / np.sqrt(gen.N)),
        matched_filter_eq_Gc_relerr=mf_err,
        corr_Toffdiag_vs_negGram=float(np.corrcoef(Toff, -Goff)[0, 1]),
    )


def main():
    res = {}
    # complete: N<=d, near-orthogonal
    g1 = BernoulliGaussian(10, 40, seed=7, p=0.15, mu=1.0, sigma=0.3)
    res["complete_N10_d40"] = lmmse_T(g1)
    # complete but with real overlap (N=d): still exact recovery iff full rank
    g2 = BernoulliGaussian(24, 24, seed=8, p=0.15, mu=1.0, sigma=0.3)
    res["square_N24_d24"] = lmmse_T(g2)
    # overcomplete / superposition: N>d
    g3 = BernoulliGaussian(48, 24, seed=9, p=0.15, mu=1.0, sigma=0.3)
    res["overcomplete_N48_d24"] = lmmse_T(g3)
    g4 = BernoulliGaussian(80, 24, seed=10, p=0.15, mu=1.0, sigma=0.3)
    res["overcomplete_N80_d24"] = lmmse_T(g4)
    json.dump(res, open("../out/check3_lmmse.json", "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
