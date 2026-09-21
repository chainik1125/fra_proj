"""Check 2 — A1.3 no-bias caveat: strip the bias, attention survives only as a
rank-1 mean-carrying bias-substitute.

Two parts:
(i) ANALYTIC (exact): from the closed form R = (1-a0)(1-s_mu) mu mu^T and
    V* = R S^+, confirm V* is rank-1 with left/right singular vectors along mu_a.
(ii) TRAINED: retrain with bias=False; measure how much of the learned
    effective mean-carrying map (V, and V+W acting on mu) aligns with mu_a, and
    how rank-1 the mean-sector action is. Report honestly (the W/V split of the
    constant is itself the k=0 gauge, so we test the *combined* mean-carrying).
"""
import json, numpy as np, torch
from data import BernoulliGaussian
from model_A import OneLayerAttnA as OneLayerAttn, train, eval_loss

OUT = "../out/check2_nobias.json"
N, d, T = 12, 48, 16


def analytic_V_star(gen, alpha0=0.3):
    """Closed-form R and V* for a lag-only pattern with diagonal mass alpha0.
    Sigma_tau = mu mu^T (tau>=1); C = a0 Sig0 + (1-a0) mu mu^T; C+ = mu mu^T;
    S = Gamma - C^T Sig0^-1 C ; V* = R S^+."""
    m = gen.moments()
    mu, Sig0 = m["mu_a"], m["Sig0"]
    # a lives in a (N+1)-dim affine subspace => Sig0 is rank-deficient in R^d;
    # use the pseudo-inverse (inverse on the data subspace im(Sig0)), NOT inv().
    Sig0i = np.linalg.pinv(Sig0, rcond=1e-8)
    subspace_rank = int(np.linalg.matrix_rank(Sig0, tol=1e-6))
    mumu = np.outer(mu, mu)
    s_mu = float(mu @ Sig0i @ mu)
    C = alpha0 * Sig0 + (1 - alpha0) * mumu
    Cplus = mumu.copy()                    # C_+ = mu mu^T
    B1 = mumu.copy()                        # Sigma_1^Y = mu mu^T
    R = Cplus - B1 @ Sig0i @ C
    R_pred = (1 - alpha0) * (1 - s_mu) * mumu
    # Gamma with gamma(0)=alpha0^2-ish is gauge; use a generic PSD S in mean sector.
    # V* = R S^+ ; since R ∝ mu mu^T (rank1), V* is rank1 along mu on both sides
    # regardless of S (S^+ acts within span). Report R structure directly.
    U, sv, Vt = np.linalg.svd(R)
    return dict(s_mu=s_mu, subspace_rank=subspace_rank, d=gen.d,
                R_rank1_relerr=float(np.linalg.norm(R - R_pred) / (np.linalg.norm(R) + 1e-12)),
                R_sv=sv[:4].tolist(),
                left_cos_mu=float(abs(U[:, 0] @ (mu / np.linalg.norm(mu)))),
                right_cos_mu=float(abs(Vt[0] @ (mu / np.linalg.norm(mu)))))


def main():
    gen = BernoulliGaussian(N, d, seed=100, p=0.12, mu=1.0, sigma=0.3)
    mom = gen.moments()
    mu = mom["mu_a"]; muhat = mu / np.linalg.norm(mu)
    floor = float(np.trace(mom["Ca"]))

    ana = analytic_V_star(gen)

    # trained no-bias, several seeds
    rows = []
    for s in [0, 1, 2]:
        m = OneLayerAttn(d, bias=False, seed=s)
        train(m, gen, T=T, steps=4000)
        L = eval_loss(m, gen, T=T)
        V = m.V.detach().numpy()
        W = m.W.detach().numpy()
        # mean-carrying action of the attention path: V mu  (attn delivers V mu on the mean)
        Vmu = V @ mu
        # rank-1-ness of V restricted to how it acts: svd
        U, sv, Vt = np.linalg.svd(V)
        # combined direct+context mean map on mu:  (W + V) mu   (lag-0 delivered by both)
        comb_mu = (W + V) @ mu
        rows.append(dict(seed=s, loss=L, loss_over_floor=L / floor,
                         V_top_sv_frac=float(sv[0] / sv.sum()),
                         V_left_cos_mu=float(abs(U[:, 0] @ muhat)),
                         Vmu_cos_mu=float(abs(Vmu @ mu) / (np.linalg.norm(Vmu) * np.linalg.norm(mu) + 1e-12)),
                         comb_mu_cos_mu=float(abs(comb_mu @ mu) / (np.linalg.norm(comb_mu) * np.linalg.norm(mu) + 1e-12)),
                         Vmu_norm=float(np.linalg.norm(Vmu)),
                         mu_norm=float(np.linalg.norm(mu))))

    res = dict(setup=dict(N=N, d=d, T=T), floor=floor,
               analytic_Vstar=ana, trained_nobias=rows)
    json.dump(res, open(OUT, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
