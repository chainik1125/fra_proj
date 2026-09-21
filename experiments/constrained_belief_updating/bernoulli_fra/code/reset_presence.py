"""Check 6 — the PRESENCE question: can FRA cut GT-feature presence?

fra_hmm_toy methodology: a ridge probe RETRAINED from scratch on the post-attention
residual at position d (resid_post = a_d + ctx_d), target = the Bayes belief about
feature i, P(z_i(t+1)=1 | y_{1:t}). Conditions on the same trained noisy reset model:
  (a) clean;
  (b) sever feature-i OV, lags k>=1 only, gain 1  (keep k=0 diagonal + current token);
  (c) sever feature-i OV, ALL lags incl k=0 diagonal, gain 1;
  (d) counter-steer: sever at gain c* (the interv null gain).
Predictions: (b) probe falls partway; (c) probe falls to the SINGLE-OBSERVATION
FLOOR (belief R^2 given only the current token, computed analytically + measured);
(d) probe ~ clean (nulling behaviour relocates, does not delete presence).
Writes out/presence.json.
"""
import json, numpy as np, torch
from reset import ResetProcess, bayes_beliefs
from model import OneLayerAttn, train

D, T = 32, 28
LAMS = [0.5, 0.85]
SIGMA, P = 1.0, 0.3
FEAT = 0                      # sever & probe feature 0 (matches interv.json sever_idx)
CSTAR = 2.91                  # interv.json rho_mm=0 null gain for feature 0
T_LO = 8                      # mid positions only (avoid boundary)


def resid_post(model, X, b, gain=0.0, lags="none"):
    """resid_post = a_d + ctx_d, with optional FRA-OV sever of feature FEAT."""
    Xt = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        A = model.pattern(Xt).mean(1)                     # B x T x s
        ctx = torch.einsum('bts,bsd->btd', A, Xt @ model.V.T)
        if gain != 0.0:
            di = torch.tensor(np.asarray(model_D[FEAT]), dtype=torch.float32)
            bt = torch.tensor(b, dtype=torch.float32)
            yhat = (Xt - bt) @ di                         # B x s = c_FEAT(s)
            Ae = A.clone()
            if lags == "ge1":                             # drop diagonal s=d
                idx = torch.arange(A.shape[1])
                Ae[:, idx, idx] = 0.0
            transported = torch.einsum('bts,bs->bt', Ae, yhat)     # B x T
            ctx = ctx - gain * transported[..., None] * (di @ model.V.T)
        resid = Xt + ctx                                  # B x T x d
    return resid.numpy()


def ridge_r2(Xtr, ytr, Xte, yte, alpha=1.0):
    mx, my = Xtr.mean(0), ytr.mean()
    Xc, Xtc = Xtr - mx, Xte - mx
    w = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(Xc.shape[1]), Xc.T @ (ytr - my))
    pred = Xtc @ w + my
    ss_res = ((yte - pred) ** 2).sum()
    ss_tot = ((yte - yte.mean()) ** 2).sum()
    return float(1 - ss_res / ss_tot)


def flatten_mid(arr):
    return arr[:, T_LO:, ...].reshape(-1, arr.shape[-1]) if arr.ndim == 3 \
        else arr[:, T_LO:].reshape(-1)


def main():
    global model_D
    g = ResetProcess(2, D, lam=LAMS, p=P, mu=1.0, sigma=SIGMA, seed=0, orthogonalize=True)
    model_D = g.D
    rho_obs = float(g.const['rho'][FEAT]); eta = float(g.const['eta'][FEAT])
    m = OneLayerAttn(D, d_h=D, bias=True, init=0.02, seed=0, pos_key=True, n_ctx=T)
    train(m, g, T=T, steps=3000, batch=256, lr=3e-3)

    # eval data + Bayes belief target for feature FEAT
    Xtr, _, cXtr = g.sample_seq(6000, T)
    Xte, _, cXte = g.sample_seq(6000, T)
    def belief(cX):
        return bayes_beliefs(g, np.concatenate([cX, cX[:, -1:, :]], axis=1))[:, :T, FEAT]
    ytr = flatten_mid(belief(cXtr)); yte = flatten_mid(belief(cXte))

    conds = {
        "a_clean":       dict(gain=0.0, lags="none"),
        "b_sever_ge1":   dict(gain=1.0, lags="ge1"),
        "c_sever_all":   dict(gain=1.0, lags="all"),
        "d_counter_cstar": dict(gain=CSTAR, lags="all"),
    }
    r2 = {}
    for name, kw in conds.items():
        Rtr = flatten_mid(resid_post(m, Xtr, g.b, **kw))
        Rte = flatten_mid(resid_post(m, Xte, g.b, **kw))
        r2[name] = ridge_r2(Rtr, ytr, Rte, yte)

    # single-observation floor: probe on the current token a_d ALONE (= what (c) leaves)
    r2_floor_meas = ridge_r2(flatten_mid(Xtr), ytr, flatten_mid(Xte), yte)
    # analytic linear floor: Corr(belief, c_FEAT(d))^2 (current value only)
    cval_tr = flatten_mid(cXtr[:, :, FEAT:FEAT+1])
    cval_te = flatten_mid(cXte[:, :, FEAT:FEAT+1])
    r2_floor_analytic = float(np.corrcoef(flatten_mid(belief(cXte)), cval_te.ravel())[0, 1] ** 2)

    res = dict(setup=dict(lams=LAMS, sigma=SIGMA, feat=FEAT, rho_obs=rho_obs, eta=eta,
                          cstar=CSTAR, T=T, t_lo=T_LO),
               probe_r2=r2,
               single_obs_floor_measured=r2_floor_meas,
               single_obs_floor_analytic_corr2=r2_floor_analytic)
    json.dump(res, open("../out/presence.json", "w"), indent=2)
    print(json.dumps(res, indent=2))
    print("\n(a)clean=%.3f  (b)k>=1=%.3f  (c)all=%.3f  (d)c*=%.3f | floor meas=%.3f analytic=%.3f"
          % (r2['a_clean'], r2['b_sever_ge1'], r2['c_sever_all'], r2['d_counter_cstar'],
             r2_floor_meas, r2_floor_analytic))


if __name__ == "__main__":
    main()
