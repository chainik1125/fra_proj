"""Check 4: B3 intervention calculus on a trained 2-feature model.

(i)  Sever feature 1's OV channel at gain c -> feature-1 tracking falls linearly,
     null at c* = 1/rho_1 (ex-ante from clean path share).
(ii) Zero collateral on feature 2 at rho_mm=0 (orthogonal d_i); collateral
     ~ (d_1.d_2)^2 under controlled superposition.
Writes out/interv.json.
"""
import json, numpy as np, torch
from reset import ResetProcess, bayes_beliefs
from model import OneLayerAttn, train

D, T = 32, 28
LAMS = [0.5, 0.85]
SIGMA = 1.0                        # noisy enough that attention carries belief
P = 0.3
STEPS = 3000


def overlap_dict(rho, d=D, seed=7):
    """d_1, d_2 with d_1.d_2 = rho (unit norm), rest random-orthogonal filler."""
    rng = np.random.default_rng(seed)
    e1 = np.zeros(d); e1[0] = 1.0
    e2 = np.zeros(d); e2[1] = 1.0
    d1 = e1
    d2 = rho * e1 + np.sqrt(1 - rho**2) * e2
    return np.stack([d1, d2])


def readouts(model, X, cX, D_np, sever_idx=None, gain=0.0, b_vec=None):
    """Model output; optional FRA-OV sever of feature sever_idx at gain via the
    d_i-readout (realistic edit). Returns feature-readout r_j(t)=d_j^T(out) (B,T,2)."""
    Xt = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        A = model.pattern(Xt).mean(1)                 # B x T x s
        Vx = Xt @ model.V.T                           # B x s x d
        ctx = torch.einsum('bts,bsd->btd', A, Vx)
        if sever_idx is not None and gain != 0.0:
            di = torch.tensor(D_np[sever_idx], dtype=torch.float32)
            b_t = torch.tensor(b_vec, dtype=torch.float32)
            yhat = (Xt - b_t) @ di                    # B x s  readout of feature i
            transported = torch.einsum('bts,bs->bt', A, yhat)   # B x T
            ctx = ctx - gain * transported[..., None] * (di @ model.V.T)
        out = Xt @ model.W.T + ctx + (model.beta if model.beta is not None else 0.0)
        Dt = torch.tensor(D_np, dtype=torch.float32)
        r = out @ Dt.T                                # B x T x 2
    return r.numpy()


def track_slope(r_j, belief_j):
    """slope of model readout r_j(t) on oracle belief belief_j(t) (both flattened)."""
    x = belief_j.ravel(); y = r_j.ravel()
    x = x - x.mean()
    return float((x @ (y - y.mean())) / (x @ x))


def run_rho(rho_mm):
    Dnp = overlap_dict(rho_mm)
    g = ResetProcess(2, D, lam=LAMS, p=P, mu=1.0, sigma=SIGMA, seed=0, D=Dnp)
    m = OneLayerAttn(D, d_h=D, bias=True, init=0.02, seed=0, pos_key=True, n_ctx=T)
    train(m, g, T=T, steps=STEPS, batch=256, lr=3e-3)
    X, _, cX = g.sample_seq(6000, T)
    Phat = bayes_beliefs(g, np.concatenate([cX, cX[:, -1:, :]], axis=1))[:, :T, :]
    B0 = Phat - g.p                                   # belief displacement
    # clean slopes
    r0 = readouts(m, X, cX, Dnp)
    s1_0 = track_slope(r0[:, :, 0], B0[:, :, 0])
    s2_0 = track_slope(r0[:, :, 1], B0[:, :, 1])
    # sever feature 1 across gains
    gains = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    s1c, s2c = [], []
    for c in gains:
        r = readouts(m, X, cX, Dnp, sever_idx=0, gain=c, b_vec=g.b)
        s1c.append(track_slope(r[:, :, 0], B0[:, :, 0]) / s1_0)
        s2c.append(track_slope(r[:, :, 1], B0[:, :, 1]) / s2_0)
    s1c = np.array(s1c); s2c = np.array(s2c)
    # rho_1 from slope of feature-1 tracking vs gain; c* = 1/rho_1
    rho1 = -np.polyfit(gains, s1c, 1)[0]              # s1c ~ 1 - rho1*c
    cstar = 1.0 / rho1
    collateral2 = float(np.max(np.abs(s2c - 1.0)))    # worst feature-2 deviation
    return dict(rho_mm=rho_mm, d1d2=float(Dnp[0] @ Dnp[1]),
                slope1_clean=s1_0, slope2_clean=s2_0, gains=gains,
                track1=[float(x) for x in s1c], track2=[float(x) for x in s2c],
                rho1_measured=float(rho1), cstar_measured=float(cstar),
                collateral2=collateral2, eta1=float(g.const['eta'][0]))


def main():
    res = [run_rho(r) for r in [0.0, 0.2, 0.4]]
    # collateral ~ (d1.d2)^2 check
    base = res[0]["collateral2"]
    for r in res:
        r["collateral2_over_d1d2sq"] = (r["collateral2"] /
                                        (r["d1d2"]**2) if r["d1d2"] != 0 else None)
        print(f"rho_mm={r['rho_mm']} d1.d2={r['d1d2']:.2f} "
              f"collateral2={r['collateral2']:.4f} "
              f"c*={r['cstar_measured']:.3f} (null gain) "
              f"track1@c=1={r['track1'][4]:.3f}")
    json.dump(dict(setup=dict(LAMS=LAMS, SIGMA=SIGMA, P=P, D=D, T=T, steps=STEPS),
                   results=res), open("../out/interv.json", "w"), indent=2)
    print("wrote ../out/interv.json")


if __name__ == "__main__":
    main()
