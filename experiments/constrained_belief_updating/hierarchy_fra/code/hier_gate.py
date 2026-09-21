"""H4 Phase-0 loss gate: attn-only on Setting H vs exact-filter + lag-only anchors.

Anchors (MSE on next observation a_{t+1}, per-a, exact via the joint filter over
sampled trajectories):
  (i)  Bayes-filter floor: E[a_{t+1}] from the exact joint posterior (=current state
       in clean obs => uses parent to predict child).
  (ii) lag-only-additive floor: best predictor whose CHILD component is a fixed
       geometric lag kernel on the child's own observations (NO parent gating) +
       linear readout.  Δ_gate = (ii) - (i) = the gating value; must be ~0 on the
       null control.
Then train an attn-only model (content QK + pos_key) and check it beats (ii),
i.e. it content-gates. Writes out/gate_<tag>.json incrementally.
"""
import sys, os, json, argparse, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bernoulli_fra", "code"))
from hier_data import HierProcess
from hier import joint_T, stationary, A, B, C
from model import OneLayerAttn, train, eval_loss
OUT = os.path.join(os.path.dirname(__file__), "..", "out")
torch.set_num_threads(4)


def bayes_child_pred(gen, ZP, ZC):
    """Exact one-step child prob P(z_C(t+1)=1 | state(t))."""
    if gen.T is not None:                                 # gated: via joint T
        st = np.where(ZP == 0, A, np.where(ZC == 1, C, B)).astype(int)
        return gen.T[st, C]
    return gen.p_C + gen.lam_C * (ZC - gen.p_C)           # null: own reset chain


def bayes_parent_pred(gen, ZP, ZC):
    if gen.T is not None:
        st = np.where(ZP == 0, A, np.where(ZC == 1, C, B)).astype(int)
        return gen.T[st, B] + gen.T[st, C]
    return gen.p_P + gen.lam_P * (ZP - gen.p_P)


def mse_floor_bayes(gen, B_=40000, T=24):
    """Per-a MSE of the exact-filter predictor of a_{t+1} (uses parent for child)."""
    ZP, ZC = gen.sample_z(B_, T + 1)
    gP = gen.mu + gen.sigma * gen.rng.standard_normal((B_, T + 1))
    gC = gen.mu + gen.sigma * gen.rng.standard_normal((B_, T + 1))
    cP = ZP * np.maximum(gP, 0); cC = ZC * np.maximum(gC, 0)
    # predictor of c_x(t+1) = m1 * P(z_x(t+1)=1 | state(t))
    Ppred = bayes_parent_pred(gen, ZP[:, :T], ZC[:, :T]) * gen.m1
    Cpred = bayes_child_pred(gen, ZP[:, :T], ZC[:, :T]) * gen.m1
    seP = (cP[:, 1:T + 1] - Ppred) ** 2
    seC = (cC[:, 1:T + 1] - Cpred) ** 2
    return float(seP.mean()), float(seC.mean())


def mse_floor_lagonly(gen, B_=40000, T=24, taumax=40):
    """Best predictor whose CHILD prediction is a geometric lag kernel on the
    child's OWN observations only (parent-blind) + optimal gain; parent prediction
    uses its own optimal reset kernel. This is the lag-only-additive constrained
    floor. We optimise the child rate + gain by 1-D scan on the exact objective."""
    ZP, ZC = gen.sample_z(B_, T + 1)
    gC = gen.mu + gen.sigma * gen.rng.standard_normal((B_, T + 1))
    cC = ZC * np.maximum(gC, 0)
    oC = (cC[:, :T] > 1e-9).astype(float)                 # child fired indicator
    yC = cC[:, 1:T + 1]                                   # target
    pbar = oC.mean()
    best = 1e9
    for eta in np.linspace(0.0, 0.95, 40):
        # feature: geometric-weighted history of child firings
        ker = eta ** np.arange(T)
        # x[b,t] = sum_{s<=t} eta^{t-s} oC[b,s]
        x = np.zeros((B_, T))
        acc = np.zeros(B_)
        for t in range(T):
            acc = eta * acc + oC[:, t]
            x[:, t] = acc
        # linear fit yC ~ a*x + c  (parent-blind)
        X = np.stack([x.ravel(), np.ones(x.size)], 1)
        coef, *_ = np.linalg.lstsq(X, yC.ravel(), rcond=None)
        pred = X @ coef
        mse = ((yC.ravel() - pred) ** 2).mean()
        best = min(best, mse)
    return float(best)


def build(gen, d, nlayers, nheads, seed, T):
    m = OneLayerAttn(d, d_h=d, bias=True, init=0.02, seed=seed, pos_key=True, n_ctx=T)
    return m


@torch.no_grad()
def component_mse(m, gen, T, B_=8000):
    """Per-feature MSE of the model's a_{t+1} prediction, projected onto d_P and d_C."""
    X, Y, _ = gen.sample_seq(B_, T)
    Xt = torch.tensor(X, dtype=torch.float32); Yt = torch.tensor(Y, dtype=torch.float32)
    pred = m(Xt).numpy(); Yn = Y
    dP = gen.D[0]; dC = gen.D[1]
    eP = ((pred - Yn) @ dP) ** 2
    eC = ((pred - Yn) @ dC) ** 2
    return float(eP.mean()), float(eC.mean())


def train_posonly(gen, d, T, steps, seed=0):
    m = OneLayerAttn(d, d_h=d, bias=True, init=0.02, seed=seed, pos_key=True, n_ctx=T)
    with torch.no_grad():
        m.Wq.zero_(); m.Wk.zero_()
    m.Wq.requires_grad_(False); m.Wk.requires_grad_(False)     # lag-only (content-blind) pattern
    train(m, gen, T=T, steps=steps, batch=256, lr=3e-3)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="H_gated")
    ap.add_argument("--gated", type=int, default=1)
    ap.add_argument("--sigma", type=float, default=0.6)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--d", type=int, default=32)
    args = ap.parse_args()
    d, T = args.d, 24
    gen = HierProcess(d=d, sigma=args.sigma, gated=bool(args.gated), seed=0)
    jpath = os.path.join(OUT, f"gate_{args.tag}.json")
    rec = dict(tag=args.tag, gated=bool(args.gated), sigma=args.sigma, status="running")
    json.dump(rec, open(jpath, "w"), indent=2)
    _, mseC_bayes_oracle = mse_floor_bayes(gen, T=T)          # state-oracle reference
    # content-gated (full QK) model
    mfull = build(gen, d, 1, 1, 0, T)
    train(mfull, gen, T=T, steps=args.steps, batch=256, lr=3e-3)
    eP_full, eC_full = component_mse(mfull, gen, T)
    # lag-only (position-only) model
    mpos = train_posonly(gen, d, T, args.steps)
    eP_pos, eC_pos = component_mse(mpos, gen, T)
    rec.update(status="done",
               child_mse_full=eC_full, child_mse_posonly=eC_pos,
               parent_mse_full=eP_full, parent_mse_posonly=eP_pos,
               delta_gate_empirical=eC_pos - eC_full,          # content-gating value (child)
               parent_collateral=eP_pos - eP_full,             # should be ~0 (parent is lag-only)
               child_mse_bayes_oracle=mseC_bayes_oracle,
               model=f"model_{args.tag}.pt")
    torch.save({"state_dict": mfull.state_dict(),
                "cfg": dict(d=d, T=T, sigma=args.sigma, gated=bool(args.gated))},
               os.path.join(OUT, f"model_{args.tag}.pt"))
    json.dump(rec, open(jpath, "w"), indent=2)
    print(json.dumps({k: round(rec[k], 5) if isinstance(rec[k], float) else rec[k]
                      for k in ["tag", "gated", "child_mse_full", "child_mse_posonly",
                                "delta_gate_empirical", "parent_collateral"]}, indent=2))


if __name__ == "__main__":
    main()
