"""Diagnosis rung 2 — the discriminating measurement: two ANALYTIC frontiers.

For each cell (gated/null x occ/clean) compute, from the EXACT process (no training):
  (1) frontier_bayes  : exact-filter child MSE (Bayes floor WITH the gate). Forward
      HMM filter on the occluded process; the occlusion is a clean missing-obs on the
      child coordinate, the parent is (essentially) clean, the ReLU false-negative
      atoms q_P,q_C are ~3e-7 (mu/sig=5) -> the emission is captured EXACTLY by the
      discrete signature (parent-on?, child-positive?). Filter is exact; the MSE is a
      Monte-Carlo average over the process (report SE).
  (2) frontier_lag    : best predictor in the THEOREM's lag-only class = affine in
      {current token} u {content-blind lag-weighted sums of past tokens}. Two forms:
        - lag_rel : ONE shared per-relative-lag linear map (the strict theorem class,
                    A_ds=alpha(d-s)); boundary-truncated at the sequence start.
        - lag_pos : per-DESTINATION-position content-blind linear regression on all
                    causal past tokens (OVER-approximates the theorem class -- gives the
                    predictor absolute-position power too, i.e. >= the trained pos-only
                    model's LINEAR expressivity). Conservative for world-B detection.
      References: lag0 (current-token-only linear) and const (marginal) bracket them.

Child MSE identity used throughout (fresh magnitude g each step, independent of state):
  MSE(pred) = E[(c_C(t+1) - pred)^2],  optimal pred = m1_C * P(z_C(t+1)=1 | info) + b.dC
so every frontier is just "best P(z_C(t+1)=1|info)" in its info class. The Bayes-vs-lag
gap is the TRUE Delta_gate (theorem units), to be compared against H2.2's 0.0189 surrogate.

Writes out/rung2_frontiers.json.
"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bernoulli_fra", "code"))
from hier_data import HierProcess, relu_moments
from hier import joint_T, null_T

OUT = os.path.join(os.path.dirname(__file__), "..", "out")
PARAMS = dict(lam_P=0.5, p_P=0.3, lam_C=0.9, p_C=0.5, mu_P=1.5, sig_P=0.3, mu_C=1.5, sig_C=0.3)
T = 24
B = 120000
EVAL_SEED = 1234
THR = 1e-6


# ---------------- exact forward filter (child one-step posterior) ----------------
def state_decomp(gated):
    """Return list of (zP,zC) per state index, child mask, transition, init dist."""
    if gated:
        decomp = [(0, 0), (1, 0), (1, 1)]            # A,B,C
        Tm = joint_T(PARAMS["lam_P"], PARAMS["p_P"], PARAMS["lam_C"], PARAMS["p_C"])
        pP, pC = PARAMS["p_P"], PARAMS["p_C"]
        init = np.array([1 - pP, pP * (1 - pC), pP * pC])
    else:
        decomp = [(0, 0), (0, 1), (1, 0), (1, 1)]    # kron(TP,TC) lexicographic
        Tm = null_T(PARAMS["lam_P"], PARAMS["p_P"], PARAMS["lam_C"], PARAMS["p_C"])
        pP, pC = PARAMS["p_P"], PARAMS["p_C"]
        init = np.array([(1 - pP) * (1 - pC), (1 - pP) * pC, pP * (1 - pC), pP * pC])
    childmask = np.array([zc for (_, zc) in decomp], float)   # zC of the NEXT state
    return decomp, Tm, init, childmask


def emission_matrix(decomp, occ, q_C, q_P, parent_blind=False):
    """E[(op,oc)][state] = P(observe signature (op,oc) | state). op,oc in {0,1}.
    parent_blind=True: the predictor does NOT observe the parent channel -> the parent
    factor is marginalized (set to 1 for every state), so op is ignored. The child
    dynamics still depend on the (hidden) parent; the filter marginalizes it optimally.
    Gap(parent_blind - full) = pure value of observing the parent = the gating value,
    which is 0 on the null control (parent irrelevant to the child)."""
    S = len(decomp)
    E = np.zeros((2, 2, S))
    for si, (zP, zC) in enumerate(decomp):
        # parent factor
        if parent_blind:
            Lp1 = Lp0 = 1.0                          # parent channel unobserved
        else:
            Lp1 = (1 - q_P) if zP == 1 else 0.0      # observe parent-on
            Lp0 = 1.0 if zP == 0 else q_P            # observe parent-off
        # child factor (occlusion is a clean missing-obs on the child coordinate)
        Lc1 = (1 - occ) * (1 - q_C) if zC == 1 else 0.0                 # observe child-pos
        Lc0 = 1.0 if zC == 0 else (occ + (1 - occ) * q_C)              # observe child-zero
        E[1, 1, si] = Lp1 * Lc1
        E[1, 0, si] = Lp1 * Lc0
        E[0, 1, si] = Lp0 * Lc1
        E[0, 0, si] = Lp0 * Lc0
    return E


def bayes_child_pi(cP_content, cC_content, gated, occ, q_C, q_P, parent_blind=False):
    """Return pi_C(t) = P(z_C(t+1)=1 | occluded obs 0..t), shape (B,T).
    cP_content, cC_content are the DE-BIASED content coords (parent/child magnitudes);
    parent-on ~= 1.5, off = 0; child on&visible ~= 1.5, occluded-or-off = 0. Threshold 0.5."""
    decomp, Tm, init, childmask = state_decomp(gated)
    E = emission_matrix(decomp, occ, q_C, q_P, parent_blind=parent_blind)
    op = (cP_content > 0.5).astype(int)     # (B,T)
    oc = (cC_content > 0.5).astype(int)
    Bn, Tn = op.shape
    alpha = np.tile(init, (Bn, 1))                         # posterior over state t
    pi = np.empty((Bn, Tn))
    for t in range(Tn):
        Et = E[op[:, t], oc[:, t], :]                     # (B,S) emission likelihood
        if t == 0:
            a = alpha * Et
        else:
            a = (alpha @ Tm) * Et
        a = a / np.clip(a.sum(1, keepdims=True), 1e-300, None)
        alpha = a
        prednext = alpha @ Tm                             # prior over state t+1
        pi[:, t] = prednext @ childmask
    return pi


# ---------------- lag-only linear frontiers (least squares, normal equations) ----
def ridge_solve(XtX, Xty, lam=1e-8):
    n = XtX.shape[0]
    return np.linalg.solve(XtX + lam * np.eye(n), Xty)


def lag_rel_mse(cP, cC, y, Lmax):
    """Strict theorem class: ONE shared per-relative-lag linear map, boundary-truncated.
    features at (b,t) = [1] + [cP(t-j),cC(t-j) for j=0..Lmax] (0 where t-j<0)."""
    Bn, Tn = cP.shape
    F = 1 + 2 * (Lmax + 1)
    XtX = np.zeros((F, F)); Xty = np.zeros(F); n = 0
    Xsy = 0.0; Xyy = 0.0
    for t in range(Tn):
        feat = np.ones((Bn, F))
        for j in range(Lmax + 1):
            s = t - j
            if s >= 0:
                feat[:, 1 + 2 * j] = cP[:, s]
                feat[:, 2 + 2 * j] = cC[:, s]
        yt = y[:, t]
        XtX += feat.T @ feat
        Xty += feat.T @ yt
        Xyy += yt @ yt
        n += Bn
    w = ridge_solve(XtX, Xty)
    sse = Xyy - 2 * w @ Xty + w @ XtX @ w
    return sse / n


def lag_pos_mse(cP, cC, y):
    """Over-approx: per-destination-position content-blind linear regression on all
    causal past tokens. Returns MSE averaged over all (b,t) and the per-t MSEs."""
    Bn, Tn = cP.shape
    sse_tot = 0.0; n = 0; per_t = []
    for t in range(Tn):
        # features [1, cP(0..t), cC(0..t)]
        feat = np.concatenate([np.ones((Bn, 1)), cP[:, :t + 1], cC[:, :t + 1]], 1)
        yt = y[:, t]
        XtX = feat.T @ feat; Xty = feat.T @ yt
        w = ridge_solve(XtX, Xty)
        res = yt - feat @ w
        s = float(res @ res)
        sse_tot += s; n += Bn; per_t.append(s / Bn)
    return sse_tot / n, per_t


def lag0_mse(cP, cC, y):
    """Current-token-only linear (lag-0). Single shared map."""
    Bn, Tn = cP.shape
    feat = np.stack([np.ones_like(cP), cP, cC], -1).reshape(-1, 3)
    yv = y.reshape(-1)
    XtX = feat.T @ feat; Xty = feat.T @ yv
    w = ridge_solve(XtX, Xty)
    res = yv - feat @ w
    return float(res @ res) / yv.size


# ---------------- per-cell driver ----------------
def run_cell(gated, occ):
    gen = HierProcess(d=32, gated=gated, child_occlude=occ, seed=EVAL_SEED, **PARAMS)
    X, Y, _ = gen.sample_seq(B, T)
    dP, dC = gen.D[0], gen.D[1]
    bP = float(gen.b @ dP); bC = float(gen.b @ dC)
    cP_content = X @ dP - bP; cC_content = X @ dC - bC   # DE-BIASED occluded content (B,T)
    y = Y @ dC                                          # target child coord = c_C(t+1)+b.dC
    m1_C = gen.m1_C
    m2_C = relu_moments(PARAMS["mu_C"], PARAMS["sig_C"])[1]
    const = bC
    q_C = float(gen.q_C)
    from scipy.stats import norm
    q_P = float(norm.cdf(-PARAMS["mu_P"] / PARAMS["sig_P"]))

    # frontier 1: Bayes (full joint filter, sees parent + child)
    pi = bayes_child_pi(cP_content, cC_content, gated, occ, q_C, q_P)
    pred_bayes = m1_C * pi + const
    err = (y - pred_bayes) ** 2
    mse_bayes = float(err.mean())
    se_bayes = float(err.std() / np.sqrt(err.size))

    # frontier 1b: parent-blind Bayes (optimal child filter that NEVER observes the parent
    # channel; still nonlinear in the child channel -> isolates the pure gating value,
    # which must be 0 on the null control).
    pi_pb = bayes_child_pi(cP_content, cC_content, gated, occ, q_C, q_P, parent_blind=True)
    pred_pb = m1_C * pi_pb + const
    mse_pblind = float(((y - pred_pb) ** 2).mean())

    # frontier 2: lag-only (bias absorbed by the intercept; content coords fine)
    mse_lag_pos, per_t = lag_pos_mse(cP_content, cC_content, y)
    mse_lag_rel = lag_rel_mse(cP_content, cC_content, y, Lmax=T - 1)
    mse_lag0 = lag0_mse(cP_content, cC_content, y)
    mse_const = float(((y - y.mean()) ** 2).mean())

    return dict(
        mse_bayes=mse_bayes, se_bayes=se_bayes, mse_pblind=mse_pblind,
        mse_lag_rel=mse_lag_rel, mse_lag_pos=mse_lag_pos,
        mse_lag0=mse_lag0, mse_const=mse_const,
        gating_value=mse_pblind - mse_bayes,          # pure parent-info value (0 on null)
        gap_bayes_vs_lagrel=mse_lag_rel - mse_bayes,
        gap_bayes_vs_lagpos=mse_lag_pos - mse_bayes,
        m1_C=m1_C, m2_C=m2_C, const=const, q_C=q_C, q_P=q_P,
        Epi=float(pi.mean()), Epi2=float((pi ** 2).mean()),
    )


def main():
    R = {"params": PARAMS, "T": T, "B": B, "eval_seed": EVAL_SEED,
         "note": "frontier_lag_rel = strict theorem class (shared per-relative-lag map); "
                 "frontier_lag_pos = over-approx (per-position content-blind linear, >= pos-only linear class)."}
    for gated in [True, False]:
        for occ in [0.5, 0.0]:
            nm = f"{'gated' if gated else 'null'}_{'occ' if occ else 'clean'}"
            print(f"[{nm}] computing frontiers ...", flush=True)
            R[nm] = run_cell(gated, occ)
            r = R[nm]
            print(f"  bayes={r['mse_bayes']:.5f}(+-{r['se_bayes']:.5f})  "
                  f"pblind={r['mse_pblind']:.5f}  lag_rel={r['mse_lag_rel']:.5f}  "
                  f"lag_pos={r['mse_lag_pos']:.5f}  lag0={r['mse_lag0']:.5f}  "
                  f"const={r['mse_const']:.5f}", flush=True)
            print(f"  GATING VALUE (pblind-bayes)={r['gating_value']:+.5f}  |  "
                  f"gap bayes-vs-lag_rel={r['gap_bayes_vs_lagrel']:+.5f}  "
                  f"bayes-vs-lag_pos={r['gap_bayes_vs_lagpos']:+.5f}", flush=True)
            json.dump(R, open(os.path.join(OUT, "rung2_frontiers.json"), "w"), indent=2)
    print("wrote out/rung2_frontiers.json", flush=True)


if __name__ == "__main__":
    main()
