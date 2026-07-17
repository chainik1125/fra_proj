"""
RATE-CONTROLLED process-level companion to §4.5: isolate the effect of cross-chain CORRELATION
on majority-decoding, holding the per-chain marginal misrate FIXED.

§4.5's transformer experiment confounds correlation with the marginal rate (mean-field spread
raises both). Here we remove that confound at the process level (no transformer, pure filters):
for each spread strength beta we binary-search eps so the per-chain misrate stays ~TARGET, then
compare the EXACT joint decoder (2^n filter) to the NAIVE independent decoder (poisson-binomial
tail of per-chain posteriors) as a function of the induced correlation rho.

Result (the rigorous version of "correlation breaks the binomial-tail picture"): at fixed rate,
the independent decoder's logical error stays high and the GAP to the optimal joint decoder grows
monotonically with rho; at strong correlation the independent (majority-of-correlated-chains)
decoder is even WORSE than a single chain, while the joint decoder still error-corrects.

Outputs results/c2_corr_decoders.pt
"""
import os, sys
import numpy as np
ROOT = os.environ.get("BAG_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import torch
from bag_moments import active
from experiments import exp_c2_spread as E

OUT = os.environ.get("BAG_OUT", os.path.join(ROOT, "results"))
N, GAMMA, PA, PM, R = 5, 0.05, 0.3, 0.7, 2
TARGET = float(os.environ.get("CORR_TARGET", "0.30"))   # fixed per-chain marginal misrate
BURN = 100


def misrate(eps, beta, seed=3):
    _, hid, _ = active.gen_redundant_active(1500, 200, N, eps, GAMMA, PA, PM,
                                            np.random.default_rng(seed), beta=beta)
    return float(hid[:, BURN:, :].mean())


def tune_eps(beta):
    lo, hi = 0.0, 0.30
    for _ in range(24):
        mid = (lo + hi) / 2
        if misrate(mid, beta) < TARGET:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def main():
    betas = [0.0, 0.06, 0.12, 0.18]
    rows = []
    print(f"=== Rate-controlled decoder sweep (per-chain misrate fixed at {TARGET}, n={N}) ===")
    print(f"{'beta':>6} {'eps':>6} {'rho':>7} {'rate':>6} {'joint_err':>10} {'indep_err':>10} "
          f"{'gap':>6} {'phys':>6}")
    for beta in betas:
        eps = tune_eps(beta)
        rng = np.random.default_rng(11)
        tk, hid, lg = active.gen_redundant_active(4000, 200, N, eps, GAMMA, PA, PM, rng, beta=beta)
        packed = active.pack_emissions(tk)
        pj = E.joint_filter(packed, N, eps, GAMMA, PA, PM, beta)[:, BURN:].reshape(-1)
        qc = active.per_chain_filter(tk, eps, GAMMA, PA, PM)[:, BURN:, :].reshape(-1, N)
        pind = active.poisson_binomial_tail(qc, R)
        tl = lg[:, BURN:].reshape(-1).astype(float)
        je = float((((pj > 0.5).astype(int)) != tl).mean())
        ie = float((((pind > 0.5).astype(int)) != tl).mean())
        phys = float(np.minimum(qc[:, 0], 1 - qc[:, 0]).mean())
        rate = float(hid[:, BURN:, :].mean()); rho = E.cross_chain_corr(hid)
        brier_j = float(((pj - tl) ** 2).mean()); brier_i = float(((pind - tl) ** 2).mean())
        rows.append({"beta": beta, "eps": eps, "rho": rho, "rate": rate, "joint_err": je,
                     "indep_err": ie, "gap": ie - je, "phys": phys,
                     "brier_joint": brier_j, "brier_indep": brier_i})
        print(f"{beta:6.2f} {eps:6.3f} {rho:+7.3f} {rate:6.3f} {je:10.3f} {ie:10.3f} "
              f"{ie - je:6.3f} {phys:6.3f}")
    torch.save({"rows": rows, "target_rate": TARGET, "n": N, "gamma": GAMMA, "pA": PA, "pM": PM},
               f"{OUT}/c2_corr_decoders.pt")
    print(f"saved {OUT}/c2_corr_decoders.pt")


if __name__ == "__main__":
    main()
