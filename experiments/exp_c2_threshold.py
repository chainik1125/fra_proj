"""
C2 (process level): error correction by redundancy, and when it FAILS.

Two facets of the active-bag error-correction model:

 (1) Independent errors -> binomial-tail suppression. With n redundant alignment blocks
     and majority decode (corrects up to r=(n-1)/2 bad blocks), logical misalignment
     p_log = P(Bin(n,p)>r) < p for p<1/2. Redundancy suppresses; more blocks => more.

 (2) Spreading/correlated errors -> an epidemic THRESHOLD. With nearest-neighbour spread
     (mean-field complete graph): a chain's corruption rate rises with the fraction of
     misaligned neighbours (rate beta), recovery rate gamma. The all-aligned state is
     stable iff R_M = beta/gamma < 1. Above R_M=1 misalignment becomes endemic at
     stationary fraction I* = 1 - 1/R_M (SIS transcritical bifurcation). This is the
     precise condition under which redundancy can no longer protect alignment.

Pure simulation vs the closed forms. Outputs results/c2_threshold.pt
"""
import os
import numpy as np
from bag_moments import active

OUT = os.environ.get("BAG_OUT", "/Users/dmitrymanning-coe/Documents/Research/Simplex/bag-sprint-2/results")


def main():
    import torch
    rng = np.random.default_rng(0)
    res = {}

    # (1) binomial-tail suppression: logical vs physical p, several n
    print("(1) binomial-tail suppression (independent blocks):")
    ps = np.linspace(0.02, 0.6, 13)
    supp = {}
    for n in [1, 3, 5, 7]:
        r = (n - 1) // 2
        supp[n] = [active.binomial_tail(n, p, r) for p in ps]
        print(f"  n={n}: p=0.3 -> p_log={active.binomial_tail(n,0.3,r):.3f}")
    res["suppression"] = {"p": ps.tolist(), "by_n": {n: supp[n] for n in supp}}

    # (2) epidemic threshold: vary R_M = beta/gamma with eps=0, measure stationary
    # fraction misaligned; expect I* = max(0, 1 - 1/R_M).
    print("(2) spread threshold (eps=0, seed 50% misaligned):")
    n = 64; gamma = 0.1; L = 600; B = 400
    RMs = np.linspace(0.0, 3.0, 16)
    endemic = []
    for RM in RMs:
        beta = RM * gamma
        # simulate: start 50% misaligned, eps=0
        cur = (rng.random((B, n)) < 0.5).astype(np.int64)
        for t in range(L):
            frac_other = (cur.sum(1, keepdims=True) - cur) / (n - 1)
            eps_eff = np.clip(beta * frac_other, 0, 1)
            u = rng.random((B, n))
            cur = np.where(cur == 0, (u < eps_eff).astype(np.int64),
                           np.where(u < gamma, 0, 1))
        I = cur.mean()
        endemic.append(float(I))
        theory = max(0.0, 1 - 1 / RM) if RM > 0 else 0.0
        print(f"  R_M={RM:.2f}: sim I*={I:.3f}  theory={theory:.3f}")
    res["threshold"] = {"R_M": RMs.tolist(), "endemic_sim": endemic,
                        "endemic_theory": [max(0.0, 1 - 1/r) if r > 0 else 0.0 for r in RMs],
                        "n": n, "gamma": gamma}
    torch.save(res, f"{OUT}/c2_threshold.pt")
    print(f"saved {OUT}/c2_threshold.pt")


if __name__ == "__main__":
    main()
