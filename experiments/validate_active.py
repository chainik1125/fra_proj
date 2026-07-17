"""
Validation gate for the active-bag generator + exact forward filter. Nothing
downstream is trusted until these pass.

  (A) Stationarity: empirical P(hidden=M) and mean filtered q_t match q*=eps/(eps+gamma).
  (B) Marginal emission: P(x=1) = (1-q*)pA + q* pM.
  (C) Filter calibration: bin positions by filtered q_t; empirical P(hidden=M|bin) ≈ bin centre
      (the forward filter is the *correct* posterior).
  (D) Next-symbol calibration: bin by Bayes next1; empirical P(x_{t+1}=1|bin) ≈ bin centre.
  (E) Redundant bag (beta=0): logical misalignment rate ≈ binomial tail P(Bin(n,q*)>r).
"""

import numpy as np
from bag_moments import active


def check(name, got, want, tol):
    ok = abs(got - want) < tol
    print(f"  [{'OK ' if ok else 'XX '}] {name}: got {got:.4f} want {want:.4f} (tol {tol})")
    return ok


def main():
    rng = np.random.default_rng(0)
    all_ok = True
    eps, gamma, pA, pM = 0.05, 0.15, 0.3, 0.7
    qstar = active.stationary_q(eps, gamma)
    B, L = 20000, 400
    toks, hid = active.gen_active_2state(B, L, eps, gamma, pA, pM, rng)
    q, qm, next1 = active.forward_filter_2state(toks, eps, gamma, pA, pM)

    print(f"=== two-state active bag (eps={eps}, gamma={gamma}, pA={pA}, pM={pM}, q*={qstar:.4f}) ===")
    # (A) stationarity (use second half to avoid transient)
    all_ok &= check("(A) E[hidden=M] (2nd half)", hid[:, L//2:].mean(), qstar, 0.005)
    all_ok &= check("(A) E[filtered q] (2nd half)", q[:, L//2:].mean(), qstar, 0.005)
    # (B) marginal emission
    all_ok &= check("(B) P(x=1)", toks[:, L//2:].mean(), (1-qstar)*pA + qstar*pM, 0.005)

    # (C) filter calibration
    print("  (C) filter calibration P(hidden=M | filtered q in bin):")
    qf = q[:, 1:].ravel(); hf = hid[:, 1:].ravel().astype(float)
    edges = np.linspace(0, 1, 11)
    cal_ok = True
    for i in range(10):
        m = (qf >= edges[i]) & (qf < edges[i+1])
        if m.sum() < 200:
            continue
        emp = hf[m].mean(); centre = qf[m].mean()
        ok = abs(emp - centre) < 0.03
        cal_ok &= ok
        print(f"      [{'OK' if ok else 'XX'}] q~{centre:.3f}: emp P(M)={emp:.3f} (n={m.sum()})")
    all_ok &= cal_ok

    # (D) next-symbol calibration
    print("  (D) next-symbol calibration P(x_{t+1}=1 | next1 in bin):")
    nf = next1[:, :-1].ravel(); xf = toks[:, 1:].ravel().astype(float)
    cal_ok = True
    for i in range(10):
        m = (nf >= edges[i]) & (nf < edges[i+1])
        if m.sum() < 200:
            continue
        emp = xf[m].mean(); centre = nf[m].mean()
        ok = abs(emp - centre) < 0.03
        cal_ok &= ok
        print(f"      [{'OK' if ok else 'XX'}] next1~{centre:.3f}: emp={emp:.3f} (n={m.sum()})")
    all_ok &= cal_ok

    # (E) redundant bag binomial tail
    print("  (E) redundant-bag logical misalignment vs binomial tail (beta=0):")
    for n in [3, 5, 7]:
        r = (n - 1) // 2
        tk, hd, logical = active.gen_redundant_active(8000, 300, n, eps, gamma, pA, pM, rng, beta=0.0)
        emp_log = logical[:, 150:].mean()
        want = active.binomial_tail(n, qstar, r)
        all_ok &= check(f"      n={n} logical rate", emp_log, want, 0.01)

    print("=" * 60)
    print("ALL CHECKS PASSED" if all_ok else "*** SOME CHECKS FAILED ***")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
