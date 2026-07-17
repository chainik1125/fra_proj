"""Validation gate for the alignment-modulated Mess3 GHMM (bag_moments/ghmm.py).

Confirms the generator + exact 6-state forward filter are mutually consistent and match
analytic ground truth, BEFORE any transformer is trained on top.
"""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bag_moments import ghmm

EPS, GAMMA, STAY, A = 0.02, 0.06, 0.30, 0.80
QSTAR = EPS / (EPS + GAMMA)
P = ghmm.joint_params_drift(EPS, GAMMA, STAY, A)

def ok(name, got, want, tol):
    s = "OK " if abs(got - want) <= tol else "FAIL"
    print(f"  [{s}] {name}: got {got:.4f} want {want:.4f} (tol {tol})")
    return s == "OK "

def main():
    rng = np.random.default_rng(0)
    print(f"=== drift-modulated Mess3 (eps={EPS} gamma={GAMMA} stay={STAY} a={A}, q*={QSTAR:.3f}) ===")
    obs, c, s = ghmm.gen_ghmm(8000, 220, P, rng)
    half = 110
    allok = True
    # (A) stationarity of the latents
    allok &= ok("E[c=M] (2nd half)", c[:, half:].mean(), QSTAR, 0.01)
    for k in range(3):
        allok &= ok(f"P(s={k}) (2nd half)", (s[:, half:] == k).mean(), 1/3, 0.01)
    # (B) marginal emission distribution: pi @ Eobs.T
    pi = ghmm.stationary_joint(P)
    pe = pi @ P["Eobs"].T
    for o in range(3):
        allok &= ok(f"P(obs={o})", (obs[:, half:] == o).mean(), pe[o], 0.01)

    # (C) joint filter calibration: P(c=M | filtered q in bin) tracks bin centre
    belief, q, mbel, nextp = ghmm.forward_filter_ghmm(obs, P)
    print("  (C) alignment-filter calibration P(c=M | filtered q in bin):")
    qf = q[:, 20:].ravel(); cf = (c[:, 20:] == 1).ravel().astype(float)
    bins = np.linspace(0, 1, 11)
    for lo, hi in zip(bins[:-1], bins[1:]):
        msk = (qf >= lo) & (qf < hi)
        if msk.sum() > 2000:
            emp = cf[msk].mean(); cen = qf[msk].mean()
            s_ = "OK" if abs(emp - cen) < 0.02 else "!!"
            print(f"      [{s_}] q~{cen:.3f}: emp P(M)={emp:.3f} (n={msk.sum()})")
            allok &= abs(emp - cen) < 0.03

    # (D) mess3 belief calibration: P(s=0 | mbelief_0 in bin) tracks bin centre
    print("  (D) mess3-belief calibration P(s=0 | filtered m0 in bin):")
    m0 = mbel[:, 20:, 0].ravel(); s0 = (s[:, 20:] == 0).ravel().astype(float)
    for lo, hi in zip(bins[:-1], bins[1:]):
        msk = (m0 >= lo) & (m0 < hi)
        if msk.sum() > 2000:
            emp = s0[msk].mean(); cen = m0[msk].mean()
            s_ = "OK" if abs(emp - cen) < 0.02 else "!!"
            print(f"      [{s_}] m0~{cen:.3f}: emp P(s=0)={emp:.3f} (n={msk.sum()})")
            allok &= abs(emp - cen) < 0.03

    # (E) next-symbol calibration: P(o_{t+1}=0 | next_p[...,0] bin)
    print("  (E) next-symbol calibration P(o_{t+1}=0 | pred in bin):")
    pn = nextp[:, 20:-1, 0].ravel()
    actual = (obs[:, 21:] == 0).ravel().astype(float)
    for lo, hi in zip(bins[:-1], bins[1:]):
        msk = (pn >= lo) & (pn < hi)
        if msk.sum() > 2000:
            emp = actual[msk].mean(); cen = pn[msk].mean()
            s_ = "OK" if abs(emp - cen) < 0.02 else "!!"
            print(f"      [{s_}] pred~{cen:.3f}: emp={emp:.3f} (n={msk.sum()})")
            allok &= abs(emp - cen) < 0.03

    print("=" * 60)
    print("ALL CHECKS PASSED" if allok else "SOME CHECKS FAILED")
    return allok

if __name__ == "__main__":
    main()
