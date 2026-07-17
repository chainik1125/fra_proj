"""
C2 metric-saturation (oracle-only, NO training): WHICH metric should certify a learned
error-correcting alignment decoder, as the code deepens?

Motivation (from the n=9 red-team, RESEARCH_LOG 2026-06-03 13:38Z): at n=9 the model's logical
ERROR RATE (0.050) had collapsed onto the trivial "always-aligned" baseline (0.0498), because a
deeper code makes logical misalignment rare -- so the error rate stopped discriminating a good
decoder from a constant one. Yet the model's posterior KL still beat a constant-marginal predictor
~15x. This script makes that observation a closed-form, oracle-anchored statement ACROSS n.

For the redundant active bag (per-block stationary q* = eps/(eps+gamma) = 0.25, majority threshold
r = (n-1)//2) we compute, purely from the exact oracle (per-chain forward filter + poisson-binomial
tail) on a large simulated eval set -- no transformer involved:

  * B(n)            = empirical marginal logical-misalignment rate  (= error of the trivial
                      "always-aligned" predictor). Anchored to closed-form binomial_tail(n,q*,r).
  * bayes_err(n)    = E[min(orac, 1-orac)]   (optimal HARD-decision error given noisy emissions).
  * err_headroom    = B(n) - bayes_err       (absolute error-rate room the optimal decoder buys).
  * rel_err_headroom= err_headroom / B(n).
  * const_kl(n)     = mean KL( constant predictor p=E[orac]  ||  true posterior orac )  -- the
                      DISTRIBUTIONAL room the optimal decoder buys (the Bayes posterior has KL 0).

CLAIM (to be shown): as n grows with q*<0.5, B(n) -> 0 (super-exponentially), so err_headroom -> 0
and the error rate loses discriminative power; but const_kl stays bounded away from 0 (the posterior
stays graded). Hence KL / log-odds-R^2 -- not the error rate -- are the metrics that certify a learned
decoder for deep codes. This generalises the single n=9 datapoint and scopes the n=3/5/7 error curve.

Oracle gate: run experiments/validate_active.py first (PASS this session). Inline, we assert the
empirical B(n) matches the closed-form binomial tail to <1%.

Outputs results/c2_metric_saturation.pt
"""
import os
import numpy as np
from bag_moments import active

OUT = os.environ.get("BAG_OUT", "results")
EPS = float(os.environ.get("C2SAT_EPS", "0.05"))
GAMMA = float(os.environ.get("C2SAT_GAMMA", "0.15"))
PA, PM, T = 0.3, 0.7, 40
QSTAR = EPS / (EPS + GAMMA)  # default = 0.25; near-threshold control: eps=0.10,gamma=0.15 -> 0.40
NS = [3, 5, 7, 9, 11, 13]
B = int(os.environ.get("C2SAT_B", "120000"))
TAG = os.environ.get("C2SAT_TAG", "")  # output suffix, e.g. "_q040"


def oracle_eval(n, rng):
    r = (n - 1) // 2
    tk, hid, _ = active.gen_redundant_active(B, T, n, EPS, GAMMA, PA, PM, rng, beta=0.0)
    qchain = active.per_chain_filter(tk, EPS, GAMMA, PA, PM)     # (B,T,n)
    qlast = qchain[:, -1, :]                                     # (B,n)
    orac = active.poisson_binomial_tail(qlast, r)                # (B,) P(L=M|obs)
    true_L = (hid[:, -1, :].sum(1) > r).astype(np.int64)         # (B,)
    e = 1e-7
    B_emp = float(orac.mean())                                   # marginal misalign rate == trivial err
    B_closed = float(active.binomial_tail(n, QSTAR, r))          # closed-form (stationary q*)
    bayes_err = float(np.minimum(orac, 1 - orac).mean())
    always_aligned_err = float((true_L == 1).mean())             # error of predicting "aligned" always
    pc = np.clip(orac.mean(), e, 1 - e); po = np.clip(orac, e, 1 - e)
    const_kl = float((po * np.log(po / pc) + (1 - po) * np.log((1 - po) / (1 - pc))).mean())
    frac_uncertain = float(((orac > 0.1) & (orac < 0.9)).mean())
    return {
        "n": n, "r": r, "B_emp": B_emp, "B_closed": B_closed,
        "bayes_err": bayes_err, "always_aligned_err": always_aligned_err,
        "err_headroom": B_emp - bayes_err, "rel_err_headroom": (B_emp - bayes_err) / max(B_emp, 1e-12),
        "const_kl": const_kl, "frac_uncertain": frac_uncertain,
        "orac_std": float(orac.std()),
    }


def main():
    rows = []
    print(f"q* = eps/(eps+gamma) = {QSTAR:.3f}; eval B={B}, T={T}")
    print(f"{'n':>3} {'B(n)':>9} {'B_closed':>9} {'bayes_err':>10} {'err_room':>9} "
          f"{'rel_room':>9} {'const_KL':>9} {'frac_unc':>9}")
    for n in NS:
        rng = np.random.default_rng(7000 + n)
        row = oracle_eval(n, rng)
        rows.append(row)
        # inline oracle gate: empirical marginal must match closed-form binomial tail
        assert abs(row["B_emp"] - row["B_closed"]) < 0.01, \
            f"n={n}: B_emp {row['B_emp']:.4f} vs closed {row['B_closed']:.4f} (>1%)"
        # sanity: trivial-predictor error == marginal misalign rate (both = B), to estimator noise
        assert abs(row["always_aligned_err"] - row["B_emp"]) < 0.01
        print(f"{n:>3} {row['B_emp']:>9.4f} {row['B_closed']:>9.4f} {row['bayes_err']:>10.4f} "
              f"{row['err_headroom']:>9.4f} {row['rel_err_headroom']:>9.3f} "
              f"{row['const_kl']:>9.4f} {row['frac_uncertain']:>9.4f}", flush=True)
    # headline ratios: distributional signal vs error-rate signal
    print("\nDISTRIBUTIONAL-vs-ERROR headroom ratio (const_kl / err_headroom):")
    for row in rows:
        ratio = row["const_kl"] / max(row["err_headroom"], 1e-9)
        print(f"  n={row['n']:>2}: const_kl/err_headroom = {ratio:7.2f} "
              f"(err_headroom {row['err_headroom']:.4f} shrinking; const_kl {row['const_kl']:.4f})")
    res = {"qstar": QSTAR, "eps": EPS, "gamma": GAMMA, "T": T, "B": B, "ns": NS, "rows": rows}
    os.makedirs(OUT, exist_ok=True)
    import torch
    torch.save(res, f"{OUT}/c2_metric_saturation{TAG}.pt")
    print(f"\nsaved {OUT}/c2_metric_saturation{TAG}.pt")


if __name__ == "__main__":
    main()
