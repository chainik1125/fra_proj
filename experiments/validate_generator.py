"""
Validation gate: the generator must match the closed-form oracles, or nothing
downstream can be trusted.  Checks:

  (A) Annealed single rollout: short-word probabilities match Beta-Bernoulli.
        P(1)=1/2, P(11)=1/3, P(10)=1/6, P(111)=1/4, and these are INVARIANT to
        the N-prior (barycenter collapse).
  (B) Quenched 2 rollouts: P(Y_{1,1}=1, Y_{2,1}=1) = 1/4 + E[1/N]/12 for several
        N-priors, and the cross-rollout transfer scales with chi_2 = E[1/N].
  (C) Independent-bag control: transfer statistic collapses to 1/4 (no sharing).
  (D) Oracle self-consistency: the exact J-partition evidence reproduces the
        closed-form J=2 first-token predictive.
"""

import numpy as np

from bag_moments import data, oracles


def check(name, got, want, tol):
    ok = abs(got - want) < tol
    print(f"  [{'OK ' if ok else 'XX '}] {name}: got {got:.5f}  want {want:.5f}  (tol {tol})")
    return ok


def main():
    rng = np.random.default_rng(0)
    B = 400_000
    all_ok = True

    print("=" * 70)
    print("(A) Annealed single rollout -> Beta-Bernoulli, invariant to N-prior")
    print("=" * 70)
    x, _ = data.gen_annealed_single(B, 3, rng)
    p1 = x[:, 0].mean()
    p11 = ((x[:, 0] == 1) & (x[:, 1] == 1)).mean()
    p10 = ((x[:, 0] == 1) & (x[:, 1] == 0)).mean()
    p111 = ((x[:, 0] == 1) & (x[:, 1] == 1) & (x[:, 2] == 1)).mean()
    all_ok &= check("P(1)", p1, 1 / 2, 0.003)
    all_ok &= check("P(11)", p11, 1 / 3, 0.003)
    all_ok &= check("P(10)", p10, 1 / 6, 0.003)
    all_ok &= check("P(111)", p111, 1 / 4, 0.003)

    print("=" * 70)
    print("(B) Quenched 2-rollout collision statistic P(1,1) = 1/4 + E[1/N]/12")
    print("=" * 70)
    for nprior in [oracles.fixed_n(1), oracles.fixed_n(2), oracles.fixed_n(5),
                   oracles.uniform_n(1, 6), oracles.zt_poisson(3.0)]:
        chi2 = nprior.mean_inv()
        qb = data.gen_quenched(B, (1, 1), nprior, rng)
        # first token of rollout1 is col 0; first token of rollout2 is col 2 (after ROLL at col1)
        y11 = qb.tokens[:, 0]
        y21 = qb.tokens[:, qb.roll_start_pos[1]]
        p_both = ((y11 == 1) & (y21 == 1)).mean()
        want = 0.25 + chi2 / 12.0
        all_ok &= check(f"P(1,1) [{nprior.name}, chi2={chi2:.4f}]", p_both, want, 0.004)

    print("=" * 70)
    print("(C) Independent-bag control: P(1,1) -> 1/4 (no cross-rollout sharing)")
    print("=" * 70)
    qb = data.gen_quenched(B, (1, 1), oracles.uniform_n(1, 6), rng, independent_bags=True)
    y11 = qb.tokens[:, 0]
    y21 = qb.tokens[:, qb.roll_start_pos[1]]
    all_ok &= check("P(1,1) independent", ((y11 == 1) & (y21 == 1)).mean(), 0.25, 0.004)

    print("=" * 70)
    print("(D) Longer-rollout transfer: E[Y2 first bit | rollout1 had s ones in t]")
    print("    empirical vs closed-form chi2*(s+1)/(t+2) + (1-chi2)/2")
    print("=" * 70)
    nprior = oracles.fixed_n(2)  # chi2 = 1/2
    chi2 = nprior.mean_inv()
    t1 = 8
    qb = data.gen_quenched(B, (t1, 1), nprior, rng)
    s1 = qb.tokens[:, :t1].sum(axis=1)
    y2 = qb.tokens[:, qb.roll_start_pos[1]]
    for s_val in [0, 2, 4, 6, 8]:
        mask = s1 == s_val
        if mask.sum() < 500:
            continue
        emp = y2[mask].mean()
        want = oracles.quenched2_first_token(s_val, t1, chi2)
        all_ok &= check(f"E[Y2|s={s_val}/{t1}] (n={mask.sum()})", emp, want, 0.012)

    print("=" * 70)
    print("(E) Oracle self-consistency: exact J-partition evidence == J=2 closed form")
    print("=" * 70)
    nprior = oracles.uniform_n(1, 6)
    chi2 = nprior.mean_inv()
    for (s1v, t1v) in [(3, 5), (0, 4), (5, 5)]:
        # predictive for first token of rollout 2 via evidence ratio
        e_no = oracles.quenchedJ_log_evidence([(s1v, t1v), (0, 1)], nprior)   # next bit = 0
        e_yes = oracles.quenchedJ_log_evidence([(s1v, t1v), (1, 1)], nprior)  # next bit = 1
        pred_partition = np.exp(e_yes) / (np.exp(e_yes) + np.exp(e_no))
        pred_closed = oracles.quenched2_first_token(s1v, t1v, chi2)
        all_ok &= check(f"oracle Y2|s={s1v}/{t1v}", pred_partition, pred_closed, 1e-9)

    print("=" * 70)
    print("ALL CHECKS PASSED" if all_ok else "*** SOME CHECKS FAILED ***")
    print("=" * 70)
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
