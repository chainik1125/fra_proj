"""
Closed-form Bayes-optimal oracles for the passive bag-of-coins model.

A coin is p in [0,1] = P(emit 1).  A *bag* is an equal-weight finite measure
G = (1/N) sum_i delta_{p_i} with p_i ~iid Unif[0,1] and N ~ nu (a prior on the
number of components).  A rollout picks one component C ~ Unif{1..N} and emits
iid Bernoulli(p_C).

Two protocols:
  * annealed (fresh bag per rollout): one rollout sees only the barycenter
    bar G = Unif[0,1]; the optimal next-token predictor is the Laplace rule
    (s+1)/(t+2) and N is information-theoretically invisible.
  * quenched (one bag reused for J rollouts): J rollouts expose the J-th moment
    measure M_J = E[G^{otimes J}].  For J=2 the relevant scalar is the collision
    susceptibility chi_2 = E[1/N] = P(two rollouts share a component).

Everything here is closed form; these functions are the ground truth that the
generator and the trained transformers are checked against.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# ----------------------------------------------------------------------------
# Bag-size priors nu(n) and their moments
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class NPrior:
    """A prior over the number of components N >= 1, on bounded support."""

    name: str
    support: tuple[int, ...]
    probs: tuple[float, ...]

    def __post_init__(self):
        assert len(self.support) == len(self.probs)
        assert abs(sum(self.probs) - 1.0) < 1e-9, "probs must sum to 1"
        assert all(n >= 1 for n in self.support), "N >= 1 required"

    def mean_inv(self) -> float:
        """chi_2 = E[1/N]."""
        return float(sum(pr / n for n, pr in zip(self.support, self.probs)))

    def mean_inv_sq(self) -> float:
        """E[1/N^2] (third-order collision scale)."""
        return float(sum(pr / (n * n) for n, pr in zip(self.support, self.probs)))

    def mean(self) -> float:
        return float(sum(pr * n for n, pr in zip(self.support, self.probs)))

    def sample(self, rng: np.random.Generator, size) -> np.ndarray:
        return rng.choice(self.support, size=size, p=self.probs)


def fixed_n(n: int) -> NPrior:
    return NPrior(f"fixed{n}", (n,), (1.0,))


def uniform_n(n_min: int, n_max: int) -> NPrior:
    support = tuple(range(n_min, n_max + 1))
    p = 1.0 / len(support)
    return NPrior(f"unif{n_min}-{n_max}", support, tuple(p for _ in support))


def zt_poisson(lam: float, n_max: int = 40) -> NPrior:
    """Zero-truncated Poisson(lam), truncated at n_max for a bounded support."""
    ns = list(range(1, n_max + 1))
    w = np.array([lam**n / math.factorial(n) for n in ns], dtype=np.float64)
    w = w / w.sum()
    return NPrior(f"ztpois{lam}", tuple(ns), tuple(float(x) for x in w))


# ----------------------------------------------------------------------------
# Beta-Bernoulli building blocks
# ----------------------------------------------------------------------------


def laplace_predictive(s: int, t: int) -> float:
    """Posterior mean of Beta(s+1, t-s+1): P(next=1 | s ones in t flips)."""
    return (s + 1.0) / (t + 2.0)


def log_block_marginal(s: int, t: int) -> float:
    """log of the Beta-Bernoulli marginal likelihood of a block with t flips, s ones.

    m(s,t) = int_0^1 p^s (1-p)^(t-s) dp = B(s+1, t-s+1) = s!(t-s)!/(t+1)!
    """
    return (
        math.lgamma(s + 1) + math.lgamma(t - s + 1) - math.lgamma(t + 2)
    )


# ----------------------------------------------------------------------------
# Quenched J=2 oracle: exact two-hypothesis (same-coin vs different-coin) mixture
# ----------------------------------------------------------------------------


def quenched2_collision_posterior(s1: int, t1: int, s2: int, t2: int, chi2: float) -> float:
    """Posterior P(C2 = C1 | rollout1=(s1,t1), rollout2-prefix=(s2,t2)).

    Prior collision prob = chi2 = E[1/N].  Same-coin hypothesis pools the two
    blocks; different-coin keeps them separate.
    """
    log_same = log_block_marginal(s1 + s2, t1 + t2)
    log_diff = log_block_marginal(s1, t1) + log_block_marginal(s2, t2)
    # posterior odds = prior odds * likelihood ratio (log space).  Clamp chi2 off
    # the {0,1} boundary so N=1 (chi2=1) and the no-collision limit are well defined.
    c = min(max(chi2, 1e-12), 1.0 - 1e-12)
    log_w_same = math.log(c) + log_same
    log_w_diff = math.log(1.0 - c) + log_diff
    m = max(log_w_same, log_w_diff)
    w_same = math.exp(log_w_same - m)
    w_diff = math.exp(log_w_diff - m)
    return w_same / (w_same + w_diff)


def quenched2_predictive(s1: int, t1: int, s2: int, t2: int, chi2: float) -> float:
    """Bayes-optimal P(next token of rollout 2 = 1).

    Mixture over same/different coin, each contributing its Laplace predictive.
    """
    q = quenched2_collision_posterior(s1, t1, s2, t2, chi2)
    pred_same = laplace_predictive(s1 + s2, t1 + t2)
    pred_diff = laplace_predictive(s2, t2)
    return q * pred_same + (1.0 - q) * pred_diff


def quenched2_first_token(s1: int, t1: int, chi2: float) -> float:
    """Special case: first token of rollout 2 (s2=t2=0).  Marginals cancel,
    so the collision posterior equals the prior chi2."""
    return chi2 * laplace_predictive(s1, t1) + (1.0 - chi2) * 0.5


# ----------------------------------------------------------------------------
# Exact J-rollout oracle via set-partition enumeration (small J, bounded N)
# ----------------------------------------------------------------------------


def _set_partitions(collection):
    """Yield all set partitions of a list (Bell number many)."""
    collection = list(collection)
    if len(collection) == 1:
        yield [collection]
        return
    first = collection[0]
    for smaller in _set_partitions(collection[1:]):
        for i, subset in enumerate(smaller):
            yield smaller[:i] + [[first] + subset] + smaller[i + 1 :]
        yield [[first]] + smaller


def _falling_factorial(n: int, k: int) -> float:
    out = 1.0
    for i in range(k):
        out *= n - i
    return out


def quenchedJ_log_evidence(blocks_st, nprior: NPrior) -> float:
    """log P(data) for J rollouts with sufficient stats blocks_st = [(s_j,t_j)],
    marginalizing over the bag size N and over all rollout->coin assignments.

    Used for the J-rollout experiment (Exp 3).  J small (<= ~6).
    """
    J = len(blocks_st)
    # precompute partitions of {0..J-1}
    partitions = list(_set_partitions(range(J)))
    # for each N, sum over partitions
    total = 0.0  # in linear space over N (few terms), log-sum over partitions
    log_terms_by_n = []
    for n, pr in zip(nprior.support, nprior.probs):
        # partitions with more blocks than n contribute 0 (can't place k blocks in n coins distinctly... actually with replacement we need distinct coins per block)
        log_part_terms = []
        for part in partitions:
            k = len(part)
            if k > n:
                continue
            # prior of this partition under uniform coin selection: (n)_k / n^J
            coef = _falling_factorial(n, k) / (n**J)
            if coef <= 0:
                continue
            log_lik = 0.0
            for block in part:
                S = sum(blocks_st[j][0] for j in block)
                T = sum(blocks_st[j][1] for j in block)
                log_lik += log_block_marginal(S, T)
            log_part_terms.append(math.log(coef) + log_lik)
        if log_part_terms:
            m = max(log_part_terms)
            lse = m + math.log(sum(math.exp(x - m) for x in log_part_terms))
            log_terms_by_n.append(math.log(pr) + lse)
    if not log_terms_by_n:
        return -math.inf
    m = max(log_terms_by_n)
    return m + math.log(sum(math.exp(x - m) for x in log_terms_by_n))


def quenchedJ_posterior_N(blocks_st, nprior: NPrior) -> dict[int, float]:
    """Posterior over N given J observed rollouts (their sufficient stats)."""
    J = len(blocks_st)
    partitions = list(_set_partitions(range(J)))
    log_unnorm = {}
    for n, pr in zip(nprior.support, nprior.probs):
        log_part_terms = []
        for part in partitions:
            k = len(part)
            if k > n:
                continue
            coef = _falling_factorial(n, k) / (n**J)
            if coef <= 0:
                continue
            log_lik = 0.0
            for block in part:
                S = sum(blocks_st[j][0] for j in block)
                T = sum(blocks_st[j][1] for j in block)
                log_lik += log_block_marginal(S, T)
            log_part_terms.append(math.log(coef) + log_lik)
        if log_part_terms:
            m = max(log_part_terms)
            lse = m + math.log(sum(math.exp(x - m) for x in log_part_terms))
            log_unnorm[n] = math.log(pr) + lse
    m = max(log_unnorm.values())
    unnorm = {n: math.exp(v - m) for n, v in log_unnorm.items()}
    Z = sum(unnorm.values())
    return {n: v / Z for n, v in unnorm.items()}
