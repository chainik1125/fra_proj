"""
Active bags: block-structured switching processes for the alignment / error-correction
experiments.

Two-state active bag (the "two behavioral coins" of the theory note):
    hidden alignment state C_t in {A (aligned), M (misaligned)}
    transition kernel  K = [[1-eps, eps],
                            [gamma, 1-gamma]]   (rows = current, cols = next)
        eps   = corruption rate  A -> M
        gamma = correction rate  M -> A
    emission  P(x=1 | A) = pA,   P(x=1 | M) = pM     (pM > pA so 1's look misaligned)

The misalignment posterior q_t = P(C_t = M | x_{1:t}) is the exact HMM forward filter;
z_t = logit(q_t) is the "alignment log-odds" coordinate.  Stationary q* = eps/(eps+gamma).

Redundant active bag (Level-2 error correction): n independent alignment chains, each a
two-state active bag; the LOGICAL state is the majority decode; optional nearest-neighbour
"spread" coupling raises a chain's corruption rate when neighbours are misaligned.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Two-state active bag
# ---------------------------------------------------------------------------

def stationary_q(eps: float, gamma: float) -> float:
    """Stationary misalignment probability q* = eps/(eps+gamma)."""
    return eps / (eps + gamma)


def gen_active_2state(B, L, eps, gamma, pA, pM, rng, q0=None):
    """Generate B sequences of L bits from the two-state active bag.

    Returns:
        tokens : (B, L) int64 bits
        hidden : (B, L) int64 alignment state (0=A, 1=M) at each step
    """
    if q0 is None:
        q0 = stationary_q(eps, gamma)
    # initial hidden state ~ Bernoulli(q0)
    cur = (rng.random(B) < q0).astype(np.int64)
    hidden = np.empty((B, L), dtype=np.int64)
    tokens = np.empty((B, L), dtype=np.int64)
    pA = float(pA); pM = float(pM)
    for t in range(L):
        hidden[:, t] = cur
        p_emit = np.where(cur == 1, pM, pA)
        tokens[:, t] = (rng.random(B) < p_emit).astype(np.int64)
        # transition
        # from A (cur==0): -> M w.p. eps ; from M (cur==1): -> A w.p. gamma
        u = rng.random(B)
        nxt = cur.copy()
        a_mask = cur == 0
        m_mask = cur == 1
        nxt[a_mask] = (u[a_mask] < eps).astype(np.int64)          # A->M if u<eps
        nxt[m_mask] = np.where(u[m_mask] < gamma, 0, 1)           # M->A if u<gamma
        cur = nxt
    return tokens, hidden


def forward_filter_2state(tokens, eps, gamma, pA, pM, q0=None):
    """Exact forward filter for the two-state active bag.

    For each position t (0-indexed), after observing x_{0..t}:
        q[:,t]      = P(C_t = M | x_{0..t})            (filtered posterior)
    Also returns, aligned to "predicting x_{t+1}":
        q_minus[:,t]= P(C_{t+1}=M | x_{0..t})          (prior after transition)
        next1[:,t]  = P(x_{t+1}=1 | x_{0..t})          (Bayes next-symbol)

    next1[:,t] uses q_minus[:,t]; the model's logit at position t predicts x_{t+1}.
    """
    B, L = tokens.shape
    if q0 is None:
        q0 = stationary_q(eps, gamma)
    q = np.empty((B, L)); q_minus = np.empty((B, L)); next1 = np.empty((B, L))
    pA = float(pA); pM = float(pM)
    # belief BEFORE seeing x_0: prior q0 (no transition before first obs)
    qm = np.full(B, q0)  # P(C_0 = M) before obs
    for t in range(L):
        x = tokens[:, t]
        # update with emission x_t
        lik_M = np.where(x == 1, pM, 1 - pM)
        lik_A = np.where(x == 1, pA, 1 - pA)
        num = qm * lik_M
        den = num + (1 - qm) * lik_A
        qt = num / den
        q[:, t] = qt
        # predict next hidden state
        qm_next = eps + (1 - eps - gamma) * qt
        q_minus[:, t] = qm_next
        next1[:, t] = (1 - qm_next) * pA + qm_next * pM
        qm = qm_next
    return q, q_minus, next1


# ---------------------------------------------------------------------------
# Redundant active bag (Level-2 redundancy / error correction)
# ---------------------------------------------------------------------------

def gen_redundant_active(B, L, n, eps, gamma, pA, pM, rng, beta=0.0, q0=None):
    """n independent (or spread-coupled) alignment chains.

    Each chain i has its own two-state alignment Z^{(i)}_t and emits a bit
    x^{(i)}_t ~ Bernoulli(p_{Z}).  With beta>0, a chain's effective corruption rate is
    raised by beta * (fraction of OTHER chains currently misaligned) -- nearest-neighbour
    "spread" on a complete graph (mean-field), giving reproduction number R_M = beta*(n-1)/n / gamma
    (approx) -- see analysis for the exact linearization.

    Returns:
        tokens : (B, L, n) int64 bits per chain
        hidden : (B, L, n) int64 alignment per chain
        logical: (B, L) int64 majority-decoded logical alignment (1 = misaligned)
    """
    if q0 is None:
        q0 = stationary_q(eps, gamma)
    cur = (rng.random((B, n)) < q0).astype(np.int64)
    hidden = np.empty((B, L, n), dtype=np.int64)
    tokens = np.empty((B, L, n), dtype=np.int64)
    pA = float(pA); pM = float(pM)
    for t in range(L):
        hidden[:, t, :] = cur
        p_emit = np.where(cur == 1, pM, pA)
        tokens[:, t, :] = (rng.random((B, n)) < p_emit).astype(np.int64)
        # spread: fraction of OTHER chains misaligned
        frac_other = (cur.sum(axis=1, keepdims=True) - cur) / max(n - 1, 1)
        eps_eff = np.clip(eps + beta * frac_other, 0.0, 1.0)  # (B,n)
        u = rng.random((B, n))
        nxt = np.where(cur == 0,
                       (u < eps_eff).astype(np.int64),            # A->M w.p. eps_eff
                       np.where(u < gamma, 0, 1))                 # M->A w.p. gamma
        cur = nxt
    r = (n - 1) // 2
    logical = (hidden.sum(axis=2) > r).astype(np.int64)  # majority misaligned
    return tokens, hidden, logical


def pack_emissions(bits):
    """Pack the n-bit emission vector at each timestep into a single token id.
    bits: (B, L, n) -> packed (B, L) in [0, 2^n)."""
    n = bits.shape[2]
    weights = (2 ** np.arange(n)).astype(np.int64)
    return (bits * weights[None, None, :]).sum(axis=2)


def poisson_binomial_tail(probs, r):
    """P(#successes > r) for independent Bernoullis with per-trial probs.
    probs: (B, n) -> (B,) tail probability of more than r successes."""
    B, n = probs.shape
    # DP over chains: dist[b, k] = P(exactly k successes so far)
    dist = np.zeros((B, n + 1)); dist[:, 0] = 1.0
    for i in range(n):
        p = probs[:, i:i+1]
        new = np.zeros_like(dist)
        new[:, 0] = dist[:, 0] * (1 - p[:, 0])
        new[:, 1:] = dist[:, 1:] * (1 - p) + dist[:, :-1] * p
        dist = new
    return dist[:, r+1:].sum(axis=1)


def per_chain_filter(tokens_chains, eps, gamma, pA, pM):
    """Run the 2-state forward filter on each chain independently.
    tokens_chains: (B, L, n) -> q (B, L, n) filtered misalignment posteriors."""
    B, L, n = tokens_chains.shape
    q = np.empty((B, L, n))
    for i in range(n):
        qi, _, _ = forward_filter_2state(tokens_chains[:, :, i], eps, gamma, pA, pM)
        q[:, :, i] = qi
    return q


def binomial_tail(n, p, r):
    """P(Bin(n,p) > r) -- the logical misalignment rate under majority decoding that
    corrects up to r bad blocks."""
    from math import comb
    return float(sum(comb(n, j) * p**j * (1 - p)**(n - j) for j in range(r + 1, n + 1)))
