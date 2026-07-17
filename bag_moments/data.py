"""
Tokenized data generators for the bag-of-coins experiments.

Token scheme:
    0 -> bit 0
    1 -> bit 1
    2 -> <ROLL> rollout delimiter (multi-rollout protocols only)

The model is NEVER shown N or the component identity -- only bits and delimiters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BIT0, BIT1, ROLL = 0, 1, 2


# ----------------------------------------------------------------------------
# Annealed single-rollout protocol (Experiment 1)
# ----------------------------------------------------------------------------


def gen_annealed_single(B: int, L: int, rng: np.random.Generator):
    """Fresh bag per rollout, one rollout observed.

    Marginally identical to: p ~ Unif[0,1]; x_t ~iid Bern(p).  The bag size N is
    irrelevant (barycenter collapse), so we don't even need to draw it.

    Returns tokens (B, L) int64 and the latent biases p (B,).
    """
    p = rng.random(B)
    x = (rng.random((B, L)) < p[:, None]).astype(np.int64)
    return x, p


# ----------------------------------------------------------------------------
# Quenched multi-rollout protocol (Experiments 2, 3)
# ----------------------------------------------------------------------------


@dataclass
class QuenchedBatch:
    tokens: np.ndarray          # (B, seqlen) int64
    roll_lens: tuple[int, ...]  # length of each rollout (fixed across batch)
    roll_token_pos: np.ndarray  # (J-1,) indices of the ROLL delimiter tokens
    roll_start_pos: np.ndarray  # (J,) index where each rollout's bits start
    N: np.ndarray               # (B,) realized bag size
    biases: list                # list of (N_b,) arrays, per-sequence coin biases
    assign: np.ndarray          # (B, J) which coin each rollout used
    bits: np.ndarray            # (B, J, max_len) the raw bits (padded with -1)

    @property
    def seqlen(self) -> int:
        return self.tokens.shape[1]


def gen_quenched(B: int, roll_lens, nprior, rng: np.random.Generator,
                 independent_bags: bool = False) -> QuenchedBatch:
    """One bag reused across J = len(roll_lens) rollouts.  Fully vectorized.

    Equal-weight bags: each sequence draws N ~ prior and a pool of N iid Unif[0,1]
    coin biases; each rollout selects a coin index c = floor(U * N) (uniform over
    {0..N-1}); two rollouts share a coin iff their indices collide.

    If independent_bags=True, each rollout instead gets a *fresh* coin (annealed
    control): collision structure destroyed, per-rollout marginal unchanged.
    """
    J = len(roll_lens)
    max_len = max(roll_lens)
    N = nprior.sample(rng, B).astype(np.int64)
    maxN = int(N.max())

    # bias pool per sequence (unused entries beyond N_b are never selected)
    pool = rng.random((B, maxN))
    assign = np.zeros((B, J), dtype=np.int64)
    pj = np.zeros((B, J))
    bidx = np.arange(B)
    for j in range(J):
        if independent_bags:
            pj[:, j] = rng.random(B)
            assign[:, j] = -1
        else:
            c = np.floor(rng.random(B) * N).astype(np.int64)  # uniform in {0..N-1}
            c = np.minimum(c, N - 1)
            assign[:, j] = c
            pj[:, j] = pool[bidx, c]

    bits = np.full((B, J, max_len), -1, dtype=np.int64)
    for j in range(J):
        Lj = roll_lens[j]
        bits[:, j, :Lj] = (rng.random((B, Lj)) < pj[:, j, None]).astype(np.int64)
    biases = pool

    # assemble tokens with ROLL delimiters between rollouts
    seqlen = sum(roll_lens) + (J - 1)
    tokens = np.zeros((B, seqlen), dtype=np.int64)
    roll_token_pos = []
    roll_start_pos = []
    col = 0
    for j in range(J):
        roll_start_pos.append(col)
        Lj = roll_lens[j]
        tokens[:, col : col + Lj] = bits[:, j, :Lj]
        col += Lj
        if j < J - 1:
            tokens[:, col] = ROLL
            roll_token_pos.append(col)
            col += 1

    return QuenchedBatch(
        tokens=tokens,
        roll_lens=tuple(roll_lens),
        roll_token_pos=np.array(roll_token_pos),
        roll_start_pos=np.array(roll_start_pos),
        N=N,
        biases=biases,
        assign=assign,
        bits=bits,
    )


def running_counts(bits_row: np.ndarray, Lj: int):
    """Return arrays (s_after_k, k) for a single rollout's bits, k=0..Lj."""
    valid = bits_row[:Lj]
    s = np.concatenate([[0], np.cumsum(valid)])
    t = np.arange(Lj + 1)
    return s, t
