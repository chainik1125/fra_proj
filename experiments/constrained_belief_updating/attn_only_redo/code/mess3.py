"""Single-Mess3 HMM: data generation + exact Bayes / constrained-belief analytics.

Convention (STANDARD Mess3, verified to satisfy the P2 invariants):

    T^{(z)}[i, j] = t(i, j) * e(z | j)
    t = (1 - 2x) I + x (1 - I)          # hidden-state transition
    e(z | j) = a if z == j else (1-a)/2  # emission from the NEXT state

so T = sum_z T^{(z)} = t, stationary pi = (1/3, 1/3, 1/3), eigenvalues of T
are {1, zeta, zeta} with zeta = 1 - 3x, and P(z) = pi T^{(z)} 1 = 1/3.

NOTE: experiments/fra_hmm_toy/mixture_data.py::mess3_transitions uses a
DIFFERENT (unnormalized) fill whose combined T has row sums 1+x and
eigenvalues {1+x, (1+x)(3a-1)/2 x2}; it does NOT satisfy the invariants
above. This module deliberately uses the standard convention instead
(checked numerically in `run_checks`).

Belief conventions (0-based token arrays z[0..L-1]):
    eta[n]  = full Bayes belief over the hidden state AFTER observing z[:n]
              (eta[0] = pi;  eta[n] = normalize(eta[n-1] @ T^{(z[n-1])}))
    r1[d]   = constrained belief (P2 eq. constrained-belief):
              r1[d] = pi + sum_{s=1..d} (pi T^{|z_s} T^{d-s} - pi)
              with T^{|z} = T^{(z)} / P(z),  P(z) = pi T^{(z)} 1.
    next-token distribution from a belief eta:  p(z') = eta @ T^{(z')} @ 1.

The transformer's residual at (0-based) position t has seen z[:t+1] and
predicts z[t+1]; the aligned beliefs are eta[t+1] / r1[t+1].
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

X_DEFAULT = 0.15
A_DEFAULT = 0.6
OUT_DIR = Path(__file__).resolve().parent / "out"


def enumerate_sequences(length: int, vocab: int = 3) -> np.ndarray:
    """All vocab^length sequences, lexicographic, FIRST token most significant.

    Sequences sharing a length-d prefix are contiguous blocks of size
    vocab^(length-d); row i*vocab^(length-d) is the first with prefix id i.
    """
    n = vocab ** length
    idx = np.arange(n)
    seqs = np.zeros((n, length), dtype=np.int64)
    for p in range(length):
        seqs[:, p] = (idx // vocab ** (length - 1 - p)) % vocab
    return seqs


def prefix_probs(m: "Mess3", seqs: np.ndarray) -> np.ndarray:
    """(n, L+1) exact P(z_{1:d}) for d = 0..L via the forward algorithm."""
    n, L = seqs.shape
    out = np.ones((n, L + 1))
    alpha = np.tile(m.pi, (n, 1))
    for d in range(L):
        alpha = np.einsum("ni,nij->nj", alpha, m.Tz[seqs[:, d]])
        out[:, d + 1] = alpha.sum(axis=1)
    return out


def mess3_matrices(x: float = X_DEFAULT, a: float = A_DEFAULT) -> np.ndarray:
    """(3, 3, 3) array Tz[z, i, j] = P(emit z AND go i->j | state i)."""
    b = (1.0 - a) / 2.0
    t = (1.0 - 2.0 * x) * np.eye(3) + x * (1.0 - np.eye(3))
    e = np.where(np.eye(3, dtype=bool), a, b)  # e[z, j] = P(z | next state j)
    return np.stack([t * e[z][None, :] for z in range(3)])  # (3,3,3)


class Mess3:
    def __init__(self, x: float = X_DEFAULT, a: float = A_DEFAULT):
        self.x, self.a = x, a
        self.Tz = mess3_matrices(x, a)          # (V=3, S=3, S=3)
        self.T = self.Tz.sum(axis=0)            # (3, 3) marginal transition
        self.pi = np.ones(3) / 3.0
        self.zeta = 1.0 - 3.0 * x
        self.pz = np.array([self.pi @ self.Tz[z] @ np.ones(3) for z in range(3)])
        self.T_cond = self.Tz / self.pz[:, None, None]   # T^{|z}
        # emission map belief -> next-token distribution: p = eta @ M, M[i,z]
        self.M = self.Tz.sum(axis=2).T          # (S, V): M[i, z] = (T^{(z)} 1)_i

    # ── sampling (predictive-distribution sampling; exact for the process) ──

    def sample(self, n_seqs: int, length: int, rng: np.random.Generator,
               return_beliefs: bool = False):
        """Sample tokens (n, L). If return_beliefs, also eta (n, L+1, 3)."""
        belief = np.tile(self.pi, (n_seqs, 1))
        tokens = np.zeros((n_seqs, length), dtype=np.int64)
        etas = np.zeros((n_seqs, length + 1, 3)) if return_beliefs else None
        if return_beliefs:
            etas[:, 0] = self.pi
        for t in range(length):
            p = belief @ self.M                      # (n, V)
            p = p / p.sum(axis=1, keepdims=True)
            u = rng.random(n_seqs)
            tok = (p.cumsum(axis=1) < u[:, None]).sum(axis=1)
            tok = np.clip(tok, 0, 2)
            tokens[:, t] = tok
            new = np.einsum("ni,nij->nj", belief, self.Tz[tok])
            belief = new / new.sum(axis=1, keepdims=True)
            if return_beliefs:
                etas[:, t + 1] = belief
        if return_beliefs:
            return tokens, etas
        return tokens

    # ── analytics ──

    def bayes_beliefs(self, tokens: np.ndarray) -> np.ndarray:
        """(n, L+1, 3) full-Bayes eta; eta[:, 0] = pi."""
        n, L = tokens.shape
        etas = np.zeros((n, L + 1, 3))
        etas[:, 0] = self.pi
        belief = np.tile(self.pi, (n, 1))
        for t in range(L):
            new = np.einsum("ni,nij->nj", belief, self.Tz[tokens[:, t]])
            belief = new / new.sum(axis=1, keepdims=True)
            etas[:, t + 1] = belief
        return etas

    def constrained_beliefs(self, tokens: np.ndarray) -> np.ndarray:
        """(n, L+1, 3) constrained r1; r1[:, 0] = pi.

        r1[d] = pi + u[d],  u[d] = u[d-1] @ T + (pi T^{|z_d} - pi)
        (valid because pi T^{d-s} = pi, so each correction propagates by T).
        """
        n, L = tokens.shape
        c = np.stack([self.pi @ self.T_cond[z] - self.pi for z in range(3)])  # (V,3)
        r1 = np.zeros((n, L + 1, 3))
        r1[:, 0] = self.pi
        u = np.zeros((n, 3))
        for d in range(1, L + 1):
            u = u @ self.T + c[tokens[:, d - 1]]
            r1[:, d] = self.pi + u
        return r1

    def next_token_dist(self, beliefs: np.ndarray) -> np.ndarray:
        """(..., 3) belief -> (..., 3) next-token distribution."""
        return beliefs @ self.M

    def bayes_ce(self, tokens: np.ndarray, t_min: int = 0) -> float:
        """Mean -log p(z_{t+1} | eta_t) over query positions t >= t_min.

        Query position t predicts tokens[:, t+1] from eta after tokens[:, :t+1].
        """
        etas = self.bayes_beliefs(tokens)
        p = self.next_token_dist(etas[:, 1:-1])          # aligned with pos t=0..L-2
        tgt = tokens[:, 1:]
        pt = np.take_along_axis(p, tgt[..., None], axis=2)[..., 0]
        return float(-np.log(np.clip(pt[:, t_min:], 1e-12, None)).mean())

    def entropy_rate_estimate(self, n_seqs: int = 400, length: int = 512,
                              burn: int = 64, seed: int = 7) -> float:
        """Estimate h = E[H(p(.|eta))] by simulating belief dynamics."""
        rng = np.random.default_rng(seed)
        tokens, etas = self.sample(n_seqs, length, rng, return_beliefs=True)
        p = self.next_token_dist(etas[:, burn:-1])
        h = -(p * np.log(np.clip(p, 1e-12, None))).sum(axis=-1)
        return float(h.mean())


# ── checks ──

def run_checks(save: bool = True) -> dict:
    m = Mess3()
    rng = np.random.default_rng(0)
    checks: dict = {"x": m.x, "a": m.a, "zeta": m.zeta}

    # structural invariants
    eigs = np.sort(np.linalg.eigvals(m.T).real)
    checks["T_row_sums"] = m.T.sum(axis=1).tolist()
    checks["T_eigs"] = eigs.tolist()
    checks["eigs_match_{1,zeta,zeta}"] = bool(
        np.allclose(eigs, np.sort([1.0, m.zeta, m.zeta]), atol=1e-12))
    checks["pi_stationary"] = bool(np.allclose(m.pi @ m.T, m.pi, atol=1e-12))
    checks["P_z"] = m.pz.tolist()
    # pi T^{|z} is the posterior over next state given z
    for z in range(3):
        post = m.pi @ m.T_cond[z]
        assert abs(post.sum() - 1.0) < 1e-12

    # r1 vs eta on short contexts
    tokens, etas = m.sample(2000, 8, rng, return_beliefs=True)
    r1 = m.constrained_beliefs(tokens)
    assert np.allclose(m.bayes_beliefs(tokens), etas, atol=1e-10)
    checks["r1_sums_to_1_max_err"] = float(np.abs(r1.sum(-1) - 1).max())
    checks["r1_min_component"] = float(r1.min())
    checks["eta_min_component"] = float(etas.min())
    diff = np.abs(r1 - etas)
    checks["r1_vs_eta_max_abs_diff_d1"] = float(diff[:, 1].max())   # should be ~0
    checks["r1_vs_eta_max_abs_diff_d2"] = float(diff[:, 2].max())
    checks["r1_vs_eta_max_abs_diff_d8"] = float(diff[:, 8].max())
    checks["r1_vs_eta_mean_abs_diff_d8"] = float(diff[:, 8].mean())

    # Bayes CE on sampled data vs entropy-rate estimate
    tokens_long = m.sample(400, 512, np.random.default_rng(1))
    ce = m.bayes_ce(tokens_long, t_min=64)
    h = m.entropy_rate_estimate()
    checks["bayes_ce_sampled"] = ce
    checks["entropy_rate_estimate"] = h
    checks["ce_vs_entropy_abs_diff"] = abs(ce - h)
    checks["unigram_ce"] = float(np.log(3.0))

    ok = (
        checks["eigs_match_{1,zeta,zeta}"] and checks["pi_stationary"]
        and checks["r1_sums_to_1_max_err"] < 1e-10
        and checks["r1_vs_eta_max_abs_diff_d1"] < 1e-10
        and checks["ce_vs_entropy_abs_diff"] < 5e-3
    )
    checks["all_ok"] = bool(ok)
    if save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "mess3_checks.json").write_text(json.dumps(checks, indent=2))
    return checks


if __name__ == "__main__":
    for k, v in run_checks().items():
        print(f"{k}: {v}")
