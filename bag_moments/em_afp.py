"""Single-phase four-component EM-AFP process.

The hidden state is a direct sum of four closed components:
    MD, MO, AD, AO
with d within-component states each.  Tokens are tagged content observations
``x_{tag, i}``, where tag is one of the four components and i is a content index.

Each token operator is block diagonal, so hidden trajectories never move between
components.  The observer's posterior over components moves by Bayes updates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


LEAVES = ("MD", "MO", "AD", "AO")
LEAF_TO_INDEX = {name: i for i, name in enumerate(LEAVES)}
PERSONA = np.array([0, 0, 1, 1], dtype=np.int64)  # M=0, A=1
DOMAIN = np.array([0, 1, 0, 1], dtype=np.int64)  # D=0, O=1


@dataclass(frozen=True)
class EMAFPConfig:
    d: int = 5
    beta_persona: float = 0.5
    beta_domain: float = 0.6
    delta: float = 0.05
    rho: float = 0.05
    seq_len: int = 20
    pi: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)

    @property
    def n_leaves(self) -> int:
        return 4

    @property
    def n_states(self) -> int:
        return self.n_leaves * self.d

    @property
    def vocab_size(self) -> int:
        return self.n_leaves * self.d


def evidence_matrix(cfg: EMAFPConfig) -> np.ndarray:
    """W[true_leaf, observed_tag]."""
    W = np.ones((cfg.n_leaves, cfg.n_leaves), dtype=np.float64)
    for leaf in range(cfg.n_leaves):
        for tag in range(cfg.n_leaves):
            if PERSONA[leaf] != PERSONA[tag]:
                W[leaf, tag] *= cfg.beta_persona
            if DOMAIN[leaf] != DOMAIN[tag]:
                W[leaf, tag] *= cfg.beta_domain
    return W


def content_matrix(cfg: EMAFPConfig) -> np.ndarray:
    """E[content, state] = P(content | within-state) up to the tag normalizer."""
    d = cfg.d
    E = np.full((d, d), cfg.delta / (d - 1), dtype=np.float64)
    np.fill_diagonal(E, 1.0 - cfg.delta)
    return E


def mixing_matrix(cfg: EMAFPConfig) -> np.ndarray:
    d = cfg.d
    return (1.0 - cfg.rho) * np.eye(d) + cfg.rho * np.ones((d, d)) / d


def initial_belief(cfg: EMAFPConfig) -> np.ndarray:
    pi = np.asarray(cfg.pi, dtype=np.float64)
    pi = pi / pi.sum()
    b = np.zeros((cfg.n_leaves, cfg.d), dtype=np.float64)
    b[:, :] = pi[:, None] / cfg.d
    return b


def token_id(tag: int, content: int, cfg: EMAFPConfig) -> int:
    return int(tag * cfg.d + content)


def split_token(tok: np.ndarray | int, cfg: EMAFPConfig):
    tok = np.asarray(tok)
    return tok // cfg.d, tok % cfg.d


def token_operator_parts(cfg: EMAFPConfig):
    W = evidence_matrix(cfg)
    E = content_matrix(cfg)
    K = mixing_matrix(cfg)
    Z = (1.0 + cfg.beta_persona) * (1.0 + cfg.beta_domain)
    return W, E, K, Z


def gen_em_afp(B: int, cfg: EMAFPConfig, rng: np.random.Generator, init=None):
    """Generate sequences from the closed-component HMM.

    Returns:
        obs: (B, L) token ids
        leaf_state: (B, L) true component index before emitting each token
        inner_state: (B, L) true within-component state before emitting each token
    """
    W, E, K, Z = token_operator_parts(cfg)
    L = cfg.seq_len
    b0 = initial_belief(cfg) if init is None else np.asarray(init, dtype=np.float64)
    p0 = b0.reshape(-1)
    p0 = p0 / p0.sum()
    flat_state = np.searchsorted(np.cumsum(p0), rng.random(B))
    leaf = flat_state // cfg.d
    inner = flat_state % cfg.d

    obs = np.empty((B, L), dtype=np.int64)
    leaf_state = np.empty((B, L), dtype=np.int64)
    inner_state = np.empty((B, L), dtype=np.int64)

    tag_cdfs = np.cumsum(W / Z, axis=1)
    content_cdfs = np.cumsum(E, axis=0)  # content cdf for each inner state
    mix_cdfs = np.cumsum(K, axis=1)

    for t in range(L):
        leaf_state[:, t] = leaf
        inner_state[:, t] = inner

        u_tag = rng.random(B)
        tag = (u_tag[:, None] > tag_cdfs[leaf]).sum(axis=1)

        u_content = rng.random(B)
        content = (u_content[:, None] > content_cdfs[:, inner].T).sum(axis=1)
        obs[:, t] = tag * cfg.d + content

        u_next = rng.random(B)
        inner = (u_next[:, None] > mix_cdfs[inner]).sum(axis=1)

    return obs, leaf_state, inner_state


def predict_next_from_prior(prior: np.ndarray, cfg: EMAFPConfig) -> np.ndarray:
    """Next-token distribution from a prior over current hidden states."""
    W, E, _, Z = token_operator_parts(cfg)
    out = np.zeros((prior.shape[0], cfg.vocab_size), dtype=np.float64)
    for tag in range(cfg.n_leaves):
        tag_lik = W[:, tag] / Z
        for content in range(cfg.d):
            tok = token_id(tag, content, cfg)
            out[:, tok] = (prior * tag_lik[None, :, None] * E[content][None, None, :]).sum(axis=(1, 2))
    return out


def forward_filter_em_afp(obs: np.ndarray, cfg: EMAFPConfig, init=None):
    """Exact forward filter.

    Returns arrays aligned so index t is after observing obs[:, t]:
        belief: (B, L, 4, d) posterior over current state
        mu_leaf: (B, L, 4) posterior component masses
        next_p: (B, L, V) Bayes next-token distribution
    """
    obs = np.asarray(obs, dtype=np.int64)
    B, L = obs.shape
    W, E, K, Z = token_operator_parts(cfg)
    prior0 = initial_belief(cfg) if init is None else np.asarray(init, dtype=np.float64)
    prior = np.repeat(prior0[None, :, :], B, axis=0)

    belief = np.empty((B, L, cfg.n_leaves, cfg.d), dtype=np.float64)
    mu_leaf = np.empty((B, L, cfg.n_leaves), dtype=np.float64)
    next_p = np.empty((B, L, cfg.vocab_size), dtype=np.float64)

    for t in range(L):
        tag, content = split_token(obs[:, t], cfg)
        tag_lik = W[:, tag].T / Z  # (B, 4)
        content_lik = E[content, :]  # (B, d)
        post = prior * tag_lik[:, :, None] * content_lik[:, None, :]
        post /= post.sum(axis=(1, 2), keepdims=True)

        belief[:, t] = post
        mu_leaf[:, t] = post.sum(axis=2)

        next_prior = np.einsum("bld,df->blf", post, K)
        next_p[:, t] = predict_next_from_prior(next_prior, cfg)
        prior = next_prior

    return belief, mu_leaf, next_p


def posterior_features(mu_leaf: np.ndarray) -> dict[str, np.ndarray]:
    """Convenience readouts from (..., 4) component masses."""
    eps = 1e-9
    md = mu_leaf[..., LEAF_TO_INDEX["MD"]]
    mo = mu_leaf[..., LEAF_TO_INDEX["MO"]]
    ad = mu_leaf[..., LEAF_TO_INDEX["AD"]]
    ao = mu_leaf[..., LEAF_TO_INDEX["AO"]]
    return {
        "mu_md": md,
        "mu_m": md + mo,
        "mu_d": md + ad,
        "chi": np.log((md * ao + eps) / (mo * ad + eps)),
    }


def component_posterior_from_tag_probs(tag_probs: np.ndarray, cfg: EMAFPConfig) -> np.ndarray:
    """Invert y = mu W / Z for component posterior estimates from tag probabilities."""
    W = evidence_matrix(cfg)
    Z = (1.0 + cfg.beta_persona) * (1.0 + cfg.beta_domain)
    mu = (tag_probs * Z) @ np.linalg.inv(W)
    mu = np.maximum(mu, 0.0)
    denom = mu.sum(axis=-1, keepdims=True)
    return np.divide(mu, denom, out=np.full_like(mu, 1.0 / cfg.n_leaves), where=denom > 0)
