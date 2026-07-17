"""Hard symmetric two-state-factor SFP HMM.

The process is (T_M direct-sum T_A) tensor (T_D direct-sum T_O).  Each factor
has a neutral state and a special state.  Global tokens are pairs of persona-side
and domain-side emissions.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


LEAVES = ("MD", "MO", "AD", "AO")
LEAF_TO_INDEX = {name: i for i, name in enumerate(LEAVES)}
PERSONA_SYMS = ("0", "1", "S_M", "S_A")
DOMAIN_SYMS = ("0", "1", "S_D", "S_O")


@dataclass(frozen=True)
class SpecialSFPConfig:
    p: float = 0.5
    epsilon: float = 0.02
    p_s: float = 0.8
    alpha_wrong_special: float = 0.0
    p_persona: float | None = None
    p_domain: float | None = None
    epsilon_persona: float | None = None
    epsilon_persona_m: float | None = None
    epsilon_persona_a: float | None = None
    epsilon_domain: float | None = None
    p_s_persona: float | None = None
    p_s_domain: float | None = None
    seq_len: int = 64
    pi: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)

    @property
    def n_leaves(self) -> int:
        return 4

    @property
    def states_per_leaf(self) -> int:
        return 4

    @property
    def n_states(self) -> int:
        return self.n_leaves * self.states_per_leaf

    @property
    def vocab_size(self) -> int:
        return 16


def token_id(persona_sym: int, domain_sym: int) -> int:
    return int(persona_sym * 4 + domain_sym)


def split_token(tok: np.ndarray | int) -> tuple[np.ndarray, np.ndarray]:
    tok = np.asarray(tok)
    return tok // 4, tok % 4


def local_ops(
    cfg: SpecialSFPConfig,
    own_special_idx: int,
    p: float | None = None,
    epsilon: float | None = None,
    p_s: float | None = None,
) -> np.ndarray:
    """Return local token operators for symbols 0, 1, special-0, special-1.

    Rows and columns are ordered N, U.  `own_special_idx` is either 2 or 3.
    The other special token has a zero operator.
    """
    p = cfg.p if p is None else p
    epsilon = cfg.epsilon if epsilon is None else epsilon
    p_s = cfg.p_s if p_s is None else p_s
    alpha = cfg.alpha_wrong_special
    if epsilon < 0 or epsilon > min(p, 1.0 - p):
        raise ValueError("epsilon must be in [0, min(p, 1-p)]")
    if alpha < 0 or alpha >= 1:
        raise ValueError("alpha_wrong_special must be in [0, 1)")
    ops = np.zeros((4, 2, 2), dtype=np.float64)
    ops[0] = np.array([[p - epsilon, epsilon], [0.0, 0.0]])
    ops[1] = np.array([[1.0 - p - epsilon, epsilon], [0.0, 0.0]])
    wrong_special_idx = 5 - own_special_idx
    ops[own_special_idx] = (1.0 - alpha) * np.array([[0.0, 0.0], [1.0 - p_s, p_s]])
    ops[wrong_special_idx] = alpha * np.array([[0.0, 0.0], [1.0 - p_s, p_s]])
    return ops


def token_operators(cfg: SpecialSFPConfig) -> np.ndarray:
    """Global token operators, shape (vocab, states, states)."""
    T = np.zeros((cfg.vocab_size, cfg.n_states, cfg.n_states), dtype=np.float64)
    # Leaf order: MD, MO, AD, AO.
    leaf_specs = [
        (2, 2),  # M, D
        (2, 3),  # M, O
        (3, 2),  # A, D
        (3, 3),  # A, O
    ]
    base_persona_kwargs = {
        "p": cfg.p if cfg.p_persona is None else cfg.p_persona,
        "epsilon": cfg.epsilon if cfg.epsilon_persona is None else cfg.epsilon_persona,
        "p_s": cfg.p_s if cfg.p_s_persona is None else cfg.p_s_persona,
    }
    domain_kwargs = {
        "p": cfg.p if cfg.p_domain is None else cfg.p_domain,
        "epsilon": cfg.epsilon if cfg.epsilon_domain is None else cfg.epsilon_domain,
        "p_s": cfg.p_s if cfg.p_s_domain is None else cfg.p_s_domain,
    }
    for leaf_i, (persona_special, domain_special) in enumerate(leaf_specs):
        persona_kwargs = dict(base_persona_kwargs)
        if persona_special == 2 and cfg.epsilon_persona_m is not None:
            persona_kwargs["epsilon"] = cfg.epsilon_persona_m
        if persona_special == 3 and cfg.epsilon_persona_a is not None:
            persona_kwargs["epsilon"] = cfg.epsilon_persona_a
        p_ops = local_ops(cfg, persona_special, **persona_kwargs)
        d_ops = local_ops(cfg, domain_special, **domain_kwargs)
        start = leaf_i * cfg.states_per_leaf
        stop = start + cfg.states_per_leaf
        for ps in range(4):
            for ds in range(4):
                T[token_id(ps, ds), start:stop, start:stop] = np.kron(p_ops[ps], d_ops[ds])
    return T


def transition_matrix(cfg: SpecialSFPConfig) -> np.ndarray:
    return token_operators(cfg).sum(axis=0)


def initial_belief(cfg: SpecialSFPConfig) -> np.ndarray:
    pi = np.asarray(cfg.pi, dtype=np.float64)
    pi = pi / pi.sum()
    b = np.zeros(cfg.n_states, dtype=np.float64)
    for leaf_i, mass in enumerate(pi):
        b[leaf_i * cfg.states_per_leaf] = mass  # (N_persona, N_domain)
    return b


def md_config(cfg: SpecialSFPConfig) -> SpecialSFPConfig:
    return replace(cfg, pi=(1.0, 0.0, 0.0, 0.0))


def predict_next(prior: np.ndarray, ops: np.ndarray) -> np.ndarray:
    return np.einsum("bs,vst->bv", prior, ops)


def gen_special_sfp(
    batch: int,
    cfg: SpecialSFPConfig,
    rng: np.random.Generator,
    init: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    ops = token_operators(cfg)
    b0 = initial_belief(cfg) if init is None else np.asarray(init, dtype=np.float64)
    b0 = b0 / b0.sum()
    state = np.searchsorted(np.cumsum(b0), rng.random(batch))
    obs = np.empty((batch, cfg.seq_len), dtype=np.int64)
    states = np.empty((batch, cfg.seq_len), dtype=np.int64)

    flat = ops.reshape(cfg.vocab_size, cfg.n_states, cfg.n_states).transpose(1, 0, 2)
    flat = flat.reshape(cfg.n_states, cfg.vocab_size * cfg.n_states)
    cdfs = np.cumsum(flat, axis=1)
    cdfs[:, -1] = 1.0

    for t in range(cfg.seq_len):
        states[:, t] = state
        u = rng.random(batch)
        event = (u[:, None] > cdfs[state]).sum(axis=1)
        obs[:, t] = event // cfg.n_states
        state = event % cfg.n_states
    return obs, states


def forward_filter(obs: np.ndarray, cfg: SpecialSFPConfig, init: np.ndarray | None = None):
    obs = np.asarray(obs, dtype=np.int64)
    batch, length = obs.shape
    ops = token_operators(cfg)
    prior0 = initial_belief(cfg) if init is None else np.asarray(init, dtype=np.float64)
    prior0 = prior0 / prior0.sum()
    prior = np.repeat(prior0[None, :], batch, axis=0)
    belief = np.empty((batch, length, cfg.n_states), dtype=np.float64)
    mu = np.empty((batch, length, cfg.n_leaves), dtype=np.float64)
    next_p = np.empty((batch, length, cfg.vocab_size), dtype=np.float64)
    likelihood = np.empty((batch, length), dtype=np.float64)

    for t in range(length):
        post = np.einsum("bs,bst->bt", prior, ops[obs[:, t]])
        z = post.sum(axis=1, keepdims=True)
        likelihood[:, t] = z[:, 0]
        post = np.divide(post, z, out=np.zeros_like(post), where=z > 0)
        belief[:, t] = post
        mu[:, t] = post.reshape(batch, cfg.n_leaves, cfg.states_per_leaf).sum(axis=2)
        next_p[:, t] = predict_next(post, ops)
        prior = post
    return belief, mu, next_p, likelihood


def leaf_posterior_from_belief(belief: np.ndarray, cfg: SpecialSFPConfig) -> np.ndarray:
    return belief.reshape(*belief.shape[:-1], cfg.n_leaves, cfg.states_per_leaf).sum(axis=-1)


def prompt_token(domain: str, neutral_persona: int = 0) -> list[int]:
    """In-support persona-neutral domain prompt plus neutral readout token.

    The first neutral token can transition the domain factor to U.  The second
    token emits the domain special.  The third neutral token lets the persona
    factor possibly transition to U, so the next-step S_M/S_A odds read out
    persona odds.
    """
    if domain == "D":
        special = 2
    elif domain == "O":
        special = 3
    else:
        raise ValueError("domain must be D or O")
    return [token_id(neutral_persona, 0), token_id(neutral_persona, special), token_id(neutral_persona, 0)]


def persona_special_odds_from_probs(probs: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p_sm = probs[:, [token_id(2, d) for d in range(4)]].sum(axis=1)
    p_sa = probs[:, [token_id(3, d) for d in range(4)]].sum(axis=1)
    return p_sm, p_sa, (p_sm + eps) / (p_sa + eps)
