"""
GHMM active bag: an alignment switch (A/M) MODULATING a 3-state nonunifilar (Mess3-style)
emitter, so each rollout carries WITHIN-BLOCK belief geometry on top of the A/M switch.

Motivation. In the two-state active bag (active.py) each block emits a single Bernoulli bit,
so the only latent is the alignment posterior q_t. Real models carry *content/capability*
structure too. We give each block an internal 3-state nonunifilar process (Mess3) whose
mixed-state (belief) geometry is the classic Sierpinski-like fractal in the 2-simplex (Shai
et al. 2405.15943). The hidden alignment state c_t in {A,M} modulates the Mess3 dynamics,
switching via the same (eps, gamma) kernel. The optimal predictor must track a JOINT 6-state
belief over (alignment c) x (Mess3 state s); from it we read two latents:
    * alignment posterior   q_t = P(c_t = M | o_{1:t})   ->  z_t = logit q_t
    * Mess3 belief simplex   m_t(s) = P(s_t = s | o_{1:t})  (a point in the 2-simplex)
The interpretability question (the "persona (x) capability factorisation"): does the
transformer linearly represent BOTH z_t and m_t, and in (near-)separable subspaces, so that
steering alignment leaves the capability (Mess3 belief) representation intact?

Two modulation regimes:
  * joint_params_drift (HEADLINE / near-factored): alignment sets the DRIFT DIRECTION of the
    content cycle (aligned +1, misaligned -1) with MATCHED stickiness and emission, so belief
    SHARPNESS is ~alignment-symmetric (the latents are only weakly correlated, |corr|~0.2) and
    alignment is carried by the cycling direction. This is the regime used for the factorisation
    claim, because the two latents are close to statistically independent.
  * joint_params (sharpness-modulated, contrast): alignment modulates transition rate x (sticky
    vs erratic), which makes belief sharpness depend strongly on alignment (corr~-0.8) -- the
    latents are ENTANGLED, so it is only a contrast, NOT used for the factorisation claim.

Everything here is exact (closed-form 6-state HMM forward filter) so it anchors the network.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Mess3 single-mode building blocks
# ---------------------------------------------------------------------------

def mess3_emit(a):
    """Emission matrix E[s_obs, state] = P(emit symbol s_obs | hidden state). 3x3.
    Each state prefers its own symbol with prob a; the other two share (1-a)/2."""
    b = (1.0 - a) / 2.0
    E = np.full((3, 3), b)
    np.fill_diagonal(E, a)
    return E  # E[obs, state]


def mess3_trans(x):
    """Symmetric state transition matrix Tr[i->j] = P(next=j | cur=i). 3x3.
    Stay with prob 1-2x, move to each of the other two with prob x. Doubly stochastic;
    stationary distribution is uniform (1/3,1/3,1/3)."""
    Tr = np.full((3, 3), x)
    np.fill_diagonal(Tr, 1.0 - 2.0 * x)
    return Tr  # Tr[from, to]


def mess3_trans_drift(stay, drift):
    """Circulant 3-state transition with a DRIFT direction (used for the factored regime).
        P(s->s)            = stay
        P(s->(s+drift)%3)  = (1-stay) * f          (move with the drift)
        P(s->(s-drift)%3)  = (1-stay) * (1-f)      (move against)
    with f fixed high (0.9) so the chain cycles in the `drift` direction. Doubly stochastic
    (stationary uniform). The aligned/misaligned modes use drift=+1 / drift=-1 with the SAME
    `stay`, so belief SHARPNESS is identical across modes (decoupling capability-belief
    geometry from the alignment latent); alignment is identifiable only from the cycling
    DIRECTION of the symbol stream, which is ~orthogonal to the within-step belief location."""
    f = 0.9
    Tr = np.zeros((3, 3))
    for s in range(3):
        Tr[s, s] = stay
        Tr[s, (s + drift) % 3] = (1 - stay) * f
        Tr[s, (s - drift) % 3] = (1 - stay) * (1 - f)
    return Tr


# ---------------------------------------------------------------------------
# Alignment-modulated GHMM: joint hidden state (c, s), 6 states; index 3*c + s
# ---------------------------------------------------------------------------

def joint_params_from(eps, gamma, TrA, TrM, EA, EM, meta=None):
    """Build the 6-state joint HMM from explicit mode transition/emission matrices.
        EA, EM : emission E[obs, state] (3x3) for aligned / misaligned
        TrA,TrM: Mess3 transition Tr[from,to] (3x3) for aligned / misaligned (uses CURRENT mode)
    """
    K = np.array([[1 - eps, eps], [gamma, 1 - gamma]])
    Eobs = np.zeros((3, 6)); Eobs[:, 0:3] = EA; Eobs[:, 3:6] = EM
    Tr_by_mode = [TrA, TrM]
    Tj = np.zeros((6, 6))
    for c in range(2):
        Tr = Tr_by_mode[c]
        for s in range(3):
            i = 3 * c + s
            for cp in range(2):
                for sp in range(3):
                    Tj[i, 3 * cp + sp] = K[c, cp] * Tr[s, sp]
    out = {"K": K, "TrA": TrA, "TrM": TrM, "EA": EA, "EM": EM, "Eobs": Eobs, "Tjoint": Tj,
           "eps": eps, "gamma": gamma}
    if meta:
        out.update(meta)
    return out


def joint_params_drift(eps, gamma, stay, a):
    """Factored regime: alignment sets the DRIFT DIRECTION of the content cycle (aligned=+1,
    misaligned=-1) with matched `stay` (sharpness) and matched emission concentration `a`.
    Belief geometry is then alignment-symmetric; alignment is carried by cycling direction."""
    TrA = mess3_trans_drift(stay, +1); TrM = mess3_trans_drift(stay, -1)
    E = mess3_emit(a)
    return joint_params_from(eps, gamma, TrA, TrM, E, E,
                             meta={"regime": "drift", "stay": stay, "a": a})


def joint_params(eps, gamma, xA, xM, aA, aM):
    """Build the 6-state joint HMM where the alignment state c modulates the WITHIN-BLOCK
    Mess3 dynamics:
        K_align[c->c']  (2x2),
        mode-dependent transition Tr_A = mess3_trans(xA), Tr_M = mess3_trans(xM),
        mode-dependent emission   EA = mess3_emit(aA),    EM = mess3_emit(aM),
        packed into Eobs[obs, (c,s)] (3x6) and Tjoint[(c,s)->(c',s')] (6x6).

    Default headline regime uses PURE-DYNAMICS modulation: aA == aM (so the per-symbol
    emission carries NO alignment info and the marginal symbol law is uniform in both
    modes), while xA != xM (aligned = sticky/coherent, misaligned = erratic). Then the
    alignment latent z is a TEMPORAL-COHERENCE property that genuinely requires integrating
    history (no last-symbol shortcut), and the Mess3 belief is the content latent. The
    transition under (c,s) uses the CURRENT mode c.
    """
    K = np.array([[1 - eps, eps], [gamma, 1 - gamma]])  # rows=from c, cols=to c'
    TrA = mess3_trans(xA); TrM = mess3_trans(xM)
    EA = mess3_emit(aA); EM = mess3_emit(aM)
    Eobs = np.zeros((3, 6))
    Eobs[:, 0:3] = EA   # c=A
    Eobs[:, 3:6] = EM   # c=M
    Tr_by_mode = [TrA, TrM]
    Tj = np.zeros((6, 6))
    for c in range(2):
        Tr = Tr_by_mode[c]
        for s in range(3):
            i = 3 * c + s
            for cp in range(2):
                for sp in range(3):
                    j = 3 * cp + sp
                    Tj[i, j] = K[c, cp] * Tr[s, sp]
    return {"K": K, "TrA": TrA, "TrM": TrM, "EA": EA, "EM": EM, "Eobs": Eobs, "Tjoint": Tj,
            "eps": eps, "gamma": gamma, "xA": xA, "xM": xM, "aA": aA, "aM": aM}


def stationary_joint(P):
    """Stationary distribution over the 6 joint states = align-stationary (x) uniform mess3."""
    eps, gamma = P["eps"], P["gamma"]
    qstar = eps / (eps + gamma)
    pi_align = np.array([1 - qstar, qstar])
    pi = np.zeros(6)
    for c in range(2):
        for s in range(3):
            pi[3 * c + s] = pi_align[c] * (1.0 / 3.0)
    return pi


def gen_ghmm(B, L, P, rng, init=None):
    """Generate B sequences of L symbols (in {0,1,2}) from the alignment-modulated Mess3.

    Returns:
        obs    : (B, L) int64 symbols
        cstate : (B, L) int64 alignment (0=A,1=M)
        sstate : (B, L) int64 mess3 state {0,1,2}
    """
    Tj = P["Tjoint"]; Eobs = P["Eobs"]
    pi = stationary_joint(P) if init is None else init
    # sample initial joint states
    cum_pi = np.cumsum(pi)
    js = np.searchsorted(cum_pi, rng.random(B))  # (B,) in [0,6)
    obs = np.empty((B, L), dtype=np.int64)
    cstate = np.empty((B, L), dtype=np.int64)
    sstate = np.empty((B, L), dtype=np.int64)
    # precompute emission cdf per joint state, transition cdf per joint state
    Ecdf = np.cumsum(Eobs, axis=0)         # (3 obs, 6) -> cdf over obs per state
    Tcdf = np.cumsum(Tj, axis=1)           # (6, 6) cdf over next state
    for t in range(L):
        c = js // 3; s = js % 3
        cstate[:, t] = c; sstate[:, t] = s
        # emit
        u = rng.random(B)
        o = (u[None, :] > Ecdf[:, js]).sum(axis=0)  # number of thresholds passed
        obs[:, t] = o
        # transition
        u2 = rng.random(B)
        js = (u2[:, None] > Tcdf[js]).sum(axis=1)
    return obs, cstate, sstate


def forward_filter_ghmm(obs, P, init=None):
    """Exact 6-state forward filter.

    Returns (all aligned so index t = after observing o_{0..t}):
        belief   : (B, L, 6) filtered joint posterior P((c,s)_t | o_{0..t})
        q        : (B, L)    alignment posterior P(c_t = M | o_{0..t})
        mbelief  : (B, L, 3) Mess3 marginal belief P(s_t = . | o_{0..t})
        next_p   : (B, L, 3) Bayes next-symbol distribution P(o_{t+1} = . | o_{0..t})
    """
    B, L = obs.shape
    Tj = P["Tjoint"]; Eobs = P["Eobs"]
    pi = stationary_joint(P) if init is None else init
    belief = np.empty((B, L, 6))
    q = np.empty((B, L))
    mbelief = np.empty((B, L, 3))
    next_p = np.empty((B, L, 3))
    bm = np.tile(pi, (B, 1))  # prior P((c,s)_0) before obs
    for t in range(L):
        o = obs[:, t]
        lik = Eobs[o, :]               # (B,6) P(o_t | state)
        post = bm * lik
        post /= post.sum(axis=1, keepdims=True)
        belief[:, t, :] = post
        q[:, t] = post[:, 3:6].sum(axis=1)               # P(M)
        mbelief[:, t, :] = post[:, 0:3] + post[:, 3:6]   # marginal over s
        # propagate
        bnext = post @ Tj                                # (B,6)
        # next-symbol distribution: sum over next states of P(next state)*P(obs|state)
        next_p[:, t, :] = bnext @ Eobs.T                 # (B,3)
        bm = bnext
    return belief, q, mbelief, next_p


def simplex_xy(m):
    """Map 2-simplex belief rows (..., 3) to 2D barycentric coords for plotting / decoding.
    Vertices at (0,0),(1,0),(0.5,sqrt3/2)."""
    m = np.asarray(m)
    x = m[..., 1] + 0.5 * m[..., 2]
    y = (np.sqrt(3) / 2.0) * m[..., 2]
    return np.stack([x, y], axis=-1)
