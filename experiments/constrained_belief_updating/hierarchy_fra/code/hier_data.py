"""Setting-H sequence generator (continuous 2-feature, gated) + exact-filter anchors.

Observation a_t = c_P d_P + c_C d_C + b, gated hierarchy (reset-on-parent-death).
Predict a_{t+1} (MSE, attn-only, per the reset predecessor). Anchors by exact
enumeration over the joint 3-state chain: (i) Bayes-filter MSE floor, (ii) best
lag-only-additive (constrained) MSE floor; their gap ~ Δ_gate (the gating value).
Null control (independent chains) must give gap ~ 0.
"""
import sys, os
import numpy as np
# read-only reuse of the reset predecessor's make_dictionary + OneLayerAttn
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bernoulli_fra", "code"))
from hier import joint_T, null_T, stationary, A, B, C


def relu_moments(mu, sigma):
    from scipy.stats import norm
    a = mu / sigma
    m = mu * norm.cdf(a) + sigma * norm.pdf(a)
    s2 = (mu**2 + sigma**2) * norm.cdf(a) + mu * sigma * norm.pdf(a)
    return m, s2


class HierProcess:
    def __init__(self, N_extra=0, d=32, lam_P=0.5, p_P=0.3, lam_C=0.85, p_C=0.5,
                 mu_P=1.5, sig_P=0.3, mu_C=1.5, sig_C=0.3, b_scale=0.5, seed=0,
                 gated=True, orthogonalize=True, obs_noise=0.0, child_occlude=0.0,
                 mu=None, sigma=None):
        """Parent CLEAN (high SNR) by default; the CHILD channel's indicator noise
        is the load-bearing dial (H2.2): set mu_C low so the ReLU false-negative
        atom q_C>0 (a fired child reads 0), or obs_noise>0 — either degrades the
        child INDICATOR and creates child look-back reliance. (mu,sigma legacy: sets
        both features.)"""
        rng = np.random.default_rng(seed)
        if mu is not None:  mu_P = mu_C = mu
        if sigma is not None: sig_P = sig_C = sigma
        self.gated = gated
        self.lam_P, self.p_P, self.lam_C, self.p_C = lam_P, p_P, lam_C, p_C
        self.mu_P, self.sig_P, self.mu_C, self.sig_C = mu_P, sig_P, mu_C, sig_C
        self.obs_noise = obs_noise
        self.child_occlude = child_occlude
        from data import make_dictionary
        from scipy.stats import norm
        self.N = 2 + N_extra
        self.D = make_dictionary(self.N, d, seed=seed, orthogonalize=orthogonalize)
        self.b = b_scale * rng.standard_normal(d)
        self.rng = rng
        self.T = joint_T(lam_P, p_P, lam_C, p_C) if gated else None
        self.m1_C, _ = relu_moments(mu_C, sig_C)          # child mean firing magnitude
        self.m1 = self.m1_C                               # (child channel is the target)
        self.q_C = float(norm.cdf(-mu_C / sig_C))         # child false-negative atom

    def _step_gated(self, zP, zC):
        rng = self.rng; B_ = len(zP)
        persistP = rng.random(B_) < self.lam_P
        newP = np.where(persistP, zP, (rng.random(B_) < self.p_P).astype(float))
        # child: gated
        juston = (newP == 1) & (zP == 0)
        stayon = (newP == 1) & (zP == 1)
        reinit = (rng.random(B_) < self.p_C).astype(float)
        persistC = rng.random(B_) < self.lam_C
        childreset = np.where(persistC, zC, (rng.random(B_) < self.p_C).astype(float))
        newC = np.zeros(B_)
        newC = np.where(juston, reinit, newC)
        newC = np.where(stayon, childreset, newC)
        newC = np.where(newP == 0, 0.0, newC)
        return newP, newC

    def _step_indep(self, zP, zC):
        rng = self.rng; B_ = len(zP)
        pP = rng.random(B_) < self.lam_P
        newP = np.where(pP, zP, (rng.random(B_) < self.p_P).astype(float))
        pC = rng.random(B_) < self.lam_C
        newC = np.where(pC, zC, (rng.random(B_) < self.p_C).astype(float))
        return newP, newC

    def sample_z(self, B_, T):
        zP = (self.rng.random(B_) < self.p_P).astype(float)
        zC = ((self.rng.random(B_) < self.p_C) & (zP == 1)).astype(float) if self.gated \
            else (self.rng.random(B_) < self.p_C).astype(float)
        ZP = np.empty((B_, T)); ZC = np.empty((B_, T))
        step = self._step_gated if self.gated else self._step_indep
        for t in range(T):
            if t > 0:
                zP, zC = step(zP, zC)
            ZP[:, t], ZC[:, t] = zP, zC
        return ZP, ZC

    def sample_seq(self, B_, T):
        ZP, ZC = self.sample_z(B_, T + 1)
        gP = self.mu_P + self.sig_P * self.rng.standard_normal((B_, T + 1))
        gC = self.mu_C + self.sig_C * self.rng.standard_normal((B_, T + 1))
        cP = ZP * np.maximum(gP, 0.0); cC = ZC * np.maximum(gC, 0.0)
        c = np.stack([cP, cC], -1)
        if self.N > 2:
            c = np.concatenate([c, np.zeros((B_, T + 1, self.N - 2))], -1)
        a_true = c @ self.D + self.b                      # target uses TRUE obs
        a_in = a_true.copy()
        if self.child_occlude > 0:                        # occlude the child in the INPUT only
            occ = (self.rng.random((B_, T + 1)) < self.child_occlude)
            a_in = a_in - (occ * cC)[:, :, None] * self.D[1][None, None, :]
        if self.obs_noise > 0:
            a_in = a_in + self.obs_noise * self.rng.standard_normal(a_in.shape)
        return a_in[:, :T, :], a_true[:, 1:, :], c[:, :T, :]  # X occluded, Y true
