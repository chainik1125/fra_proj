"""Setting-A Bernoulli-Gaussian generator (SynthSAEBench §3), plus analytic/MC moments.

a = D^T c + b,  c_i = z_i * ReLU(mu_i + sigma_i eps_i),  z_i ~ Bern(p_i).
Sigma = I (independent features) unless a copula factor is passed.
Rows of D are unit-norm; superposition controlled by N vs d and an optional
random-then-orthogonalise switch.
"""
import numpy as np


def make_dictionary(N, d, rho_target=None, seed=0, orthogonalize=False):
    """Unit-norm rows D in R^{N x d}. If orthogonalize and N<=d, Gram-Schmidt."""
    rng = np.random.default_rng(seed)
    D = rng.standard_normal((N, d))
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    if orthogonalize and N <= d:
        Q, _ = np.linalg.qr(D.T)          # d x N orthonormal columns
        D = Q.T[:N]
        D /= np.linalg.norm(D, axis=1, keepdims=True)
    return D


class BernoulliGaussian:
    def __init__(self, N, d, seed=0, p=0.1, mu=1.0, sigma=0.3, b_scale=0.5,
                 orthogonalize=False):
        rng = np.random.default_rng(seed)
        self.N, self.d = N, d
        self.D = make_dictionary(N, d, seed=seed, orthogonalize=orthogonalize)
        self.p = np.full(N, p) if np.isscalar(p) else np.asarray(p, float)
        self.mu = np.full(N, mu) if np.isscalar(mu) else np.asarray(mu, float)
        self.sigma = np.full(N, sigma) if np.isscalar(sigma) else np.asarray(sigma, float)
        self.b = b_scale * rng.standard_normal(d)
        self.rng = rng

    # ---- sampling ----
    def sample_c(self, B):
        z = (self.rng.random((B, self.N)) < self.p).astype(float)
        g = self.mu + self.sigma * self.rng.standard_normal((B, self.N))
        c = z * np.maximum(g, 0.0)
        return c

    def sample_a(self, B):
        c = self.sample_c(B)
        a = c @ self.D + self.b
        return a, c

    def sample_seq(self, B, T):
        """iid-across-time sequence: a[b,t,:], and next-vector targets y=a shifted."""
        c = self.sample_c(B * (T + 1)).reshape(B, T + 1, self.N)
        a = c @ self.D + self.b               # B x (T+1) x d
        return a[:, :T, :], a[:, 1:, :], c[:, :T, :]   # X, Y(next), c_X

    # ---- moments (MC, large sample = "exact enough") ----
    def moments(self, B=400_000):
        a, c = self.sample_a(B)
        mu_a = a.mean(0)
        Ca = np.cov(a, rowvar=False)          # Cov(a)
        Sig0 = Ca + np.outer(mu_a, mu_a)      # E[a a^T]
        cbar = c.mean(0)
        Sc = np.cov(c, rowvar=False)          # Cov(c)
        G = self.D @ self.D.T                 # Gram (N x N)
        return dict(mu_a=mu_a, Ca=Ca, Sig0=Sig0, cbar=cbar, Sc=Sc, G=G)


def gram_offdiag_stats(D):
    G = D @ D.T
    N = G.shape[0]
    off = G[~np.eye(N, dtype=bool)]
    rho_mm = np.mean([np.max(np.abs(G[i][np.arange(N) != i])) for i in range(N)])
    return dict(rho_mm=float(rho_mm), max_abs_off=float(np.abs(off).max()),
                mean_abs_off=float(np.abs(off).mean()))
