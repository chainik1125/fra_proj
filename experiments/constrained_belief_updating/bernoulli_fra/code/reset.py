"""Setting-B reset process: temporal generator + closed-form theory (B_reset.md).

Per feature i: z_i(t) reset chain, eigenvalue lambda_i (redraw prob 1-lambda_i,
redraw ~ Bern(p_i)); c_i(t) = z_i(t) ReLU(mu_i + sigma_i eps_i(t)), eps redrawn
each step; a_t = D^T c(t) + b.

Theory constants (exact rectified-Gaussian moments):
  m = E[ReLU(mu+sig e)],  s2 = E[.^2],  A = m^2 p(1-p),  nu = p (s2-m^2),
  rho = nu/A = observation noise-to-signal.  Optimal attention rate eta(rho,lambda)
  = in-disc root of  rho*lam*eta^2 + [(lam^2-1)-rho*(lam^2+1)]*eta + rho*lam = 0.
"""
import numpy as np
from scipy.stats import norm
from data import make_dictionary


# ---------- exact rectified-Gaussian moments ----------
def relu_moments(mu, sigma):
    """m=E[ReLU(mu+sig e)], s2=E[ReLU^2], for e~N(0,1). Vectorized over arrays."""
    mu = np.asarray(mu, float); sigma = np.asarray(sigma, float)
    a = mu / sigma
    Phi = norm.cdf(a); phi = norm.pdf(a)
    m = mu * Phi + sigma * phi
    s2 = (mu**2 + sigma**2) * Phi + mu * sigma * phi
    q = norm.cdf(-a)                      # ReLU false-negative atom P(fired->0)
    return m, s2, q


def feature_constants(lam, p, mu, sigma):
    """Returns dict of A, nu, rho, m, s2, q, eta (per feature; arrays)."""
    lam = np.atleast_1d(np.asarray(lam, float))
    p = np.broadcast_to(np.atleast_1d(p), lam.shape).astype(float)
    mu = np.broadcast_to(np.atleast_1d(mu), lam.shape).astype(float)
    sigma = np.broadcast_to(np.atleast_1d(sigma), lam.shape).astype(float)
    m, s2, q = relu_moments(mu, sigma)
    A = m**2 * p * (1 - p)
    nu = p * (s2 - m**2)
    rho = nu / A
    eta = np.array([eta_of(l, r) for l, r in zip(lam, rho)])
    return dict(A=A, nu=nu, rho=rho, m=m, s2=s2, q=q, eta=eta, lam=lam, p=p)


def eta_of(lam, rho):
    """In-disc root of the palindromic quadratic (optimal attention rate)."""
    coeffs = [rho * lam, (lam**2 - 1) - rho * (lam**2 + 1), rho * lam]
    r = np.roots(coeffs)
    ind = [x.real for x in r if abs(x) < 1 - 1e-12 and abs(x.imag) < 1e-9]
    return float(sorted(ind, key=abs)[0]) if ind else float('nan')


# ---------- generator ----------
class ResetProcess:
    def __init__(self, N, d, lam, seed=0, p=0.3, mu=1.0, sigma=0.3, b_scale=0.5,
                 orthogonalize=True, D=None):
        rng = np.random.default_rng(seed)
        self.N, self.d = N, d
        self.D = D if D is not None else make_dictionary(N, d, seed=seed,
                                                         orthogonalize=orthogonalize)
        b = lambda x: np.full(N, x) if np.isscalar(x) else np.asarray(x, float)
        self.lam, self.p, self.mu, self.sigma = b(lam), b(p), b(mu), b(sigma)
        self.b = b_scale * rng.standard_normal(d)
        self.rng = rng
        self.const = feature_constants(self.lam, self.p, self.mu, self.sigma)

    def sample_zc(self, B, T):
        """Returns z (B,T,N) indicators and c (B,T,N) magnitudes over T steps."""
        N = self.N
        z = np.empty((B, T, N))
        cur = (self.rng.random((B, N)) < self.p).astype(float)   # stationary start
        for t in range(T):
            if t > 0:
                persist = self.rng.random((B, N)) < self.lam
                redraw = (self.rng.random((B, N)) < self.p).astype(float)
                cur = np.where(persist, cur, redraw)
            z[:, t, :] = cur
        g = self.mu + self.sigma * self.rng.standard_normal((B, T, N))
        c = z * np.maximum(g, 0.0)
        return z, c

    def sample_seq(self, B, T):
        """X=a[:T], Y=a[1:T+1] (next-vector target), c_X. Shapes (B,T,d)/(B,T,N)."""
        z, c = self.sample_zc(B, T + 1)
        a = c @ self.D + self.b
        return a[:, :T, :], a[:, 1:, :], c[:, :T, :]

    def autocov_check(self, B=200000, T=40):
        z, _ = self.sample_zc(B, T)
        out = []
        for tau in [1, 2, 5, 10]:
            cv = np.mean((z[:, tau:, 0] - self.p[0]) * (z[:, :-tau, 0] - self.p[0]))
            out.append((tau, float(cv), float(self.p[0]*(1-self.p[0])*self.lam[0]**tau)))
        return out


# ---------- exact Bayes filter + prediction floors (per feature, orthogonal) ----------
def bayes_filter_mse(gen, B=20000, T=24):
    """Exact 2-state Bayes filter per feature; returns per-a one-step MSE (Bayes
    floor) summed over features, plus the naive prior-only floor."""
    z, c = gen.sample_zc(B, T + 1)
    lam, p, m, q = gen.lam, gen.p, gen.const['m'], gen.const['q']
    mse_bayes = 0.0; mse_prior = 0.0
    for i in range(gen.N):
        y = c[:, :, i]                       # observation = readout (rho_mm=0)
        b = np.full(B, p[i])                 # filter posterior b_i(t)=P(z=1|y_1:t)
        Pi = m[i]                            # E[magnitude | fired]
        se_b = np.zeros(B); se_p = np.zeros(B)
        for t in range(T):
            yt = y[:, t]
            pred_prior = p[i] + lam[i] * (b - p[i])          # one-step ahead
            # update with obs at t
            ppred = p[i] + lam[i] * (b - p[i])               # prior for z(t)
            fired = yt > 1e-12
            b_new = np.where(fired, 1.0,
                             ppred * q[i] / (1 - ppred * (1 - q[i]) + 1e-30))
            b = b_new
            # predict c_i(t+1) = m_i * P(z(t+1)=1|y_1:t)
            Phat = p[i] + lam[i] * (b - p[i])
            ctrue = c[:, t + 1, i]
            se_b += (ctrue - Pi * Phat) ** 2
            se_p += (ctrue - Pi * p[i]) ** 2
        mse_bayes += se_b.mean() / T
        mse_prior += se_p.mean() / T
    return dict(mse_bayes=float(mse_bayes), mse_prior=float(mse_prior))


def bayes_beliefs(gen, c):
    """One-step belief Phat[b,t,i]=P(z_i(t+1)=1 | y_i(1:t)) for observations c (B,T,N)."""
    B, T, Nn = c.shape
    lam, p, q = gen.lam, gen.p, gen.const['q']
    Phat = np.empty((B, T, Nn))
    for i in range(Nn):
        b = np.full(B, p[i])
        for t in range(T):
            ppred = p[i] + lam[i] * (b - p[i])
            fired = c[:, t, i] > 1e-12
            b = np.where(fired, 1.0, ppred * q[i] / (1 - ppred * (1 - q[i]) + 1e-30))
            Phat[:, t, i] = p[i] + lam[i] * (b - p[i])
    return Phat


def additive_floor_mse(gen, T=24, taumax=60):
    """Best geometric-additive (attention) one-step MSE per feature, at the optimal
    eta (=Kalman/Wiener steady state for AR(1)+noise). Summed over features."""
    tot = 0.0
    for i in range(gen.N):
        A, nu, lam = gen.const['A'][i], gen.const['nu'][i], gen.lam[i]
        eta = gen.const['eta'][i]
        G = A * lam ** np.arange(taumax + 1); G[0] = A + nu
        a = eta ** np.arange(1, taumax + 1)
        from scipy.linalg import toeplitz
        Gmat = toeplitz(G)
        VX0 = G[0]; CX0X1 = a @ G[1:taumax + 1]
        VX1 = a @ Gmat[1:, 1:] @ a
        CyX0 = G[1]; CyX1 = a[:taumax - 1] @ G[2:taumax + 1]
        Sig = np.array([[VX0, CX0X1], [CX0X1, VX1]]); cv = np.array([CyX0, CyX1])
        tot += G[0] - cv @ np.linalg.solve(Sig, cv)
    return float(tot)


def geometric_rate_fit(alpha, tau_lo=1, tau_hi=None, amin=1e-4):
    """Fit alpha(tau) ~ c*eta^tau over tau in [tau_lo, tau_hi]; log-linear."""
    T = len(alpha)
    tau_hi = tau_hi or (T - 2)
    taus, ys = [], []
    for tau in range(tau_lo, tau_hi + 1):
        if alpha[tau] > amin:
            taus.append(tau); ys.append(np.log(alpha[tau]))
    if len(taus) < 2:
        return float('nan')
    slope = np.polyfit(taus, ys, 1)[0]
    return float(np.exp(slope))


if __name__ == "__main__":
    g = ResetProcess(3, 32, lam=0.7, p=0.3, mu=1.0, sigma=0.3, seed=0)
    print("autocov (tau, sim, theory):", g.autocov_check())
    print("const:", {k: np.round(v, 4) for k, v in g.const.items()})
    print("eta grid over sigma:")
    for s in [0.05, 0.15, 0.3, 0.6, 1.2, 2.4]:
        c = feature_constants(0.7, 0.3, 1.0, s)
        print(f"  sigma={s}: rho={c['rho'][0]:.4f} eta={c['eta'][0]:.4f}")
