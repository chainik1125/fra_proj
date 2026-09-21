"""Shared machinery for the SAE ground-truth-recovery verification (VERIFY_SAE.md).

Provides: a TopK SAE (adapted from fra_hmm_toy/toy_model.py, NOT import-edited),
a self-contained trainer over precomputed activations, correlated/hierarchical
Bernoulli-Gaussian generators (extending data.py's independent one), and the full
recovery-metric suite:
    MCC (Hungarian), uniqueness, dead latents, precision/recall/F1 as classifiers,
    the mixing matrix chi = W_dec @ pinv(D) (signed regression, not max-cosine),
    and the severability residual  ||(I - P_rowspace(chi)) e_i||  per GT feature.

Everything is CPU/numpy/torch, no LayerNorm anywhere.
"""
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from scipy.stats import norm


# ----------------------------------------------------------------------------
# TopK SAE (vendored + adapted from experiments/fra_hmm_toy/toy_model.py)
# ----------------------------------------------------------------------------
class TopKSAE(nn.Module):
    def __init__(self, d_in, d_sae, k, use_relu=True, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.d_in, self.d_sae, self.k, self.use_relu = d_in, d_sae, k, use_relu
        self.W_enc = nn.Parameter(torch.empty(d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        nn.init.kaiming_uniform_(self.W_enc, a=5 ** 0.5)
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.T)
            self._normalize_decoder()

    def _normalize_decoder(self):
        norms = self.W_dec.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self.W_dec.data.div_(norms)

    def encode(self, x):
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc
        if self.use_relu:
            pre = torch.relu(pre)
        vals, idx = pre.topk(self.k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(-1, idx, vals)
        return z

    def decode(self, z):
        return z @ self.W_dec + self.b_dec

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


def train_sae(acts, d_sae, k, steps=4000, batch=4096, lr=1e-3, seed=0,
              b_dec_init=True, log=False):
    """Train a TopK SAE on a fixed activation matrix `acts` (Nsamp x d_in)."""
    acts_t = torch.as_tensor(acts, dtype=torch.float32)
    d_in = acts_t.shape[1]
    sae = TopKSAE(d_in, d_sae, k, seed=seed)
    if b_dec_init:
        with torch.no_grad():
            sae.b_dec.copy_(acts_t.mean(0))
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    n = acts_t.shape[0]
    for it in range(steps):
        idx = torch.randint(0, n, (batch,))
        x = acts_t[idx]
        xhat, z = sae(x)
        loss = ((xhat - x) ** 2).sum(-1).mean()
        opt.zero_grad(); loss.backward()
        opt.step()
        with torch.no_grad():
            sae._normalize_decoder()
        if log and it % 1000 == 0:
            print(f"    sae step {it:5d}  recon {loss.item():.5f}")
    return sae


# ----------------------------------------------------------------------------
# Generators: correlated (Gaussian copula) + hierarchical Bernoulli-Gaussian.
# Independent case lives in data.py (BernoulliGaussian); these add structure.
# ----------------------------------------------------------------------------
def make_dictionary(N, d, seed=0, orthogonalize=False):
    rng = np.random.default_rng(seed)
    D = rng.standard_normal((N, d))
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    if orthogonalize and N <= d:
        Q, _ = np.linalg.qr(D.T)
        D = Q.T[:N]
        D /= np.linalg.norm(D, axis=1, keepdims=True)
    return D


def rho_mm(D):
    G = D @ D.T
    N = G.shape[0]
    return float(np.mean([np.max(np.abs(G[i][np.arange(N) != i])) for i in range(N)]))


class CorrHierGen:
    """Bernoulli-Gaussian with an optional Gaussian copula (corr) and/or a
    parent->child hierarchy gate. corr_scale=0 & no parents => data.py's setting.

    - copula: g ~ N(0, Sigma), Sigma = F F^T + diag(delta) with unit diagonal;
      z_i = 1[g_i > Phi^{-1}(1-p_i)]  (SS eq., matches marginal p_i).
    - hierarchy: `parents[i]` = index of i's parent (or -1). child gated:
      c_child *= 1[c_parent > 0].
    """
    def __init__(self, N, d, seed=0, p=0.06, mu=1.0, sigma=0.5, b_scale=0.5,
                 orthogonalize=False, corr_rank=0, corr_scale=0.0, parents=None):
        rng = np.random.default_rng(seed)
        self.N, self.d = N, d
        self.D = make_dictionary(N, d, seed=seed, orthogonalize=orthogonalize)
        self.p = np.full(N, p) if np.isscalar(p) else np.asarray(p, float)
        self.mu = np.full(N, mu) if np.isscalar(mu) else np.asarray(mu, float)
        self.sigma = np.full(N, sigma) if np.isscalar(sigma) else np.asarray(sigma, float)
        self.b = b_scale * rng.standard_normal(d)
        self.tau = norm.ppf(1 - self.p)                    # copula thresholds
        self.parents = np.full(N, -1) if parents is None else np.asarray(parents, int)
        # low-rank copula factor
        if corr_rank > 0 and corr_scale > 0:
            F = corr_scale * rng.standard_normal((N, corr_rank))
            ss = (F ** 2).sum(1)
            # keep unit diagonal: clip factor rows whose norm exceeds ~0.97
            scale = np.minimum(1.0, np.sqrt(0.97 / np.maximum(ss, 1e-9)))
            F = F * scale[:, None]
            self.F = F
            self.delta = np.clip(1 - (F ** 2).sum(1), 1e-6, 1.0)
        else:
            self.F, self.delta = None, None
        self.rng = rng

    def sample_c(self, B):
        if self.F is not None:
            r = self.F.shape[1]
            g = self.rng.standard_normal((B, r)) @ self.F.T \
                + np.sqrt(self.delta) * self.rng.standard_normal((B, self.N))
        else:
            g = self.rng.standard_normal((B, self.N))
        z = (g > self.tau).astype(float)
        mag = self.mu + self.sigma * self.rng.standard_normal((B, self.N))
        c = z * np.maximum(mag, 0.0)
        # hierarchy gate (single level; parent must be resolved first — assume
        # parents have lower index, true for our configs)
        for i in range(self.N):
            par = self.parents[i]
            if par >= 0:
                c[:, i] *= (c[:, par] > 0)
        return c

    def sample_a(self, B):
        c = self.sample_c(B)
        return c @ self.D + self.b, c


# ----------------------------------------------------------------------------
# Recovery metrics
# ----------------------------------------------------------------------------
def _unit(M):
    return M / np.clip(np.linalg.norm(M, axis=1, keepdims=True), 1e-12, None)


def mcc_hungarian(W_dec, D):
    """MCC = mean matched |cos| via Hungarian, over min(L,N) matched pairs.
    Returns (mcc, matched_latent_idx, matched_gt_idx)."""
    Wn, Dn = _unit(np.asarray(W_dec)), _unit(np.asarray(D))
    S = np.abs(Wn @ Dn.T)                  # L x N absolute cosine
    L, N = S.shape
    li, gi = linear_sum_assignment(-S)     # maximize
    mcc = S[li, gi].mean()
    return float(mcc), li, gi


def uniqueness(W_dec, D):
    Wn, Dn = _unit(np.asarray(W_dec)), _unit(np.asarray(D))
    S = np.abs(Wn @ Dn.T)
    best = S.argmax(1)                     # dominant GT per latent
    return float(len(set(best.tolist())) / len(best))


def dead_latents(sae, acts, batch=20000):
    x = torch.as_tensor(acts[:batch], dtype=torch.float32)
    with torch.no_grad():
        z = sae.encode(x)
    fired = (z > 0).any(0).cpu().numpy()
    return int((~fired).sum()), fired


def chi_matrix(W_dec, D):
    """chi (L x N): minimum-norm regression  w_l ~ sum_i chi_li d_i.
    chi = W_dec @ pinv(D). Also returns off-dictionary residual fraction per
    latent (||w - chi D|| / ||w||)."""
    W = np.asarray(W_dec)
    D = np.asarray(D)
    chi = W @ np.linalg.pinv(D)            # L x N
    recon = chi @ D
    res = np.linalg.norm(W - recon, axis=1) / np.clip(np.linalg.norm(W, axis=1), 1e-12, None)
    return chi, res


def severability(chi, tol=1e-6):
    """Per GT feature i: residual of severing = ||(I - P) e_i||, P = chi^+ chi
    the projector onto rowspace(chi). residual~0 => 'sever feature i' realizable
    as a latent-set op (e_i in rowspace(chi)). Returns (residuals[N], rank)."""
    chi = np.asarray(chi)
    N = chi.shape[1]
    P = np.linalg.pinv(chi) @ chi          # N x N projector onto rowspace
    resid = np.linalg.norm((np.eye(N) - P), axis=0)  # column-norm = ||(I-P)e_i||
    rank = int(np.linalg.matrix_rank(chi, tol=1e-6))
    return resid, rank


def classifier_prf(sae, acts, z_gt, li, gi):
    """Per matched (latent li[k] -> GT gi[k]): precision/recall/F1 of
    (latent fires>0) predicting (z_gt>0). Returns mean p, r, f1."""
    x = torch.as_tensor(acts, dtype=torch.float32)
    with torch.no_grad():
        zc = sae.encode(x).cpu().numpy()
    ps, rs, fs = [], [], []
    for lat, gt in zip(li, gi):
        pred = zc[:, lat] > 0
        true = z_gt[:, gt] > 0
        tp = np.sum(pred & true)
        fp = np.sum(pred & ~true)
        fn = np.sum(~pred & true)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        ps.append(prec); rs.append(rec); fs.append(f1)
    return float(np.mean(ps)), float(np.mean(rs)), float(np.mean(fs))


def recovery_report(sae, D, acts, z_gt):
    """One-call bundle of all recovery metrics for a trained SAE."""
    W_dec = sae.W_dec.detach().cpu().numpy()
    mcc, li, gi = mcc_hungarian(W_dec, D)
    uniq = uniqueness(W_dec, D)
    ndead, _ = dead_latents(sae, acts)
    chi, offres = chi_matrix(W_dec, D)
    sev, rank = severability(chi)
    prec, rec, f1 = classifier_prf(sae, acts, z_gt, li, gi)
    # chi diagonal strength after matching (signed): pick chi[li, gi]
    diag_signed = chi[li, gi]
    # per-matched-latent off-target chi mass (row-normalized): 1 - |diag|/||row||
    rn = np.linalg.norm(chi[li], axis=1)
    offtarget = 1 - np.abs(diag_signed) / np.clip(rn, 1e-12, None)
    return dict(
        mcc=mcc, uniqueness=uniq, dead=ndead, L=W_dec.shape[0], N=D.shape[0],
        prec=prec, recall=rec, f1=f1,
        chi_rank=rank, sev_max=float(sev.max()), sev_mean=float(sev.mean()),
        n_unsever=int((sev > 1e-3).sum()),
        offdict_res_mean=float(offres.mean()), offdict_res_max=float(offres.max()),
        chi_diag_mean=float(np.abs(diag_signed).mean()),
        offtarget_mean=float(offtarget.mean()),
        _chi=chi, _sev=sev, _li=li, _gi=gi,
    )
