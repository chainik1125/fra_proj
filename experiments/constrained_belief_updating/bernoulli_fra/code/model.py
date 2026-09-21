"""Minimal 1-layer, 1-head attention-only next-vector predictor (no LayerNorm,
so the FRA algebra of A2 is exact: W_QK = Wq^T Wk).

    f_t = beta + W a_t + sum_{s<=t} A_ts V a_s
    A_ts = softmax_s( (Wq a_t + bq) . (Wk a_s + bk) / sqrt(d_h) ),  causal.

Effective direct path W (d x d), effective OV V (d x d), attention Wq,Wk (d x d_h).
FRA in the GT basis (dictionary rows d_i):
    Q_ij = d_i^T Wq Wk^T d_j / sqrt(d_h)      (feature x feature score)
    O_ij = d_i^T V d_j                          (feature j value read by feature i)
"""
import numpy as np
import torch
import torch.nn as nn


class OneLayerAttn(nn.Module):
    def __init__(self, d, d_h=None, bias=True, init=0.02, seed=0,
                 pos_key=False, n_ctx=0, n_heads=1):
        """pos_key: add a learned key-side positional embedding KP[s] to the key,
        giving the pattern a lag/position handle (needed for temporal/reset data;
        Setting-A static data leaves it off). n_ctx sizes KP. n_heads>1 splits d_h
        into heads with separate softmax, summed OV (P2 two-head realization)."""
        super().__init__()
        torch.manual_seed(seed)
        d_h = d_h or d
        assert d_h % n_heads == 0
        self.d, self.d_h, self.use_bias = d, d_h, bias
        self.pos_key, self.n_ctx, self.n_heads = pos_key, n_ctx, n_heads
        self.dph = d_h // n_heads
        self.W = nn.Parameter(init * torch.randn(d, d))
        self.V = nn.Parameter(init * torch.randn(d, d))
        self.Wq = nn.Parameter(init * torch.randn(d, d_h))
        self.Wk = nn.Parameter(init * torch.randn(d, d_h))
        self.bq = nn.Parameter(torch.zeros(d_h))
        self.bk = nn.Parameter(torch.zeros(d_h))
        self.KP = nn.Parameter(init * torch.randn(n_ctx, d_h)) if pos_key else None
        self.beta = nn.Parameter(torch.zeros(d)) if bias else None

    def pattern(self, X):
        # X: B x T x d ; returns A: B x nH x T x s (per-head causal softmax)
        q = X @ self.Wq + self.bq            # B x T x d_h
        k = X @ self.Wk + self.bk
        T = X.shape[1]
        if self.pos_key:
            k = k + self.KP[:T]              # key-side positional embedding
        B = X.shape[0]
        q = q.view(B, T, self.n_heads, self.dph)
        k = k.view(B, T, self.n_heads, self.dph)
        scores = torch.einsum('bthc,bshc->bhts', q, k) / (self.dph ** 0.5)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float('-inf'))
        return torch.softmax(scores, dim=-1)   # B x nH x T x s

    def forward(self, X):
        A = self.pattern(X)                    # B x nH x T x s
        Vx = X @ self.V.T                      # B x s x d  (OV shared across heads)
        ctx = torch.einsum('bhts,bsd->btd', A, Vx) / self.n_heads
        out = X @ self.W.T + ctx
        if self.use_bias:
            out = out + self.beta
        return out

    @torch.no_grad()
    def mean_lag(self, X, t_lo=0):
        """Mean-lag kernel alpha(tau)=mean_{b, t>=max(tau,t_lo)} A_{t,t-tau},
        head-summed. t_lo excludes boundary rows (sprint-2 uses mid rows)."""
        A = self.pattern(X).mean(1)            # B x T x s  (head mean)
        T = X.shape[1]
        out = np.zeros(T)
        for tau in range(T):
            lo = max(tau, t_lo)
            if lo >= T:
                continue
            idx = torch.arange(lo, T)
            out[tau] = A[:, idx, idx - tau].mean().item()
        return out

    # ---- FRA readouts in GT basis ----
    @torch.no_grad()
    def fra_QO(self, D):
        """D: N x d numpy. Returns Q (NxN score), O (NxN OV)."""
        Dt = torch.tensor(D, dtype=self.W.dtype)
        Wqk = self.Wq @ self.Wk.T / (self.d_h ** 0.5)     # d x d
        Q = (Dt @ Wqk @ Dt.T).cpu().numpy()
        O = (Dt @ self.V.T @ Dt.T).cpu().numpy()
        return Q, O


def train(model, gen, T=16, steps=3000, batch=256, lr=3e-3, floor=None, log=False):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for it in range(steps):
        X, Y, _ = gen.sample_seq(batch, T)
        X = torch.tensor(X, dtype=torch.float32)
        Y = torch.tensor(Y, dtype=torch.float32)
        pred = model(X)
        loss = ((pred - Y) ** 2).sum(-1).mean()      # mean over B,T of ||.||^2
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        if log and it % 500 == 0:
            print(f"  step {it:5d}  loss {loss.item():.5f}"
                  + (f"  floor {floor:.5f}" if floor else ""))
    return np.array(losses)


@torch.no_grad()
def eval_loss(model, gen, T=16, B=8000):
    X, Y, _ = gen.sample_seq(B, T)
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    return float(((model(X) - Y) ** 2).sum(-1).mean())
