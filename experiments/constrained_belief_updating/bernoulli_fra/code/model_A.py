"""Setting-A dedicated 1-layer 1-head attention-only next-vector predictor.
(Private copy so Setting-A checks don't race the shared model.py, which the
Setting-B agent extends for temporal data.) No LayerNorm => FRA algebra exact:
W_QK = Wq Wk^T / sqrt(d_h).

    f_t = beta + W a_t + sum_{s<=t} A_ts V a_s ,  A causal softmax.
"""
import numpy as np
import torch
import torch.nn as nn


class OneLayerAttnA(nn.Module):
    def __init__(self, d, d_h=None, bias=True, init=0.02, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        d_h = d_h or d
        self.d, self.d_h, self.use_bias = d, d_h, bias
        self.W = nn.Parameter(init * torch.randn(d, d))
        self.V = nn.Parameter(init * torch.randn(d, d))
        self.Wq = nn.Parameter(init * torch.randn(d, d_h))
        self.Wk = nn.Parameter(init * torch.randn(d, d_h))
        self.bq = nn.Parameter(torch.zeros(d_h))
        self.bk = nn.Parameter(torch.zeros(d_h))
        self.beta = nn.Parameter(torch.zeros(d)) if bias else None

    def pattern(self, X):
        q = X @ self.Wq + self.bq
        k = X @ self.Wk + self.bk
        scores = torch.einsum('bth,bsh->bts', q, k) / (self.d_h ** 0.5)
        T = X.shape[1]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float('-inf'))
        return torch.softmax(scores, dim=-1)

    def forward(self, X):
        A = self.pattern(X)
        ctx = torch.einsum('bts,bsd->btd', A, X @ self.V.T)
        out = X @ self.W.T + ctx
        if self.use_bias:
            out = out + self.beta
        return out

    @torch.no_grad()
    def fra_QO(self, D):
        Dt = torch.tensor(D, dtype=self.W.dtype)
        Wqk = self.Wq @ self.Wk.T / (self.d_h ** 0.5)
        Q = (Dt @ Wqk @ Dt.T).cpu().numpy()
        O = (Dt @ self.V.T @ Dt.T).cpu().numpy()
        return Q, O


def train(model, gen, T=16, steps=3000, batch=256, lr=3e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for it in range(steps):
        X, Y, _ = gen.sample_seq(batch, T)
        X = torch.tensor(X, dtype=torch.float32)
        Y = torch.tensor(Y, dtype=torch.float32)
        loss = ((model(X) - Y) ** 2).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return float(loss.item())


@torch.no_grad()
def eval_loss(model, gen, T=16, B=8000):
    X, Y, _ = gen.sample_seq(B, T)
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    return float(((model(X) - Y) ** 2).sum(-1).mean())
