"""Multi-layer attention-only (no MLP) continuous next-vector predictor for Setting H.

Own module (namespaced) — the reset-gate is 2-hop (gate child keys by most-recent
parent-off), so it needs >=2 attention layers. Each block: resid += attn(resid),
content QK + key positional embedding, pre-LN, causal. Readout linear.
"""
import numpy as np, torch, torch.nn as nn


class Block(nn.Module):
    def __init__(self, d, n_ctx, init=0.02, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.Wq = nn.Parameter(init * torch.randn(d, d, generator=g))
        self.Wk = nn.Parameter(init * torch.randn(d, d, generator=g))
        self.Wv = nn.Parameter(init * torch.randn(d, d, generator=g))
        self.Wo = nn.Parameter(init * torch.randn(d, d, generator=g))
        self.KP = nn.Parameter(init * torch.randn(n_ctx, d, generator=g))
        self.bq = nn.Parameter(0.02 * torch.randn(d, generator=g))
        self.ln = nn.LayerNorm(d)

    def forward(self, x, key_project=None, b_vec=None):
        h = self.ln(x)
        hk = h
        if key_project is not None:                        # project a direction out of KEYS
            dP = key_project
            hk = h - (h @ dP)[..., None] * dP
        q = h @ self.Wq + self.bq
        k = hk @ self.Wk + self.KP[:x.shape[1]]
        T = x.shape[1]
        sc = torch.einsum('btd,bsd->bts', q, k) / (q.shape[-1] ** 0.5)
        sc = sc.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float('-inf'))
        A = torch.softmax(sc, -1)
        ctx = torch.einsum('bts,bsd->btd', A, (h @ self.Wv) @ self.Wo.T)
        return x + ctx


class MLAttn(nn.Module):
    def __init__(self, d, n_layers=2, n_ctx=24, init=0.02, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.blocks = nn.ModuleList([Block(d, n_ctx, init, seed + i) for i in range(n_layers)])
        self.Wout = nn.Parameter(init * torch.randn(d, d))
        self.beta = nn.Parameter(torch.zeros(d))
        self.d = d

    def forward(self, X, key_project=None):
        x = X
        for blk in self.blocks:
            x = blk(x, key_project=key_project)
        return x @ self.Wout.T + self.beta


def train_ml(model, gen, T=24, steps=8000, batch=256, lr=3e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for it in range(steps):
        X, Y, _ = gen.sample_seq(batch, T)
        Xt = torch.tensor(X, dtype=torch.float32); Yt = torch.tensor(Y, dtype=torch.float32)
        loss = ((model(Xt) - Yt) ** 2).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def child_handle(model, gen, T=24, B_=12000):
    """child-MSE rise when the PARENT direction is projected out of the KEYS."""
    X, Y, _ = gen.sample_seq(B_, T)
    Xt = torch.tensor(X, dtype=torch.float32)
    dP = torch.tensor(gen.D[0], dtype=torch.float32); dC = gen.D[1]
    base = model(Xt).numpy()
    cut = model(Xt, key_project=dP).numpy()
    ch = float((((cut - Y) @ dC) ** 2).mean() - (((base - Y) @ dC) ** 2).mean())
    pa = float((((cut - Y) @ gen.D[0]) ** 2).mean() - (((base - Y) @ gen.D[0]) ** 2).mean())
    return ch, pa, float((((base - Y) @ dC) ** 2).mean())
