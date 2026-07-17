"""
Linear probes + causal ablation utilities (dependency-free: torch/numpy only).

  * extract_resid: residual-stream activations at chosen layer/positions.
  * ridge_probe: closed-form ridge regression with a held-out split -> test R².
  * logistic_probe: multinomial logistic regression (Adam) -> test accuracy.
  * direction_for_target: the probe weight vector = a causal direction to ablate.
"""

from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def extract_resid(model, tokens: torch.Tensor, layer: int, pos):
    """Residual stream after `layer` at input position(s) `pos`.
    pos: int or 1D index tensor/array. Returns (N, d_model) numpy."""
    _, resids = model.run_with_resid(tokens)
    r = resids[layer]  # (B, T, d)
    if isinstance(pos, int):
        out = r[:, pos, :]
    else:
        pos = torch.as_tensor(pos, device=r.device)
        out = r[:, pos, :]
        if out.dim() == 3:
            out = out.reshape(-1, out.shape[-1])
    return out.detach().cpu().numpy()


def _split(n, frac, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    k = int(n * frac)
    return idx[:k], idx[k:]


def ridge_probe(X, y, alpha=1.0, train_frac=0.7, seed=0):
    """Closed-form ridge. Returns dict with test R², weights, bias."""
    X = np.asarray(X, np.float64); y = np.asarray(y, np.float64).ravel()
    tr, te = _split(len(X), train_frac, seed)
    mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-8
    Xtr = (X[tr] - mu) / sd; Xte = (X[te] - mu) / sd
    ym = y[tr].mean()
    d = Xtr.shape[1]
    A = Xtr.T @ Xtr + alpha * np.eye(d)
    w = np.linalg.solve(A, Xtr.T @ (y[tr] - ym))
    pred = Xte @ w + ym
    ss_res = ((y[te] - pred) ** 2).sum()
    ss_tot = ((y[te] - y[te].mean()) ** 2).sum() + 1e-12
    r2 = 1 - ss_res / ss_tot
    # direction in raw activation space (undo standardization scaling)
    direction = w / sd
    return {"r2": float(r2), "w": w, "bias": float(ym), "mu": mu, "sd": sd,
            "direction": direction, "pred_test": pred, "y_test": y[te], "test_idx": te}


def logistic_probe(X, y_int, n_classes=None, train_frac=0.7, seed=0, steps=400, lr=0.05,
                   l2=1e-3, device="cpu"):
    """Multinomial logistic regression probe. Returns test accuracy + baseline."""
    X = np.asarray(X, np.float64); y_int = np.asarray(y_int).astype(np.int64).ravel()
    classes, y = np.unique(y_int, return_inverse=True)
    K = len(classes) if n_classes is None else n_classes
    tr, te = _split(len(X), train_frac, seed)
    mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-8
    Xtr = torch.tensor((X[tr] - mu) / sd, dtype=torch.float32, device=device)
    Xte = torch.tensor((X[te] - mu) / sd, dtype=torch.float32, device=device)
    ytr = torch.tensor(y[tr], device=device); yte = torch.tensor(y[te], device=device)
    W = torch.zeros(Xtr.shape[1], K, requires_grad=True, device=device)
    b = torch.zeros(K, requires_grad=True, device=device)
    opt = torch.optim.Adam([W, b], lr=lr)
    lossf = torch.nn.CrossEntropyLoss()
    for _ in range(steps):
        opt.zero_grad()
        logit = Xtr @ W + b
        loss = lossf(logit, ytr) + l2 * (W ** 2).sum()
        loss.backward(); opt.step()
    with torch.no_grad():
        acc = (torch.argmax(Xte @ W + b, 1) == yte).float().mean().item()
    # majority-class baseline on the test split
    vals, counts = np.unique(y[tr], return_counts=True)
    maj = vals[np.argmax(counts)]
    baseline = (y[te] == maj).mean()
    return {"acc": float(acc), "baseline": float(baseline), "n_classes": int(K)}
