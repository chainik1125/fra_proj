"""
Shared helpers for FRA computation.

Extracted from validation.py, fra_func.py, and fra_crosscoder.py to
eliminate duplication of weight extraction, top-k sparsification,
and reconstruction metrics.
"""

import math

import numpy as np
import torch
from transformer_lens import HookedTransformer


# ── Weight extraction (GQA-aware) ────────────────────────────────────────


def get_W_K(model: HookedTransformer, layer: int, head: int) -> torch.Tensor:
    """Index W_K correctly for both MHA and GQA models.

    TransformerLens may or may not expand KV heads to match Q heads.
    This helper handles both cases.
    """
    W_K = model.blocks[layer].attn.W_K
    if W_K.shape[0] == model.cfg.n_heads:
        return W_K[head]
    n_kv = W_K.shape[0]
    heads_per_kv = model.cfg.n_heads // n_kv
    return W_K[head // heads_per_kv]


def get_qk_weights(model: HookedTransformer, layer: int, head: int):
    """Get (W_Q, W_K, b_Q, b_K) for a query head, handling GQA correctly."""
    W_Q = model.blocks[layer].attn.W_Q[head]
    b_Q = model.blocks[layer].attn.b_Q[head]
    n_kv = model.blocks[layer].attn.W_K.shape[0]
    n_q = model.blocks[layer].attn.W_Q.shape[0]
    kv_head = head * n_kv // n_q
    W_K = model.blocks[layer].attn.W_K[kv_head]
    b_K = model.blocks[layer].attn.b_K[kv_head]
    return W_Q, W_K, b_Q, b_K


def get_attn_scale(model: HookedTransformer, layer: int) -> float:
    """Return sqrt(d_head) for the given layer."""
    d_head = model.blocks[layer].attn.W_Q.shape[-1]
    return math.sqrt(d_head)


# ── Top-k sparsification ─────────────────────────────────────────────────


def topk_sparsify(
    feature_activations: torch.Tensor,
    top_k: int | None,
) -> torch.Tensor:
    """Keep only top-k features per position by absolute activation magnitude.

    Args:
        feature_activations: [seq_len, d_sae] tensor of feature activations.
        top_k: Number of features to keep per position.  None keeps all
               active (non-zero) features without truncation.

    Returns:
        [seq_len, d_sae] tensor with at most top_k non-zero entries per row.
    """
    if top_k is None:
        return feature_activations

    seq_len = feature_activations.shape[0]
    topk_features = []
    for pos in range(seq_len):
        feat = feature_activations[pos]
        n_active = (feat != 0).sum().item()

        if n_active > 0:
            k = min(top_k, n_active)
            _, topk_idx = torch.topk(feat.abs(), k)
            sparse_feat = torch.zeros_like(feat)
            sparse_feat[topk_idx] = feat[topk_idx]
        else:
            sparse_feat = torch.zeros_like(feat)

        topk_features.append(sparse_feat)

    return torch.stack(topk_features)


# ── FRA aggregation ──────────────────────────────────────────────────────


def fra_sum_to_attn(sparse_tensor: torch.Tensor, seq_len: int) -> np.ndarray:
    """Sum FRA sparse tensor over feature dims -> [seq, seq]."""
    indices = sparse_tensor.indices().cpu().numpy()
    values = sparse_tensor.values().cpu().numpy()
    out = np.zeros((seq_len, seq_len))
    np.add.at(out, (indices[0], indices[1]), values)
    return out


def fra_max_to_attn(sparse_tensor: torch.Tensor, seq_len: int) -> np.ndarray:
    """Max-pool FRA sparse tensor over feature dims -> [seq, seq]."""
    indices = sparse_tensor.indices().cpu().numpy()
    values = sparse_tensor.values().cpu().numpy()
    out = np.full((seq_len, seq_len), -np.inf)
    np.maximum.at(out, (indices[0], indices[1]), values)
    out[out == -np.inf] = 0.0
    return out


def fra_mean_to_attn(sparse_tensor: torch.Tensor, seq_len: int) -> np.ndarray:
    """Mean-pool FRA sparse tensor over feature dims -> [seq, seq]."""
    indices = sparse_tensor.indices().cpu().numpy()
    values = sparse_tensor.values().cpu().numpy()
    s = np.zeros((seq_len, seq_len))
    c = np.zeros((seq_len, seq_len))
    np.add.at(s, (indices[0], indices[1]), values)
    np.add.at(c, (indices[0], indices[1]), 1)
    return np.divide(s, c, where=c > 0, out=np.zeros_like(s))


# ── Reconstruction metrics ───────────────────────────────────────────────


def compute_errors(actual: np.ndarray, reconstructed: np.ndarray,
                   eps: float = 1e-3) -> dict:
    """Compute reconstruction error metrics with small-denominator handling."""
    diff = np.abs(actual - reconstructed)
    denom = np.maximum(np.abs(actual), eps)

    a_flat = actual.flatten().astype(np.float64)
    r_flat = reconstructed.flatten().astype(np.float64)
    norm_a = np.linalg.norm(a_flat)
    norm_r = np.linalg.norm(r_flat)

    return {
        "mean_rel_err": float(np.mean(diff / denom)),
        "median_rel_err": float(np.median(diff / denom)),
        "mean_abs_err": float(np.mean(diff)),
        "cosine_sim": float(
            np.dot(a_flat, r_flat) / (norm_a * norm_r + 1e-10)
        ),
        "r_squared": float(
            1 - np.sum((a_flat - r_flat) ** 2)
            / (np.sum((a_flat - a_flat.mean()) ** 2) + 1e-10)
        ),
        "fro_rel_err": float(np.linalg.norm(diff) / (norm_a + 1e-10)),
    }


def print_errors(label: str, errs: dict) -> None:
    """Pretty-print error metrics."""
    print(f"  {label}:")
    print(f"    Frobenius rel error : {errs['fro_rel_err']:.4f} ({errs['fro_rel_err']*100:.1f}%)")
    print(f"    Cosine similarity   : {errs['cosine_sim']:.6f}")
    print(f"    R-squared           : {errs['r_squared']:.6f}")
    print(f"    Mean elem rel error : {errs['mean_rel_err']:.4f}  (median: {errs['median_rel_err']:.4f})")
    print(f"    Mean abs error      : {errs['mean_abs_err']:.6f}")
