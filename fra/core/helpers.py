"""
Shared helpers for FRA analysis.

Weight extraction, reconstruction metrics, pair ranking, and FRA
aggregation utilities used across dashboard, analysis, and viz modules.

FRA-specific computation (``topk_sparsify``, ``compute_fra_sparse``,
``apply_rope_to_projected``) lives in ``fra.core.fra``.
"""

import math
from collections import defaultdict

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
                   eps: float = 1e-3, exclude_bos: bool = False) -> dict:
    """Compute reconstruction error metrics with small-denominator handling.

    Args:
        exclude_bos: When True, exclude position 0 (BOS token) from metrics.
            For 2-D attention matrices this removes both row 0 and column 0.
            For 1-D vectors this removes element 0.
    """
    if exclude_bos:
        if actual.ndim >= 2 and actual.shape[0] == actual.shape[1]:
            # Square matrix (attention scores): remove BOS row and column
            actual = actual[1:, 1:]
            reconstructed = reconstructed[1:, 1:]
        else:
            # Activation matrix [seq, d_model] or 1-D: remove position 0 only
            actual = actual[1:]
            reconstructed = reconstructed[1:]

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


# ── Pair ranking ─────────────────────────────────────────────────────────


def aggregate_pairs(indices_np, values_np, diagonal=None):
    """Aggregate sparse 4D FRA entries by (q_feat, k_feat) pair.

    Args:
        indices_np: [4, nnz] array (q_pos, k_pos, q_feat, k_feat)
        values_np: [nnz] array
        diagonal: None=all pairs, True=only i==j, False=only i!=j

    Returns:
        List of (q_feat, k_feat, sum_abs, count, max_abs) tuples.
    """
    q_feats = indices_np[2, :]
    k_feats = indices_np[3, :]
    abs_vals = np.abs(values_np)

    if diagonal is True:
        mask = q_feats == k_feats
        q_feats, k_feats, abs_vals = q_feats[mask], k_feats[mask], abs_vals[mask]
    elif diagonal is False:
        mask = q_feats != k_feats
        q_feats, k_feats, abs_vals = q_feats[mask], k_feats[mask], abs_vals[mask]

    pair_sum: dict = defaultdict(float)
    pair_count: dict = defaultdict(int)
    pair_max: dict = defaultdict(float)
    for q, k, v in zip(q_feats, k_feats, abs_vals):
        pair_sum[(int(q), int(k))] += float(v)
        pair_count[(int(q), int(k))] += 1
        if float(v) > pair_max[(int(q), int(k))]:
            pair_max[(int(q), int(k))] = float(v)

    return [
        (q, k, pair_sum[(q, k)], pair_count[(q, k)], pair_max[(q, k)])
        for (q, k) in pair_sum
    ]


def rank_pairs(indices_np, values_np, top_k=50, diagonal=None, mode="sum"):
    """Return top-k (q_feat, k_feat) pairs ranked by aggregation mode.

    Modes:
        sum  -- total absolute interaction strength
        avg  -- mean absolute interaction per position-pair occurrence
        max  -- single strongest position-pair interaction

    Args:
        indices_np: [4, nnz] array (q_pos, k_pos, q_feat, k_feat)
        values_np: [nnz] array
        top_k: Number of pairs to return.
        diagonal: None=all pairs, True=only i==j, False=only i!=j
        mode: Ranking criterion ("sum", "avg", or "max").

    Returns:
        List of (q_feat, k_feat, sum_abs, count, max_abs) tuples,
        sorted descending by *mode*.
    """
    pairs = aggregate_pairs(indices_np, values_np, diagonal=diagonal)
    if mode == "sum":
        pairs.sort(key=lambda x: x[2], reverse=True)
    elif mode == "avg":
        pairs.sort(key=lambda x: x[2] / max(x[3], 1), reverse=True)
    elif mode == "max":
        pairs.sort(key=lambda x: x[4], reverse=True)
    return pairs[:top_k]


def get_position_heatmap(indices_np, values_np, q_feat, k_feat, seq_len):
    """Extract [seq_len, seq_len] heatmap for a specific (q_feat, k_feat) pair."""
    mask = (indices_np[2] == q_feat) & (indices_np[3] == k_feat)
    q_pos = indices_np[0, mask]
    k_pos = indices_np[1, mask]
    vals = np.abs(values_np[mask])

    mat = np.zeros((seq_len, seq_len))
    for qp, kp, v in zip(q_pos, k_pos, vals):
        mat[qp, kp] += v
    return mat
