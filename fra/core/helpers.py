"""
Shared helpers for FRA computation.

Extracted from validation.py, fra_func.py, and fra_crosscoder.py to
eliminate duplication of weight extraction, top-k sparsification,
and reconstruction metrics.
"""

import math
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
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


# ── Core FRA loop ───────────────────────────────────────────────────────


def compute_fra_sparse(
    topk_features: torch.Tensor,
    W_dec: torch.Tensor,
    W_Q: torch.Tensor,
    W_K: torch.Tensor,
    attn_scale: float,
    *,
    rms: torch.Tensor | None = None,
    dec_norms: torch.Tensor | None = None,
    chunk_size: int = 16,
    verbose: bool = False,
    layer_head_label: str = "",
) -> torch.sparse_coo_tensor:
    """Compute the 4-D FRA sparse tensor from pre-computed feature activations.

    This is the shared inner loop used by both ``get_sentence_fra_batch``
    (single-model SAE path) and ``get_sentence_fra_crosscoder`` (two-model
    crosscoder path).  All model-specific setup (activation extraction,
    encoding, weight lookup) happens in the caller.

    Args:
        topk_features: ``[seq_len, d_sae]`` already top-k sparsified.
        W_dec:  ``[d_sae, d_model]`` decoder weight matrix.
        W_Q:    ``[d_model, d_head]`` query projection for one head.
        W_K:    ``[d_model, d_head]`` key projection for one head.
        attn_scale: ``sqrt(d_head)`` scaling factor.
        rms:    Optional ``[seq_len]`` per-position RMS values for RMSNorm
                correction.  When provided, each interaction is divided by
                ``rms[query_idx] * rms[key_idx]``.
        dec_norms: Optional ``[d_sae]`` decoder-weight norms for SAEs
                trained with ``rescale_acts_by_decoder_norm=True``.
        chunk_size: Number of query positions per GPU batch before flushing
                to CPU.  Controls peak GPU memory.
        verbose: Show progress bar.
        layer_head_label: Label for progress bar (e.g. ``"L5H3"``).

    Returns:
        ``torch.sparse_coo_tensor`` on CPU, shape
        ``[seq_len, seq_len, d_sae, d_sae]``, coalesced.
    """
    seq_len = topk_features.shape[0]
    d_sae = topk_features.shape[1]
    shape = (seq_len, seq_len, d_sae, d_sae)

    all_indices_cpu: list[torch.Tensor] = []
    all_values_cpu: list[torch.Tensor] = []

    total_pairs = seq_len * (seq_len + 1) // 2
    pbar = tqdm(
        total=total_pairs,
        desc=f"Computing FRA ({layer_head_label})" if layer_head_label else "Computing FRA",
        disable=not verbose,
    )

    for q_start in range(0, seq_len, chunk_size):
        q_end = min(q_start + chunk_size, seq_len)
        chunk_indices: list[torch.Tensor] = []
        chunk_values: list[torch.Tensor] = []

        for query_idx in range(q_start, q_end):
            q_feat = topk_features[query_idx]
            q_active = torch.where(q_feat != 0)[0]

            if len(q_active) == 0:
                pbar.update(query_idx + 1)
                continue

            q_vecs = W_dec[q_active]                          # [n_q, d_model]
            q_proj = q_vecs @ W_Q                             # [n_q, d_head]
            q_scales = q_feat[q_active]                       # [n_q]
            if dec_norms is not None:
                q_scales = q_scales / dec_norms[q_active]

            for key_idx in range(query_idx + 1):
                k_feat = topk_features[key_idx]
                k_active = torch.where(k_feat != 0)[0]

                if len(k_active) == 0:
                    pbar.update(1)
                    continue

                k_vecs = W_dec[k_active]                      # [n_k, d_model]
                k_proj = k_vecs @ W_K                         # [n_k, d_head]
                k_scales = k_feat[k_active]                   # [n_k]
                if dec_norms is not None:
                    k_scales = k_scales / dec_norms[k_active]

                int_matrix = (q_proj @ k_proj.T) / attn_scale  # [n_q, n_k]
                int_matrix = int_matrix * q_scales.unsqueeze(1) * k_scales.unsqueeze(0)

                if rms is not None:
                    int_matrix = int_matrix / (rms[query_idx] * rms[key_idx])

                mask = int_matrix.abs() > 1e-10
                if mask.any():
                    local_r, local_c = torch.where(mask)
                    n_int = len(local_r)

                    pos_indices = torch.empty((4, n_int), dtype=torch.long)
                    pos_indices[0] = query_idx
                    pos_indices[1] = key_idx
                    pos_indices[2] = q_active[local_r].cpu()
                    pos_indices[3] = k_active[local_c].cpu()

                    chunk_indices.append(pos_indices)
                    chunk_values.append(int_matrix[mask].detach().cpu().float())

                pbar.update(1)

        all_indices_cpu.extend(chunk_indices)
        all_values_cpu.extend(chunk_values)

        device = topk_features.device
        if device.type != "cpu":
            torch.cuda.empty_cache()

    pbar.close()

    if all_indices_cpu:
        indices = torch.cat(all_indices_cpu, dim=1)
        values = torch.cat(all_values_cpu)
        fra_sparse = torch.sparse_coo_tensor(
            indices, values, size=shape, device="cpu", dtype=torch.float32,
        ).coalesce()
    else:
        fra_sparse = torch.sparse_coo_tensor(
            torch.zeros((4, 0), dtype=torch.long),
            torch.zeros(0, dtype=torch.float32),
            size=shape, device="cpu",
        )

    return fra_sparse


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
