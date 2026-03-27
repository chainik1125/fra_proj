"""
Feature-Resolved Attention (FRA) for Induction Heads -- library functions.

Provides data-independent attention analysis and DI computation helpers.
"""

import math

import numpy as np
import torch
from transformer_lens import HookedTransformer

from fra.core.fra import attention_pattern_QK


def data_independent_attention(model: HookedTransformer, layer: int, head: int, sae_dec: torch.Tensor):
    """
    Compute data-independent feature-resolved attention pattern.

    This shows which features naturally attend to which other features
    based solely on the SAE decoder weights, without any specific input data.

    Args:
        model: The transformer model
        layer: Layer index
        head: Head index
        sae_dec: SAE decoder matrix (W_dec) of shape [d_sae, d_model]

    Returns:
        interaction_matrix: Data-independent attention pattern [d_sae, d_sae]
    """
    # Use SAE decoder weights directly as "activations" for all features
    # This shows the inherent feature-to-feature attention preferences
    query_activations_for_features = sae_dec
    key_activations_for_features = sae_dec

    # Compute attention pattern using the decoder weights
    interaction_matrix_unscaled = attention_pattern_QK(
        model, layer, head,
        query_activations_for_features, False,  # No bias for queries
        key_activations_for_features, False     # No bias for keys
    )

    return interaction_matrix_unscaled


def compute_di_row(
    W_dec: torch.Tensor,
    W_Q: torch.Tensor,
    W_K: torch.Tensor,
    query_feature: int,
) -> np.ndarray:
    """Compute DI(query_feature, j) for all key features j.

    Returns array of shape [d_sae].
    """
    q_vec = W_dec[query_feature] @ W_Q          # [d_head]
    K_all = W_dec @ W_K                         # [d_sae, d_head]
    d_head = W_Q.shape[-1]
    di_row = (K_all @ q_vec) / math.sqrt(d_head)  # [d_sae]
    return di_row.detach().cpu().float().numpy()


def compute_global_di_topk(
    W_dec: torch.Tensor,
    W_Q: torch.Tensor,
    W_K: torch.Tensor,
    top_k: int = 50,
    chunk_size: int = 1024,
    n_sample_rows: int = 500,
    progress_callback=None,
) -> dict:
    """Scan all (i,j) pairs for global top-k by |DI|.

    Also samples random rows for the distribution histogram.

    Returns dict with keys: query_ids, key_ids, di_values, hist_sample.
    """
    d_sae = W_dec.shape[0]
    d_head = W_Q.shape[-1]
    scale = math.sqrt(d_head)
    K_all = W_dec @ W_K                          # [d_sae, d_head]
    n_chunks = math.ceil(d_sae / chunk_size)

    # Sample random rows for histogram
    rng = np.random.default_rng(42)
    sample_idxs = rng.choice(d_sae, size=min(n_sample_rows, d_sae), replace=False)
    Q_sample = W_dec[torch.tensor(sample_idxs, device=W_dec.device)] @ W_Q
    hist_sample = ((Q_sample @ K_all.T) / scale).detach().cpu().float().numpy().ravel()

    # Running top-k across chunks
    top_vals = torch.empty(0, device=W_dec.device)
    top_q = torch.empty(0, dtype=torch.long, device=W_dec.device)
    top_k_buf = torch.empty(0, dtype=torch.long, device=W_dec.device)

    for c in range(n_chunks):
        start = c * chunk_size
        end = min(start + chunk_size, d_sae)
        Q_chunk = W_dec[start:end] @ W_Q         # [chunk, d_head]
        DI_chunk = (Q_chunk @ K_all.T) / scale   # [chunk, d_sae]

        flat_abs = DI_chunk.abs().flatten()
        k_local = min(top_k, flat_abs.numel())
        _, idx_c = torch.topk(flat_abs, k_local)
        q_c = idx_c // d_sae + start
        k_c = idx_c % d_sae
        signed_c = DI_chunk.flatten()[idx_c]

        # Merge with running top-k
        all_signed = torch.cat([top_vals, signed_c])
        all_q = torch.cat([top_q, q_c])
        all_k = torch.cat([top_k_buf, k_c])

        k_merge = min(top_k, all_signed.numel())
        _, keep = torch.topk(all_signed.abs(), k_merge)
        top_vals = all_signed[keep]
        top_q = all_q[keep]
        top_k_buf = all_k[keep]

        if progress_callback:
            progress_callback(c + 1, n_chunks)

    order = torch.argsort(top_vals.abs(), descending=True)
    return {
        "query_ids": top_q[order].cpu().numpy(),
        "key_ids": top_k_buf[order].cpu().numpy(),
        "di_values": top_vals[order].detach().cpu().float().numpy(),
        "hist_sample": hist_sample,
        "hist_sample_idxs": sample_idxs,
        "d_sae": d_sae,
    }
