"""
Feature-Resolved Attention (FRA) for Induction Heads — library functions.

Provides data-independent attention analysis, FRA computation,
and feature interaction extraction used by the induction head experiments.
"""

import torch
from typing import Dict, Any, List, Tuple
from tqdm import tqdm
from transformer_lens import HookedTransformer
from fra.core.fra import attention_pattern_QK
from fra.coders.sae_lens import SAELensAttentionSAE


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


def get_attention_activations(
    model: HookedTransformer,
    input_text: str,
    layer: int,
    max_length: int = 128
) -> torch.Tensor:
    """
    Get attention activations for SAE Lens SAEs (hook_z format).

    Args:
        model: The HookedTransformer model
        input_text: Input text to analyze
        layer: Which layer to get activations from
        max_length: Maximum sequence length

    Returns:
        Tensor of shape (sequence_length, n_heads * d_head) = (seq_len, 768)
    """
    # Tokenize
    tokens = model.tokenizer.encode(input_text)
    if max_length is not None and len(tokens) > max_length:
        tokens = tokens[:max_length]

    device = next(model.parameters()).device
    tokens = torch.tensor(tokens).unsqueeze(0).to(device)

    # Get hook_z activations
    hook_name = f"blocks.{layer}.attn.hook_z"
    _, cache = model.run_with_cache(tokens, names_filter=[hook_name])

    # Shape: [batch=1, seq_len, n_heads=12, d_head=64]
    z = cache[hook_name].squeeze(0)  # Remove batch dimension

    # Flatten heads dimension: [seq_len, 12*64=768]
    z_flat = z.flatten(-2, -1)

    return z_flat


@torch.no_grad()
def compute_fra(
    model: HookedTransformer,
    sae: SAELensAttentionSAE,
    text: str,
    layer: int,
    head: int,
    max_length: int = 128,
    top_k: int = 20,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Compute Feature-Resolved Attention for a text sample.

    Args:
        model: The transformer model
        sae: The SAE wrapper
        text: Input text to analyze
        layer: Which layer to analyze
        head: Which attention head to analyze
        max_length: Maximum sequence length
        top_k: Number of top features to keep per position
        verbose: Whether to show progress

    Returns:
        Dictionary containing:
            - fra_matrix: Sparse FRA matrix [d_sae, d_sae]
            - fra_matrix_abs: Absolute value version
            - seq_len: Sequence length
            - total_pairs: Number of position pairs processed
            - nnz: Number of non-zero entries
            - density: Sparsity of the matrix
            - avg_l0: Average L0 norm (features per token)
            - sparsity: Percentage sparsity
    """
    device = next(model.parameters()).device

    # Get activations
    if verbose:
        print("Getting attention activations...")
    activations = get_attention_activations(model, text, layer=layer, max_length=max_length)
    seq_len = min(activations.shape[0], max_length)

    # Encode to SAE features
    if verbose:
        print("Encoding to SAE features...")
    feature_activations = sae.encode(activations)  # [seq_len, d_sae]

    # Calculate sparsity statistics
    l0_per_token = (feature_activations != 0).sum(-1).float()
    avg_l0 = l0_per_token.mean().item()
    sparsity = 1 - (avg_l0 / sae.d_sae)

    if verbose:
        print(f"  Average L0: {avg_l0:.1f} features per token")
        print(f"  Sparsity: {sparsity*100:.2f}%")

    # Keep only top-k features per position
    if verbose:
        print(f"Selecting top-{top_k} features per position...")

    topk_features = []
    for pos in range(seq_len):
        feat = feature_activations[pos]
        active_mask = feat != 0
        n_active = active_mask.sum().item()

        if n_active > 0:
            k = min(top_k, n_active)
            topk_vals, topk_idx = torch.topk(feat.abs(), k)
            sparse_feat = torch.zeros_like(feat)
            sparse_feat[topk_idx] = feat[topk_idx]
        else:
            sparse_feat = torch.zeros_like(feat)

        topk_features.append(sparse_feat)

    topk_features = torch.stack(topk_features)

    # Get attention weights for the specified head
    W_Q = model.blocks[layer].attn.W_Q[head]
    W_K = model.blocks[layer].attn.W_K[head]

    # Process all position pairs (lower triangle for causal attention)
    total_pairs = seq_len * (seq_len + 1) // 2

    if verbose:
        pbar = tqdm(total=total_pairs, desc=f"Computing FRA (L{layer}H{head})")

    # Accumulate interactions in COO format
    row_indices = []
    col_indices = []
    values = []

    pair_count = 0
    for key_idx in range(seq_len):
        for query_idx in range(key_idx, seq_len):
            q_feat = topk_features[query_idx]
            k_feat = topk_features[key_idx]

            q_active = torch.where(q_feat != 0)[0]
            k_active = torch.where(k_feat != 0)[0]

            if len(q_active) == 0 or len(k_active) == 0:
                pair_count += 1
                if verbose:
                    pbar.update(1)
                continue

            # Get decoder vectors for active features
            q_vecs = sae.W_dec[q_active]
            k_vecs = sae.W_dec[k_active]

            # Compute attention scores
            q_proj = torch.matmul(q_vecs, W_Q)
            k_proj = torch.matmul(k_vecs, W_K)
            int_matrix = torch.matmul(q_proj, k_proj.T)

            # Scale by feature activations
            int_matrix = int_matrix * q_feat[q_active].unsqueeze(1) * k_feat[k_active].unsqueeze(0)

            # Find non-zero interactions
            mask = int_matrix.abs() > 1e-10
            if mask.any():
                local_r, local_c = torch.where(mask)

                # Convert to global feature indices
                global_rows = q_active[local_r].cpu().tolist()
                global_cols = k_active[local_c].cpu().tolist()
                vals = int_matrix[mask].cpu().tolist()

                row_indices.extend(global_rows)
                col_indices.extend(global_cols)
                values.extend(vals)

            pair_count += 1
            if verbose:
                pbar.update(1)

    if verbose:
        pbar.close()

    # Create sparse matrices
    d_sae = sae.d_sae

    if len(row_indices) > 0:
        indices = torch.tensor([row_indices, col_indices], dtype=torch.long, device=device)
        vals = torch.tensor(values, dtype=torch.float32, device=device)

        # Average by number of position pairs
        vals = vals / pair_count if pair_count > 0 else vals

        # Create sparse matrix
        fra_matrix = torch.sparse_coo_tensor(
            indices, vals,
            size=(d_sae, d_sae),
            device=device,
            dtype=torch.float32
        ).coalesce()

        # Absolute value version
        abs_vals = vals.abs()
        fra_matrix_abs = torch.sparse_coo_tensor(
            indices, abs_vals,
            size=(d_sae, d_sae),
            device=device,
            dtype=torch.float32
        ).coalesce()

        nnz = fra_matrix._nnz()
        density = nnz / (d_sae * d_sae)
    else:
        # Empty matrices if no interactions found
        empty_indices = torch.zeros((2, 0), dtype=torch.long, device=device)
        empty_vals = torch.zeros(0, dtype=torch.float32, device=device)

        fra_matrix = torch.sparse_coo_tensor(
            empty_indices, empty_vals,
            size=(d_sae, d_sae),
            device=device
        )
        fra_matrix_abs = fra_matrix.clone()
        nnz = 0
        density = 0

    return {
        'fra_matrix': fra_matrix,
        'fra_matrix_abs': fra_matrix_abs,
        'seq_len': seq_len,
        'total_pairs': pair_count,
        'nnz': nnz,
        'density': density,
        'avg_l0': avg_l0,
        'sparsity': sparsity
    }


def get_top_feature_interactions(fra_matrix: torch.sparse.FloatTensor, top_k: int = 10) -> List[Tuple[int, int, float]]:
    """
    Get top-k feature interactions from FRA matrix.

    Args:
        fra_matrix: Sparse FRA matrix
        top_k: Number of top interactions to return

    Returns:
        List of (query_feature, key_feature, interaction_strength) tuples
    """
    if fra_matrix._nnz() == 0:
        return []

    indices = fra_matrix._indices()
    values = fra_matrix._values()

    k = min(top_k, len(values))
    _, top_idx = torch.topk(values.abs(), k)

    results = []
    for idx in top_idx:
        q_feat = indices[0, idx].item()
        k_feat = indices[1, idx].item()
        value = values[idx].item()
        results.append((q_feat, k_feat, value))

    return results


def analyze_self_interactions(fra_matrix: torch.sparse.FloatTensor, threshold: float = 0.001) -> List[Tuple[int, float]]:
    """
    Find self-interactions (potential induction behavior) in FRA matrix.

    Args:
        fra_matrix: Sparse FRA matrix
        threshold: Minimum interaction strength to consider

    Returns:
        List of (feature_id, self_interaction_strength) tuples
    """
    indices = fra_matrix._indices()
    values = fra_matrix._values()

    # Find diagonal elements (self-interactions)
    self_mask = indices[0] == indices[1]

    if not self_mask.any():
        return []

    self_features = indices[0, self_mask]
    self_values = values[self_mask]

    # Filter by threshold
    strong_mask = self_values.abs() > threshold

    results = []
    for feat, val in zip(self_features[strong_mask].tolist(), self_values[strong_mask].tolist()):
        results.append((feat, val))

    # Sort by interaction strength
    results.sort(key=lambda x: abs(x[1]), reverse=True)

    return results
