"""
FRA computation for model-diffing crosscoders.

This module provides ``get_sentence_fra_crosscoder``, a variant of
``get_sentence_fra_batch`` that works with crosscoders operating on two
models (e.g. Gemma 2B base + instruct).

The key differences from the single-model FRA:
  1. Both models are run to produce stacked activations for the crosscoder.
  2. GQA (grouped-query attention) is handled when indexing W_K.
  3. The crosscoder and attention layers can differ (the crosscoder was
     trained at one layer; we can analyze attention at another).
  4. RMSNorm correction: the crosscoder decoder vectors live in residual-
     stream space, but W_Q / W_K project from post-RMSNorm space.  We
     account for this exactly by folding gamma into W_Q / W_K and dividing
     each position pair's interactions by rms_q * rms_k (both scalars
     computed from the full residual stream at each position).
"""

import torch
from typing import Any, Dict

from tqdm import tqdm
from transformer_lens import HookedTransformer


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _get_W_K(model: HookedTransformer, layer: int, head: int) -> torch.Tensor:
    """Index W_K correctly for both MHA and GQA models.

    TransformerLens may or may not expand KV heads to match Q heads.
    This helper handles both cases.
    """
    W_K = model.blocks[layer].attn.W_K
    if W_K.shape[0] == model.cfg.n_heads:
        # Standard MHA or already-expanded GQA
        return W_K[head]
    # GQA with un-expanded KV heads
    n_kv = W_K.shape[0]
    heads_per_kv = model.cfg.n_heads // n_kv
    return W_K[head // heads_per_kv]


# ------------------------------------------------------------------
# Core FRA function
# ------------------------------------------------------------------

@torch.no_grad()
def get_sentence_fra_crosscoder(
    base_model: HookedTransformer,
    it_model: HookedTransformer,
    crosscoder: Any,
    text: str,
    layer: int = 14,
    head: int = 0,
    crosscoder_layer: int = 13,
    max_length: int = 128,
    top_k: int = 20,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Compute 4-D FRA tensor using a model-diffing crosscoder.

    The crosscoder was trained on ``hook_resid_post`` at ``crosscoder_layer``.
    That residual stream state is the input to the *next* layer's attention
    (``hook_resid_pre`` at ``crosscoder_layer + 1``), so by default we
    analyse attention at ``crosscoder_layer + 1``.  The same activations are
    used for crosscoder encoding and for the RMSNorm correction.

    Args:
        base_model:  HookedTransformer for model-index 0 (base).
        it_model:    HookedTransformer for model-index 1 (instruct).
        crosscoder:  ``GemmaCrosscoderFRA`` instance.
        text:        Input text to analyse.
        layer:       Attention layer whose W_Q / W_K to use.  Defaults to
                     ``crosscoder_layer + 1`` (the layer that reads from the
                     crosscoder's residual stream).
        head:        Attention head index.
        crosscoder_layer:
            Layer from which to extract residual-stream activations for the
            crosscoder (default 13, matching the published checkpoint).
            Uses ``hook_resid_post`` at this layer.
        max_length:  Maximum sequence length.
        top_k:       Keep only top-k features per position.
        verbose:     Show progress bar.

    Returns:
        Dict with keys:
          - ``fra_tensor_sparse``  (torch.sparse_coo_tensor, 4-D)
          - ``shape``
          - ``seq_len``
          - ``total_interactions``
    """
    target_model = base_model if crosscoder.model_idx == 0 else it_model
    other_model = it_model if crosscoder.model_idx == 0 else base_model
    device = next(target_model.parameters()).device

    # ---- tokenise ----
    tokens = target_model.tokenizer.encode(text)
    if max_length is not None and len(tokens) > max_length:
        tokens = tokens[:max_length]
    tokens_tensor = torch.tensor(tokens).unsqueeze(0).to(device)

    # ---- extract activations ----
    # The crosscoder was trained on hook_resid_post at crosscoder_layer.
    # That same residual stream state is hook_resid_pre at layer (= crosscoder_layer + 1),
    # i.e. the input to the attention whose W_Q / W_K we decompose through.
    # We use this single state for both crosscoder encoding and RMS correction.
    hook_name = f"blocks.{crosscoder_layer}.hook_resid_post"

    _, target_cache = target_model.run_with_cache(
        tokens_tensor, names_filter=[hook_name],
    )
    _, other_cache = other_model.run_with_cache(
        tokens_tensor, names_filter=[hook_name],
    )

    target_act = target_cache[hook_name].squeeze(0)      # [seq, d_model]
    other_act = other_cache[hook_name].squeeze(0)
    seq_len = target_act.shape[0]

    # Stack in crosscoder order: [base, instruct]
    if crosscoder.model_idx == 0:
        x_stacked = torch.stack([target_act, other_act], dim=1)
    else:
        x_stacked = torch.stack([other_act, target_act], dim=1)

    # ---- encode through crosscoder ----
    if verbose:
        print(f"Encoding {seq_len} positions through crosscoder...")
    feature_activations = crosscoder.encode(x_stacked)   # [seq, d_sae]
    d_sae = feature_activations.shape[-1]

    # ---- top-k sparsification ----
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
    topk_features = torch.stack(topk_features)

    # ---- RMSNorm correction ----
    # The crosscoder decoder vectors live in residual-stream space, but
    # W_Q / W_K project from post-RMSNorm space.  RMSNorm does:
    #
    #   output = (x / rms(x)) * gamma
    #
    # where rms(x) = sqrt(mean(x^2) + eps) is a scalar per position, and
    # gamma is a fixed learnable per-element scale.
    #
    # Since rms is scalar, the attention score decomposes exactly:
    #
    #   attn(q, k) = 1/(rms_q * rms_k)
    #                * sum_ij a_qi * a_kj * (W_dec[i] @ W_Q_eff) . (W_dec[j] @ W_K_eff)
    #
    # where W_Q_eff = diag(gamma) @ W_Q,  W_K_eff = diag(gamma) @ W_K.

    gamma = target_model.blocks[layer].ln1.weight           # [d_model]
    eps = target_model.cfg.eps

    # Per-position RMS from the actual full residual stream
    # (target_act is both the crosscoder input and the RMSNorm input)
    rms = (target_act.pow(2).mean(dim=-1) + eps).sqrt()   # [seq]

    # Fold gamma into W_Q / W_K (one-time)
    W_Q = target_model.blocks[layer].attn.W_Q[head]       # [d_model, d_head]
    W_K = _get_W_K(target_model, layer, head)              # [d_model, d_head]
    W_Q_eff = gamma.unsqueeze(1) * W_Q                    # [d_model, d_head]
    W_K_eff = gamma.unsqueeze(1) * W_K                    # [d_model, d_head]

    W_dec = crosscoder.W_dec                              # [d_sae, d_model]

    # ---- FRA loop (lower-triangular, causal) ----
    all_indices = []
    all_values = []
    total_pairs = seq_len * (seq_len + 1) // 2

    pbar = tqdm(total=total_pairs, desc=f"Computing FRA (L{layer}H{head})",
                disable=not verbose)

    for key_idx in range(seq_len):
        for query_idx in range(key_idx, seq_len):
            q_feat = topk_features[query_idx]
            k_feat = topk_features[key_idx]

            q_active = torch.where(q_feat != 0)[0]
            k_active = torch.where(k_feat != 0)[0]

            if len(q_active) == 0 or len(k_active) == 0:
                pbar.update(1)
                continue

            q_vecs = W_dec[q_active]                       # [n_q, d_model]
            k_vecs = W_dec[k_active]                       # [n_k, d_model]

            q_proj = torch.matmul(q_vecs, W_Q_eff)        # [n_q, d_head]
            k_proj = torch.matmul(k_vecs, W_K_eff)        # [n_k, d_head]
            int_matrix = torch.matmul(q_proj, k_proj.T)   # [n_q, n_k]

            # Scale by activation magnitudes and RMSNorm correction.
            # The 1/(rms_q * rms_k) factor accounts for the input-dependent
            # part of RMSNorm; gamma is already folded into W_Q_eff / W_K_eff.
            int_matrix = (
                int_matrix
                * q_feat[q_active].unsqueeze(1)
                * k_feat[k_active].unsqueeze(0)
                / (rms[query_idx] * rms[key_idx])
            )

            mask = int_matrix.abs() > 1e-10
            if mask.any():
                local_r, local_c = torch.where(mask)
                n = len(local_r)
                pos_indices = torch.zeros((4, n), dtype=torch.long)
                pos_indices[0, :] = query_idx
                pos_indices[1, :] = key_idx
                pos_indices[2, :] = q_active[local_r]
                pos_indices[3, :] = k_active[local_c]

                all_indices.append(pos_indices)
                all_values.append(int_matrix[mask])

            pbar.update(1)

    pbar.close()

    # ---- assemble sparse tensor ----
    shape = (seq_len, seq_len, d_sae, d_sae)

    if all_indices:
        indices = torch.cat(all_indices, dim=1).to(device)
        values = torch.cat(all_values).to(device)
        fra_tensor_sparse = torch.sparse_coo_tensor(
            indices, values, size=shape, device=device, dtype=torch.float32,
        ).coalesce()
        total_interactions = fra_tensor_sparse._nnz()
    else:
        empty_idx = torch.zeros((4, 0), dtype=torch.long, device=device)
        empty_val = torch.zeros(0, dtype=torch.float32, device=device)
        fra_tensor_sparse = torch.sparse_coo_tensor(
            empty_idx, empty_val, size=shape, device=device,
        )
        total_interactions = 0

    if verbose:
        density = total_interactions / max(seq_len * seq_len * top_k * top_k, 1)
        print(
            f"4D FRA tensor: shape={shape}, "
            f"nnz={total_interactions:,}, density={density:.2%}"
        )

    return {
        "fra_tensor_sparse": fra_tensor_sparse,
        "shape": shape,
        "seq_len": seq_len,
        "total_interactions": total_interactions,
    }
