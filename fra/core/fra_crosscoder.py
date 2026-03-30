"""
FRA computation for model-diffing crosscoders.

This module provides ``get_sentence_fra_crosscoder``, a variant of
``get_sentence_fra_batch`` that works with crosscoders operating on two
models (e.g. Gemma 2B base + instruct).

The key differences from the single-model FRA:
  1. Both models are run to produce stacked activations for the crosscoder.
  2. GQA (grouped-query attention) is handled when indexing W_K.
  3. The attention layer is always ``crosscoder_layer + 1`` — the layer
     whose input is exactly the residual stream the crosscoder encodes.
  4. RMSNorm correction: the crosscoder decoder vectors live in residual-
     stream space, but W_Q / W_K project from post-RMSNorm space.  We
     account for this exactly by folding gamma into W_Q / W_K and dividing
     each position pair's interactions by rms_q * rms_k (both scalars
     computed from the full residual stream at each position).
"""

import math

import torch
from typing import Any, Dict

from transformer_lens import HookedTransformer

from fra.core.helpers import compute_fra_sparse, get_W_K, topk_sparsify

@torch.no_grad()
def get_sentence_fra_crosscoder(
    base_model: HookedTransformer,
    it_model: HookedTransformer,
    crosscoder: Any,
    tokens: list,
    head: int = 0,
    crosscoder_layer: int = 13,
    max_length: int = 128,
    top_k: int | None = 20,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Compute 4-D FRA tensor using a model-diffing crosscoder.

    The crosscoder was trained on ``hook_resid_post`` at ``crosscoder_layer``.
    That residual stream state is the input to the *next* layer's attention
    (``hook_resid_pre`` at ``crosscoder_layer + 1``), so we always analyse
    attention at ``crosscoder_layer + 1``.  The same activations are used
    for crosscoder encoding and for the RMSNorm correction.

    The attention layer is **not** independently configurable — it is always
    ``crosscoder_layer + 1``.  This ensures the FRA decomposition corresponds
    to the features the crosscoder actually learned.

    Args:
        base_model:  HookedTransformer for model-index 0 (base).
        it_model:    HookedTransformer for model-index 1 (instruct).
        crosscoder:  ``GemmaCrosscoderFRA`` instance.
        tokens:      Token IDs to analyse.  Caller is responsible for
                     tokenization (including chat template if needed).
        head:        Attention head index.
        crosscoder_layer:
            Layer from which to extract residual-stream activations for the
            crosscoder (default 13, matching the published checkpoint).
            Uses ``hook_resid_post`` at this layer.  The attention layer
            is derived as ``crosscoder_layer + 1``.
        max_length:  Maximum sequence length.
        top_k:       Keep only top-k features per position.
        verbose:     Show progress bar.

    Returns:
        Dict with keys:
          - ``fra_tensor_sparse``  (torch.sparse_coo_tensor, 4-D)
          - ``shape``
          - ``seq_len``
          - ``total_interactions``
          - ``feature_activations``  (torch.Tensor, [seq, d_sae])
    """
    layer = crosscoder_layer + 1
    target_model = base_model if crosscoder.model_idx == 0 else it_model
    other_model = it_model if crosscoder.model_idx == 0 else base_model
    device = next(target_model.parameters()).device

    # ---- truncate if needed ----
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
    topk_features = topk_sparsify(feature_activations, top_k)

    # ---- RMSNorm correction ----
    # The crosscoder decoder vectors live in residual-stream space, but
    # W_Q / W_K project from post-RMSNorm space.  RMSNorm does:
    #
    #   output = (x / rms(x)) * gamma
    #
    # TransformerLens folds gamma into W_Q / W_K when loading Gemma
    # (fold_ln=True converts "RMS" -> "RMSPre"), so W_Q and W_K already
    # include the gamma factor.  The only remaining correction is the
    # per-position scalar 1/rms(x).
    #
    # The RMS must be computed from the residual stream that actually
    # feeds into the attention layer's RMSNorm — i.e. hook_resid_pre at
    # ``layer``, which equals hook_resid_post at ``crosscoder_layer``.
    # Since layer is always crosscoder_layer + 1, ``target_act`` is
    # exactly that residual stream.

    eps = target_model.cfg.eps
    rms = (target_act.float().pow(2).mean(dim=-1) + eps).sqrt()   # [seq], float32 for numerical stability

    # W_Q / W_K already have gamma folded in by TransformerLens
    W_Q = target_model.blocks[layer].attn.W_Q[head]       # [d_model, d_head]
    W_K = get_W_K(target_model, layer, head)              # [d_model, d_head]
    d_head = W_Q.shape[-1]
    attn_scale = math.sqrt(d_head)

    W_dec = crosscoder.W_dec                              # [d_sae, d_model]

    fra_tensor_sparse = compute_fra_sparse(
        topk_features, W_dec, W_Q, W_K, attn_scale,
        rms=rms,
        chunk_size=16,
        verbose=verbose,
        layer_head_label=f"L{layer}H{head}",
    )

    # Move to device if it fits
    if str(device) != "cpu":
        try:
            fra_tensor_sparse = fra_tensor_sparse.to(device)
        except RuntimeError:
            if verbose:
                print("Warning: sparse tensor too large for GPU, keeping on CPU.")

    total_interactions = fra_tensor_sparse._nnz()
    shape = fra_tensor_sparse.shape

    if verbose:
        _k = top_k if top_k is not None else d_sae
        density = total_interactions / max(seq_len * seq_len * _k * _k, 1)
        print(
            f"4D FRA tensor: shape={shape}, "
            f"nnz={total_interactions:,}, density={density:.2%}"
        )

    return {
        "fra_tensor_sparse": fra_tensor_sparse,
        "shape": shape,
        "seq_len": seq_len,
        "total_interactions": total_interactions,
        "feature_activations": feature_activations,
        "topk_features": topk_features,  # [seq_len, d_sae], top-k filtered activations
    }
