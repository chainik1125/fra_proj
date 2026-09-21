"""Adapter exposing our dense FRA to the repo's conformance harness.

Why bother, when we have our own exactness test: the harness's synthetic case is
built with a **non-zero** ``b_dec``, ``b_Q`` and ``b_K``, and its
``reconstruct_masked_scores_from_fra`` adds back the Eqs 13-15 feat x bias,
bias x feat and bias x bias terms. Our own toy zeroes all of those by
construction, so our test cannot exercise them. Running against the harness
checks our tensor is right in the presence of biases too -- which is precisely
what the "add corrections back one at a time" commits will need.

Convention: the harness expects the FRA tensor **unscaled** (its
``compute_raw_qk`` compares against a bare ``q @ k.T``) and divides by
``attn_scale`` only after adding the bias terms. So this adapter passes
``attn_scale=1.0``. See :mod:`fra.toy.fra` for the discussion.
"""

from __future__ import annotations

from typing import Any

import torch

from fra.toy.fra import fra_qk


def _qk_weights(model: Any, layer: int, head: int) -> tuple[torch.Tensor, torch.Tensor]:
    """W_Q and the GQA-mapped W_K, matching the harness's own helper."""
    W_Q = model.blocks[layer].attn.W_Q[head]
    n_q = model.blocks[layer].attn.W_Q.shape[0]
    n_kv = model.blocks[layer].attn.W_K.shape[0]
    W_K = model.blocks[layer].attn.W_K[head * n_kv // n_q]
    return W_Q, W_K


def _topk_sparsify(features: torch.Tensor, top_k: int | None) -> torch.Tensor:
    """Keep the top-k features per position by absolute activation."""
    if top_k is None:
        return features
    out = torch.zeros_like(features)
    for pos in range(features.shape[0]):
        row = features[pos]
        n_active = int((row != 0).sum())
        if n_active == 0:
            continue
        _, idx = torch.topk(row.abs(), min(top_k, n_active))
        out[pos, idx] = row[idx]
    return out


def compute_toy_fra(
    model: Any,
    sae: Any,
    text: str,
    layer: int = 0,
    head: int = 0,
    hook_point: str = "ln1.hook_normalized",
    top_k_features: int | None = None,
    chunk_size: int = 16,  # unused: dense at this scale
    max_length: int | None = None,
    normalize_by_decoder_norm: bool | None = False,
    prepend_bos: bool = False,
    run_validation: bool = False,
) -> dict[str, Any]:
    """Dense FRA, returned in the harness's sparse-COO contract."""
    del chunk_size, run_validation

    tokens = model.tokenizer.encode(text, add_special_tokens=prepend_bos)
    if max_length is not None:
        tokens = tokens[:max_length]
    tokens_t = torch.tensor(tokens).unsqueeze(0)

    hook_name = f"blocks.{layer}.{hook_point}"
    _, cache = model.run_with_cache(tokens_t, names_filter=[hook_name])
    acts = cache[hook_name].squeeze(0)  # [seq, d_model]

    features = sae.encode(acts)  # [seq, d_sae]
    features = _topk_sparsify(features, top_k_features)

    W_dec = sae.W_dec
    if normalize_by_decoder_norm:
        W_dec = W_dec / W_dec.norm(dim=-1, keepdim=True)

    W_Q, W_K = _qk_weights(model, layer, head)
    # attn_scale = 1.0: the harness applies the scale itself, after the biases.
    G = (W_dec @ W_Q) @ (W_dec @ W_K).T

    dense = fra_qk(features, G, causal=True)  # [seq, seq, d_sae, d_sae]
    seq_len, _, d_sae, _ = dense.shape

    return {
        "fra_tensor_sparse": dense.to_sparse().coalesce(),
        "shape": (seq_len, seq_len, d_sae, d_sae),
        "seq_len": seq_len,
    }
