"""
Feature-Resolved Attention (FRA) computation.

Public entry points (full pipeline: tokens → result dict):
  - ``compute_fra``:            one model, one coder
  - ``compute_fra_model_diff``: two models whose residuals are stacked before
                                encoding (model-diffing setup)

Both accept pre-tokenized token lists.  Internally they call ``_build_fra_result``,
which handles weight lookup, RoPE extraction, and delegates to ``_compute_fra_sparse``.
``_compute_fra_sparse`` is the private tensor kernel: it operates purely on
pre-computed feature activations and weight matrices with no knowledge of models
or tokens.
"""

import math

import numpy as np
import torch
from typing import Any, Dict
from einops import einsum
from tqdm import tqdm
from transformer_lens import HookedTransformer

from fra.core.activations import get_llm_activations
from fra.core.helpers import get_W_K, apply_rope_to_projected, _extract_rope_params


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

    d_sae = feature_activations.shape[1]
    k = min(top_k, d_sae)

    # Vectorized: one topk + scatter instead of a Python loop over positions
    topk_idx = torch.topk(feature_activations.abs(), k=k, dim=-1).indices  # [seq, k]
    out = torch.zeros_like(feature_activations)
    out.scatter_(-1, topk_idx, feature_activations.gather(-1, topk_idx))
    return out


# ── Core FRA loop ───────────────────────────────────────────────────────


def _compute_fra_sparse(
    topk_features: torch.Tensor,
    W_dec: torch.Tensor,
    W_Q: torch.Tensor,
    W_K: torch.Tensor,
    attn_scale: float,
    *,
    rms: torch.Tensor | None = None,
    dec_norms: torch.Tensor | None = None,
    rope_sin: torch.Tensor | None = None,
    rope_cos: torch.Tensor | None = None,
    rotary_dim: int | None = None,
    rotary_adjacent_pairs: bool = False,
    chunk_size: int = 16,
    verbose: bool = False,
    layer_head_label: str = "",
) -> torch.sparse_coo_tensor:
    """Tensor kernel: compute the 4-D FRA sparse tensor from pre-computed inputs.

    This is the inner loop only.  It has no knowledge of models, tokenization,
    or encoding — all of that lives in ``compute_fra`` / ``compute_fra_model_diff``
    and their shared helper ``_build_fra_result``.  Call those instead unless
    you already have feature activations and weight matrices in hand.

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
        rope_sin: Optional ``[n_ctx, rotary_dim]`` precomputed sine table
                from the attention block.  Pass for RoPE models (Gemma, Llama).
        rope_cos: Optional ``[n_ctx, rotary_dim]`` precomputed cosine table.
        rotary_dim: Number of RoPE dimensions (``model.cfg.rotary_dim``).
        rotary_adjacent_pairs: RoPE pair ordering flag from model config.
        chunk_size: Unused; kept for API compatibility.
        verbose: Show progress bar.
        layer_head_label: Label for progress bar (e.g. ``"L5H3"``).

    Returns:
        ``torch.sparse_coo_tensor`` on CPU, shape
        ``[seq_len, seq_len, d_sae, d_sae]``, coalesced.
    """
    seq_len = topk_features.shape[0]
    d_sae = topk_features.shape[1]
    device = topk_features.device
    shape = (seq_len, seq_len, d_sae, d_sae)

    use_rope = rope_sin is not None

    # ── Pre-compute key-side projections for every position — O(T) GPU ops ──
    # Each position's data is computed exactly once, not once per attending query.
    k_active_list: list = [None] * seq_len   # active feature index tensor per pos
    k_proj_list:   list = [None] * seq_len   # [n_k, d_head] RoPE-rotated projection
    k_scales_list: list = [None] * seq_len   # [n_k] weighted activations

    for t in range(seq_len):
        feat = topk_features[t]
        active = feat.nonzero(as_tuple=True)[0]
        if len(active) == 0:
            continue
        proj = W_dec[active] @ W_K                              # [n_k, d_head]
        scales = feat[active]
        if dec_norms is not None:
            scales = scales / dec_norms[active]
        if use_rope:
            proj = apply_rope_to_projected(
                proj, t, rope_sin, rope_cos, rotary_dim, rotary_adjacent_pairs,
            )
        k_active_list[t] = active
        k_proj_list[t]   = proj
        k_scales_list[t] = scales

    # Count total active key features to pre-allocate buffers
    total_k = sum(len(a) for a in k_active_list if a is not None)

    if total_k == 0:
        return torch.sparse_coo_tensor(
            torch.zeros((4, 0), dtype=torch.long),
            torch.zeros(0, dtype=torch.float32),
            size=shape, device="cpu",
        )

    # Pre-allocate contiguous key buffers.  Filled incrementally as the
    # query loop advances, so that position t is a key candidate for all
    # query positions t' ≥ t (causal masking is natural).
    k_proj_buf   = torch.empty(total_k, W_K.shape[-1], dtype=torch.float32, device=device)
    k_scales_buf = torch.empty(total_k, dtype=torch.float32, device=device)
    k_active_buf = torch.empty(total_k, dtype=torch.long, device=device)
    k_pos_buf    = torch.empty(total_k, dtype=torch.long, device=device)
    fill_ptr = 0

    all_indices_cpu: list[torch.Tensor] = []
    all_values_cpu:  list[torch.Tensor] = []

    pbar = tqdm(
        total=seq_len,
        desc=f"Computing FRA ({layer_head_label})" if layer_head_label else "Computing FRA",
        disable=not verbose,
    )

    # ── Main loop: O(T) iterations, one batched matmul per query position ───
    for query_idx in range(seq_len):
        # Extend key buffer: position query_idx is now a valid key target.
        k_a = k_active_list[query_idx]
        if k_a is not None:
            n_k = len(k_a)
            k_proj_buf  [fill_ptr:fill_ptr + n_k] = k_proj_list[query_idx]
            k_scales_buf[fill_ptr:fill_ptr + n_k] = k_scales_list[query_idx]
            k_active_buf[fill_ptr:fill_ptr + n_k] = k_a
            k_pos_buf   [fill_ptr:fill_ptr + n_k] = query_idx
            fill_ptr += n_k

        if fill_ptr == 0:
            pbar.update(1)
            continue

        # Query projection
        q_feat   = topk_features[query_idx]
        q_active = q_feat.nonzero(as_tuple=True)[0]
        if len(q_active) == 0:
            pbar.update(1)
            continue

        q_proj = W_dec[q_active] @ W_Q                         # [n_q, d_head]
        q_scales = q_feat[q_active]
        if dec_norms is not None:
            q_scales = q_scales / dec_norms[q_active]
        if use_rope:
            q_proj = apply_rope_to_projected(
                q_proj, query_idx, rope_sin, rope_cos, rotary_dim, rotary_adjacent_pairs,
            )

        # Single batched matmul against all key positions so far [n_q, total_k_so_far]
        k_p = k_proj_buf[:fill_ptr]
        k_s = k_scales_buf[:fill_ptr]
        k_a_view = k_active_buf[:fill_ptr]
        k_pos_view = k_pos_buf[:fill_ptr]

        int_matrix = (q_proj @ k_p.T) / attn_scale             # [n_q, total_k]
        int_matrix = int_matrix * q_scales.unsqueeze(1) * k_s.unsqueeze(0)

        if rms is not None:
            int_matrix = int_matrix / (rms[query_idx] * rms[k_pos_view].unsqueeze(0))

        mask = int_matrix.abs() > 1e-10
        if mask.any():
            local_r, local_c = torch.where(mask)
            pos_indices = torch.stack([
                torch.full((len(local_r),), query_idx, dtype=torch.long),
                k_pos_view[local_c].cpu(),
                q_active[local_r].cpu(),
                k_a_view[local_c].cpu(),
            ], dim=0)
            all_indices_cpu.append(pos_indices)
            all_values_cpu.append(int_matrix[mask].detach().cpu().float())

        pbar.update(1)

    pbar.close()

    if all_indices_cpu:
        indices = torch.cat(all_indices_cpu, dim=1)
        values  = torch.cat(all_values_cpu)
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


def _build_fra_result(
    model: HookedTransformer,
    layer: int,
    head: int,
    feature_activations: torch.Tensor,
    W_dec: torch.Tensor,
    device,
    *,
    top_k: int | None = 20,
    topk_features: torch.Tensor | None = None,
    rms_activations: torch.Tensor | None = None,
    rms: torch.Tensor | None = None,
    dec_norms: torch.Tensor | None = None,
    chunk_size: int = 16,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Shared post-encoding FRA construction.

    Handles top-k sparsification, weight extraction, RMSNorm correction,
    RoPE, ``_compute_fra_sparse``, device transfer, and result assembly.

    Args:
        model: Target model (for W_Q / W_K and RoPE parameters).
        layer: Attention layer index.
        head: Attention head index.
        feature_activations: ``[seq, d_sae]`` raw encoded features.
        W_dec: ``[d_sae, d_model]`` decoder weights.
        device: Torch device for the result tensor.
        top_k: Features per position to keep (None = all).  Ignored when
            ``topk_features`` is provided.
        topk_features: Pre-computed ``[seq, d_sae]`` sparsified tensor.  If
            provided, ``topk_sparsify`` is skipped (useful when calling
            per-head to avoid redundant work).
        rms_activations: ``[seq, d_model]`` residual stream for RMSNorm
            correction, or None to skip.
        rms: Pre-computed ``[seq]`` RMSNorm denominators.  If provided,
            skips recomputation from ``rms_activations``.
        dec_norms: ``[d_sae]`` decoder norms for rescale correction, or None.
        chunk_size: GPU batch size for ``_compute_fra_sparse``.
        verbose: Show progress.

    Returns:
        Dict with ``fra_tensor_sparse``, ``shape``, ``seq_len``,
        ``total_interactions``, ``feature_activations``, ``topk_features``.
    """
    if topk_features is None:
        topk_features = topk_sparsify(feature_activations, top_k).float()

    # Attention weights — float32 for accumulation precision
    W_Q = model.blocks[layer].attn.W_Q[head].float()       # [d_model, d_head]
    W_K_mat = get_W_K(model, layer, head).float()           # [d_model, d_head]
    d_head = W_Q.shape[-1]
    attn_scale = math.sqrt(d_head)

    # Normalization correction
    #
    # TransformerLens folds gamma/beta into W_Q / W_K.  For LNPre models
    # we also fold the centering projection P into W_dec and b_dec at
    # init (FRACoder fold_ln=True), so W_dec and rms_activations arrive
    # here already centered.
    W_dec_corr = W_dec.float()

    if rms is None and rms_activations is not None:
        eps = model.cfg.eps
        rms = (rms_activations.float().pow(2).mean(dim=-1) + eps).sqrt()

    # RoPE
    rope_sin, rope_cos, rotary_dim, rotary_adjacent_pairs = (
        _extract_rope_params(model, layer)
    )

    fra_tensor_sparse = _compute_fra_sparse(
        topk_features, W_dec_corr, W_Q, W_K_mat, attn_scale,
        rms=rms,
        dec_norms=dec_norms,
        rope_sin=rope_sin,
        rope_cos=rope_cos,
        rotary_dim=rotary_dim,
        rotary_adjacent_pairs=rotary_adjacent_pairs,
        chunk_size=chunk_size,
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

    seq_len = feature_activations.shape[0]
    d_sae = feature_activations.shape[-1]
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
        "topk_features": topk_features,
    }


# ── Legacy analysis helpers ────────────────────────────────────────────


def lower_triangular_mask(pattern: np.ndarray) -> np.ma.MaskedArray:
    """Apply lower triangular mask to attention pattern."""
    mask = np.triu(np.ones(pattern.shape), k=1)
    return np.ma.array(np.tril(pattern, k=0), mask=mask)


def attention_pattern_QK(llm: Any, layer: int, head: int, q_input: torch.Tensor,
                        q_do_bias: bool, k_input: torch.Tensor, k_do_bias: bool) -> np.ndarray:
    """
    Compute attention pattern from query and key inputs.

    Args:
        layer: Layer index
        head: Head index
        q_input: Query input tensor
        q_do_bias: Whether to add query bias
        k_input: Key input tensor
        k_do_bias: Whether to add key bias

    Returns:
        Attention scores as numpy array
    """
    W_Q = llm.blocks[layer].attn.W_Q[head]
    b_Q = llm.blocks[layer].attn.b_Q[head]
    W_K = llm.blocks[layer].attn.W_K[head]
    b_K = llm.blocks[layer].attn.b_K[head]

    q = einsum(W_Q, q_input, "d a, s d -> s a")
    if q_do_bias:
        q += b_Q

    k = einsum(W_K, k_input, "d a, s d -> s a")
    if k_do_bias:
        k += b_K

    d_head = W_Q.shape[-1]
    attention_scores = einsum(q, k, "q a, k a -> q k") / math.sqrt(d_head)

    return attention_scores.detach().cpu().numpy()


def analyze_feature_attention_interactions(model: Any, sae: Any, layer: int, head: int,
                                           input_text: str, query_position: int, key_position: int,
                                           hook_point: str = "hook_attn_out") -> Dict:
    """
    Analyze interactions between features in attention.

    Args:
        layer: Layer index
        head: Head index
        input_text: Input text to analyze
        query_position: Query position
        key_position: Key position
    """
    activations_SD = get_llm_activations(model,input_text,hook_point=hook_point,layers=layer)
    feature_activations_SH = sae.encode(activations_SD)

    feature_activations_query = feature_activations_SH[query_position]
    query_active_features = torch.where(feature_activations_query != 0)[0]

    feature_activations_key = feature_activations_SH[key_position]
    key_active_features = torch.where(feature_activations_key != 0)[0]

    query_activations_for_features = sae.W_dec[query_active_features]
    key_activations_for_features = sae.W_dec[key_active_features]

    interaction_matrix_unscaled = attention_pattern_QK(model, layer, head,
                                                        query_activations_for_features, False,
                                                        key_activations_for_features, False)



    # Convert to numpy after using for indexing
    query_features_tensor = query_active_features
    key_features_tensor = key_active_features

    matrix_scaling = feature_activations_query[query_features_tensor].unsqueeze(1) * \
                    feature_activations_key[key_features_tensor].unsqueeze(0)
    matrix_scaling = matrix_scaling.detach().cpu().numpy()

    query_active_features = query_features_tensor.cpu().numpy()
    key_active_features = key_features_tensor.cpu().numpy()

    return {
        'query_active_features': query_active_features,
        'key_active_features': key_active_features,
        'interaction_matrix_unscaled': interaction_matrix_unscaled,
        'matrix_scaling': matrix_scaling,
        'interaction_matrix': interaction_matrix_unscaled * matrix_scaling
    }


def get_sentence_averages(llm:Any,sae:Any,layer:int,head:int,input_text:str,hook_point:str="attn.hook_z"):
	text_length=128
	hidden_dim=sae.d_sae
	data_dep_int_matrix=np.zeros((hidden_dim,hidden_dim))
	data_dep_int_matrix_abs=np.zeros((hidden_dim,hidden_dim))
	data_dep_localization_matrix=np.zeros((hidden_dim,hidden_dim))
	count=0
	for key_index in tqdm(range(text_length), disable=True):
		for query_index in range(key_index,text_length):
				feature_analysis=analyze_feature_attention_interactions(llm,sae,layer,head,input_text,query_index,key_index,hook_point)
				int_matrix=feature_analysis["interaction_matrix"]
				query_active_features=feature_analysis["query_active_features"]
				key_active_features=feature_analysis["key_active_features"]
				data_independent=feature_analysis["interaction_matrix_unscaled"]

				resized_data_dependent_int=np.zeros((hidden_dim,hidden_dim))
				resized_data_dependent_int[query_active_features[:,None],key_active_features[None,:]]=int_matrix

				resized_data_dependent_localization=np.zeros((hidden_dim,hidden_dim))
				resized_data_dependent_localization[query_active_features[:,None],key_active_features[None,:]]=np.abs(int_matrix)*(query_index-key_index)




				data_dep_int_matrix=data_dep_int_matrix+resized_data_dependent_int
				data_dep_int_matrix_abs=data_dep_int_matrix_abs+np.abs(resized_data_dependent_int)
				data_dep_localization_matrix=data_dep_localization_matrix+resized_data_dependent_localization

				count+=1

	data_dep_int_matrix/=count
	data_dep_localization_matrix=data_dep_localization_matrix/np.clip(data_dep_int_matrix_abs,a_min=1,a_max=None)
	data_dep_int_matrix_abs/=count

	return data_dep_int_matrix,data_dep_int_matrix_abs,data_dep_localization_matrix




# ═══════════════════════════════════════════════════════════════════════
# Head-independent encoding helpers
# ═══════════════════════════════════════════════════════════════════════


def _encode_sae(
    model: HookedTransformer,
    sae: Any,
    tokens_tensor: torch.Tensor,
    layer: int,
    hook_point: str = "ln1.hook_normalized",
    *,
    activation: torch.Tensor | None = None,
    normalize_by_decoder_norm: bool | None = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Head-independent SAE encoding.

    Runs the model forward pass (unless *activation* is provided), encodes
    through the SAE, and computes all artifacts needed by
    ``_build_fra_result``.

    Args:
        model: The transformer model.
        sae: SAE object with ``.encode()`` and ``.W_dec``.
        tokens_tensor: ``[1, seq]`` token tensor (on device).
        layer: Layer index.
        hook_point: Hook point relative to ``blocks.{layer}.``.
        activation: Pre-computed ``[seq, d_model]`` activation tensor.  If
            provided the model forward pass is skipped — useful when the
            caller captured multiple hooks in a single ``run_with_cache``.
        normalize_by_decoder_norm: Override decoder-norm correction.
        verbose: Print progress.

    Returns:
        Dict with ``feature_activations``, ``W_dec``, ``dec_norms``,
        ``rms_activations``, ``normalized``.
    """
    device = next(model.parameters()).device

    if activation is None:
        hook_name = f"blocks.{layer}.{hook_point}"
        _, cache = model.run_with_cache(
            tokens_tensor, names_filter=[hook_name],
        )
        activation = cache[hook_name].squeeze(0)

    # hook_z is [seq_len, n_heads, d_head] → flatten to [seq_len, n_heads*d_head]
    if activation.dim() == 3:
        activation = activation.flatten(-2, -1)

    # Encode to SAE features
    if verbose:
        print(f"Encoding {activation.shape[0]} positions to SAE features...")

    feature_activations = sae.encode(activation)

    # Norm coefficient correction (e.g. Gemma-Scope)
    if sae._norm_coeff is not None:
        feature_activations = feature_activations / sae._norm_coeff

    W_dec = sae.W_dec

    # Decoder norm detection
    if normalize_by_decoder_norm is None:
        dec_norms = sae.dec_norms
    elif normalize_by_decoder_norm:
        dec_norms = W_dec.norm(dim=-1)
    else:
        dec_norms = None
    if dec_norms is not None and verbose:
        print("Applying decoder-norm correction (rescale_acts_by_decoder_norm)")

    # RMSNorm correction setup
    if "resid" in hook_point:
        b_dec = sae.b_dec
        rms_activations = feature_activations @ W_dec + b_dec
    else:
        rms_activations = None

    return {
        "feature_activations": feature_activations,
        "W_dec": W_dec,
        "dec_norms": dec_norms,
        "rms_activations": rms_activations,
        "normalized": dec_norms is not None,
    }


def _encode_crosscoder(
    base_model: HookedTransformer,
    it_model: HookedTransformer,
    crosscoder: Any,
    tokens_tensor: torch.Tensor,
    crosscoder_layer: int,
    *,
    target_activation: torch.Tensor | None = None,
    other_activation: torch.Tensor | None = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Head-independent crosscoder encoding.

    Runs both model forward passes (unless activations are provided),
    encodes through the crosscoder, and prepares RMSNorm artifacts.

    Args:
        base_model: HookedTransformer for model-index 0 (base).
        it_model: HookedTransformer for model-index 1 (instruct).
        crosscoder: ``FRACoder`` instance.
        tokens_tensor: ``[1, seq]`` token tensor (on device).
        crosscoder_layer: Crosscoder residual-stream layer.
        target_activation: Pre-computed ``[seq, d_model]`` from target model.
        other_activation: Pre-computed ``[seq, d_model]`` from other model.
        verbose: Print progress.

    Returns:
        Dict with ``feature_activations``, ``W_dec``, ``dec_norms``,
        ``rms_activations``.
    """
    target_model = base_model if crosscoder.model_idx == 0 else it_model
    other_model = it_model if crosscoder.model_idx == 0 else base_model
    device = next(target_model.parameters()).device

    hook_name = f"blocks.{crosscoder_layer}.hook_resid_post"

    if target_activation is None:
        _, target_cache = target_model.run_with_cache(
            tokens_tensor, names_filter=[hook_name],
        )
        target_activation = target_cache[hook_name].squeeze(0)

    if other_activation is None:
        _, other_cache = other_model.run_with_cache(
            tokens_tensor, names_filter=[hook_name],
        )
        other_activation = other_cache[hook_name].squeeze(0)

    # Stack in crosscoder order: [base, instruct]
    if crosscoder.model_idx == 0:
        x_stacked = torch.stack([target_activation, other_activation], dim=1)
    else:
        x_stacked = torch.stack([other_activation, target_activation], dim=1)

    if verbose:
        print(
            f"Encoding {target_activation.shape[0]} positions through crosscoder..."
        )
    feature_activations = crosscoder.encode(x_stacked)

    # RMSNorm correction always needed — decode in float32 for precision
    b_dec_rms = crosscoder.b_dec.float().to(device)
    rms_activations = (
        feature_activations.float() @ crosscoder.W_dec.float() + b_dec_rms
    )

    return {
        "feature_activations": feature_activations,
        "W_dec": crosscoder.W_dec,
        "dec_norms": None,
        "rms_activations": rms_activations,
    }


# ═══════════════════════════════════════════════════════════════════════
# Entry point: single-model path
# ═══════════════════════════════════════════════════════════════════════


@torch.no_grad()
def compute_fra(
    model: HookedTransformer,
    coder: Any,
    tokens: list,
    layer: int,
    head: int,
    max_length: int = 128,
    top_k: int | None = 20,
    verbose: bool = False,
    hook_point: str = "ln1.hook_normalized",
    chunk_size: int = 16,
    normalize_by_decoder_norm: bool | None = None,
) -> Dict[str, Any]:
    """Compute the 4-D FRA sparse tensor for a token sequence using one model.

    hook_point controls which activation the coder was trained on:
      - "ln1.hook_normalized"  (default, correct for FRA): decoder vectors live in
        the same d_model space that W_Q / W_K project from.  This is the only
        mathematically correct choice.
      - "attn.hook_z"          (legacy, for pre-trained hook_z SAEs): decoder
        vectors are in concatenated-heads space, not d_model space; the QK
        attention score computation is therefore approximate.

    Args:
        model: The transformer model.
        coder: Any object with .encode() and .W_dec attributes (SAE or FRACoder).
        tokens: Pre-tokenized token IDs.  Caller is responsible for tokenization
                (including BOS / chat template if needed).
        layer: Which layer to analyze.
        head: Which attention head to analyze.
        max_length: Truncate tokens to this length.
        top_k: Number of top features to keep per position.
        verbose: Whether to show progress.
        hook_point: Hookpoint the coder was trained on (relative to blocks.{layer}.)
        chunk_size: Query positions per GPU batch.  Reduce for large coders to
                    avoid GPU OOM; set to seq_len to process everything at once.
        normalize_by_decoder_norm: Divide feature activations by decoder weight
                    norms (rescale_acts_by_decoder_norm).  None auto-detects from
                    the coder config; True/False forces the behaviour.

    Returns:
        Dictionary containing:
            - fra_tensor_sparse: Sparse 4D tensor indices and values
            - shape: Shape of the full tensor [seq_len, seq_len, d_sae, d_sae]
            - seq_len: Actual sequence length
            - total_interactions: Total number of non-zero interactions
            - normalized: Whether decoder-norm normalization was applied
    """
    device = next(model.parameters()).device

    if max_length is not None and len(tokens) > max_length:
        tokens = tokens[:max_length]
    tokens_tensor = torch.tensor(tokens).unsqueeze(0).to(device)

    encoded = _encode_sae(
        model, coder, tokens_tensor, layer, hook_point,
        normalize_by_decoder_norm=normalize_by_decoder_norm,
        verbose=verbose,
    )

    result = _build_fra_result(
        model, layer, head,
        encoded["feature_activations"], encoded["W_dec"], device,
        top_k=top_k,
        rms_activations=encoded["rms_activations"],
        dec_norms=encoded["dec_norms"],
        chunk_size=chunk_size,
        verbose=verbose,
    )
    result["normalized"] = encoded["normalized"]
    return result


# ═══════════════════════════════════════════════════════════════════════
# Entry point: paired-model path
# ═══════════════════════════════════════════════════════════════════════


@torch.no_grad()
def compute_fra_model_diff(
    base_model: HookedTransformer,
    it_model: HookedTransformer,
    coder: Any,
    tokens: list,
    head: int = 0,
    coder_layer: int = 13,
    max_length: int = 128,
    top_k: int | None = 20,
    verbose: bool = False,
    chunk_size: int = 16,
) -> Dict[str, Any]:
    """Compute the 4-D FRA sparse tensor using residuals from two models.

    Both models are run on the same token sequence; their residuals at
    ``coder_layer`` are stacked and passed to the coder for encoding.
    The coder was trained on ``hook_resid_post`` at ``coder_layer``, which
    is the input to the next layer's attention, so the FRA decomposition
    always targets attention at ``coder_layer + 1``.

    The attention layer is **not** independently configurable — it is always
    ``coder_layer + 1``.  This ensures the FRA decomposition corresponds
    to the features the coder actually learned.

    Args:
        base_model:  HookedTransformer for model-index 0 (base).
        it_model:    HookedTransformer for model-index 1 (instruct).
        coder:       ``FRACoder`` instance trained on paired residuals.
        tokens:      Pre-tokenized token IDs.  Caller is responsible for
                     tokenization (including chat template if needed).
        head:        Attention head index.
        coder_layer: Layer from which to extract residual-stream activations
                     (default 13, matching the published checkpoint).
                     Uses ``hook_resid_post`` at this layer.  The attention
                     layer analyzed is derived as ``coder_layer + 1``.
        max_length:  Truncate tokens to this length.
        top_k:       Keep only top-k features per position.
        verbose:     Show progress bar.
        chunk_size:  Query positions per GPU batch (controls peak memory).

    Returns:
        Dict with keys:
          - ``fra_tensor_sparse``  (torch.sparse_coo_tensor, 4-D)
          - ``shape``
          - ``seq_len``
          - ``total_interactions``
          - ``feature_activations``  (torch.Tensor, [seq, d_sae])
          - ``topk_features``  (torch.Tensor, [seq, d_sae])
    """
    layer = coder_layer + 1
    target_model = base_model if coder.model_idx == 0 else it_model
    device = next(target_model.parameters()).device

    if max_length is not None and len(tokens) > max_length:
        tokens = tokens[:max_length]
    tokens_tensor = torch.tensor(tokens).unsqueeze(0).to(device)

    encoded = _encode_crosscoder(
        base_model, it_model, coder, tokens_tensor, coder_layer,
        verbose=verbose,
    )

    return _build_fra_result(
        target_model, layer, head,
        encoded["feature_activations"], encoded["W_dec"], device,
        top_k=top_k,
        rms_activations=encoded["rms_activations"],
        chunk_size=chunk_size,
        verbose=verbose,
    )


@torch.no_grad()
def compute_fra_model_diff_all_heads(
    base_model: HookedTransformer,
    it_model: HookedTransformer,
    coder: Any,
    tokens: list,
    n_heads: int,
    coder_layer: int = 13,
    max_length: int = 128,
    top_k: int | None = 20,
    verbose: bool = False,
    chunk_size: int = 16,
) -> Dict[str, Any]:
    """Compute the 4-D FRA sparse tensor for every attention head in one call.

    Runs both model forward passes and the crosscoder encoding **once** (not
    once per head), then calls ``_build_fra_result`` for each head using the
    shared encoding artifacts.  This is ``n_heads`` times more efficient than
    calling :func:`compute_fra_model_diff` in a loop.

    Args:
        base_model: HookedTransformer for model-index 0 (base).
        it_model:   HookedTransformer for model-index 1 (instruct).
        coder:      ``FRACoder`` instance trained on paired residuals.
        tokens:     Pre-tokenized token IDs.
        n_heads:    Number of attention heads in the target layer.
        coder_layer: Crosscoder residual-stream layer (default 13).
        max_length:  Truncate tokens to this length.
        top_k:       Keep only top-k features per position.
        verbose:     Show progress bar per head.
        chunk_size:  Unused; kept for API compatibility.

    Returns:
        Dict with:
          - ``fra_sparse_dict``: ``dict[int, sparse_coo_tensor]`` head → FRA.
          - ``feature_activations``: ``[seq, d_sae]`` float32 on CPU.
          - ``topk_features``: ``[seq, d_sae]`` float32.
          - ``seq_len``: int.
    """
    layer = coder_layer + 1
    target_model = base_model if coder.model_idx == 0 else it_model
    device = next(target_model.parameters()).device

    if max_length is not None and len(tokens) > max_length:
        tokens = tokens[:max_length]
    tokens_tensor = torch.tensor(tokens).unsqueeze(0).to(device)

    # Single encode: 2 model forward passes shared across all heads
    encoded = _encode_crosscoder(
        base_model, it_model, coder, tokens_tensor, coder_layer, verbose=verbose,
    )

    feature_activations = encoded["feature_activations"]
    W_dec = encoded["W_dec"]

    # Head-independent artifacts computed once
    topk_features = topk_sparsify(feature_activations, top_k).float()

    eps = target_model.cfg.eps
    rms = (encoded["rms_activations"].float().pow(2).mean(dim=-1) + eps).sqrt()

    fra_sparse_dict: Dict[int, Any] = {}
    for h in range(n_heads):
        result = _build_fra_result(
            target_model, layer, h,
            feature_activations, W_dec, device,
            topk_features=topk_features,
            rms=rms,
            chunk_size=chunk_size,
            verbose=verbose,
        )
        fra_sparse_dict[h] = result["fra_tensor_sparse"].cpu()

    return {
        "fra_sparse_dict": fra_sparse_dict,
        "feature_activations": feature_activations.cpu().float(),
        "topk_features": topk_features.cpu(),
        "seq_len": feature_activations.shape[0],
    }


if __name__ == "__main__":
    print('main character')
