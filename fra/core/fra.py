import math

from transformer_lens import HookedTransformer
import torch
import numpy as np
from typing import Any, Dict
from einops import einsum
from fra.core.activations import get_llm_activations
from fra.core.helpers import topk_sparsify
from tqdm import tqdm



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
    
    # if self.feature_activations_active_mean is not None:
    #     interaction_matrix_unscaled *= self.feature_activations_active_mean[query_active_features][:, np.newaxis]
    #     interaction_matrix_unscaled *= self.feature_activations_active_mean[key_active_features][np.newaxis, :]
    #     matrix_scaling /= self.feature_activations_active_mean[query_active_features][:, np.newaxis]
    #     matrix_scaling /= self.feature_activations_active_mean[key_active_features][np.newaxis, :]
    
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

    


@torch.no_grad()
def get_sentence_fra_batch(
    model: HookedTransformer,
    sae: Any,
    text: str,
    layer: int,
    head: int,
    max_length: int = 128,
    top_k: int | None = 20,
    verbose: bool = False,
    hook_point: str = "ln1.hook_normalized",
    chunk_size: int = 16,
    normalize_by_decoder_norm: bool | None = None,
    prepend_bos: bool | None = None,
) -> Dict[str, Any]:
    """
    Compute full 4D Feature-Resolved Attention tensor for a sentence.
    Returns a sparse representation to avoid memory issues.

    hook_point controls which activation the SAE was trained on:
      - "ln1.hook_normalized"  (default, correct for FRA): decoder vectors live in
        the same d_model space that W_Q / W_K project from.  This is the only
        mathematically correct choice.
      - "attn.hook_z"          (legacy, for pre-trained hook_z SAEs): decoder
        vectors are in concatenated-heads space, not d_model space; the QK
        attention score computation is therefore approximate.

    Args:
        model: The transformer model
        sae: The SAE (any object with .encode() and .W_dec attributes)
        text: Input text to analyze
        layer: Which layer to analyze
        head: Which attention head to analyze
        max_length: Maximum sequence length
        top_k: Number of top features to keep per position
        verbose: Whether to show progress
        hook_point: Hookpoint the SAE was trained on (relative to blocks.{layer}.)
        chunk_size: Number of query positions to process per GPU batch before
                    flushing results to CPU.  Reduce for large SAEs (e.g. Gemma-Scope)
                    to avoid GPU OOM.  Set to seq_len to process everything at once.
        normalize_by_decoder_norm: Whether to divide feature activations by
                    decoder weight norms to match SAEs trained with
                    rescale_acts_by_decoder_norm=True.  None (default) auto-detects
                    from the SAE config.  True/False forces the behaviour.
        prepend_bos: Whether to include special tokens (BOS) in tokenization.
                    None (default) uses the tokenizer's default behaviour.
                    True/False forces add_special_tokens on/off.

    Returns:
        Dictionary containing:
            - fra_tensor_sparse: Sparse 4D tensor indices and values
            - shape: Shape of the full tensor [seq_len, seq_len, d_sae, d_sae]
            - seq_len: Actual sequence length
            - total_interactions: Total number of non-zero interactions
            - normalized: Whether decoder-norm normalization was applied
    """
    device = next(model.parameters()).device

    # Tokenise and truncate
    if prepend_bos is not None:
        tokens = model.tokenizer.encode(text, add_special_tokens=prepend_bos)
    else:
        tokens = model.tokenizer.encode(text)
    if max_length is not None and len(tokens) > max_length:
        tokens = tokens[:max_length]

    tokens_tensor = torch.tensor(tokens).unsqueeze(0).to(device)
    hook_name = f"blocks.{layer}.{hook_point}"
    _, cache = model.run_with_cache(tokens_tensor, names_filter=[hook_name])

    act = cache[hook_name].squeeze(0)  # remove batch dim
    seq_len = act.shape[0]

    # hook_z is [seq_len, n_heads, d_head] → flatten to [seq_len, n_heads*d_head]
    # ln1.hook_normalized is already [seq_len, d_model]
    if act.dim() == 3:
        act = act.flatten(-2, -1)

    # Encode to SAE features
    if verbose:
        print(f"Encoding {seq_len} positions to SAE features...")

    if hasattr(sae, 'encode'):
        feature_activations = sae.encode(act)   # [seq_len, d_sae]
    else:
        feature_activations = sae.sae.encode(act)

    # If the SAE normalizes inputs (e.g. Gemma-Scope), the feature activations
    # are in the normalized scale.  Divide by the norm coefficient so that
    # FRA[q,k,i,j] sums to the actual (un-normalized) QK attention score.
    # Top-k ranking is unaffected (same scalar per position).
    if hasattr(sae, '_norm_coeff') and sae._norm_coeff is not None:
        feature_activations = feature_activations / sae._norm_coeff

    d_sae = feature_activations.shape[-1]

    topk_features = topk_sparsify(feature_activations, top_k)

    # Get attention weights — handle GQA (e.g. Gemma-2: 8 Q heads, 4 KV heads)
    W_Q = model.blocks[layer].attn.W_Q[head]       # [d_model, d_head]
    n_kv = model.blocks[layer].attn.W_K.shape[0]
    n_q  = model.blocks[layer].attn.W_Q.shape[0]
    kv_head = head * n_kv // n_q                   # for GPT-2: kv_head == head
    W_K = model.blocks[layer].attn.W_K[kv_head]   # [d_model, d_head]
    d_head = W_Q.shape[-1]
    attn_scale = math.sqrt(d_head)

    # Get decoder weights
    if hasattr(sae, 'W_dec'):
        W_dec = sae.W_dec   # [d_sae, d_model]
    else:
        W_dec = sae.sae.W_dec

    # Handle rescale_acts_by_decoder_norm: when the SAE was trained with this
    # flag, encode() returns activations scaled UP by ||W_dec[i]||, and decode()
    # scales them back DOWN.  The true per-feature contribution to x is
    #   (f[i] / ||W_dec[i]||) * W_dec[i]
    # but without correction FRA would use f[i] * W_dec[i], inflating each
    # entry by ||W_dec[q_feat]|| * ||W_dec[k_feat]||.
    if normalize_by_decoder_norm is None:
        inner = sae.sae if hasattr(sae, 'sae') else sae
        cfg = getattr(inner, 'cfg', None)
        do_normalize = getattr(cfg, 'rescale_acts_by_decoder_norm', False) if cfg else False
    else:
        do_normalize = normalize_by_decoder_norm

    if do_normalize:
        dec_norms = W_dec.norm(dim=-1)  # [d_sae]
        if verbose:
            print(f"Applying decoder-norm correction (rescale_acts_by_decoder_norm)")
    else:
        dec_norms = None

    # Collect all sparse interactions on CPU (GPU only holds one chunk at a time).
    # Loop order: query-outer, key-inner.  For each query position we pre-compute
    # q_proj = W_dec[q_active] @ W_Q  once, then reuse across all key positions.
    # chunk_size controls how many query positions are batched before we flush to CPU,
    # bounding GPU memory to O(chunk_size × top_k² × d_head) at any time.
    all_indices_cpu: list[torch.Tensor] = []   # each: [4, n_int], CPU, long
    all_values_cpu:  list[torch.Tensor] = []   # each: [n_int],    CPU, float32

    total_pairs = seq_len * (seq_len + 1) // 2
    if verbose:
        pbar = tqdm(total=total_pairs, desc=f"Computing 4D FRA (L{layer}H{head})")

    for q_start in range(0, seq_len, chunk_size):
        q_end = min(q_start + chunk_size, seq_len)
        chunk_indices: list[torch.Tensor] = []
        chunk_values:  list[torch.Tensor] = []

        for query_idx in range(q_start, q_end):
            q_feat   = topk_features[query_idx]           # [d_sae]
            q_active = torch.where(q_feat != 0)[0]        # [n_q]

            if len(q_active) == 0:
                if verbose:
                    pbar.update(query_idx + 1)  # query_idx+1 key positions skipped
                continue

            # Pre-compute query projection once for all key positions in this row
            q_vecs  = W_dec[q_active]          # [n_q, d_model]
            q_proj  = q_vecs @ W_Q             # [n_q, d_head]
            q_scales = q_feat[q_active]        # [n_q]
            if dec_norms is not None:
                q_scales = q_scales / dec_norms[q_active]

            for key_idx in range(query_idx + 1):   # causal: key ≤ query
                k_feat   = topk_features[key_idx]
                k_active = torch.where(k_feat != 0)[0]

                if len(k_active) == 0:
                    if verbose:
                        pbar.update(1)
                    continue

                k_vecs = W_dec[k_active]       # [n_k, d_model]
                k_proj = k_vecs @ W_K          # [n_k, d_head]

                k_scales = k_feat[k_active]    # [n_k]
                if dec_norms is not None:
                    k_scales = k_scales / dec_norms[k_active]

                int_matrix = (q_proj @ k_proj.T) / attn_scale              # [n_q, n_k]
                int_matrix = int_matrix * q_scales.unsqueeze(1) * k_scales.unsqueeze(0)

                mask = int_matrix.abs() > 1e-10
                if mask.any():
                    local_r, local_c = torch.where(mask)
                    n_int = len(local_r)

                    # Build index tensor on CPU immediately — no GPU memory held
                    pos_indices = torch.empty((4, n_int), dtype=torch.long)
                    pos_indices[0] = query_idx
                    pos_indices[1] = key_idx
                    pos_indices[2] = q_active[local_r].cpu()
                    pos_indices[3] = k_active[local_c].cpu()

                    chunk_indices.append(pos_indices)
                    chunk_values.append(int_matrix[mask].detach().cpu().float())

                if verbose:
                    pbar.update(1)

        # Flush this chunk to main CPU lists and free GPU intermediates
        all_indices_cpu.extend(chunk_indices)
        all_values_cpu.extend(chunk_values)
        device_str = device.type if hasattr(device, 'type') else str(device)
        if device_str != "cpu":
            torch.cuda.empty_cache()

    if verbose:
        pbar.close()

    # Combine all interactions
    shape = (seq_len, seq_len, d_sae, d_sae)
    if len(all_indices_cpu) > 0:
        indices_cpu = torch.cat(all_indices_cpu, dim=1)  # [4, total_nnz]
        values_cpu  = torch.cat(all_values_cpu)          # [total_nnz]

        fra_tensor_sparse = torch.sparse_coo_tensor(
            indices_cpu, values_cpu,
            size=shape,
            device="cpu",
            dtype=torch.float32,
        ).coalesce()

        # Move to device only if it fits (for large SAEs keep on CPU)
        if str(device) != "cpu":
            try:
                fra_tensor_sparse = fra_tensor_sparse.to(device)
            except RuntimeError:
                if verbose:
                    print("Warning: sparse tensor too large for GPU, keeping on CPU.")

        total_interactions = fra_tensor_sparse._nnz()
    else:
        empty_indices = torch.zeros((4, 0), dtype=torch.long)
        empty_values  = torch.zeros(0, dtype=torch.float32)
        fra_tensor_sparse = torch.sparse_coo_tensor(
            empty_indices, empty_values, size=shape, device="cpu"
        )
        total_interactions = 0

    if verbose:
        _k = top_k if top_k is not None else d_sae
        density = total_interactions / (seq_len * seq_len * _k * _k)
        print(f"4D FRA tensor: shape={shape}, nnz={total_interactions:,}, density={density:.2%}")
        sparse_mem = (total_interactions * 5 * 4) / (1024**2)   # 4 indices + 1 value
        dense_mem  = (seq_len * seq_len * d_sae * d_sae * 4) / (1024**3)
        print(f"Memory: sparse={sparse_mem:.2f}MB vs dense={dense_mem:.2f}GB")

    return {
        'fra_tensor_sparse': fra_tensor_sparse,
        'shape': shape,
        'seq_len': seq_len,
        'total_interactions': total_interactions,
        'normalized': do_normalize,
    }


if __name__ == "__main__":
    print('main character')