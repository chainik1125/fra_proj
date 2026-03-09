from transformer_lens import HookedTransformer
import torch
import numpy as np
from typing import Any, Dict
from einops import einsum
from fra.activation_utils import get_llm_activations
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
        
    attention_scores = einsum(q, k, "q a, k a -> q k")

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
    top_k: int = 20,
    verbose: bool = False,
    hook_point: str = "ln1.hook_normalized",
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

    Returns:
        Dictionary containing:
            - fra_tensor_sparse: Sparse 4D tensor indices and values
            - shape: Shape of the full tensor [seq_len, seq_len, d_sae, d_sae]
            - seq_len: Actual sequence length
            - total_interactions: Total number of non-zero interactions
    """
    device = next(model.parameters()).device

    # Tokenise and truncate
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
    
    d_sae = feature_activations.shape[-1]
    
    # Keep only top-k features per position
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
    
    # Get attention weights
    W_Q = model.blocks[layer].attn.W_Q[head]
    W_K = model.blocks[layer].attn.W_K[head]
    
    # Get decoder weights
    if hasattr(sae, 'W_dec'):
        W_dec = sae.W_dec
    else:
        W_dec = sae.sae.W_dec
    
    # Collect all sparse interactions
    # Format: [query_pos, key_pos, query_feat, key_feat] -> value
    all_indices = []
    all_values = []
    
    total_pairs = seq_len * (seq_len + 1) // 2
    if verbose:
        pbar = tqdm(total=total_pairs, desc=f"Computing 4D FRA (L{layer}H{head})")
    
    for key_idx in range(seq_len):
        for query_idx in range(key_idx, seq_len):  # Lower triangular
            q_feat = topk_features[query_idx]
            k_feat = topk_features[key_idx]
            
            q_active = torch.where(q_feat != 0)[0]
            k_active = torch.where(k_feat != 0)[0]
            
            if len(q_active) == 0 or len(k_active) == 0:
                if verbose:
                    pbar.update(1)
                continue
            
            # Get decoder vectors
            q_vecs = W_dec[q_active]
            k_vecs = W_dec[k_active]
            
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
                
                # Create 4D indices: [query_pos, key_pos, query_feat, key_feat]
                n_interactions = len(local_r)
                pos_indices = torch.zeros((4, n_interactions), dtype=torch.long)
                pos_indices[0, :] = query_idx  # Query position
                pos_indices[1, :] = key_idx    # Key position
                pos_indices[2, :] = q_active[local_r]  # Query feature
                pos_indices[3, :] = k_active[local_c]  # Key feature
                
                all_indices.append(pos_indices)
                all_values.append(int_matrix[mask])
            
            if verbose:
                pbar.update(1)
    
    if verbose:
        pbar.close()
    
    # Combine all interactions
    if len(all_indices) > 0:
        indices = torch.cat(all_indices, dim=1).to(device)
        values = torch.cat(all_values).to(device)
        
        # Create sparse 4D tensor
        shape = (seq_len, seq_len, d_sae, d_sae)
        fra_tensor_sparse = torch.sparse_coo_tensor(
            indices, values,
            size=shape,
            device=device,
            dtype=torch.float32
        ).coalesce()
        
        total_interactions = fra_tensor_sparse._nnz()
    else:
        # Empty tensor
        shape = (seq_len, seq_len, d_sae, d_sae)
        empty_indices = torch.zeros((4, 0), dtype=torch.long, device=device)
        empty_values = torch.zeros(0, dtype=torch.float32, device=device)
        
        fra_tensor_sparse = torch.sparse_coo_tensor(
            empty_indices, empty_values,
            size=shape,
            device=device
        )
        total_interactions = 0
    
    if verbose:
        density = total_interactions / (seq_len * seq_len * top_k * top_k)
        print(f"4D FRA tensor: shape={shape}, nnz={total_interactions:,}, density={density:.2%}")
        
        # Memory estimate
        sparse_mem = (total_interactions * (4 + 1) * 4) / (1024**2)  # 4 indices + 1 value, 4 bytes each
        dense_mem = (seq_len * seq_len * d_sae * d_sae * 4) / (1024**3)  # GB
        print(f"Memory: sparse={sparse_mem:.2f}MB vs dense={dense_mem:.2f}GB")
    
    return {
        'fra_tensor_sparse': fra_tensor_sparse,
        'shape': shape,
        'seq_len': seq_len,
        'total_interactions': total_interactions
    }


if __name__ == "__main__":
    print('main character')