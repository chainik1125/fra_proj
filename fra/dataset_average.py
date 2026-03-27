#!/usr/bin/env python
"""
Simple function to compute average feature interactions over dataset samples.
"""

import torch
from typing import List, Dict, Any, Optional
from transformer_lens import HookedTransformer
from tqdm import tqdm

from fra.fra_func import get_sentence_fra_batch
from fra.utils import infer_hook_point_from_sae, load_model_and_sae_from_config


def compute_dataset_average(
    model: HookedTransformer,
    sae: Any,
    dataset_texts: List[str],
    layer: int = 5,
    head: int = 0,
    filter_self_interactions: bool = True,
    hook_point: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Compute average feature interactions over dataset samples.
    
    Args:
        model: The transformer model
        sae: SAE for feature extraction
        dataset_texts: List of texts to analyze
        layer: Layer to analyze
        head: Head to analyze
        filter_self_interactions: Whether to filter self-interactions
        verbose: Whether to show progress
        
    Returns:
        Dictionary with:
            - interaction_sum: Sparse dict of summed interactions
            - interaction_count: Sparse dict of counts
            - num_samples: Number of samples processed
            - d_sae: SAE dimension
    """
    d_sae = sae.W_dec.shape[0] if hasattr(sae, "W_dec") else sae.sae.W_dec.shape[0]
    device = next(model.parameters()).device
    hook_point = hook_point or infer_hook_point_from_sae(sae)
    
    # Use dictionaries for sparse accumulation
    interaction_sum = {}  # (q_feat, k_feat) -> sum of strengths
    interaction_count = {}  # (q_feat, k_feat) -> count
    
    num_samples = len(dataset_texts)
    
    if verbose:
        print(f"Computing average over {num_samples} samples...")
        iterator = tqdm(range(num_samples), desc="Processing samples")
    else:
        iterator = range(num_samples)
    
    for sample_idx in iterator:
        text = dataset_texts[sample_idx]
        
        # Get FRA for this sample
        fra_result = get_sentence_fra_batch(
            model, sae, text, layer, head,
            max_length=128, top_k=20, verbose=False,
            hook_point=hook_point,
        )
        
        if fra_result is None:
            continue
        
        # Process the sparse FRA tensor
        fra_sparse = fra_result['fra_tensor_sparse']
        indices = fra_sparse.indices()
        values = fra_sparse.values()
        
        # Accumulate interactions
        for i in range(indices.shape[1]):
            q_pos, k_pos, q_feat, k_feat = indices[:, i].tolist()
            strength = values[i].item()
            
            # Skip self-interactions if requested
            if filter_self_interactions and q_feat == k_feat:
                continue
            
            # Update running statistics
            pair_key = (q_feat, k_feat)
            interaction_sum[pair_key] = interaction_sum.get(pair_key, 0.0) + abs(strength)
            interaction_count[pair_key] = interaction_count.get(pair_key, 0) + 1
        
        # CRITICAL: Clear tensors from GPU memory
        del fra_sparse
        del fra_result
        del indices
        del values
        if device == "cuda":
            torch.cuda.empty_cache()
    
    if verbose:
        print(f"\n✅ Processed {num_samples} samples")
        print(f"   Found {len(interaction_sum)} unique feature pairs")
        total_interactions = sum(interaction_count.values())
        print(f"   Total interactions: {total_interactions:,}")
        
        # Compute and show top averages
        if interaction_sum:
            avg_list = []
            for (q_feat, k_feat), sum_val in interaction_sum.items():
                count = interaction_count[(q_feat, k_feat)]
                avg = sum_val / count
                avg_list.append(((q_feat, k_feat), avg, count))
            
            avg_list.sort(key=lambda x: x[1], reverse=True)
            
            print(f"\n🔥 Top 5 average interactions:")
            for i, ((q_feat, k_feat), avg, count) in enumerate(avg_list[:5]):
                print(f"   {i+1}. Features ({q_feat}, {k_feat}): {avg:.4f} (seen {count} times)")
    
    return {
        'interaction_sum': interaction_sum,
        'interaction_count': interaction_count,
        'num_samples': num_samples,
        'd_sae': d_sae
    }


if __name__ == "__main__":
    # Test with 5 samples
    import time
    
    torch.set_grad_enabled(False)
    
    print("Testing dataset average computation...")
    print("="*60)
    
    # Load model and SAE
    print("\nLoading model and SAE...")
    t0 = time.time()
    model, sae, config = load_model_and_sae_from_config()
    layer = int(config["sae"]["layer"])
    hook_point = config["sae"].get("hook_point") or infer_hook_point_from_sae(sae)
    print(f"Loading time: {time.time()-t0:.1f}s")
    
    # Test texts
    test_texts = [
        "The cat sat on the mat. The cat was happy.",
        "Alice went to the store. She bought milk.",
        "The student who studied hard passed. The students who studied hard passed.",
        "Bob likes to play chess. Bob is very good at chess.",
        "The weather today is sunny. The weather yesterday was rainy."
    ]
    
    # Compute averages
    print(f"\nComputing averages over {len(test_texts)} samples...")
    t0 = time.time()
    
    results = compute_dataset_average(
        model=model,
        sae=sae,
        dataset_texts=test_texts,
        layer=layer,
        head=0,
        filter_self_interactions=True,
        hook_point=hook_point,
        verbose=True
    )
    
    t1 = time.time()
    
    print(f"\n⏱️ Total time: {t1-t0:.1f}s")
    print(f"⏱️ Per sample: {(t1-t0)/len(test_texts):.1f}s")
    print(f"\n✅ This should be reasonable (few seconds per sample)")