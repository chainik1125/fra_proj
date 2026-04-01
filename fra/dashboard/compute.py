"""Computation helper functions for the FRA Dashboard."""

import torch

from fra.dashboard.loaders import (
    load_crosscoder,
    load_model,
    load_model_gemma,
    load_model_pair,
    load_sae_gemma,
    load_sae_hub,
    load_sae_local,
)


def run_fra(
    text: str,
    layer: int,
    head: int,
    hook_point: str,
    sae_type: str,
    sae_hub_release: str,
    sae_hub_id: str,
    sae_local_path: str,
    top_k_features: int | None,
    device: str,
    model_name: str = "gpt2-small",
    chunk_size: int = 16,
    hf_token: str = "",
    include_special_tokens: bool = True,
) -> dict:
    """Compute FRA and return numpy-serialisable result dict."""
    from fra.core.fra import get_sentence_fra_batch

    if sae_type == "gemma":
        model = load_model_gemma(model_name, device, hf_token)
    else:
        model = load_model(model_name, device, hf_token)

    if sae_type == "hub":
        sae = load_sae_hub(sae_hub_release, sae_hub_id, device)
    elif sae_type == "gemma":
        sae = load_sae_gemma(sae_hub_release, sae_hub_id, device)
    else:
        sae = load_sae_local(sae_local_path, layer, device)

    with torch.no_grad():
        fra_result = get_sentence_fra_batch(
            model, sae, text,
            layer=layer, head=head,
            max_length=128, top_k=top_k_features,
            hook_point=hook_point,
            chunk_size=chunk_size,
            prepend_bos=include_special_tokens,
        )

        # Standard attention pattern + pre-softmax scores for comparison
        tokens = model.tokenizer.encode(
            text, add_special_tokens=include_special_tokens,
        )[:128]
        tok_tensor = torch.tensor(tokens).unsqueeze(0).to(device)
        attn_pattern_hook = f"blocks.{layer}.attn.hook_pattern"
        attn_scores_hook = f"blocks.{layer}.attn.hook_attn_scores"
        _, attn_cache = model.run_with_cache(
            tok_tensor, names_filter=[attn_pattern_hook, attn_scores_hook]
        )
        attn_pattern = attn_cache[attn_pattern_hook][0, head].cpu().float().numpy()  # [S, S]
        attn_scores = attn_cache[attn_scores_hook][0, head].cpu().float().numpy()    # [S, S]

        token_strs = [model.tokenizer.decode([t]) for t in tokens]

    sparse = fra_result["fra_tensor_sparse"]
    return {
        "indices_np": sparse.indices().cpu().numpy(),   # [4, nnz]
        "values_np": sparse.values().cpu().float().numpy(),     # [nnz]
        "shape": fra_result["shape"],
        "seq_len": fra_result["seq_len"],
        "total_interactions": fra_result["total_interactions"],
        "feat_acts_np": fra_result["feature_activations"].cpu().float().numpy(),  # [seq_len, d_sae], raw
        "topk_acts_np": fra_result["topk_features"].cpu().float().numpy(),       # [seq_len, d_sae], top-k filtered
        "attn_pattern_np": attn_pattern,                # [seq_len, seq_len]
        "attn_scores_np": attn_scores,                  # [seq_len, seq_len]
        "token_strs": token_strs,
        "tokens": tokens[:fra_result["seq_len"]],
    }


def run_fra_crosscoder(
    tokens: list,
    head: int,
    crosscoder_layer: int,
    crosscoder_repo_id: str,
    model_idx: int,
    base_model_name: str,
    it_model_name: str,
    top_k_features: int | None,
    device: str,
    subfolder: str = "",
    it_arch_name: str = "",
) -> dict:
    """Compute FRA with a model-diffing crosscoder, same return format as run_fra."""
    from fra.core.fra import get_sentence_fra_crosscoder

    base_model, it_model = load_model_pair(
        base_model_name, it_model_name, device, it_arch_name,
    )
    crosscoder = load_crosscoder(crosscoder_repo_id, model_idx, device, subfolder)
    target_model = base_model if model_idx == 0 else it_model
    layer = crosscoder_layer + 1

    with torch.no_grad():
        fra_result = get_sentence_fra_crosscoder(
            base_model, it_model, crosscoder, tokens,
            head=head,
            crosscoder_layer=crosscoder_layer,
            max_length=128, top_k=top_k_features,
            verbose=True,
        )

        feat_acts = fra_result["feature_activations"]

        # Standard attention pattern + pre-softmax scores for comparison
        tok_tensor = torch.tensor(tokens[:128]).unsqueeze(0).to(device)
        attn_pattern_hook = f"blocks.{layer}.attn.hook_pattern"
        attn_scores_hook = f"blocks.{layer}.attn.hook_attn_scores"
        _, attn_cache = target_model.run_with_cache(
            tok_tensor, names_filter=[attn_pattern_hook, attn_scores_hook],
        )
        attn_pattern = attn_cache[attn_pattern_hook][0, head].cpu().float().numpy()
        attn_scores = attn_cache[attn_scores_hook][0, head].cpu().float().numpy()

        token_strs = [target_model.tokenizer.decode([t]) for t in tokens[:128]]

    sparse = fra_result["fra_tensor_sparse"]
    return {
        "indices_np": sparse.indices().cpu().numpy(),
        "values_np": sparse.values().cpu().float().numpy(),
        "shape": fra_result["shape"],
        "seq_len": fra_result["seq_len"],
        "total_interactions": fra_result["total_interactions"],
        "feat_acts_np": feat_acts.cpu().float().numpy(),         # [seq_len, d_sae], raw
        "topk_acts_np": fra_result["topk_features"].cpu().float().numpy(),  # [seq_len, d_sae], top-k filtered
        "attn_pattern_np": attn_pattern,
        "attn_scores_np": attn_scores,
        "token_strs": token_strs,
        "tokens": tokens[:fra_result["seq_len"]],  # actual token IDs used for FRA computation
    }


def _load_di_weights(sae_type, head, device, **kw):
    """Load (W_dec, W_Q, W_K, attn_layer) for DI computation."""
    from fra.core.helpers import get_W_K

    if sae_type == "crosscoder":
        base, it = load_model_pair(
            kw["base_model_name"], kw["it_model_name"], device, kw["cc_it_arch"],
        )
        model = base if kw["model_idx"] == 0 else it
        cc = load_crosscoder(
            kw["crosscoder_repo_id"], kw["model_idx"], device, kw["cc_subfolder"],
        )
        W_dec = cc.W_dec
        attn_layer = int(kw["crosscoder_layer"]) + 1
    elif sae_type == "sae_gemma":
        model = load_model_gemma(
            kw.get("model_name", "gemma-2-2b"), device, kw.get("hf_token", ""),
        )
        sae_obj = load_sae_gemma(kw["sae_hub_release"], kw["sae_hub_id"], device)
        W_dec = sae_obj.W_dec
        attn_layer = int(kw["layer"])
    else:
        _model_name = kw.get("model_name", "gpt2-small")
        model = load_model(_model_name, device)
        if sae_type in ("hub", "sae_hub"):
            sae_obj = load_sae_hub(kw["sae_hub_release"], kw["sae_hub_id"], device)
        else:
            sae_obj = load_sae_local(kw["sae_local_path"], int(kw["layer"]), device)
        W_dec = sae_obj.W_dec
        attn_layer = int(kw["layer"])

    W_Q = model.blocks[attn_layer].attn.W_Q[head]
    W_K = get_W_K(model, attn_layer, head)
    return W_dec, W_Q, W_K, attn_layer
