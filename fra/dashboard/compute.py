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


# ---------------------------------------------------------------------------
# Packing helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def _pack_fra_result(model, layer, head, fra_result, tokens, device):
    """Convert raw FRA output into a numpy-serialisable dict.

    Extracts the standard attention pattern / scores via ``run_with_cache``,
    decodes token strings, and packs everything into the dict format that
    every downstream tab expects.

    .. note::

       Prefer :func:`_pack_fra_from_cache` in multi-head workflows to avoid
       redundant forward passes.
    """
    tok_tensor = torch.tensor(tokens[:128]).unsqueeze(0).to(device)
    attn_pattern_hook = f"blocks.{layer}.attn.hook_pattern"
    attn_scores_hook = f"blocks.{layer}.attn.hook_attn_scores"
    _, attn_cache = model.run_with_cache(
        tok_tensor, names_filter=[attn_pattern_hook, attn_scores_hook],
    )
    attn_pattern = attn_cache[attn_pattern_hook][0, head].cpu().float().numpy()
    attn_scores = attn_cache[attn_scores_hook][0, head].cpu().float().numpy()

    token_strs = [model.tokenizer.decode([t]) for t in tokens[:128]]
    softcap = getattr(model.cfg, "attn_scores_soft_cap", 0.0) or 0.0

    sparse = fra_result["fra_tensor_sparse"]
    return {
        "indices_np": sparse.indices().cpu().numpy(),
        "values_np": sparse.values().cpu().float().numpy(),
        "shape": fra_result["shape"],
        "seq_len": fra_result["seq_len"],
        "total_interactions": fra_result["total_interactions"],
        "feat_acts_np": fra_result["feature_activations"].cpu().float().numpy(),
        "topk_acts_np": fra_result["topk_features"].cpu().float().numpy(),
        "attn_pattern_np": attn_pattern,
        "attn_scores_np": attn_scores,
        "token_strs": token_strs,
        "tokens": tokens[:fra_result["seq_len"]],
        "softcap": softcap,
    }


def _pack_fra_from_cache(fra_result, attn_pattern_np, attn_scores_np,
                         token_strs, tokens, softcap):
    """Pack an FRA result dict using pre-computed attention arrays.

    Same output format as :func:`_pack_fra_result` but avoids a model
    forward pass by accepting already-extracted attention data.
    """
    sparse = fra_result["fra_tensor_sparse"]
    return {
        "indices_np": sparse.indices().cpu().numpy(),
        "values_np": sparse.values().cpu().float().numpy(),
        "shape": fra_result["shape"],
        "seq_len": fra_result["seq_len"],
        "total_interactions": fra_result["total_interactions"],
        "feat_acts_np": fra_result["feature_activations"].cpu().float().numpy(),
        "topk_acts_np": fra_result["topk_features"].cpu().float().numpy(),
        "attn_pattern_np": attn_pattern_np,
        "attn_scores_np": attn_scores_np,
        "token_strs": token_strs,
        "tokens": tokens[:fra_result["seq_len"]],
        "softcap": softcap,
    }


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
    from fra.core.fra import compute_fra

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

    tokens = model.tokenizer.encode(
        text, add_special_tokens=include_special_tokens,
    )[:128]

    with torch.no_grad():
        fra_result = compute_fra(
            model, sae, tokens,
            layer=layer, head=head,
            top_k=top_k_features,
            hook_point=hook_point,
            chunk_size=chunk_size,
        )

    return _pack_fra_result(model, layer, head, fra_result, tokens, device)


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
    from fra.core.fra import compute_fra_pair

    base_model, it_model = load_model_pair(
        base_model_name, it_model_name, device, it_arch_name,
    )
    crosscoder = load_crosscoder(crosscoder_repo_id, model_idx, device, subfolder)
    target_model = base_model if model_idx == 0 else it_model
    layer = crosscoder_layer + 1

    with torch.no_grad():
        fra_result = compute_fra_pair(
            base_model, it_model, crosscoder, tokens,
            head=head,
            coder_layer=crosscoder_layer,
            max_length=128, top_k=top_k_features,
            verbose=True,
        )

    return _pack_fra_result(target_model, layer, head, fra_result, tokens, device)


def _load_di_weights(sae_type, head, device, **kw):
    """Load (W_dec, W_Q, W_K, attn_layer, rope_params) for DI computation."""
    from fra.core.helpers import get_W_K, _extract_rope_params

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
    rope_params = _extract_rope_params(model, attn_layer)
    return W_dec, W_Q, W_K, attn_layer, rope_params


# ---------------------------------------------------------------------------
# Efficient encode-once / build-per-head helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def encode_fra(
    text: str,
    layer: int,
    hook_point: str,
    sae_type: str,
    sae_hub_release: str,
    sae_hub_id: str,
    sae_local_path: str,
    device: str,
    model_name: str = "gpt2-small",
    hf_token: str = "",
    include_special_tokens: bool = True,
):
    """Head-independent encode: single combined forward pass + SAE encode.

    Returns ``(encoded, attn_cache, model, tokens)`` where *encoded* is the
    dict from ``_encode_sae`` and *attn_cache* holds pre-computed attention
    pattern / scores tensors for **all** heads.
    """
    from fra.core.fra import _encode_sae

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

    tokens = model.tokenizer.encode(
        text, add_special_tokens=include_special_tokens,
    )[:128]
    tok_tensor = torch.tensor(tokens).unsqueeze(0).to(device)

    # Single forward pass capturing activations AND attention hooks
    act_hook = f"blocks.{layer}.{hook_point}"
    pattern_hook = f"blocks.{layer}.attn.hook_pattern"
    scores_hook = f"blocks.{layer}.attn.hook_attn_scores"
    _, cache = model.run_with_cache(
        tok_tensor,
        names_filter=[act_hook, pattern_hook, scores_hook],
    )

    activation = cache[act_hook].squeeze(0)
    encoded = _encode_sae(
        model, sae, tok_tensor, layer, hook_point,
        activation=activation,
    )

    attn_cache = {
        "pattern": cache[pattern_hook],   # [1, n_heads, seq, seq]
        "scores": cache[scores_hook],     # [1, n_heads, seq, seq]
    }
    return encoded, attn_cache, model, tokens


@torch.no_grad()
def encode_fra_crosscoder(
    tokens: list,
    crosscoder_layer: int,
    crosscoder_repo_id: str,
    model_idx: int,
    base_model_name: str,
    it_model_name: str,
    device: str,
    subfolder: str = "",
    it_arch_name: str = "",
):
    """Head-independent encode: crosscoder path with combined forward pass.

    The target model's forward pass captures both the residual-stream
    activations (at ``crosscoder_layer``) and the attention pattern / scores
    (at ``crosscoder_layer + 1``) in a single ``run_with_cache`` call.

    Returns ``(encoded, attn_cache, target_model, tokens, attn_layer)``.
    """
    from fra.core.fra import _encode_crosscoder

    base_model, it_model = load_model_pair(
        base_model_name, it_model_name, device, it_arch_name,
    )
    crosscoder = load_crosscoder(crosscoder_repo_id, model_idx, device, subfolder)
    target_model = base_model if model_idx == 0 else it_model
    other_model = it_model if model_idx == 0 else base_model
    attn_layer = crosscoder_layer + 1

    tokens = list(tokens[:128])
    tok_tensor = torch.tensor(tokens).unsqueeze(0).to(device)

    # Target model: combined forward pass (residual + attention hooks)
    resid_hook = f"blocks.{crosscoder_layer}.hook_resid_post"
    pattern_hook = f"blocks.{attn_layer}.attn.hook_pattern"
    scores_hook = f"blocks.{attn_layer}.attn.hook_attn_scores"
    _, target_cache = target_model.run_with_cache(
        tok_tensor,
        names_filter=[resid_hook, pattern_hook, scores_hook],
    )
    target_activation = target_cache[resid_hook].squeeze(0)

    # Other model: only needs residual activation
    _, other_cache = other_model.run_with_cache(
        tok_tensor, names_filter=[resid_hook],
    )
    other_activation = other_cache[resid_hook].squeeze(0)

    encoded = _encode_crosscoder(
        base_model, it_model, crosscoder, tok_tensor, crosscoder_layer,
        target_activation=target_activation,
        other_activation=other_activation,
    )

    attn_cache = {
        "pattern": target_cache[pattern_hook],
        "scores": target_cache[scores_hook],
    }
    return encoded, attn_cache, target_model, tokens, attn_layer


@torch.no_grad()
def build_fra_head(encoded, attn_cache, model, layer, head, tokens, device,
                   top_k=None, chunk_size=16, topk_features=None, rms=None,
                   token_strs=None, softcap=None):
    """Build FRA result for one head from pre-encoded data.

    Uses pre-computed *attn_cache* instead of running a forward pass, and
    accepts optional pre-computed *topk_features* / *rms* to avoid
    redundant work across heads.

    Returns the same numpy-serialisable dict as :func:`run_fra`.
    """
    from fra.core.fra import _build_fra_result

    fra_result = _build_fra_result(
        model, layer, head,
        encoded["feature_activations"], encoded["W_dec"], device,
        top_k=top_k,
        topk_features=topk_features,
        rms_activations=encoded["rms_activations"],
        rms=rms,
        dec_norms=encoded["dec_norms"],
        chunk_size=chunk_size,
    )

    attn_pattern_np = attn_cache["pattern"][0, head].cpu().float().numpy()
    attn_scores_np = attn_cache["scores"][0, head].cpu().float().numpy()

    if token_strs is None:
        token_strs = [model.tokenizer.decode([t]) for t in tokens[:128]]
    if softcap is None:
        softcap = getattr(model.cfg, "attn_scores_soft_cap", 0.0) or 0.0

    return _pack_fra_from_cache(
        fra_result, attn_pattern_np, attn_scores_np,
        token_strs, tokens, softcap,
    )


# ---------------------------------------------------------------------------
# Multi-layer crosscoder path
# ---------------------------------------------------------------------------


@torch.no_grad()
def encode_fra_multilayer(
    tokens: list,
    attn_layers: list[int],
    coder,
    model,
    device: str,
):
    """Head-independent encode for a multi-layer crosscoder.

    Runs a single forward pass caching all hookpoints the coder needs plus
    attention patterns/scores for every requested attention layer.  Encodes
    once through the crosscoder to get shared feature activations.

    Returns ``(encoded, attn_caches, model, tokens)`` where *attn_caches*
    maps each attention layer to its ``{pattern, scores}`` tensors.
    """
    tokens = list(tokens[:128])
    tok_tensor = torch.tensor(tokens).unsqueeze(0).to(device)

    # Hooks: all coder hookpoints + attn pattern/scores per requested layer
    hooks_to_cache = list(coder.hookpoints)
    for layer in attn_layers:
        hooks_to_cache.append(f"blocks.{layer}.attn.hook_pattern")
        hooks_to_cache.append(f"blocks.{layer}.attn.hook_attn_scores")

    _, cache = model.run_with_cache(tok_tensor, names_filter=hooks_to_cache)

    # Stack hookpoint activations for the crosscoder encoder
    # Shape: [seq, n_hookpoints, d_model] then add model dim
    act_list = [cache[hp].squeeze(0) for hp in coder.hookpoints]
    stacked = torch.stack(act_list, dim=1)            # [seq, n_hookpoints, d_model]
    stacked = stacked.unsqueeze(1)                     # [seq, 1, n_hookpoints, d_model]

    feature_activations = coder.encode(stacked)        # [seq, d_sae]

    # Per-layer attention caches
    attn_caches = {}
    for layer in attn_layers:
        attn_caches[layer] = {
            "pattern": cache[f"blocks.{layer}.attn.hook_pattern"],
            "scores": cache[f"blocks.{layer}.attn.hook_attn_scores"],
        }

    # Build encoded dict per attention layer (different W_dec slice each)
    encoded_per_layer = {}
    for layer in attn_layers:
        W_dec = coder.decoder_for_attn_layer(layer)
        b_dec = coder.b_dec.float().to(device)
        rms_activations = feature_activations.float() @ W_dec.float() + b_dec
        encoded_per_layer[layer] = {
            "feature_activations": feature_activations,
            "W_dec": W_dec,
            "dec_norms": coder.dec_norms,
            "rms_activations": rms_activations,
        }

    return encoded_per_layer, attn_caches, model, tokens


@torch.no_grad()
def build_fra_head_multilayer(
    encoded, attn_cache, model, layer, head, tokens, device,
    top_k=None, chunk_size=16, topk_features=None, rms=None,
    token_strs=None, softcap=None,
):
    """Build FRA for one head at one layer from multi-layer encoded data.

    Same return format as :func:`build_fra_head`.
    """
    from fra.core.fra import _build_fra_result

    fra_result = _build_fra_result(
        model, layer, head,
        encoded["feature_activations"], encoded["W_dec"], device,
        top_k=top_k,
        topk_features=topk_features,
        rms_activations=encoded["rms_activations"],
        rms=rms,
        dec_norms=encoded["dec_norms"],
        chunk_size=chunk_size,
    )

    attn_pattern_np = attn_cache["pattern"][0, head].cpu().float().numpy()
    attn_scores_np = attn_cache["scores"][0, head].cpu().float().numpy()

    if token_strs is None:
        token_strs = [model.tokenizer.decode([t]) for t in tokens[:128]]
    if softcap is None:
        softcap = getattr(model.cfg, "attn_scores_soft_cap", 0.0) or 0.0

    return _pack_fra_from_cache(
        fra_result, attn_pattern_np, attn_scores_np,
        token_strs, tokens, softcap,
    )
