"""Legacy GPT-2 hook_z FRA regression test outside the conformance gate."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fra.fra_func import get_sentence_fra_batch


def fra_sum_to_attn(sparse_tensor: torch.Tensor, seq_len: int) -> np.ndarray:
    """Sum a sparse FRA tensor over the feature dimensions."""
    indices = sparse_tensor.indices().cpu().numpy()
    values = sparse_tensor.values().cpu().numpy()
    out = np.zeros((seq_len, seq_len), dtype=np.float64)
    np.add.at(out, (indices[0], indices[1]), values)
    return out


class SyntheticFeatureWrapper:
    """Keep a real decoder but override encode() with synthetic features."""

    def __init__(self, base_sae, synthetic_features: torch.Tensor):
        self._base_sae = base_sae
        self._features = synthetic_features
        self.W_dec = base_sae.W_dec
        self.b_dec = base_sae.b_dec
        self.d_sae = base_sae.d_sae

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        expected = self.decode(self._features)
        if not torch.allclose(x, expected, rtol=1e-5, atol=1e-5):
            raise AssertionError("Injected GPT-2 activations do not match synthetic U @ W_dec + b_dec")
        return self._features.clone()

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return self._base_sae.decode(features)


class CachedActivationModel:
    """Wrap a real model but override run_with_cache for one hook."""

    def __init__(self, base_model, hook_name: str, activations: torch.Tensor):
        self._base_model = base_model
        self._hook_name = hook_name
        self._activations = activations
        self.blocks = base_model.blocks
        self.tokenizer = base_model.tokenizer
        self.cfg = base_model.cfg

    def parameters(self):
        return self._base_model.parameters()

    def run_with_cache(self, tokens: torch.Tensor, names_filter):
        if isinstance(names_filter, str):
            names = [names_filter]
        else:
            names = list(names_filter)

        if tokens.shape[1] != self._activations.shape[0]:
            raise AssertionError("Injected activation length does not match tokenized prompt length")

        cache = {}
        for name in names:
            if name != self._hook_name:
                raise KeyError(f"Unsupported hook requested in cached model wrapper: {name}")
            cache[name] = self._activations.unsqueeze(0)
        return None, cache


def _get_qk_weights(model, layer: int, head: int) -> tuple[torch.Tensor, torch.Tensor]:
    W_Q = model.blocks[layer].attn.W_Q[head]
    n_kv = model.blocks[layer].attn.W_K.shape[0]
    n_q = model.blocks[layer].attn.W_Q.shape[0]
    kv_head = head * n_kv // n_q
    W_K = model.blocks[layer].attn.W_K[kv_head]
    return W_Q, W_K


def _get_biases(model, layer: int, head: int) -> tuple[torch.Tensor, torch.Tensor]:
    b_Q = model.blocks[layer].attn.b_Q[head]
    n_kv = model.blocks[layer].attn.b_K.shape[0]
    n_q = model.blocks[layer].attn.b_Q.shape[0]
    kv_head = head * n_kv // n_q
    b_K = model.blocks[layer].attn.b_K[kv_head]
    return b_Q, b_K


def _masked_scores(scores: np.ndarray) -> np.ndarray:
    masked = scores.copy()
    masked[np.triu_indices_from(masked, k=1)] = -np.inf
    return masked


def _softmax_rows(masked_scores: np.ndarray) -> np.ndarray:
    return torch.softmax(torch.tensor(masked_scores, dtype=torch.float64), dim=-1).cpu().numpy()


def test_gpt2_hook_z_regression_with_synthetic_features():
    transformer_lens = pytest.importorskip("transformer_lens")
    pytest.importorskip("sae_lens")

    from fra.sae_lens_wrapper import SAELensAttentionSAE

    HookedTransformer = transformer_lens.HookedTransformer

    layer = 5
    head = 1
    device = "cpu"
    text = "The cat sat on the mat."
    hook_point = "attn.hook_z"

    try:
        base_model = HookedTransformer.from_pretrained("gpt2-small", device=device)
        base_sae = SAELensAttentionSAE("gpt2-small-hook-z-kk", f"blocks.{layer}.hook_z", device=device)
    except Exception as exc:
        pytest.skip(f"GPT-2 / SAE weights unavailable locally: {exc}")

    tokens = base_model.tokenizer.encode(text, add_special_tokens=False)
    seq_len = len(tokens)
    d_sae = base_sae.d_sae

    feature_ids = [0, 17, 1234]
    gen = torch.Generator().manual_seed(0)
    feature_values = torch.randn(
        seq_len,
        len(feature_ids),
        generator=gen,
        dtype=base_sae.W_dec.dtype,
        device=base_sae.W_dec.device,
    ) * 0.5

    features = torch.zeros(seq_len, d_sae, dtype=base_sae.W_dec.dtype, device=base_sae.W_dec.device)
    features[:, feature_ids] = feature_values

    x_flat = base_sae.decode(features)
    x_nobias = x_flat - base_sae.b_dec

    n_heads = base_model.cfg.n_heads
    d_head = base_model.blocks[layer].attn.W_Q.shape[-1]
    if x_flat.shape[-1] != n_heads * d_head:
        pytest.skip(
            f"Decoder dimension {x_flat.shape[-1]} does not match hook_z size {n_heads * d_head}"
        )
    x_hook = x_flat.reshape(seq_len, n_heads, d_head)

    wrapped_model = CachedActivationModel(
        base_model,
        hook_name=f"blocks.{layer}.{hook_point}",
        activations=x_hook,
    )
    wrapped_sae = SyntheticFeatureWrapper(base_sae, features)

    fra_result = get_sentence_fra_batch(
        model=wrapped_model,
        sae=wrapped_sae,
        text=text,
        layer=layer,
        head=head,
        max_length=128,
        top_k=len(feature_ids),
        verbose=False,
        hook_point=hook_point,
        chunk_size=16,
        normalize_by_decoder_norm=False,
        prepend_bos=False,
    )
    fra_sum = fra_sum_to_attn(fra_result["fra_tensor_sparse"], seq_len)

    W_Q, W_K = _get_qk_weights(base_model, layer, head)
    b_Q, b_K = _get_biases(base_model, layer, head)

    q_nobias = x_nobias @ W_Q
    k_nobias = x_nobias @ W_K
    raw_qk = np.tril((q_nobias @ k_nobias.T).detach().cpu().numpy())

    assert np.allclose(fra_sum, raw_qk, rtol=1e-5, atol=1e-5)

    combined_q_bias = base_sae.b_dec @ W_Q + b_Q
    combined_k_bias = base_sae.b_dec @ W_K + b_K
    term_q = (q_nobias @ combined_k_bias).detach().cpu().numpy()
    term_k = (k_nobias @ combined_q_bias).detach().cpu().numpy()
    term_const = float(torch.dot(combined_q_bias, combined_k_bias).item())

    reconstructed_scores = (
        fra_sum + term_q[:, None] + term_k[None, :] + term_const
    ) / base_model.blocks[layer].attn.attn_scale
    reconstructed_scores = _masked_scores(reconstructed_scores)

    q_full = x_flat @ W_Q + b_Q
    k_full = x_flat @ W_K + b_K
    actual_scores = (
        (q_full @ k_full.T) / base_model.blocks[layer].attn.attn_scale
    ).detach().cpu().numpy()
    actual_scores = _masked_scores(actual_scores)

    finite_mask = np.isfinite(actual_scores)
    assert np.array_equal(np.isneginf(reconstructed_scores), np.isneginf(actual_scores))
    assert np.allclose(
        reconstructed_scores[finite_mask],
        actual_scores[finite_mask],
        rtol=1e-5,
        atol=1e-5,
    )

    reconstructed_pattern = _softmax_rows(reconstructed_scores)
    actual_pattern = _softmax_rows(actual_scores)
    assert np.allclose(
        reconstructed_pattern,
        actual_pattern,
        rtol=1e-6,
        atol=1e-6,
    )
