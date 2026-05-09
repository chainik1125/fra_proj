"""
Gemma FRA exactness test with synthetic features.

Uses the real Gemma-2-2b model and real Gemma-Scope SAE decoder.
Solves for synthetic feature activations u such that u @ W_dec = x exactly,
then runs the production FRA pipeline (with RoPE + RMSNorm + attn_scale)
and verifies that summing over the feature dimensions recovers the actual
attention scores.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from fra.core.fra import get_sentence_fra_batch, compute_fra_sparse, topk_sparsify
from fra.core.helpers import (
    get_W_K,
    get_attn_scale,
    project_qk,
    fra_sum_to_attn,
    _extract_rope_params,
    apply_rope_to_projected,
)


# ---------------------------------------------------------------------------
# Mock SAE that returns precomputed exact features
# ---------------------------------------------------------------------------

class MockGemmaSAE:
    """Wraps a real Gemma-Scope decoder with synthetic exact features."""

    def __init__(self, W_dec, features, b_dec):
        self.W_dec = W_dec
        self.d_sae = W_dec.shape[0]
        self.b_dec = b_dec
        self._features = features

    def encode(self, x):
        return self._features


# ---------------------------------------------------------------------------
# Solve for exact features
# ---------------------------------------------------------------------------

def solve_exact_features(x, W_dec, b_dec):
    """Solve for u such that u @ W_dec + b_dec == x exactly.

    Uses pivoted QR to select the best-conditioned d_model decoder
    directions, then solves the square system in float64.
    """
    from scipy.linalg import qr as scipy_qr

    d_model = x.shape[1]
    d_sae = W_dec.shape[0]

    _, _, perm = scipy_qr(W_dec.detach().float().numpy().T, pivoting=True)
    selected = perm[:d_model]

    target = (x.float() - b_dec.float()).double()
    D_sub = W_dec[selected].float().double()

    u_sub = torch.linalg.solve(D_sub.T, target.T).T

    features = torch.zeros(x.shape[0], d_sae, device=x.device, dtype=x.dtype)
    features[:, selected] = u_sub.to(x.dtype)
    return features


# ---------------------------------------------------------------------------
# Fixture: real Gemma model + real SAE decoder + synthetic features
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gemma_case():
    """Load real Gemma-2-2b, real Gemma-Scope SAE, solve for exact features."""
    from transformer_lens import HookedTransformer
    from sae_lens import SAE

    layer = 12
    head = 0
    text = (
        "The transformer architecture revolutionized natural language processing "
        "by replacing recurrence with self-attention mechanisms."
    )
    hook_point = "hook_resid_pre"

    # Load real model
    model = HookedTransformer.from_pretrained(
        "gemma-2-2b", device="cpu", dtype=torch.float32,
    )

    # Load real Gemma-Scope SAE
    result = SAE.from_pretrained(
        release="gemma-scope-2b-pt-res",
        sae_id=f"layer_{layer}/width_16k/average_l0_82",
        device="cpu",
    )
    sae_raw = result[0] if isinstance(result, tuple) else result

    W_dec = sae_raw.W_dec.detach().float()
    b_dec = torch.zeros(W_dec.shape[1])  # zero bias for clean test

    # Get real activations
    tokens = model.tokenizer.encode(text, add_special_tokens=False)
    tokens_tensor = torch.tensor(tokens).unsqueeze(0)
    hook_name = f"blocks.{layer}.{hook_point}"
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens_tensor, names_filter=[hook_name])
    x = cache[hook_name].squeeze(0).float()

    # Solve for exact features
    features = solve_exact_features(x, W_dec, b_dec)
    d_model = x.shape[1]

    mock_sae = MockGemmaSAE(W_dec, features, b_dec)

    return {
        "model": model,
        "sae": mock_sae,
        "text": text,
        "layer": layer,
        "head": head,
        "hook_point": hook_point,
        "top_k": d_model,
        "x": x,
        "features": features,
        "W_dec": W_dec,
        "b_dec": b_dec,
        "seq_len": x.shape[0],
        "d_model": d_model,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGemmaFRAExactness:
    """End-to-end Gemma FRA test with real model, real decoder, synthetic features."""

    def test_reconstruction_close(self, gemma_case):
        """u @ W_dec + b_dec ~= x."""
        c = gemma_case
        recon = c["features"].float() @ c["W_dec"].float() + c["b_dec"].float()
        err = (recon - c["x"].float()).abs().max().item()
        assert err < 1e-2, f"Reconstruction error: {err:.2e}"

    def test_fra_sum_matches_attention(self, gemma_case):
        """sum_{i,j} FRA[q,k,i,j] matches the RoPE+RMSNorm attention scores."""
        c = gemma_case

        # Run the production FRA pipeline
        fra_result = get_sentence_fra_batch(
            model=c["model"], sae=c["sae"], text=c["text"],
            layer=c["layer"], head=c["head"],
            max_length=128, top_k=c["top_k"], verbose=True,
            hook_point=c["hook_point"], chunk_size=8,
            normalize_by_decoder_norm=False, prepend_bos=False,
        )
        fra_sum = fra_sum_to_attn(fra_result["fra_tensor_sparse"], c["seq_len"])
        fra_sum_causal = np.tril(fra_sum)

        # Ground truth: use project_qk which handles RoPE + RMSNorm
        x_hat = c["features"].float() @ c["W_dec"].float() + c["b_dec"].float()
        q_full, k_full, q_nobias, k_nobias = project_qk(
            c["model"], c["layer"], c["head"],
            x_hat, c["b_dec"], needs_rms=True,
        )
        attn_scale = get_attn_scale(c["model"], c["layer"])
        gt = np.tril(
            (q_nobias @ k_nobias.T / attn_scale).detach().cpu().numpy()
        )

        # Report errors before asserting
        max_err = np.max(np.abs(fra_sum_causal - gt))
        gt_scale = np.max(np.abs(gt))
        print(f"Max abs error: {max_err:.4e}")
        print(f"GT scale: {gt_scale:.4e}")
        print(f"Relative: {max_err / (gt_scale + 1e-10):.4e}")

        assert np.allclose(fra_sum_causal, gt, rtol=5e-2, atol=5e-2), (
            f"FRA sum mismatch: max err = {max_err:.2e}, "
            f"relative = {max_err / (gt_scale + 1e-10):.2e}"
        )

    def test_fra_sum_multiple_heads(self, gemma_case):
        """Verify across multiple heads to catch GQA indexing bugs."""
        c = gemma_case

        x_hat = c["features"].float() @ c["W_dec"].float() + c["b_dec"].float()
        attn_scale = get_attn_scale(c["model"], c["layer"])

        for head in [0, 3]:  # head 0 and 3 map to different KV heads in GQA
            fra_result = get_sentence_fra_batch(
                model=c["model"], sae=c["sae"], text=c["text"],
                layer=c["layer"], head=head,
                max_length=128, top_k=c["top_k"], verbose=False,
                hook_point=c["hook_point"], chunk_size=8,
                normalize_by_decoder_norm=False, prepend_bos=False,
            )
            fra_sum = np.tril(fra_sum_to_attn(
                fra_result["fra_tensor_sparse"], c["seq_len"]
            ))

            _, _, q_nobias, k_nobias = project_qk(
                c["model"], c["layer"], head,
                x_hat, c["b_dec"], needs_rms=True,
            )
            gt = np.tril(
                (q_nobias @ k_nobias.T / attn_scale).detach().cpu().numpy()
            )

            assert np.allclose(fra_sum, gt, rtol=5e-2, atol=5e-2), (
                f"FRA mismatch head {head}: "
                f"max err = {np.max(np.abs(fra_sum - gt)):.2e}"
            )

    def test_causal_mask(self, gemma_case):
        """Upper triangle of FRA sum should be zero."""
        c = gemma_case
        fra_result = get_sentence_fra_batch(
            model=c["model"], sae=c["sae"], text=c["text"],
            layer=c["layer"], head=c["head"],
            max_length=128, top_k=c["top_k"], verbose=False,
            hook_point=c["hook_point"], chunk_size=8,
            normalize_by_decoder_norm=False, prepend_bos=False,
        )
        fra_sum = fra_sum_to_attn(fra_result["fra_tensor_sparse"], c["seq_len"])
        upper = np.triu(fra_sum, k=1)
        assert np.allclose(upper, 0, atol=1e-10)
