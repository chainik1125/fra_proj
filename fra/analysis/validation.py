"""
FRA Reconstruction Validation — library functions.

Three control experiments to validate FRA reconstruction quality:

  (a) Attention map reconstruction — FRA sum over features vs actual QK scores
      Sub-comparisons isolate error sources:
        a1. FRA sum vs actual raw QK    (total error: SAE + top-k + b_dec)
        a2. SAE recon QK vs actual QK   (SAE-only error)
        a3. FRA sum vs SAE QK no-bias   (top-k truncation error only)

  (b) SAE residual stream reconstruction — decode(encode(x)) vs x

  (c) Loss recovery — patch one head's pre-softmax attention scores with
      FRA-reconstructed scores, compare patched vs unpatched vs zero-ablated loss.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from fra.core.fra import get_sentence_fra_batch
from fra.core.helpers import (
    compute_errors,
    fra_max_to_attn,
    fra_mean_to_attn,
    fra_sum_to_attn,
    get_qk_weights,
    print_errors,
)

# ── Default test texts ────────────────────────────────────────────────────

DEFAULT_TEXTS = [
    "The cat sat on the mat. The cat was happy. A dog lay on the rug. The dog was tired.",
    "When John and Mary went to the store, John gave a drink to Mary.",
    "The president of the United States gave a speech about the economy and foreign policy.",
]

# Longer, more diverse texts for Gemma — better coverage of training distribution.
# Short simple sentences tend to have atypical residual stream norms at deeper layers,
# pushing SAE L0 far from its training-time average.
GEMMA_TEXTS = [
    (
        "In a recent study published in Nature, researchers found that the rate of"
        " ice loss in Antarctica has accelerated significantly over the past decade."
        " The findings suggest that sea level rise could exceed earlier projections"
        " by as much as thirty percent. Scientists warn that without immediate action"
        " to reduce greenhouse gas emissions, coastal cities around the world face"
        " unprecedented flooding risks within the next fifty years."
    ),
    (
        "The quick brown fox jumps over the lazy dog. Mary had a little lamb whose"
        " fleece was white as snow. Every day the farmer walked to the market to sell"
        " his vegetables and buy supplies for the week ahead. The children played in"
        " the park while their parents sat on benches reading newspapers and talking"
        " about the weather and local politics."
    ),
    (
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    a, b = 0, 1\n"
        "    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\n"
        "# The Fibonacci sequence appears throughout nature, from the spiral arrangement"
        " of leaves on a stem to the breeding patterns of rabbits. Leonardo of Pisa,"
        " known as Fibonacci, introduced these numbers to Western mathematics in 1202."
    ),
    (
        "The transformer architecture, introduced in the paper Attention Is All You Need,"
        " revolutionized natural language processing by replacing recurrent layers with"
        " self-attention mechanisms. Each attention head computes query, key, and value"
        " projections from the input embeddings, then uses scaled dot-product attention"
        " to produce a weighted combination of values. This allows the model to attend"
        " to different positions in the input sequence simultaneously."
    ),
    (
        "Tokyo is the capital of Japan and one of the most populous metropolitan areas"
        " in the world. The city blends ultramodern architecture with traditional temples"
        " and gardens. Its efficient public transportation system moves millions of"
        " commuters daily. From the bustling streets of Shibuya to the serene grounds"
        " of the Imperial Palace, Tokyo offers a fascinating contrast between the old"
        " and the new."
    ),
]


# ── Model / SAE loading ──────────────────────────────────────────────────


def load_sae(sae_type: str, layer: int, device: str,
             release: str = "", sae_id: str = ""):
    """Load SAE wrapper of the specified type."""
    if sae_type == "hub":
        from fra.coders.sae_lens import SAELensAttentionSAE
        return SAELensAttentionSAE("gpt2-small-hook-z-kk",
                                   f"blocks.{layer}.hook_z", device=device)
    elif sae_type == "gemma":
        from fra.coders.sae_lens import GemmaScopeSAE
        rel = release or "gemma-scope-2b-pt-res"
        sid = sae_id or f"layer_{layer}/width_16k/average_l0_82"
        return GemmaScopeSAE(rel, sid, device=device)
    else:
        from fra.coders.sae_lens import LocalLn1SAE
        ckpt = str(Path(__file__).parent.parent.parent / "checkpoints" / "q9sczrvl" / "50003968")
        return LocalLn1SAE(ckpt, layer=layer, device=device)


# ── Test (a): Attention map reconstruction ────────────────────────────────


@torch.no_grad()
def test_attention_reconstruction(model, sae, text, layer, head,
                                  hook_point, top_k, chunk_size):
    """
    Compare FRA-reconstructed attention to actual pre-softmax QK scores.

    Three sub-comparisons:
      a1. FRA sum vs actual raw QK (end-to-end)
      a2. SAE full-recon QK vs actual raw QK (SAE approximation error)
      a3. FRA sum vs SAE QK without b_dec (top-k truncation error)
    """
    device = next(model.parameters()).device
    tokens = model.tokenizer.encode(text)[:128]
    tok_tensor = torch.tensor(tokens).unsqueeze(0).to(device)
    seq_len = len(tokens)

    # ── Activations ──
    hook_name = f"blocks.{layer}.{hook_point}"
    _, cache = model.run_with_cache(tok_tensor, names_filter=[hook_name])
    x = cache[hook_name].squeeze(0)          # [seq, d_model]
    if x.dim() == 3:
        x = x.flatten(-2, -1)

    W_Q, W_K, _, _ = get_qk_weights(model, layer, head)
    attn_scale = model.blocks[layer].attn.attn_scale

    # ── 1. Actual QK (ground truth, scaled by 1/sqrt(d_head)) ──
    # FRA tensors now include this scaling, so ground truth must match.
    actual_qk = (((x @ W_Q) @ (x @ W_K).T) / attn_scale).cpu().numpy()

    # ── 2. SAE full-reconstruction QK (includes b_dec, scaled) ──
    features = sae.encode(x)
    x_hat = sae.decode(features)               # includes b_dec
    sae_qk = (((x_hat @ W_Q) @ (x_hat @ W_K).T) / attn_scale).cpu().numpy()

    # ── 3. SAE QK without b_dec (what FRA decomposes into, scaled) ──
    b_dec = sae.b_dec if hasattr(sae, "b_dec") else sae.sae.b_dec
    x_hat_nobias = x_hat - b_dec
    sae_qk_nobias = (((x_hat_nobias @ W_Q) @ (x_hat_nobias @ W_K).T) / attn_scale).cpu().numpy()

    # ── 4. FRA computation (normalized = auto-detect) ──
    fra_result = get_sentence_fra_batch(
        model, sae, text, layer, head,
        max_length=128, top_k=top_k, hook_point=hook_point,
        chunk_size=chunk_size, verbose=False,
        normalize_by_decoder_norm=None,  # auto-detect
    )
    sparse = fra_result["fra_tensor_sparse"]
    fra_sum_corr = fra_sum_to_attn(sparse, seq_len)

    # Max and mean aggregation modes
    fra_max = fra_max_to_attn(sparse, seq_len)
    fra_mean = fra_mean_to_attn(sparse, seq_len)

    # ── Compare only lower triangle (causal region) ──
    causal = np.tril(np.ones((seq_len, seq_len)))
    actual_qk *= causal
    sae_qk *= causal
    sae_qk_nobias *= causal
    fra_sum_corr *= causal
    fra_max *= causal
    fra_mean *= causal

    results = {
        "a1_fra_vs_actual": compute_errors(actual_qk, fra_sum_corr),
        "a2_sae_vs_actual": compute_errors(actual_qk, sae_qk),
        "a3_fra_vs_sae_nobias": compute_errors(sae_qk_nobias, fra_sum_corr),
        # Max/mean vs actual QK (only sum is mathematically correct; max/mean are diagnostic)
        "fra_max_vs_actual": compute_errors(actual_qk, fra_max),
        "fra_mean_vs_actual": compute_errors(actual_qk, fra_mean),
        "normalized": fra_result.get("normalized", False),
        "seq_len": seq_len,
        "nnz": fra_result["total_interactions"],
    }
    return results, fra_result


# ── Test (b): SAE residual stream reconstruction ─────────────────────────


@torch.no_grad()
def test_sae_reconstruction(model, sae, text, layer, hook_point):
    """Measure SAE encode->decode quality on the residual stream."""
    device = next(model.parameters()).device
    tokens = model.tokenizer.encode(text)[:128]
    tok_tensor = torch.tensor(tokens).unsqueeze(0).to(device)

    hook_name = f"blocks.{layer}.{hook_point}"
    _, cache = model.run_with_cache(tok_tensor, names_filter=[hook_name])
    x = cache[hook_name].squeeze(0)
    if x.dim() == 3:
        x = x.flatten(-2, -1)

    features = sae.encode(x)
    x_hat = sae.decode(features)

    x_np = x.cpu().numpy()
    x_hat_np = x_hat.cpu().numpy()

    # Per-token L0 (active features)
    per_token_l0 = (features != 0).sum(dim=-1).float()
    n_active = per_token_l0.mean().item()
    sparsity = (features == 0).float().mean().item()

    # Per-token residual stream norms (for distribution diagnostics)
    per_token_norms = x.norm(dim=-1)

    # Diagnostic norms
    x_norm = float(np.linalg.norm(x_np))
    x_hat_norm = float(np.linalg.norm(x_hat_np))
    diff_norm = float(np.linalg.norm(x_np - x_hat_np))

    return {
        "recon": compute_errors(x_np, x_hat_np),
        "avg_active_features": n_active,
        "l0_min": float(per_token_l0.min().item()),
        "l0_max": float(per_token_l0.max().item()),
        "l0_std": float(per_token_l0.std().item()),
        "sparsity": sparsity,
        "d_sae": features.shape[-1],
        "x_norm": x_norm,
        "x_hat_norm": x_hat_norm,
        "diff_norm": diff_norm,
        "token_norm_mean": float(per_token_norms.mean().item()),
        "token_norm_min": float(per_token_norms.min().item()),
        "token_norm_max": float(per_token_norms.max().item()),
        "seq_len": len(tokens),
    }


# ── Test (c): Loss recovery ──────────────────────────────────────────────


@torch.no_grad()
def test_loss_recovery(model, sae, text, layer, head,
                       hook_point, top_k, chunk_size):
    """
    Patch a single head's pre-softmax attention scores and measure loss.

    Returns losses for: unpatched, FRA-patched, SAE-patched, zero-ablated,
    plus recovery ratios.

    NOTE: TransformerLens applies the causal mask BEFORE hook_attn_scores,
    so when we overwrite the scores we must re-apply the causal mask ourselves.
    """
    device = next(model.parameters()).device
    tokens = model.tokenizer.encode(text)[:128]
    tok_tensor = torch.tensor(tokens).unsqueeze(0).to(device)
    seq_len = len(tokens)

    if seq_len < 3:
        return None

    shift_labels = tok_tensor[0, 1:]  # next-token targets

    # ── 1. Unpatched loss ──
    logits_clean = model(tok_tensor)
    unpatched_loss = F.cross_entropy(logits_clean[0, :-1], shift_labels).item()

    # ── 2. Compute reconstructions ──
    hook_name = f"blocks.{layer}.{hook_point}"
    _, cache = model.run_with_cache(tok_tensor, names_filter=[hook_name])
    x = cache[hook_name].squeeze(0)
    if x.dim() == 3:
        x = x.flatten(-2, -1)

    W_Q, W_K, b_Q, b_K = get_qk_weights(model, layer, head)
    attn_scale = model.blocks[layer].attn.attn_scale

    # SAE reconstruction
    features = sae.encode(x)
    x_hat = sae.decode(features)  # includes b_dec

    # FRA (normalized = auto-detect)
    fra_result = get_sentence_fra_batch(
        model, sae, text, layer, head,
        max_length=128, top_k=top_k, hook_point=hook_point,
        chunk_size=chunk_size, verbose=False,
        normalize_by_decoder_norm=None,  # auto-detect
    )
    sparse = fra_result["fra_tensor_sparse"]
    fra_sum = fra_sum_to_attn(sparse, seq_len)

    # ── Build FRA full scores (with bias corrections) ──
    # FRA_sum ≈ x_hat_nobias @ W_Q @ (x_hat_nobias @ W_K)^T
    # Full score = (x_hat_nobias + combined_bias_q) . (x_hat_nobias + combined_bias_k) / scale
    # where combined biases absorb b_dec, b_Q, b_K
    b_dec = sae.b_dec if hasattr(sae, "b_dec") else sae.sae.b_dec

    x_hat_nobias = (x_hat - b_dec)  # [seq, d_model]
    q_nobias = (x_hat_nobias @ W_Q).cpu().numpy()   # [seq, d_head]
    k_nobias = (x_hat_nobias @ W_K).cpu().numpy()

    # Combined bias: b_dec contribution + attention bias
    combined_q_bias = (b_dec @ W_Q + b_Q).cpu().numpy()  # [d_head]
    combined_k_bias = (b_dec @ W_K + b_K).cpu().numpy()

    # Full score[q,k] = (q_nobias[q] + cqb) . (k_nobias[k] + ckb) / scale
    # = q_nobias[q].k_nobias[k] + q_nobias[q].ckb + cqb.k_nobias[k] + cqb.ckb
    # FRA_sum ≈ first term; add remaining three:
    term_q = q_nobias @ combined_k_bias       # [seq], per-query
    term_k = k_nobias @ combined_q_bias       # [seq], per-key
    term_const = np.dot(combined_q_bias, combined_k_bias)  # scalar

    # fra_sum already includes 1/sqrt(d_head) from the FRA computation;
    # only the bias correction terms are unscaled dot products.
    fra_full = fra_sum + (
        term_q[:, None] + term_k[None, :] + term_const
    ) / attn_scale

    # Apply causal mask (upper triangle -> -inf)
    causal_mask = np.triu(np.full((seq_len, seq_len), float("-inf")), k=1)
    fra_full += causal_mask

    fra_scores_t = torch.tensor(
        fra_full, dtype=torch.float32, device=device
    )

    # ── SAE full scores (direct recomputation, no top-k) ──
    q_full = x_hat @ W_Q + b_Q   # [seq, d_head]
    k_full = x_hat @ W_K + b_K
    sae_full = (q_full @ k_full.T) / attn_scale
    # Apply causal mask
    mask_t = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1
    )
    sae_scores_t = sae_full + mask_t

    # ── Hook name for patching ──
    score_hook = f"blocks.{layer}.attn.hook_attn_scores"

    # ── 3. FRA-patched ──
    def fra_hook(attn_scores, hook):
        attn_scores[0, head, :seq_len, :seq_len] = fra_scores_t
        return attn_scores

    fra_logits = model.run_with_hooks(
        tok_tensor, fwd_hooks=[(score_hook, fra_hook)]
    )
    fra_loss = F.cross_entropy(fra_logits[0, :-1], shift_labels).item()

    # ── 4. SAE-patched (no top-k truncation) ──
    def sae_hook(attn_scores, hook):
        attn_scores[0, head, :seq_len, :seq_len] = sae_scores_t
        return attn_scores

    sae_logits = model.run_with_hooks(
        tok_tensor, fwd_hooks=[(score_hook, sae_hook)]
    )
    sae_loss = F.cross_entropy(sae_logits[0, :-1], shift_labels).item()

    # ── 5. Zero-ablation (uniform attention after softmax) ──
    def zero_hook(attn_scores, hook):
        # Set to 0 -> softmax gives ~uniform over causal positions
        # The causal mask was already applied by TL before our hook,
        # so positions in the upper triangle are already -inf. Setting
        # the lower triangle to 0 gives uniform within causal window.
        attn_scores[0, head, :seq_len, :seq_len] = (
            torch.zeros((seq_len, seq_len), device=device)
            + mask_t
        )
        return attn_scores

    zero_logits = model.run_with_hooks(
        tok_tensor, fwd_hooks=[(score_hook, zero_hook)]
    )
    zero_loss = F.cross_entropy(zero_logits[0, :-1], shift_labels).item()

    # ── Recovery ratios ──
    # head_contribution = how much zeroing the head hurts (positive = head helps)
    head_contribution = zero_loss - unpatched_loss

    return {
        "unpatched_loss": unpatched_loss,
        "fra_patched_loss": fra_loss,
        "sae_patched_loss": sae_loss,
        "zero_ablation_loss": zero_loss,
        "head_contribution": head_contribution,
        # 1 - patched/unpatched (user-requested metric)
        "fra_loss_ratio": 1 - fra_loss / unpatched_loss,
        "sae_loss_ratio": 1 - sae_loss / unpatched_loss,
        # Recovery: fraction of head contribution preserved (1 = perfect, 0 = zero-ablation)
        "fra_recovery": (
            (zero_loss - fra_loss) / (head_contribution + 1e-10)
            if head_contribution > 0.01 else float("nan")
        ),
        "sae_recovery": (
            (zero_loss - sae_loss) / (head_contribution + 1e-10)
            if head_contribution > 0.01 else float("nan")
        ),
    }


# ── Averaging across texts ────────────────────────────────────────────────


def avg_dicts(dicts, keys):
    """Average numeric values across a list of dicts."""
    result = {}
    for k in keys:
        vals = [d[k] for d in dicts if d is not None and k in d and not np.isnan(d[k])]
        result[k] = float(np.mean(vals)) if vals else float("nan")
    return result


def avg_error_dicts(dicts):
    """Average error metric dicts."""
    keys = ["fro_rel_err", "cosine_sim", "r_squared", "mean_rel_err", "median_rel_err", "mean_abs_err"]
    return avg_dicts(dicts, keys)


# ── Crosscoder validation ─────────────────────────────────────────────────


def test_crosscoder_attention_reconstruction(
    target_model, base_model, it_model, crosscoder,
    tokens, layer, head, crosscoder_layer, top_k=20,
):
    """Test (a) for crosscoders: FRA sum vs actual pre-softmax scores."""
    from fra.core.fra_crosscoder import get_sentence_fra_crosscoder

    device = next(target_model.parameters()).device
    tokens = tokens[:128]
    seq_len = len(tokens)

    fra_result = get_sentence_fra_crosscoder(
        base_model, it_model, crosscoder, tokens,
        head=head, crosscoder_layer=crosscoder_layer,
        max_length=128, top_k=top_k, verbose=False,
    )
    sparse = fra_result["fra_tensor_sparse"]
    fra_sum = fra_sum_to_attn(sparse, seq_len)

    # Ground truth: actual pre-softmax scores from target model
    tok_tensor = torch.tensor(tokens).unsqueeze(0).to(device)
    attn_scores_hook = f"blocks.{layer}.attn.hook_attn_scores"
    _, cache = target_model.run_with_cache(
        tok_tensor, names_filter=[attn_scores_hook],
    )
    actual_scores = cache[attn_scores_hook][0, head].cpu().numpy()  # [seq, seq]

    # Compare only causal region
    causal = np.tril(np.ones((seq_len, seq_len)))
    fra_sum *= causal
    actual_scores_causal = actual_scores[:seq_len, :seq_len] * causal

    return {
        "fra_vs_actual": compute_errors(actual_scores_causal, fra_sum),
        "seq_len": seq_len,
        "nnz": fra_result["total_interactions"],
    }


def test_crosscoder_reconstruction(
    base_model, it_model, crosscoder, tokens, crosscoder_layer,
):
    """Test (b) for crosscoders: encode-decode quality."""
    device = next(base_model.parameters()).device
    tokens = tokens[:128]
    tok_tensor = torch.tensor(tokens).unsqueeze(0).to(device)

    hook_name = f"blocks.{crosscoder_layer}.hook_resid_post"
    _, base_cache = base_model.run_with_cache(
        tok_tensor, names_filter=[hook_name],
    )
    _, it_cache = it_model.run_with_cache(
        tok_tensor, names_filter=[hook_name],
    )
    base_act = base_cache[hook_name].squeeze(0)
    it_act = it_cache[hook_name].squeeze(0)

    # Stack in crosscoder order
    if crosscoder.model_idx == 0:
        x_stacked = torch.stack([base_act, it_act], dim=1)
    else:
        x_stacked = torch.stack([it_act, base_act], dim=1)

    target_act = base_act if crosscoder.model_idx == 0 else it_act

    # Encode and decode
    features = crosscoder.encode(x_stacked)
    x_hat_stacked = crosscoder.decode(features)
    target_hat = x_hat_stacked[:, crosscoder.model_idx]  # [seq, d_model]

    x_np = target_act.cpu().float().numpy()
    xhat_np = target_hat.cpu().float().numpy()

    active_per_token = (features != 0).sum(dim=-1).float()

    return {
        "recon": compute_errors(x_np, xhat_np),
        "d_sae": features.shape[-1],
        "avg_active_features": float(active_per_token.mean()),
        "sparsity": float((features == 0).float().mean()),
        "l0_min": float(active_per_token.min()),
        "l0_max": float(active_per_token.max()),
        "l0_std": float(active_per_token.std()),
        "seq_len": len(tokens),
        "x_norm": float(np.linalg.norm(x_np)),
        "x_hat_norm": float(np.linalg.norm(xhat_np)),
        "diff_norm": float(np.linalg.norm(x_np - xhat_np)),
        "token_norm_mean": float(np.linalg.norm(x_np, axis=-1).mean()),
        "token_norm_min": float(np.linalg.norm(x_np, axis=-1).min()),
        "token_norm_max": float(np.linalg.norm(x_np, axis=-1).max()),
    }


def test_crosscoder_loss_recovery(
    target_model, base_model, it_model, crosscoder,
    tokens, layer, head, crosscoder_layer, top_k=20,
):
    """Test (c) for crosscoders: loss recovery via attention patching."""
    from fra.core.fra_crosscoder import get_sentence_fra_crosscoder

    device = next(target_model.parameters()).device
    tokens = tokens[:128]
    seq_len = len(tokens)
    if seq_len < 3:
        return None

    tok_tensor = torch.tensor(tokens).unsqueeze(0).to(device)
    shift_labels = tok_tensor[0, 1:]

    # Unpatched loss
    logits_clean = target_model(tok_tensor)
    unpatched_loss = F.cross_entropy(logits_clean[0, :-1], shift_labels).item()

    # FRA computation
    fra_result = get_sentence_fra_crosscoder(
        base_model, it_model, crosscoder, tokens,
        head=head, crosscoder_layer=crosscoder_layer,
        max_length=128, top_k=top_k, verbose=False,
    )
    sparse = fra_result["fra_tensor_sparse"]
    fra_sum = fra_sum_to_attn(sparse, seq_len)

    # For crosscoders: no bias correction needed (b_Q=0, b_K=0 for Gemma/Llama,
    # and FRA already includes 1/sqrt(d_head) + RMSNorm correction)
    causal_mask = np.triu(np.full((seq_len, seq_len), float("-inf")), k=1)
    fra_scores = fra_sum + causal_mask
    fra_scores_t = torch.tensor(fra_scores, dtype=torch.float32, device=device)

    # Zero-ablated: uniform attention
    zero_scores = torch.zeros((seq_len, seq_len), device=device)
    zero_scores += torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1,
    )

    # Patch and measure
    hook_name = f"blocks.{layer}.attn.hook_attn_scores"

    def _patch(scores_tensor):
        def hook(value, hook):
            value[0, head] = scores_tensor
            return value
        with target_model.hooks([(hook_name, hook)]):
            logits = target_model(tok_tensor)
        return F.cross_entropy(logits[0, :-1], shift_labels).item()

    fra_loss = _patch(fra_scores_t)
    zero_loss = _patch(zero_scores)

    head_contribution = zero_loss - unpatched_loss
    fra_recovery = (
        (zero_loss - fra_loss) / head_contribution
        if abs(head_contribution) > 1e-6 else float("nan")
    )

    return {
        "unpatched_loss": unpatched_loss,
        "fra_patched_loss": fra_loss,
        "zero_ablation_loss": zero_loss,
        "head_contribution": head_contribution,
        "fra_recovery": fra_recovery,
        "fra_loss_ratio": 1 - fra_loss / unpatched_loss if unpatched_loss > 0 else 0,
    }


def run_crosscoder_validation(args):
    """Run all three validation tests for a crosscoder model pair."""
    from fra.coders.crosscoder import GemmaCrosscoderFRA
    from transformer_lens import HookedTransformer

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Parse crosscoder type
    is_gemma_cc = args.model == "crosscoder-gemma"

    if is_gemma_cc:
        base_name = "google/gemma-2-2b"
        it_name = "google/gemma-2-2b-it"
        it_arch_name = ""
        repo_id = "science-of-finetuning/gemma-2-2b-L13-k100-lr1e-04-local-shuffling-CCLoss"
        cc_layer = args.layer if args.layer is not None else 13
        subfolder = ""
        model_idx = args.model_idx
        n_heads = 8
        texts = args.text and [args.text] or GEMMA_TEXTS
    else:  # crosscoder-llama
        base_name = "meta-llama/Llama-3.1-8B"
        it_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
        it_arch_name = "meta-llama/Llama-3.1-8B"
        repo_id = "mitroitskii/Crosscoder-Llama-3.1-8B-vs-Llama-R1-Distill-8B"
        cc_layer = args.layer if args.layer is not None else 15
        layer_map = {7: "BatchTopK-Crosscoder/L7R", 15: "BatchTopK-Crosscoder/L15R",
                     23: "BatchTopK-Crosscoder/L23R"}
        subfolder = layer_map.get(cc_layer, f"BatchTopK-Crosscoder/L{cc_layer}R")
        model_idx = args.model_idx
        n_heads = 32
        texts = args.text and [args.text] or GEMMA_TEXTS

    layer = cc_layer + 1
    head = min(args.head, n_heads - 1)
    top_k = args.top_k
    model_label = "base" if model_idx == 0 else "instruct/reasoning"

    print("=" * 65)
    print("  FRA Crosscoder Validation")
    print("=" * 65)
    print(f"  Base model    : {base_name}")
    print(f"  Target model  : {it_name}")
    print(f"  Analysing     : model {model_idx} ({model_label})")
    print(f"  Crosscoder    : {repo_id}")
    print(f"  CC layer      : {cc_layer} -> attention L{layer}")
    print(f"  Head          : {head}")
    print(f"  Top-K         : {top_k}")
    print(f"  Device        : {device}")
    print(f"  Texts         : {len(texts)}")
    print("=" * 65)

    # Load models
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_grad_enabled(False)
    print("\nLoading base model...", end=" ", flush=True)
    base_model = HookedTransformer.from_pretrained(
        base_name, device=device, dtype=torch.float16,
    )
    print("done.")

    print("Loading target model...", end=" ", flush=True)
    if it_arch_name:
        hf_model = AutoModelForCausalLM.from_pretrained(it_name, dtype=torch.float16)
        hf_tokenizer = AutoTokenizer.from_pretrained(it_name)
        it_model = HookedTransformer.from_pretrained(
            it_arch_name, device=device, dtype=torch.float16,
            hf_model=hf_model, tokenizer=hf_tokenizer,
        )
        del hf_model
    else:
        it_model = HookedTransformer.from_pretrained(
            it_name, device=device, dtype=torch.float16,
        )
    print("done.")

    print("Loading crosscoder...", end=" ", flush=True)
    if subfolder:
        crosscoder = GemmaCrosscoderFRA.from_cc_weights(
            repo_id, subfolder, model_idx=model_idx, device=device,
        )
    else:
        crosscoder = GemmaCrosscoderFRA.from_pretrained(
            repo_id, model_idx=model_idx, device=device,
        )
    print(f"done. (d_sae={crosscoder.W_dec.shape[0]})")

    target_model = base_model if model_idx == 0 else it_model

    # Tokenize texts
    all_a, all_b, all_c = [], [], []
    for i, text in enumerate(texts):
        short = text[:60] + "..." if len(text) > 60 else text
        print(f"\n--- Text {i+1}/{len(texts)}: \"{short}\"")

        tokens = target_model.tokenizer.encode(text)[:128]

        print("  Running test (a): attention reconstruction...", flush=True)
        a_result = test_crosscoder_attention_reconstruction(
            target_model, base_model, it_model, crosscoder,
            tokens, layer, head, cc_layer, top_k,
        )
        all_a.append(a_result)
        print(f"    seq_len={a_result['seq_len']}, nnz={a_result['nnz']:,}")

        print("  Running test (b): crosscoder reconstruction...", flush=True)
        b_result = test_crosscoder_reconstruction(
            base_model, it_model, crosscoder, tokens, cc_layer,
        )
        all_b.append(b_result)
        print(f"    L0={b_result['avg_active_features']:.0f}/{b_result['d_sae']}, "
              f"seq_len={b_result['seq_len']}")

        print("  Running test (c): loss recovery...", flush=True)
        c_result = test_crosscoder_loss_recovery(
            target_model, base_model, it_model, crosscoder,
            tokens, layer, head, cc_layer, top_k,
        )
        all_c.append(c_result)
        if c_result:
            print(f"    unpatched={c_result['unpatched_loss']:.4f}, "
                  f"fra={c_result['fra_patched_loss']:.4f}, "
                  f"zero={c_result['zero_ablation_loss']:.4f}")

    # Print results
    print("\n" + "=" * 65)
    print("  RESULTS (averaged over {} text{})".format(
        len(texts), "s" if len(texts) > 1 else ""))
    print("=" * 65)

    # Test (a)
    print("\nTest (a): Attention Map Reconstruction (crosscoder)")
    print("-" * 50)
    a_avg = avg_error_dicts([r["fra_vs_actual"] for r in all_a])
    print_errors("FRA sum vs actual pre-softmax scores", a_avg)

    # Test (b)
    print("\nTest (b): Crosscoder Reconstruction")
    print("-" * 50)
    b_avg = avg_error_dicts([r["recon"] for r in all_b])
    print_errors("decode(encode(x)) vs x", b_avg)
    avg_active = np.mean([r["avg_active_features"] for r in all_b])
    avg_sparsity = np.mean([r["sparsity"] for r in all_b])
    print(f"  Avg active features : {avg_active:.1f}")
    print(f"  Sparsity            : {avg_sparsity:.4f}")

    # Test (c)
    valid_c = [r for r in all_c if r is not None]
    print(f"\nTest (c): Loss Recovery (L{layer} H{head})")
    print("-" * 50)
    if valid_c:
        c_avg = avg_dicts(valid_c, [
            "unpatched_loss", "fra_patched_loss",
            "zero_ablation_loss", "head_contribution",
            "fra_recovery", "fra_loss_ratio",
        ])
        print(f"  Unpatched loss     : {c_avg['unpatched_loss']:.4f}")
        print(f"  FRA-patched loss   : {c_avg['fra_patched_loss']:.4f}")
        print(f"  Zero-ablated loss  : {c_avg['zero_ablation_loss']:.4f}")
        print(f"  FRA recovery       : {c_avg['fra_recovery']:.4f}")
    else:
        print("  No valid results (texts too short?)")

    # Summary
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    attn_err = a_avg["fro_rel_err"]
    resid_err = b_avg["fro_rel_err"]
    fra_rec = c_avg.get("fra_recovery", float("nan")) if valid_c else float("nan")
    print(f"  (a) Attention error  : {attn_err:.1%}  (target <50%)  "
          f"[{'PASS' if attn_err < 0.50 else 'FAIL'}]")
    print(f"  (b) Residual error   : {resid_err:.1%}  (target <20%)  "
          f"[{'PASS' if resid_err < 0.20 else 'FAIL'}]")
    if not np.isnan(fra_rec):
        print(f"  (c) Loss recovery    : {fra_rec:.2f}    (target >0.70) "
              f"[{'PASS' if fra_rec > 0.70 else 'FAIL'}]")
    print("=" * 65)
