"""
FRA Off-Diagonal Ablation Study — library functions.

Ablate the strongest cross-feature (off-diagonal, i != j) interactions in
FRA and measure impact on model output.

Conditions:
  1. unpatched        — normal model forward pass (ground truth)
  2. fra_full         — patch with full FRA reconstruction (FRA baseline)
  3. offdiag_top_K    — ablate top-K off-diagonal (i!=j) feature pairs
  4. random_K         — ablate K random off-diagonal pairs (control)
  5. ondiag_top_K     — ablate top-K on-diagonal (i==j) pairs (control)
  6. zero             — uniform attention (worst-case baseline)

Metrics per condition:
  - Cross-entropy loss
  - KL divergence from unpatched
  - Top-1 prediction change fraction
  - Loss recovery ratio
"""

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from fra.core.fra import get_sentence_fra_batch
from fra.core.helpers import (
    compute_bias_correction,
    fra_sum_to_attn,
    project_qk,
    rank_pairs,
)

from fra.coders import load_sae

# ── Texts ─────────────────────────────────────────────────────────────────

ABLATION_TEXTS = [
    "The cat sat on the mat. The cat was happy. A dog lay on the rug. The dog was tired.",
    "When John and Mary went to the store, John gave a drink to Mary.",
    "The president of the United States gave a speech about the economy and foreign policy.",
    (
        "In a recent study published in Nature, researchers found that the rate of"
        " ice loss in Antarctica has accelerated significantly over the past decade."
        " The findings suggest that sea level rise could exceed earlier projections."
    ),
    (
        "The quick brown fox jumps over the lazy dog. Mary had a little lamb whose"
        " fleece was white as snow. Every day the farmer walked to the market to sell"
        " his vegetables and buy supplies for the week ahead."
    ),
    (
        "The transformer architecture revolutionized natural language processing by"
        " replacing recurrent layers with self-attention mechanisms. Each attention"
        " head computes query, key, and value projections from the input embeddings."
    ),
    (
        "Tokyo is the capital of Japan and one of the most populous metropolitan areas"
        " in the world. The city blends ultramodern architecture with traditional temples."
    ),
    (
        "Alice was beginning to get very tired of sitting by her sister on the bank,"
        " and of having nothing to do: once or twice she had peeped into the book her"
        " sister was reading, but it had no pictures or conversations in it."
    ),
]


# ── Sparse tensor ablation ────────────────────────────────────────────────


def ablate_fra_pairs(fra_sparse, pairs_to_ablate, d_sae):
    """
    Remove specific (q_feat, k_feat) pairs from the sparse FRA tensor.

    Args:
        fra_sparse: 4D sparse COO [seq, seq, d_sae, d_sae]
        pairs_to_ablate: list of (q_feat, k_feat) tuples
        d_sae: SAE hidden dimension (for hash encoding)

    Returns:
        New sparse tensor with those pairs removed.
    """
    if not pairs_to_ablate:
        return fra_sparse

    indices = fra_sparse.indices()  # [4, nnz] on CPU
    values = fra_sparse.values()

    # Hash-encode pairs for fast lookup
    q_feats = indices[2].long()
    k_feats = indices[3].long()
    pair_keys = q_feats * d_sae + k_feats

    ablate_keys = torch.tensor(
        [q * d_sae + k for q, k in pairs_to_ablate],
        dtype=torch.long, device=pair_keys.device,
    )
    keep_mask = ~torch.isin(pair_keys, ablate_keys)

    new_indices = indices[:, keep_mask]
    new_values = values[keep_mask]

    return torch.sparse_coo_tensor(
        new_indices, new_values, size=fra_sparse.shape
    ).coalesce()


# ── Bias corrections & score reconstruction ───────────────────────────────


@torch.no_grad()
def compute_bias_corrections(model, sae, text, layer, head, hook_point, max_length=128):
    """
    Compute the bias correction matrix needed to go from FRA sum to full
    pre-softmax attention scores.

    Returns dict with: bias_correction, seq_len, tok_tensor,
                       shift_labels, unpatched_loss, unpatched_logits.
    """
    device = next(model.parameters()).device
    tokens = model.tokenizer.encode(text)[:max_length]
    tok_tensor = torch.tensor(tokens).unsqueeze(0).to(device)
    seq_len = len(tokens)

    if seq_len < 3:
        return None

    shift_labels = tok_tensor[0, 1:]

    # Unpatched forward pass (ground truth)
    logits_clean = model(tok_tensor)
    unpatched_loss = F.cross_entropy(logits_clean[0, :-1], shift_labels).item()

    # Get activations for SAE
    hook_name = f"blocks.{layer}.{hook_point}"
    _, cache = model.run_with_cache(tok_tensor, names_filter=[hook_name])
    x = cache[hook_name].squeeze(0)
    if x.dim() == 3:
        x = x.flatten(-2, -1)

    # SAE reconstruction
    features = sae.encode(x)
    x_hat = sae.decode(features).float()
    b_dec = sae.b_dec if hasattr(sae, "b_dec") else sae.sae.b_dec

    attn_scale = model.blocks[layer].attn.attn_scale
    q_full, k_full, q_nobias, k_nobias = project_qk(
        model, layer, head, x_hat, b_dec,
        needs_rms="resid" in hook_point,
    )
    bias_corr = compute_bias_correction(
        q_full, k_full, q_nobias, k_nobias, attn_scale,
    ).cpu().numpy()

    return {
        "bias_correction": bias_corr,
        "seq_len": seq_len,
        "tok_tensor": tok_tensor,
        "shift_labels": shift_labels,
        "unpatched_loss": unpatched_loss,
        "unpatched_logits": logits_clean,
    }


def reconstruct_scores(fra_sum_2d, bias, device, actual_bos_scores=None,
                       softcap: float = 0.0):
    """
    Build full pre-softmax attention scores from FRA sum + bias corrections.

    Args:
        fra_sum_2d: [seq, seq] numpy array (FRA collapsed over feature dims)
        bias: dict from compute_bias_corrections
        device: torch device
        actual_bos_scores: Optional [seq, seq] numpy array of actual attention
            scores.  When provided, row 0 and column 0 (BOS positions) are
            copied from the actual scores into the reconstruction.  Use this
            when the SAE was not trained on BOS activations.
        softcap: Attention logit soft-cap value (e.g. 50.0 for Gemma-2).
            When > 0, applies ``softcap * tanh(scores / softcap)`` before
            the causal mask, matching the model's own attention implementation.

    Returns:
        [seq, seq] torch tensor ready to patch into hook_attn_scores.
    """
    seq_len = bias["seq_len"]
    # fra_sum_2d already includes 1/sqrt(d_head) scaling from the FRA
    # computation.  bias_correction is also pre-scaled.
    scores = fra_sum_2d + bias["bias_correction"][:seq_len, :seq_len]

    # Logit soft-capping (Gemma-2): applied before causal mask
    if softcap > 0:
        scores = softcap * np.tanh(scores / softcap)

    # Causal mask
    causal = np.triu(np.full((seq_len, seq_len), float("-inf")), k=1)
    scores += causal

    scores_t = torch.tensor(scores, dtype=torch.float32, device=device)

    if actual_bos_scores is not None:
        actual_t = torch.tensor(
            actual_bos_scores[:seq_len, :seq_len],
            dtype=torch.float32, device=device,
        )
        scores_t[0, :] = actual_t[0, :]
        scores_t[:, 0] = actual_t[:, 0]

    return scores_t


# ── Run a single ablation condition ───────────────────────────────────────


@torch.no_grad()
def run_condition(model, layer, head, tok_tensor, shift_labels,
                  scores_tensor, unpatched_logits):
    """
    Patch one head's attention scores and measure metrics.

    Returns dict: loss, kl_div, top1_change_frac.
    """
    seq_len = scores_tensor.shape[0]
    score_hook = f"blocks.{layer}.attn.hook_attn_scores"

    def hook_fn(attn_scores, hook):
        attn_scores[0, head, :seq_len, :seq_len] = scores_tensor
        return attn_scores

    patched_logits = model.run_with_hooks(
        tok_tensor, fwd_hooks=[(score_hook, hook_fn)]
    )

    loss = F.cross_entropy(patched_logits[0, :-1], shift_labels).item()

    # KL divergence (position-averaged)
    # Use log_softmax + log_target=True to avoid 0 * -inf = NaN from vocabulary underflow
    kl = F.kl_div(
        F.log_softmax(patched_logits[0, :-1].float(), dim=-1),
        F.log_softmax(unpatched_logits[0, :-1].float(), dim=-1),
        reduction="batchmean",
        log_target=True,
    ).item()

    # Top-1 prediction change
    pred_clean = unpatched_logits[0, :-1].argmax(dim=-1)
    pred_patched = patched_logits[0, :-1].argmax(dim=-1)
    top1_change = (pred_clean != pred_patched).float().mean().item()

    return {"loss": loss, "kl_div": kl, "top1_change_frac": top1_change}


# ── Run all conditions for one sample + one head ─────────────────────────


@torch.no_grad()
def run_single_sample(model, sae, text, layer, head, hook_point,
                      k_values, top_k_features=20, chunk_size=16,
                      rank_mode="sum"):
    """
    Run full ablation experiment for one text and one head.

    Returns dict keyed by condition name -> metrics dict.
    """
    device = next(model.parameters()).device

    # 1. Bias corrections + unpatched baseline
    bias = compute_bias_corrections(model, sae, text, layer, head, hook_point)
    if bias is None:
        return None

    seq_len = bias["seq_len"]
    tok_tensor = bias["tok_tensor"]
    shift_labels = bias["shift_labels"]
    unpatched_logits = bias["unpatched_logits"]

    # 2. Compute FRA
    fra_result = get_sentence_fra_batch(
        model, sae, text, layer, head,
        max_length=128, top_k=top_k_features, hook_point=hook_point,
        chunk_size=chunk_size, verbose=False,
        normalize_by_decoder_norm=None,
    )
    fra_sparse = fra_result["fra_tensor_sparse"]
    d_sae = fra_sparse.shape[2]

    # 3. Rank feature pairs (off-diagonal and on-diagonal)
    _indices_np = fra_sparse.indices().cpu().numpy()
    _values_np = fra_sparse.values().cpu().numpy()
    offdiag_pairs = rank_pairs(_indices_np, _values_np, top_k=len(_values_np), diagonal=False, mode=rank_mode)
    ondiag_pairs = rank_pairs(_indices_np, _values_np, top_k=len(_values_np), diagonal=True, mode=rank_mode)

    n_offdiag = len(offdiag_pairs)
    n_ondiag = len(ondiag_pairs)

    # 4. FRA full reconstruction (baseline)
    fra_sum_full = fra_sum_to_attn(fra_sparse, seq_len)
    scores_full = reconstruct_scores(fra_sum_full, bias, device)

    # 5. Zero-ablation scores
    mask_t = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1
    )
    scores_zero = torch.zeros((seq_len, seq_len), device=device) + mask_t

    # 6. Run conditions
    results = {}

    # Unpatched
    results["unpatched"] = {
        "loss": bias["unpatched_loss"], "kl_div": 0.0, "top1_change_frac": 0.0,
        "k": 0,
    }

    # FRA full
    r = run_condition(model, layer, head, tok_tensor, shift_labels,
                      scores_full, unpatched_logits)
    results["fra_full"] = {**r, "k": 0}

    # Zero
    r = run_condition(model, layer, head, tok_tensor, shift_labels,
                      scores_zero, unpatched_logits)
    results["zero"] = {**r, "k": 0}

    # For each k value: off-diagonal, random, on-diagonal
    for k in k_values:
        # Off-diagonal ablation
        k_eff = min(k, n_offdiag)
        pairs_off = [(p[0], p[1]) for p in offdiag_pairs[:k_eff]]
        fra_ablated = ablate_fra_pairs(fra_sparse, pairs_off, d_sae)
        fra_sum_abl = fra_sum_to_attn(fra_ablated, seq_len)
        scores_abl = reconstruct_scores(fra_sum_abl, bias, device)
        r = run_condition(model, layer, head, tok_tensor, shift_labels,
                          scores_abl, unpatched_logits)
        results[f"offdiag_{k}"] = {**r, "k": k_eff, "n_available": n_offdiag}

        # Random off-diagonal (same k, random pairs)
        if n_offdiag > 0:
            rng = np.random.RandomState(42 + k)
            rand_idx = rng.choice(n_offdiag, size=min(k, n_offdiag), replace=False)
            pairs_rand = [(offdiag_pairs[i][0], offdiag_pairs[i][1]) for i in rand_idx]
            fra_rand = ablate_fra_pairs(fra_sparse, pairs_rand, d_sae)
            fra_sum_rand = fra_sum_to_attn(fra_rand, seq_len)
            scores_rand = reconstruct_scores(fra_sum_rand, bias, device)
            r = run_condition(model, layer, head, tok_tensor, shift_labels,
                              scores_rand, unpatched_logits)
            results[f"random_{k}"] = {**r, "k": min(k, n_offdiag)}
        else:
            results[f"random_{k}"] = results["fra_full"].copy()

        # On-diagonal ablation
        k_on = min(k, n_ondiag)
        pairs_on = [(p[0], p[1]) for p in ondiag_pairs[:k_on]]
        fra_on = ablate_fra_pairs(fra_sparse, pairs_on, d_sae)
        fra_sum_on = fra_sum_to_attn(fra_on, seq_len)
        scores_on = reconstruct_scores(fra_sum_on, bias, device)
        r = run_condition(model, layer, head, tok_tensor, shift_labels,
                          scores_on, unpatched_logits)
        results[f"ondiag_{k}"] = {**r, "k": k_on, "n_available": n_ondiag}

    # Add metadata
    results["_meta"] = {
        "seq_len": seq_len,
        "n_offdiag_pairs": n_offdiag,
        "n_ondiag_pairs": n_ondiag,
        "nnz": fra_result["total_interactions"],
        "head_contribution": results["zero"]["loss"] - results["unpatched"]["loss"],
    }

    return results


# ── Aggregate and print ───────────────────────────────────────────────────


def aggregate_results(all_results, k_values):
    """Average results across texts for each condition."""
    conditions = ["unpatched", "fra_full", "zero"]
    for k in k_values:
        conditions += [f"offdiag_{k}", f"random_{k}", f"ondiag_{k}"]

    agg = {}
    for cond in conditions:
        losses = [r[cond]["loss"] for r in all_results if cond in r]
        kls = [r[cond]["kl_div"] for r in all_results if cond in r]
        t1s = [r[cond]["top1_change_frac"] for r in all_results if cond in r]
        if losses:
            agg[cond] = {
                "loss": np.mean(losses),
                "loss_std": np.std(losses),
                "kl_div": np.mean(kls),
                "top1_change_frac": np.mean(t1s),
                "n": len(losses),
            }
    return agg


def print_results(agg, k_values):
    """Print a formatted results table."""
    zero_loss = agg.get("zero", {}).get("loss", float("nan"))
    unpatched_loss = agg.get("unpatched", {}).get("loss", float("nan"))
    head_contrib = zero_loss - unpatched_loss

    print(f"\n{'Condition':<25} {'Loss':>8} {'dLoss':>8} {'KL':>10} {'Top1%':>7} {'Recovery':>9}")
    print("-" * 70)

    for cond in ["unpatched", "fra_full", "zero"]:
        if cond not in agg:
            continue
        a = agg[cond]
        dloss = a["loss"] - unpatched_loss
        recovery = (zero_loss - a["loss"]) / (head_contrib + 1e-10) if head_contrib > 0.01 else float("nan")
        print(f"  {cond:<23} {a['loss']:>8.4f} {dloss:>+8.4f} {a['kl_div']:>10.4f} "
              f"{a['top1_change_frac']*100:>6.1f}% {recovery:>8.3f}")

    for k in k_values:
        print(f"  --- k={k} ---")
        for prefix in ["offdiag", "random", "ondiag"]:
            cond = f"{prefix}_{k}"
            if cond not in agg:
                continue
            a = agg[cond]
            dloss = a["loss"] - unpatched_loss
            recovery = (zero_loss - a["loss"]) / (head_contrib + 1e-10) if head_contrib > 0.01 else float("nan")
            print(f"  {cond:<23} {a['loss']:>8.4f} {dloss:>+8.4f} {a['kl_div']:>10.4f} "
                  f"{a['top1_change_frac']*100:>6.1f}% {recovery:>8.3f}")


# ── Head screening ────────────────────────────────────────────────────────


@torch.no_grad()
def screen_heads(model, texts, layer, hook_point, max_length=128):
    """
    Quick zero-ablation sweep to find heads with the largest contribution.
    Returns list of (head_idx, avg_head_contribution) sorted descending.
    """
    device = next(model.parameters()).device
    n_heads = model.cfg.n_heads
    contributions = defaultdict(list)

    for text in texts:
        tokens = model.tokenizer.encode(text)[:max_length]
        tok_tensor = torch.tensor(tokens).unsqueeze(0).to(device)
        seq_len = len(tokens)
        if seq_len < 3:
            continue

        shift_labels = tok_tensor[0, 1:]
        logits_clean = model(tok_tensor)
        clean_loss = F.cross_entropy(logits_clean[0, :-1], shift_labels).item()

        score_hook = f"blocks.{layer}.attn.hook_attn_scores"
        mask_t = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1
        )

        for h in range(n_heads):
            def zero_hook(attn_scores, hook, _h=h):
                attn_scores[0, _h, :seq_len, :seq_len] = (
                    torch.zeros((seq_len, seq_len), device=device) + mask_t
                )
                return attn_scores

            zero_logits = model.run_with_hooks(
                tok_tensor, fwd_hooks=[(score_hook, zero_hook)]
            )
            zero_loss = F.cross_entropy(zero_logits[0, :-1], shift_labels).item()
            contributions[h].append(zero_loss - clean_loss)

    result = [(h, np.mean(v)) for h, v in contributions.items()]
    result.sort(key=lambda x: abs(x[1]), reverse=True)
    return result
