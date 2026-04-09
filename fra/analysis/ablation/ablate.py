"""FRA ablation — sparse tensor manipulation and score reconstruction.

Functions for ablating feature pairs from the FRA sparse tensor, reconstructing
attention scores, computing per-query FRA interactions, and measuring the causal
impact of ablations on model output.
"""

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from fra.core.fra import get_sentence_fra_batch
from fra.core.helpers import (
    apply_rope_to_projected,
    compute_bias_correction,
    fra_sum_to_attn,
    project_qk,
    rank_pairs,
)
from fra.core.coder import FRACoder

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
    """Remove specific (q_feat, k_feat) pairs from the sparse FRA tensor.

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


@torch.no_grad()
def compute_fra_new_query(
    feat_q: torch.Tensor,
    feat_k_all: torch.Tensor,
    pairs: list,
    W_dec: torch.Tensor,
    W_Q: torch.Tensor,
    W_K: torch.Tensor,
    attn_scale: float,
    rms_q: float,
    rms_k_all: torch.Tensor,
    rope_params: tuple,
    q_pos: int,
) -> torch.Tensor:
    """Compute FRA interaction scores for a new query attending to T key positions.

    For each (feat_i, feat_j) pair, computes the contribution of feature i at
    the new query position to the attention score at each of the T key positions.
    The result can be subtracted from ``hook_attn_scores[0, head, 0, :T]`` to
    ablate those pair contributions during generation.

    All tensors must be on CPU; results are returned on CPU.

    Args:
        feat_q: ``[d_sae]`` feature activations at the new query token.
        feat_k_all: ``[T, d_sae]`` feature activations at all T key positions.
        pairs: List of ``(q_feat_idx, k_feat_idx)`` tuples to ablate.
        W_dec: ``[d_sae, d_model]`` decoder weight matrix.
        W_Q: ``[d_model, d_head]`` query projection (no bias).
        W_K: ``[d_model, d_head]`` key projection (no bias, GQA-aware).
        attn_scale: ``sqrt(d_head)``.
        rms_q: RMSNorm denominator at the new query position.
        rms_k_all: ``[T]`` RMSNorm denominators at all key positions.
        rope_params: ``(rope_sin, rope_cos, rotary_dim, rotary_adjacent_pairs)``.
            ``rope_sin`` / ``rope_cos`` are ``None`` when the model has no RoPE.
        q_pos: Absolute sequence position index of the new query token.

    Returns:
        ``[T]`` float32 CPU tensor of FRA interaction scores for the selected pairs.
    """
    T = feat_k_all.shape[0]
    scores = torch.zeros(T, dtype=torch.float32)
    if not pairs or T == 0:
        return scores

    rope_sin, rope_cos, rotary_dim, rotary_adj = rope_params
    use_rope = rope_sin is not None

    W_dec = W_dec.float()
    W_Q = W_Q.float()
    W_K = W_K.float()
    feat_q = feat_q.float()
    feat_k_all = feat_k_all.float()
    rms_k_all = rms_k_all.float()

    for feat_i, feat_j in pairs:
        q_act = feat_q[feat_i].item()
        if q_act == 0.0:
            continue
        k_acts = feat_k_all[:, feat_j]  # [T]
        if k_acts.abs().max().item() == 0.0:
            continue

        # Query: project through W_Q + RoPE at q_pos
        q_vec = (W_dec[feat_i] @ W_Q).unsqueeze(0)  # [1, d_head]
        if use_rope:
            q_vec = apply_rope_to_projected(
                q_vec, q_pos, rope_sin, rope_cos, rotary_dim, rotary_adj,
            )

        # Keys: project through W_K + RoPE at each of T positions
        k_vec_base = (W_dec[feat_j] @ W_K).unsqueeze(0)  # [1, d_head]
        d_head = k_vec_base.shape[-1]
        if use_rope:
            k_vecs = torch.empty(T, d_head, dtype=torch.float32)
            for t in range(T):
                k_vecs[t] = apply_rope_to_projected(
                    k_vec_base, t, rope_sin, rope_cos, rotary_dim, rotary_adj,
                ).squeeze(0)
            interactions = (q_vec @ k_vecs.T).squeeze(0) / attn_scale  # [T]
        else:
            dot = (q_vec @ k_vec_base.T).item()
            interactions = torch.full((T,), dot / attn_scale, dtype=torch.float32)

        scores += q_act * k_acts * interactions / (rms_q * rms_k_all)

    return scores


# ── Bias corrections & score reconstruction ───────────────────────────────


@torch.no_grad()
def compute_bias_corrections(model, sae, text, layer, head, hook_point, max_length=128):
    """Compute the bias correction matrix needed to go from FRA sum to full
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

    logits_clean = model(tok_tensor)
    unpatched_loss = F.cross_entropy(logits_clean[0, :-1], shift_labels).item()

    hook_name = f"blocks.{layer}.{hook_point}"
    _, cache = model.run_with_cache(tok_tensor, names_filter=[hook_name])
    x = cache[hook_name].squeeze(0)
    if x.dim() == 3:
        x = x.flatten(-2, -1)

    features = sae.encode(x)
    x_hat = sae.decode(features).float()
    b_dec = sae.b_dec

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
    """Build full pre-softmax attention scores from FRA sum + bias corrections.

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
            the causal mask.

    Returns:
        [seq, seq] torch tensor ready to patch into hook_attn_scores.
    """
    seq_len = bias["seq_len"]
    scores = fra_sum_2d + bias["bias_correction"][:seq_len, :seq_len]

    if softcap > 0:
        scores = softcap * np.tanh(scores / softcap)

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
    """Patch attention scores and measure metrics.

    When *head* is an int, patches a single head's scores (``scores_tensor``
    is a ``[seq, seq]`` tensor).  When *head* is ``None``, patches multiple
    heads simultaneously (``scores_tensor`` is a ``dict[int, Tensor]``
    mapping head indices to ``[seq, seq]`` tensors).

    Returns dict: loss, kl_div, top1_change_frac.
    """
    score_hook = f"blocks.{layer}.attn.hook_attn_scores"

    if head is not None:
        seq_len = scores_tensor.shape[0]

        def hook_fn(attn_scores, hook):
            attn_scores[0, head, :seq_len, :seq_len] = scores_tensor
            return attn_scores
    else:
        seq_len = next(iter(scores_tensor.values())).shape[0]

        def hook_fn(attn_scores, hook):
            for h, s in scores_tensor.items():
                attn_scores[0, h, :s.shape[0], :s.shape[1]] = s
            return attn_scores

    patched_logits = model.run_with_hooks(
        tok_tensor, fwd_hooks=[(score_hook, hook_fn)]
    )

    loss = F.cross_entropy(patched_logits[0, :-1], shift_labels).item()

    kl = F.kl_div(
        F.log_softmax(patched_logits[0, :-1].float(), dim=-1),
        F.log_softmax(unpatched_logits[0, :-1].float(), dim=-1),
        reduction="batchmean",
        log_target=True,
    ).item()

    pred_clean = unpatched_logits[0, :-1].argmax(dim=-1)
    pred_patched = patched_logits[0, :-1].argmax(dim=-1)
    top1_change = (pred_clean != pred_patched).float().mean().item()

    return {"loss": loss, "kl_div": kl, "top1_change_frac": top1_change}


# ── Run all conditions for one sample + one head ─────────────────────────


@torch.no_grad()
def run_single_sample(model, sae, text, layer, head, hook_point,
                      k_values, top_k_features=20, chunk_size=16,
                      rank_mode="sum"):
    """Run full ablation experiment for one text and one head.

    Returns dict keyed by condition name -> metrics dict.
    """
    device = next(model.parameters()).device

    bias = compute_bias_corrections(model, sae, text, layer, head, hook_point)
    if bias is None:
        return None

    seq_len = bias["seq_len"]
    tok_tensor = bias["tok_tensor"]
    shift_labels = bias["shift_labels"]
    unpatched_logits = bias["unpatched_logits"]

    fra_result = get_sentence_fra_batch(
        model, sae, text, layer, head,
        max_length=128, top_k=top_k_features, hook_point=hook_point,
        chunk_size=chunk_size, verbose=False,
        normalize_by_decoder_norm=None,
    )
    fra_sparse = fra_result["fra_tensor_sparse"]
    d_sae = fra_sparse.shape[2]

    _indices_np = fra_sparse.indices().cpu().numpy()
    _values_np = fra_sparse.values().cpu().numpy()
    offdiag_pairs = rank_pairs(_indices_np, _values_np, top_k=len(_values_np), diagonal=False, mode=rank_mode)
    ondiag_pairs = rank_pairs(_indices_np, _values_np, top_k=len(_values_np), diagonal=True, mode=rank_mode)

    n_offdiag = len(offdiag_pairs)
    n_ondiag = len(ondiag_pairs)

    fra_sum_full = fra_sum_to_attn(fra_sparse, seq_len)
    scores_full = reconstruct_scores(fra_sum_full, bias, device)

    mask_t = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1
    )
    scores_zero = torch.zeros((seq_len, seq_len), device=device) + mask_t

    results = {}

    results["unpatched"] = {
        "loss": bias["unpatched_loss"], "kl_div": 0.0, "top1_change_frac": 0.0,
        "k": 0,
    }

    r = run_condition(model, layer, head, tok_tensor, shift_labels,
                      scores_full, unpatched_logits)
    results["fra_full"] = {**r, "k": 0}

    r = run_condition(model, layer, head, tok_tensor, shift_labels,
                      scores_zero, unpatched_logits)
    results["zero"] = {**r, "k": 0}

    for k in k_values:
        k_eff = min(k, n_offdiag)
        pairs_off = [(p[0], p[1]) for p in offdiag_pairs[:k_eff]]
        fra_ablated = ablate_fra_pairs(fra_sparse, pairs_off, d_sae)
        fra_sum_abl = fra_sum_to_attn(fra_ablated, seq_len)
        scores_abl = reconstruct_scores(fra_sum_abl, bias, device)
        r = run_condition(model, layer, head, tok_tensor, shift_labels,
                          scores_abl, unpatched_logits)
        results[f"offdiag_{k}"] = {**r, "k": k_eff, "n_available": n_offdiag}

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

        k_on = min(k, n_ondiag)
        pairs_on = [(p[0], p[1]) for p in ondiag_pairs[:k_on]]
        fra_on = ablate_fra_pairs(fra_sparse, pairs_on, d_sae)
        fra_sum_on = fra_sum_to_attn(fra_on, seq_len)
        scores_on = reconstruct_scores(fra_sum_on, bias, device)
        r = run_condition(model, layer, head, tok_tensor, shift_labels,
                          scores_on, unpatched_logits)
        results[f"ondiag_{k}"] = {**r, "k": k_on, "n_available": n_ondiag}

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
    """Quick zero-ablation sweep to find heads with the largest contribution.

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
