#!/usr/bin/env python
"""
FRA Off-Diagonal Ablation Study
================================
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

Run:
  python -m fra.ablation_study                         # GPT-2, local ln1, L2
  python -m fra.ablation_study --heads 0 1 5 --k 10 50 100
  python -m fra.ablation_study --sae hub --layer 5     # hook_z SAE at L5
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformer_lens import HookedTransformer
from fra.fra_func import get_sentence_fra_batch
from fra.validation import (
    load_sae,
    get_qk_weights,
    fra_sum_to_attn,
)

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


# ── Feature pair ranking ──────────────────────────────────────────────────


def rank_feature_pairs(fra_sparse, diagonal=None, mode="sum"):
    """
    Rank (q_feat, k_feat) pairs by aggregated absolute strength.

    Args:
        fra_sparse: 4D sparse COO tensor [seq, seq, d_sae, d_sae]
        diagonal: None=all, True=only i==j, False=only i!=j
        mode: "sum" | "avg" | "max"

    Returns:
        List of (q_feat, k_feat, sum_abs, count) sorted descending.
    """
    indices = fra_sparse.indices().cpu().numpy()  # [4, nnz]
    values = fra_sparse.values().cpu().numpy()

    q_feats = indices[2]
    k_feats = indices[3]
    abs_vals = np.abs(values)

    if diagonal is True:
        mask = q_feats == k_feats
        q_feats, k_feats, abs_vals = q_feats[mask], k_feats[mask], abs_vals[mask]
    elif diagonal is False:
        mask = q_feats != k_feats
        q_feats, k_feats, abs_vals = q_feats[mask], k_feats[mask], abs_vals[mask]

    pair_sum: dict = defaultdict(float)
    pair_count: dict = defaultdict(int)
    pair_max: dict = defaultdict(float)

    for q, k, v in zip(q_feats, k_feats, abs_vals):
        key = (int(q), int(k))
        pair_sum[key] += float(v)
        pair_count[key] += 1
        pair_max[key] = max(pair_max[key], float(v))

    pairs = [
        (q, k, pair_sum[(q, k)], pair_count[(q, k)], pair_max[(q, k)])
        for (q, k) in pair_sum
    ]

    if mode == "sum":
        pairs.sort(key=lambda x: x[2], reverse=True)
    elif mode == "avg":
        pairs.sort(key=lambda x: x[2] / max(x[3], 1), reverse=True)
    elif mode == "max":
        pairs.sort(key=lambda x: x[4], reverse=True)

    return pairs


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
    Compute the bias correction terms needed to go from FRA sum to full
    pre-softmax attention scores.

    Returns dict with: term_q, term_k, term_const, attn_scale, seq_len,
                       tokens, unpatched_loss, unpatched_logits.
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

    W_Q, W_K, b_Q, b_K = get_qk_weights(model, layer, head)
    attn_scale = model.blocks[layer].attn.attn_scale

    # SAE reconstruction
    features = sae.encode(x)
    x_hat = sae.decode(features)
    b_dec = sae.b_dec if hasattr(sae, "b_dec") else sae.sae.b_dec

    x_hat_nobias = x_hat - b_dec
    q_nobias = (x_hat_nobias @ W_Q).cpu().numpy()
    k_nobias = (x_hat_nobias @ W_K).cpu().numpy()

    combined_q_bias = (b_dec @ W_Q + b_Q).cpu().numpy()
    combined_k_bias = (b_dec @ W_K + b_K).cpu().numpy()

    term_q = q_nobias @ combined_k_bias       # [seq]
    term_k = k_nobias @ combined_q_bias       # [seq]
    term_const = np.dot(combined_q_bias, combined_k_bias)

    return {
        "term_q": term_q,
        "term_k": term_k,
        "term_const": term_const,
        "attn_scale": attn_scale,
        "seq_len": seq_len,
        "tok_tensor": tok_tensor,
        "shift_labels": shift_labels,
        "unpatched_loss": unpatched_loss,
        "unpatched_logits": logits_clean,
    }


def reconstruct_scores(fra_sum_2d, bias, device):
    """
    Build full pre-softmax attention scores from FRA sum + bias corrections.

    Args:
        fra_sum_2d: [seq, seq] numpy array (FRA collapsed over feature dims)
        bias: dict from compute_bias_corrections
        device: torch device

    Returns:
        [seq, seq] torch tensor ready to patch into hook_attn_scores.
    """
    seq_len = bias["seq_len"]
    # fra_sum_2d already includes 1/sqrt(d_head) scaling from the FRA
    # computation.  Only the bias correction terms are unscaled dot products
    # and need dividing by attn_scale.
    scores = fra_sum_2d + (
        bias["term_q"][:, None]
        + bias["term_k"][None, :]
        + bias["term_const"]
    ) / bias["attn_scale"]

    # Causal mask
    causal = np.triu(np.full((seq_len, seq_len), float("-inf")), k=1)
    scores += causal

    return torch.tensor(scores, dtype=torch.float32, device=device)


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
    p = F.softmax(unpatched_logits[0, :-1], dim=-1)
    q = F.softmax(patched_logits[0, :-1], dim=-1)
    kl = F.kl_div(q.log(), p, reduction="batchmean").item()

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

    Returns dict keyed by condition name → metrics dict.
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
    offdiag_pairs = rank_feature_pairs(fra_sparse, diagonal=False, mode=rank_mode)
    ondiag_pairs = rank_feature_pairs(fra_sparse, diagonal=True, mode=rank_mode)

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


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="FRA Off-Diagonal Ablation Study")
    parser.add_argument("--model", choices=["gpt2", "gemma"], default="gpt2")
    parser.add_argument("--sae", choices=["hub", "local", "gemma"], default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--heads", type=int, nargs="+", default=None,
                        help="Heads to test (default: auto-select top 3 by contribution)")
    parser.add_argument("--k", type=int, nargs="+", default=[10, 50, 100, 500],
                        help="Number of feature pairs to ablate")
    parser.add_argument("--top-k-features", type=int, default=20,
                        help="Top-K SAE features per position in FRA")
    parser.add_argument("--n-texts", type=int, default=None,
                        help="Number of texts to use (default: all)")
    parser.add_argument("--rank-mode", choices=["sum", "avg", "max"], default="sum",
                        help="How to rank feature pairs for ablation")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save", type=str, default=None,
                        help="Save JSON results to this path")
    parser.add_argument("--chunk-size", type=int, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    is_gemma = args.model == "gemma"

    # Model/SAE defaults
    if is_gemma:
        sae_type = args.sae or "gemma"
        sae_layer = args.layer if args.layer is not None else 12
        layer = sae_layer + 1
        hook_point = "hook_resid_pre"
        chunk_size = args.chunk_size or 1
    else:
        sae_type = args.sae or "local"
        layer = args.layer if args.layer is not None else 2
        chunk_size = args.chunk_size or 16
        if sae_type == "hub":
            hook_point = "attn.hook_z"
        else:
            hook_point = "ln1.hook_normalized"

    k_values = args.k
    texts = ABLATION_TEXTS[:args.n_texts] if args.n_texts else ABLATION_TEXTS

    print("=" * 70)
    print("  FRA Off-Diagonal Ablation Study")
    print("=" * 70)
    print(f"  Model      : {'gemma-2-2b' if is_gemma else 'gpt2'}")
    print(f"  SAE        : {sae_type} ({hook_point})")
    print(f"  Layer      : {layer}")
    print(f"  K values   : {k_values}")
    print(f"  Rank mode  : {args.rank_mode}")
    print(f"  FRA top-k  : {args.top_k_features}")
    print(f"  Texts      : {len(texts)}")
    print(f"  Device     : {device}")

    # Load SAE
    sae_load_layer = sae_layer if is_gemma else layer
    print("\nLoading SAE...", end=" ", flush=True)
    sae = load_sae(sae_type, sae_load_layer, device)
    print(f"done. (d_sae={sae.d_sae})")

    # Load model
    if is_gemma:
        model_kw = {
            "fold_ln": False, "center_unembed": False,
            "center_writing_weights": False, "fold_value_biases": False,
            "refactor_factored_attn_matrices": False,
        }
    elif sae_type == "local":
        model_kw = {"fold_ln": False, "center_unembed": True, "center_writing_weights": True}
    else:
        model_kw = {}

    print("Loading model...", end=" ", flush=True)
    model = HookedTransformer.from_pretrained(
        "gemma-2-2b" if is_gemma else "gpt2", device=device, **model_kw
    )
    print("done.")

    # Head selection
    if args.heads is not None:
        heads = args.heads
    else:
        print("\nScreening heads (zero-ablation)...", flush=True)
        head_contribs = screen_heads(model, texts[:3], layer, hook_point)
        heads = [h for h, c in head_contribs[:3]]
        print("  Head contributions (top 5):")
        for h, c in head_contribs[:5]:
            print(f"    H{h}: {c:+.4f} ({'helps' if c > 0 else 'hurts/neutral'})")
        print(f"  Selected heads: {heads}")

    print(f"\n  Heads      : {heads}")
    print("=" * 70)

    # Run study
    all_per_head_results = {}

    for head in heads:
        print(f"\n{'='*70}")
        print(f"  HEAD {head}")
        print(f"{'='*70}")

        all_results = []
        for i, text in enumerate(texts):
            short = text[:55] + "..." if len(text) > 55 else text
            print(f"\n  Text {i+1}/{len(texts)}: \"{short}\"")

            result = run_single_sample(
                model, sae, text, layer, head, hook_point,
                k_values=k_values,
                top_k_features=args.top_k_features,
                chunk_size=chunk_size,
                rank_mode=args.rank_mode,
            )
            if result is None:
                print("    Skipped (too short)")
                continue

            meta = result["_meta"]
            print(f"    seq={meta['seq_len']}, nnz={meta['nnz']:,}, "
                  f"offdiag_pairs={meta['n_offdiag_pairs']}, "
                  f"ondiag_pairs={meta['n_ondiag_pairs']}, "
                  f"head_contrib={meta['head_contribution']:+.4f}")
            print(f"    fra_full loss={result['fra_full']['loss']:.4f}, "
                  f"zero loss={result['zero']['loss']:.4f}")

            all_results.append(result)

        if all_results:
            agg = aggregate_results(all_results, k_values)
            print(f"\n  Aggregated results for L{layer} H{head} "
                  f"({len(all_results)} texts):")
            print_results(agg, k_values)
            all_per_head_results[head] = agg

    # Overall summary
    print(f"\n\n{'='*70}")
    print(f"  OVERALL SUMMARY")
    print(f"{'='*70}")

    for head, agg in all_per_head_results.items():
        unp = agg.get("unpatched", {}).get("loss", float("nan"))
        fra = agg.get("fra_full", {}).get("loss", float("nan"))
        zero = agg.get("zero", {}).get("loss", float("nan"))
        hc = zero - unp

        print(f"\n  L{layer} H{head}:  unpatched={unp:.4f}  fra_full={fra:.4f}  "
              f"zero={zero:.4f}  head_contrib={hc:+.4f}")

        if hc < 0.01:
            print(f"    Head contribution too small for meaningful ablation analysis.")
            continue

        for k in k_values:
            off = agg.get(f"offdiag_{k}", {})
            rnd = agg.get(f"random_{k}", {})
            on = agg.get(f"ondiag_{k}", {})

            off_dloss = off.get("loss", unp) - unp
            rnd_dloss = rnd.get("loss", unp) - unp
            on_dloss = on.get("loss", unp) - unp

            off_rec = (zero - off.get("loss", zero)) / (hc + 1e-10)
            rnd_rec = (zero - rnd.get("loss", zero)) / (hc + 1e-10)
            on_rec = (zero - on.get("loss", zero)) / (hc + 1e-10)

            print(f"    k={k:>4}:  offdiag dL={off_dloss:+.4f} rec={off_rec:.3f}  |  "
                  f"random dL={rnd_dloss:+.4f} rec={rnd_rec:.3f}  |  "
                  f"ondiag dL={on_dloss:+.4f} rec={on_rec:.3f}")

    # Save JSON
    if args.save:
        save_data = {
            "config": {
                "model": args.model, "sae": sae_type, "layer": layer,
                "heads": heads, "k_values": k_values,
                "rank_mode": args.rank_mode, "n_texts": len(texts),
            },
            "per_head": {
                str(h): {cond: vals for cond, vals in agg.items()}
                for h, agg in all_per_head_results.items()
            },
        }
        with open(args.save, "w") as f:
            json.dump(save_data, f, indent=2, default=str)
        print(f"\nResults saved to {args.save}")

    print("\nDone.")


if __name__ == "__main__":
    main()
