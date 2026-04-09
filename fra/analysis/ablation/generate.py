"""FRA ablation during autoregressive generation.

KV-cached generation helpers supporting two ablation modes:

  - Prefill ablation (generate_with_prefill_ablation): the ablated patch is
    baked into the prompt's KV cache during prefill; all subsequent generation
    steps run with no hooks.  Cost is a single patched forward pass over the
    prompt; generation itself is identical to unablated decoding.

  - Per-step ablation (generate_with_ablation): each new query token's FRA
    interactions with all previous positions are additionally suppressed at
    every generation step.  The loop is architecture-agnostic: callers supply
    an encode_fn that extracts feature activations for the new token from the
    target model's residual stream.  Isolating this step means the same loop
    body handles single-model SAEs and multi-model crosscoders without
    branching on architecture type inside the library.
"""

import torch

from fra.analysis.ablation.ablate import compute_fra_new_query, reconstruct_scores
from fra.core.helpers import fra_sum_to_attn


@torch.no_grad()
def build_patch_scores(cfg, fra_data, fra_sparse, fra_abl_sparse, proj, ablation_type, device):
    """Build ``[seq, seq]`` patch scores tensor (with softcap + causal mask applied).

    Args:
        ablation_type: ``"coder_recon"`` or ``"fra_sum"``.
    """
    seq_len = fra_data["seq_len"]
    sc = proj.get("softcap", 0.0) or 0.0
    exclude_bos = not cfg.get("trained_on_bos", True)
    actual_bos = fra_data["attn_scores_np"] if exclude_bos else None
    bias = {"bias_correction": proj["bias_corr_np"], "seq_len": seq_len}

    fra_sum_full = fra_sum_to_attn(fra_sparse, seq_len)
    fra_sum_abl = fra_sum_to_attn(fra_abl_sparse, seq_len)

    if ablation_type == "fra_sum":
        return reconstruct_scores(
            fra_sum_abl, bias, device,
            actual_bos_scores=actual_bos, softcap=sc,
        )

    # "coder_recon": baseline is (q_full @ k_full.T) / attn_scale
    q_full = proj["q_full"]
    k_full = proj["k_full"]
    attn_scale = proj["attn_scale"]

    coder_recon_pre = (q_full @ k_full.T) / attn_scale
    pair_delta = torch.tensor(
        fra_sum_full - fra_sum_abl, dtype=torch.float32, device=device,
    )
    patched_pre = coder_recon_pre - pair_delta

    if sc > 0:
        patched_pre = sc * torch.tanh(patched_pre / sc)
    mask_t = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1,
    )
    patch_scores = patched_pre + mask_t

    if actual_bos is not None:
        actual_t = torch.tensor(
            actual_bos[:seq_len, :seq_len], dtype=torch.float32, device=device,
        )
        patch_scores[0, :] = actual_t[0, :]
        patch_scores[:, 0] = actual_t[:, 0]

    return patch_scores


@torch.no_grad()
def prefill_with_patch(model, tok_ids, patch_scores, layer, head, device):
    """Run model on prompt with patched attention scores; return ``(kv, first_new_tok)``."""
    from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCache

    seq_len = len(tok_ids)
    tok_t = torch.tensor(tok_ids, dtype=torch.long, device=device).unsqueeze(0)
    kv = HookedTransformerKeyValueCache.init_cache(model.cfg, device, 1)

    def _hook(attn_scores, hook):
        attn_scores[0, head, :seq_len, :seq_len] = patch_scores
        return attn_scores

    logits = model.run_with_hooks(
        tok_t,
        fwd_hooks=[(f"blocks.{layer}.attn.hook_attn_scores", _hook)],
        past_kv_cache=kv,
    )
    return kv, int(logits[0, -1].argmax(-1).item())


@torch.no_grad()
def prefill_no_hooks(model, tok_ids, device):
    """Run model on prompt with no hooks; return ``(kv, first_new_tok)``."""
    from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCache

    tok_t = torch.tensor(tok_ids, dtype=torch.long, device=device).unsqueeze(0)
    kv = HookedTransformerKeyValueCache.init_cache(model.cfg, device, 1)
    logits = model.run_with_hooks(tok_t, fwd_hooks=[], past_kv_cache=kv)
    return kv, int(logits[0, -1].argmax(-1).item())


@torch.no_grad()
def generate_with_prefill_ablation(model, kv_cache, first_new_tok, max_new_tokens, eos_id, device):
    """Greedy generation from a KV cache with no per-step ablation hooks.

    Used for both the clean baseline and the prefill-only ablation condition:
    in both cases the prompt's KV cache already encodes whatever patching was
    applied during prefill, and subsequent generation steps run unmodified.
    The name reflects that any ablation lives entirely in the prefill, not here.
    """
    new_ids = []
    cur_tok_id = first_new_tok

    for _ in range(max_new_tokens):
        if eos_id is not None and cur_tok_id == eos_id:
            break
        new_ids.append(cur_tok_id)
        if len(new_ids) >= max_new_tokens:
            break
        cur_tok = torch.tensor([[cur_tok_id]], dtype=torch.long, device=device)
        logits = model.run_with_hooks(cur_tok, fwd_hooks=[], past_kv_cache=kv_cache)
        cur_tok_id = int(logits[0, -1].argmax(-1).item())

    return new_ids


@torch.no_grad()
def generate_with_ablation(
    target_model,
    target_kv,
    first_new_tok,
    max_new_tokens,
    eos_id,
    encode_fn,          # (cur_tok: Tensor[1,1], target_resid: Tensor[1, d_model]) -> Tensor[d_sae]
    cc_layer,           # layer at which to intercept the residual for encoding
    layer,              # attention layer whose scores are patched; must satisfy cc_layer < layer
    head,
    pairs_to_ablate,
    W_dec,              # [d_sae, d_model] CPU float32
    b_dec,              # [d_model] CPU float32
    W_Q,                # [d_model, d_head] CPU float32
    W_K,                # [d_model, d_head] CPU float32
    attn_scale,
    feat_acts_prompt,   # [seq, d_sae] — feature activations for all prompt positions
    rms_prompt,         # [seq] CPU float32 — RMSNorm denominators for prompt positions
    rope_params,        # (sin, cos, dim, adj) all CPU / None
    eps,                # model epsilon for numerically stable RMS computation
    device,
):
    """Greedy generation with per-step FRA ablation applied to every new query row.

    At each step, two hooks fire inside a single target_model forward pass:

      1. resid_hook at cc_layer captures the target residual and calls encode_fn
         to obtain feature activations for the new token.  encode_fn is an
         opaque callable — it may run a second model internally (multi-model
         crosscoder) or encode directly (single-model SAE).  Either way the
         generation loop sees the same interface and does not branch on
         architecture type.

      2. attn_hook at ``layer`` subtracts the FRA scores computed by
         compute_fra_new_query from the new query row of the attention scores.

    The two hooks must fire resid → attn, which is guaranteed by
    cc_layer < layer.  Passing both to a single run_with_hooks call lets
    TransformerLens execute them in one forward pass.

    feat_acts_all / rms_all grow by one entry each step so that
    compute_fra_new_query has access to key-side features for every token
    in the sequence so far.
    """
    W_dec_dev = W_dec.to(device).float()
    b_dec_dev = b_dec.to(device).float()

    # Accumulate feature activations and RMS denominators across all positions
    # (prompt + generated tokens).  These are the key-side inputs to
    # compute_fra_new_query; they must grow in lockstep with the KV cache.
    feat_acts_all = torch.tensor(feat_acts_prompt, dtype=torch.float32)  # CPU [seq, d_sae]
    rms_all = rms_prompt.cpu().float()                                    # CPU [seq]

    resid_hook_name = f"blocks.{cc_layer}.hook_resid_post"
    attn_hook_name = f"blocks.{layer}.attn.hook_attn_scores"

    new_ids = []
    cur_tok_id = first_new_tok

    for _ in range(max_new_tokens):
        if eos_id is not None and cur_tok_id == eos_id:
            break
        new_ids.append(cur_tok_id)
        if len(new_ids) >= max_new_tokens:
            break

        cur_tok = torch.tensor([[cur_tok_id]], dtype=torch.long, device=device)
        T = feat_acts_all.shape[0]

        # Shared state between the two hooks in this step.  A dict avoids
        # nonlocal and is safe when closures are defined inside a loop.
        _step = {}
        _fa_snap = feat_acts_all  # snapshot so default-arg binding captures the current value
        _rms_snap = rms_all

        def _hook_resid(
            target_resid, hook,
            _fa=_fa_snap, _rk=_rms_snap, _q_pos=T, _ss=_step,
        ):
            # encode_fn abstracts over encoding architecture: a single-model
            # SAE encodes target_resid directly; a multi-model crosscoder runs
            # the paired model here and stacks both residuals before encoding.
            # Either way we receive feature activations [d_sae] for this token.
            fa_new = encode_fn(cur_tok, target_resid[0].float())  # [d_sae]

            x_hat_new = fa_new.to(W_dec_dev.device) @ W_dec_dev + b_dec_dev
            rms_new = float((x_hat_new.pow(2).mean() + eps).sqrt().item())

            fra_scores = compute_fra_new_query(
                feat_q=fa_new.cpu(),
                feat_k_all=_fa,
                pairs=pairs_to_ablate,
                W_dec=W_dec,
                W_Q=W_Q,
                W_K=W_K,
                attn_scale=attn_scale,
                rms_q=rms_new,
                rms_k_all=_rk,
                rope_params=rope_params,
                q_pos=_q_pos,
            )  # [T] CPU
            _ss["fra_scores"] = fra_scores.to(device)
            _ss["fa_new"] = fa_new.cpu()
            _ss["rms_new"] = rms_new
            return target_resid

        def _hook_attn(attn_scores, hook, _ss=_step):
            if "fra_scores" in _ss:
                s = _ss["fra_scores"]
                T_cur = min(len(s), attn_scores.shape[-1])
                attn_scores[0, head, 0, :T_cur] -= s[:T_cur]
            return attn_scores

        logits = target_model.run_with_hooks(
            cur_tok,
            fwd_hooks=[(resid_hook_name, _hook_resid), (attn_hook_name, _hook_attn)],
            past_kv_cache=target_kv,
        )

        # Append this token's features and RMS so future steps see it as a key position.
        if "fa_new" in _step:
            feat_acts_all = torch.cat(
                [feat_acts_all, _step["fa_new"].unsqueeze(0)], dim=0,
            )
            rms_all = torch.cat(
                [rms_all, torch.tensor([_step["rms_new"]], dtype=torch.float32)], dim=0,
            )

        cur_tok_id = int(logits[0, -1].argmax(-1).item())

    return new_ids
