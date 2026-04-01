"""Tab — Reconstruction & Ablation (unified).

Three top-level sections:
  1. Heatmaps — FRA vs standard attention (pregenerated)
  2. Metrics — Reconstruction Quality (pregenerated) + Patched Loss (behind Run button)
  3. Ablations — custom pair ablation (behind Run button)
"""

import html as html_lib

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as _F

from fra.core.fra import _extract_rope_params, apply_rope_to_projected
from fra.core.helpers import compute_errors, fra_sum_to_attn, get_W_K, rank_pairs
from fra.dashboard.loaders import (
    load_crosscoder,
    load_model,
    load_model_gemma,
    load_model_pair,
    load_sae_gemma,
    load_sae_hub,
    load_sae_local,
)
from fra.dashboard.state import get_fra_config, get_fra_data
from fra.dashboard.widgets import _show_heatmap, make_heatmap


# ── Helpers ──────────────────────────────────────────────────────────────────


def _get_softcap(model, layer):
    """Return the attention logit soft-cap value, or 0 if the model doesn't use it."""
    return getattr(model.cfg, "attn_scores_soft_cap", 0.0) or 0.0


def _apply_softcap_np(scores, softcap):
    """Apply tanh soft-capping to a numpy score array (no-op if softcap==0)."""
    if softcap > 0:
        return softcap * np.tanh(scores / softcap)
    return scores


def _apply_softcap_t(scores, softcap):
    """Apply tanh soft-capping to a torch score tensor (no-op if softcap==0)."""
    if softcap > 0:
        return softcap * torch.tanh(scores / softcap)
    return scores


@torch.no_grad()
def _compute_recon_metrics(cfg, fra_data, device):
    """Compute SAE/crosscoder encode→decode reconstruction quality.

    Uses cached model/SAE loaders, stores result in session state so it's
    only computed once per FRA run.
    """
    sae_type = cfg.get("sae_type", "")
    is_crosscoder = sae_type == "crosscoder"
    trained_on_bos = cfg.get("trained_on_bos", True)
    exclude_bos = not trained_on_bos

    if is_crosscoder:
        base, it, target, crosscoder, model_idx, cc_layer = (
            _load_crosscoder_resources(cfg, device)
        )
        tokens = fra_data["tokens"][:128]
        tok_tensor = torch.tensor(tokens).unsqueeze(0).to(device)
        hook_name = f"blocks.{cc_layer}.hook_resid_post"
        _, base_cache = base.run_with_cache(tok_tensor, names_filter=[hook_name])
        _, it_cache = it.run_with_cache(tok_tensor, names_filter=[hook_name])
        base_act = base_cache[hook_name].squeeze(0)
        it_act = it_cache[hook_name].squeeze(0)
        # Always stack [base, instruct] — crosscoder order is fixed
        x_stacked = torch.stack([base_act, it_act], dim=1)
        target_act = base_act if crosscoder.model_idx == 0 else it_act
        features = crosscoder.encode(x_stacked)
        x_hat_stacked = crosscoder.decode(features)
        target_hat = x_hat_stacked[:, crosscoder.model_idx]
        x_np = target_act.cpu().float().numpy()
        xhat_np = target_hat.detach().cpu().float().numpy()
    else:
        model, sae = _load_model_sae(cfg, device)
        layer_ = cfg["layer"]
        hook_point = cfg.get("hook_point", "ln1.hook_normalized")
        tokens = fra_data["tokens"][:128]
        tok_tensor = torch.tensor(tokens).unsqueeze(0).to(device)
        hook_name = f"blocks.{layer_}.{hook_point}"
        _, cache = model.run_with_cache(tok_tensor, names_filter=[hook_name])
        act = cache[hook_name].squeeze(0)
        if act.dim() == 3:
            act = act.flatten(-2, -1)
        features = sae.encode(act)
        x_hat = sae.decode(features)
        x_np = act.cpu().float().numpy()
        xhat_np = x_hat.cpu().float().numpy()

    per_token_l0 = (features != 0).sum(dim=-1).float()
    return {
        "recon": compute_errors(x_np, xhat_np, exclude_bos=exclude_bos),
        "avg_active_features": float(per_token_l0.mean()),
        "sparsity": float((features == 0).float().mean()),
        "l0_min": float(per_token_l0.min()),
        "l0_max": float(per_token_l0.max()),
        "l0_std": float(per_token_l0.std()),
        "d_sae": features.shape[-1],
    }


def _load_model_sae(cfg, device):
    """Load model and SAE using cached dashboard loaders."""
    sae_type = cfg.get("sae_type", "")
    model_name = cfg.get("model_name", "gpt2-small")
    hf = cfg.get("hf_token", "")

    sae_hub_release = st.session_state.get("_sidebar_sae_hub_release", "")
    sae_hub_id = st.session_state.get("_sidebar_sae_hub_id", "")
    sae_local_path = st.session_state.get("_sidebar_sae_local_path", "")

    if sae_type in ("sae_gemma", "gemma"):
        model = load_model_gemma(model_name, device, hf)
        sae = load_sae_gemma(sae_hub_release, sae_hub_id, device)
    elif sae_type in ("sae_hub", "hub"):
        model = load_model(model_name, device)
        sae = load_sae_hub(sae_hub_release, sae_hub_id, device)
    else:
        model = load_model(model_name, device)
        sae = load_sae_local(sae_local_path, int(cfg["layer"]), device)

    return model, sae


def _load_crosscoder_resources(cfg, device):
    """Load model pair, target model, and crosscoder for crosscoder presets."""
    base_model_name = st.session_state.get("_sidebar_base_model_name", "")
    it_model_name = st.session_state.get("_sidebar_it_model_name", "")
    cc_it_arch = st.session_state.get("_sidebar_cc_it_arch", "")
    model_idx = st.session_state.get("_sidebar_model_idx", 0)
    cc_repo_id = st.session_state.get("_sidebar_crosscoder_repo_id", "")
    cc_subfolder = st.session_state.get("_sidebar_cc_subfolder", "")
    crosscoder_layer = st.session_state.get(
        "_sidebar_crosscoder_layer", cfg["layer"] - 1,
    )

    base, it = load_model_pair(base_model_name, it_model_name, device, cc_it_arch)
    target = base if model_idx == 0 else it
    crosscoder = load_crosscoder(cc_repo_id, model_idx, device, cc_subfolder)
    return base, it, target, crosscoder, model_idx, crosscoder_layer


def _run_loss_metrics(model, sae, cfg, fra_data, device, exclude_bos=False):
    """Compute FRA / SAE / zero loss patching using stored FRA data."""
    from fra.analysis.ablation import run_condition

    layer_ = cfg["layer"]
    head_ = cfg["head"]
    seq_len = fra_data["seq_len"]
    actual_bos = fra_data["attn_scores_np"] if exclude_bos else None

    # Use stored tokens (same as FRA computation) for consistency
    tokens = fra_data["tokens"][:seq_len]
    tok_t = torch.tensor(tokens).unsqueeze(0).to(device)
    if len(tokens) < 3:
        return None
    shift = tok_t[0, 1:]
    unp_log = model(tok_t)
    unpatched_loss = _F.cross_entropy(unp_log[0, :-1], shift).item()

    # SAE decode stored features → full reconstruction
    feat_acts = torch.tensor(
        fra_data["feat_acts_np"], dtype=torch.float32, device=device,
    )
    x_hat = sae.decode(feat_acts).float()

    # Decoder bias and no-bias reconstruction
    b_dec = (sae.b_dec if hasattr(sae, "b_dec") else sae.sae.b_dec).float()
    x_hat_nobias = x_hat - b_dec

    # Weight matrices
    W_Q = model.blocks[layer_].attn.W_Q[head_].float()
    W_K = get_W_K(model, layer_, head_).float()
    b_Q = model.blocks[layer_].attn.b_Q[head_].float()
    n_kv = model.cfg.n_key_value_heads or model.cfg.n_heads
    kv_head = head_ // (model.cfg.n_heads // n_kv)
    b_K = model.blocks[layer_].attn.b_K[kv_head].float()
    attn_scale = model.blocks[layer_].attn.attn_scale

    # Full Q/K (with bias) and nobias Q/K (what FRA decomposes)
    q_full = x_hat @ W_Q + b_Q
    k_full = x_hat @ W_K + b_K
    q_nobias = x_hat_nobias @ W_Q
    k_nobias = x_hat_nobias @ W_K

    # Apply RoPE to ALL Q/K vectors (must match compute_fra_sparse)
    rope_sin, rope_cos, rotary_dim, rotary_adjacent_pairs = (
        _extract_rope_params(model, layer_)
    )
    if rope_sin is not None:
        for pos in range(seq_len):
            q_full[pos] = apply_rope_to_projected(
                q_full[pos].unsqueeze(0), pos, rope_sin, rope_cos,
                rotary_dim, rotary_adjacent_pairs,
            ).squeeze(0)
            k_full[pos] = apply_rope_to_projected(
                k_full[pos].unsqueeze(0), pos, rope_sin, rope_cos,
                rotary_dim, rotary_adjacent_pairs,
            ).squeeze(0)
            q_nobias[pos] = apply_rope_to_projected(
                q_nobias[pos].unsqueeze(0), pos, rope_sin, rope_cos,
                rotary_dim, rotary_adjacent_pairs,
            ).squeeze(0)
            k_nobias[pos] = apply_rope_to_projected(
                k_nobias[pos].unsqueeze(0), pos, rope_sin, rope_cos,
                rotary_dim, rotary_adjacent_pairs,
            ).squeeze(0)

    softcap = _get_softcap(model, layer_)
    mask_t = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device),
        diagonal=1,
    )

    # SAE scores: full reconstruction through Q/K
    scores_sae = _apply_softcap_t(
        (q_full @ k_full.T) / attn_scale, softcap,
    ) + mask_t

    # FRA scores: fra_sum + RoPE-aware bias correction
    # (mirrors crosscoder path — exact in RoPE space, only top-k error remains)
    d_sae = fra_data["feat_acts_np"].shape[1]
    fra_sparse = torch.sparse_coo_tensor(
        torch.tensor(fra_data["indices_np"], dtype=torch.long),
        torch.tensor(fra_data["values_np"], dtype=torch.float32),
        size=torch.Size([seq_len, seq_len, d_sae, d_sae]),
    ).coalesce()
    fra_sum = fra_sum_to_attn(fra_sparse, seq_len)

    bias_correction = (
        (q_full @ k_full.T) - (q_nobias @ k_nobias.T)
    ) / attn_scale
    scores_fra = _apply_softcap_t(
        torch.tensor(fra_sum, dtype=torch.float32, device=device) + bias_correction,
        softcap,
    ) + mask_t

    # Copy actual BOS scores when SAE wasn't trained on BOS
    if actual_bos is not None:
        actual_t = torch.tensor(
            actual_bos[:seq_len, :seq_len], dtype=torch.float32, device=device,
        )
        scores_fra[0, :] = actual_t[0, :]
        scores_fra[:, 0] = actual_t[:, 0]
        scores_sae[0, :] = actual_t[0, :]
        scores_sae[:, 0] = actual_t[:, 0]

    # Zero scores
    scores_zero = torch.zeros((seq_len, seq_len), device=device) + mask_t

    # Run conditions
    r_fra = run_condition(model, layer_, head_, tok_t, shift, scores_fra, unp_log)
    r_sae = run_condition(model, layer_, head_, tok_t, shift, scores_sae, unp_log)
    r_zero = run_condition(model, layer_, head_, tok_t, shift, scores_zero, unp_log)

    return {
        "unpatched_loss": unpatched_loss,
        "fra": r_fra,
        "sae": r_sae,
        "zero": r_zero,
        "fra_sparse": fra_sparse,
        "d_sae": d_sae,
    }


def _run_crosscoder_loss_metrics(
    cfg, fra_data, device, exclude_bos=False,
    *, target, crosscoder, model_idx, crosscoder_layer,
):
    """Compute crosscoder / FRA / zero loss patching for crosscoder path."""
    from fra.analysis.ablation import run_condition

    layer_ = cfg["layer"]
    head_ = cfg["head"]
    seq_len = fra_data["seq_len"]

    tok_t = torch.tensor(fra_data["tokens"]).unsqueeze(0).to(device)
    shift = tok_t[0, 1:]

    logits_clean = target(tok_t)
    unpatched_loss = _F.cross_entropy(logits_clean[0, :-1], shift).item()

    d_sae = fra_data["feat_acts_np"].shape[1]
    fra_sparse = torch.sparse_coo_tensor(
        torch.tensor(fra_data["indices_np"], dtype=torch.long),
        torch.tensor(fra_data["values_np"], dtype=torch.float32),
        size=torch.Size([seq_len, seq_len, d_sae, d_sae]),
    ).coalesce()
    actual_bos = fra_data["attn_scores_np"] if exclude_bos else None

    mask_t = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device),
        diagonal=1,
    )

    # ── Crosscoder-patched condition ──
    # Decode stored features → reconstructed residual stream activations,
    # then apply RMSNorm and project through W_Q / W_K to get attention scores.
    # Decode in float32 to match FRA computation precision (which uses
    # W_dec.float() for per-feature projections).
    feat_acts = torch.tensor(
        fra_data["feat_acts_np"], dtype=torch.float32, device=device,
    )
    b_dec = crosscoder._crosscoder.decoder.bias[model_idx].float().to(device)
    x_hat = feat_acts @ crosscoder.W_dec.float() + b_dec  # [seq, d_model]

    # RMSNorm: crosscoder decodes into residual-stream space, but W_Q / W_K
    # (with gamma folded in by TransformerLens) expect post-RMSNorm input.
    eps = target.cfg.eps
    rms = (x_hat.pow(2).mean(dim=-1, keepdim=True) + eps).sqrt()
    x_hat_norm = x_hat / rms

    W_Q = target.blocks[layer_].attn.W_Q[head_].float()
    W_K = get_W_K(target, layer_, head_).float()
    n_kv = target.cfg.n_key_value_heads or target.cfg.n_heads
    kv_head = head_ // (target.cfg.n_heads // n_kv)
    b_Q = target.blocks[layer_].attn.b_Q[head_].float()
    b_K = target.blocks[layer_].attn.b_K[kv_head].float()
    attn_scale = target.blocks[layer_].attn.attn_scale

    # Full Q/K (with decoder bias contribution and attention biases)
    q_full = x_hat_norm @ W_Q + b_Q
    k_full = x_hat_norm @ W_K + b_K

    # Q/K from feature directions only (no decoder bias) — what FRA decomposes
    x_hat_nobias_norm = (x_hat - b_dec) / rms
    q_nobias = x_hat_nobias_norm @ W_Q
    k_nobias = x_hat_nobias_norm @ W_K

    # Apply RoPE to all Q/K vectors
    rope_sin, rope_cos, rotary_dim, rotary_adjacent_pairs = (
        _extract_rope_params(target, layer_)
    )
    if rope_sin is not None:
        for pos in range(seq_len):
            q_full[pos] = apply_rope_to_projected(
                q_full[pos].unsqueeze(0), pos, rope_sin, rope_cos,
                rotary_dim, rotary_adjacent_pairs,
            ).squeeze(0)
            k_full[pos] = apply_rope_to_projected(
                k_full[pos].unsqueeze(0), pos, rope_sin, rope_cos,
                rotary_dim, rotary_adjacent_pairs,
            ).squeeze(0)
            q_nobias[pos] = apply_rope_to_projected(
                q_nobias[pos].unsqueeze(0), pos, rope_sin, rope_cos,
                rotary_dim, rotary_adjacent_pairs,
            ).squeeze(0)
            k_nobias[pos] = apply_rope_to_projected(
                k_nobias[pos].unsqueeze(0), pos, rope_sin, rope_cos,
                rotary_dim, rotary_adjacent_pairs,
            ).squeeze(0)

    softcap = _get_softcap(target, layer_)
    scores_coder = _apply_softcap_t(
        (q_full @ k_full.T) / attn_scale, softcap,
    ) + mask_t

    # ── FRA-patched condition ──
    # FRA sum ≈ (q_nobias @ k_nobias.T) / attn_scale (feature pairs only).
    # The decoder bias creates cross-terms that FRA doesn't capture, so we
    # compute the correction matrix: full_scores - nobias_scores.
    fra_sum = fra_sum_to_attn(fra_sparse, seq_len)
    bias_correction = (
        (q_full @ k_full.T) - (q_nobias @ k_nobias.T)
    ) / attn_scale
    scores_fra = _apply_softcap_t(
        torch.tensor(fra_sum, dtype=torch.float32, device=device) + bias_correction,
        softcap,
    ) + mask_t

    if actual_bos is not None:
        actual_t = torch.tensor(
            actual_bos[:seq_len, :seq_len], dtype=torch.float32, device=device,
        )
        scores_coder[0, :] = actual_t[0, :]
        scores_coder[:, 0] = actual_t[:, 0]
        scores_fra[0, :] = actual_t[0, :]
        scores_fra[:, 0] = actual_t[:, 0]

    # ── Zero condition ──
    scores_zero = torch.zeros((seq_len, seq_len), device=device) + mask_t

    # ── Run all conditions ──
    r_fra = run_condition(target, layer_, head_, tok_t, shift, scores_fra, logits_clean)
    r_coder = run_condition(target, layer_, head_, tok_t, shift, scores_coder, logits_clean)
    r_zero = run_condition(target, layer_, head_, tok_t, shift, scores_zero, logits_clean)

    return {
        "unpatched_loss": unpatched_loss,
        "fra": r_fra,
        "sae": r_coder,
        "zero": r_zero,
        "fra_sparse": fra_sparse,
        "d_sae": d_sae,
    }


# ── Ablation sub-section ────────────────────────────────────────────────────


def _render_ablation(cfg, fra_data, seq_len, token_strs, device, exclude_bos=False):
    """Render the custom pair ablation UI inside an expander."""
    from fra.analysis.ablation import (
        ablate_fra_pairs,
        reconstruct_scores,
        run_condition,
    )

    _agg = cfg.get("agg_mode", "sum")
    sae_type = cfg.get("sae_type", "")

    all_pairs = rank_pairs(
        fra_data["indices_np"], fra_data["values_np"],
        top_k=100, diagonal=None, mode=_agg,
    )
    if not all_pairs:
        st.warning("No feature pairs found.")
        return

    offdiag = [p for p in all_pairs if p[0] != p[1]]
    ondiag = [p for p in all_pairs if p[0] == p[1]]
    st.markdown(
        f"**{len(offdiag)}** off-diagonal pairs, "
        f"**{len(ondiag)}** on-diagonal pairs in top 100.",
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        n_ablate = st.slider(
            "Number of top pairs to ablate",
            min_value=1, max_value=min(50, len(offdiag) or 1),
            value=min(10, len(offdiag) or 1),
        )
        abl_target = st.radio(
            "Ablation target",
            ["Top off-diagonal (i\u2260j)", "Top on-diagonal (i==j)",
             "Random off-diagonal"],
            help=(
                "**Off-diagonal**: cross-feature interactions. "
                "**On-diagonal**: self-interactions. Random is a control."
            ),
        )

    with col2:
        st.markdown("**Pairs to ablate:**")
        if abl_target.startswith("Top off"):
            sel_pairs = offdiag[:n_ablate]
        elif abl_target.startswith("Top on"):
            sel_pairs = ondiag[:n_ablate]
        else:
            import random as _rng_mod
            _rng = _rng_mod.Random(42)
            sel_pairs = _rng.sample(offdiag, min(n_ablate, len(offdiag)))

        for q, k, s, cnt, mx in sel_pairs[:15]:
            avg = s / max(cnt, 1)
            marker = "\u27f2" if q == k else "\u2192"
            st.text(f"  F{q} {marker} F{k}  (avg={avg:.4f}, sum={s:.4f})")
        if len(sel_pairs) > 15:
            st.text(f"  ... and {len(sel_pairs) - 15} more")

    run_abl = st.button("\u25b6  Run Ablation", type="primary")
    if not run_abl:
        return

    with st.spinner("Running ablation\u2026"):
        layer_ = cfg["layer"]
        head_ = cfg["head"]

        d_sae = fra_data["feat_acts_np"].shape[1]
        fra_sparse = torch.sparse_coo_tensor(
            torch.tensor(fra_data["indices_np"], dtype=torch.long),
            torch.tensor(fra_data["values_np"], dtype=torch.float32),
            size=torch.Size([seq_len, seq_len, d_sae, d_sae]),
        ).coalesce()

        # Load model + compute bias (same pattern as before)
        if sae_type == "crosscoder":
            _, _, target, _, _, _ = _load_crosscoder_resources(cfg, device)
            tok_t = torch.tensor(fra_data["tokens"]).unsqueeze(0).to(device)
            shift = tok_t[0, 1:]
            logits_clean = target(tok_t)
            unp_loss = _F.cross_entropy(logits_clean[0, :-1], shift).item()
            bias = {
                "term_q": np.zeros(seq_len),
                "term_k": np.zeros(seq_len),
                "term_const": 0.0,
                "attn_scale": 1.0,
                "seq_len": seq_len,
                "tok_tensor": tok_t,
                "shift_labels": shift,
                "unpatched_loss": unp_loss,
                "unpatched_logits": logits_clean,
            }
            _target = target
        else:
            from fra.analysis.ablation import compute_bias_corrections
            model, sae = _load_model_sae(cfg, device)
            hook = cfg.get("hook_point", "attn.hook_z")
            bias = compute_bias_corrections(
                model, sae, cfg["text"], layer_, head_, hook,
            )
            if bias is not None and bias["seq_len"] != seq_len:
                bias["term_q"] = bias["term_q"][:seq_len]
                bias["term_k"] = bias["term_k"][:seq_len]
                bias["seq_len"] = seq_len
                bias["tok_tensor"] = bias["tok_tensor"][:, :seq_len]
                bias["shift_labels"] = bias["tok_tensor"][0, 1:]
                _logits = model(bias["tok_tensor"])
                bias["unpatched_loss"] = _F.cross_entropy(
                    _logits[0, :-1], bias["shift_labels"],
                ).item()
                bias["unpatched_logits"] = _logits
            _target = model

        if bias is None:
            st.error("Text too short for ablation.")
            return

        actual_bos = fra_data["attn_scores_np"] if exclude_bos else None
        _sc = fra_data.get("softcap", 0.0) or 0.0
        fra_sum_full = fra_sum_to_attn(fra_sparse, seq_len)
        scores_full = reconstruct_scores(fra_sum_full, bias, device,
                                         actual_bos_scores=actual_bos, softcap=_sc)

        pairs_to_abl = [(int(p[0]), int(p[1])) for p in sel_pairs]
        fra_ablated = ablate_fra_pairs(fra_sparse, pairs_to_abl, d_sae)
        fra_sum_abl = fra_sum_to_attn(fra_ablated, seq_len)
        scores_abl = reconstruct_scores(fra_sum_abl, bias, device,
                                        actual_bos_scores=actual_bos, softcap=_sc)

        mask_t = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1,
        )
        scores_zero = torch.zeros((seq_len, seq_len), device=device) + mask_t

        tok_t = bias["tok_tensor"]
        shift = bias["shift_labels"]
        unp_log = bias["unpatched_logits"]

        r_full = run_condition(_target, layer_, head_, tok_t, shift, scores_full, unp_log)
        r_abl = run_condition(_target, layer_, head_, tok_t, shift, scores_abl, unp_log)
        r_zero = run_condition(_target, layer_, head_, tok_t, shift, scores_zero, unp_log)

    # Display ablation results
    st.markdown("---")
    st.markdown("**Ablation Results**")

    hc = r_zero["loss"] - bias["unpatched_loss"]
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Unpatched loss", f"{bias['unpatched_loss']:.4f}")
    mc2.metric(
        "FRA full loss", f"{r_full['loss']:.4f}",
        delta=f"{r_full['loss'] - bias['unpatched_loss']:+.4f}",
    )
    mc3.metric(
        "Ablated loss", f"{r_abl['loss']:.4f}",
        delta=f"{r_abl['loss'] - bias['unpatched_loss']:+.4f}",
    )
    mc4.metric(
        "Zero-ablated loss", f"{r_zero['loss']:.4f}",
        delta=f"{r_zero['loss'] - bias['unpatched_loss']:+.4f}",
    )

    mc5, mc6, mc7 = st.columns(3)
    mc5.metric("Ablation KL div", f"{r_abl['kl_div']:.4f}")
    mc6.metric("Top-1 changed", f"{r_abl['top1_change_frac']*100:.1f}%")
    if hc > 0.01:
        rec_full = (r_zero["loss"] - r_full["loss"]) / hc
        rec_abl = (r_zero["loss"] - r_abl["loss"]) / hc
        mc7.metric(
            "Recovery (full\u2192ablated)",
            f"{rec_full:.3f} \u2192 {rec_abl:.3f}",
            delta=f"{rec_abl - rec_full:+.3f}",
        )

    # Attention heatmap comparison
    st.markdown("**Attention score comparison**")
    ticks = list(range(seq_len))
    labels = [html_lib.escape(t) for t in token_strs]

    def _abl_hm(scores_np, hover):
        disp = scores_np.copy()
        disp[np.triu_indices_from(disp, k=1)] = np.nan
        return make_heatmap(disp, ticks, ticks, hover_label=hover, colorscale="RdBu", zmid=0)

    hm1, hm2, hm3 = st.columns(3)
    with hm1:
        _show_heatmap(
            _abl_hm(scores_full.cpu().numpy(), "Score"),
            ticks, labels, seq_len, compact_height=350, key="abl_full",
        )
        st.caption("FRA Full")
    with hm2:
        _show_heatmap(
            _abl_hm(scores_abl.cpu().numpy(), "Score"),
            ticks, labels, seq_len, compact_height=350, key="abl_after",
        )
        st.caption("After Ablation")
    with hm3:
        diff_np = scores_abl.cpu().numpy() - scores_full.cpu().numpy()
        _show_heatmap(
            _abl_hm(diff_np, "Score"),
            ticks, labels, seq_len, compact_height=350, key="abl_diff",
        )
        st.caption("Difference (ablated \u2212 full)")


# ── Main render ──────────────────────────────────────────────────────────────


def render(tab):
    with tab:
        fra_data = get_fra_data()
        if fra_data is None:
            st.info("Click **\u25b6 Compute FRA** in the sidebar to see reconstruction results.")
            return

        cfg = get_fra_config()
        layer_ = cfg["layer"]
        head_ = cfg["head"]
        seq_len = fra_data["seq_len"]
        token_strs = fra_data["token_strs"][:seq_len]
        sae_type = cfg.get("sae_type", "")
        is_crosscoder = sae_type == "crosscoder"
        trained_on_bos = cfg.get("trained_on_bos", True)
        exclude_bos = not trained_on_bos
        device = "cuda" if torch.cuda.is_available() else "cpu"

        coder_label = "Crosscoder" if is_crosscoder else "SAE"

        # Shared FRA data
        _softcap = fra_data.get("softcap", 0.0) or 0.0
        idxs = fra_data["indices_np"]
        vals = fra_data["values_np"]
        causal_mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
        attn_tick_vals = list(range(seq_len))
        attn_tick_labels = [html_lib.escape(t) for t in token_strs]

        # ═══════════════════════════════════════════════════════════════
        # Section 1: Heatmaps
        # ═══════════════════════════════════════════════════════════════

        st.subheader(f"Heatmaps \u2014 L{layer_} H{head_}")

        # Standard logits
        std_logits = fra_data["attn_scores_np"][:seq_len, :seq_len].copy()
        std_logits[causal_mask] = np.nan

        # Standard probs
        std_probs = fra_data["attn_pattern_np"][:seq_len, :seq_len].copy()
        std_probs[causal_mask] = np.nan

        # FRA logits — apply softcap to match what the model does
        fra_logits = np.zeros((seq_len, seq_len))
        for qp, kp, v in zip(idxs[0], idxs[1], vals):
            if qp < seq_len and kp < seq_len:
                fra_logits[qp, kp] += v
        fra_logits = _apply_softcap_np(fra_logits, _softcap)

        # Copy actual BOS row/col into FRA logits when SAE wasn't trained on BOS
        if exclude_bos:
            fra_logits[0, :] = fra_data["attn_scores_np"][0, :seq_len]
            fra_logits[:, 0] = fra_data["attn_scores_np"][:seq_len, 0]

        # FRA probs
        fra_probs = np.full((seq_len, seq_len), np.nan)
        for q in range(seq_len):
            row = fra_logits[q, :q + 1]
            row_exp = np.exp(row - row.max())
            fra_probs[q, :q + 1] = row_exp / row_exp.sum()

        fra_logits_display = fra_logits.copy()
        fra_logits_display[causal_mask] = np.nan

        # Row 1: Logits
        st.markdown("#### Pre-softmax logits")
        col_std_logit, col_fra_logit = st.columns(2)

        with col_std_logit:
            st.markdown("**Standard** (masked QK scores)")
            _show_heatmap(
                make_heatmap(std_logits, attn_tick_vals, attn_tick_vals, hover_label="Logit", zmid=0),
                attn_tick_vals, attn_tick_labels, seq_len,
                compact_height=380, key="attn_std_logit",
            )

        with col_fra_logit:
            st.markdown("**FRA** (signed sum over feature pairs)")
            _show_heatmap(
                make_heatmap(fra_logits_display, attn_tick_vals, attn_tick_vals, hover_label="Logit", zmid=0),
                attn_tick_vals, attn_tick_labels, seq_len,
                compact_height=380, key="attn_fra_logit",
            )

        # Row 2: Probs
        st.markdown("#### Post-softmax probabilities")
        col_std_prob, col_fra_prob = st.columns(2)

        with col_std_prob:
            st.markdown("**Standard** (attention weights)")
            _show_heatmap(
                make_heatmap(std_probs, attn_tick_vals, attn_tick_vals, hover_label="Weight"),
                attn_tick_vals, attn_tick_labels, seq_len,
                compact_height=380, key="attn_std_prob",
            )

        with col_fra_prob:
            st.markdown("**FRA** (softmax of FRA logits)")
            _show_heatmap(
                make_heatmap(fra_probs, attn_tick_vals, attn_tick_vals, hover_label="Weight"),
                attn_tick_vals, attn_tick_labels, seq_len,
                compact_height=380, key="attn_fra_prob",
            )

        # ═══════════════════════════════════════════════════════════════
        # Section 2: Metrics
        # ═══════════════════════════════════════════════════════════════

        st.markdown("---")
        st.subheader(f"Metrics \u2014 L{layer_} H{head_}")

        # ── 2a: Reconstruction Quality (pregenerated) ─────────────────

        st.markdown("#### Reconstruction Quality")

        if exclude_bos:
            st.caption(
                "BOS position excluded from all metrics "
                "(coder not trained on BOS activations).",
            )

        # FRA → attention score reconstruction
        st.markdown(f"**FRA \u2192 Attention Scores**")
        st.caption(
            "FRA-reconstructed attention vs the model's actual "
            "pre-softmax scores (causal region only).",
        )

        _causal_bool = np.tril(np.ones((seq_len, seq_len), dtype=bool))
        _std_causal = np.where(
            _causal_bool,
            fra_data["attn_scores_np"][:seq_len, :seq_len],
            0.0,
        )
        _fra_causal = np.zeros((seq_len, seq_len))
        for qp, kp, v in zip(idxs[0], idxs[1], vals):
            if qp < seq_len and kp < seq_len:
                _fra_causal[qp, kp] += v
        _fra_causal = _apply_softcap_np(_fra_causal, _softcap)
        _fra_causal[~_causal_bool] = 0.0

        errs = compute_errors(_std_causal, _fra_causal, exclude_bos=exclude_bos)

        vm1, vm2, vm3, vm4 = st.columns(4)
        vm1.metric("Frobenius rel. error", f"{errs['fro_rel_err']:.1%}")
        vm2.metric("Cosine similarity", f"{errs['cosine_sim']:.4f}")
        vm3.metric("R\u00b2", f"{errs['r_squared']:.4f}")
        vm4.metric("Mean abs. error", f"{errs['mean_abs_err']:.4f}")

        _pass = errs["fro_rel_err"] < 0.50
        if _pass:
            st.success(f"Attention reconstruction: PASS (Frobenius error {errs['fro_rel_err']:.1%} < 50%)")
        else:
            st.warning(f"Attention reconstruction: FAIL (Frobenius error {errs['fro_rel_err']:.1%} >= 50%)")

        # SAE / crosscoder → residual stream reconstruction
        if "_recon_metrics" not in st.session_state:
            with st.spinner(f"Computing {coder_label} reconstruction quality\u2026"):
                st.session_state["_recon_metrics"] = _compute_recon_metrics(
                    cfg, fra_data, device,
                )
        recon_m = st.session_state.get("_recon_metrics")
        if recon_m is not None:
            st.markdown(f"**{coder_label} \u2192 Residual Stream**")
            st.caption(
                f"How well the {coder_label.lower()} reconstructs the residual stream "
                "activations (encode then decode).",
            )
            recon = recon_m["recon"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Frobenius rel. error", f"{recon['fro_rel_err']:.1%}")
            c2.metric("Cosine similarity", f"{recon['cosine_sim']:.4f}")
            c3.metric("R\u00b2", f"{recon['r_squared']:.4f}")
            c4.metric("Mean abs. error", f"{recon['mean_abs_err']:.4f}")

            c5, c6, c7 = st.columns(3)
            c5.metric("Avg active features (L0)", f"{recon_m['avg_active_features']:.0f}")
            c6.metric("L0 range", f"{recon_m['l0_min']:.0f} \u2013 {recon_m['l0_max']:.0f}")
            c7.metric("Sparsity", f"{recon_m['sparsity']:.1%}")

        # ── 2b: Patched Loss (behind Run button) ─────────────────────

        st.markdown("---")
        st.markdown("#### Patched Loss")
        st.caption(
            "Patch one attention head's pre-softmax scores with "
            f"{coder_label}-reconstructed or FRA-reconstructed scores "
            "and measure the impact on next-token prediction loss.",
        )

        run_loss = st.button(
            "\u25b6  Run Patched Loss", type="primary",
            help=(
                f"Load the {coder_label}, reconstruct attention scores, "
                "and measure patched vs unpatched loss."
            ),
        )

        if run_loss:
            with st.spinner("Computing patched losses\u2026"):
                if is_crosscoder:
                    base, it, target, crosscoder, model_idx, cc_layer = (
                        _load_crosscoder_resources(cfg, device)
                    )
                    loss_r = _run_crosscoder_loss_metrics(
                        cfg, fra_data, device, exclude_bos=exclude_bos,
                        target=target, crosscoder=crosscoder,
                        model_idx=model_idx, crosscoder_layer=cc_layer,
                    )
                else:
                    model, sae = _load_model_sae(cfg, device)
                    loss_r = _run_loss_metrics(
                        model, sae, cfg, fra_data, device,
                        exclude_bos=exclude_bos,
                    )
                st.session_state["_val_loss"] = loss_r

        loss_r = st.session_state.get("_val_loss")

        if loss_r is None:
            st.caption(
                "Press **Run Patched Loss** to compute loss recovery metrics.",
            )
        elif loss_r.get("sae") is not None:
            hc = loss_r["zero"]["loss"] - loss_r["unpatched_loss"]
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Unpatched loss", f"{loss_r['unpatched_loss']:.4f}")
            c2.metric(
                f"{coder_label}-patched",
                f"{loss_r['sae']['loss']:.4f}",
                delta=f"{loss_r['sae']['loss'] - loss_r['unpatched_loss']:+.4f}",
            )
            c3.metric(
                "FRA-patched",
                f"{loss_r['fra']['loss']:.4f}",
                delta=f"{loss_r['fra']['loss'] - loss_r['unpatched_loss']:+.4f}",
            )
            c4.metric(
                "Zero-ablated",
                f"{loss_r['zero']['loss']:.4f}",
                delta=f"{loss_r['zero']['loss'] - loss_r['unpatched_loss']:+.4f}",
            )
            if hc > 0.01:
                sae_rec = (loss_r["zero"]["loss"] - loss_r["sae"]["loss"]) / hc
                fra_rec = (loss_r["zero"]["loss"] - loss_r["fra"]["loss"]) / hc
                c5.metric(
                    "Recovery",
                    f"{coder_label}: {sae_rec:.3f}",
                    delta=f"FRA: {fra_rec:.3f}",
                    delta_color="off",
                )
            else:
                c5.metric("Recovery", "N/A")

            st.caption(
                f"With all features and no top-k truncation, {coder_label}-patched "
                "and FRA-patched losses should match.",
            )

        # ═══════════════════════════════════════════════════════════════
        # Section 3: Ablations
        # ═══════════════════════════════════════════════════════════════

        st.markdown("---")
        st.subheader(f"Ablations \u2014 L{layer_} H{head_}")
        st.caption(
            "Ablate selected feature pairs from the FRA tensor and measure the "
            "impact on model output. This reveals which cross-feature interactions "
            "are causally important to this attention head's computation.",
        )
        _render_ablation(cfg, fra_data, seq_len, token_strs, device, exclude_bos=exclude_bos)
