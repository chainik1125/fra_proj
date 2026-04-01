"""Tab — Reconstruction & Ablation (unified).

Section A: Activation reconstruction — SAE encode→decode quality + SAE-patched loss.
Section B: Attention reconstruction — FRA heatmaps, metrics, FRA-patched loss, ablation.
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
    from fra.analysis.ablation import (
        compute_bias_corrections,
        reconstruct_scores,
        run_condition,
    )

    layer_ = cfg["layer"]
    head_ = cfg["head"]
    seq_len = fra_data["seq_len"]
    hook = cfg.get("hook_point", "attn.hook_z")
    actual_bos = fra_data["attn_scores_np"] if exclude_bos else None

    bias = compute_bias_corrections(model, sae, cfg["text"], layer_, head_, hook)
    if bias is None:
        return None

    # Handle seq_len mismatch
    if bias["seq_len"] != seq_len:
        bias["term_q"] = bias["term_q"][:seq_len]
        bias["term_k"] = bias["term_k"][:seq_len]
        bias["seq_len"] = seq_len
        bias["tok_tensor"] = bias["tok_tensor"][:, :seq_len]
        bias["shift_labels"] = bias["tok_tensor"][0, 1:]
        logits_trim = model(bias["tok_tensor"])
        bias["unpatched_loss"] = _F.cross_entropy(
            logits_trim[0, :-1], bias["shift_labels"],
        ).item()
        bias["unpatched_logits"] = logits_trim

    # FRA scores from stored data
    d_sae = fra_data["feat_acts_np"].shape[1]
    fra_sparse = torch.sparse_coo_tensor(
        torch.tensor(fra_data["indices_np"], dtype=torch.long),
        torch.tensor(fra_data["values_np"], dtype=torch.float32),
        size=torch.Size([seq_len, seq_len, d_sae, d_sae]),
    ).coalesce()
    fra_sum = fra_sum_to_attn(fra_sparse, seq_len)
    scores_fra = reconstruct_scores(fra_sum, bias, device, actual_bos_scores=actual_bos)

    # SAE scores from stored feature activations
    feat_acts = torch.tensor(
        fra_data["feat_acts_np"], dtype=torch.float32, device=device,
    )
    x_hat = sae.decode(feat_acts).float()

    W_Q = model.blocks[layer_].attn.W_Q[head_].float()
    W_K = get_W_K(model, layer_, head_).float()
    b_Q = model.blocks[layer_].attn.b_Q[head_].float()
    n_kv = model.cfg.n_key_value_heads or model.cfg.n_heads
    kv_head = head_ // (model.cfg.n_heads // n_kv)
    b_K = model.blocks[layer_].attn.b_K[kv_head].float()
    attn_scale = model.blocks[layer_].attn.attn_scale

    q_full = x_hat @ W_Q + b_Q
    k_full = x_hat @ W_K + b_K

    # Apply RoPE (must match what compute_fra_sparse does)
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

    sae_scores = (q_full @ k_full.T) / attn_scale
    mask_t = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device),
        diagonal=1,
    )
    scores_sae = sae_scores + mask_t

    if actual_bos is not None:
        actual_t = torch.tensor(
            actual_bos[:seq_len, :seq_len], dtype=torch.float32, device=device,
        )
        scores_sae[0, :] = actual_t[0, :]
        scores_sae[:, 0] = actual_t[:, 0]

    # Zero scores
    scores_zero = torch.zeros((seq_len, seq_len), device=device) + mask_t

    # Run conditions
    tok_t = bias["tok_tensor"]
    shift = bias["shift_labels"]
    unp_log = bias["unpatched_logits"]

    r_fra = run_condition(model, layer_, head_, tok_t, shift, scores_fra, unp_log)
    r_sae = run_condition(model, layer_, head_, tok_t, shift, scores_sae, unp_log)
    r_zero = run_condition(model, layer_, head_, tok_t, shift, scores_zero, unp_log)

    return {
        "unpatched_loss": bias["unpatched_loss"],
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
    feat_acts = torch.tensor(
        fra_data["feat_acts_np"], dtype=torch.float32, device=device,
    )
    x_hat_stacked = crosscoder.decode(feat_acts.to(crosscoder.W_dec.dtype))  # [seq, 2, d_model]
    x_hat = x_hat_stacked[:, model_idx].float()           # [seq, d_model]

    # Decoder bias: crosscoder.decode() adds this, but FRA only decomposes
    # the feature-weighted decoder directions (without bias).
    b_dec = crosscoder._crosscoder.decoder.bias[model_idx].float().to(device)

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

    scores_coder = (q_full @ k_full.T) / attn_scale + mask_t

    # ── FRA-patched condition ──
    # FRA sum ≈ (q_nobias @ k_nobias.T) / attn_scale (feature pairs only).
    # The decoder bias creates cross-terms that FRA doesn't capture, so we
    # compute the correction matrix: full_scores - nobias_scores.
    fra_sum = fra_sum_to_attn(fra_sparse, seq_len)
    bias_correction = (
        (q_full @ k_full.T) - (q_nobias @ k_nobias.T)
    ) / attn_scale
    scores_fra = (
        torch.tensor(fra_sum, dtype=torch.float32, device=device)
        + bias_correction
        + mask_t
    )

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
        fra_sum_full = fra_sum_to_attn(fra_sparse, seq_len)
        scores_full = reconstruct_scores(fra_sum_full, bias, device, actual_bos_scores=actual_bos)

        pairs_to_abl = [(int(p[0]), int(p[1])) for p in sel_pairs]
        fra_ablated = ablate_fra_pairs(fra_sparse, pairs_to_abl, d_sae)
        fra_sum_abl = fra_sum_to_attn(fra_ablated, seq_len)
        scores_abl = reconstruct_scores(fra_sum_abl, bias, device, actual_bos_scores=actual_bos)

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

        # ═══════════════════════════════════════════════════════════════
        # Section A: Activation Reconstruction
        # ═══════════════════════════════════════════════════════════════

        st.subheader(f"Activation Reconstruction \u2014 L{layer_}")

        # Reconstruction tests button (runs both Section A and loss portions of B)
        run_val = st.button(
            "\u25b6  Run Reconstruction Tests", type="primary",
            help="Compute reconstruction quality and loss patching metrics.",
        )

        if run_val:
            with st.spinner("Running reconstruction tests\u2026"):
                if is_crosscoder:
                    from fra.analysis.validation import (
                        test_crosscoder_reconstruction,
                    )

                    base, it, target, crosscoder, model_idx, cc_layer = (
                        _load_crosscoder_resources(cfg, device)
                    )
                    sae_r = test_crosscoder_reconstruction(
                        base, it, crosscoder, fra_data["tokens"], cc_layer,
                    )
                    loss_r = _run_crosscoder_loss_metrics(
                        cfg, fra_data, device, exclude_bos=exclude_bos,
                        target=target, crosscoder=crosscoder,
                        model_idx=model_idx, crosscoder_layer=cc_layer,
                    )
                    st.session_state["_val_sae"] = sae_r
                    st.session_state["_val_loss"] = loss_r
                else:
                    from fra.analysis.validation import test_sae_reconstruction

                    model, sae = _load_model_sae(cfg, device)
                    hook = cfg.get("hook_point", "attn.hook_z")
                    sae_r = test_sae_reconstruction(
                        model, sae, cfg["text"], layer_, hook,
                        trained_on_bos=trained_on_bos,
                    )
                    loss_r = _run_loss_metrics(
                        model, sae, cfg, fra_data, device,
                        exclude_bos=exclude_bos,
                    )
                    st.session_state["_val_sae"] = sae_r
                    st.session_state["_val_loss"] = loss_r

        # Display Section A (coder encode→decode metrics)
        sae_r = st.session_state.get("_val_sae")
        loss_r = st.session_state.get("_val_loss")
        coder_label = "Crosscoder" if is_crosscoder else "SAE"

        if sae_r is not None:
            st.markdown(f"#### {coder_label} Reconstruction Quality")
            recon = sae_r["recon"]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Frobenius rel. error", f"{recon['fro_rel_err']:.1%}")
            c2.metric("Cosine similarity", f"{recon['cosine_sim']:.4f}")
            c3.metric("R\u00b2", f"{recon['r_squared']:.4f}")
            c4.metric("Mean abs. error", f"{recon['mean_abs_err']:.4f}")

        # Display coder-patched loss (Section A portion)
        if loss_r is not None and loss_r.get("sae") is not None:
            st.markdown(f"#### Loss Recovery ({coder_label}-patched)")
            hc = loss_r["zero"]["loss"] - loss_r["unpatched_loss"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Unpatched loss", f"{loss_r['unpatched_loss']:.4f}")
            c2.metric(
                f"{coder_label}-patched loss",
                f"{loss_r['sae']['loss']:.4f}",
                delta=f"{loss_r['sae']['loss'] - loss_r['unpatched_loss']:+.4f}",
            )
            c3.metric(
                "Zero-ablated loss",
                f"{loss_r['zero']['loss']:.4f}",
                delta=f"{loss_r['zero']['loss'] - loss_r['unpatched_loss']:+.4f}",
            )
            if hc > 0.01:
                sae_rec = (loss_r["zero"]["loss"] - loss_r["sae"]["loss"]) / hc
                c4.metric("Loss recovered", f"{sae_rec:.3f}")
            else:
                c4.metric("Loss recovered", "N/A")

        st.markdown("---")

        # ═══════════════════════════════════════════════════════════════
        # Section B: Attention Reconstruction (FRA)
        # ═══════════════════════════════════════════════════════════════

        st.subheader(f"Attention Reconstruction (FRA) \u2014 L{layer_} H{head_}")

        # ── Heatmaps (always available from fra_data) ──

        attn_tick_vals = list(range(seq_len))
        attn_tick_labels = [html_lib.escape(t) for t in token_strs]

        # Standard logits
        std_logits = fra_data["attn_scores_np"][:seq_len, :seq_len].copy()
        causal_mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
        std_logits[causal_mask] = np.nan

        # Standard probs
        std_probs = fra_data["attn_pattern_np"][:seq_len, :seq_len].copy()
        std_probs[causal_mask] = np.nan

        # FRA logits
        idxs = fra_data["indices_np"]
        vals = fra_data["values_np"]
        fra_logits = np.zeros((seq_len, seq_len))
        for qp, kp, v in zip(idxs[0], idxs[1], vals):
            if qp < seq_len and kp < seq_len:
                fra_logits[qp, kp] += v

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

        # ── Reconstruction quality metrics ──

        st.markdown("---")
        st.markdown("#### Reconstruction Metrics")
        st.caption(
            "Quantitative comparison of FRA-reconstructed attention vs the "
            "model's actual pre-softmax scores (causal region only).",
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
        _fra_causal[~_causal_bool] = 0.0

        errs = compute_errors(_std_causal, _fra_causal, exclude_bos=exclude_bos)

        if exclude_bos:
            st.caption(
                "Note: BOS position excluded from metrics "
                "(SAE not trained on BOS activations).",
            )

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

        # ── FRA loss recovery (from reconstruction test run) ──

        if loss_r is not None:
            st.markdown("#### Loss Recovery (FRA-patched)")
            hc = loss_r["zero"]["loss"] - loss_r["unpatched_loss"]
            lc1, lc2, lc3, lc4 = st.columns(4)
            lc1.metric("Unpatched", f"{loss_r['unpatched_loss']:.4f}")
            lc2.metric(
                "FRA-patched",
                f"{loss_r['fra']['loss']:.4f}",
                delta=f"{loss_r['fra']['loss'] - loss_r['unpatched_loss']:+.4f}",
            )
            lc3.metric(
                "Zero-ablated",
                f"{loss_r['zero']['loss']:.4f}",
                delta=f"{loss_r['zero']['loss'] - loss_r['unpatched_loss']:+.4f}",
            )
            if hc > 0.01:
                fra_rec = (loss_r["zero"]["loss"] - loss_r["fra"]["loss"]) / hc
                lc4.metric("Loss recovered", f"{fra_rec:.3f}")
            else:
                lc4.metric("Loss recovered", "N/A")

            st.caption(
                "Note: with all features and no top-k truncation, "
                f"FRA-patched loss should equal the {coder_label}-patched "
                "loss above.",
            )

        # ═══════════════════════════════════════════════════════════════
        # Ablation Expander
        # ═══════════════════════════════════════════════════════════════

        st.markdown("---")
        with st.expander("Advanced: Custom Pair Ablation"):
            st.caption(
                "Ablate selected feature pairs from the FRA tensor and measure the "
                "impact on model output. This reveals which cross-feature interactions "
                "are causally important to this attention head's computation.",
            )
            _render_ablation(cfg, fra_data, seq_len, token_strs, device, exclude_bos=exclude_bos)
