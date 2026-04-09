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

from fra.core.helpers import (
    compute_bias_correction,
    compute_errors,
    fra_sum_to_attn,
    get_attn_scale,
    project_qk,
    rank_pairs,
)
from fra.dashboard.loaders import (
    load_crosscoder,
    load_model,
    load_model_gemma,
    load_model_pair,
    load_sae_gemma,
    load_sae_hub,
    load_sae_local,
)
from fra.dashboard.state import get_active_fra_data, get_fra_config, get_fra_data, get_fra_data_all
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
def _get_projection_cache(cfg, fra_data, device):
    """Compute and cache QK projections and bias correction.

    Single source of truth for bias correction and coder-score components,
    shared by heatmaps, loss-patching, and ablation sections.

    Cached per head in ``st.session_state["_projection_cache_{head}"]``.
    All entries are cleared when a new FRA run is triggered (see app.py).

    Returns dict with keys:
        bias_corr_np:  ``[seq, seq]`` numpy bias correction matrix.
        bias_corr:     Same as tensor (on *device*).
        q_full:        ``[seq, d_head]`` full reconstruction projected through W_Q.
        k_full:        ``[seq, d_head]`` full reconstruction projected through W_K.
        attn_scale:    ``sqrt(d_head)``.
        softcap:       Attention logit soft-cap value (0 if unused).
    """
    head_ = cfg["head"]
    key = f"_projection_cache_{head_}"
    if key in st.session_state:
        return st.session_state[key]

    layer_ = cfg["layer"]
    head_ = cfg["head"]
    sae_type = cfg.get("sae_type", "")
    seq_len = fra_data["seq_len"]

    feat_acts = torch.tensor(
        fra_data["feat_acts_np"][:seq_len], dtype=torch.float32, device=device,
    )

    if sae_type == "crosscoder":
        _, _, target, crosscoder, _, _ = _load_crosscoder_resources(cfg, device)
        b_dec = crosscoder.b_dec.float().to(device)
        x_hat = feat_acts @ crosscoder.W_dec.float() + b_dec
        model = target
        needs_rms = True
    else:
        model, sae = _load_model_sae(cfg, device)
        x_hat = sae.decode(feat_acts).float()
        b_dec = sae.b_dec.float()
        needs_rms = "resid" in cfg.get("hook_point", "")

    q_full, k_full, q_nobias, k_nobias = project_qk(
        model, layer_, head_, x_hat, b_dec, needs_rms=needs_rms,
    )
    attn_scale = get_attn_scale(model, layer_)
    softcap = _get_softcap(model, layer_)
    bias_corr = compute_bias_correction(
        q_full, k_full, q_nobias, k_nobias, attn_scale,
    )
    bias_corr_sl = bias_corr[:seq_len, :seq_len]

    result = {
        "bias_corr_np": bias_corr_sl.cpu().numpy(),
        "bias_corr": bias_corr_sl.to(device),
        "q_full": q_full.to(device),
        "k_full": k_full.to(device),
        "attn_scale": attn_scale,
        "softcap": softcap,
    }
    st.session_state[key] = result
    return result


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


def _run_loss_metrics(proj, fra_scores, cfg, fra_data, device, exclude_bos=False):
    """Compute FRA / coder / zero loss patching.

    Uses the pre-computed ``fra_scores`` (the canonical causally-masked FRA
    reconstruction) and cached QK projections from ``_get_projection_cache``.
    Only the model forward pass is done here.

    Args:
        proj: Projection cache dict from ``_get_projection_cache``.
        fra_scores: ``[seq, seq]`` numpy array — the FRA-reconstructed
            attention scores (with bias correction, softcap, causal mask,
            and BOS override already applied).
        cfg: Dashboard config dict (must contain ``layer`` and ``head``).
        fra_data: Stored FRA results dict.
        device: Torch device.
        exclude_bos: Copy actual BOS scores into coder reconstruction.
    """
    from fra.analysis.ablation import run_condition

    layer_ = cfg["layer"]
    head_ = cfg["head"]
    sae_type = cfg.get("sae_type", "")
    seq_len = fra_data["seq_len"]

    # Reload model (st.cache_resource makes this a dict lookup)
    if sae_type == "crosscoder":
        _, _, model, _, _, _ = _load_crosscoder_resources(cfg, device)
    else:
        model, _ = _load_model_sae(cfg, device)

    tokens = fra_data["tokens"][:seq_len]
    tok_t = torch.tensor(tokens).unsqueeze(0).to(device)
    if len(tokens) < 3:
        return None
    shift = tok_t[0, 1:]
    logits_clean = model(tok_t)
    unpatched_loss = _F.cross_entropy(logits_clean[0, :-1], shift).item()

    mask_t = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device),
        diagonal=1,
    )

    # Coder scores: full reconstruction through Q/K
    q_full = proj["q_full"]
    k_full = proj["k_full"]
    scores_coder = _apply_softcap_t(
        (q_full @ k_full.T) / proj["attn_scale"], proj["softcap"],
    ) + mask_t
    if exclude_bos:
        actual_t = torch.tensor(
            fra_data["attn_scores_np"][:seq_len, :seq_len],
            dtype=torch.float32, device=device,
        )
        scores_coder[0, :] = actual_t[0, :]
        scores_coder[:, 0] = actual_t[:, 0]

    # FRA scores: use the canonical reconstruction directly
    scores_fra = torch.tensor(fra_scores, dtype=torch.float32, device=device)

    # Zero scores
    scores_zero = torch.zeros((seq_len, seq_len), device=device) + mask_t

    # Run conditions
    r_fra = run_condition(model, layer_, head_, tok_t, shift, scores_fra, logits_clean)
    r_coder = run_condition(model, layer_, head_, tok_t, shift, scores_coder, logits_clean)
    r_zero = run_condition(model, layer_, head_, tok_t, shift, scores_zero, logits_clean)

    return {
        "unpatched_loss": unpatched_loss,
        "fra": r_fra,
        "sae": r_coder,
        "zero": r_zero,
    }


def _run_loss_metrics_all_heads(cfg, fra_data, fra_data_all, device, n_heads,
                                fra_scores=None, head_=None, exclude_bos=False):
    """Compute loss patching with all heads patched simultaneously.

    Builds coder-reconstructed *and* FRA-reconstructed scores for every
    head, then patches all heads in a single forward pass per condition.
    """
    from fra.analysis.ablation import run_condition

    layer_ = cfg["layer"]
    seq_len = fra_data["seq_len"]

    model, coder_dict, zero_dict, fra_dict = _compute_all_heads_coder_scores(
        cfg, fra_data, device, n_heads,
        exclude_bos=exclude_bos, fra_data_all=fra_data_all,
    )

    tokens = fra_data["tokens"][:seq_len]
    tok_t = torch.tensor(tokens).unsqueeze(0).to(device)
    if len(tokens) < 3:
        return None
    shift = tok_t[0, 1:]
    logits_clean = model(tok_t)
    unpatched_loss = _F.cross_entropy(logits_clean[0, :-1], shift).item()

    r_coder = run_condition(model, layer_, None, tok_t, shift,
                            coder_dict, logits_clean)
    r_zero = run_condition(model, layer_, None, tok_t, shift,
                           zero_dict, logits_clean)

    r_fra = None
    if fra_dict is not None and len(fra_dict) == n_heads:
        r_fra = run_condition(model, layer_, None, tok_t, shift,
                              fra_dict, logits_clean)

    return {
        "unpatched_loss": unpatched_loss,
        "fra": r_fra,
        "sae": r_coder,
        "zero": r_zero,
    }


    # Ablation logic moved to fra.dashboard.tabs.ablation


# ── All-heads helpers ────────────────────────────────────────────────────────


@torch.no_grad()
def _compute_all_heads_coder_scores(cfg, fra_data, device, n_heads,
                                    exclude_bos=False, fra_data_all=None):
    """Compute coder-reconstructed attention scores for every head.

    Encodes through the SAE/crosscoder once (x_hat is shared), then
    projects through each head's W_Q/W_K.

    When *fra_data_all* is provided, also builds FRA-reconstructed scores
    per head (sparse FRA tensor summed to ``[seq, seq]`` + per-head bias
    correction + softcap + causal mask).

    Returns ``(model, coder_dict, zero_dict, fra_dict)`` where each dict
    maps ``{head: [seq, seq] tensor}``.  *fra_dict* is ``None`` when
    *fra_data_all* is not supplied.
    """
    layer_ = cfg["layer"]
    sae_type = cfg.get("sae_type", "")
    seq_len = fra_data["seq_len"]

    feat_acts = torch.tensor(
        fra_data["feat_acts_np"][:seq_len], dtype=torch.float32, device=device,
    )

    if sae_type == "crosscoder":
        _, _, model, crosscoder, _, _ = _load_crosscoder_resources(cfg, device)
        b_dec = crosscoder.b_dec.float().to(device)
        x_hat = feat_acts @ crosscoder.W_dec.float() + b_dec
        needs_rms = True
    else:
        model, sae = _load_model_sae(cfg, device)
        x_hat = sae.decode(feat_acts).float()
        b_dec = sae.b_dec.float()
        needs_rms = "resid" in cfg.get("hook_point", "")

    attn_scale = get_attn_scale(model, layer_)
    softcap = _get_softcap(model, layer_)
    mask_t = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device),
        diagonal=1,
    )

    coder_dict = {}
    zero_dict = {}
    fra_dict = {} if fra_data_all is not None else None
    zero_scores = torch.zeros((seq_len, seq_len), device=device) + mask_t

    for h in range(n_heads):
        q_full, k_full, q_nobias, k_nobias = project_qk(
            model, layer_, h, x_hat, b_dec, needs_rms=needs_rms,
        )
        scores = _apply_softcap_t(
            (q_full @ k_full.T) / attn_scale, softcap,
        ) + mask_t
        if exclude_bos:
            actual_t = torch.tensor(
                fra_data["attn_scores_np"][:seq_len, :seq_len],
                dtype=torch.float32, device=device,
            )
            scores[0, :] = actual_t[0, :]
            scores[:, 0] = actual_t[:, 0]
        coder_dict[h] = scores
        zero_dict[h] = zero_scores

        # FRA-reconstructed scores for this head
        if fra_dict is not None and h in fra_data_all:
            hd = fra_data_all[h]
            h_idxs = hd["indices_np"]
            h_vals = hd["values_np"]
            # Vectorised scatter-add over position pairs
            fra_np = np.zeros((seq_len, seq_len))
            mask = (h_idxs[0] < seq_len) & (h_idxs[1] < seq_len)
            np.add.at(fra_np, (h_idxs[0][mask], h_idxs[1][mask]), h_vals[mask])
            fra_sum = torch.tensor(fra_np, dtype=torch.float32, device=device)
            bias_corr = compute_bias_correction(
                q_full, k_full, q_nobias, k_nobias, attn_scale,
            )[:seq_len, :seq_len]
            fra_h = _apply_softcap_t(fra_sum + bias_corr, softcap) + mask_t
            if exclude_bos:
                actual_t = torch.tensor(
                    hd["attn_scores_np"][:seq_len, :seq_len],
                    dtype=torch.float32, device=device,
                )
                fra_h[0, :] = actual_t[0, :]
                fra_h[:, 0] = actual_t[:, 0]
            fra_dict[h] = fra_h

    return model, coder_dict, zero_dict, fra_dict


# ── Main render ──────────────────────────────────────────────────────────────


def render(tab):
    with tab:
        if get_fra_data() is None:
            st.info("Click **\u25b6 Compute FRA** in the sidebar to see reconstruction results.")
            return

        cfg = get_fra_config()
        layer_ = cfg["layer"]
        fra_data_all = get_fra_data_all()

        # Head selector (visible only when all-heads data exists)
        fra_data, head_ = get_active_fra_data("reconstruction")
        n_heads = cfg.get("n_heads", 1)
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

        # Bias correction: b_dec cross-terms that FRA doesn't capture
        # Use the tab-selected head, not the sidebar head stored in cfg.
        _cfg_head = {**cfg, "head": head_}
        with st.spinner("Computing bias correction\u2026"):
            proj = _get_projection_cache(_cfg_head, fra_data, device)
        bias_corr_np = proj["bias_corr_np"]

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

        # FRA-reconstructed attention scores (single source of truth).
        # Matches the model's hook_attn_scores: post-softcap, post-causal-mask.
        fra_scores = np.zeros((seq_len, seq_len))
        for qp, kp, v in zip(idxs[0], idxs[1], vals):
            if qp < seq_len and kp < seq_len:
                fra_scores[qp, kp] += v
        fra_scores += bias_corr_np
        fra_scores = _apply_softcap_np(fra_scores, _softcap)
        fra_scores[causal_mask] = -np.inf

        # Copy actual BOS row/col when SAE wasn't trained on BOS
        if exclude_bos:
            fra_scores[0, :] = fra_data["attn_scores_np"][0, :seq_len]
            fra_scores[:, 0] = fra_data["attn_scores_np"][:seq_len, 0]

        # Display version: -inf → NaN for plotting
        fra_logits_display = fra_scores.copy()
        fra_logits_display[causal_mask] = np.nan

        # FRA probs: softmax over causal region
        fra_probs = np.full((seq_len, seq_len), np.nan)
        for q in range(seq_len):
            row = fra_scores[q, :q + 1]
            row_exp = np.exp(row - row.max())
            fra_probs[q, :q + 1] = row_exp / row_exp.sum()

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
            st.markdown("**FRA** (feature pairs + bias correction)")
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
        _fra_causal = np.where(_causal_bool, fra_scores, 0.0)

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

        # All-heads option
        _can_all_heads = fra_data_all is not None
        patch_all_heads = False
        if _can_all_heads:
            patch_all_heads = st.checkbox(
                f"Patch all {n_heads} heads in layer {layer_}",
                value=False,
                key="patch_all_heads_loss",
                help=(
                    "Reconstruct and patch attention scores for every head "
                    "simultaneously.  Shows the total layer-level effect."
                ),
            )

        _target_desc = (
            f"all {n_heads} heads" if patch_all_heads
            else "one attention head"
        )
        st.caption(
            f"Patch {_target_desc}'s pre-softmax scores with "
            f"{coder_label}-reconstructed or FRA-reconstructed scores "
            "and measure the impact on next-token prediction loss.",
        )

        _loss_key = "_val_loss_all" if patch_all_heads else f"_val_loss_{head_}"

        run_loss = st.button(
            "\u25b6  Run Patched Loss", type="primary",
            help=(
                f"Load the {coder_label}, reconstruct attention scores, "
                "and measure patched vs unpatched loss."
            ),
        )

        if run_loss:
            with st.spinner("Computing patched losses\u2026"):
                if patch_all_heads:
                    loss_r = _run_loss_metrics_all_heads(
                        cfg, fra_data, fra_data_all, device, n_heads,
                        fra_scores=fra_scores, head_=head_,
                        exclude_bos=exclude_bos,
                    )
                else:
                    loss_r = _run_loss_metrics(
                        proj, fra_scores, _cfg_head, fra_data, device,
                        exclude_bos=exclude_bos,
                    )
                st.session_state[_loss_key] = loss_r

        loss_r = st.session_state.get(_loss_key)

        if loss_r is None:
            st.caption(
                "Press **Run Patched Loss** to compute loss recovery metrics.",
            )
        elif loss_r.get("sae") is not None:
            hc = loss_r["zero"]["loss"] - loss_r["unpatched_loss"]

            # Reference row
            r1, r2, r3 = st.columns(3)
            r1.metric("Unpatched loss", f"{loss_r['unpatched_loss']:.4f}")
            r2.metric(
                "Zero-ablated loss",
                f"{loss_r['zero']['loss']:.4f}",
                delta=f"{loss_r['zero']['loss'] - loss_r['unpatched_loss']:+.4f}",
            )
            if hc < 0:
                r3.metric(
                    "Headroom", f"{hc:.4f} (negative)",
                    help=(
                        "Zeroing this head improves loss \u2014 "
                        "the head hurts on this input."
                    ),
                )
            elif hc < 0.01:
                r3.metric("Headroom", "< 0.01")
            else:
                r3.metric("Headroom (zero \u2212 unpatched)", f"{hc:.4f}")

            has_recovery = hc > 0.01

            # Patched conditions: loss + recovery side by side
            _has_fra_loss = loss_r.get("fra") is not None
            if _has_fra_loss:
                st.markdown(
                    f"| | **{coder_label}-patched** | **FRA-patched** |\n"
                    f"|---|---|---|\n"
                    f"| **Loss** | {loss_r['sae']['loss']:.4f} "
                    f"({loss_r['sae']['loss'] - loss_r['unpatched_loss']:+.4f}) "
                    f"| {loss_r['fra']['loss']:.4f} "
                    f"({loss_r['fra']['loss'] - loss_r['unpatched_loss']:+.4f}) |\n"
                    + (
                        f"| **Recovery** | "
                        f"{(loss_r['zero']['loss'] - loss_r['sae']['loss']) / hc:.3f} | "
                        f"{(loss_r['zero']['loss'] - loss_r['fra']['loss']) / hc:.3f} |\n"
                        if has_recovery else ""
                    ),
                )
            else:
                # Fallback: only coder-patched (no FRA column)
                st.markdown(
                    f"| | **{coder_label}-patched** |\n"
                    f"|---|---|\n"
                    f"| **Loss** | {loss_r['sae']['loss']:.4f} "
                    f"({loss_r['sae']['loss'] - loss_r['unpatched_loss']:+.4f}) |\n"
                    + (
                        f"| **Recovery** | "
                        f"{(loss_r['zero']['loss'] - loss_r['sae']['loss']) / hc:.3f} |\n"
                        if has_recovery else ""
                    ),
                )

            if _has_fra_loss:
                st.caption(
                    f"With all features and no top-k truncation, {coder_label}-patched "
                    "and FRA-patched losses should match.",
                )

        # Ablation section moved to dedicated Ablation tab.
