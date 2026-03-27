"""Tab 3 -- Validation (Standard vs FRA Attention comparison)."""

import html as html_lib

import numpy as np
import streamlit as st

from fra.core.helpers import compute_errors
from fra.dashboard._state import get_fra_config, get_fra_data
from fra.dashboard._widgets import _show_heatmap, make_heatmap


def render(tab):
    with tab:
        fra_data = get_fra_data()
        if fra_data is None:
            st.info("Click **\u25b6 Compute FRA** in the sidebar to see attention comparison.")
            return

        cfg = get_fra_config()
        layer_ = cfg["layer"]
        head_ = cfg["head"]
        seq_len = fra_data["seq_len"]
        token_strs = fra_data["token_strs"][:seq_len]

        st.subheader(f"Standard vs FRA Attention \u2014 L{layer_} H{head_}")

        # Use integer positions to avoid Plotly merging duplicate token labels
        attn_tick_vals = list(range(seq_len))
        attn_tick_labels = [html_lib.escape(t) for t in token_strs]

        # -- Build the four matrices --

        # Standard logits: masked pre-softmax scores (upper triangle = -inf -> NaN for display)
        std_logits = fra_data["attn_scores_np"][:seq_len, :seq_len].copy()
        causal_mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
        std_logits[causal_mask] = np.nan

        # Standard probs: post-softmax (already causal)
        std_probs = fra_data["attn_pattern_np"][:seq_len, :seq_len].copy()
        std_probs[causal_mask] = np.nan

        # FRA logits: signed sum over feature pairs per position
        idxs = fra_data["indices_np"]
        vals = fra_data["values_np"]
        fra_logits = np.zeros((seq_len, seq_len))
        q_pos_all = idxs[0, :]
        k_pos_all = idxs[1, :]
        for qp, kp, v in zip(q_pos_all, k_pos_all, vals):
            if qp < seq_len and kp < seq_len:
                fra_logits[qp, kp] += v

        # FRA probs: row-wise softmax of FRA logits (causal: only over k <= q)
        fra_probs = np.full((seq_len, seq_len), np.nan)
        for q in range(seq_len):
            row = fra_logits[q, :q + 1]
            row_exp = np.exp(row - row.max())
            fra_probs[q, :q + 1] = row_exp / row_exp.sum()

        fra_logits[causal_mask] = np.nan

        # -- Row 1: Logits --

        st.markdown("#### Pre-softmax logits")
        col_std_logit, col_fra_logit = st.columns(2)

        with col_std_logit:
            st.markdown("**Standard** (masked QK scores)")
            _show_heatmap(
                make_heatmap(std_logits, attn_tick_vals, "Logit", zmid=0),
                attn_tick_vals, attn_tick_labels, seq_len,
                compact_height=380, key="attn_std_logit",
            )

        with col_fra_logit:
            st.markdown("**FRA** (signed sum over feature pairs)")
            _show_heatmap(
                make_heatmap(fra_logits, attn_tick_vals, "Logit", zmid=0),
                attn_tick_vals, attn_tick_labels, seq_len,
                compact_height=380, key="attn_fra_logit",
            )

        # -- Row 2: Probs --

        st.markdown("#### Post-softmax probabilities")
        col_std_prob, col_fra_prob = st.columns(2)

        with col_std_prob:
            st.markdown("**Standard** (attention weights)")
            _show_heatmap(
                make_heatmap(std_probs, attn_tick_vals, "Weight"),
                attn_tick_vals, attn_tick_labels, seq_len,
                compact_height=380, key="attn_std_prob",
            )

        with col_fra_prob:
            st.markdown("**FRA** (softmax of FRA logits)")
            _show_heatmap(
                make_heatmap(fra_probs, attn_tick_vals, "Weight"),
                attn_tick_vals, attn_tick_labels, seq_len,
                compact_height=380, key="attn_fra_prob",
            )

        # -- Reconstruction quality metrics --

        st.markdown("---")
        st.markdown("#### Reconstruction Metrics")
        st.caption(
            "Quantitative comparison of FRA-reconstructed attention vs the "
            "model's actual pre-softmax scores (causal region only)."
        )

        # Compare FRA logits (before NaN masking) to standard logits.
        _causal_mask_bool = np.tril(np.ones((seq_len, seq_len), dtype=bool))
        _std_causal = np.where(
            _causal_mask_bool,
            fra_data["attn_scores_np"][:seq_len, :seq_len],
            0.0,
        )
        # Rebuild fra_logits without NaN masking for metrics
        _fra_causal = np.zeros((seq_len, seq_len))
        for qp, kp, v in zip(q_pos_all, k_pos_all, vals):
            if qp < seq_len and kp < seq_len:
                _fra_causal[qp, kp] += v
        _fra_causal[~_causal_mask_bool] = 0.0

        errs = compute_errors(_std_causal, _fra_causal)

        vm1, vm2, vm3, vm4 = st.columns(4)
        vm1.metric("Frobenius rel. error", f"{errs['fro_rel_err']:.1%}")
        vm2.metric("Cosine similarity", f"{errs['cosine_sim']:.4f}")
        vm3.metric("R\u00b2", f"{errs['r_squared']:.4f}")
        vm4.metric("Mean abs. error", f"{errs['mean_abs_err']:.4f}")

        _pass_attn = errs["fro_rel_err"] < 0.50
        if _pass_attn:
            st.success(f"Attention reconstruction: PASS (Frobenius error {errs['fro_rel_err']:.1%} < 50%)")
        else:
            st.warning(f"Attention reconstruction: FAIL (Frobenius error {errs['fro_rel_err']:.1%} >= 50%)")
