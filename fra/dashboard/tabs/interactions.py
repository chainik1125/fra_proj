"""Tab 1 -- Top Interactions."""

import html as html_lib

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from fra.core.helpers import get_position_heatmap, rank_pairs, aggregate_pairs
from fra.dashboard._loaders import fetch_neuronpedia
from fra.dashboard._state import get_fra_config, get_fra_data
from fra.dashboard._widgets import (
    _show_heatmap,
    neuronpedia_embed_url,
    token_activation_bar,
)


def render(tab):
    with tab:
        fra_data = get_fra_data()
        if fra_data is None:
            st.info("Click **\u25b6 Compute FRA** in the sidebar to see interactions.")
            return

        cfg = get_fra_config()
        layer_ = cfg["layer"]
        head_ = cfg["head"]
        seq_len = fra_data["seq_len"]
        token_strs = fra_data["token_strs"][:seq_len]

        _agg = cfg.get("agg_mode", "sum")
        _metric_label = {"sum": "sum", "avg": "avg", "max": "max"}.get(_agg, "sum")

        def _pair_metric(q, k, s, c, m):
            if _agg == "avg":
                return s / max(c, 1)
            if _agg == "max":
                return m
            return s

        _diagonal = False if cfg["filter_self"] else None
        pairs = rank_pairs(
            fra_data["indices_np"],
            fra_data["values_np"],
            top_k=cfg["top_k_pairs"],
            diagonal=_diagonal,
            mode=_agg,
        )

        all_pairs_for_bottom = aggregate_pairs(fra_data["indices_np"], fra_data["values_np"], diagonal=_diagonal)
        all_pairs_for_bottom.sort(key=lambda x: x[2])
        bottom_pairs = all_pairs_for_bottom[:cfg["top_k_pairs"]]

        rank_mode = st.radio(
            "Show:",
            ["Top interactions (strongest)", "Least interactions (weakest)"],
            horizontal=True,
            label_visibility="collapsed",
        )
        active_pairs = pairs if rank_mode.startswith("Top") else bottom_pairs

        if not active_pairs:
            st.warning("No interactions found with current filters.")
            return

        col_list, col_detail = st.columns([1, 2])

        with col_list:
            st.subheader("Feature pairs")
            pair_labels = [
                f"F{q}\u2192F{k}  ({_pair_metric(q, k, s, c, m):.3f} {_metric_label})"
                + ("  \u27f2" if q == k else "")
                for q, k, s, c, m in active_pairs
            ]
            selected_idx = st.radio(
                "Select a pair to inspect:",
                range(len(active_pairs)),
                format_func=lambda i: pair_labels[i],
                label_visibility="collapsed",
            )

        with col_detail:
            q_sel, k_sel, strength_sel, count_sel, max_sel = active_pairs[selected_idx]
            is_self = q_sel == k_sel

            st.subheader(
                f"Feature {q_sel} \u2192 Feature {k_sel}"
                + ("  \u27f2 self" if is_self else "")
            )
            st.caption(
                f"Total absolute strength: **{strength_sel:.4f}** | "
                f"Position-pair occurrences: **{count_sel}**"
            )
            if is_self:
                st.info(
                    "Self-interaction: query and key are the **same** feature. "
                    "This is a candidate for a **conceptual induction head** channel."
                )

            # --- Per-token activation bars ---
            feat_acts = fra_data["feat_acts_np"]  # [seq_len, d_sae]
            q_acts = feat_acts[:, q_sel]
            k_acts = feat_acts[:, k_sel]
            topk_acts = fra_data.get("topk_acts_np")
            q_topk = topk_acts[:, q_sel] if topk_acts is not None else None
            k_topk = topk_acts[:, k_sel] if topk_acts is not None else None

            barA, barB = st.columns(2)
            with barA:
                st.markdown(f"**Query feature {q_sel}** \u2014 token activations")
                st.plotly_chart(
                    token_activation_bar(
                        token_strs, q_acts, "rgba(102,126,234,0.75)",
                        topk_activations=q_topk,
                    ),
                    use_container_width=True,
                )
            with barB:
                st.markdown(f"**Key feature {k_sel}** \u2014 token activations")
                st.plotly_chart(
                    token_activation_bar(
                        token_strs, k_acts, "rgba(118,75,162,0.75)",
                        topk_activations=k_topk,
                    ),
                    use_container_width=True,
                )

            # --- Position heatmap: [seq, seq] for this pair ---
            st.markdown("**Position heatmap** \u2014 where does this pair interact?")
            pos_mat = get_position_heatmap(
                fra_data["indices_np"],
                fra_data["values_np"],
                q_sel, k_sel, seq_len,
            )
            tick_labels = [html_lib.escape(t) for t in token_strs]
            tick_vals = list(range(len(token_strs)))
            fig_pos = go.Figure(go.Heatmap(
                z=pos_mat,
                x=tick_vals,
                y=tick_vals,
                colorscale="Blues",
                hovertemplate=(
                    "Q-pos: %{y}<br>K-pos: %{x}<br>Strength: %{z:.4f}"
                    "<extra></extra>"
                ),
            ))
            _show_heatmap(fig_pos, tick_vals, tick_labels, seq_len,
                          x_title="Key token", y_title="Query token",
                          key="pos_heatmap")

            # --- Neuronpedia iframes ---
            if cfg["supports_neuronpedia"]:
                np_col1, np_col2 = st.columns(2)
                with np_col1:
                    desc_q = fetch_neuronpedia(layer_, q_sel)
                    st.markdown(
                        f"**Neuronpedia \u2014 F{q_sel}:** _{desc_q}_"
                    )
                    st.components.v1.iframe(
                        neuronpedia_embed_url(layer_, q_sel),
                        height=380,
                    )
                with np_col2:
                    desc_k = fetch_neuronpedia(layer_, k_sel)
                    st.markdown(
                        f"**Neuronpedia \u2014 F{k_sel}:** _{desc_k}_"
                    )
                    st.components.v1.iframe(
                        neuronpedia_embed_url(layer_, k_sel),
                        height=380,
                    )
            else:
                st.info(
                    "Neuronpedia is only available with the hub (hook_z) SAE. "
                    "Switch SAE type in the sidebar to enable it."
                )
