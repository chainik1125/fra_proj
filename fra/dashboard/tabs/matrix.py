"""Tab 2 -- Feature Matrix heatmap."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from fra.core.helpers import rank_pairs
from fra.dashboard.state import get_active_fra_data, get_fra_config, get_fra_data


def render(tab):
    with tab:
        if get_fra_data() is None:
            st.info("Click **\u25b6 Compute FRA** in the sidebar to see the feature matrix.")
            return

        cfg = get_fra_config()
        layer_ = cfg["layer"]

        fra_data, head_ = get_active_fra_data("matrix")


        _agg = cfg.get("agg_mode", "sum")

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

        st.subheader(f"FRA Feature Interaction Matrix \u2014 L{layer_} H{head_}")
        st.caption(
            f"Each cell shows the **{_agg}** interaction strength across "
            "position pairs. Only the top features appearing in the ranked list "
            "are shown."
        )

        if not pairs:
            st.warning("No pairs to display.")
            return

        # Collect unique features from top pairs
        top_features = []
        seen = set()
        for q, k, *_ in pairs:
            for f in (q, k):
                if f not in seen:
                    seen.add(f)
                    top_features.append(f)
            if len(top_features) >= 30:
                break

        feat_to_idx = {f: i for i, f in enumerate(top_features)}
        n = len(top_features)
        matrix = np.zeros((n, n))

        for q, k, s, c, m in pairs:
            if q in feat_to_idx and k in feat_to_idx:
                val = _pair_metric(q, k, s, c, m)
                matrix[feat_to_idx[q], feat_to_idx[k]] += val

        labels = [f"F{f}" for f in top_features]

        fig_mat = go.Figure(go.Heatmap(
            z=matrix,
            x=labels,
            y=labels,
            colorscale="Viridis",
            hovertemplate=(
                "Query: %{y}<br>Key: %{x}<br>Strength: %{z:.4f}<extra></extra>"
            ),
        ))
        fig_mat.update_layout(
            height=600,
            xaxis_title="Key Feature",
            yaxis_title="Query Feature",
            yaxis_autorange="reversed",
        )
        st.plotly_chart(fig_mat, use_container_width=True)
