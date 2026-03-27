"""Shared UI component functions for the FRA Dashboard."""

import html as html_lib

import numpy as np
import plotly.graph_objects as go
import streamlit as st


def neuronpedia_embed_url(layer: int, feature_id: int) -> str:
    return (
        f"https://www.neuronpedia.org/gpt2-small/{layer}-att-kk/{feature_id}"
        f"?embed=true&embedexplanation=true&embedplots=true&embedtest=false"
    )


def _render_prompt_card(entry, feature_id, idx):
    """Render a single max-act prompt card with token-level highlighting."""
    is_safe = entry.get("is_safe")
    if is_safe is True:
        badge = '<span style="background:#28a745;color:white;padding:2px 8px;border-radius:4px;font-weight:bold;font-size:0.85em;">SAFE</span>'
        highlight_color = "66,133,244"  # blue
    elif is_safe is False:
        badge = '<span style="background:#dc3545;color:white;padding:2px 8px;border-radius:4px;font-weight:bold;font-size:0.85em;">UNSAFE</span>'
        highlight_color = "220,53,69"  # red
    else:
        badge = '<span style="background:#6c757d;color:white;padding:2px 8px;border-radius:4px;font-weight:bold;font-size:0.85em;">?</span>'
        highlight_color = "108,117,125"  # gray

    cats = entry.get("categories", [])
    cats_str = ", ".join(cats[:3]) if cats else ""

    max_act = entry.get("max_act", 0)
    n_active = entry.get("n_active_tokens", 0)
    n_tokens = entry.get("n_tokens", 0)

    # Header: badge + stats
    header_html = f"{badge}"
    if cats_str:
        header_html += f'&nbsp; <span style="color:#888;font-size:0.85em;">{html_lib.escape(cats_str)}</span>'
    header_html += (
        f'&nbsp;&nbsp; <span style="font-size:0.85em;">'
        f"max act: <b>{max_act:.3f}</b> &middot; "
        f"active tokens: {n_active}/{n_tokens}</span>"
    )

    # Token-level highlighted text
    token_strs_list = entry.get("token_strs")
    token_acts_list = entry.get("token_acts")

    if token_strs_list and token_acts_list:
        # Full token-level highlighting
        max_val = max(token_acts_list) if token_acts_list else 1.0
        if max_val == 0:
            max_val = 1.0
        spans = []
        for tok, act in zip(token_strs_list, token_acts_list):
            escaped = html_lib.escape(tok)
            if act > 0:
                alpha = min(0.15 + 0.85 * (act / max_val), 1.0)
                spans.append(
                    f'<span title="act={act:.4f}" style="background:rgba({highlight_color},{alpha:.2f});'
                    f'padding:1px 2px;border-radius:2px;font-family:monospace;font-size:0.88em;">'
                    f'{escaped}</span>'
                )
            else:
                spans.append(
                    f'<span style="font-family:monospace;font-size:0.88em;">{escaped}</span>'
                )
        text_html = " ".join(spans)
    else:
        # Fallback for old JSON without token_strs/token_acts
        text_html = (
            f'<span style="font-family:monospace;font-size:0.88em;">'
            f'{html_lib.escape(entry.get("prompt", ""))}</span>'
        )
        top_tokens = entry.get("top_tokens", [])
        if top_tokens:
            tok_parts = [
                f'{html_lib.escape(t["token"])} ({t["act"]:.3f})'
                for t in top_tokens[:5]
            ]
            text_html += (
                f'<br><span style="color:#888;font-size:0.82em;">'
                f'Top tokens: {", ".join(tok_parts)}</span>'
            )

    st.markdown(
        f'<div style="margin-bottom:12px;">'
        f'<div style="margin-bottom:4px;">{header_html}</div>'
        f'<div>{text_html}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    if idx < 100:  # avoid rendering 100+ dividers
        st.markdown("---")


def token_activation_bar(token_strs, activations, color, height=220, topk_activations=None):
    """Return a Plotly bar chart of per-token activations.

    If *topk_activations* is provided (the top-k filtered values used during FRA
    computation), positions where the feature was dropped by top-k filtering are
    shown as grey bars so the user can see which activations were invisible to FRA.
    """
    tick_vals = list(range(len(token_strs)))
    tick_labels = [html_lib.escape(t) for t in token_strs]

    if topk_activations is not None:
        # Per-bar color: grey where dropped (raw != 0 but topk == 0), normal otherwise
        colors = [
            "rgba(180,180,180,0.5)" if (act != 0 and topk == 0) else color
            for act, topk in zip(activations, topk_activations)
        ]
    else:
        colors = color

    fig = go.Figure(go.Bar(
        x=tick_vals,
        y=activations,
        marker_color=colors,
    ))
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=0, b=30),
        xaxis=dict(tickvals=tick_vals, ticktext=tick_labels),
        yaxis_title="Activation",
        showlegend=False,
    )
    return fig


def _show_heatmap(fig, tick_vals, tick_labels, seq_len, *,
                  x_title="Key", y_title="Query",
                  compact_height=300, key=None):
    """Render a heatmap without tick labels by default.

    Tick data is always embedded in the figure so a 'Show tokens' toggle
    button (inside the Plotly chart) can reveal them.  This button works
    both in the inline compact view and in Streamlit's full-screen modal
    (arrow icon, appears on hover over the chart).
    """
    fig.update_layout(
        height=compact_height,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(
            title=x_title,
            tickvals=tick_vals, ticktext=tick_labels,
            tickangle=90, showticklabels=False,
        ),
        yaxis=dict(
            title=y_title,
            tickvals=tick_vals, ticktext=tick_labels,
            autorange="reversed", showticklabels=False,
        ),
        updatemenus=[dict(
            type="buttons",
            direction="right",
            x=0.0, xanchor="left",
            y=1.0, yanchor="bottom",
            pad={"t": 4},
            showactive=True,
            buttons=[dict(
                label="Show tokens",
                method="relayout",
                args=[{
                    "xaxis.showticklabels": True,
                    "yaxis.showticklabels": True,
                }],
                args2=[{
                    "xaxis.showticklabels": False,
                    "yaxis.showticklabels": False,
                }],
            )],
        )],
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


def make_heatmap(z, x, y, hover_label="Value", colorscale="RdBu", zmid=None):
    """Generic plotly heatmap figure."""
    return go.Figure(go.Heatmap(
        z=z, x=x, y=y,
        colorscale=colorscale, zmid=zmid,
        hovertemplate=f"Q: %{{y}}<br>K: %{{x}}<br>{hover_label}: %{{z:.4f}}<extra></extra>",
    ))
