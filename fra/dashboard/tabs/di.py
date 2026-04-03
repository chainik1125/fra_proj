"""Tab 5 -- Data-Independent (QK circuit) analysis."""

import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

from fra.analysis.induction import compute_di_row, compute_global_di_topk
from fra.core.helpers import rank_pairs
from fra.dashboard.compute import _load_di_weights
from fra.dashboard.state import PRESETS, get_fra_config, get_fra_data


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------

def _ensure_di_weights(sae_type, head, device, _wt_kw):
    """Load and cache DI weight matrices + RoPE params."""
    if st.session_state["_tab5_di_weights"] is None:
        W_dec, W_Q, W_K_, _attn, rope_params = _load_di_weights(
            sae_type, int(head), device, **_wt_kw,
        )
        st.session_state["_tab5_di_weights"] = (W_dec, W_Q, W_K_)
        st.session_state["_tab5_rope_params"] = rope_params
    return st.session_state["_tab5_di_weights"]


def _get_di_row(query_id, sae_type, head, device, _wt_kw, delta=0):
    """Compute (and cache) the full DI row for *query_id*."""
    cache = st.session_state["_tab5_di_row_cache"]
    if query_id not in cache:
        W_dec, W_Q, W_K_ = _ensure_di_weights(sae_type, head, device, _wt_kw)
        rope_params = st.session_state.get("_tab5_rope_params")
        cache[query_id] = compute_di_row(
            W_dec, W_Q, W_K_, query_id,
            rope_params=rope_params, delta=delta,
        )
    return cache[query_id]


def _di_for_pair(q, k, sae_type, head, device, _wt_kw, delta=0):
    """Return the scalar DI value for a (query, key) pair."""
    return float(_get_di_row(q, sae_type, head, device, _wt_kw, delta=delta)[k])


def _render_di_histogram(di_values, stats, mark_val=None, mark_label=None):
    """Render DI distribution histogram with SD lines."""
    mean, std = stats["mean"], stats["std"]
    fig = go.Figure()
    # Pre-bin with numpy to avoid sending millions of raw points to the browser.
    # Clip to +/-3 SD to zoom in on the meaningful range.
    clip_lo, clip_hi = mean - 3 * std, mean + 3 * std
    clipped = di_values[(di_values >= clip_lo) & (di_values <= clip_hi)]
    counts, bin_edges = np.histogram(clipped, bins=200)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    fig.add_trace(go.Bar(
        x=bin_centers, y=counts,
        width=(bin_edges[1] - bin_edges[0]),
        marker_color="rgba(102,126,234,0.6)",
    ))
    for mult, dash in [(0, "dot"), (1, "dash"), (2, "solid")]:
        for sign in [1, -1]:
            val = mean + sign * mult * std
            if mult == 0 and sign == -1:
                continue  # don't double-draw mean
            fig.add_vline(
                x=val, line_dash=dash,
                line_color="gray" if mult < 2 else "orange",
                annotation_text=f"{'+' if sign > 0 else '-'}{mult}\u03c3" if mult > 0 else "\u03bc",
                annotation_position="top left" if sign > 0 else "top right",
            )
    if mark_val is not None:
        fig.add_vline(
            x=mark_val, line_dash="dash", line_color="red",
            annotation_text=mark_label or "",
        )
    fig.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_title="DI value",
        yaxis_title="Count",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render(tab):
    with tab:
        fra_data = get_fra_data()
        cfg = get_fra_config()
        _has_fra = fra_data is not None

        st.subheader("Feature Resolved QK Circuit \u2014 Data-Independent")
        st.caption(
            "Analyse feature pairs through the QK circuit's inherent geometry. "
            "Select a pair manually, from data-dependent (DD) FRA rankings, or from "
            "a global data-independent (DI) scan, then compare DD and DI signals."
        )

        # RoPE detection -- drives delta slider visibility
        _model_name = cfg.get("model_name", "") if cfg else ""
        _sae_type = cfg.get("sae_type", "") if cfg else ""
        _is_rope = (
            "gemma" in _model_name
            or "llama" in _model_name.lower()
            or _sae_type == "crosscoder"
        )

        import pandas as pd

        # Read sidebar state
        preset_name = st.session_state.get("_sidebar_preset_name", "")
        preset = PRESETS.get(preset_name, {})
        sae_type = st.session_state.get("_sidebar_sae_type", cfg.get("sae_type", "") if cfg else "")
        head = st.session_state.get("_sidebar_head", cfg["head"] if cfg else 0)
        layer = st.session_state.get("_sidebar_layer", cfg["layer"] if cfg else 0)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        base_model_name = st.session_state.get("_sidebar_base_model_name", "")
        it_model_name = st.session_state.get("_sidebar_it_model_name", "")
        cc_it_arch = st.session_state.get("_sidebar_cc_it_arch", "")
        model_idx = st.session_state.get("_sidebar_model_idx", 0)
        crosscoder_repo_id = st.session_state.get("_sidebar_crosscoder_repo_id", "")
        cc_subfolder = st.session_state.get("_sidebar_cc_subfolder", "")
        crosscoder_layer = st.session_state.get("_sidebar_crosscoder_layer", 13)
        sae_hub_release = st.session_state.get("_sidebar_sae_hub_release", "")
        sae_hub_id = st.session_state.get("_sidebar_sae_hub_id", "")
        sae_local_path = st.session_state.get("_sidebar_sae_local_path", "")
        hf_token = st.session_state.get("_sidebar_hf_token", "")

        # Shared weight-loading kwargs
        _wt_kw = dict(
            base_model_name=base_model_name, it_model_name=it_model_name,
            cc_it_arch=cc_it_arch, model_idx=model_idx,
            crosscoder_repo_id=crosscoder_repo_id, cc_subfolder=cc_subfolder,
            crosscoder_layer=crosscoder_layer,
            sae_hub_release=sae_hub_release, sae_hub_id=sae_hub_id,
            sae_local_path=sae_local_path, layer=layer,
            model_name=preset.get("model", "gpt2-small"),
            hf_token=hf_token if sae_type == "sae_gemma" else "",
        )

        # -- Cached DI computation helpers --
        _di_cache_key = (sae_type, int(head), int(layer),
                         preset.get("model", ""), preset.get("repo_id", ""),
                         preset.get("release", ""), cc_subfolder)

        if st.session_state.get("_tab5_di_cache_key") != _di_cache_key:
            st.session_state["_tab5_di_weights"] = None
            st.session_state["_tab5_rope_params"] = None
            st.session_state["_tab5_di_row_cache"] = {}
            st.session_state["_tab5_dd_di_comparison"] = None
            st.session_state["_tab5_global_di"] = None
            st.session_state["_tab5_global_sample"] = None
            st.session_state["_tab5_di_cache_key"] = _di_cache_key
            st.session_state["_tab5_rope_delta"] = 0
            st.session_state.pop("_t5_man_confirmed", None)
            st.session_state.pop("_t5_dd_confirmed", None)

        for _k, _default in [
            ("_tab5_di_weights", None),
            ("_tab5_rope_params", None),
            ("_tab5_di_row_cache", {}),
            ("_tab5_dd_di_comparison", None),
            ("_tab5_global_di", None),
            ("_tab5_global_sample", None),
            ("_tab5_rope_delta", 0),
        ]:
            if _k not in st.session_state:
                st.session_state[_k] = _default

        # Convenience closures that capture current sidebar state
        def _ew():
            return _ensure_di_weights(sae_type, head, device, _wt_kw)

        def _gdr(query_id):
            _delta = st.session_state.get("_tab5_rope_delta", 0)
            return _get_di_row(query_id, sae_type, head, device, _wt_kw, delta=_delta)

        def _dfp(q, k):
            _delta = st.session_state.get("_tab5_rope_delta", 0)
            return _di_for_pair(q, k, sae_type, head, device, _wt_kw, delta=_delta)

        # Helper to compute pair metric with current agg mode
        _agg = cfg.get("agg_mode", "sum") if cfg else "sum"

        def _pair_metric(q, k, s, c, m):
            if _agg == "avg":
                return s / max(c, 1)
            if _agg == "max":
                return m
            return s

        # Reuse pairs from FRA if available
        pairs = []
        if _has_fra:
            _diagonal = False if cfg["filter_self"] else None
            pairs = rank_pairs(
                fra_data["indices_np"],
                fra_data["values_np"],
                top_k=cfg["top_k_pairs"],
                diagonal=_diagonal,
                mode=_agg,
            )

        # -- Section 1: Pair Selector + Metrics --
        st.markdown("### Pair selector")

        if _is_rope:
            def _on_delta_change():
                """Invalidate DI caches that depend on the delta value."""
                st.session_state["_tab5_di_row_cache"] = {}
                st.session_state["_tab5_dd_di_comparison"] = None
                st.session_state["_tab5_global_di"] = None
                st.session_state["_tab5_global_sample"] = None

            _rope_delta = st.slider(
                "RoPE relative position (delta)",
                min_value=0, max_value=20, value=st.session_state.get("_tab5_rope_delta", 0),
                key="_t5_rope_delta_slider",
                on_change=_on_delta_change,
                help=(
                    "Relative position offset (i − j) for the RoPE rotation matrix "
                    "W_R^(δ). At δ=0, DI scores are position-independent (no rotation). "
                    "Higher values show how feature interactions change at larger "
                    "query–key distances."
                ),
            )
            st.session_state["_tab5_rope_delta"] = _rope_delta
            if _rope_delta == 0:
                st.caption(
                    "δ=0: DI scores are position-independent (no RoPE rotation applied)."
                )
            else:
                st.caption(
                    f"δ={_rope_delta}: DI scores include the RoPE rotation W_R^({_rope_delta})."
                )

        _t5_pair_mode = st.radio(
            "Selection mode",
            ["Manual", "Top DD pairs", "Top DI pairs"],
            horizontal=True,
            key="_t5_pair_mode",
            help=(
                "**Top DD pairs** \u2014 the same FRA-ranked pairs as Tab 1, reusing the "
                "**Top-K pairs to display** sidebar setting and the current input's computation.\n\n"
                "**Top DI pairs** \u2014 ranked by data-independent QK score, scanning all feature "
                "pairs regardless of input. K is set by the slider that appears below."
            ),
        )

        _t5_q_sel = None
        _t5_k_sel = None

        if _t5_pair_mode == "Manual":
            _mc1, _mc2, _mc3 = st.columns([2, 2, 1])
            with _mc1:
                _t5_q_input = st.number_input(
                    "Query feature ID", min_value=0, value=0, key="_t5_man_q",
                )
            with _mc2:
                _t5_k_input = st.number_input(
                    "Key feature ID", min_value=0, value=0, key="_t5_man_k",
                )
            with _mc3:
                _t5_man_go = st.button("Go", key="_t5_man_go", use_container_width=True)
            if _t5_man_go:
                st.session_state["_t5_man_confirmed"] = (_t5_q_input, _t5_k_input)
            if "_t5_man_confirmed" in st.session_state:
                _t5_q_sel, _t5_k_sel = st.session_state["_t5_man_confirmed"]

        elif _t5_pair_mode == "Top DD pairs":
            if not _has_fra:
                st.info("Compute FRA first to use data-dependent pair ranking.")
            else:
                _t5_dd_pairs = pairs
                if not _t5_dd_pairs:
                    st.warning("No DD pairs found.")
                else:
                    _t5_dd_labels = [
                        f"F{q}\u2192F{k} (score={_pair_metric(q, k, s, c, m):.4f})"
                        for q, k, s, c, m in _t5_dd_pairs
                    ]
                    _dd_col_sel, _dd_col_btn = st.columns([4, 1])
                    with _dd_col_sel:
                        _t5_dd_idx = st.selectbox(
                            "Pick a DD pair", range(len(_t5_dd_pairs)),
                            format_func=lambda i: _t5_dd_labels[i],
                            key="_t5_dd_sel",
                        )
                    with _dd_col_btn:
                        st.markdown("")
                        st.markdown("")
                        _t5_dd_go = st.button("View", key="_t5_dd_go", use_container_width=True)
                    if _t5_dd_go:
                        st.session_state["_t5_dd_confirmed"] = (
                            _t5_dd_pairs[_t5_dd_idx][0], _t5_dd_pairs[_t5_dd_idx][1],
                        )
                    if "_t5_dd_confirmed" in st.session_state:
                        _t5_q_sel, _t5_k_sel = st.session_state["_t5_dd_confirmed"]

        else:  # Top DI pairs
            _t5_di_k = st.slider(
                "Top-K DI pairs", min_value=10, max_value=200, value=50,
                key="_t5_di_k",
            )

            # Invalidate cached result if k changed since last scan
            _gdi_cached = st.session_state["_tab5_global_di"]
            if _gdi_cached is not None and _gdi_cached.get("top_k_used") != _t5_di_k:
                st.session_state["_tab5_global_di"] = None

            _t5_di_go = st.button(
                "Compute global DI top-k", type="primary", key="_t5_di_go",
            )
            if _t5_di_go:
                with st.spinner("Running global DI scan..."):
                    W_dec, W_Q, W_K_ = _ew()
                    progress = st.progress(0, text="Scanning feature pairs...")

                    def _t5_di_progress(i, n):
                        progress.progress(i / n, text=f"Chunk {i}/{n}...")

                    result = compute_global_di_topk(
                        W_dec, W_Q, W_K_, top_k=_t5_di_k,
                        progress_callback=_t5_di_progress,
                        rope_params=st.session_state.get("_tab5_rope_params"),
                        delta=st.session_state.get("_tab5_rope_delta", 0),
                    )
                    result["top_k_used"] = _t5_di_k
                    progress.empty()
                    st.session_state["_tab5_global_di"] = result

            if st.session_state["_tab5_global_di"] is not None:
                _gdi = st.session_state["_tab5_global_di"]
                _gdi_labels = [
                    f"F{q}\u2192F{k} (DI={v:.4f})"
                    for q, k, v in zip(_gdi["query_ids"], _gdi["key_ids"], _gdi["di_values"])
                ]
                _gdi_idx = st.selectbox(
                    "Pick a DI pair", range(len(_gdi_labels)),
                    format_func=lambda i: _gdi_labels[i],
                    key="_t5_gdi_sel",
                )
                _t5_q_sel = int(_gdi["query_ids"][_gdi_idx])
                _t5_k_sel = int(_gdi["key_ids"][_gdi_idx])
            else:
                if _t5_pair_mode == "Top DI pairs":
                    st.caption("Click the button above to compute global DI pairs.")

        # -- Metrics + Heatmap for the selected pair --
        if _t5_q_sel is not None and _t5_k_sel is not None:
            st.markdown("---")
            st.markdown(f"#### Pair F{_t5_q_sel} \u2192 F{_t5_k_sel}")

            # Compute DI for this pair
            _t5_di_val = _dfp(_t5_q_sel, _t5_k_sel)
            _t5_di_row = _gdr(_t5_q_sel)

            # DI rank / percentile among ALL key features
            _t5_abs_row = np.abs(_t5_di_row)
            _t5_abs_val = abs(_t5_di_val)
            _t5_di_rank = int(np.sum(_t5_abs_row >= _t5_abs_val))  # 1-based rank
            _t5_di_pct = 100.0 * (1.0 - _t5_di_rank / len(_t5_abs_row))

            # DD score (only if FRA is available)
            _t5_dd_val = None
            if _has_fra:
                _t5_agg = cfg.get("agg_mode", "sum")
                for _pq, _pk, _ps, _pc, _pm in pairs:
                    if _pq == _t5_q_sel and _pk == _t5_k_sel:
                        _t5_dd_val = _pair_metric(_pq, _pk, _ps, _pc, _pm)
                        break

            # Metric cards
            _mc1, _mc2, _mc3, _mc4 = st.columns(4)
            _mc1.metric(
                "DD score",
                f"{_t5_dd_val:.4f}" if _t5_dd_val is not None else "N/A",
            )
            _mc2.metric("DI score", f"{_t5_di_val:.4f}")
            _mc3.metric(
                "F\u2096 rank by |DI|",
                f"{_t5_di_rank:,} / {len(_t5_di_row):,}",
                help=(
                    f"How many key features score higher |DI| than F{_t5_k_sel} "
                    f"when F{_t5_q_sel} is the query. "
                    f"Rank 1 means F{_t5_k_sel} is the single strongest key."
                ),
            )
            _mc4.metric("DI percentile", f"{_t5_di_pct:.1f}%")

            # -- Three DI distribution histograms --
            st.markdown("#### DI distributions")

            # --- Histogram 1: row distribution (F_q -> all keys) ---
            st.markdown(f"**Row distribution \u2014 F{_t5_q_sel} as query**")
            st.caption(
                f"DI(F{_t5_q_sel}, F\u2095) for every key feature F\u2095 in the dictionary "
                f"({len(_t5_di_row):,} values). "
                f"Red line marks the selected key F{_t5_k_sel}."
            )
            _t5_di_stats = {
                "mean": float(np.mean(_t5_di_row)),
                "std": float(np.std(_t5_di_row)),
            }
            _render_di_histogram(
                _t5_di_row, _t5_di_stats,
                mark_val=_t5_di_val,
                mark_label=f"F{_t5_q_sel}\u2192F{_t5_k_sel}",
            )

            # --- Histogram 2: global sample (all feature pairs) ---
            st.markdown("**Global distribution \u2014 all feature pairs (sampled)**")
            st.caption(
                "Approximate distribution of DI(F\u1d62, F\u2095) across all query\u2013key pairs, "
                "estimated from 500 random query rows. "
                "Red line marks the selected pair."
            )
            if st.session_state.get("_tab5_global_sample") is None:
                with st.spinner("Computing global DI sample\u2026"):
                    from fra.core.helpers import apply_rope_to_projected

                    _gs_W_dec, _gs_W_Q, _gs_W_K = _ew()
                    _gs_rng = np.random.default_rng(42)
                    _gs_idxs = _gs_rng.choice(
                        _gs_W_dec.shape[0],
                        size=min(500, _gs_W_dec.shape[0]),
                        replace=False,
                    )
                    _gs_Q = _gs_W_dec[torch.tensor(_gs_idxs, device=_gs_W_dec.device)] @ _gs_W_Q
                    _gs_K = _gs_W_dec @ _gs_W_K
                    _gs_delta = st.session_state.get("_tab5_rope_delta", 0)
                    _gs_rp = st.session_state.get("_tab5_rope_params")
                    if _gs_rp is not None and _gs_rp[0] is not None and _gs_delta > 0:
                        _gs_Q = apply_rope_to_projected(
                            _gs_Q, _gs_delta, *_gs_rp,
                        )
                        _gs_K = apply_rope_to_projected(
                            _gs_K, 0, *_gs_rp,
                        )
                    _gs_scale = math.sqrt(_gs_W_Q.shape[-1])
                    _gs_vals = ((_gs_Q @ _gs_K.T) / _gs_scale).detach().cpu().float().numpy().ravel()
                    st.session_state["_tab5_global_sample"] = _gs_vals
            _gs_sample = st.session_state["_tab5_global_sample"]
            _gs_stats = {
                "mean": float(np.mean(_gs_sample)),
                "std": float(np.std(_gs_sample)),
            }
            _render_di_histogram(
                _gs_sample, _gs_stats,
                mark_val=_t5_di_val,
                mark_label=f"F{_t5_q_sel}\u2192F{_t5_k_sel}",
            )

            # --- Histogram 3: DD pairs (top interacting features on input text) ---
            if _has_fra:
                st.markdown(f"**Input-specific distribution \u2014 top {len(pairs)} FRA pairs**")
                st.caption(
                    f"DI scores for the top {len(pairs)} feature pairs by "
                    f"{cfg.get('agg_mode', 'sum')} FRA strength on the current input. "
                    f"Red line marks the selected pair."
                )
                _dd_di_vals = np.array([_dfp(q, k) for q, k, *_ in pairs])
                _dd_stats = {
                    "mean": float(np.mean(_dd_di_vals)),
                    "std": float(np.std(_dd_di_vals)),
                }
                _pair_in_top = any(
                    q == _t5_q_sel and k == _t5_k_sel for q, k, *_ in pairs
                )
                if not _pair_in_top:
                    st.warning(
                        f"F{_t5_q_sel}\u2192F{_t5_k_sel} does not appear in the top "
                        f"{len(pairs)} FRA pairs for the current input \u2014 "
                        "red line shows where it would fall."
                    )
                _render_di_histogram(
                    _dd_di_vals, _dd_stats,
                    mark_val=_t5_di_val,
                    mark_label=f"F{_t5_q_sel}\u2192F{_t5_k_sel}",
                )
            else:
                st.info("Compute FRA on an input to see the input-specific DI distribution.")

        # -- Section 2: DD vs DI Comparison --
        st.markdown("---")
        st.markdown("### DD vs DI comparison")

        if not _has_fra:
            st.info(
                "Compute FRA first to compare data-dependent and data-independent "
                "rankings for the top pairs."
            )
        else:
            _t5_cmp_agg = cfg.get("agg_mode", "sum")
            _t5_cmp_n = 30
            _t5_cmp_pairs = rank_pairs(
                fra_data["indices_np"], fra_data["values_np"],
                top_k=_t5_cmp_n, diagonal=None, mode=_t5_cmp_agg,
            )

            _t5_cmp_go = st.button(
                f"Compute DI for top {len(_t5_cmp_pairs)} DD pairs",
                type="primary",
                key="_t5_cmp_go",
            )

            if _t5_cmp_go and _t5_cmp_pairs:
                _cmp_rows = []
                _cmp_progress = st.progress(0, text="Computing DI for DD pairs...")
                for _ci, (_cq, _ck, _cs, _cc, _cm) in enumerate(_t5_cmp_pairs):
                    _cdi = _dfp(_cq, _ck)
                    _cmp_rows.append({
                        "q": _cq, "k": _ck,
                        "dd": _pair_metric(_cq, _ck, _cs, _cc, _cm),
                        "di": _cdi,
                    })
                    _cmp_progress.progress(
                        (_ci + 1) / len(_t5_cmp_pairs),
                        text=f"Pair {_ci + 1}/{len(_t5_cmp_pairs)}...",
                    )
                _cmp_progress.empty()
                st.session_state["_tab5_dd_di_comparison"] = _cmp_rows

            if st.session_state["_tab5_dd_di_comparison"] is not None:
                _cmp = st.session_state["_tab5_dd_di_comparison"]

                # Scatter: DD score vs |DI| score
                _scatter_dd = [r["dd"] for r in _cmp]
                _scatter_di = [abs(r["di"]) for r in _cmp]
                _scatter_labels = [f"F{r['q']}\u2192F{r['k']}" for r in _cmp]

                # Determine which point is the selected pair (if any)
                _sel_mask = [
                    (_t5_q_sel is not None and r["q"] == _t5_q_sel
                     and _t5_k_sel is not None and r["k"] == _t5_k_sel)
                    for r in _cmp
                ]
                _scatter_colors = [
                    "red" if s else "rgba(102,126,234,0.7)" for s in _sel_mask
                ]
                _scatter_sizes = [12 if s else 7 for s in _sel_mask]

                _fig_scatter = go.Figure()
                _fig_scatter.add_trace(go.Scatter(
                    x=_scatter_dd,
                    y=_scatter_di,
                    mode="markers",
                    marker=dict(color=_scatter_colors, size=_scatter_sizes),
                    text=_scatter_labels,
                    hovertemplate="%{text}<br>DD: %{x:.4f}<br>|DI|: %{y:.4f}<extra></extra>",
                ))
                _fig_scatter.update_layout(
                    height=400,
                    margin=dict(l=0, r=0, t=30, b=0),
                    xaxis_title=f"DD score ({_t5_cmp_agg})",
                    yaxis_title="|DI| score",
                    showlegend=False,
                )
                st.plotly_chart(_fig_scatter, use_container_width=True, key="_t5_scatter")

                # Sortable table
                _tbl_rows = []
                for _ri, _r in enumerate(_cmp):
                    _is_sel = _sel_mask[_ri]
                    _q_str = f"\u2192 **F{_r['q']}**" if _is_sel else f"F{_r['q']}"
                    _k_str = f"\u2192 **F{_r['k']}**" if _is_sel else f"F{_r['k']}"
                    _tbl_rows.append({
                        "DD Rank": _ri + 1,
                        "Query": _q_str,
                        "Key": _k_str,
                        "DD Score": round(_r["dd"], 4),
                        "|DI| Score": round(abs(_r["di"]), 4),
                    })
                # Sort by DI rank (descending |DI|)
                _tbl_sorted = sorted(_tbl_rows, key=lambda x: x["|DI| Score"], reverse=True)
                for _di_rank, _row in enumerate(_tbl_sorted, 1):
                    _row["DI Rank"] = _di_rank
                # Restore DD rank order for display
                _tbl_sorted.sort(key=lambda x: x["DD Rank"])

                _tbl_df = pd.DataFrame(_tbl_sorted)[
                    ["DD Rank", "Query", "Key", "DD Score", "|DI| Score", "DI Rank"]
                ]
                st.dataframe(_tbl_df, use_container_width=True, hide_index=True)
