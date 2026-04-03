"""Tab 4 -- Max-Act Examples."""

from pathlib import Path

import streamlit as st
import torch

from fra.dashboard.loaders import load_crosscoder, load_model_pair
from fra.dashboard.state import PRESETS, get_fra_data
from fra.dashboard.widgets import _render_prompt_card


def render(tab):
    with tab:
        from fra.analysis.max_act import (
            compute_max_acts as _compute_max_acts,
            list_available_features as _list_available_features,
            load_prompts as _load_prompts,
            load_reasoning_prompts as _load_reasoning_prompts,
            load_results as _load_results,
            save_results as _save_results,
        )

        # Read sidebar state needed by this tab
        cfg = st.session_state.get("fra_config", {})
        sae_type = cfg.get("sae_type", st.session_state.get("_sidebar_sae_type", ""))
        preset_name = st.session_state.get("_sidebar_preset_name", "")
        preset = PRESETS.get(preset_name, {})

        _is_reasoning_preset = (
            sae_type == "crosscoder"
            and preset.get("is_reasoning", False)
        ) if sae_type == "crosscoder" else False

        st.subheader("Max-Act Examples")
        if _is_reasoning_preset:
            st.caption(
                "Browse max-activating examples for crosscoder features across "
                "math problems with full R1 reasoning traces (OpenR1-Math-220k)."
            )
        else:
            st.caption(
                "Browse max-activating examples for crosscoder features across a corpus "
                "of safe and unsafe prompts."
            )

        _RESULTS_DIR = str(Path(__file__).parent.parent.parent.parent / "results")

        # --- Mode selector ---
        ma_mode = st.radio("Mode", ["Load existing", "Compute new"], horizontal=True)

        # Read crosscoder-specific sidebar vars from session state
        base_model_name = st.session_state.get("_sidebar_base_model_name", "")
        it_model_name = st.session_state.get("_sidebar_it_model_name", "")
        cc_it_arch = st.session_state.get("_sidebar_cc_it_arch", "")
        crosscoder_repo_id = st.session_state.get("_sidebar_crosscoder_repo_id", "")
        model_idx = st.session_state.get("_sidebar_model_idx", 0)
        cc_subfolder = st.session_state.get("_sidebar_cc_subfolder", "")
        crosscoder_layer = st.session_state.get("_sidebar_crosscoder_layer", 13)
        apply_chat_template = st.session_state.get("_sidebar_apply_chat_template", False)

        if ma_mode == "Compute new":
            # --- Compute mode ---
            ma_col_input, ma_col_btn = st.columns([3, 1])
            with ma_col_input:
                ma_feat_str = st.text_input(
                    "Feature IDs (comma-separated)",
                    value="53124",
                    help="e.g. 53124, 24613",
                )
                ma_n_prompts = st.number_input(
                    "Number of prompts",
                    min_value=10, max_value=5000, value=200, step=10,
                )
            with ma_col_btn:
                st.markdown("")  # spacing
                st.markdown("")
                ma_run = st.button("Run", type="primary", use_container_width=True)

            if ma_run and sae_type != "crosscoder":
                st.error("Max-act computation requires a crosscoder preset (select one in the sidebar).")
                ma_run = False

            if ma_run:
                # Parse feature IDs
                try:
                    ma_feature_ids = [int(x.strip()) for x in ma_feat_str.split(",") if x.strip()]
                except ValueError:
                    st.error("Invalid feature IDs. Enter comma-separated integers.")
                    ma_feature_ids = []

                if ma_feature_ids:
                    ma_device = "cuda" if torch.cuda.is_available() else "cpu"

                    with st.status("Loading prompts...", expanded=True) as status:
                        if _is_reasoning_preset:
                            prompts = _load_reasoning_prompts(ma_n_prompts)
                        else:
                            prompts = _load_prompts(ma_n_prompts)
                        status.update(label=f"Loaded {len(prompts)} prompts. Loading models...")

                        base_model, it_model = load_model_pair(
                            base_model_name, it_model_name, ma_device, cc_it_arch,
                        )
                        cc = load_crosscoder(
                            crosscoder_repo_id, model_idx, ma_device, cc_subfolder,
                        )
                        status.update(label="Running prompts...")

                    progress = st.progress(0, text="Processing prompts...")

                    def _ma_progress(i, n):
                        progress.progress((i + 1) / n, text=f"Processing prompt {i+1}/{n}...")

                    results = _compute_max_acts(
                        base_model, it_model, cc, ma_feature_ids, prompts,
                        apply_template=apply_chat_template,
                        crosscoder_layer=crosscoder_layer, device=ma_device,
                        progress_callback=_ma_progress,
                    )
                    progress.empty()

                    # Save to disk and session state
                    _ma_dataset = "OpenR1-Math-220k" if _is_reasoning_preset else "BeaverTails+UltraChat"
                    for fid in ma_feature_ids:
                        _save_results(fid, results[fid], len(results[fid]),
                                      _ma_dataset, _RESULTS_DIR)

                    st.session_state["max_act_results"] = results
                    st.session_state["max_act_feature_ids"] = ma_feature_ids
                    st.success(f"Computed and saved results for {len(ma_feature_ids)} feature(s).")

        else:
            # --- Load mode ---
            available = _list_available_features(_RESULTS_DIR)
            if not available:
                st.info(f"No pre-computed results found in `{_RESULTS_DIR}/`.")
            else:
                ma_load_fid = st.selectbox(
                    "Feature",
                    available,
                    format_func=lambda fid: f"F{fid}",
                )
                if ma_load_fid is not None:
                    loaded = _load_results(ma_load_fid, _RESULTS_DIR)
                    if loaded:
                        st.session_state["max_act_results"] = {ma_load_fid: loaded["prompts"]}
                        st.session_state["max_act_feature_ids"] = [ma_load_fid]

        # --- Results display (shared) ---
        if "max_act_results" in st.session_state and st.session_state.get("max_act_feature_ids"):
            ma_results = st.session_state["max_act_results"]
            ma_fids = st.session_state["max_act_feature_ids"]

            # Feature selector if multiple
            if len(ma_fids) > 1:
                ma_display_fid = st.selectbox(
                    "Display feature",
                    ma_fids,
                    format_func=lambda fid: f"F{fid}",
                    key="ma_display_fid",
                )
            else:
                ma_display_fid = ma_fids[0]

            entries = ma_results.get(ma_display_fid, [])
            if not entries:
                st.warning(f"No entries for feature {ma_display_fid}.")
            else:
                # Filter controls (rendered before metrics so BOS toggle affects them)
                _fc1, _fc2 = st.columns([2, 1])
                with _fc1:
                    ma_filter = st.radio(
                        "Filter",
                        ["All", "Safe only", "Unsafe only"],
                        horizontal=True,
                        key="ma_filter",
                    )
                with _fc2:
                    ma_ignore_bos = st.toggle(
                        "Ignore BOS",
                        value=False,
                        key="ma_ignore_bos",
                        help="Exclude the BOS (beginning-of-sequence) token from "
                             "activation stats and sorting. BOS often dominates "
                             "max activation, masking the real signal.",
                    )

                # Apply BOS filtering: strip position-0 token and recalculate stats
                if ma_ignore_bos:
                    adjusted = []
                    for e in entries:
                        t_strs = e.get("token_strs")
                        t_acts = e.get("token_acts")
                        if t_strs and t_acts and len(t_strs) > 1:
                            new_acts = t_acts[1:]
                            active_vals = [v for v in new_acts if v > 0]
                            new_max = max(active_vals) if active_vals else 0.0
                            new_mean = sum(active_vals) / len(active_vals) if active_vals else 0.0
                            new_top = [
                                t for t in e.get("top_tokens", []) if t["pos"] != 0
                            ]
                            adjusted.append({
                                **e,
                                "token_strs": t_strs[1:],
                                "token_acts": new_acts,
                                "max_act": new_max,
                                "mean_act": new_mean,
                                "n_active_tokens": len(active_vals),
                                "n_tokens": len(new_acts),
                                "top_tokens": new_top,
                            })
                        else:
                            adjusted.append(e)
                    entries = sorted(adjusted, key=lambda x: x["max_act"], reverse=True)

                # Summary metrics
                total = len(entries)
                active = [e for e in entries if e["max_act"] > 0]
                safe_entries = [e for e in entries if e.get("is_safe") is True]
                unsafe_entries = [e for e in entries if e.get("is_safe") is False]
                safe_active = [e for e in active if e.get("is_safe") is True]
                unsafe_active = [e for e in active if e.get("is_safe") is False]

                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Total prompts", total)
                mc2.metric("Active", f"{len(active)}/{total}")
                mc3.metric("Safe active", f"{len(safe_active)}/{len(safe_entries)}" if safe_entries else "N/A")
                mc4.metric("Unsafe active", f"{len(unsafe_active)}/{len(unsafe_entries)}" if unsafe_entries else "N/A")

                st.markdown("---")

                if ma_filter == "Safe only":
                    display_entries = [e for e in entries if e.get("is_safe") is True]
                elif ma_filter == "Unsafe only":
                    display_entries = [e for e in entries if e.get("is_safe") is False]
                else:
                    display_entries = entries

                # Pagination
                if "ma_page_size" not in st.session_state:
                    st.session_state["ma_page_size"] = 15
                page_size = st.session_state["ma_page_size"]
                page_entries = display_entries[:page_size]

                st.caption(f"Showing {len(page_entries)} of {len(display_entries)} prompts "
                           f"(sorted by max activation)")

                # Render prompt cards
                for idx, entry in enumerate(page_entries):
                    _render_prompt_card(entry, ma_display_fid, idx)

                # Load more button
                if page_size < len(display_entries):
                    if st.button("Load more (+10)"):
                        st.session_state["ma_page_size"] = page_size + 10
                        st.rerun()
