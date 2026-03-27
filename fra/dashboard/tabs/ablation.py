"""Tab 6 -- Ablation studies."""

import html as html_lib

import numpy as np
import streamlit as st
import torch

from fra.core.helpers import rank_pairs
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


def render(tab):
    with tab:
        fra_data = get_fra_data()
        if fra_data is None:
            st.info("Click **\u25b6 Compute FRA** in the sidebar to run ablation studies.")
            return

        cfg = get_fra_config()
        layer_ = cfg["layer"]
        head_ = cfg["head"]
        seq_len = fra_data["seq_len"]
        token_strs = fra_data["token_strs"][:seq_len]
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Read sidebar state
        sae_type = cfg.get("sae_type", "")
        base_model_name = st.session_state.get("_sidebar_base_model_name", "")
        it_model_name = st.session_state.get("_sidebar_it_model_name", "")
        cc_it_arch = st.session_state.get("_sidebar_cc_it_arch", "")
        model_idx = st.session_state.get("_sidebar_model_idx", 0)
        cc_subfolder = st.session_state.get("_sidebar_cc_subfolder", "")
        sae_hub_release = st.session_state.get("_sidebar_sae_hub_release", "")
        sae_hub_id = st.session_state.get("_sidebar_sae_hub_id", "")
        sae_local_path = st.session_state.get("_sidebar_sae_local_path", "")
        crosscoder_repo_id = st.session_state.get("_sidebar_crosscoder_repo_id", "")

        _agg = cfg.get("agg_mode", "sum")

        def _pair_metric(q, k, s, c, m):
            if _agg == "avg":
                return s / max(c, 1)
            if _agg == "max":
                return m
            return s

        st.subheader(f"Feature-Pair Ablation \u2014 L{layer_} H{head_}")
        st.caption(
            "Ablate selected feature pairs from the FRA tensor and measure the "
            "impact on model output. This reveals which cross-feature interactions "
            "are causally important to this attention head's computation."
        )

        # Build pair selection UI
        _abl_mode = cfg.get("agg_mode", "sum")
        all_pairs_for_ablation = rank_pairs(
            fra_data["indices_np"], fra_data["values_np"],
            top_k=100, diagonal=None, mode=_abl_mode,
        )

        if not all_pairs_for_ablation:
            st.warning("No feature pairs found.")
            return

        offdiag_list = [p for p in all_pairs_for_ablation if p[0] != p[1]]
        ondiag_list = [p for p in all_pairs_for_ablation if p[0] == p[1]]

        st.markdown(f"**{len(offdiag_list)}** off-diagonal pairs, "
                    f"**{len(ondiag_list)}** on-diagonal pairs in top 100.")

        abl_col1, abl_col2 = st.columns([1, 1])

        with abl_col1:
            n_ablate = st.slider(
                "Number of top pairs to ablate",
                min_value=1, max_value=min(50, len(offdiag_list) or 1),
                value=min(10, len(offdiag_list) or 1),
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

        with abl_col2:
            st.markdown("**Pairs to ablate:**")
            if abl_target.startswith("Top off"):
                _sel_pairs = offdiag_list[:n_ablate]
            elif abl_target.startswith("Top on"):
                _sel_pairs = ondiag_list[:n_ablate]
            else:
                import random as _rng_mod
                _rng = _rng_mod.Random(42)
                _sel_pairs = _rng.sample(
                    offdiag_list, min(n_ablate, len(offdiag_list)),
                )

            for _i, (q, k, s, cnt, mx) in enumerate(_sel_pairs[:15]):
                avg = s / max(cnt, 1)
                marker = "\u27f2" if q == k else "\u2192"
                st.text(f"  F{q} {marker} F{k}  (avg={avg:.4f}, sum={s:.4f})")
            if len(_sel_pairs) > 15:
                st.text(f"  ... and {len(_sel_pairs) - 15} more")

        run_abl = st.button("\u25b6  Run Ablation", type="primary")

        if run_abl:
            with st.spinner("Running ablation..."):
                import torch.nn.functional as _F
                from fra.analysis.ablation import (
                    ablate_fra_pairs,
                    reconstruct_scores,
                    run_condition,
                )
                from fra.core.helpers import fra_sum_to_attn

                # Rebuild sparse tensor from stored indices/values
                d_sae_val = fra_data["feat_acts_np"].shape[1]
                sp_indices = torch.tensor(
                    fra_data["indices_np"], dtype=torch.long,
                )
                sp_values = torch.tensor(
                    fra_data["values_np"], dtype=torch.float32,
                )
                sp_size = torch.Size(
                    [seq_len, seq_len, d_sae_val, d_sae_val],
                )
                fra_sparse = torch.sparse_coo_tensor(
                    sp_indices, sp_values, size=sp_size,
                ).coalesce()

                _sae_type = cfg.get("sae_type", sae_type)

                if _sae_type == "crosscoder":
                    # Crosscoder path: no bias correction needed
                    _cc_base, _cc_it = load_model_pair(
                        base_model_name, it_model_name, device,
                        cc_it_arch,
                    )
                    _cc_target = (
                        _cc_base if model_idx == 0 else _cc_it
                    )

                    # Use the exact tokens FRA was computed on (may include chat template)
                    _cc_tok_t = torch.tensor(
                        fra_data["tokens"],
                    ).unsqueeze(0).to(device)
                    _cc_shift = _cc_tok_t[0, 1:]

                    _cc_logits = _cc_target(_cc_tok_t)
                    _cc_unp_loss = _F.cross_entropy(
                        _cc_logits[0, :-1], _cc_shift,
                    ).item()

                    # Build bias dict with zero corrections
                    _cc_bias = {
                        "term_q": np.zeros(seq_len),
                        "term_k": np.zeros(seq_len),
                        "term_const": 0.0,
                        "attn_scale": 1.0,  # already scaled
                        "seq_len": seq_len,
                        "tok_tensor": _cc_tok_t,
                        "shift_labels": _cc_shift,
                        "unpatched_loss": _cc_unp_loss,
                        "unpatched_logits": _cc_logits,
                    }
                    _target_model = _cc_target
                    _bias = _cc_bias
                else:
                    # Single-SAE path
                    from fra.analysis.ablation import (
                        compute_bias_corrections,
                    )

                    _model_name = cfg.get("model_name", "gpt2-small")
                    _hf = cfg.get("hf_token", "")
                    _hook = cfg.get("hook_point", "attn.hook_z")

                    if _sae_type in ("sae_gemma", "gemma"):
                        _mdl = load_model_gemma(
                            _model_name, device, _hf,
                        )
                        _sae_obj = load_sae_gemma(
                            sae_hub_release, sae_hub_id, device,
                        )
                    elif _sae_type in ("sae_hub", "hub"):
                        _mdl = load_model(_model_name, device)
                        _sae_obj = load_sae_hub(
                            sae_hub_release, sae_hub_id, device,
                        )
                    else:
                        _mdl = load_model(_model_name, device)
                        _sae_obj = load_sae_local(
                            sae_local_path, int(layer_), device,
                        )

                    _bias = compute_bias_corrections(
                        _mdl, _sae_obj, cfg["text"],
                        layer_, head_, _hook,
                    )
                    if _bias is not None and _bias["seq_len"] != seq_len:
                        _bias["term_q"] = _bias["term_q"][:seq_len]
                        _bias["term_k"] = _bias["term_k"][:seq_len]
                        _bias["seq_len"] = seq_len
                        _bias["tok_tensor"] = _bias["tok_tensor"][
                            :, :seq_len
                        ]
                        _bias["shift_labels"] = _bias["tok_tensor"][
                            0, 1:
                        ]
                        _logits_trim = _mdl(_bias["tok_tensor"])
                        _bias["unpatched_loss"] = _F.cross_entropy(
                            _logits_trim[0, :-1],
                            _bias["shift_labels"],
                        ).item()
                        _bias["unpatched_logits"] = _logits_trim
                    _target_model = _mdl

                if _bias is None:
                    st.error("Text too short for ablation.")
                else:
                    # Full FRA scores (baseline)
                    fra_sum_full = fra_sum_to_attn(fra_sparse, seq_len)
                    scores_full = reconstruct_scores(
                        fra_sum_full, _bias, device,
                    )

                    # Ablated scores
                    pairs_to_abl = [
                        (int(p[0]), int(p[1])) for p in _sel_pairs
                    ]
                    fra_ablated = ablate_fra_pairs(
                        fra_sparse, pairs_to_abl, d_sae_val,
                    )
                    fra_sum_abl = fra_sum_to_attn(fra_ablated, seq_len)
                    scores_abl = reconstruct_scores(
                        fra_sum_abl, _bias, device,
                    )

                    # Zero scores
                    _mask_t = torch.triu(
                        torch.full(
                            (seq_len, seq_len), float("-inf"),
                            device=device,
                        ),
                        diagonal=1,
                    )
                    scores_zero = (
                        torch.zeros((seq_len, seq_len), device=device)
                        + _mask_t
                    )

                    _tok_t = _bias["tok_tensor"]
                    _shift = _bias["shift_labels"]
                    _unp_log = _bias["unpatched_logits"]

                    r_full = run_condition(
                        _target_model, layer_, head_,
                        _tok_t, _shift, scores_full, _unp_log,
                    )
                    r_abl = run_condition(
                        _target_model, layer_, head_,
                        _tok_t, _shift, scores_abl, _unp_log,
                    )
                    r_zero = run_condition(
                        _target_model, layer_, head_,
                        _tok_t, _shift, scores_zero, _unp_log,
                    )

                    # Display results
                    st.markdown("---")
                    st.subheader("Ablation Results")

                    hc = r_zero["loss"] - _bias["unpatched_loss"]
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    mc1.metric(
                        "Unpatched loss",
                        f"{_bias['unpatched_loss']:.4f}",
                    )
                    mc2.metric(
                        "FRA full loss",
                        f"{r_full['loss']:.4f}",
                        delta=f"{r_full['loss'] - _bias['unpatched_loss']:+.4f}",
                    )
                    mc3.metric(
                        "Ablated loss",
                        f"{r_abl['loss']:.4f}",
                        delta=f"{r_abl['loss'] - _bias['unpatched_loss']:+.4f}",
                    )
                    mc4.metric(
                        "Zero-ablated loss",
                        f"{r_zero['loss']:.4f}",
                        delta=f"{r_zero['loss'] - _bias['unpatched_loss']:+.4f}",
                    )

                    mc5, mc6, mc7 = st.columns(3)
                    mc5.metric(
                        "Ablation KL div",
                        f"{r_abl['kl_div']:.4f}",
                    )
                    mc6.metric(
                        "Top-1 changed",
                        f"{r_abl['top1_change_frac']*100:.1f}%",
                    )
                    if hc > 0.01:
                        rec_full = (
                            r_zero["loss"] - r_full["loss"]
                        ) / hc
                        rec_abl = (
                            r_zero["loss"] - r_abl["loss"]
                        ) / hc
                        mc7.metric(
                            "Recovery (full\u2192ablated)",
                            f"{rec_full:.3f} \u2192 {rec_abl:.3f}",
                            delta=f"{rec_abl - rec_full:+.3f}",
                        )

                    # Attention heatmap comparison
                    st.markdown("**Attention score comparison**")
                    _abl_ticks = list(range(seq_len))
                    _abl_labels = [
                        html_lib.escape(t) for t in token_strs
                    ]

                    def _abl_heatmap(scores_np, hover_label):
                        disp = scores_np.copy()
                        disp[np.triu_indices_from(disp, k=1)] = np.nan
                        return make_heatmap(
                            disp, _abl_ticks, _abl_ticks,
                            hover_label=hover_label, colorscale="RdBu", zmid=0,
                        )

                    hm1, hm2, hm3 = st.columns(3)
                    with hm1:
                        _show_heatmap(
                            _abl_heatmap(
                                scores_full.cpu().numpy(), "Score",
                            ),
                            _abl_ticks, _abl_labels, seq_len,
                            compact_height=350, key="abl_full",
                        )
                        st.caption("FRA Full")
                    with hm2:
                        _show_heatmap(
                            _abl_heatmap(
                                scores_abl.cpu().numpy(),
                                "Score",
                            ),
                            _abl_ticks, _abl_labels, seq_len,
                            compact_height=350, key="abl_after",
                        )
                        st.caption("After Ablation")
                    with hm3:
                        _diff_np = (
                            scores_abl.cpu().numpy()
                            - scores_full.cpu().numpy()
                        )
                        _show_heatmap(
                            _abl_heatmap(_diff_np, "Score"),
                            _abl_ticks, _abl_labels, seq_len,
                            compact_height=350, key="abl_diff",
                        )
                        st.caption("Difference (ablated \u2212 full)")
