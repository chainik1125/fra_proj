"""
FRA Dashboard -- Feature-Resolved Attention interactive viewer.

Run with:
    streamlit run fra/dashboard/app.py
"""

import html as html_lib
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import torch

from fra.dashboard.state import PRESETS
from fra.dashboard.loaders import (
    load_crosscoder,
    load_model,
    load_model_gemma,
    load_model_pair,
    load_sae_gemma,
    load_sae_hub,
    load_sae_local,
    load_lora_model,
    load_wandb_crosscoder,
)
from fra.core.helpers import aggregate_pairs, rank_pairs
from fra.dashboard.compute import (
    build_fra_head,
    build_fra_head_multilayer,
    encode_fra,
    encode_fra_model_diff,
    encode_fra_multilayer,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FRA Dashboard",
    page_icon="\U0001f9e0",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Configuration")

    text = st.text_area(
        "Input text",
        value=(
            "The cat sat on the mat. "
            "The cat was happy. "
            "A dog lay on the rug. "
            "The dog was tired."
        ),
        height=130,
    )

    st.subheader("Model & SAE")

    preset_name = st.selectbox("Preset", list(PRESETS.keys()))
    preset = PRESETS[preset_name]
    sae_type = preset["type"]
    supports_neuronpedia = preset.get("supports_neuronpedia", False)
    hook_point = preset.get("hook_point", "")

    # -- Layer selector ----------------------------------------------------
    if sae_type == "crosscoder_multilayer":
        available_layers = preset["layers"]
        layer = st.selectbox(
            "Attention layer",
            available_layers,
            index=available_layers.index(preset["default_layer"]),
        )
        # crosscoder-only defaults (not used for multilayer)
        crosscoder_layer = 0
        cc_subfolder = ""
        cc_it_arch = ""
    elif sae_type == "crosscoder":
        available_layers = sorted(preset["layers"].keys())
        crosscoder_layer = st.selectbox(
            "Crosscoder layer",
            available_layers,
            index=available_layers.index(preset["default_layer"]),
        )
        cc_subfolder = preset["layers"][crosscoder_layer]
        cc_it_arch = preset.get("it_arch_name", "")
        layer = crosscoder_layer + 1
        st.caption(f"Attention layer: **{layer}** (crosscoder layer + 1)")
    else:
        layer = st.number_input(
            "Layer",
            min_value=min(preset["layers"]),
            max_value=max(preset["layers"]),
            value=preset["default_layer"],
        )
        # defaults for crosscoder-only variables
        crosscoder_layer = 13
        cc_subfolder = ""
        cc_it_arch = ""

    # -- Head selector -----------------------------------------------------
    head = st.number_input("Head", 0, preset["n_heads"] - 1, value=0)
    n_heads = preset["n_heads"]
    compute_all_heads = st.checkbox(
        f"Compute all {n_heads} heads",
        value=False,
        help=(
            "Compute FRA for every head in this layer. "
            "Enables per-tab head selection and all-heads patching/ablation."
        ),
    )

    # -- Type-specific extras ----------------------------------------------
    # Multi-layer crosscoder extras
    if sae_type == "crosscoder_multilayer":
        base_model_name = preset["base_model"]
        base_lora = preset["base_lora"]
        sleeper_lora = preset["sleeper_lora"]

        st.caption(f"Base: `{base_model_name}`")
        st.caption(f"Base LoRA: `{base_lora}`")
        st.caption(f"Sleeper LoRA: `{sleeper_lora}`")

        model_idx = st.radio(
            "Analyse attention of",
            [0, 1],
            format_func=lambda i: preset["model_labels"][i],
            horizontal=True,
        )

        st.caption(preset.get("ram_note", ""))

        # not used for multilayer path
        crosscoder_repo_id = ""
        it_model_name = ""
        apply_chat_template = False
        reasoning_trace = ""
        reasoning_response = ""
        sae_hub_release = ""
        sae_hub_id = ""
        sae_local_path = ""
        hf_token = ""
        chunk_size = 16

    # Single-layer crosscoder extras
    elif sae_type == "crosscoder":
        crosscoder_repo_id = preset["repo_id"]
        base_model_name = preset["base_model"]
        it_model_name = preset["it_model"]

        st.caption(f"Base: `{base_model_name}`")
        st.caption(f"Target: `{it_model_name}`")

        model_idx = st.radio(
            "Analyse attention of",
            [0, 1],
            format_func=lambda i: preset["model_labels"][i],
            horizontal=True,
        )

        apply_chat_template = st.checkbox(
            "Apply chat template",
            value=True,
            help=(
                "Wrap the input text using the target model's "
                "tokenizer.apply_chat_template(). Required for "
                "safety / reasoning features to activate."
            ),
        )

        is_reasoning = preset.get("is_reasoning", False)
        reasoning_trace = ""
        reasoning_response = ""
        if is_reasoning and apply_chat_template:
            reasoning_trace = st.text_area(
                "Reasoning trace",
                value="",
                height=100,
                help=(
                    "Optional. The model's chain-of-thought. "
                    "The chat template already adds <think>; this "
                    "text goes inside the thinking block. "
                    "Leave blank to analyse only the user prompt."
                ),
            )
            reasoning_response = st.text_area(
                "Response",
                value="",
                height=100,
                help=(
                    "Optional. The model's final response "
                    "after </think>. Requires a reasoning trace."
                ),
            )

        st.caption(preset["ram_note"])

        # not used for crosscoder path
        sae_hub_release = ""
        sae_hub_id = ""
        sae_local_path = ""
        hf_token = ""
        chunk_size = 16
    else:
        # defaults for crosscoder-only variables
        crosscoder_repo_id = ""
        base_model_name = ""
        it_model_name = ""
        model_idx = 0
        apply_chat_template = False
        reasoning_trace = ""
        reasoning_response = ""

        if sae_type == "sae_hub":
            sae_hub_release = preset["release"]
            sae_hub_id = preset["sae_id_template"].format(layer=layer)
            sae_local_path = ""
            hf_token = ""
            chunk_size = 16
        elif sae_type == "sae_local":
            sae_hub_release = ""
            sae_hub_id = ""
            default_local = str(
                Path(__file__).parent.parent.parent / preset["checkpoint_path"].lstrip("./")
            )
            sae_local_path = st.text_input("Checkpoint path", value=default_local)
            if not Path(sae_local_path).exists():
                st.warning("Checkpoint not found. Train with `python train_sae.py`.")
            hf_token = ""
            chunk_size = 16
        elif sae_type == "sae_gemma":
            sae_hub_release = preset["release"]
            sae_layer = layer - 1
            sae_hub_id = preset["sae_id_template"].format(layer=sae_layer)
            sae_local_path = ""
            st.caption(
                f"SAE trained on `resid_post[{sae_layer}]` "
                f"→ activations from `resid_pre[{layer}]`"
            )
            if preset.get("hf_token_required"):
                hf_token = st.text_input(
                    "HuggingFace token",
                    type="password",
                    help="Required to download Gemma model weights.",
                )
            else:
                hf_token = ""
            chunk_size = st.slider(
                "Chunk size",
                min_value=1,
                max_value=32,
                value=preset.get("chunk_size_default", 1),
                help="Batch chunk size for FRA computation. Lower = less VRAM.",
            )

    # -- Ranking mode ------------------------------------------------------
    _RANK_LABELS = {"sum": "Sum |FRA|", "avg": "Avg |FRA|", "max": "Max |FRA|"}
    agg_mode = st.radio(
        "Rank feature pairs by:",
        list(_RANK_LABELS.keys()),
        format_func=lambda k: _RANK_LABELS[k],
        horizontal=True,
    )

    if compute_all_heads:
        _MH_LABELS = {
            "sum": "Sum across heads",
            "max": "Max across heads",
            "avg": "Avg across heads",
            "min": "Min across heads",
            "count": "Count (# heads)",
        }
        multi_head_agg = st.radio(
            "Rank pairs across heads by:",
            list(_MH_LABELS.keys()),
            format_func=lambda k: _MH_LABELS[k],
            horizontal=True,
        )
    else:
        multi_head_agg = None

    # -- Common compute settings -------------------------------------------
    st.subheader("Compute settings")
    use_all_features = st.checkbox(
        "Use ALL features (debug)",
        value=False,
        help="Skip top-k truncation and use every active SAE feature. "
             "Isolates reconstruction error from feature selection error. "
             "Very memory/compute heavy!",
    )
    top_k_feat = st.slider(
        "Top-K features / position", 5, 50, 20,
        disabled=use_all_features,
    )
    top_k_pairs = st.slider("Top-K pairs to display", 10, 100, 30)
    filter_self = st.checkbox("Filter self-interactions (q==k)", value=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.caption(f"Device: {device}")

    compute_btn = st.button("\u25b6  Compute FRA", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Store sidebar variables in session state for tabs
# ---------------------------------------------------------------------------

st.session_state["_sidebar_preset_name"] = preset_name
st.session_state["_sidebar_sae_type"] = sae_type
st.session_state["_sidebar_head"] = head
st.session_state["_sidebar_layer"] = layer
st.session_state["_sidebar_base_model_name"] = base_model_name
st.session_state["_sidebar_it_model_name"] = it_model_name
st.session_state["_sidebar_cc_it_arch"] = cc_it_arch
st.session_state["_sidebar_model_idx"] = model_idx
st.session_state["_sidebar_crosscoder_repo_id"] = crosscoder_repo_id
st.session_state["_sidebar_cc_subfolder"] = cc_subfolder
st.session_state["_sidebar_crosscoder_layer"] = crosscoder_layer
st.session_state["_sidebar_sae_hub_release"] = sae_hub_release
st.session_state["_sidebar_sae_hub_id"] = sae_hub_id
st.session_state["_sidebar_sae_local_path"] = sae_local_path
st.session_state["_sidebar_hf_token"] = hf_token
st.session_state["_sidebar_apply_chat_template"] = apply_chat_template
st.session_state["_sidebar_compute_all_heads"] = compute_all_heads
st.session_state["_sidebar_multi_head_agg"] = multi_head_agg

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("\U0001f9e0 Feature-Resolved Attention Dashboard")
st.caption("Decomposing attention through SAE feature space.")

# ---------------------------------------------------------------------------
# Trigger computation
# ---------------------------------------------------------------------------

if compute_btn:
    # Determine which heads to compute
    heads_to_compute = list(range(n_heads)) if compute_all_heads else [int(head)]

    # -- Tokenisation (shared across heads) --------------------------------
    fra_tokens = None  # only used for crosscoder/multilayer paths
    if sae_type == "crosscoder_multilayer":
        _lora = base_lora if model_idx == 0 else sleeper_lora
        with st.spinner("Loading model & crosscoder\u2026"):
            _ml_model = load_lora_model(base_model_name, _lora, device)
            _ml_coder = load_wandb_crosscoder(
                preset["crosscoder_name"], preset["wandb_download_dir"],
                int(model_idx), device,
            )
        fra_tokens = _ml_model.tokenizer.encode(text)
    elif sae_type == "crosscoder":
        with st.spinner("Loading models & crosscoder\u2026"):
            load_model_pair(base_model_name, it_model_name, device, cc_it_arch)
            load_crosscoder(crosscoder_repo_id, model_idx, device, cc_subfolder)

        base_model, it_model = load_model_pair(
            base_model_name, it_model_name, device, cc_it_arch,
        )
        target_model = base_model if model_idx == 0 else it_model
        if apply_chat_template:
            if reasoning_trace.strip() or reasoning_response.strip():
                prefix = it_model.tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                full_text = prefix + reasoning_trace.strip()
                if reasoning_response.strip():
                    full_text += "\n</think>\n" + reasoning_response.strip()
                fra_tokens = it_model.tokenizer.encode(
                    full_text, add_special_tokens=False,
                )
            else:
                fra_tokens = it_model.tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    tokenize=True,
                    add_generation_prompt=True,
                )
            with st.expander("Templated input"):
                st.code(it_model.tokenizer.decode(fra_tokens))
        else:
            fra_tokens = target_model.tokenizer.encode(text)
    else:
        _run_sae_type = {"sae_hub": "hub", "sae_local": "local", "sae_gemma": "gemma"}[sae_type]
        _run_model = preset.get("model", "gpt2-small")

        with st.spinner("Loading model & SAE\u2026"):
            if sae_type == "sae_gemma":
                load_model_gemma(_run_model, device, hf_token)
                load_sae_gemma(sae_hub_release, sae_hub_id, device)
            else:
                load_model(_run_model, device)
                if sae_type == "sae_hub":
                    load_sae_hub(sae_hub_release, sae_hub_id, device)
                elif sae_local_path and Path(sae_local_path).exists():
                    load_sae_local(sae_local_path, int(layer), device)

    # -- Encode once, build per head ----------------------------------------
    _top_k = None if use_all_features else top_k_feat
    fra_data_all = {}
    _progress = st.progress(0, text="Encoding\u2026") if compute_all_heads else None

    if sae_type == "crosscoder_multilayer":
        _encoded_per_layer, _attn_caches, _model, _tokens = (
            encode_fra_multilayer(
                tokens=fra_tokens,
                attn_layers=[int(layer)],
                coder=_ml_coder,
                model=_ml_model,
                device=device,
            )
        )
        _encoded = _encoded_per_layer[int(layer)]
        _attn_cache = _attn_caches[int(layer)]
        _attn_layer = int(layer)
    elif sae_type == "crosscoder":
        _encoded, _attn_cache, _model, _tokens, _attn_layer = (
            encode_fra_model_diff(
                tokens=fra_tokens,
                crosscoder_layer=int(crosscoder_layer),
                crosscoder_repo_id=crosscoder_repo_id,
                model_idx=int(model_idx),
                base_model_name=base_model_name,
                it_model_name=it_model_name,
                device=device,
                subfolder=cc_subfolder,
                it_arch_name=cc_it_arch,
            )
        )
    else:
        _encoded, _attn_cache, _model, _tokens = encode_fra(
            text=text,
            layer=int(layer),
            hook_point=hook_point,
            sae_type=_run_sae_type,
            sae_hub_release=sae_hub_release,
            sae_hub_id=sae_hub_id,
            sae_local_path=sae_local_path,
            device=device,
            model_name=_run_model,
            hf_token=hf_token,
            include_special_tokens=True,
        )
        _attn_layer = int(layer)

    # Pre-compute head-independent artifacts once
    from fra.core.fra import topk_sparsify
    _topk_features = topk_sparsify(
        _encoded["feature_activations"], _top_k,
    ).float()
    _rms = None
    if _encoded["rms_activations"] is not None:
        _eps = _model.cfg.eps
        _rms = (
            _encoded["rms_activations"].float().pow(2).mean(dim=-1) + _eps
        ).sqrt()
    _token_strs = [_model.tokenizer.decode([t]) for t in _tokens[:128]]
    _softcap = getattr(_model.cfg, "attn_scores_soft_cap", 0.0) or 0.0

    for _i, _h in enumerate(heads_to_compute):
        if _progress is not None:
            _progress.progress(
                _i / len(heads_to_compute),
                text=f"Computing FRA for head {_h + 1}/{n_heads}\u2026",
            )
        fra_data_all[_h] = build_fra_head(
            _encoded, _attn_cache, _model, _attn_layer, int(_h),
            _tokens, device,
            top_k=_top_k, chunk_size=chunk_size,
            topk_features=_topk_features, rms=_rms,
            token_strs=_token_strs, softcap=_softcap,
        )

    if _progress is not None:
        _progress.progress(1.0, text="Done!")

    # Primary head's data for backward compatibility
    fra_data = fra_data_all[int(head)]
    st.session_state["fra_data"] = fra_data
    st.session_state["fra_data_all"] = fra_data_all if compute_all_heads else None

    # Clear cached reconstruction tab results from previous runs
    st.session_state.pop("_recon_metrics", None)
    st.session_state.pop("_val_loss_all", None)
    # Clear all per-head caches (projection + loss)
    for _k in [k for k in st.session_state
               if k.startswith("_projection_cache_") or k.startswith("_val_loss_")]:
        del st.session_state[_k]
    st.session_state["fra_config"] = {
        "layer": int(layer),
        "head": int(head),
        "text": text,
        "supports_neuronpedia": supports_neuronpedia,
        "filter_self": filter_self,
        "top_k_pairs": top_k_pairs,
        "agg_mode": agg_mode,
        "sae_type": sae_type,
        "model_name": preset.get("model", "gpt2-small"),
        "hf_token": hf_token if sae_type == "sae_gemma" else "",
        "hook_point": preset.get("hook_point", ""),
        "trained_on_bos": preset.get("trained_on_bos", True),
        "n_heads": n_heads,
        "compute_all_heads": compute_all_heads,
        "multi_head_agg": multi_head_agg,
        "is_multi_layer": preset.get("is_multi_layer", False),
    }
    _total = sum(d["total_interactions"] for d in fra_data_all.values())
    _head_label = f"all {n_heads} heads" if compute_all_heads else f"head {head}"
    st.success(
        f"Done \u2014 {_total:,} non-zero interactions found ({_head_label})."
    )

# ---------------------------------------------------------------------------
# Main results area
# ---------------------------------------------------------------------------

_has_fra = "fra_data" in st.session_state

# Summary row + tokenised text (only when FRA has been computed)
if _has_fra:
    fra_data = st.session_state["fra_data"]
    cfg = st.session_state["fra_config"]
    layer_ = cfg["layer"]
    head_ = cfg["head"]
    seq_len = fra_data["seq_len"]
    token_strs = fra_data["token_strs"][:seq_len]

    # Recompute pairs (filter / top_k / ranking may change without recomputing FRA)
    _agg = cfg.get("agg_mode", "sum")
    _diagonal = False if cfg["filter_self"] else None
    _mh_agg = cfg.get("multi_head_agg")
    _fra_data_all = st.session_state.get("fra_data_all")

    if _fra_data_all is not None and _mh_agg is not None:
        _idx_dict = {h: d["indices_np"] for h, d in _fra_data_all.items()}
        _val_dict = {h: d["values_np"] for h, d in _fra_data_all.items()}
        pairs = rank_pairs(
            _idx_dict, _val_dict,
            top_k=cfg["top_k_pairs"],
            diagonal=_diagonal,
            mode=_agg,
            multi_head_agg=_mh_agg,
        )
        # Bottom pairs: use same multi-head aggregation, but sort ascending
        _all_mh = rank_pairs(
            _idx_dict, _val_dict,
            top_k=0,  # 0 = return all
            diagonal=_diagonal,
            mode=_agg,
            multi_head_agg=_mh_agg,
        )
        _all_mh.sort(key=lambda x: x[2])
        bottom_pairs = _all_mh[:cfg["top_k_pairs"]]
    else:
        pairs = rank_pairs(
            fra_data["indices_np"],
            fra_data["values_np"],
            top_k=cfg["top_k_pairs"],
            diagonal=_diagonal,
            mode=_agg,
        )
        # Bottom pairs: weakest by sum
        _all_pairs_for_bottom = aggregate_pairs(fra_data["indices_np"], fra_data["values_np"], diagonal=_diagonal)
        _all_pairs_for_bottom.sort(key=lambda x: x[2])
        bottom_pairs = _all_pairs_for_bottom[:cfg["top_k_pairs"]]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tokens", seq_len)
    if _fra_data_all is not None:
        _total_nnz = sum(d["total_interactions"] for d in _fra_data_all.values())
        c2.metric("Non-zero interactions", f"{_total_nnz:,}",
                  help=f"Summed across all {len(_fra_data_all)} heads")
    else:
        c2.metric("Non-zero interactions", f"{fra_data['total_interactions']:,}")
    if _fra_data_all is not None:
        _all_unique = set()
        for d in _fra_data_all.values():
            _all_unique.update(
                (int(t[0]), int(t[1]))
                for t in aggregate_pairs(d["indices_np"], d["values_np"], diagonal=_diagonal)
            )
        total_unique = len(_all_unique)
        c3.metric("Unique feature pairs", f"{total_unique:,}",
                  help=f"Union across all {len(_fra_data_all)} heads")
    else:
        total_unique = len(aggregate_pairs(fra_data["indices_np"], fra_data["values_np"], diagonal=_diagonal))
        c3.metric("Unique feature pairs", f"{total_unique:,}")
    _head_str = "all" if _fra_data_all is not None else f"H{head_}"
    c4.metric("Layer / Head", f"L{layer_} / {_head_str}")

    tok_html = " ".join(
        f'<span style="background:#e9ecef;padding:2px 5px;border-radius:3px;'
        f'font-family:monospace;font-size:0.9em;">{html_lib.escape(t)}</span>'
        for t in token_strs
    )
    st.markdown(tok_html, unsafe_allow_html=True)
    st.markdown("")

    _metric_label = {"sum": "sum", "avg": "avg", "max": "max"}.get(_agg, "sum")

    def _pair_metric(q, k, s, c, m):
        if _agg == "avg":
            return s / max(c, 1)
        if _agg == "max":
            return m
        return s

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "\U0001f4ca Top Interactions",
    "\U0001f525 Feature Matrix",
    "\u2705 Reconstruction",
    "\U0001f9e9 Max-Act Examples",
    "\U0001f517 Data-Independent",
    "\u2702 Ablation",
])

from fra.dashboard.tabs import ablation, interactions, matrix, reconstruction, max_act, di

interactions.render(tab1)
matrix.render(tab2)
reconstruction.render(tab3)
max_act.render(tab4)
di.render(tab5)
ablation.render(tab6)

# ---------------------------------------------------------------------------
# Persist active tab across reruns via URL query param + JS injection.
# Tab clicks are pure frontend in Streamlit (no rerun), so this is loop-free:
#   user clicks tab i → JS writes ?tab=i to URL (no rerun)
#   widget fires rerun → Python reads ?tab=i → JS clicks tab i back
# ---------------------------------------------------------------------------
_active_tab = int(st.query_params.get("tab", 0))
components.html(
    f"""
    <script>
    (function() {{
        const TARGET = {_active_tab};
        function restore() {{
            const btns = window.parent.document.querySelectorAll('button[role="tab"]');
            if (!btns.length) {{ setTimeout(restore, 50); return; }}
            if (btns[TARGET] && btns[TARGET].getAttribute('aria-selected') !== 'true') {{
                btns[TARGET].click();
            }}
            btns.forEach((btn, i) => {{
                if (!btn.dataset.fraTabWired) {{
                    btn.dataset.fraTabWired = '1';
                    btn.addEventListener('click', () => {{
                        const url = new URL(window.parent.location.href);
                        url.searchParams.set('tab', i);
                        window.parent.history.replaceState(null, '', url.toString());
                    }});
                }}
            }});
        }}
        restore();
    }})();
    </script>
    """,
    height=0,
)
