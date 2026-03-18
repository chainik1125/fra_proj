"""
FRA Dashboard — Feature-Resolved Attention interactive viewer.

Run with:
    streamlit run fra/streamlit_app.py
"""

import html as html_lib
import math
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
import requests
import streamlit as st
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FRA Dashboard",
    page_icon="🧠",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Cached resource loaders (persist across reruns, keyed by args)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_model(model_name: str, device: str):
    from transformer_lens import HookedTransformer
    torch.set_grad_enabled(False)
    return HookedTransformer.from_pretrained(model_name, device=device)


@st.cache_resource(show_spinner=False)
def load_sae_hub(release: str, sae_id: str, device: str):
    from fra.sae_lens_wrapper import SAELensAttentionSAE
    return SAELensAttentionSAE(release, sae_id, device=device)


@st.cache_resource(show_spinner=False)
def load_sae_local(checkpoint_path: str, layer: int, device: str):
    from fra.sae_lens_wrapper import LocalLn1SAE
    return LocalLn1SAE(checkpoint_path, layer=layer, device=device)


@st.cache_resource(show_spinner=False)
def load_model_pair(base_name: str, it_name: str, device: str,
                    it_arch_name: str = ""):
    """Load both base and target models for a crosscoder pair.

    If the target model isn't in TransformerLens's registry (e.g.
    DeepSeek-R1-Distill-Llama-8B), pass ``it_arch_name`` to specify a
    compatible architecture name and load via ``hf_model``.
    """
    from transformers import AutoModelForCausalLM
    from transformer_lens import HookedTransformer
    torch.set_grad_enabled(False)
    base = HookedTransformer.from_pretrained(
        base_name, device=device, dtype=torch.float16,
    )
    if it_arch_name:
        from transformers import AutoTokenizer
        hf_model = AutoModelForCausalLM.from_pretrained(
            it_name, dtype=torch.float16,
        )
        hf_tokenizer = AutoTokenizer.from_pretrained(it_name)
        it = HookedTransformer.from_pretrained(
            it_arch_name, device=device, dtype=torch.float16,
            hf_model=hf_model, tokenizer=hf_tokenizer,
        )
        del hf_model
    else:
        it = HookedTransformer.from_pretrained(
            it_name, device=device, dtype=torch.float16,
        )
    return base, it


@st.cache_resource(show_spinner=False)
def load_crosscoder(repo_id: str, model_idx: int, device: str, subfolder: str = ""):
    from fra.crosscoder_wrapper import GemmaCrosscoderFRA
    if subfolder:
        return GemmaCrosscoderFRA.from_cc_weights(
            repo_id, subfolder, model_idx=model_idx, device=device,
        )
    return GemmaCrosscoderFRA.from_pretrained(
        repo_id, model_idx=model_idx, device=device,
    )


# ---------------------------------------------------------------------------
# Crosscoder presets
# ---------------------------------------------------------------------------

CROSSCODER_PRESETS = {
    "Gemma 2B — Base vs Instruct": {
        "base_model": "google/gemma-2-2b",
        "it_model": "google/gemma-2-2b-it",
        "it_arch_name": "",
        "repo_id": "science-of-finetuning/gemma-2-2b-L13-k100-lr1e-04-local-shuffling-CCLoss",
        "layers": {13: ""},
        "default_layer": 13,
        "max_layer": 25,
        "n_heads": 8,
        "model_labels": ("Base (model 0)", "Instruct (model 1)"),
        "ram_note": "~12 GB GPU RAM (both Gemma 2B models fp16 + crosscoder).",
    },
    "Llama 8B — Base vs R1-Distill (Reasoning)": {
        "base_model": "meta-llama/Llama-3.1-8B",
        "it_model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "it_arch_name": "meta-llama/Llama-3.1-8B",
        "repo_id": "mitroitskii/Crosscoder-Llama-3.1-8B-vs-Llama-R1-Distill-8B",
        "layers": {
            7: "BatchTopK-Crosscoder/L7R",
            15: "BatchTopK-Crosscoder/L15R",
            23: "BatchTopK-Crosscoder/L23R",
        },
        "default_layer": 15,
        "max_layer": 31,
        "n_heads": 32,
        "model_labels": ("Base (model 0)", "Reasoning (model 1)"),
        "ram_note": "~36 GB GPU RAM (both Llama 8B models fp16 + crosscoder).",
    },
}


@st.cache_data
def fetch_neuronpedia(layer: int, feature_id: int) -> str:
    """Fetch feature explanation from Neuronpedia API (cached)."""
    try:
        url = (
            f"https://www.neuronpedia.org/api/feature/gpt2-small"
            f"/{layer}-att-kk/{feature_id}"
        )
        r = requests.get(url, timeout=4)
        if r.status_code == 200:
            data = r.json()
            expls = data.get("explanations", [])
            if expls:
                return expls[0].get("description", f"Feature {feature_id}")
    except Exception:
        pass
    return f"Feature {feature_id}"


def neuronpedia_embed_url(layer: int, feature_id: int) -> str:
    return (
        f"https://www.neuronpedia.org/gpt2-small/{layer}-att-kk/{feature_id}"
        f"?embed=true&embedexplanation=true&embedplots=true&embedtest=false"
    )


# ---------------------------------------------------------------------------
# Computation helpers
# ---------------------------------------------------------------------------

def run_fra(
    text: str,
    layer: int,
    head: int,
    hook_point: str,
    sae_type: str,
    sae_hub_release: str,
    sae_hub_id: str,
    sae_local_path: str,
    top_k_features: int,
    device: str,
) -> dict:
    """Compute FRA and return numpy-serialisable result dict."""
    from fra.fra_func import get_sentence_fra_batch

    model = load_model("gpt2-small", device)

    if sae_type == "hub":
        sae = load_sae_hub(sae_hub_release, sae_hub_id, device)
    else:
        sae = load_sae_local(sae_local_path, layer, device)

    with torch.no_grad():
        fra_result = get_sentence_fra_batch(
            model, sae, text,
            layer=layer, head=head,
            max_length=128, top_k=top_k_features,
            hook_point=hook_point,
        )

        # Also grab feature activations for token-level display
        hook_name = f"blocks.{layer}.{hook_point}"
        tokens = model.tokenizer.encode(text)[:128]
        tok_tensor = torch.tensor(tokens).unsqueeze(0).to(device)
        _, cache = model.run_with_cache(tok_tensor, names_filter=[hook_name])
        act = cache[hook_name].squeeze(0)
        if act.dim() == 3:
            act = act.flatten(-2, -1)
        feat_acts = sae.encode(act)  # [seq_len, d_sae]

        # Standard attention pattern + pre-softmax scores for comparison
        attn_pattern_hook = f"blocks.{layer}.attn.hook_pattern"
        attn_scores_hook = f"blocks.{layer}.attn.hook_attn_scores"
        _, attn_cache = model.run_with_cache(
            tok_tensor, names_filter=[attn_pattern_hook, attn_scores_hook]
        )
        attn_pattern = attn_cache[attn_pattern_hook][0, head].cpu().numpy()  # [S, S]
        attn_scores = attn_cache[attn_scores_hook][0, head].cpu().numpy()    # [S, S]

        token_strs = [model.tokenizer.decode([t]) for t in tokens]

    sparse = fra_result["fra_tensor_sparse"]
    return {
        "indices_np": sparse.indices().cpu().numpy(),   # [4, nnz]
        "values_np": sparse.values().cpu().numpy(),     # [nnz]
        "shape": fra_result["shape"],
        "seq_len": fra_result["seq_len"],
        "total_interactions": fra_result["total_interactions"],
        "feat_acts_np": feat_acts.cpu().numpy(),        # [seq_len, d_sae]
        "attn_pattern_np": attn_pattern,                # [seq_len, seq_len]
        "attn_scores_np": attn_scores,                  # [seq_len, seq_len]
        "token_strs": token_strs,
    }


def run_fra_crosscoder(
    tokens: list,
    head: int,
    crosscoder_layer: int,
    crosscoder_repo_id: str,
    model_idx: int,
    base_model_name: str,
    it_model_name: str,
    top_k_features: int,
    device: str,
    subfolder: str = "",
    it_arch_name: str = "",
) -> dict:
    """Compute FRA with a model-diffing crosscoder, same return format as run_fra."""
    from fra.fra_crosscoder import get_sentence_fra_crosscoder

    base_model, it_model = load_model_pair(
        base_model_name, it_model_name, device, it_arch_name,
    )
    crosscoder = load_crosscoder(crosscoder_repo_id, model_idx, device, subfolder)
    target_model = base_model if model_idx == 0 else it_model
    layer = crosscoder_layer + 1

    with torch.no_grad():
        fra_result = get_sentence_fra_crosscoder(
            base_model, it_model, crosscoder, tokens,
            head=head,
            crosscoder_layer=crosscoder_layer,
            max_length=128, top_k=top_k_features,
            verbose=True,
        )

        feat_acts = fra_result["feature_activations"]

        # Standard attention pattern + pre-softmax scores for comparison
        tok_tensor = torch.tensor(tokens[:128]).unsqueeze(0).to(device)
        attn_pattern_hook = f"blocks.{layer}.attn.hook_pattern"
        attn_scores_hook = f"blocks.{layer}.attn.hook_attn_scores"
        _, attn_cache = target_model.run_with_cache(
            tok_tensor, names_filter=[attn_pattern_hook, attn_scores_hook],
        )
        attn_pattern = attn_cache[attn_pattern_hook][0, head].cpu().numpy()
        attn_scores = attn_cache[attn_scores_hook][0, head].cpu().numpy()

        token_strs = [target_model.tokenizer.decode([t]) for t in tokens[:128]]

    sparse = fra_result["fra_tensor_sparse"]
    return {
        "indices_np": sparse.indices().cpu().numpy(),
        "values_np": sparse.values().cpu().numpy(),
        "shape": fra_result["shape"],
        "seq_len": fra_result["seq_len"],
        "total_interactions": fra_result["total_interactions"],
        "feat_acts_np": feat_acts.cpu().numpy(),
        "attn_pattern_np": attn_pattern,
        "attn_scores_np": attn_scores,
        "token_strs": token_strs,
    }


def _aggregate_pairs(indices_np, values_np, filter_self=False):
    """Aggregate by (q_feat, k_feat) returning {pair: (sum_abs, count)}."""
    q_feats = indices_np[2, :]
    k_feats = indices_np[3, :]
    abs_vals = np.abs(values_np)

    if filter_self:
        mask = q_feats != k_feats
        q_feats, k_feats, abs_vals = q_feats[mask], k_feats[mask], abs_vals[mask]

    pair_sum: dict = defaultdict(float)
    pair_count: dict = defaultdict(int)
    for q, k, v in zip(q_feats, k_feats, abs_vals):
        pair_sum[(int(q), int(k))] += float(v)
        pair_count[(int(q), int(k))] += 1

    return [
        (q, k, pair_sum[(q, k)], pair_count[(q, k)])
        for (q, k) in pair_sum
    ]


def compute_di_row(
    W_dec: torch.Tensor,
    W_Q: torch.Tensor,
    W_K: torch.Tensor,
    query_feature: int,
) -> np.ndarray:
    """Compute DI(query_feature, j) for all key features j.

    Returns array of shape [d_sae].
    """
    q_vec = W_dec[query_feature] @ W_Q          # [d_head]
    K_all = W_dec @ W_K                         # [d_sae, d_head]
    d_head = W_Q.shape[-1]
    di_row = (K_all @ q_vec) / math.sqrt(d_head)  # [d_sae]
    return di_row.detach().cpu().float().numpy()


def get_top_pairs(indices_np, values_np, top_k=50, filter_self=False):
    """Return top-k pairs by total absolute strength (strongest interactions)."""
    pairs = _aggregate_pairs(indices_np, values_np, filter_self)
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_k]


def get_bottom_pairs(indices_np, values_np, top_k=50, filter_self=False):
    """Return bottom-k pairs by total absolute strength (weakest interactions)."""
    pairs = _aggregate_pairs(indices_np, values_np, filter_self)
    pairs.sort(key=lambda x: x[2])
    return pairs[:top_k]


def get_position_heatmap(indices_np, values_np, q_feat, k_feat, seq_len):
    """Extract [seq_len, seq_len] heatmap for a specific (q_feat, k_feat) pair."""
    mask = (indices_np[2] == q_feat) & (indices_np[3] == k_feat)
    q_pos = indices_np[0, mask]
    k_pos = indices_np[1, mask]
    vals = np.abs(values_np[mask])

    mat = np.zeros((seq_len, seq_len))
    for qp, kp, v in zip(q_pos, k_pos, vals):
        mat[qp, kp] += v
    return mat




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


def token_activation_bar(token_strs, activations, color, height=220):
    """Return a Plotly bar chart of per-token activations."""
    tick_vals = list(range(len(token_strs)))
    tick_labels = [html_lib.escape(t) for t in token_strs]
    fig = go.Figure(go.Bar(
        x=tick_vals,
        y=activations,
        marker_color=color,
    ))
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=0, b=30),
        xaxis=dict(tickvals=tick_vals, ticktext=tick_labels),
        yaxis_title="Activation",
        showlegend=False,
    )
    return fig




# ---------------------------------------------------------------------------
# Sidebar — configuration
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

    sae_option = st.radio(
        "SAE",
        [
            "GPT-2 — Hub hook_z (Neuronpedia)",
            "GPT-2 — Local ln1 (trained)",
            "Crosscoder (model-diffing)",
        ],
        index=0,
    )

    if sae_option.startswith("Crosscoder"):
        sae_type = "crosscoder"
        supports_neuronpedia = False

        preset_name = st.selectbox("Preset", list(CROSSCODER_PRESETS.keys()))
        preset = CROSSCODER_PRESETS[preset_name]

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

        available_layers = sorted(preset["layers"].keys())
        crosscoder_layer = st.selectbox(
            "Crosscoder layer",
            available_layers,
            index=available_layers.index(preset["default_layer"]),
        )
        cc_subfolder = preset["layers"][crosscoder_layer]
        cc_it_arch = preset["it_arch_name"]

        layer = crosscoder_layer + 1
        st.caption(f"Attention layer: **{layer}** (crosscoder layer + 1)")
        head = st.number_input("Head", 0, preset["n_heads"] - 1, value=0)

        apply_chat_template = st.checkbox(
            "Apply chat template",
            value=True,
            help=(
                "Wrap the input text using the target model's "
                "tokenizer.apply_chat_template(). Required for "
                "safety / reasoning features to activate."
            ),
        )

        st.caption(preset["ram_note"])

        # not used for crosscoder path
        hook_point = ""
        sae_hub_release = ""
        sae_hub_id = ""
        sae_local_path = ""
    else:
        crosscoder_repo_id = ""
        base_model_name = ""
        it_model_name = ""
        model_idx = 0
        crosscoder_layer = 13
        cc_subfolder = ""
        cc_it_arch = ""
        apply_chat_template = False

        col_l, col_h = st.columns(2)
        with col_l:
            layer = st.number_input("Layer", 0, 11, value=5)
        with col_h:
            head = st.number_input("Head", 0, 11, value=1)

        if sae_option.startswith("GPT-2 — Hub"):
            sae_type = "hub"
            hook_point = "attn.hook_z"
            sae_hub_release = "gpt2-small-hook-z-kk"
            sae_hub_id = f"blocks.{layer}.hook_z"
            supports_neuronpedia = True
            sae_local_path = ""
        else:
            sae_type = "local"
            hook_point = "ln1.hook_normalized"
            sae_hub_release = ""
            sae_hub_id = ""
            supports_neuronpedia = False
            default_local = str(
                Path(__file__).parent.parent / "checkpoints" / "q9sczrvl" / "50003968"
            )
            sae_local_path = st.text_input("Checkpoint path", value=default_local)
            if not Path(sae_local_path).exists():
                st.warning("Checkpoint not found. Train with `python train_sae.py`.")

    st.subheader("Compute settings")
    top_k_feat = st.slider("Top-K features / position", 5, 50, 20)
    top_k_pairs = st.slider("Top-K pairs to display", 10, 100, 30)
    filter_self = st.checkbox("Filter self-interactions (q==k)", value=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.caption(f"Device: {device}")

    compute_btn = st.button("▶  Compute FRA", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🧠 Feature-Resolved Attention Dashboard")
st.caption("Decomposing attention through SAE feature space.")

# ---------------------------------------------------------------------------
# Trigger computation
# ---------------------------------------------------------------------------

if compute_btn:
    if sae_type == "crosscoder":
        with st.spinner("Loading models & crosscoder…"):
            load_model_pair(base_model_name, it_model_name, device, cc_it_arch)
            load_crosscoder(crosscoder_repo_id, model_idx, device, cc_subfolder)

        base_model, it_model = load_model_pair(
            base_model_name, it_model_name, device, cc_it_arch,
        )
        target_model = base_model if model_idx == 0 else it_model
        if apply_chat_template:
            fra_tokens = it_model.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=True,
                add_generation_prompt=True,
            )
            with st.expander("Templated input"):
                st.code(it_model.tokenizer.decode(fra_tokens))
        else:
            fra_tokens = target_model.tokenizer.encode(text)

        with st.spinner("Computing Feature-Resolved Attention (crosscoder)…"):
            fra_data = run_fra_crosscoder(
                tokens=fra_tokens,
                head=int(head),
                crosscoder_layer=int(crosscoder_layer),
                crosscoder_repo_id=crosscoder_repo_id,
                model_idx=int(model_idx),
                base_model_name=base_model_name,
                it_model_name=it_model_name,
                top_k_features=top_k_feat,
                device=device,
                subfolder=cc_subfolder,
                it_arch_name=cc_it_arch,
            )
    else:
        with st.spinner("Loading model & SAE…"):
            load_model("gpt2-small", device)
            if sae_type == "hub":
                load_sae_hub(sae_hub_release, sae_hub_id, device)
            elif Path(sae_local_path).exists():
                load_sae_local(sae_local_path, int(layer), device)

        with st.spinner("Computing Feature-Resolved Attention…"):
            fra_data = run_fra(
                text=text,
                layer=int(layer),
                head=int(head),
                hook_point=hook_point,
                sae_type=sae_type,
                sae_hub_release=sae_hub_release,
                sae_hub_id=sae_hub_id,
                sae_local_path=sae_local_path,
                top_k_features=top_k_feat,
                device=device,
            )

    st.session_state["fra_data"] = fra_data
    st.session_state["fra_config"] = {
        "layer": int(layer),
        "head": int(head),
        "text": text,
        "supports_neuronpedia": supports_neuronpedia,
        "filter_self": filter_self,
        "top_k_pairs": top_k_pairs,
    }
    st.success(
        f"Done — {fra_data['total_interactions']:,} non-zero interactions found."
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

    # Recompute pairs (filter / top_k may change without recomputing FRA)
    pairs = get_top_pairs(
        fra_data["indices_np"],
        fra_data["values_np"],
        top_k=cfg["top_k_pairs"],
        filter_self=cfg["filter_self"],
    )
    bottom_pairs = get_bottom_pairs(
        fra_data["indices_np"],
        fra_data["values_np"],
        top_k=cfg["top_k_pairs"],
        filter_self=cfg["filter_self"],
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tokens", seq_len)
    c2.metric("Non-zero interactions", f"{fra_data['total_interactions']:,}")
    total_unique = len(_aggregate_pairs(fra_data["indices_np"], fra_data["values_np"], cfg["filter_self"]))
    c3.metric("Unique feature pairs", f"{total_unique:,}")
    c4.metric("Layer / Head", f"L{layer_} / H{head_}")

    tok_html = " ".join(
        f'<span style="background:#e9ecef;padding:2px 5px;border-radius:3px;'
        f'font-family:monospace;font-size:0.9em;">{html_lib.escape(t)}</span>'
        for t in token_strs
    )
    st.markdown(tok_html, unsafe_allow_html=True)
    st.markdown("")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Top Interactions",
    "🔥 Feature Matrix",
    "🔍 Attention Comparison",
    "🧩 Max-Act Examples",
])

# ── Tab 1: Top / Least Interactions ────────────────────────────────────────

with tab1:
    if not _has_fra:
        st.info("Click **▶ Compute FRA** in the sidebar to see interactions.")
    else:
        rank_mode = st.radio(
            "Show:",
            ["Top interactions (strongest)", "Least interactions (weakest)"],
            horizontal=True,
            label_visibility="collapsed",
        )
        active_pairs = pairs if rank_mode.startswith("Top") else bottom_pairs

        if not active_pairs:
            st.warning("No interactions found with current filters.")
        else:
            col_list, col_detail = st.columns([1, 2])

            with col_list:
                st.subheader("Feature pairs")
                pair_labels = [
                    f"F{q}→F{k}  ({s:.3f})"
                    + ("  ⟲" if q == k else "")
                    for q, k, s, _ in active_pairs
                ]
                selected_idx = st.radio(
                    "Select a pair to inspect:",
                    range(len(active_pairs)),
                    format_func=lambda i: pair_labels[i],
                    label_visibility="collapsed",
                )

            with col_detail:
                q_sel, k_sel, strength_sel, count_sel = active_pairs[selected_idx]
                is_self = q_sel == k_sel

                st.subheader(
                    f"Feature {q_sel} → Feature {k_sel}"
                    + ("  ⟲ self" if is_self else "")
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

                barA, barB = st.columns(2)
                with barA:
                    st.markdown(f"**Query feature {q_sel}** — token activations")
                    st.plotly_chart(
                        token_activation_bar(
                            token_strs, q_acts, "rgba(102,126,234,0.75)"
                        ),
                        use_container_width=True,
                    )
                with barB:
                    st.markdown(f"**Key feature {k_sel}** — token activations")
                    st.plotly_chart(
                        token_activation_bar(
                            token_strs, k_acts, "rgba(118,75,162,0.75)"
                        ),
                        use_container_width=True,
                    )

                # --- Position heatmap: [seq, seq] for this pair ---
                st.markdown("**Position heatmap** — where does this pair interact?")
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
                fig_pos.update_layout(
                    height=300,
                    margin=dict(l=0, r=0, t=0, b=0),
                    xaxis=dict(title="Key token", tickvals=tick_vals, ticktext=tick_labels),
                    yaxis=dict(title="Query token", tickvals=tick_vals, ticktext=tick_labels, autorange="reversed"),
                )
                st.plotly_chart(fig_pos, use_container_width=True)

                # --- Data-Independent Ranking ---
                st.markdown(
                    "**Data-independent ranking** — is this pair "
                    "inherently coupled via the QK circuit?"
                )
                st.caption(
                    "DI(i,j) = (W_dec[i]·W_Q)·(W_dec[j]·W_K)ᵀ/√d_h measures "
                    "how strongly the model's **weights alone** couple two features, "
                    "ignoring activations. A pair can top the FRA list via high "
                    "activations even if its DI rank is moderate. The histogram "
                    "shows DI(query, j) for **all** key features j — a high |DI| "
                    "rank means the QK circuit specifically wires this pair "
                    "(positive = promotes attention, negative = suppresses), "
                    "rather than the interaction being driven by co-activation alone."
                )

                from fra.fra_crosscoder import _get_W_K

                if sae_type == "crosscoder":
                    _di_base, _di_it = load_model_pair(
                        base_model_name, it_model_name, device, cc_it_arch,
                    )
                    _di_model = _di_base if model_idx == 0 else _di_it
                    _di_cc = load_crosscoder(
                        crosscoder_repo_id, model_idx, device, cc_subfolder,
                    )
                    _di_W_dec = _di_cc.W_dec
                    _di_layer = int(crosscoder_layer) + 1
                else:
                    _di_model = load_model("gpt2-small", device)
                    if sae_type == "hub":
                        _di_sae = load_sae_hub(sae_hub_release, sae_hub_id, device)
                    else:
                        _di_sae = load_sae_local(sae_local_path, int(layer), device)
                    _di_W_dec = _di_sae.W_dec
                    _di_layer = cfg["layer"]

                _di_W_Q = _di_model.blocks[_di_layer].attn.W_Q[head_]
                _di_W_K = _get_W_K(_di_model, _di_layer, head_)

                # Cache DI row — only recompute when query feature changes
                _di_key = (q_sel, _di_layer, head_)
                if st.session_state.get("_di_cache_key") != _di_key:
                    st.session_state["_di_row"] = compute_di_row(
                        _di_W_dec, _di_W_Q, _di_W_K, q_sel,
                    )
                    st.session_state["_di_cache_key"] = _di_key
                di_row = st.session_state["_di_row"]

                target_val = float(di_row[k_sel])
                abs_row = np.abs(di_row)
                rank = int((abs_row > abs(target_val)).sum()) + 1
                total = len(di_row)
                percentile = (1 - rank / total) * 100

                di_m1, di_m2, di_m3 = st.columns(3)
                di_m1.metric("DI value", f"{target_val:.4f}")
                di_m2.metric("Rank (by |DI|)", f"{rank:,}", help=f"Out of {total:,} features")
                di_m3.metric("Percentile", f"{percentile:.2f}%")

                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=di_row, nbinsx=200,
                    marker_color="rgba(102,126,234,0.6)",
                ))
                fig_hist.add_vline(
                    x=target_val, line_dash="dash", line_color="red",
                    annotation_text=f"F{q_sel}→F{k_sel}: {target_val:.4f}",
                )
                fig_hist.update_layout(
                    height=300,
                    margin=dict(l=0, r=0, t=30, b=0),
                    xaxis_title="DI value",
                    yaxis_title="Count",
                    showlegend=False,
                )
                st.plotly_chart(fig_hist, use_container_width=True)

                # --- Neuronpedia iframes ---
                if cfg["supports_neuronpedia"]:
                    np_col1, np_col2 = st.columns(2)
                    with np_col1:
                        desc_q = fetch_neuronpedia(layer_, q_sel)
                        st.markdown(
                            f"**Neuronpedia — F{q_sel}:** _{desc_q}_"
                        )
                        st.components.v1.iframe(
                            neuronpedia_embed_url(layer_, q_sel),
                            height=380,
                        )
                    with np_col2:
                        desc_k = fetch_neuronpedia(layer_, k_sel)
                        st.markdown(
                            f"**Neuronpedia — F{k_sel}:** _{desc_k}_"
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

# ── Tab 2: Feature Matrix ──────────────────────────────────────────────────

with tab2:
    if not _has_fra:
        st.info("Click **▶ Compute FRA** in the sidebar to see the feature matrix.")
    else:
        st.subheader(f"FRA Feature Interaction Matrix — L{layer_} H{head_}")
        st.caption(
            "Each cell shows the total absolute interaction strength summed over "
            "all position pairs. Only the top features appearing in the ranked list "
            "are shown."
        )

        if not pairs:
            st.warning("No pairs to display.")
        else:
            # Collect unique features from top pairs
            top_features = []
            seen = set()
            for q, k, _, _ in pairs:
                for f in (q, k):
                    if f not in seen:
                        seen.add(f)
                        top_features.append(f)
                if len(top_features) >= 30:
                    break

            feat_to_idx = {f: i for i, f in enumerate(top_features)}
            n = len(top_features)
            matrix = np.zeros((n, n))

            for q, k, strength, _ in pairs:
                if q in feat_to_idx and k in feat_to_idx:
                    matrix[feat_to_idx[q], feat_to_idx[k]] += strength

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

# ── Tab 3: Attention Comparison ────────────────────────────────────────────

with tab3:
    if not _has_fra:
        st.info("Click **▶ Compute FRA** in the sidebar to see attention comparison.")
    else:
        st.subheader(f"Standard vs FRA Attention — L{layer_} H{head_}")

        # Use integer positions to avoid Plotly merging duplicate token labels
        attn_tick_vals = list(range(seq_len))
        attn_tick_labels = [html_lib.escape(t) for t in token_strs]

        # -- Build the four matrices --

        # Standard logits: masked pre-softmax scores (upper triangle = -inf → NaN for display)
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

        # -- Shared layout helper --

        def _attn_heatmap(z, hover_label, colorscale="RdBu", zmid=None):
            fig = go.Figure(go.Heatmap(
                z=z,
                x=attn_tick_vals,
                y=attn_tick_vals,
                colorscale=colorscale,
                zmid=zmid,
                hovertemplate=(
                    f"Q: %{{y}}<br>K: %{{x}}<br>{hover_label}: %{{z:.4f}}"
                    "<extra></extra>"
                ),
            ))
            fig.update_layout(
                height=380,
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(title="Key", tickvals=attn_tick_vals, ticktext=attn_tick_labels),
                yaxis=dict(title="Query", tickvals=attn_tick_vals, ticktext=attn_tick_labels, autorange="reversed"),
            )
            return fig

        # -- Row 1: Logits --

        st.markdown("#### Pre-softmax logits")
        col_std_logit, col_fra_logit = st.columns(2)

        with col_std_logit:
            st.markdown("**Standard** (masked QK scores)")
            st.plotly_chart(
                _attn_heatmap(std_logits, "Logit", zmid=0),
                use_container_width=True,
            )

        with col_fra_logit:
            st.markdown("**FRA** (signed sum over feature pairs)")
            st.plotly_chart(
                _attn_heatmap(fra_logits, "Logit", zmid=0),
                use_container_width=True,
            )

        # -- Row 2: Probs --

        st.markdown("#### Post-softmax probabilities")
        col_std_prob, col_fra_prob = st.columns(2)

        with col_std_prob:
            st.markdown("**Standard** (attention weights)")
            st.plotly_chart(
                _attn_heatmap(std_probs, "Weight"),
                use_container_width=True,
            )

        with col_fra_prob:
            st.markdown("**FRA** (softmax of FRA logits)")
            st.plotly_chart(
                _attn_heatmap(fra_probs, "Weight"),
                use_container_width=True,
            )

# ── Tab 4: Max-Act Examples ────────────────────────────────────────────────

with tab4:
    from fra.max_act import (
        compute_max_acts as _compute_max_acts,
        list_available_features as _list_available_features,
        load_prompts as _load_prompts,
        load_results as _load_results,
        save_results as _save_results,
    )

    st.subheader("Max-Act Examples")
    st.caption(
        "Browse max-activating examples for crosscoder features across a corpus "
        "of safe and unsafe prompts."
    )

    _RESULTS_DIR = str(Path(__file__).parent.parent / "results")

    # --- Mode selector ---
    ma_mode = st.radio("Mode", ["Load existing", "Compute new"], horizontal=True)

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
                for fid in ma_feature_ids:
                    _save_results(fid, results[fid], len(results[fid]),
                                  "BeaverTails+UltraChat", _RESULTS_DIR)

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
