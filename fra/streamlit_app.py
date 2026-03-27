"""
FRA Dashboard — Feature-Resolved Attention interactive viewer.

Run with:
    streamlit run fra/streamlit_app.py
"""

import html as html_lib
import json
import sys
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote

import numpy as np
import plotly.graph_objects as go
import requests
import streamlit as st
import torch
import torch.nn.functional as F

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

@st.cache_resource
def load_model(model_name: str, device: str, hf_token: str = ""):
    from transformer_lens import HookedTransformer
    torch.set_grad_enabled(False)
    kwargs = {}
    if hf_token:
        kwargs["token"] = hf_token
    return HookedTransformer.from_pretrained(model_name, device=device, **kwargs)


@st.cache_resource
def load_sae_hub(release: str, sae_id: str, device: str):
    from fra.sae_lens_wrapper import SAELensAttentionSAE
    return SAELensAttentionSAE(release, sae_id, device=device)


@st.cache_resource
def load_sae_local(checkpoint_path: str, layer: int, device: str):
    from fra.sae_lens_wrapper import LocalLn1SAE
    return LocalLn1SAE(checkpoint_path, layer=layer, device=device)


@st.cache_resource
def load_sae_gemma(release: str, sae_id: str, device: str):
    from fra.sae_lens_wrapper import GemmaScopeSAE
    return GemmaScopeSAE(release, sae_id, device=device)


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


@st.cache_data
def load_sae_explanations(model_id: str, layer: int) -> dict[int, str]:
    """Load local SAE feature explanations for a model/layer from JSONL batches."""
    explanations: dict[int, str] = {}
    if not model_id:
        return explanations

    explanations_dir = Path(__file__).parent / "SAE_Explanations"
    if not explanations_dir.exists():
        return explanations

    layer_prefix = f"{layer}-"
    for jsonl_path in sorted(explanations_dir.glob("batch-*.jsonl")):
        try:
            with jsonl_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if str(row.get("modelId", "")) != model_id:
                        continue
                    if not str(row.get("layer", "")).startswith(layer_prefix):
                        continue

                    idx = row.get("index")
                    desc = str(row.get("description", "")).strip()
                    if idx is None or not desc:
                        continue

                    try:
                        explanations[int(idx)] = desc
                    except (ValueError, TypeError):
                        continue
        except OSError:
            continue

    return explanations


def format_feature_label(feature_id: int, explanations: dict[int, str], max_len: int = 64) -> str:
    """Render user-facing feature labels using explanation text when available."""
    desc = explanations.get(int(feature_id), "")
    if not desc:
        return f"Feature #{feature_id}"

    desc = " ".join(desc.split())
    if len(desc) > max_len:
        desc = f"{desc[: max_len - 3]}..."
    return f"{desc} [#{feature_id}]"


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
    model_name: str = "gpt2-small",
    chunk_size: int = 16,
    hf_token: str = "",
    include_special_tokens: bool = True,
) -> dict:
    """Compute FRA and return numpy-serialisable result dict."""
    from fra.fra_func import get_sentence_fra_batch

    model = load_model(model_name, device, hf_token)

    if sae_type == "hub":
        sae = load_sae_hub(sae_hub_release, sae_hub_id, device)
    elif sae_type == "gemma":
        sae = load_sae_gemma(sae_hub_release, sae_hub_id, device)
    else:
        sae = load_sae_local(sae_local_path, layer, device)

    with torch.no_grad():
        fra_result = get_sentence_fra_batch(
            model, sae, text,
            layer=layer, head=head,
            max_length=128, top_k=top_k_features,
            hook_point=hook_point,
            chunk_size=chunk_size,
            prepend_bos=include_special_tokens,
        )

        # Also grab feature activations for token-level display
        hook_name = f"blocks.{layer}.{hook_point}"
        tokens = model.tokenizer.encode(
            text, add_special_tokens=include_special_tokens
        )[:128]
        tok_tensor = torch.tensor(tokens).unsqueeze(0).to(device)
        _, cache = model.run_with_cache(tok_tensor, names_filter=[hook_name])
        act = cache[hook_name].squeeze(0)
        if act.dim() == 3:
            act = act.flatten(-2, -1)
        feat_acts = sae.encode(act)  # [seq_len, d_sae]

        # Standard attention pattern for comparison
        attn_hook = f"blocks.{layer}.attn.hook_pattern"
        _, attn_cache = model.run_with_cache(
            tok_tensor, names_filter=[attn_hook]
        )
        attn_pattern = attn_cache[attn_hook][0, head].cpu().numpy()  # [S, S]

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
        "token_strs": token_strs,
    }


def _aggregate_pairs(indices_np, values_np, filter_self=False):
    """
    Aggregate by (q_feat, k_feat).

    Returns list of (q_feat, k_feat, sum_abs, count, max_abs).

    Three natural ranking signals exposed here:
      i)  sum_abs          — total absolute strength (biased toward frequently active pairs)
      ii) sum_abs / count  — mean only over position-pairs where the pair fires
                             (unbiased, expected to surface cleanest pairs)
      iii) max_abs         — single strongest occurrence across the sentence
    """
    q_feats  = indices_np[2, :]
    k_feats  = indices_np[3, :]
    abs_vals = np.abs(values_np)

    if filter_self:
        mask = q_feats != k_feats
        q_feats, k_feats, abs_vals = q_feats[mask], k_feats[mask], abs_vals[mask]

    pair_sum:   dict = defaultdict(float)
    pair_count: dict = defaultdict(int)
    pair_max:   dict = defaultdict(float)

    for q, k, v in zip(q_feats, k_feats, abs_vals):
        key = (int(q), int(k))
        pair_sum[key]    += float(v)
        pair_count[key]  += 1
        if float(v) > pair_max[key]:
            pair_max[key] = float(v)

    return [
        (q, k, pair_sum[(q, k)], pair_count[(q, k)], pair_max[(q, k)])
        for (q, k) in pair_sum
    ]


def get_ranked_pairs(indices_np, values_np, top_k=50, filter_self=False, mode="avg"):
    """
    Return top-k pairs ranked by the chosen aggregation mode.

    mode:
      "sum"  — total absolute strength summed over all position-pairs  (ranking i)
      "avg"  — mean absolute strength over non-zero position-pairs      (ranking ii)
      "max"  — maximum single-position-pair absolute strength            (ranking iii)
    """
    pairs = _aggregate_pairs(indices_np, values_np, filter_self)
    if mode == "sum":
        pairs.sort(key=lambda x: x[2], reverse=True)
    elif mode == "avg":
        pairs.sort(key=lambda x: x[2] / max(x[3], 1), reverse=True)
    elif mode == "max":
        pairs.sort(key=lambda x: x[4], reverse=True)
    else:
        pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_k]


# Keep old names as thin wrappers for backward compat
def get_top_pairs(indices_np, values_np, top_k=50, filter_self=False):
    return get_ranked_pairs(indices_np, values_np, top_k, filter_self, mode="sum")


def get_bottom_pairs(indices_np, values_np, top_k=50, filter_self=False):
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


def token_activation_bar(token_strs, activations, color, height=220):
    """Return a Plotly bar chart of per-token activations."""
    fig = go.Figure(go.Bar(
        x=[html_lib.escape(t) for t in token_strs],
        y=activations,
        marker_color=color,
    ))
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=0, b=30),
        xaxis_title=None,
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

    model_choice = st.radio(
        "Model",
        ["GPT-2 Small", "Gemma-2 2B"],
        horizontal=True,
    )
    is_gemma = model_choice == "Gemma-2 2B"
    model_name = "gemma-2-2b" if is_gemma else "gpt2-small"
    max_layer = 25 if is_gemma else 11
    max_head  = 7  if is_gemma else 11

    col_l, col_h = st.columns(2)
    with col_l:
        layer = st.number_input("Layer", 0, max_layer, value=12 if is_gemma else 5)
    with col_h:
        head = st.number_input("Head",  0, max_head,  value=0)

    if is_gemma:
        sae_option = st.radio(
            "SAE",
            ["Gemma-Scope — resid_pre"],
            index=0,
        )
        sae_type = "gemma"
        hook_point = "hook_resid_pre"
        supports_neuronpedia = False
        sae_local_path = ""
        sae_hub_release = st.text_input(
            "Release", value="gemma-scope-2b-pt-res"
        )
        sae_hub_id = st.text_input(
            "SAE ID", value=f"layer_{int(layer)}/width_16k/average_l0_82"
        )
    else:
        sae_option = st.radio(
            "SAE",
            ["Hub — hook_z (Neuronpedia)", "Local — ln1 (trained)"],
            index=0,
        )
        if sae_option.startswith("Hub"):
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
    # Gemma-Scope has large d_sae — default to small chunks to avoid OOM
    default_chunk = 1 if is_gemma else 16
    chunk_size = st.slider("Chunk size (↓ = less GPU mem)", 1, 32, default_chunk)
    top_k_pairs = st.slider("Top-K pairs to display", 10, 100, 30)
    filter_self = st.checkbox("Filter self-interactions (q==k)", value=False)
    include_special_tokens = st.checkbox(
        "Include special tokens (BOS)",
        value=True,
        help="Gemma adds a BOS token by default. Uncheck to exclude it. GPT-2 has no BOS.",
    )

    st.subheader("Ranking mode")
    agg_mode = st.radio(
        "Rank feature pairs by:",
        options=["avg", "sum", "max"],
        format_func={
            "avg": "(ii) Non-zero avg — mean strength when pair fires",
            "sum": "(i)  Sum — total strength over all positions",
            "max": "(iii) Max — strongest single occurrence",
        }.get,
        index=0,
        help=(
            "**(i) Sum**: total |FRA| summed over all position-pairs. Biased toward "
            "pairs that fire often.\n\n"
            "**(ii) Non-zero avg** (recommended): mean |FRA| divided only by the "
            "number of position-pairs where the pair actually fires. Best for "
            "finding cleanest semantic interactions.\n\n"
            "**(iii) Max**: the single highest |FRA| value anywhere in the sentence. "
            "Good for finding the strongest individual occurrence."
        ),
    )

    if is_gemma:
        hf_token = st.text_input(
            "HuggingFace token (for Gemma)",
            type="password",
            help="Required if you haven't run `huggingface-cli login`. Get yours at huggingface.co/settings/tokens",
        )
    else:
        hf_token = ""

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
    with st.spinner("Loading model & SAE…"):
        load_model(model_name, device, hf_token)
        if sae_type == "hub":
            load_sae_hub(sae_hub_release, sae_hub_id, device)
        elif sae_type == "gemma":
            load_sae_gemma(sae_hub_release, sae_hub_id, device)
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
            model_name=model_name,
            chunk_size=int(chunk_size),
            hf_token=hf_token,
            include_special_tokens=include_special_tokens,
        )

    st.session_state["fra_data"] = fra_data
    st.session_state["fra_config"] = {
        "layer": int(layer),
        "head": int(head),
        "text": text,
        "model_name": model_name,
        "supports_neuronpedia": supports_neuronpedia,
        "filter_self": filter_self,
        "top_k_pairs": top_k_pairs,
        "agg_mode": agg_mode,
    }
    st.success(
        f"Done — {fra_data['total_interactions']:,} non-zero interactions found."
    )

# ---------------------------------------------------------------------------
# Main results area
# ---------------------------------------------------------------------------

if "fra_data" not in st.session_state:
    st.info("Configure the sidebar and click **▶ Compute FRA** to begin.")
    with st.expander("What is Feature-Resolved Attention?"):
        st.markdown(
            """
**Feature-Resolved Attention (FRA)** replaces the standard `[seq, seq]` attention
matrix with a `[seq, seq, d_sae, d_sae]` tensor, where each entry captures how much
**SAE query-feature _i_** at position _q_ attends to **SAE key-feature _j_** at
position _k_.

This lets us ask:
- Which _semantic_ features in the query attend strongly to which key features?
- Are there **conceptual induction heads** that copy specific concepts across positions?
- How do feature-level interactions differ from token-level ones?
"""
        )
    st.stop()

fra_data = st.session_state["fra_data"]
cfg = st.session_state["fra_config"]
layer_ = cfg["layer"]
head_ = cfg["head"]
model_name_ = cfg.get("model_name", "")
seq_len = fra_data["seq_len"]
token_strs = fra_data["token_strs"][:seq_len]
use_sae_explanations = model_name_ == "gemma-2-2b"
sae_explanations = load_sae_explanations(model_name_, layer_) if use_sae_explanations else {}


def display_feature_label(feature_id: int, max_len: int = 64) -> str:
    if use_sae_explanations:
        return format_feature_label(feature_id, sae_explanations, max_len=max_len)
    return f"F{feature_id}"

# Recompute pairs (filter / top_k / agg_mode may change without recomputing FRA)
agg_mode = cfg.get("agg_mode", "avg")
pairs = get_ranked_pairs(
    fra_data["indices_np"],
    fra_data["values_np"],
    top_k=cfg["top_k_pairs"],
    filter_self=cfg["filter_self"],
    mode=agg_mode,
)
bottom_pairs = get_bottom_pairs(
    fra_data["indices_np"],
    fra_data["values_np"],
    top_k=cfg["top_k_pairs"],
    filter_self=cfg["filter_self"],
)

# ---------------------------------------------------------------------------
# Summary row
# ---------------------------------------------------------------------------

c1, c2, c3, c4 = st.columns(4)
c1.metric("Tokens", seq_len)
c2.metric("Non-zero interactions", f"{fra_data['total_interactions']:,}")
total_unique = len(_aggregate_pairs(fra_data["indices_np"], fra_data["values_np"], cfg["filter_self"]))
c3.metric("Unique feature pairs", f"{total_unique:,}")
c4.metric(f"Layer / Head", f"L{layer_} / H{head_}")

# Tokenised text display
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
    "🔬 Ablation",
])

# ── Tab 1: Top / Least Interactions ────────────────────────────────────────

with tab1:
    agg_labels = {"avg": "(ii) Non-zero avg", "sum": "(i) Sum", "max": "(iii) Max"}
    st.caption(f"Ranking by: **{agg_labels.get(agg_mode, agg_mode)}** — change in sidebar.")

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
            # Label shows the active ranking metric
            def _pair_label(i):
                q, k, s, cnt, mx = active_pairs[i]
                avg_s = s / max(cnt, 1)
                if agg_mode == "avg":
                    score_str = f"avg={avg_s:.3f}"
                elif agg_mode == "max":
                    score_str = f"max={mx:.3f}"
                else:
                    score_str = f"sum={s:.3f}"
                suffix = "  ⟲" if q == k else ""
                q_lbl = display_feature_label(q, max_len=38)
                k_lbl = display_feature_label(k, max_len=38)
                return f"{q_lbl} → {k_lbl}  {score_str}{suffix}"

            selected_idx = st.radio(
                "Select a pair to inspect:",
                range(len(active_pairs)),
                format_func=_pair_label,
                label_visibility="collapsed",
            )

        with col_detail:
            q_sel, k_sel, strength_sel, count_sel, max_sel = active_pairs[selected_idx]
            avg_sel = strength_sel / max(count_sel, 1)
            is_self = q_sel == k_sel
            q_full = display_feature_label(q_sel, max_len=120)
            k_full = display_feature_label(k_sel, max_len=120)

            st.subheader(
                f"{q_full} → {k_full}"
                + ("  ⟲ self" if is_self else "")
            )
            m1, m2, m3 = st.columns(3)
            m1.metric("(i) Sum |FRA|",   f"{strength_sel:.4f}")
            m2.metric("(ii) Non-zero avg", f"{avg_sel:.4f}", help=f"over {count_sel} position-pairs")
            m3.metric("(iii) Max |FRA|",  f"{max_sel:.4f}")
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
                st.markdown(f"**Query: {q_full}** — token activations")
                st.plotly_chart(
                    token_activation_bar(
                        token_strs, q_acts, "rgba(102,126,234,0.75)"
                    ),
                    use_container_width=True,
                )
            with barB:
                st.markdown(f"**Key: {k_full}** — token activations")
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
            fig_pos = go.Figure(go.Heatmap(
                z=pos_mat,
                x=[html_lib.escape(t) for t in token_strs],
                y=[html_lib.escape(t) for t in token_strs],
                colorscale="Blues",
                hovertemplate=(
                    "Q-pos: %{y}<br>K-pos: %{x}<br>Strength: %{z:.4f}"
                    "<extra></extra>"
                ),
            ))
            fig_pos.update_layout(
                height=300,
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis_title="Key token",
                yaxis_title="Query token",
                yaxis_autorange="reversed",
            )
            st.plotly_chart(fig_pos, use_container_width=True)

            # --- Neuronpedia iframes ---
            if cfg["supports_neuronpedia"]:
                np_col1, np_col2 = st.columns(2)
                with np_col1:
                    desc_q = fetch_neuronpedia(layer_, q_sel)
                    st.markdown(
                        f"**Neuronpedia — {q_full}:** _{desc_q}_"
                    )
                    st.components.v1.iframe(
                        neuronpedia_embed_url(layer_, q_sel),
                        height=380,
                    )
                with np_col2:
                    desc_k = fetch_neuronpedia(layer_, k_sel)
                    st.markdown(
                        f"**Neuronpedia — {k_full}:** _{desc_k}_"
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
    st.subheader(f"FRA Feature Interaction Matrix — L{layer_} H{head_}")
    mode_desc = {"avg": "non-zero average", "sum": "sum", "max": "max"}.get(agg_mode, agg_mode)
    st.caption(
        f"Each cell shows the **{mode_desc}** absolute interaction strength "
        "over all position pairs. Only features appearing in the ranked list are shown."
    )

    if not pairs:
        st.warning("No pairs to display.")
    else:
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
        # Build matrix using the currently active ranking score
        matrix = np.zeros((n, n))
        for q, k, s, cnt, mx in pairs:
            if q in feat_to_idx and k in feat_to_idx:
                if agg_mode == "avg":
                    score = s / max(cnt, 1)
                elif agg_mode == "max":
                    score = mx
                else:
                    score = s
                matrix[feat_to_idx[q], feat_to_idx[k]] += score

        labels = [display_feature_label(f, max_len=28) for f in top_features]

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
    st.subheader(f"Standard vs FRA Attention — L{layer_} H{head_}")

    col_std, col_fra = st.columns(2)

    with col_std:
        st.markdown("**Standard token-level attention** (post-softmax)")
        attn = fra_data["attn_pattern_np"][:seq_len, :seq_len]
        fig_attn = go.Figure(go.Heatmap(
            z=attn,
            x=[html_lib.escape(t) for t in token_strs],
            y=[html_lib.escape(t) for t in token_strs],
            colorscale="RdBu",
            hovertemplate=(
                "Q: %{y}<br>K: %{x}<br>Weight: %{z:.4f}<extra></extra>"
            ),
        ))
        fig_attn.update_layout(
            height=420,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis_title="Key",
            yaxis_title="Query",
            yaxis_autorange="reversed",
        )
        st.plotly_chart(fig_attn, use_container_width=True)

    with col_fra:
        st.markdown(
            "**FRA attention** — summed over all feature pairs, per position"
        )
        # Collapse feature dims: sum abs(value) for each (q_pos, k_pos)
        idxs = fra_data["indices_np"]
        vals_abs = np.abs(fra_data["values_np"])
        fra_pos_mat = np.zeros((seq_len, seq_len))
        q_pos_all = idxs[0, :]
        k_pos_all = idxs[1, :]
        for qp, kp, v in zip(q_pos_all, k_pos_all, vals_abs):
            if qp < seq_len and kp < seq_len:
                fra_pos_mat[qp, kp] += v

        fig_fra_attn = go.Figure(go.Heatmap(
            z=fra_pos_mat,
            x=[html_lib.escape(t) for t in token_strs],
            y=[html_lib.escape(t) for t in token_strs],
            colorscale="RdBu",
            hovertemplate=(
                "Q: %{y}<br>K: %{x}<br>FRA strength: %{z:.4f}<extra></extra>"
            ),
        ))
        fig_fra_attn.update_layout(
            height=420,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis_title="Key",
            yaxis_title="Query",
            yaxis_autorange="reversed",
        )
        st.plotly_chart(fig_fra_attn, use_container_width=True)

    st.info(
        "The FRA attention matrix collapses the feature dimensions — it shows "
        "the *total feature-level interaction* at each position pair, comparable "
        "to the standard attention weight. Differences between the two highlight "
        "where FRA captures additional structure beyond raw token attention."
    )

# ── Tab 4: Ablation ──────────────────────────────────────────────────────

with tab4:
    st.subheader(f"Feature-Pair Ablation — L{layer_} H{head_}")
    st.caption(
        "Ablate selected feature pairs from the FRA tensor and measure the "
        "impact on model output. This reveals which cross-feature interactions "
        "are causally important to this attention head's computation."
    )

    # Build pair selection UI
    all_pairs_for_ablation = get_ranked_pairs(
        fra_data["indices_np"], fra_data["values_np"],
        top_k=100, filter_self=False, mode=agg_mode,
    )

    if not all_pairs_for_ablation:
        st.warning("No feature pairs found. Compute FRA first.")
    else:
        # Separate off-diagonal and on-diagonal
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
                ["Top off-diagonal (i≠j)", "Top on-diagonal (i==j)", "Random off-diagonal"],
                help=(
                    "**Off-diagonal**: cross-feature interactions (feature i attending to "
                    "different feature j). **On-diagonal**: self-interactions (same feature "
                    "at query and key). Random is a control."
                ),
            )

        with abl_col2:
            st.markdown("**Pairs to ablate:**")
            if abl_target.startswith("Top off"):
                selected_pairs = offdiag_list[:n_ablate]
            elif abl_target.startswith("Top on"):
                selected_pairs = ondiag_list[:n_ablate]
            else:
                import random
                rng = random.Random(42)
                selected_pairs = rng.sample(offdiag_list, min(n_ablate, len(offdiag_list)))

            for i, (q, k, s, cnt, mx) in enumerate(selected_pairs[:15]):
                avg = s / max(cnt, 1)
                marker = "⟲" if q == k else "→"
                st.text(f"  F{q} {marker} F{k}  (avg={avg:.4f}, sum={s:.4f})")
            if len(selected_pairs) > 15:
                st.text(f"  ... and {len(selected_pairs) - 15} more")

        run_abl = st.button("▶  Run Ablation", type="primary")

        if run_abl:
            with st.spinner("Running ablation..."):
                from fra.ablation_study import (
                    ablate_fra_pairs,
                    compute_bias_corrections,
                    reconstruct_scores,
                    run_condition,
                )

                # Get model and SAE from cache
                mdl = load_model(model_name, device, hf_token)
                if sae_type == "hub":
                    sae_obj = load_sae_hub(sae_hub_release, sae_hub_id, device)
                elif sae_type == "gemma":
                    sae_obj = load_sae_gemma(sae_hub_release, sae_hub_id, device)
                else:
                    sae_obj = load_sae_local(sae_local_path, int(layer_), device)

                # Bias corrections
                bias = compute_bias_corrections(
                    mdl, sae_obj, cfg["text"], layer_, head_, hook_point
                )

                if bias is None:
                    st.error("Text too short for ablation.")
                else:
                    # The dashboard may have excluded BOS or truncated differently
                    # than compute_bias_corrections (which re-tokenizes).
                    # Truncate bias vectors to match the FRA's seq_len.
                    bias_seq = bias["seq_len"]
                    if bias_seq != seq_len:
                        # Trim bias terms to dashboard's seq_len
                        bias["term_q"] = bias["term_q"][:seq_len]
                        bias["term_k"] = bias["term_k"][:seq_len]
                        bias["seq_len"] = seq_len
                        # Also trim token tensor and labels
                        bias["tok_tensor"] = bias["tok_tensor"][:, :seq_len]
                        bias["shift_labels"] = bias["tok_tensor"][0, 1:]
                        # Re-run unpatched loss with trimmed tokens
                        logits_trim = mdl(bias["tok_tensor"])
                        bias["unpatched_loss"] = F.cross_entropy(
                            logits_trim[0, :-1], bias["shift_labels"]
                        ).item()
                        bias["unpatched_logits"] = logits_trim

                    # Rebuild sparse tensor from stored indices/values
                    d_sae_val = fra_data["feat_acts_np"].shape[1]
                    sp_indices = torch.tensor(fra_data["indices_np"], dtype=torch.long)
                    sp_values = torch.tensor(fra_data["values_np"], dtype=torch.float32)
                    sp_size = torch.Size([seq_len, seq_len, d_sae_val, d_sae_val])
                    fra_sparse = torch.sparse_coo_tensor(sp_indices, sp_values, size=sp_size).coalesce()

                    # Full FRA scores (baseline)
                    from fra.validation import fra_sum_to_attn
                    fra_sum_full = fra_sum_to_attn(fra_sparse, seq_len)
                    scores_full = reconstruct_scores(fra_sum_full, bias, device)

                    # Ablated scores
                    pairs_to_abl = [(int(p[0]), int(p[1])) for p in selected_pairs]
                    fra_ablated = ablate_fra_pairs(fra_sparse, pairs_to_abl, d_sae_val)
                    fra_sum_abl = fra_sum_to_attn(fra_ablated, seq_len)
                    scores_abl = reconstruct_scores(fra_sum_abl, bias, device)

                    # Zero scores
                    mask_t = torch.triu(
                        torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1
                    )
                    scores_zero = torch.zeros((seq_len, seq_len), device=device) + mask_t

                    # Run conditions
                    tok_t = bias["tok_tensor"]
                    shift_lab = bias["shift_labels"]
                    unp_logits = bias["unpatched_logits"]

                    r_full = run_condition(mdl, layer_, head_, tok_t, shift_lab, scores_full, unp_logits)
                    r_abl = run_condition(mdl, layer_, head_, tok_t, shift_lab, scores_abl, unp_logits)
                    r_zero = run_condition(mdl, layer_, head_, tok_t, shift_lab, scores_zero, unp_logits)

                    # Display results
                    st.markdown("---")
                    st.subheader("Ablation Results")

                    hc = r_zero["loss"] - bias["unpatched_loss"]
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    mc1.metric("Unpatched loss", f"{bias['unpatched_loss']:.4f}")
                    mc2.metric("FRA full loss", f"{r_full['loss']:.4f}",
                               delta=f"{r_full['loss'] - bias['unpatched_loss']:+.4f}")
                    mc3.metric("Ablated loss", f"{r_abl['loss']:.4f}",
                               delta=f"{r_abl['loss'] - bias['unpatched_loss']:+.4f}")
                    mc4.metric("Zero-ablated loss", f"{r_zero['loss']:.4f}",
                               delta=f"{r_zero['loss'] - bias['unpatched_loss']:+.4f}")

                    mc5, mc6, mc7 = st.columns(3)
                    mc5.metric("Ablation KL div", f"{r_abl['kl_div']:.4f}",
                               help="KL divergence from unpatched distribution")
                    mc6.metric("Top-1 predictions changed",
                               f"{r_abl['top1_change_frac']*100:.1f}%")
                    if hc > 0.01:
                        rec_full = (r_zero['loss'] - r_full['loss']) / hc
                        rec_abl = (r_zero['loss'] - r_abl['loss']) / hc
                        mc7.metric("Recovery (full→ablated)",
                                   f"{rec_full:.3f} → {rec_abl:.3f}",
                                   delta=f"{rec_abl - rec_full:+.3f}")

                    # Attention heatmap comparison
                    st.markdown("**Attention score comparison**")
                    hm1, hm2, hm3 = st.columns(3)

                    def _score_heatmap(scores_np, title):
                        # Mask upper triangle for display
                        disp = scores_np.copy()
                        disp[np.triu_indices_from(disp, k=1)] = np.nan
                        fig = go.Figure(go.Heatmap(
                            z=disp,
                            x=[html_lib.escape(t) for t in token_strs],
                            y=[html_lib.escape(t) for t in token_strs],
                            colorscale="RdBu",
                            zmid=0,
                            hovertemplate="Q: %{y}<br>K: %{x}<br>Score: %{z:.2f}<extra></extra>",
                        ))
                        fig.update_layout(
                            title=title, height=350,
                            margin=dict(l=0, r=0, t=30, b=0),
                            yaxis_autorange="reversed",
                        )
                        return fig

                    with hm1:
                        st.plotly_chart(
                            _score_heatmap(scores_full.cpu().numpy(), "FRA Full"),
                            use_container_width=True,
                        )
                    with hm2:
                        st.plotly_chart(
                            _score_heatmap(scores_abl.cpu().numpy(), "After Ablation"),
                            use_container_width=True,
                        )
                    with hm3:
                        diff = scores_abl.cpu().numpy() - scores_full.cpu().numpy()
                        st.plotly_chart(
                            _score_heatmap(diff, "Difference (Abl - Full)"),
                            use_container_width=True,
                        )