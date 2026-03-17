"""
FRA Dashboard — Feature-Resolved Attention interactive viewer.

Run with:
    streamlit run fra/streamlit_app.py
"""

import html as html_lib
import sys
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote

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

@st.cache_resource
def load_model(model_name: str, device: str):
    from transformer_lens import HookedTransformer
    torch.set_grad_enabled(False)
    return HookedTransformer.from_pretrained(model_name, device=device)


@st.cache_resource
def load_sae_hub(release: str, sae_id: str, device: str):
    from fra.sae_lens_wrapper import SAELensAttentionSAE
    return SAELensAttentionSAE(release, sae_id, device=device)


@st.cache_resource
def load_sae_local(checkpoint_path: str, layer: int, device: str):
    from fra.sae_lens_wrapper import LocalLn1SAE
    return LocalLn1SAE(checkpoint_path, layer=layer, device=device)


@st.cache_resource
def load_gemma_pair(base_name: str, it_name: str, device: str):
    """Load both Gemma 2B base and instruct models."""
    from transformer_lens import HookedTransformer
    torch.set_grad_enabled(False)
    base = HookedTransformer.from_pretrained(
        base_name, device=device, dtype=torch.float16,
    )
    it = HookedTransformer.from_pretrained(
        it_name, device=device, dtype=torch.float16,
    )
    return base, it


@st.cache_resource
def load_crosscoder(repo_id: str, model_idx: int, device: str):
    from fra.crosscoder_wrapper import GemmaCrosscoderFRA
    return GemmaCrosscoderFRA.from_pretrained(
        repo_id, model_idx=model_idx, device=device,
    )


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
) -> dict:
    """Compute FRA with a model-diffing crosscoder, same return format as run_fra."""
    from fra.fra_crosscoder import get_sentence_fra_crosscoder

    base_model, it_model = load_gemma_pair(base_model_name, it_model_name, device)
    crosscoder = load_crosscoder(crosscoder_repo_id, model_idx, device)
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


def get_active_query_features(indices_np, values_np):
    """Return list of (q_feat, total_outgoing_strength, n_key_features) sorted by strength."""
    q_feats = indices_np[2, :]
    k_feats = indices_np[3, :]
    abs_vals = np.abs(values_np)

    unique_q = np.unique(q_feats)
    results = []
    for qf in unique_q:
        mask = q_feats == qf
        total_strength = float(abs_vals[mask].sum())
        n_keys = len(np.unique(k_feats[mask]))
        results.append((int(qf), total_strength, n_keys))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def get_top_keys_for_query(indices_np, values_np, query_feat, top_k=30):
    """Return top key features for a query feature: (k_feat, sum_abs, count) sorted by sum_abs."""
    q_feats = indices_np[2, :]
    k_feats = indices_np[3, :]
    abs_vals = np.abs(values_np)

    mask = q_feats == query_feat
    k_masked = k_feats[mask]
    v_masked = abs_vals[mask]

    unique_keys = np.unique(k_masked)
    results = []
    for kf in unique_keys:
        k_mask = k_masked == kf
        results.append((int(kf), float(v_masked[k_mask].sum()), int(k_mask.sum())))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


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


def save_feature_data(query_feat, key_features, fra_data, token_strs, output_dir="results/features"):
    """Save query feature and its top key features to human-readable JSON files."""
    import json

    feat_acts_np = fra_data["feat_acts_np"]
    indices_np = fra_data["indices_np"]
    values_np = fra_data["values_np"]
    seq_len = fra_data["seq_len"]

    out = Path(output_dir) / f"F{query_feat}"
    out.mkdir(parents=True, exist_ok=True)

    q_acts = feat_acts_np[:seq_len, query_feat].tolist()
    total_strength = sum(s for _, s, _ in key_features)

    summary = {
        "query_feature": query_feat,
        "total_outgoing_strength": round(total_strength, 6),
        "token_strs": token_strs,
        "query_activations": [round(a, 6) for a in q_acts],
        "top_key_features": [
            {
                "feature_id": kf,
                "interaction_strength": round(s, 6),
                "count": c,
                "pct_of_total": round(100 * s / total_strength, 1) if total_strength > 0 else 0,
            }
            for kf, s, c in key_features
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    for kf, _, _ in key_features:
        k_acts = feat_acts_np[:seq_len, kf].tolist()

        # Position pairs where this (query, key) interaction occurs
        mask = (indices_np[2] == query_feat) & (indices_np[3] == kf)
        q_pos = indices_np[0, mask].tolist()
        k_pos = indices_np[1, mask].tolist()
        pair_vals = np.abs(values_np[mask]).tolist()

        tokens_with_acts = [
            {
                "token": token_strs[i],
                "pos": i,
                "query_act": round(float(q_acts[i]), 6),
                "key_act": round(float(k_acts[i]), 6),
            }
            for i in range(seq_len)
            if q_acts[i] != 0 or k_acts[i] != 0
        ]

        key_data = {
            "key_feature": kf,
            "key_activations": [round(a, 6) for a in k_acts],
            "position_pairs": [
                {"q_pos": int(qp), "k_pos": int(kp), "strength": round(float(v), 6)}
                for qp, kp, v in zip(q_pos, k_pos, pair_vals)
            ],
            "tokens_with_activations": tokens_with_acts,
        }
        (out / f"F{kf}.json").write_text(json.dumps(key_data, indent=2))

    return str(out)


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
            "Gemma 2B — Crosscoder (model-diffing)",
        ],
        index=0,
    )

    if sae_option.startswith("Gemma"):
        sae_type = "crosscoder"
        supports_neuronpedia = False

        crosscoder_repo_id = st.text_input(
            "Crosscoder HF repo",
            value="science-of-finetuning/gemma-2-2b-L13-k100-lr1e-04-local-shuffling-CCLoss",
        )
        base_model_name = st.text_input("Base model", value="google/gemma-2-2b")
        it_model_name = st.text_input("Instruct model", value="google/gemma-2-2b-it")
        model_idx = st.radio(
            "Analyse attention of",
            [0, 1],
            format_func=lambda i: "Base (model 0)" if i == 0 else "Instruct (model 1)",
            horizontal=True,
        )
        crosscoder_layer = st.number_input(
            "Crosscoder layer (activations)", 0, 25, value=13,
        )
        layer = crosscoder_layer + 1
        st.caption(f"Attention layer: **{layer}** (crosscoder layer + 1)")
        head = st.number_input("Head", 0, 7, value=0)

        apply_chat_template = st.checkbox(
            "Apply chat template",
            value=True,
            help=(
                "Wrap the input text using the instruct model's "
                "tokenizer.apply_chat_template(). Required for "
                "refusal / safety features to activate."
            ),
        )

        st.caption(
            "Requires ~12 GB GPU RAM for both Gemma 2B models (fp16) "
            "plus the crosscoder. Gemma weights are gated — accept the "
            "license on HuggingFace and run `huggingface-cli login` first."
        )

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
        with st.spinner("Loading Gemma models & crosscoder…"):
            load_gemma_pair(base_model_name, it_model_name, device)
            load_crosscoder(crosscoder_repo_id, model_idx, device)

        base_model, it_model = load_gemma_pair(base_model_name, it_model_name, device)
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
    "🧩 Max-Act Examples",
])

# ── Tab 1: Top / Least Interactions ────────────────────────────────────────

with tab1:
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
    st.subheader(f"Max-Act Examples — L{layer_} H{head_}")

    active_q_feats = get_active_query_features(
        fra_data["indices_np"], fra_data["values_np"],
    )

    if not active_q_feats:
        st.warning("No active query features found.")
    else:
        q_options = {
            f"F{qf} (strength: {s:.3f}, {n} key features)": qf
            for qf, s, n in active_q_feats
        }
        selected_q_label = st.selectbox(
            "Query feature",
            list(q_options.keys()),
        )
        selected_q_feat = q_options[selected_q_label]

        top_keys = get_top_keys_for_query(
            fra_data["indices_np"], fra_data["values_np"],
            selected_q_feat, top_k=30,
        )
        total_q_strength = sum(s for _, s, _ in top_keys)

        if not top_keys:
            st.warning("No key features found for this query feature.")
        else:
            col_keys, col_detail = st.columns([1, 2])

            with col_keys:
                st.markdown("**Key features** (ranked by interaction strength)")
                key_labels = [
                    f"F{kf} — {s:.3f} ({100 * s / total_q_strength:.0f}%)"
                    if total_q_strength > 0
                    else f"F{kf} — {s:.3f}"
                    for kf, s, _ in top_keys
                ]
                selected_key_idx = st.radio(
                    "Select key feature:",
                    range(len(top_keys)),
                    format_func=lambda i: key_labels[i],
                    label_visibility="collapsed",
                )

            with col_detail:
                k_feat, k_strength, k_count = top_keys[selected_key_idx]
                feat_acts = fra_data["feat_acts_np"]
                q_acts = feat_acts[:seq_len, selected_q_feat]
                k_acts = feat_acts[:seq_len, k_feat]
                tick_labels = [html_lib.escape(t) for t in token_strs]
                tick_vals = list(range(len(token_strs)))

                # -- Query feature activation heatmap --
                st.markdown(f"**Query F{selected_q_feat}** — per-token activations")
                fig_q = go.Figure(go.Heatmap(
                    z=[q_acts.tolist()],
                    x=tick_vals,
                    y=[""],
                    colorscale="Blues",
                    hovertemplate="Token: %{x}<br>Activation: %{z:.4f}<extra></extra>",
                ))
                fig_q.update_layout(
                    height=100,
                    margin=dict(l=0, r=0, t=0, b=30),
                    xaxis=dict(tickvals=tick_vals, ticktext=tick_labels),
                    yaxis=dict(showticklabels=False),
                )
                st.plotly_chart(fig_q, use_container_width=True)

                # -- Key feature activation heatmap --
                st.markdown(f"**Key F{k_feat}** — per-token activations")
                fig_k = go.Figure(go.Heatmap(
                    z=[k_acts.tolist()],
                    x=tick_vals,
                    y=[""],
                    colorscale="Purples",
                    hovertemplate="Token: %{x}<br>Activation: %{z:.4f}<extra></extra>",
                ))
                fig_k.update_layout(
                    height=100,
                    margin=dict(l=0, r=0, t=0, b=30),
                    xaxis=dict(tickvals=tick_vals, ticktext=tick_labels),
                    yaxis=dict(showticklabels=False),
                )
                st.plotly_chart(fig_k, use_container_width=True)

                # -- Position interaction heatmap --
                st.markdown(f"**Position heatmap** — F{selected_q_feat} → F{k_feat}")
                pos_mat = get_position_heatmap(
                    fra_data["indices_np"], fra_data["values_np"],
                    selected_q_feat, k_feat, seq_len,
                )
                fig_pos = go.Figure(go.Heatmap(
                    z=pos_mat,
                    x=tick_vals,
                    y=tick_vals,
                    colorscale="Blues",
                    hovertemplate="Q-pos: %{y}<br>K-pos: %{x}<br>Strength: %{z:.4f}<extra></extra>",
                ))
                fig_pos.update_layout(
                    height=300,
                    margin=dict(l=0, r=0, t=0, b=0),
                    xaxis=dict(title="Key token", tickvals=tick_vals, ticktext=tick_labels),
                    yaxis=dict(title="Query token", tickvals=tick_vals, ticktext=tick_labels, autorange="reversed"),
                )
                st.plotly_chart(fig_pos, use_container_width=True)

                # -- Interaction summary --
                pct = (100 * k_strength / total_q_strength) if total_q_strength > 0 else 0
                st.caption(
                    f"Total strength: **{k_strength:.4f}** | "
                    f"Position pairs: **{k_count}** | "
                    f"% of query total: **{pct:.1f}%**"
                )

                # -- Save button --
                if st.button(f"💾 Save F{selected_q_feat} data", key="save_feature"):
                    save_path = save_feature_data(
                        selected_q_feat, top_keys, fra_data, token_strs,
                    )
                    st.success(f"Saved to {save_path}/")