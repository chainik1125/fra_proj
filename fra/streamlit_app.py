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
def load_model(model_name: str, device: str, hf_token: str = ""):
    from transformer_lens import HookedTransformer
    torch.set_grad_enabled(False)
    kwargs = {}
    if hf_token:
        kwargs["token"] = hf_token
    return HookedTransformer.from_pretrained(model_name, device=device, **kwargs)


@st.cache_resource(show_spinner=False)
def load_sae_hub(release: str, sae_id: str, device: str):
    from fra.sae_lens_wrapper import SAELensAttentionSAE
    return SAELensAttentionSAE(release, sae_id, device=device)


@st.cache_resource(show_spinner=False)
def load_sae_local(checkpoint_path: str, layer: int, device: str):
    from fra.sae_lens_wrapper import LocalLn1SAE
    return LocalLn1SAE(checkpoint_path, layer=layer, device=device)


@st.cache_resource(show_spinner=False)
def load_sae_gemma(release: str, sae_id: str, device: str):
    from fra.sae_lens_wrapper import GemmaScopeSAE
    return GemmaScopeSAE(release, sae_id, device=device)


@st.cache_resource(show_spinner=False)
def load_model_gemma(model_name: str, device: str, hf_token: str = ""):
    from transformer_lens import HookedTransformer
    torch.set_grad_enabled(False)
    kwargs = {
        "fold_ln": False,
        "center_unembed": False,
        "center_writing_weights": False,
        "fold_value_biases": False,
        "refactor_factored_attn_matrices": False,
    }
    if hf_token:
        kwargs["token"] = hf_token
    return HookedTransformer.from_pretrained(model_name, device=device, **kwargs)


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
# Presets — every model / SAE combination the dashboard supports
# ---------------------------------------------------------------------------

PRESETS = {
    "GPT-2 Small — SAE (hook_z, Neuronpedia)": {
        "type": "sae_hub",
        "model": "gpt2-small",
        "release": "gpt2-small-hook-z-kk",
        "sae_id_template": "blocks.{layer}.hook_z",
        "hook_point": "attn.hook_z",
        "layers": list(range(12)),
        "default_layer": 5,
        "n_heads": 12,
        "supports_neuronpedia": True,
    },
    "GPT-2 Small — SAE (ln1, local)": {
        "type": "sae_local",
        "model": "gpt2-small",
        "checkpoint_path": "./checkpoints/q9sczrvl/50003968",
        "hook_point": "ln1.hook_normalized",
        "layers": list(range(12)),
        "default_layer": 2,
        "n_heads": 12,
        "supports_neuronpedia": False,
    },
    "Gemma-2 2B — Gemma-Scope (resid_pre)": {
        "type": "sae_gemma",
        "model": "gemma-2-2b",
        "release": "gemma-scope-2b-pt-res",
        "sae_id_template": "layer_{layer}/width_16k/average_l0_82",
        "hook_point": "hook_resid_pre",
        "layers": list(range(26)),
        "default_layer": 12,
        "n_heads": 8,
        "supports_neuronpedia": False,
        "hf_token_required": True,
        "chunk_size_default": 1,
    },
    "Gemma-2 2B — Crosscoder (Base vs Instruct)": {
        "type": "crosscoder",
        "base_model": "google/gemma-2-2b",
        "it_model": "google/gemma-2-2b-it",
        "it_arch_name": "",
        "repo_id": "science-of-finetuning/gemma-2-2b-L13-k100-lr1e-04-local-shuffling-CCLoss",
        "layers": {13: ""},
        "default_layer": 13,
        "n_heads": 8,
        "model_labels": ("Base (model 0)", "Instruct (model 1)"),
        "ram_note": "~12 GB GPU RAM (both Gemma 2B models fp16 + crosscoder).",
        "is_reasoning": False,
    },
    "Llama 8B — Crosscoder (Base vs R1-Distill)": {
        "type": "crosscoder",
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
        "n_heads": 32,
        "model_labels": ("Base (model 0)", "Reasoning (model 1)"),
        "ram_note": "~36 GB GPU RAM (both Llama 8B models fp16 + crosscoder).",
        "is_reasoning": True,
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
    model_name: str = "gpt2-small",
    chunk_size: int = 16,
    hf_token: str = "",
    include_special_tokens: bool = True,
) -> dict:
    """Compute FRA and return numpy-serialisable result dict."""
    from fra.fra_func import get_sentence_fra_batch

    if sae_type == "gemma":
        model = load_model_gemma(model_name, device, hf_token)
    else:
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
    """Aggregate by (q_feat, k_feat) returning (q, k, sum_abs, count, max_abs)."""
    q_feats = indices_np[2, :]
    k_feats = indices_np[3, :]
    abs_vals = np.abs(values_np)

    if filter_self:
        mask = q_feats != k_feats
        q_feats, k_feats, abs_vals = q_feats[mask], k_feats[mask], abs_vals[mask]

    pair_sum: dict = defaultdict(float)
    pair_count: dict = defaultdict(int)
    pair_max: dict = defaultdict(float)
    for q, k, v in zip(q_feats, k_feats, abs_vals):
        pair_sum[(int(q), int(k))] += float(v)
        pair_count[(int(q), int(k))] += 1
        if float(v) > pair_max[(int(q), int(k))]:
            pair_max[(int(q), int(k))] = float(v)

    return [
        (q, k, pair_sum[(q, k)], pair_count[(q, k)], pair_max[(q, k)])
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


def compute_di_col(
    W_dec: torch.Tensor,
    W_Q: torch.Tensor,
    W_K: torch.Tensor,
    key_feature: int,
) -> np.ndarray:
    """Compute DI(i, key_feature) for all query features i.

    Returns array of shape [d_sae].
    """
    k_vec = W_dec[key_feature] @ W_K            # [d_head]
    Q_all = W_dec @ W_Q                         # [d_sae, d_head]
    d_head = W_Q.shape[-1]
    di_col = (Q_all @ k_vec) / math.sqrt(d_head)  # [d_sae]
    return di_col.detach().cpu().float().numpy()


def compute_global_di_topk(
    W_dec: torch.Tensor,
    W_Q: torch.Tensor,
    W_K: torch.Tensor,
    top_k: int = 50,
    chunk_size: int = 1024,
    n_sample_rows: int = 500,
    progress_callback=None,
) -> dict:
    """Scan all (i,j) pairs for global top-k by |DI|.

    Also samples random rows for the distribution histogram.

    Returns dict with keys: query_ids, key_ids, di_values, hist_sample.
    """
    d_sae = W_dec.shape[0]
    d_head = W_Q.shape[-1]
    scale = math.sqrt(d_head)
    K_all = W_dec @ W_K                          # [d_sae, d_head]
    n_chunks = math.ceil(d_sae / chunk_size)

    # Sample random rows for histogram
    rng = np.random.default_rng(42)
    sample_idxs = rng.choice(d_sae, size=min(n_sample_rows, d_sae), replace=False)
    Q_sample = W_dec[torch.tensor(sample_idxs, device=W_dec.device)] @ W_Q
    hist_sample = ((Q_sample @ K_all.T) / scale).detach().cpu().float().numpy().ravel()

    # Running top-k across chunks
    top_vals = torch.empty(0, device=W_dec.device)
    top_q = torch.empty(0, dtype=torch.long, device=W_dec.device)
    top_k_buf = torch.empty(0, dtype=torch.long, device=W_dec.device)

    for c in range(n_chunks):
        start = c * chunk_size
        end = min(start + chunk_size, d_sae)
        Q_chunk = W_dec[start:end] @ W_Q         # [chunk, d_head]
        DI_chunk = (Q_chunk @ K_all.T) / scale   # [chunk, d_sae]

        flat_abs = DI_chunk.abs().flatten()
        k_local = min(top_k, flat_abs.numel())
        _, idx_c = torch.topk(flat_abs, k_local)
        q_c = idx_c // d_sae + start
        k_c = idx_c % d_sae
        signed_c = DI_chunk.flatten()[idx_c]

        # Merge with running top-k
        all_signed = torch.cat([top_vals, signed_c])
        all_q = torch.cat([top_q, q_c])
        all_k = torch.cat([top_k_buf, k_c])

        k_merge = min(top_k, all_signed.numel())
        _, keep = torch.topk(all_signed.abs(), k_merge)
        top_vals = all_signed[keep]
        top_q = all_q[keep]
        top_k_buf = all_k[keep]

        if progress_callback:
            progress_callback(c + 1, n_chunks)

    order = torch.argsort(top_vals.abs(), descending=True)
    return {
        "query_ids": top_q[order].cpu().numpy(),
        "key_ids": top_k_buf[order].cpu().numpy(),
        "di_values": top_vals[order].detach().cpu().float().numpy(),
        "hist_sample": hist_sample,
        "hist_sample_idxs": sample_idxs,
        "d_sae": d_sae,
    }


def sample_di_bands(
    di_values: np.ndarray,
    feature_ids: np.ndarray | None = None,
    n_per_band: int = 5,
) -> dict:
    """Pick representative features at distribution bands.

    Returns dict with band_name -> list of (feature_id, di_value).
    Includes '_stats' key with mean and std.
    """
    if feature_ids is None:
        feature_ids = np.arange(len(di_values))

    mean = float(np.mean(di_values))
    std = float(np.std(di_values))
    abs_vals = np.abs(di_values)

    bands = {}

    # Top positive / negative
    bands["Top |DI| (positive)"] = [
        (int(feature_ids[i]), float(di_values[i]))
        for i in np.argsort(-di_values)[:n_per_band]
    ]
    bands["Top |DI| (negative)"] = [
        (int(feature_ids[i]), float(di_values[i]))
        for i in np.argsort(di_values)[:n_per_band]
    ]

    # Band thresholds
    for label, target in [
        ("+2 SD", mean + 2 * std),
        ("+1 SD", mean + 1 * std),
        ("Near mean", mean),
    ]:
        dist = np.abs(di_values - target)
        nearest = np.argsort(dist)[:n_per_band]
        bands[label] = [
            (int(feature_ids[i]), float(di_values[i]))
            for i in nearest
        ]

    # Lowest |DI|
    bands["Low |DI|"] = [
        (int(feature_ids[i]), float(di_values[i]))
        for i in np.argsort(abs_vals)[:n_per_band]
    ]

    bands["_stats"] = {"mean": mean, "std": std}
    return bands


def _load_di_weights(sae_type, head, device, **kw):
    """Load (W_dec, W_Q, W_K, attn_layer) for DI computation."""
    from fra.fra_crosscoder import _get_W_K

    if sae_type == "crosscoder":
        base, it = load_model_pair(
            kw["base_model_name"], kw["it_model_name"], device, kw["cc_it_arch"],
        )
        model = base if kw["model_idx"] == 0 else it
        cc = load_crosscoder(
            kw["crosscoder_repo_id"], kw["model_idx"], device, kw["cc_subfolder"],
        )
        W_dec = cc.W_dec
        attn_layer = int(kw["crosscoder_layer"]) + 1
    elif sae_type == "sae_gemma":
        model = load_model_gemma(
            kw.get("model_name", "gemma-2-2b"), device, kw.get("hf_token", ""),
        )
        sae_obj = load_sae_gemma(kw["sae_hub_release"], kw["sae_hub_id"], device)
        W_dec = sae_obj.W_dec
        attn_layer = int(kw["layer"])
    else:
        _model_name = kw.get("model_name", "gpt2-small")
        model = load_model(_model_name, device)
        if sae_type in ("hub", "sae_hub"):
            sae_obj = load_sae_hub(kw["sae_hub_release"], kw["sae_hub_id"], device)
        else:
            sae_obj = load_sae_local(kw["sae_local_path"], int(kw["layer"]), device)
        W_dec = sae_obj.W_dec
        attn_layer = int(kw["layer"])

    W_Q = model.blocks[attn_layer].attn.W_Q[head]
    W_K = _get_W_K(model, attn_layer, head)
    return W_dec, W_Q, W_K, attn_layer


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


def get_ranked_pairs(indices_np, values_np, top_k=50, filter_self=False, mode="avg"):
    """Return top-k pairs ranked by the selected aggregation mode.

    Modes:
        sum  -- total absolute interaction strength
        avg  -- mean absolute interaction per position-pair occurrence
        max  -- single strongest position-pair interaction
    """
    pairs = _aggregate_pairs(indices_np, values_np, filter_self)
    if mode == "sum":
        pairs.sort(key=lambda x: x[2], reverse=True)
    elif mode == "avg":
        pairs.sort(key=lambda x: x[2] / max(x[3], 1), reverse=True)
    elif mode == "max":
        pairs.sort(key=lambda x: x[4], reverse=True)
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



_HEATMAP_TICK_THRESHOLD = 15


def _show_heatmap(fig, tick_vals, tick_labels, seq_len, *,
                  x_title="Key", y_title="Query",
                  compact_height=300, key=None):
    """Render a heatmap with compact/expanded modes for long sequences.

    When *seq_len* ≤ ``_HEATMAP_TICK_THRESHOLD`` the chart is shown inline
    with token tick-labels.  Otherwise a small unlabelled preview is shown
    with an expander that reveals a full-width, labelled version sized to
    fit all tokens comfortably.
    """
    if seq_len <= _HEATMAP_TICK_THRESHOLD:
        fig.update_layout(
            height=compact_height,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(title=x_title, tickvals=tick_vals, ticktext=tick_labels),
            yaxis=dict(title=y_title, tickvals=tick_vals, ticktext=tick_labels,
                       autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True, key=key)
    else:
        # Compact preview — no tick labels, fixed small height
        preview = go.Figure(fig)
        preview.update_layout(
            height=250,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(title=x_title, showticklabels=False),
            yaxis=dict(title=y_title, showticklabels=False,
                       autorange="reversed"),
        )
        st.plotly_chart(preview, use_container_width=True,
                        key=f"{key}_preview" if key else None)

        with st.expander("Expand full heatmap"):
            full_height = max(500, seq_len * 18)
            fig.update_layout(
                height=full_height,
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(title=x_title, tickvals=tick_vals,
                           ticktext=tick_labels, tickangle=90),
                yaxis=dict(title=y_title, tickvals=tick_vals,
                           ticktext=tick_labels, autorange="reversed"),
            )
            st.plotly_chart(fig, use_container_width=True,
                            key=f"{key}_expand" if key else None)


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

    preset_name = st.selectbox("Preset", list(PRESETS.keys()))
    preset = PRESETS[preset_name]
    sae_type = preset["type"]
    supports_neuronpedia = preset.get("supports_neuronpedia", False)
    hook_point = preset.get("hook_point", "")

    # ── Layer selector ──────────────────────────────────────────────────
    if sae_type == "crosscoder":
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

    # ── Head selector ───────────────────────────────────────────────────
    head = st.number_input("Head", 0, preset["n_heads"] - 1, value=0)

    # ── Type-specific extras ────────────────────────────────────────────
    # Crosscoder extras
    if sae_type == "crosscoder":
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
                Path(__file__).parent.parent / preset["checkpoint_path"].lstrip("./")
            )
            sae_local_path = st.text_input("Checkpoint path", value=default_local)
            if not Path(sae_local_path).exists():
                st.warning("Checkpoint not found. Train with `python train_sae.py`.")
            hf_token = ""
            chunk_size = 16
        elif sae_type == "sae_gemma":
            sae_hub_release = preset["release"]
            sae_hub_id = preset["sae_id_template"].format(layer=layer)
            sae_local_path = ""
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

    # ── Ranking mode ────────────────────────────────────────────────────
    _RANK_LABELS = {"sum": "Sum |FRA|", "avg": "Avg |FRA|", "max": "Max |FRA|"}
    agg_mode = st.radio(
        "Rank feature pairs by:",
        list(_RANK_LABELS.keys()),
        format_func=lambda k: _RANK_LABELS[k],
        horizontal=True,
    )

    # ── Common compute settings ─────────────────────────────────────────
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
            if reasoning_trace.strip() or reasoning_response.strip():
                # The R1 template strips <think> from assistant messages,
                # so for full-trace analysis we build the string manually
                # and tokenize directly (standard practice — see
                # mitroitskii/interp-experiments/reasoning_circuits).
                prefix = it_model.tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                # prefix already ends with <|Assistant|><think>\n
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
        # Map preset types to internal run_fra sae_type codes
        _run_sae_type = {"sae_hub": "hub", "sae_local": "local", "sae_gemma": "gemma"}[sae_type]
        _run_model = preset.get("model", "gpt2-small")

        with st.spinner("Loading model & SAE…"):
            if sae_type == "sae_gemma":
                load_model_gemma(_run_model, device, hf_token)
                load_sae_gemma(sae_hub_release, sae_hub_id, device)
            else:
                load_model(_run_model, device)
                if sae_type == "sae_hub":
                    load_sae_hub(sae_hub_release, sae_hub_id, device)
                elif sae_local_path and Path(sae_local_path).exists():
                    load_sae_local(sae_local_path, int(layer), device)

        with st.spinner("Computing Feature-Resolved Attention…"):
            fra_data = run_fra(
                text=text,
                layer=int(layer),
                head=int(head),
                hook_point=hook_point,
                sae_type=_run_sae_type,
                sae_hub_release=sae_hub_release,
                sae_hub_id=sae_hub_id,
                sae_local_path=sae_local_path,
                top_k_features=top_k_feat,
                device=device,
                model_name=_run_model,
                chunk_size=chunk_size,
                hf_token=hf_token,
                include_special_tokens=(sae_type != "sae_gemma"),
            )

    st.session_state["fra_data"] = fra_data
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

    # Recompute pairs (filter / top_k / ranking may change without recomputing FRA)
    _agg = cfg.get("agg_mode", "sum")
    pairs = get_ranked_pairs(
        fra_data["indices_np"],
        fra_data["values_np"],
        top_k=cfg["top_k_pairs"],
        filter_self=cfg["filter_self"],
        mode=_agg,
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
    "📊 Top Interactions",
    "🔥 Feature Matrix",
    "✅ Validation",
    "🧩 Max-Act Examples",
    "🔗 Data-Independent",
    "🔬 Ablation",
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
                    f"F{q}→F{k}  ({_pair_metric(q, k, s, c, m):.3f} {_metric_label})"
                    + ("  ⟲" if q == k else "")
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
                _show_heatmap(fig_pos, tick_vals, tick_labels, seq_len,
                              x_title="Key token", y_title="Query token",
                              key="pos_heatmap")

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
            f"Each cell shows the **{_agg}** interaction strength across "
            "position pairs. Only the top features appearing in the ranked list "
            "are shown."
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

        # -- Shared figure builder --

        def _attn_fig(z, hover_label, colorscale="RdBu", zmid=None):
            return go.Figure(go.Heatmap(
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

        # -- Row 1: Logits --

        st.markdown("#### Pre-softmax logits")
        col_std_logit, col_fra_logit = st.columns(2)

        with col_std_logit:
            st.markdown("**Standard** (masked QK scores)")
            _show_heatmap(
                _attn_fig(std_logits, "Logit", zmid=0),
                attn_tick_vals, attn_tick_labels, seq_len,
                compact_height=380, key="attn_std_logit",
            )

        with col_fra_logit:
            st.markdown("**FRA** (signed sum over feature pairs)")
            _show_heatmap(
                _attn_fig(fra_logits, "Logit", zmid=0),
                attn_tick_vals, attn_tick_labels, seq_len,
                compact_height=380, key="attn_fra_logit",
            )

        # -- Row 2: Probs --

        st.markdown("#### Post-softmax probabilities")
        col_std_prob, col_fra_prob = st.columns(2)

        with col_std_prob:
            st.markdown("**Standard** (attention weights)")
            _show_heatmap(
                _attn_fig(std_probs, "Weight"),
                attn_tick_vals, attn_tick_labels, seq_len,
                compact_height=380, key="attn_std_prob",
            )

        with col_fra_prob:
            st.markdown("**FRA** (softmax of FRA logits)")
            _show_heatmap(
                _attn_fig(fra_probs, "Weight"),
                attn_tick_vals, attn_tick_labels, seq_len,
                compact_height=380, key="attn_fra_prob",
            )

        # -- Reconstruction quality metrics --

        st.markdown("---")
        st.markdown("#### Reconstruction Metrics")
        st.caption(
            "Quantitative comparison of FRA-reconstructed attention vs the "
            "model's actual pre-softmax scores (causal region only)."
        )

        # Compare FRA logits (before NaN masking) to standard logits
        _causal = np.tril(np.ones((seq_len, seq_len)))
        _std_causal = fra_data["attn_scores_np"][:seq_len, :seq_len] * _causal
        _fra_causal = fra_logits.copy()
        _fra_causal[causal_mask] = 0.0
        _fra_causal *= _causal

        _diff = np.abs(_std_causal - _fra_causal)
        _norm_std = np.linalg.norm(_std_causal.flatten())
        _fro_rel = float(np.linalg.norm(_diff) / (_norm_std + 1e-10))
        _flat_s = _std_causal.flatten().astype(np.float64)
        _flat_f = _fra_causal.flatten().astype(np.float64)
        _cos = float(np.dot(_flat_s, _flat_f) / (
            np.linalg.norm(_flat_s) * np.linalg.norm(_flat_f) + 1e-10
        ))
        _r2 = float(1 - np.sum((_flat_s - _flat_f) ** 2) / (
            np.sum((_flat_s - _flat_s.mean()) ** 2) + 1e-10
        ))

        vm1, vm2, vm3, vm4 = st.columns(4)
        vm1.metric("Frobenius rel. error", f"{_fro_rel:.1%}")
        vm2.metric("Cosine similarity", f"{_cos:.4f}")
        vm3.metric("R²", f"{_r2:.4f}")
        vm4.metric("Mean abs. error", f"{float(np.mean(_diff)):.4f}")

        _pass_attn = _fro_rel < 0.50
        if _pass_attn:
            st.success(f"Attention reconstruction: PASS (Frobenius error {_fro_rel:.1%} < 50%)")
        else:
            st.warning(f"Attention reconstruction: FAIL (Frobenius error {_fro_rel:.1%} >= 50%)")

# ── Tab 4: Max-Act Examples ────────────────────────────────────────────────

with tab4:
    from fra.max_act import (
        compute_max_acts as _compute_max_acts,
        list_available_features as _list_available_features,
        load_prompts as _load_prompts,
        load_reasoning_prompts as _load_reasoning_prompts,
        load_results as _load_results,
        save_results as _save_results,
    )

    _is_reasoning_preset = (
        sae_type == "crosscoder"
        and PRESETS.get(preset_name, {}).get("is_reasoning", False)
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

# ── Tab 5: QK Circuit ────────────────────────────────────────────────────

def _render_band_table(bands: dict, id_label: str = "Feature"):
    """Render band comparison as a streamlit table."""
    rows = []
    for band_name, entries in bands.items():
        if band_name.startswith("_"):
            continue
        for fid, val in entries:
            rows.append({"Band": band_name, id_label: f"F{fid}", "DI": f"{val:.6f}"})
    import pandas as pd
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _decode_global_band_pairs(bands: dict, sample_idxs: np.ndarray, d_sae: int):
    """Decode flat hist_sample indices into (query_feat, key_feat) per band."""
    decoded = {}
    for band_name, entries in bands.items():
        if band_name.startswith("_"):
            continue
        pairs = []
        for flat_idx, val in entries:
            q_sample = flat_idx // d_sae
            k_feat = flat_idx % d_sae
            q_feat = int(sample_idxs[q_sample]) if q_sample < len(sample_idxs) else flat_idx
            pairs.append((q_feat, k_feat, val))
        decoded[band_name] = pairs
    return decoded


def _render_global_band_table(bands: dict, sample_idxs: np.ndarray, d_sae: int):
    """Render band table for global scan, decoding flat indices to (query, key) pairs."""
    decoded = _decode_global_band_pairs(bands, sample_idxs, d_sae)
    rows = []
    for band_name, pairs in decoded.items():
        for q_feat, k_feat, val in pairs:
            rows.append({
                "Band": band_name,
                "Query Feature": f"F{q_feat}",
                "Key Feature": f"F{k_feat}",
                "DI": f"{val:.6f}",
            })
    import pandas as pd
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_di_histogram(di_values, stats, mark_val=None, mark_label=None):
    """Render DI distribution histogram with SD lines."""
    mean, std = stats["mean"], stats["std"]
    fig = go.Figure()
    # Pre-bin with numpy to avoid sending millions of raw points to the browser.
    # Clip to ±3 SD to zoom in on the meaningful range.
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
                annotation_text=f"{'+' if sign > 0 else '-'}{mult}σ" if mult > 0 else "μ",
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


with tab5:
    st.subheader("QK Circuit — Data-Independent")
    st.caption(
        "Analyse feature pairs through the QK circuit's inherent geometry. "
        "Select a pair manually, from data-dependent (DD) FRA rankings, or from "
        "a global data-independent (DI) scan, then compare DD and DI signals."
    )

    import pandas as pd

    # Shared weight-loading kwargs (safe to build even before FRA is computed)
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

    # ── helpers for cached DI computation ──────────────────────────────────
    if "_tab5_di_weights" not in st.session_state:
        st.session_state["_tab5_di_weights"] = None
    if "_tab5_di_row_cache" not in st.session_state:
        st.session_state["_tab5_di_row_cache"] = {}
    if "_tab5_dd_di_comparison" not in st.session_state:
        st.session_state["_tab5_dd_di_comparison"] = None
    if "_tab5_global_di" not in st.session_state:
        st.session_state["_tab5_global_di"] = None

    def _ensure_di_weights():
        """Load and cache DI weight matrices."""
        if st.session_state["_tab5_di_weights"] is None:
            W_dec, W_Q, W_K_, _attn = _load_di_weights(
                sae_type, int(head), device, **_wt_kw,
            )
            st.session_state["_tab5_di_weights"] = (W_dec, W_Q, W_K_)
        return st.session_state["_tab5_di_weights"]

    def _get_di_row(query_id):
        """Compute (and cache) the full DI row for *query_id*."""
        cache = st.session_state["_tab5_di_row_cache"]
        if query_id not in cache:
            W_dec, W_Q, W_K_ = _ensure_di_weights()
            cache[query_id] = compute_di_row(W_dec, W_Q, W_K_, query_id)
        return cache[query_id]

    def _di_for_pair(q, k):
        """Return the scalar DI value for a (query, key) pair."""
        return float(_get_di_row(q)[k])

    # ── Section 1: Pair Selector + Metrics ─────────────────────────────────
    st.markdown("### Pair selector")

    _t5_pair_mode = st.radio(
        "Selection mode",
        ["Manual", "Top DD pairs", "Top DI pairs"],
        horizontal=True,
        key="_t5_pair_mode",
    )

    _t5_q_sel = None
    _t5_k_sel = None

    if _t5_pair_mode == "Manual":
        _mc1, _mc2 = st.columns(2)
        with _mc1:
            _t5_q_sel = st.number_input(
                "Query feature ID", min_value=0, value=0, key="_t5_man_q",
            )
        with _mc2:
            _t5_k_sel = st.number_input(
                "Key feature ID", min_value=0, value=0, key="_t5_man_k",
            )

    elif _t5_pair_mode == "Top DD pairs":
        if not _has_fra:
            st.info("Compute FRA first to use data-dependent pair ranking.")
        else:
            _t5_agg = cfg.get("agg_mode", "sum")
            _t5_dd_pairs = get_ranked_pairs(
                fra_data["indices_np"], fra_data["values_np"],
                top_k=30, filter_self=False, mode=_t5_agg,
            )
            if not _t5_dd_pairs:
                st.warning("No DD pairs found.")
            else:
                _t5_dd_labels = [
                    f"F{q}\u2192F{k} (score={_pair_metric(q, k, s, c, m):.4f})"
                    for q, k, s, c, m in _t5_dd_pairs
                ]
                _t5_dd_idx = st.selectbox(
                    "Pick a DD pair", range(len(_t5_dd_pairs)),
                    format_func=lambda i: _t5_dd_labels[i],
                    key="_t5_dd_sel",
                )
                _t5_q_sel = _t5_dd_pairs[_t5_dd_idx][0]
                _t5_k_sel = _t5_dd_pairs[_t5_dd_idx][1]

    else:  # Top DI pairs
        _t5_di_go = st.button(
            "Compute global DI top-k", type="primary", key="_t5_di_go",
        )
        if _t5_di_go:
            with st.spinner("Running global DI scan..."):
                W_dec, W_Q, W_K_ = _ensure_di_weights()
                progress = st.progress(0, text="Scanning feature pairs...")

                def _t5_di_progress(i, n):
                    progress.progress(i / n, text=f"Chunk {i}/{n}...")

                result = compute_global_di_topk(
                    W_dec, W_Q, W_K_, top_k=50,
                    progress_callback=_t5_di_progress,
                )
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

    # ── Metrics + Heatmap for the selected pair ────────────────────────────
    if _t5_q_sel is not None and _t5_k_sel is not None:
        st.markdown("---")
        st.markdown(f"#### Pair F{_t5_q_sel} \u2192 F{_t5_k_sel}")

        # Compute DI for this pair
        _t5_di_val = _di_for_pair(_t5_q_sel, _t5_k_sel)
        _t5_di_row = _get_di_row(_t5_q_sel)

        # DI rank / percentile among ALL key features
        _t5_abs_row = np.abs(_t5_di_row)
        _t5_abs_val = abs(_t5_di_val)
        _t5_di_rank = int(np.sum(_t5_abs_row >= _t5_abs_val))  # 1-based rank
        _t5_di_pct = 100.0 * (1.0 - _t5_di_rank / len(_t5_abs_row))

        # DD score (only if FRA is available)
        _t5_dd_val = None
        if _has_fra:
            _t5_agg = cfg.get("agg_mode", "sum")
            # Search for this pair in ranked list
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
        _mc3.metric("DI |rank|", f"{_t5_di_rank:,} / {len(_t5_di_row):,}")
        _mc4.metric("DI percentile", f"{_t5_di_pct:.1f}%")

        # DI histogram with this pair marked
        _t5_di_stats = {
            "mean": float(np.mean(_t5_di_row)),
            "std": float(np.std(_t5_di_row)),
        }
        _render_di_histogram(
            _t5_di_row, _t5_di_stats,
            mark_val=_t5_di_val,
            mark_label=f"F{_t5_q_sel}\u2192F{_t5_k_sel}",
        )

        # Position heatmap (only with FRA data)
        if _has_fra:
            st.markdown("**Position heatmap** — where does this pair interact?")
            _t5_pos_mat = get_position_heatmap(
                fra_data["indices_np"], fra_data["values_np"],
                _t5_q_sel, _t5_k_sel, seq_len,
            )
            if _t5_pos_mat.sum() > 0:
                _t5_tick_labels = [html_lib.escape(t) for t in token_strs]
                _t5_tick_vals = list(range(len(token_strs)))
                _t5_fig_pos = go.Figure(go.Heatmap(
                    z=_t5_pos_mat,
                    x=_t5_tick_vals,
                    y=_t5_tick_vals,
                    colorscale="Blues",
                    hovertemplate=(
                        "Q-pos: %{y}<br>K-pos: %{x}<br>"
                        "Strength: %{z:.4f}<extra></extra>"
                    ),
                ))
                _show_heatmap(
                    _t5_fig_pos, _t5_tick_vals, _t5_tick_labels, seq_len,
                    x_title="Key token", y_title="Query token",
                    key="_t5_pos_heatmap",
                )
            else:
                st.caption("No interactions at any position for this pair.")

    # ── Section 2: DD vs DI Comparison ─────────────────────────────────────
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
        _t5_cmp_pairs = get_ranked_pairs(
            fra_data["indices_np"], fra_data["values_np"],
            top_k=_t5_cmp_n, filter_self=False, mode=_t5_cmp_agg,
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
                _cdi = _di_for_pair(_cq, _ck)
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

    # ── Section 3: Exploration (expander) ──────────────────────────────────
    st.markdown("---")
    with st.expander("Explore single features / global scan"):

        qk_mode = st.radio(
            "Mode",
            ["Fixed Query", "Fixed Key", "Global"],
            horizontal=True,
            key="qk_mode",
        )

        if qk_mode == "Fixed Query":
            qk_col_in, qk_col_btn = st.columns([3, 1])
            with qk_col_in:
                qk_fq_id = st.number_input("Query feature ID", min_value=0, value=0, key="qk_fq_id")
            with qk_col_btn:
                st.markdown("")
                st.markdown("")
                qk_fq_go = st.button("Compute", type="primary", key="qk_fq_btn")

            if qk_fq_go:
                W_dec, W_Q, W_K_ = _ensure_di_weights()
                di_row = compute_di_row(W_dec, W_Q, W_K_, qk_fq_id)
                bands = sample_di_bands(di_row)
                st.session_state["qk_fq_result"] = {
                    "di_vals": di_row, "bands": bands,
                    "query_id": qk_fq_id, "head": int(head),
                }

            if "qk_fq_result" in st.session_state:
                r = st.session_state["qk_fq_result"]
                st.markdown(f"**Query F{r['query_id']}** — top key features by |DI| (H{r['head']})")
                _render_di_histogram(r["di_vals"], r["bands"]["_stats"])
                _render_band_table(r["bands"], id_label="Key Feature")

        elif qk_mode == "Fixed Key":
            qk_col_in, qk_col_btn = st.columns([3, 1])
            with qk_col_in:
                qk_fk_id = st.number_input("Key feature ID", min_value=0, value=0, key="qk_fk_id")
            with qk_col_btn:
                st.markdown("")
                st.markdown("")
                qk_fk_go = st.button("Compute", type="primary", key="qk_fk_btn")

            if qk_fk_go:
                W_dec, W_Q, W_K_ = _ensure_di_weights()
                di_col = compute_di_col(W_dec, W_Q, W_K_, qk_fk_id)
                bands = sample_di_bands(di_col)
                st.session_state["qk_fk_result"] = {
                    "di_vals": di_col, "bands": bands,
                    "key_id": qk_fk_id, "head": int(head),
                }

            if "qk_fk_result" in st.session_state:
                r = st.session_state["qk_fk_result"]
                st.markdown(f"**Key F{r['key_id']}** — top query features by |DI| (H{r['head']})")
                _render_di_histogram(r["di_vals"], r["bands"]["_stats"])
                _render_band_table(r["bands"], id_label="Query Feature")

        else:  # Global
            st.warning(
                "Global scan iterates over all feature pairs. "
                "Takes ~30-60s on GPU, longer on CPU."
            )
            qk_g_topk = st.slider("Top-K global pairs", 10, 200, 50, key="qk_g_topk")
            qk_g_go = st.button("Run Global Scan", type="primary", key="qk_g_btn")

            if qk_g_go:
                W_dec, W_Q, W_K_ = _ensure_di_weights()
                progress = st.progress(0, text="Scanning feature pairs...")

                def _g_progress(i, n):
                    progress.progress(i / n, text=f"Chunk {i}/{n}...")

                result = compute_global_di_topk(
                    W_dec, W_Q, W_K_, top_k=qk_g_topk,
                    progress_callback=_g_progress,
                )
                progress.empty()

                # Build bands from the sampled histogram data
                bands = sample_di_bands(result["hist_sample"])

                st.session_state["qk_global_result"] = {
                    **result, "bands": bands, "head": int(head),
                }

            if "qk_global_result" in st.session_state:
                r = st.session_state["qk_global_result"]
                st.markdown(f"**Global top pairs by |DI|** (H{r['head']})")
                _render_di_histogram(r["hist_sample"], r["bands"]["_stats"])

                # Top pairs table
                top_df = pd.DataFrame({
                    "Query Feature": [f"F{q}" for q in r["query_ids"]],
                    "Key Feature": [f"F{k}" for k in r["key_ids"]],
                    "DI": [f"{v:.6f}" for v in r["di_values"]],
                })
                st.dataframe(top_df, use_container_width=True, hide_index=True)

                st.markdown("**Distribution bands** (sampled from ~500 random query rows)")
                _render_global_band_table(r["bands"], r["hist_sample_idxs"], r["d_sae"])

# ── Tab 6: Ablation ────────────────────────────────────────────────────────

with tab6:
    if not _has_fra:
        st.info("Click **▶ Compute FRA** in the sidebar to run ablation studies.")
    else:
        st.subheader(f"Feature-Pair Ablation — L{layer_} H{head_}")
        st.caption(
            "Ablate selected feature pairs from the FRA tensor and measure the "
            "impact on model output. This reveals which cross-feature interactions "
            "are causally important to this attention head's computation."
        )

        # Build pair selection UI
        _abl_mode = cfg.get("agg_mode", "sum")
        all_pairs_for_ablation = get_ranked_pairs(
            fra_data["indices_np"], fra_data["values_np"],
            top_k=100, filter_self=False, mode=_abl_mode,
        )

        if not all_pairs_for_ablation:
            st.warning("No feature pairs found.")
        else:
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
                    ["Top off-diagonal (i≠j)", "Top on-diagonal (i==j)",
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
                    marker = "⟲" if q == k else "→"
                    st.text(f"  F{q} {marker} F{k}  (avg={avg:.4f}, sum={s:.4f})")
                if len(_sel_pairs) > 15:
                    st.text(f"  ... and {len(_sel_pairs) - 15} more")

            run_abl = st.button("▶  Run Ablation", type="primary")

            if run_abl:
                with st.spinner("Running ablation..."):
                    import torch.nn.functional as _F
                    from fra.ablation_study import (
                        ablate_fra_pairs,
                        reconstruct_scores,
                        run_condition,
                    )
                    from fra.validation import fra_sum_to_attn

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
                        # (b_Q=0, b_K=0; FRA already has 1/sqrt(d_head)
                        #  + RMSNorm)
                        _cc_base, _cc_it = load_model_pair(
                            base_model_name, it_model_name, device,
                            cc_it_arch,
                        )
                        _cc_target = (
                            _cc_base if model_idx == 0 else _cc_it
                        )

                        # Get tokens from the FRA computation
                        _cc_tokens = _cc_target.tokenizer.encode(
                            cfg["text"],
                        )[:128]
                        _cc_tok_t = torch.tensor(
                            _cc_tokens,
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
                        from fra.ablation_study import (
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
                                "Recovery (full→ablated)",
                                f"{rec_full:.3f} → {rec_abl:.3f}",
                                delta=f"{rec_abl - rec_full:+.3f}",
                            )

                        # Attention heatmap comparison
                        st.markdown("**Attention score comparison**")
                        _abl_ticks = list(range(seq_len))
                        _abl_labels = [
                            html_lib.escape(t) for t in token_strs
                        ]

                        def _abl_heatmap(scores_np, title):
                            disp = scores_np.copy()
                            disp[np.triu_indices_from(disp, k=1)] = np.nan
                            return go.Figure(go.Heatmap(
                                z=disp, x=_abl_ticks, y=_abl_ticks,
                                colorscale="RdBu", zmid=0,
                                hovertemplate=(
                                    "Q: %{y}<br>K: %{x}<br>"
                                    "Score: %{z:.2f}<extra></extra>"
                                ),
                            ))

                        hm1, hm2, hm3 = st.columns(3)
                        with hm1:
                            _show_heatmap(
                                _abl_heatmap(
                                    scores_full.cpu().numpy(), "FRA Full",
                                ),
                                _abl_ticks, _abl_labels, seq_len,
                                compact_height=350, key="abl_full",
                            )
                            st.caption("FRA Full")
                        with hm2:
                            _show_heatmap(
                                _abl_heatmap(
                                    scores_abl.cpu().numpy(),
                                    "After Ablation",
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
                                _abl_heatmap(_diff_np, "Difference"),
                                _abl_ticks, _abl_labels, seq_len,
                                compact_height=350, key="abl_diff",
                            )
                            st.caption("Difference (ablated − full)")
