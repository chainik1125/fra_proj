"""Cached resource loaders (persist across Streamlit reruns, keyed by args)."""

import requests
import streamlit as st
import torch


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
    from fra.coders.sae_lens import SAELensAttentionSAE
    return SAELensAttentionSAE(release, sae_id, device=device)


@st.cache_resource(show_spinner=False)
def load_sae_local(checkpoint_path: str, layer: int, device: str):
    from fra.coders.sae_lens import LocalLn1SAE
    return LocalLn1SAE(checkpoint_path, layer=layer, device=device)


@st.cache_resource(show_spinner=False)
def load_sae_gemma(release: str, sae_id: str, device: str):
    from fra.coders.sae_lens import GemmaScopeSAE
    return GemmaScopeSAE(release, sae_id, device=device)


@st.cache_resource(show_spinner=False)
def load_model_gemma(model_name: str, device: str, hf_token: str = ""):
    from transformer_lens import HookedTransformer
    torch.set_grad_enabled(False)
    kwargs = {}
    if hf_token:
        kwargs["token"] = hf_token
    return HookedTransformer.from_pretrained_no_processing(model_name, device=device, dtype=torch.float16, **kwargs)


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
    from fra.coders.crosscoder import GemmaCrosscoderFRA
    if subfolder:
        return GemmaCrosscoderFRA.from_cc_weights(
            repo_id, subfolder, model_idx=model_idx, device=device,
        )
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
