"""Cached resource loaders (persist across Streamlit reruns, keyed by args).

Each loader is a thin wrapper around a constructor or classmethod, decorated
with ``@st.cache_resource``.  Streamlit hashes function arguments for cache
keys, so each distinct construction path needs its own function — a single
generic ``load_coder(**kwargs)`` would be fragile (kwarg ordering, extra keys,
etc. can bust the cache).  If Streamlit's caching story improves these
wrappers can be consolidated.
"""

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
    from fra.core.coder import FRACoder
    return FRACoder.from_sae_lens(release, sae_id, device=device)


@st.cache_resource(show_spinner=False)
def load_sae_local(checkpoint_path: str, layer: int, device: str):
    from fra.core.coder import FRACoder
    return FRACoder.from_local_sae(checkpoint_path, layer=layer, device=device)


@st.cache_resource(show_spinner=False)
def load_sae_gemma(release: str, sae_id: str, device: str):
    from fra.core.coder import FRACoder
    return FRACoder.from_gemma_scope(release, sae_id, device=device)


@st.cache_resource(show_spinner=False)
def load_model_gemma(model_name: str, device: str, hf_token: str = ""):
    from transformer_lens import HookedTransformer
    torch.set_grad_enabled(False)
    kwargs = {}
    if hf_token:
        kwargs["token"] = hf_token
    return HookedTransformer.from_pretrained(model_name, device=device, dtype=torch.float16, **kwargs)


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
    from fra.core.coder import FRACoder
    return FRACoder.from_hf_crosscoder(
        repo_id, model_idx=model_idx, device=device, subfolder=subfolder,
    )


@st.cache_resource(show_spinner=False)
def load_lora_model(base_model_repo: str, lora_repo: str, device: str):
    """Load a HookedTransformer with a LoRA adapter applied."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from transformer_lens import HookedTransformer
    torch.set_grad_enabled(False)
    hf_tokenizer = AutoTokenizer.from_pretrained(base_model_repo)
    hf_model = AutoModelForCausalLM.from_pretrained(base_model_repo)
    hf_model = PeftModel.from_pretrained(hf_model, lora_repo)
    hf_model = hf_model.merge_and_unload()
    return HookedTransformer.from_pretrained(
        base_model_repo,
        hf_model=hf_model,
        tokenizer=hf_tokenizer,
        device=device,
    )


@st.cache_resource(show_spinner=False)
def load_wandb_crosscoder(crosscoder_name: str, download_dir: str,
                          model_idx: int, device: str):
    """Load a multi-layer crosscoder from W&B artifacts."""
    from fra.core.coder import FRACoder
    return FRACoder.from_wandb_crosscoder(
        crosscoder_name, download_dir, model_idx=model_idx, device=device,
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
