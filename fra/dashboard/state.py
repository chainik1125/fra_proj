"""Dashboard presets and session-state helpers."""

import streamlit as st

# ---------------------------------------------------------------------------
# Presets -- every model / SAE combination the dashboard supports
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
        "trained_on_bos": True,
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
        "trained_on_bos": True,
    },
    "Gemma-2 2B — Gemma-Scope (resid_pre)": {
        "type": "sae_gemma",
        "model": "gemma-2-2b",
        "release": "gemma-scope-2b-pt-res",
        "sae_id_template": "layer_{layer}/width_16k/average_l0_82",
        "hook_point": "hook_resid_pre",
        "layers": list(range(1, 26)),
        "default_layer": 12,
        "n_heads": 8,
        "supports_neuronpedia": False,
        "hf_token_required": True,
        "chunk_size_default": 1,
        "trained_on_bos": False,
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
        "trained_on_bos": False,
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
        "trained_on_bos": True,
    },
}


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------

def get_fra_data():
    return st.session_state.get("fra_data")


def get_fra_config():
    return st.session_state.get("fra_config")
