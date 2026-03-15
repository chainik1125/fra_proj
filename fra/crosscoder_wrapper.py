"""
Wrapper for model-diffing crosscoders (science-of-finetuning format) for FRA.

Loads a crosscoder from HuggingFace that jointly encodes residual stream
activations from a base model and an instruct model.  Exposes the interface
that the FRA pipeline expects: .encode() returning [seq, d_sae] and .W_dec
of shape [d_sae, d_model].

Usage:
    xcoder = GemmaCrosscoderFRA.from_pretrained(
        "science-of-finetuning/gemma-2-2b-crosscoder-l13-mu4.1e-02-lr1e-04",
        model_idx=0,  # 0 = base, 1 = instruct
    )
"""

import json

import torch
import torch.nn.functional as F


class GemmaCrosscoderFRA:
    """FRA-compatible wrapper for a model-diffing crosscoder.

    The crosscoder was trained on stacked residual-stream activations from two
    models (base + instruct) at a single layer.  Its decoder weight has shape
    ``(n_models, d_sae, d_model)`` -- one set of directions per model.

    For FRA we need:
      - ``W_dec``: ``(d_sae, d_model)`` -- one model's decoder directions.
        These are projected through W_Q / W_K to compute feature-resolved
        attention scores.
      - ``encode(x_stacked)``: takes ``(seq, n_models, d_model)`` and returns
        ``(seq, d_sae)`` sparse feature activations.

    Note: the decoder vectors live in *residual-stream* space, while W_Q / W_K
    project from *post-LayerNorm* space.  This is an approximation (the same
    one made by hook_z SAEs in the original dashboard).
    """

    def __init__(
        self,
        W_enc: torch.Tensor,       # (n_models, d_model, d_sae)
        b_enc: torch.Tensor,       # (d_sae,)
        W_dec: torch.Tensor,       # (n_models, d_sae, d_model)
        b_dec: torch.Tensor,       # (n_models, d_model) -- centering bias
        model_idx: int = 0,
        device: str = "cuda",
    ):
        self.device = device
        self.model_idx = model_idx

        self._W_enc = W_enc.to(device)          # (n_models, d_model, d_sae)
        self.b_enc = b_enc.to(device)            # (d_sae,)
        self._W_dec_full = W_dec.to(device)      # (n_models, d_sae, d_model)
        self._b_dec = b_dec.to(device)           # (n_models, d_model)

        self.n_models = W_enc.shape[0]
        self.d_model = W_enc.shape[1]
        self.d_sae = W_enc.shape[2]
        self.d_in = self.d_model

        # Single-model decoder slice for FRA
        self.W_dec = self._W_dec_full[model_idx].contiguous()  # (d_sae, d_model)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = "science-of-finetuning/gemma-2-2b-crosscoder-l13-mu4.1e-02-lr1e-04",
        model_idx: int = 0,
        device: str = "cuda",
    ) -> "GemmaCrosscoderFRA":
        """Download and load a crosscoder from HuggingFace.

        Expects the ``dictionary_learning`` save format with:
          - ``config.json``  (activation_dim, dict_size, num_layers)
          - ``model.safetensors``
        """
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        config_path = hf_hub_download(repo_id, "config.json")
        weights_path = hf_hub_download(repo_id, "model.safetensors")

        with open(config_path) as f:
            config = json.load(f)

        sd = load_file(weights_path, device="cpu")

        return cls(
            W_enc=sd["encoder.weight"],
            b_enc=sd["encoder.bias"],
            W_dec=sd["decoder.weight"],
            b_dec=sd["decoder.bias"],
            model_idx=model_idx,
            device=device,
        )

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------

    def encode(self, x_stacked: torch.Tensor) -> torch.Tensor:
        """Encode stacked residual-stream activations from both models.

        Args:
            x_stacked: ``(seq_len, n_models, d_model)``

        Returns:
            ``(seq_len, d_sae)`` -- sparse feature activations (ReLU).
        """
        x_centered = x_stacked - self._b_dec          # (seq, n_models, d_model)
        # Contract over model and d_model dims:
        #   (seq, n_models, d_model) @ (n_models, d_model, d_sae) -> (seq, d_sae)
        pre_act = torch.einsum("smd,mdl->sl", x_centered, self._W_enc) + self.b_enc
        return F.relu(pre_act)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode features back to stacked residual-stream space.

        Args:
            features: ``(seq_len, d_sae)``

        Returns:
            ``(seq_len, n_models, d_model)``
        """
        # (seq, d_sae) @ (n_models, d_sae, d_model) -> (seq, n_models, d_model)
        return torch.einsum("sl,mld->smd", features, self._W_dec_full) + self._b_dec

    def feature_sparsity(self, features: torch.Tensor) -> float:
        return (features == 0).float().mean().item()
