"""
Wrapper for model-diffing crosscoders (science-of-finetuning format) for FRA.

Loads a crosscoder from HuggingFace that jointly encodes residual stream
activations from a base model and an instruct model.  Exposes the interface
that the FRA pipeline expects: .encode() returning [seq, d_sae] and .W_dec
of shape [d_sae, d_model].

Delegates all weight loading, encoding, and activation-function logic to the
``dictionary_learning`` package (BatchTopKCrossCoder / CrossCoder), which
correctly handles BatchTopK sparsification, activation normalization, and
the HuggingFace hub format.

Usage:
    xcoder = GemmaCrosscoderFRA.from_pretrained(
        "science-of-finetuning/gemma-2-2b-L13-k100-lr1e-04-local-shuffling-CCLoss",
        model_idx=0,  # 0 = base, 1 = instruct
    )
"""

from __future__ import annotations

import torch

from dictionary_learning import BatchTopKCrossCoder, CrossCoder


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

    This class delegates encoding and weight management to the
    ``dictionary_learning`` package's own CrossCoder / BatchTopKCrossCoder
    implementation, which correctly handles BatchTopK activation, threshold-
    based sparsification, and activation normalization.
    """

    def __init__(
        self,
        crosscoder: CrossCoder,
        model_idx: int = 0,
    ):
        self._crosscoder = crosscoder
        self.model_idx = model_idx

        self.d_sae = crosscoder.dict_size
        self.d_model = crosscoder.activation_dim

        # Single-model decoder slice for FRA: (d_sae, d_model)
        # decoder.weight has shape (n_models, d_sae, d_model)
        self.W_dec = crosscoder.decoder.weight[model_idx].detach().contiguous()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = "science-of-finetuning/gemma-2-2b-L13-k100-lr1e-04-local-shuffling-CCLoss",
        model_idx: int = 0,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> "GemmaCrosscoderFRA":
        """Download and load a crosscoder from HuggingFace.

        Uses ``dictionary_learning.BatchTopKCrossCoder.from_pretrained``
        with ``from_hub=True``, which handles downloading ``config.json``
        and ``model.safetensors`` via ``PyTorchModelHubMixin``.

        Falls back to ``CrossCoder`` if the model is not a BatchTopK
        variant.
        """
        try:
            crosscoder = BatchTopKCrossCoder.from_pretrained(
                repo_id,
                from_hub=True,
                device=device,
                dtype=dtype,
            )
        except Exception:
            crosscoder = CrossCoder.from_pretrained(
                repo_id,
                from_hub=True,
                device=device,
                dtype=dtype,
            )

        return cls(crosscoder=crosscoder, model_idx=model_idx)

    @classmethod
    def from_cc_weights(
        cls,
        repo_id: str,
        subfolder: str,
        model_idx: int = 0,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> "GemmaCrosscoderFRA":
        """Load from the older ``cc_weights.pt`` + ``config.json`` format.

        Used by crosscoders that weren't saved via PyTorchModelHubMixin
        (e.g. Mitroitskii reasoning crosscoder).  The state-dict keys are
        identical to the ``dictionary_learning`` format so we just need to
        read the config manually and instantiate the class ourselves.
        """
        import json
        from huggingface_hub import hf_hub_download

        weights_path = hf_hub_download(repo_id, f"{subfolder}/cc_weights.pt")
        config_path = hf_hub_download(repo_id, f"{subfolder}/config.json")

        with open(config_path) as f:
            cfg = json.load(f)

        # Handle nested config (trainer wrapper)
        tcfg = cfg.get("trainer", cfg)
        crosscoder = BatchTopKCrossCoder(
            activation_dim=tcfg["activation_dim"],
            dict_size=tcfg["dict_size"],
            num_layers=2,
            k=tcfg.get("k", 100),
        )
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        crosscoder.load_state_dict(state_dict, strict=False)
        crosscoder = crosscoder.to(device=device, dtype=dtype)

        return cls(crosscoder=crosscoder, model_idx=model_idx)

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(self, x_stacked: torch.Tensor) -> torch.Tensor:
        """Encode stacked residual-stream activations from both models.

        Delegates to the ``dictionary_learning`` crosscoder's ``encode()``
        method, which handles activation normalization and the correct
        sparsification (BatchTopK threshold or ReLU) automatically.

        Args:
            x_stacked: ``(seq_len, n_models, d_model)``

        Returns:
            ``(seq_len, d_sae)`` -- sparse feature activations.
        """
        return self._crosscoder.encode(x_stacked)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode features back to stacked residual-stream space.

        Args:
            features: ``(seq_len, d_sae)``

        Returns:
            ``(seq_len, n_models, d_model)``
        """
        return self._crosscoder.decode(features)

    def feature_sparsity(self, features: torch.Tensor) -> float:
        return (features == 0).float().mean().item()
