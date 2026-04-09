"""Unified coder for SAEs and crosscoders in the FRA pipeline.

All dictionary-learning objects (SAEs, crosscoders, multi-layer crosscoders)
are loaded through ``FRACoder`` classmethods and present the same interface
to the rest of the codebase.

Classmethods handle loading from different sources:
  - ``from_sae_lens``       : SAE Lens hub SAEs (hook_z or ln1)
  - ``from_local_sae``      : locally-trained SAE checkpoints
  - ``from_gemma_scope``    : Gemma-Scope residual-stream SAEs
  - ``from_hf_crosscoder``  : HuggingFace model-diffing crosscoders
  - ``from_wandb_crosscoder``: W&B multi-layer crosscoders (tiny-sleepers)
  - ``from_state_dict``     : raw weight tensors (legacy / simple SAEs)
"""

from __future__ import annotations

import torch


class FRACoder:
    """Unified coder for SAEs and crosscoders in the FRA pipeline.

    After construction every instance exposes the same interface regardless
    of the underlying library or loading source:

      - ``W_dec``            : ``[d_sae, d_model]`` decoder for the default layer
      - ``b_dec``            : ``[d_model]`` decoder bias
      - ``d_sae``, ``d_model``: dimensions
      - ``encode(x)``        : ``[..., d_sae]`` sparse feature activations
      - ``decode(features)``  : reconstruction
      - ``decoder_for_layer(i)`` : ``[d_sae, d_model]`` decoder for layer *i*
      - ``model_idx``         : which model's decoder is exposed (0 for SAEs)
      - ``n_models``, ``n_layers``, ``hookpoints``, ``is_multi_layer``
      - ``dec_norms``         : ``[d_sae]`` or ``None``
      - ``_norm_coeff``       : set during encode for Gemma-Scope, else ``None``
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        *,
        raw_encoder,
        raw_decoder,
        W_dec_full: torch.Tensor,
        b_dec: torch.Tensor,
        d_sae: int,
        d_model: int,
        n_models: int = 1,
        n_layers: int = 1,
        model_idx: int = 0,
        hookpoints: list[str] | None = None,
        attn_layers: list[int] | None = None,
        normalize_activations: bool = False,
        dec_norms: torch.Tensor | None = None,
    ):
        """
        Args:
            raw_encoder: callable(x) -> [seq, d_sae]
            raw_decoder: callable(features) -> reconstruction
            W_dec_full: [d_sae, n_models, n_layers, d_model] canonical shape
            b_dec: [d_model] decoder bias
            d_sae, d_model: dimensions
            n_models: 1 for SAEs, 2 for model-diffing crosscoders
            n_layers: 1 for single-layer, >1 for multi-layer
            model_idx: which model's decoder to expose (0 for SAEs)
            hookpoints: hookpoint names (length = n_layers)
            attn_layers: attention layer each hookpoint feeds into (length =
                n_layers).  Required for multi-layer coders so the pipeline
                knows which decoder slice to use for each attention head.
            normalize_activations: Gemma-Scope-style input norm rescaling
            dec_norms: [d_sae] decoder weight norms (rescale_acts_by_decoder_norm)
        """
        self._raw_encoder = raw_encoder
        self._raw_decoder = raw_decoder
        self._W_dec_full = W_dec_full
        self._b_dec = b_dec
        self.d_sae = d_sae
        self.d_model = d_model
        self.n_models = n_models
        self.n_layers = n_layers
        self.model_idx = model_idx
        self.hookpoints = hookpoints or []
        self.attn_layers = attn_layers or []
        self._normalize_activations = normalize_activations
        self.dec_norms = dec_norms
        self._norm_coeff: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    @property
    def W_dec(self) -> torch.Tensor:
        """Decoder weight for the default (first) layer: ``[d_sae, d_model]``."""
        return self.decoder_for_layer(0)

    @property
    def b_dec(self) -> torch.Tensor:
        """Decoder bias: ``[d_model]``."""
        return self._b_dec

    @property
    def is_multi_layer(self) -> bool:
        return self.n_layers > 1

    def decoder_for_layer(self, layer_idx: int = 0) -> torch.Tensor:
        """Return ``[d_sae, d_model]`` decoder for hookpoint at *layer_idx*."""
        return self._W_dec_full[:, self.model_idx, layer_idx, :]

    def decoder_for_attn_layer(self, attn_layer: int) -> torch.Tensor:
        """Return ``[d_sae, d_model]`` decoder for the hookpoint feeding
        the given attention layer.

        Requires ``attn_layers`` to have been set at init (multi-layer coders).
        """
        try:
            hook_idx = self.attn_layers.index(attn_layer)
        except ValueError:
            raise ValueError(
                f"Attention layer {attn_layer} not in attn_layers={self.attn_layers}"
            )
        return self.decoder_for_layer(hook_idx)

    def hookpoint_for_attn_layer(self, attn_layer: int) -> str:
        """Return the hookpoint name that feeds the given attention layer."""
        hook_idx = self.attn_layers.index(attn_layer)
        return self.hookpoints[hook_idx]

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode activations to sparse features.

        Input shape depends on the coder type:
          - SAE:                  ``[seq, d_model]``
          - single-layer XC:     ``[seq, n_models, d_model]``
          - multi-layer XC:      ``[batch, n_models, n_layers, d_model]``

        Returns:
            ``[seq, d_sae]`` sparse feature activations.
        """
        if self._normalize_activations and x.dim() == 2:
            x_norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            self._norm_coeff = (self.d_model ** 0.5) / x_norms
            x = x * self._norm_coeff
        return self._raw_encoder(x)

    @torch.no_grad()
    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode features back to activation space."""
        x_hat = self._raw_decoder(features)
        if self._normalize_activations and self._norm_coeff is not None:
            x_hat = x_hat / self._norm_coeff
        return x_hat

    # ------------------------------------------------------------------
    # Classmethods — loading from different sources
    # ------------------------------------------------------------------

    @classmethod
    def from_sae_lens(
        cls,
        release: str,
        sae_id: str,
        device: str = "cuda",
    ) -> FRACoder:
        """Load a pre-trained SAE from the SAE Lens hub.

        Handles both hook_z (concatenated-heads) and ln1 (d_model) SAEs.
        """
        from sae_lens import SAE

        sae = SAE.from_pretrained(release, sae_id, device=device)
        if hasattr(sae, "turn_off_forward_pass_hook_z_reshaping"):
            sae.turn_off_forward_pass_hook_z_reshaping()

        d_sae = sae.cfg.d_sae
        d_model = sae.cfg.d_in

        W_dec = sae.W_dec.detach()                      # [d_sae, d_model]
        W_dec_full = W_dec.unsqueeze(1).unsqueeze(2)     # [d_sae, 1, 1, d_model]
        b_dec = sae.b_dec.detach()

        cfg = sae.cfg
        rescale = getattr(cfg, "rescale_acts_by_decoder_norm", False)
        dec_norms = W_dec.norm(dim=-1) if rescale else None

        def _encode(x):
            if x.dim() == 3 and x.shape[-2:] == (12, 64):
                x = x.flatten(-2, -1)
            if x.dim() > 2:
                batch_shape = x.shape[:-1]
                x = x.reshape(-1, d_model)
                f = sae.encode(x)
                return f.reshape(*batch_shape, d_sae)
            return sae.encode(x)

        def _decode(f):
            if f.dim() > 2:
                batch_shape = f.shape[:-1]
                f = f.reshape(-1, d_sae)
                r = sae.decode(f)
                return r.reshape(*batch_shape, d_model)
            return sae.decode(f)

        return cls(
            raw_encoder=_encode,
            raw_decoder=_decode,
            W_dec_full=W_dec_full,
            b_dec=b_dec,
            d_sae=d_sae,
            d_model=d_model,
            dec_norms=dec_norms,
        )

    @classmethod
    def from_local_sae(
        cls,
        checkpoint_path: str,
        layer: int,
        device: str = "cuda",
    ) -> FRACoder:
        """Load a locally-trained SAE saved by ``sae.save_model()``."""
        from sae_lens import SAE

        sae = SAE.load_from_disk(checkpoint_path, device=device)
        sae = sae.to(device)

        d_sae = sae.cfg.d_sae
        d_model = sae.cfg.d_in

        W_dec = sae.W_dec.detach()
        W_dec_full = W_dec.unsqueeze(1).unsqueeze(2)
        b_dec = sae.b_dec.detach()

        cfg = sae.cfg
        rescale = getattr(cfg, "rescale_acts_by_decoder_norm", False)
        dec_norms = W_dec.norm(dim=-1) if rescale else None

        return cls(
            raw_encoder=sae.encode,
            raw_decoder=sae.decode,
            W_dec_full=W_dec_full,
            b_dec=b_dec,
            d_sae=d_sae,
            d_model=d_model,
            dec_norms=dec_norms,
        )

    @classmethod
    def from_gemma_scope(
        cls,
        release: str,
        sae_id: str,
        device: str = "cuda",
    ) -> FRACoder:
        """Load a Gemma-Scope residual-stream SAE.

        Sets ``normalize_activations=True`` so that encode/decode apply
        the per-token constant-norm rescaling Gemma-Scope was trained with.
        """
        from sae_lens import SAE

        result = SAE.from_pretrained(release, sae_id, device=device)
        sae = result[0] if isinstance(result, tuple) else result

        d_sae = sae.cfg.d_sae
        d_model = sae.cfg.d_in

        W_dec = sae.W_dec.detach()
        W_dec_full = W_dec.unsqueeze(1).unsqueeze(2)
        b_dec = sae.b_dec.detach()

        return cls(
            raw_encoder=sae.encode,
            raw_decoder=sae.decode,
            W_dec_full=W_dec_full,
            b_dec=b_dec,
            d_sae=d_sae,
            d_model=d_model,
            normalize_activations=True,
        )

    @classmethod
    def from_hf_crosscoder(
        cls,
        repo_id: str,
        model_idx: int = 0,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        subfolder: str = "",
    ) -> FRACoder:
        """Load a model-diffing crosscoder from HuggingFace.

        Handles both ``BatchTopKCrossCoder`` (standard format) and the older
        ``cc_weights.pt`` format (Mitroitskii reasoning crosscoder).
        """
        from dictionary_learning import BatchTopKCrossCoder, CrossCoder

        if subfolder:
            return cls._from_cc_weights(
                repo_id, subfolder, model_idx, device, dtype,
            )

        try:
            cc = BatchTopKCrossCoder.from_pretrained(
                repo_id, from_hub=True, device=device, dtype=dtype,
            )
        except Exception:
            cc = CrossCoder.from_pretrained(
                repo_id, from_hub=True, device=device, dtype=dtype,
            )

        return cls._wrap_dictionary_learning_cc(cc, model_idx)

    @classmethod
    def _from_cc_weights(
        cls,
        repo_id: str,
        subfolder: str,
        model_idx: int,
        device: str,
        dtype: torch.dtype,
    ) -> FRACoder:
        """Load from the older ``cc_weights.pt`` + ``config.json`` format."""
        import json
        from huggingface_hub import hf_hub_download
        from dictionary_learning import BatchTopKCrossCoder

        weights_path = hf_hub_download(repo_id, f"{subfolder}/cc_weights.pt")
        config_path = hf_hub_download(repo_id, f"{subfolder}/config.json")

        with open(config_path) as f:
            cfg = json.load(f)

        tcfg = cfg.get("trainer", cfg)
        cc = BatchTopKCrossCoder(
            activation_dim=tcfg["activation_dim"],
            dict_size=tcfg["dict_size"],
            num_layers=2,
            k=tcfg.get("k", 100),
        )
        state_dict = torch.load(
            weights_path, map_location=device, weights_only=True,
        )
        cc.load_state_dict(state_dict, strict=False)
        cc = cc.to(device=device, dtype=dtype)

        return cls._wrap_dictionary_learning_cc(cc, model_idx)

    @classmethod
    def _wrap_dictionary_learning_cc(
        cls, cc, model_idx: int,
    ) -> FRACoder:
        """Build an FRACoder from a ``dictionary_learning`` CrossCoder."""
        d_sae = cc.dict_size
        d_model = cc.activation_dim

        # decoder.weight: [n_models, d_sae, d_model]
        W_dec_per_model = cc.decoder.weight.detach()     # [n_models, d_sae, d_model]
        n_models = W_dec_per_model.shape[0]
        # canonical: [d_sae, n_models, n_layers=1, d_model]
        W_dec_full = W_dec_per_model.permute(1, 0, 2).unsqueeze(2).contiguous()
        b_dec = cc.decoder.bias[model_idx].detach().contiguous()

        return cls(
            raw_encoder=cc.encode,
            raw_decoder=cc.decode,
            W_dec_full=W_dec_full,
            b_dec=b_dec,
            d_sae=d_sae,
            d_model=d_model,
            n_models=n_models,
            model_idx=model_idx,
        )

    @classmethod
    def from_wandb_crosscoder(
        cls,
        crosscoder_name: str,
        download_dir: str,
        model_idx: int = 0,
        device: str = "cuda",
    ) -> FRACoder:
        """Load a multi-layer crosscoder from W&B artifacts (tiny-sleepers).

        The crosscoder has ``W_dec_HXD`` of shape
        ``[hidden_dim, n_models, n_hookpoints, d_model]`` — already in
        canonical form.
        """
        from pathlib import Path
        from crosscode.models.acausal_crosscoder import ModelHookpointAcausalCrosscoder

        path = Path(download_dir) / crosscoder_name
        cc = ModelHookpointAcausalCrosscoder.load(path, device=device)

        hookpoints = [
            "blocks.0.hook_resid_pre",
            "blocks.0.hook_resid_post",
            "blocks.1.hook_resid_post",
            "blocks.2.hook_resid_post",
            "blocks.3.hook_resid_post",
        ]
        # Each hookpoint feeds into the next attention layer.
        # hook_resid_pre at block 0 → attn layer 0, hook_resid_post at
        # block 0 → attn layer 1, etc.  The last hookpoint (resid_post
        # block 3) has no downstream attention layer.
        attn_layers = [0, 1, 2, 3, -1]

        # W_dec_LMPD: [n_latents, n_models, n_hookpoints, d_model]
        W_dec_full = cc.W_dec_LMPD.detach()
        d_sae = W_dec_full.shape[0]
        n_models = W_dec_full.shape[1]
        n_layers = W_dec_full.shape[2]
        d_model = W_dec_full.shape[3]

        b_dec_raw = getattr(cc, "b_dec_MPD", None)
        if b_dec_raw is not None:
            b_dec = b_dec_raw[0, 0, :].detach()
        else:
            b_dec = torch.zeros(d_model, device=W_dec_full.device)

        return cls(
            raw_encoder=lambda x: cc.forward_train(x).latents_BL,
            raw_decoder=cc.decode_BMPD,
            W_dec_full=W_dec_full,
            b_dec=b_dec,
            d_sae=d_sae,
            d_model=d_model,
            n_models=n_models,
            n_layers=n_layers,
            model_idx=model_idx,
            hookpoints=hookpoints,
            attn_layers=attn_layers,
        )

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        device: str = "cuda",
    ) -> FRACoder:
        """Load from a raw state-dict with W_enc, W_dec, b_enc, b_dec keys.

        Provides a simple ReLU SAE encode/decode.
        """
        import torch.nn.functional as F

        W_enc = state_dict["W_enc"].to(device)  # [d_model, d_sae]
        W_dec = state_dict["W_dec"].to(device)  # [d_sae, d_model]
        b_enc = state_dict["b_enc"].to(device)  # [d_sae]
        b_dec = state_dict["b_dec"].to(device)  # [d_model]

        d_model, d_sae = W_enc.shape

        W_dec_full = W_dec.unsqueeze(1).unsqueeze(2)  # [d_sae, 1, 1, d_model]

        def _encode(x):
            return F.relu(x @ W_enc + b_enc)

        def _decode(f):
            return f @ W_dec + b_dec

        return cls(
            raw_encoder=_encode,
            raw_decoder=_decode,
            W_dec_full=W_dec_full,
            b_dec=b_dec,
            d_sae=d_sae,
            d_model=d_model,
        )
