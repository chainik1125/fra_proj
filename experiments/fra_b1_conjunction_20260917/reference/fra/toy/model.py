"""The 1L/1H attention-only model, with every Appendix-A correction stripped.

This module is the single place where the "bare equations" decisions live, so
adding a correction back later is a diff in one file (brief section 2).

What is stripped, and why
-------------------------
* ``attn_only=True``      -- no MLP. A 1L/1H attention-only transformer can
                             express skip-trigrams and nothing else, which is
                             exactly the QK/OV factorisation FRA decomposes.
* ``normalization_type=None`` -- no LayerNorm, so no A.4 RMSNorm correction.
* absolute (learned) positions, **zeroed and frozen** -- no RoPE, so no A.3
  correction. See below.
* ``b_Q = b_K = 0``, frozen (``cfg.zero_qk_biases``) -- kills the Eqs 13-15 bias
  terms, so the acceptance test is the bare equation
  ``FRA_QK[q,k].sum() == s[q,k]`` with nothing added.
* ``b_V = b_O = 0``, frozen (``cfg.zero_ov_biases``) -- the OV analogue. With them live,
  ``attn_out = (sum_k A[q,k] (x_k W_V + b_V)) W_O + b_O`` carries two constant
  residues (``b_V W_O`` survives because the attention row sums to one, and
  ``b_O`` is unconditional) that the feature decomposition cannot express, so OV
  exactness would fail even though QK exactness held. No expressiveness is lost:
  both are position-independent constants, and ``unembed.b_U`` already spans that.

Both bias groups are **config flags**, not hardcoded. Turning one back on is
precisely one of the "add corrections back one at a time" commits, and it should
be a flag flip rather than a diff.
* ``W_E`` frozen to the planted construction -- see below.

The two freezes that the ground truth depends on
------------------------------------------------
``W_pos = 0``: oracle FRA requires ``resid_pre == f @ W_dec`` exactly. A
positional term is a residue the feature basis cannot express, so exactness
would fail by construction. This is safe *here* because the task is
content-addressed ("attend where mu* fires"), not positional -- causal masking
supplies all the ordering the rule needs.

``W_E`` frozen at ``M @ feature_directions``: if the embedding trains, the model
drifts off the planted directions and the ground truth becomes fiction. That
voids the experiment rather than merely degrading it.

Consequence: the only trainable parameters are ``W_Q, W_K, W_V, W_O`` and the
unembedding. The model's entire learnable content is the QK/OV circuit.
"""

from __future__ import annotations

import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from fra.toy.config import ToyConfig
from fra.toy.dgp import FeatureMatchedRetrieval

RESID_PRE_HOOK = "blocks.0.hook_resid_pre"


def build_config(cfg: ToyConfig) -> HookedTransformerConfig:
    return HookedTransformerConfig(
        n_layers=1,
        n_heads=1,
        d_model=cfg.d_model,
        d_head=cfg.d_head,
        d_mlp=None,
        attn_only=True,
        act_fn=None,
        normalization_type=None,
        positional_embedding_type="standard",
        d_vocab=cfg.d_vocab,
        n_ctx=cfg.seq_len,
        device="cpu",
        seed=cfg.seed,
    )


def build_model(dgp: FeatureMatchedRetrieval) -> HookedTransformer:
    """Construct the model and apply every freeze the ground truth depends on."""
    cfg = dgp.cfg
    model = HookedTransformer(build_config(cfg))

    with torch.no_grad():
        # W_E is the planted construction; this is what makes oracle FRA exact.
        model.embed.W_E.copy_(dgp.embedding_matrix)
        # No positional signal at all.
        model.pos_embed.W_pos.zero_()
        for bias in frozen_biases(cfg):
            getattr(model.blocks[0].attn, bias).zero_()

    model.embed.W_E.requires_grad_(False)
    model.pos_embed.W_pos.requires_grad_(False)
    for bias in frozen_biases(cfg):
        getattr(model.blocks[0].attn, bias).requires_grad_(False)

    return model


def frozen_biases(cfg: ToyConfig) -> tuple[str, ...]:
    """Attention biases zeroed and frozen under the current correction flags."""
    names: tuple[str, ...] = ()
    if cfg.zero_qk_biases:
        names += ("b_Q", "b_K")
    if cfg.zero_ov_biases:
        names += ("b_V", "b_O")
    return names


def trainable_parameters(model: HookedTransformer) -> list[tuple[str, torch.nn.Parameter]]:
    """Named parameters that will actually receive gradients."""
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def assert_freezes_hold(model: HookedTransformer, dgp: FeatureMatchedRetrieval) -> None:
    """Re-check the freezes. Cheap, and catches an optimizer that was handed
    ``model.parameters()`` wholesale instead of the trainable subset."""
    assert torch.equal(model.embed.W_E, dgp.embedding_matrix), "W_E drifted off the planted directions"
    assert torch.all(model.pos_embed.W_pos == 0), "W_pos is no longer zero"
    for bias in frozen_biases(dgp.cfg):
        assert torch.all(getattr(model.blocks[0].attn, bias) == 0), f"{bias} is no longer zero"
