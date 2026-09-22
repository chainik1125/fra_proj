"""Numerical invariants for release conversion and channel-specific transport."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "lib"))
import pytest
import torch
import scope


def fake_sae(d=32, f=8):
    sae = scope.ScopeSAE.__new__(scope.ScopeSAE)
    sae.residual_layer = 0
    sae.W_enc = torch.randn(d, f) * .2
    sae.W_dec = torch.randn(f, d) * .2
    sae.b_enc = torch.ones(f)
    sae.b_dec = torch.zeros(d)
    sae.threshold = .1
    return sae


def test_checkpoint_normalization_preserves_reference_decoder_algebra():
    torch.manual_seed(91)
    d, f = 4096, 9
    state = {"encoder.weight": torch.randn(f, d) * .03,
             "decoder.weight": torch.randn(d, f) * .02,
             "encoder.bias": torch.randn(f), "decoder.bias": torch.randn(d)}
    cfg = {"hook_point_in": "blocks.7.hook_resid_post", "act_fn": "jumprelu",
           "apply_decoder_bias_to_pre_encoder": False, "d_model": d,
           "norm_activation": "dataset-wise", "dataset_average_activation_norm": {"in": 8., "out": 16.},
           "jump_relu_threshold": .35}
    sae = scope.ScopeSAE(state, cfg, 7, device="cpu")
    x = torch.randn(5, d)
    # Independent unfused normalized-domain reference: transform the threshold
    # per feature, then decode in original units. Unequal in/out scales catch
    # accidental assumptions that would be invisible in residual-only fixtures.
    si, so = 8., 4.
    norm = (state["decoder.weight"] * si / so).norm(dim=0)
    pre = (si * x) @ state["encoder.weight"].T + state["encoder.bias"]
    active = torch.where(pre > cfg["jump_relu_threshold"] * si / norm, pre, 0)
    ref = (active @ state["decoder.weight"].T + state["decoder.bias"]) / so
    torch.testing.assert_close(sae(x), ref, atol=1e-5, rtol=1e-4)


def test_transport_uses_full_token_scale_and_valid_mask():
    torch.manual_seed(3)
    sae = fake_sae()
    x = torch.randn(2, 4, 32)
    x[1] *= 9
    valid = torch.tensor([[False, True, True, True], [True, True, False, True]])
    z = sae.encode(x)
    got = scope.transported_deltas(sae, x, valid, {"method": "ov", "features": [2]}, 1e-5)
    expected = -(z[..., 2] * valid)[..., None] * sae.W_dec[2] / scope.rms_scale(x, 1e-5)
    assert set(got) == {"V"}
    torch.testing.assert_close(got["V"], expected)
    assert not got["V"][~valid].count_nonzero()


def test_qkov_gates_use_unscaled_residual_feature_activity():
    sae = fake_sae(d=4, f=4)
    sae.W_enc = torch.eye(4)
    sae.W_dec = torch.eye(4)
    sae.b_enc.zero_()
    x = torch.tensor([[[1., 2., 0., 1.], [1., 0., 3., 1.], [1., 2., 3., 1.]]])
    out = scope.transported_deltas(sae, x, torch.ones(1, 3, dtype=torch.bool),
                                   {"method": "qkov", "features": [0, 1, 2]}, 1e-5)
    assert out["Q"][0, :, 0].count_nonzero() == 3
    assert out["K"][0, :, 1].count_nonzero() == 1
    assert out["V"][0, :, 2].count_nonzero() == 1


def test_real_tiny_llama_channel_isolation_zero_identity_and_cleanup():
    from transformers import LlamaConfig, LlamaForCausalLM
    import steering
    torch.manual_seed(7)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)).eval()
    sae = fake_sae()
    scope.install_transport()
    tokens = torch.tensor([[1, 5, 9, 12], [1, 6, 8, 11]])
    valid = torch.tensor([[False, True, True, True]] * 2)
    candidate = {"method": "ov", "features": [0]}
    block = model.model.layers[1]
    observed = {}

    def capture(name):
        def hook(m, i, o):
            observed[name] = o.detach().clone()
        return hook

    def measure(alpha, cand=candidate):
        observed.clear()
        with steering.intervention(model, sae, 1, valid, cand, alpha, "input"):
            handles = [getattr(block.self_attn, c + "_proj").register_forward_hook(capture(c)) for c in ("q", "k", "v")]
            try:
                with torch.no_grad():
                    logits = model(tokens).logits.clone()
            finally:
                for h in handles:
                    h.remove()
        return logits, {k: v.clone() for k, v in observed.items()}

    baseline, channels = measure(0)
    edited, modified = measure(2)
    for c in ("q", "k"):
        torch.testing.assert_close(channels[c], modified[c], rtol=0, atol=0)
    assert (channels["v"] - modified["v"]).abs().max() > 0
    torch.testing.assert_close(channels["v"][~valid], modified["v"][~valid], rtol=0, atol=0)
    restored, _ = measure(0)
    torch.testing.assert_close(baseline, restored, rtol=0, atol=0)
    assert not block.input_layernorm._forward_hooks
    assert all(not getattr(block.self_attn, c + "_proj")._forward_hooks for c in ("q", "k", "v"))


def test_transport_only_edits_first_prefill():
    from types import SimpleNamespace
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(12)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)).eval()
    sae = fake_sae()
    scope.install_transport()
    block = model.model.layers[1]
    valid = torch.ones(1, 3, dtype=torch.bool)
    x = torch.randn(1, 3, 32)
    y = torch.randn(1, 1, 32)
    candidate = {"method": "single", "features": [0]}
    ref = block.input_layernorm(y)
    with scope.transported_intervention(model, sae, 1, valid, candidate, 2):
        block.input_layernorm(x)
        got = block.input_layernorm(y)
    torch.testing.assert_close(got, ref, rtol=0, atol=0)
