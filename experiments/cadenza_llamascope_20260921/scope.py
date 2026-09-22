"""Llama Scope checkpoint loading and conditional RMSNorm feature transport.

Imported only on Simplex. Layer numbers on residual SAEs and attention are distinct.
Loading follows OpenMOSS b932639, sae.py: standardize_parameters_of_dataset_activation_scaling
and transform_to_unit_decoder_norm, then its released JumpReLU threshold.
"""
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(part)
    return h.hexdigest()


class ScopeSAE:
    def __init__(self, state, cfg, residual_layer, device="cuda"):
        self.residual_layer = residual_layer
        assert cfg["hook_point_in"] == f"blocks.{residual_layer}.hook_resid_post"
        assert cfg["act_fn"] == "jumprelu" and not cfg["apply_decoder_bias_to_pre_encoder"]
        assert cfg["d_model"] == 4096 and cfg["norm_activation"] == "dataset-wise"
        # Do the published load-time transformations in FP32 to avoid cumulative
        # BF16 parameter rounding. Both input and output scale are accounted for.
        s_in = math.sqrt(cfg["d_model"]) / cfg["dataset_average_activation_norm"]["in"]
        s_out = math.sqrt(cfg["d_model"]) / cfg["dataset_average_activation_norm"]["out"]
        dec = state["decoder.weight"].float().to(device) * (s_in / s_out)
        enc = state["encoder.weight"].float().to(device)
        bias = state["encoder.bias"].float().to(device) / s_in
        norms = dec.norm(dim=0)
        assert bool((norms > 0).all()) and bool(torch.isfinite(norms).all())
        self.W_dec = (dec / norms).T.contiguous()
        self.W_enc = (enc * norms[:, None]).T.contiguous()
        self.b_enc = bias * norms
        self.b_dec = state["decoder.bias"].float().to(device) / s_out
        self.threshold = cfg["jump_relu_threshold"]
        self.cfg = cfg

    def encode(self, x):
        with torch.autocast("cuda", enabled=False):
            h = x.float() @ self.W_enc + self.b_enc
            return torch.where(h > self.threshold, h, 0)

    def decode(self, z):
        with torch.autocast("cuda", enabled=False):
            return z.float() @ self.W_dec + self.b_dec

    def __call__(self, x):
        return self.decode(self.encode(x))


def load_scope(expansion, layer, revision, run):
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from remote import atomic_json
    repo = f"OpenMOSS-Team/Llama3_1-8B-Base-LXR-{expansion}x"
    folder = f"Llama3_1-8B-Base-L{layer}R-{expansion}x"
    paths = [hf_hub_download(repo, f"{folder}/{name}", revision=revision, token=False)
             for name in ("hyperparams.json", "checkpoints/final.safetensors")]
    cfg = json.loads(Path(paths[0]).read_text())
    sae = ScopeSAE(load_file(paths[1]), cfg, layer)
    atomic_json(run / "sae_source.json", {
        "repo": repo, "revision": revision, "residual_layer": layer,
        "attention_layer": layer + 1, "release_config": cfg,
        "files": {name: {"path": p, "sha256": sha256(p)} for name, p in zip(("config", "weights"), paths)},
        "parameter_precision": "FP32 inference after published normalization transforms",
        "reference_loading_code": "OpenMOSS/Llamascopium@b932639261697c0642d077c05f2cb7e463d058cb/src/lm_saes/sae.py",
        "model_transfer": "Llama-3.1-8B-Base SAE applied to Dolphin/Llama-3.0 sleeper A",
    })
    return sae


def rms_scale(x, eps):
    return (x.float().square().mean(-1, keepdim=True) + eps).sqrt()


def capture_rank_input(model, tokenizer, sae, layer, batch):
    from train import capture_attention
    from steering import encode
    residual = capture_attention(model, sae.residual_layer, batch, "resid_post")
    assert layer == sae.residual_layer + 1
    norm = model.model.layers[layer].input_layernorm
    scale = rms_scale(residual, norm.variance_epsilon)
    raw = (residual.float() / scale).to(residual.dtype)
    z = encode(sae, residual) / scale.reshape(-1, 1)
    return raw, z


def transported_deltas(sae, residual, valid, candidate, eps):
    from steering import feature_deltas
    scale = rms_scale(residual, eps)
    return {k: v / scale for k, v in feature_deltas(sae, residual, valid, candidate).items()}


@contextmanager
def transported_intervention(model, sae, layer, valid, candidate, alpha, hook_kind="input"):
    import steering
    if not isinstance(sae, ScopeSAE) or hook_kind != "input" or candidate is None or alpha == 0:
        with ORIGINAL_INTERVENTION(model, sae, layer, valid, candidate, alpha, hook_kind):
            yield
        return
    assert layer == sae.residual_layer + 1
    block = model.model.layers[layer]
    state = {"calls": 0, "active": False, "projected": {}}

    def norm_hook(module, inputs, output):
        state["active"] = state["calls"] == 0
        state["calls"] += 1
        if not state["active"]:
            return output
        residual = inputs[0]
        assert residual.shape[:-1] == valid.shape
        deltas = transported_deltas(sae, residual, valid, candidate, module.variance_epsilon)
        if "input" in deltas:
            return output + (alpha * deltas["input"] * module.weight).to(output.dtype)
        for channel, delta in deltas.items():
            projection = getattr(block.self_attn, channel.lower() + "_proj")
            state["projected"][channel] = F.linear(delta * module.weight.float(), projection.weight.float())
        return output

    handles = [block.input_layernorm.register_forward_hook(norm_hook)]
    for channel in ("Q", "K", "V"):
        def patch(module, inputs, output, channel=channel):
            if state["active"] and channel in state["projected"]:
                return output + (alpha * state["projected"][channel]).to(output.dtype)
            return output
        handles.append(getattr(block.self_attn, channel.lower() + "_proj").register_forward_hook(patch))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


ORIGINAL_INTERVENTION = None
ORIGINAL_RANK = None


def install_transport():
    global ORIGINAL_INTERVENTION
    import steering
    if ORIGINAL_INTERVENTION is None:
        ORIGINAL_INTERVENTION = steering.intervention
        steering.intervention = transported_intervention


def rank_transport(model, tokenizer, sae, cfg, pairs, protocol, run):
    """Reuse the existing FRA mathematics, replacing only its capture/encode pair.

    The scoped patches are restored even on exceptions. Actual Q/K attention
    uses native normalized x; feature coefficients use z(residual)/rms(residual).
    """
    import steering
    original_capture, original_encode = steering.capture_attention, steering.encode
    pending = {}

    def encode(s, raw):
        assert s is sae and raw is pending["raw"]
        return pending.pop("z")

    # capture_rank_input imports steering.encode itself; bind a direct reference
    # to the original encoder so the capture does not recursively call this patch.
    def capture(m, layer, batch, kind):
        from train import capture_attention
        assert layer == sae.residual_layer + 1 and kind == "input"
        residual = capture_attention(m, sae.residual_layer, batch, "resid_post")
        norm = m.model.layers[layer].input_layernorm
        scale = rms_scale(residual, norm.variance_epsilon)
        raw = (residual.float() / scale).to(residual.dtype)
        pending["raw"] = raw
        pending["z"] = original_encode(sae, residual) / scale.reshape(-1, 1)
        return raw

    steering.capture_attention, steering.encode = capture, encode
    try:
        return ORIGINAL_RANK(model, tokenizer, sae, cfg, pairs, protocol, run)
    finally:
        steering.capture_attention, steering.encode = original_capture, original_encode
