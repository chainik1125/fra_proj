"""Remote-only single-GPU, bounded-memory Cadenza TopK SAE pilot."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import time

from config import (Config, DATASET, DATASET_REVISION, MODEL_REVISIONS,
                    TRIGGER, class_of, prompt_key, schedules, user_question)
from remote import atomic_json


def require_remote(run):
    # Executing this script on the laptop must fail BEFORE importing ML / downloading.
    expected = os.environ.get("FRA_SAE_REMOTE_RUN")
    if platform.system() != "Linux" or not expected or Path(expected).resolve() != run:
        raise RuntimeError("Use launch.py: training and all ML downloads are remote-only")
    if not os.environ.get("CUDA_VISIBLE_DEVICES") or not os.environ.get("HF_HOME"):
        raise RuntimeError("Remote GPU/cache environment was not configured")


def prepare_data(cfg):
    from datasets import load_dataset
    rows = {
        split: list(load_dataset(DATASET, revision=DATASET_REVISION,
                                 split=split, streaming=True, token=False))
        for split in ("train", "test")
    }
    # Use the official test split, excluding ALL of its questions from SAE training,
    # including their trigger-stripped twins, even if not selected for the small eval.
    heldout_keys = {prompt_key(row["text"]) for row in rows["test"]}
    pools = {label: [] for label in ("sleeper", "non_sleeper")}
    eval_pools = {label: [] for label in pools}
    dropped = 0
    for row in rows["train"]:
        label = class_of(row)
        if prompt_key(row["text"]) in heldout_keys:
            dropped += 1
        else:
            pools[label].append(row["text"])
    for row in rows["test"]:
        eval_pools[class_of(row)].append(row["text"])
    if any(not values for values in [*pools.values(), *eval_pools.values()]):
        raise ValueError("Both training and evaluation require nonempty classes")
    rng = random.Random(cfg.seed + 1)
    for values in eval_pools.values():
        rng.shuffle(values)
    n = min(cfg.eval_per_class, *(len(values) for values in eval_pools.values()))
    eval_pools = {label: values[:n] for label, values in eval_pools.items()}
    return pools, eval_pools, {
        "dataset": DATASET, "revision": DATASET_REVISION,
        "source_rows": {split: len(values) for split, values in rows.items()},
        "train_pool_rows": {label: len(values) for label, values in pools.items()},
        "eval_rows": {label: len(values) for label, values in eval_pools.items()},
        "excluded_train_rows_sharing_test_question": dropped,
        "rows_with_trigger_only_outside_user_turn": {
            split: sum(TRIGGER in r["text"] and TRIGGER not in user_question(r["text"])
                       for r in values) for split, values in rows.items()},
        "balance_unit": "examples, not tokens", "completions": "original teacher-forced dataset text",
        "sampling": "each class shuffled and cycled independently; 50/50 in every harvest batch",
        "heldout_scope": "held out from SAE training; this LM was already finetuned on Cadenza",
    }


class BalancedTexts:
    def __init__(self, pools, seed):
        self.pools = {label: list(values) for label, values in pools.items()}
        self.rng = random.Random(seed)
        self.positions = {label: 0 for label in pools}
        self.epochs = {label: 0 for label in pools}
        self.examples = {label: 0 for label in pools}
        for values in self.pools.values():
            self.rng.shuffle(values)

    def next_batch(self, size):
        batch = []
        for _ in range(size // 2):
            for label, pool in self.pools.items():
                if self.positions[label] == len(pool):
                    self.rng.shuffle(pool)
                    self.positions[label] = 0
                    self.epochs[label] += 1
                batch.append((pool[self.positions[label]], label))
                self.positions[label] += 1
                self.examples[label] += 1
        self.rng.shuffle(batch)
        return batch


def tokenize(tokenizer, texts, context_size, device="cuda"):
    batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                      max_length=context_size, add_special_tokens=False).to(device)
    valid = batch["attention_mask"].bool()
    for token in (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id):
        if token is not None:
            valid &= batch["input_ids"] != token
    return batch, valid


class StopAtAttention(Exception):
    pass


def attention_module(model, layer, hook_kind):
    block = model.model.layers[layer]
    return {"input": block.input_layernorm, "output": block.self_attn.o_proj,
            "resid_mid": block.post_attention_layernorm, "resid_post": block}[hook_kind]


def normalized_input(module, inputs):
    """TL ln1.hook_normalized: native RMSNorm arithmetic, before learned gain."""
    import torch
    x = inputs[0]
    value = x.float()
    return (value * torch.rsqrt(value.square().mean(-1, keepdim=True)
                               + module.variance_epsilon)).to(x.dtype)


def hook_activations(module, inputs, output, hook_kind):
    if hook_kind == "input":
        return normalized_input(module, inputs)
    if hook_kind == "resid_mid":
        return inputs[0]
    if hook_kind == "resid_post":
        return output[0] if isinstance(output, tuple) else output
    if hook_kind == "output":
        return output
    raise ValueError("Unknown activation hook")


def register_stream_replacement(model, layer, hook_kind, transform):
    """Replace the actual stream, including BOTH paths out of resid_mid.

HF Llama saves `residual = hidden_states` immediately before calling MLP RMSNorm.
Thus an out-of-place norm pre-hook would change only the MLP branch. During
no-grad evaluation, copy into that shared residual tensor before RMSNorm so the
skip connection and MLP both receive exactly the replacement, without a lossy
subtract/re-add through attention output. Other hooks are out-of-place.
"""
    import torch
    module = attention_module(model, layer, hook_kind)
    if hook_kind == "resid_mid":
        def pre_hook(module, inputs):
            if torch.is_grad_enabled():
                raise RuntimeError("resid_mid replacement requires no-grad evaluation")
            inputs[0].copy_(transform(inputs[0]).to(inputs[0].dtype))
        return module.register_forward_pre_hook(pre_hook)

    def post_hook(module, inputs, output):
        raw = hook_activations(module, inputs, output, hook_kind)
        changed = transform(raw)
        if hook_kind == "input":
            changed = changed * module.weight
        changed = changed.to(raw.dtype)
        return (changed, *output[1:]) if isinstance(output, tuple) else changed
    return module.register_forward_hook(post_hook)


def capture_attention(model, layer, batch, hook_kind="input"):
    """Capture attention input/output or residual mid/post; stop later computation."""
    captured = []

    def hook(module, inputs, output):
        captured.append(hook_activations(module, inputs, output, hook_kind).detach())
        raise StopAtAttention

    handle = attention_module(model, layer, hook_kind).register_forward_hook(hook)
    try:
        model.model(**batch, use_cache=False)
    except StopAtAttention:
        pass
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError("The requested attention stream was not captured exactly once")
    return captured[0]


class ActivationStore:
    """One reusable BF16 buffer, plus at most one harvest minibatch of leftovers."""
    def __init__(self, model, tokenizer, cfg, texts):
        import torch
        self.model, self.tokenizer, self.cfg, self.texts = model, tokenizer, cfg, texts
        capacity = min(cfg.buffer_tokens, cfg.training_tokens)
        self.storage = torch.empty((capacity, cfg.d_in), device="cuda", dtype=torch.bfloat16)
        self.pending = None
        self.harvested_valid_tokens = 0
        self.valid_by_class = {label: 0 for label in texts.pools}
        self.refills = 0
        self.harvest_seconds = 0.0

    def fill(self, count):
        import torch
        start = time.monotonic()
        position = 0
        empty_batches = 0
        with torch.no_grad():
            while position < count:
                if self.pending is None or len(self.pending) == 0:
                    examples = self.texts.next_batch(self.cfg.harvest_batch)
                    batch, valid = tokenize(self.tokenizer, [x[0] for x in examples], self.cfg.context_size)
                    acts = capture_attention(self.model, self.cfg.layer, batch, self.cfg.hook_kind)
                    self.pending = acts[valid].to(torch.bfloat16)
                    self.harvested_valid_tokens += len(self.pending)
                    for (_, label), row_mask in zip(examples, valid):
                        self.valid_by_class[label] += int(row_mask.sum())
                    del acts, batch, valid
                    if not len(self.pending):
                        empty_batches += 1
                        if empty_batches > 20:
                            raise RuntimeError("No usable activations in 20 consecutive batches")
                        continue
                take = min(count - position, len(self.pending))
                self.storage[position:position + take].copy_(self.pending[:take])
                # A view retains at most one harvest batch, never a growing stream.
                self.pending = self.pending[take:]
                position += take
        torch.cuda.synchronize()
        self.harvest_seconds += time.monotonic() - start
        self.refills += 1
        return self.storage[:count]


def new_sae(cfg, device="cuda"):
    import torch
    from sae_lens.saes.topk_sae import TopKSAE, TopKSAEConfig
    from sae_lens.saes.sae import SAEMetadata
    sae = TopKSAE(TopKSAEConfig(
        d_in=cfg.d_in, d_sae=cfg.d_sae, k=cfg.k, dtype="float32", device=device,
        apply_b_dec_to_input=False, normalize_activations="none",
        rescale_acts_by_decoder_norm=True,
        metadata=SAEMetadata(model_name=cfg.model_name, model_revision=MODEL_REVISIONS[cfg.variant],
                             hook_name=cfg.hook_name, hook_layer=cfg.layer,
                             context_size=cfg.context_size, training_tokens=cfg.training_tokens),
    ))
    with torch.no_grad():
        sae.W_dec.mul_(0.5 / sae.W_dec.norm(dim=-1, keepdim=True))
        sae.W_enc.copy_(sae.W_dec.T)
        sae.b_enc.zero_()
        sae.b_dec.zero_()
    return sae


def reconstruct(sae, raw, scale):
    import torch
    with torch.autocast("cuda", dtype=torch.bfloat16):
        features = sae.encode(raw.float() * scale)
        reconstruction = sae.decode(features).float() / scale
    return reconstruction, features


def reconstruction_eval(model, tokenizer, sae, scale, cfg, pools):
    import torch
    result = {}
    with torch.no_grad():
        for label, texts in pools.items():
            n, sse, l0, cosine_sum, norm_ratio_sum = 0, 0.0, 0.0, 0.0, 0.0
            sum_x = torch.zeros(cfg.d_in, device="cuda", dtype=torch.float64)
            sum_x2 = torch.zeros_like(sum_x)
            active = torch.zeros(cfg.d_sae, device="cuda", dtype=torch.bool)
            for start in range(0, len(texts), cfg.harvest_batch):
                batch, valid = tokenize(tokenizer, texts[start:start + cfg.harvest_batch], cfg.context_size)
                acts = capture_attention(model, cfg.layer, batch, cfg.hook_kind)[valid]
                for raw in acts.split(cfg.batch_tokens):
                    prediction, features = reconstruct(sae, raw, scale)
                    x = raw.float()
                    sse += float((prediction - x).square().sum())
                    cosine_sum += float(torch.nn.functional.cosine_similarity(prediction, x, dim=-1).sum())
                    norm_ratio_sum += float((prediction.norm(dim=-1) / x.norm(dim=-1).clamp_min(1e-8)).sum())
                    l0 += float((features > 0).sum())
                    active |= (features > 0).any(dim=0)
                    sum_x += x.double().sum(0)
                    sum_x2 += x.double().square().sum(0)
                    n += len(x)
            if not n:
                raise RuntimeError(f"No valid eval tokens for {label}")
            total_variance = float((sum_x2 - sum_x.square() / n).sum())
            fvu = sse / total_variance if total_variance > 0 else None
            result[label] = {"examples": len(texts), "tokens": n,
                             "mse_per_component": sse / (n * cfg.d_in), "fvu": fvu,
                             "explained_variance": None if fvu is None else 1 - fvu,
                             "cosine_similarity": cosine_sum / n, "l2_norm_ratio": norm_ratio_sum / n,
                             "l0": l0 / n, "features_active_on_eval": int(active.sum()),
                             "features_inactive_on_eval_fraction": float((~active).float().mean())}
    # Equal class weighting, distinct from pooling all tokens of unequal lengths.
    result["macro_50_50"] = {
        key: sum(result[label][key] for label in pools) / 2
        for key in ("mse_per_component", "l0")
    }
    if all(result[label]["fvu"] is not None for label in pools):
        result["macro_50_50"]["fvu"] = sum(result[label]["fvu"] for label in pools) / 2
    return result


def ce_eval(model, tokenizer, sae, scale, cfg, pools):
    """Teacher-forced CE before/after SAE replacement; chunk the large vocabulary head."""
    import torch
    import torch.nn.functional as F
    result = {}
    with torch.no_grad():
        for label, texts in pools.items():
            sums = {"baseline": 0.0, "sae_replacement": 0.0, "zero_ablation": 0.0}
            tokens = 0
            for start in range(0, len(texts), cfg.harvest_batch):
                batch, valid = tokenize(tokenizer, texts[start:start + cfg.harvest_batch], cfg.context_size)
                target_valid = valid[:, 1:] & batch["attention_mask"][:, :-1].bool()
                targets = batch["input_ids"][:, 1:][target_valid]
                tokens += len(targets)

                def replace(values):
                    changed = values.clone()
                    raw = values[valid]
                    predictions = [reconstruct(sae, x, scale)[0].to(values.dtype)
                                   for x in raw.split(cfg.batch_tokens)]
                    if predictions:
                        changed[valid] = torch.cat(predictions)
                    return changed

                def ablate(values):
                    changed = values.clone()
                    changed[valid] = 0
                    return changed

                for mode in sums:
                    handle = None
                    if mode == "sae_replacement":
                        handle = register_stream_replacement(model, cfg.layer, cfg.hook_kind, replace)
                    elif mode == "zero_ablation":
                        handle = register_stream_replacement(model, cfg.layer, cfg.hook_kind, ablate)
                    try:
                        hidden = model.model(**batch, use_cache=False).last_hidden_state
                    finally:
                        if handle is not None:
                            handle.remove()
                    predictors = hidden[:, :-1][target_valid]
                    for h, target in zip(predictors.split(128), targets.split(128)):
                        logits = model.lm_head(h).float()
                        sums[mode] += float(F.cross_entropy(logits, target, reduction="sum"))
                    del hidden, predictors
            if not tokens:
                raise RuntimeError("No valid CE targets")
            result[label] = {key + "_ce": value / tokens for key, value in sums.items()}
            result[label]["ce_increase"] = (sums["sae_replacement"] - sums["baseline"]) / tokens
            ablation_effect = (sums["zero_ablation"] - sums["baseline"]) / tokens
            result[label]["zero_ablation_ce_increase"] = ablation_effect
            result[label]["ce_loss_recovered"] = (
                (sums["zero_ablation"] - sums["sae_replacement"]) / (sums["zero_ablation"] - sums["baseline"])
                if ablation_effect > 1e-4 else None)
            result[label]["tokens"] = tokens
    result["macro_50_50"] = {
        key: sum(result[label][key] for label in pools) / 2
        for key in ("baseline_ce", "sae_replacement_ce", "zero_ablation_ce", "ce_increase")
    }
    return result


def export_sae(sae, scale, cfg, destination, tokens=None):
    """Fold both dataset scaling and norm-weighted TopK into an ordinary TopK SAE."""
    import torch
    from sae_lens.saes.topk_sae import TopKSAE, TopKSAEConfig
    export_cfg = TopKSAEConfig.from_dict(sae.cfg.to_dict())
    export_cfg.device = "cpu"
    export_cfg.k = sae.activation_fn.k
    export_cfg.rescale_acts_by_decoder_norm = False
    export_cfg.metadata.activation_scale_folded = scale
    export_cfg.metadata.training_tokens = cfg.training_tokens if tokens is None else tokens
    export_cfg.metadata.target_training_tokens = cfg.training_tokens
    export_cfg.metadata.training_recipe = "LlamaScope-inspired Cadenza pilot; see manifest.json"
    exported = TopKSAE(export_cfg)
    with torch.no_grad():
        norms = sae.W_dec.norm(dim=-1).clamp_min(1e-8)
        exported.W_enc.copy_((sae.W_enc * norms[None, :] * scale).cpu())
        exported.b_enc.copy_((sae.b_enc * norms).cpu())
        exported.W_dec.copy_((sae.W_dec / norms[:, None] / scale).cpu())
        exported.b_dec.copy_((sae.b_dec / scale).cpu())
    exported.save_model(destination)


def verify_export(sae, scale, sample, path):
    import torch
    from sae_lens import SAE
    reloaded = SAE.load_from_disk(str(path), device="cuda")
    precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        with torch.no_grad():
            expected = sae(sample.float() * scale).float() / scale
            actual = reloaded(sample.float()).float()
            torch.testing.assert_close(actual, expected, rtol=5e-4, atol=2e-4)
            maximum = float((actual - expected).abs().max())
    finally:
        torch.set_float32_matmul_precision(precision)
    del reloaded
    return {"passed": True, "max_abs_error": maximum, "n_tokens": len(sample)}


def retain_checkpoint(run, sae, scale, cfg, tokens, step, probe, recent_counts=None, window_tokens=None):
    """Immutable, load-tested SAE exports at exact activation-token milestones."""
    parent = run / "checkpoints"
    parent.mkdir(exist_ok=True)
    destination = parent / f"tokens_{tokens:09d}"
    temporary = parent / f".tokens_{tokens:09d}.partial"
    if destination.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint {destination}")
    export_sae(sae, scale, cfg, temporary, tokens=tokens)
    checked = verify_export(sae, scale, probe, temporary)
    record = {"tokens": tokens, "optimizer_step": step, "k": sae.activation_fn.k,
              "path": str(destination), "reload_check": checked}
    if recent_counts is not None:
        record.update(activity_window_tokens=window_tokens,
                      dead_fraction_recent_window=float((recent_counts == 0).float().mean()),
                      below_1e_minus6_frequency_fraction=float(
                          (recent_counts.float() / max(1, window_tokens) < 1e-6).float().mean()))
    atomic_json(temporary / "checkpoint.json", record)
    temporary.rename(destination)
    atomic_json(run / "checkpoint_status.json", record)
    print(json.dumps({"event": "checkpoint_saved", **record}), flush=True)
    return record


def checkpoint(run, sae, optimizer, cfg, tokens, step, scale):
    import torch
    # Only one latest training snapshot is retained. It intentionally omits activations.
    # An optimizer snapshot is provided, but automatic exact-stream resume is not implemented.
    target = run / "checkpoint_latest.pt"
    temporary = run / "checkpoint_latest.tmp"
    torch.save({"sae": sae.state_dict(), "optimizer": optimizer.state_dict(),
                "config": asdict(cfg), "tokens": tokens, "step": step, "scale": scale,
                "activation_buffer_saved": False, "exact_stream_resume_supported": False}, temporary)
    temporary.replace(target)


def training_reference(run, cfg):
    """Read-only equality guard for the explicitly requested hook-only rerun."""
    run = run.resolve()
    task = json.loads((run / "task.json").read_text()) if (run / "task.json").exists() else {}
    if not task.get("parameter_reference"):
        return None
    source = Path(task["parameter_reference"]).resolve()
    if source.parent != run.parent or source == run:
        raise ValueError("Training reference must be a different sibling run")
    content = (source / "config.json").read_bytes()
    reference = json.loads(content)
    summary = json.loads((source / "summary.json").read_text())
    expected = {**reference, "hook_kind": cfg.hook_kind}
    if (reference.get("hook_kind") != "input" or cfg.hook_kind not in ("resid_mid", "resid_post")
            or expected != asdict(cfg) or summary.get("state") != "complete"
            or summary.get("tokens_trained") != cfg.training_tokens
            or summary.get("model") != cfg.model_name
            or summary.get("hook") != f"blocks.{cfg.layer}.ln1.hook_normalized"):
        raise ValueError("Residual training must match its completed input SAE config except hook_kind")
    return {"run": str(source), "config_sha256": hashlib.sha256(content).hexdigest(),
            "changed_fields": {"hook_kind": {"from": "input", "to": cfg.hook_kind}},
            "normalization": "same data-dependent mean-norm calibration rule, recalibrated for this hook"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run = Path(args.run_dir).resolve()
    require_remote(run)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = Config(**json.loads((run / "config.json").read_text())).validate()
    reference = training_reference(run, cfg)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    # Show this worker as a CUDA process while remote downloads occur.
    gpu_claim = torch.empty(64 * 1024**2, device="cuda", dtype=torch.float32)
    gpu_claim.zero_()
    torch.cuda.synchronize()
    started = time.monotonic()
    manifest = cfg.manifest()
    if reference is not None:
        manifest["parameter_reference"] = reference
    manifest.update(gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
                    lm_dtype="bfloat16", sae_parameters="float32", sae_matmuls="bfloat16 autocast",
                    activation_cache="GPU RAM only, never persisted",
                    reference_differences=["Llama 3 sleeper, not Llama 3.1 base",
                                           "50/50 Cadenza examples, not SlimPajama",
                                           "right-padded/truncated separate conversations, no cross-example packing",
                                           "FP32 SAE parameters and Adam states with BF16 matmuls",
                                           "warmup and annealing scaled proportionally to the token budget",
                                           "TopK retained at inference, no JumpReLU calibration"],
                    loss="MSE divided by batch total variance; no auxiliary dead-feature loss")
    atomic_json(run / "manifest.json", manifest)
    pools, eval_pools, data_info = prepare_data(cfg)
    atomic_json(run / "data_summary.json", data_info)
    print(json.dumps({"event": "data_ready", **data_info}), flush=True)
    print(json.dumps({"event": "loading_remote_model", "model": cfg.model_name}), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, revision=MODEL_REVISIONS[cfg.variant],
                                              token=False)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, revision=MODEL_REVISIONS[cfg.variant], token=False,
        torch_dtype=torch.bfloat16, device_map={"": "cuda:0"},
        attn_implementation="sdpa", low_cpu_mem_usage=True,
    ).eval().requires_grad_(False)
    if model.config.hidden_size != cfg.d_in or model.config.num_hidden_layers != 32:
        raise ValueError("Unexpected Llama architecture")
    del gpu_claim
    texts = BalancedTexts(pools, cfg.seed)
    store = ActivationStore(model, tokenizer, cfg, texts)
    buffer = store.fill(cfg.next_buffer_size(0))
    calibration = buffer[:min(8192, len(buffer))].float()
    scale = math.sqrt(cfg.d_in) / float(calibration.norm(dim=-1).mean())
    if not math.isfinite(scale) or scale <= 0:
        raise RuntimeError("Invalid activation normalization")
    probe = calibration[:32].clone()
    del calibration
    sae = new_sae(cfg)
    optimizer = torch.optim.Adam(sae.parameters(), lr=cfg.lr,
                                 betas=(cfg.adam_beta1, cfg.adam_beta2), fused=True)
    print(json.dumps({"event": "initial_eval", "scale": scale}), flush=True)
    initial_eval = reconstruction_eval(model, tokenizer, sae, scale, cfg, eval_pools)
    atomic_json(run / "eval_initial.json", initial_eval)
    active = torch.zeros(cfg.d_sae, dtype=torch.bool, device="cuda")
    recent_counts = torch.zeros(cfg.d_sae, dtype=torch.long, device="cuda")
    recent_window_start = 0
    tokens, step, next_checkpoint = 0, 0, cfg.checkpoint_every_tokens
    train_seconds = 0.0
    last_loss = None
    checkpoints = []
    with (run / "metrics.jsonl").open("a", buffering=1) as log:
        while tokens < cfg.training_tokens:
            permutation = torch.randperm(len(buffer), device="cuda")
            torch.cuda.synchronize()
            train_start = time.monotonic()
            for position in range(0, len(buffer), cfg.batch_tokens):
                raw = buffer[permutation[position:position + cfg.batch_tokens]]
                x = raw.float() * scale
                lr, k = schedules(cfg, step)
                sae.activation_fn.k = k
                optimizer.param_groups[0]["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    features = sae.encode(x)
                    prediction = sae.decode(features)
                mse = (prediction.float() - x).square().sum(-1).mean()
                variance = (x - x.mean(0)).square().sum(-1).mean().clamp_min(1e-8)
                loss = mse / variance
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at step {step}")
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(sae.parameters(), cfg.clip_grad_norm,
                                                          error_if_nonfinite=True)
                optimizer.step()
                with torch.no_grad():
                    # Reference-style max norm constraint, not forced unit normalization.
                    sae.W_dec.div_(sae.W_dec.norm(dim=-1, keepdim=True).clamp_min(1.0))
                    active |= (features > 0).any(0)
                    recent_counts += (features > 0).sum(0)
                tokens += len(raw)
                step += 1
                last_loss = float(loss.detach())
                if step % cfg.log_every_steps == 0 or tokens == cfg.training_tokens:
                    progress = {"event": "train", "tokens": tokens, "target": cfg.training_tokens,
                                "step": step, "loss": last_loss, "lr": lr, "k": k,
                                "l0": float((features > 0).float().sum(-1).mean()),
                                "grad_norm_before_clip": float(grad_norm),
                                "elapsed_seconds": time.monotonic() - started,
                                "buffer_refills": store.refills,
                                "peak_gpu_GiB": torch.cuda.max_memory_allocated() / 1024**3}
                    log.write(json.dumps(progress, allow_nan=False) + "\n")
                    print(json.dumps(progress), flush=True)
                    atomic_json(run / "progress.json", progress)
                if tokens >= next_checkpoint:
                    if tokens != next_checkpoint:
                        raise RuntimeError("A checkpoint boundary was crossed without shortening the batch")
                    checkpoints.append(retain_checkpoint(run, sae, scale, cfg, tokens, step, probe,
                                                          recent_counts, tokens - recent_window_start))
                    checkpoint(run, sae, optimizer, cfg, tokens, step, scale)
                    recent_counts.zero_()
                    recent_window_start = tokens
                    next_checkpoint += cfg.checkpoint_every_tokens
            torch.cuda.synchronize()
            train_seconds += time.monotonic() - train_start
            if tokens < cfg.training_tokens:
                buffer = store.fill(cfg.next_buffer_size(tokens))
    if tokens != cfg.training_tokens:
        raise RuntimeError(f"Token accounting failed: {tokens} != {cfg.training_tokens}")
    sae.activation_fn.k = cfg.k
    if not checkpoints or checkpoints[-1]["tokens"] != tokens:
        checkpoints.append(retain_checkpoint(run, sae, scale, cfg, tokens, step, probe,
                                              recent_counts, tokens - recent_window_start))
        checkpoint(run, sae, optimizer, cfg, tokens, step, scale)
    if step != cfg.steps:
        raise RuntimeError(f"Optimizer-step accounting failed: {step} != {cfg.steps}")
    print(json.dumps({"event": "final_evaluation", "tokens": tokens}), flush=True)
    final_eval = reconstruction_eval(model, tokenizer, sae, scale, cfg, eval_pools)
    atomic_json(run / "eval_final.json", final_eval)
    ce = ce_eval(model, tokenizer, sae, scale, cfg, eval_pools) if cfg.eval_ce else None
    atomic_json(run / "eval_ce.json", ce)
    export_path = run / "sae_final"
    export_path.symlink_to(Path("checkpoints") / f"tokens_{tokens:09d}", target_is_directory=True)
    reload_check = checkpoints[-1]["reload_check"]
    summary = {
        "state": "complete", "model": cfg.model_name, "hook": cfg.hook_name,
        "tokens_trained": tokens, "steps": step, "final_k": cfg.k, "final_loss": last_loss,
        "examples_harvested": texts.examples, "class_epochs_completed": texts.epochs,
        "valid_tokens_harvested": store.harvested_valid_tokens,
        "valid_tokens_harvested_by_class": store.valid_by_class,
        "unused_final_harvest_tokens": store.harvested_valid_tokens - tokens,
        "buffer_refills": store.refills,
        "activation_buffer_capacity_tokens": min(cfg.buffer_tokens, cfg.training_tokens),
        "train_seconds": train_seconds, "harvest_seconds": store.harvest_seconds,
        "total_seconds_including_load_eval_save": time.monotonic() - started,
        "peak_gpu_GiB": torch.cuda.max_memory_allocated() / 1024**3,
        "never_active_during_training_fraction": float((~active).float().mean()),
        "initial_reconstruction": initial_eval, "final_reconstruction": final_eval,
        "teacher_forced_ce": ce, "checkpoint_reload": reload_check,
        "retained_checkpoints": checkpoints,
        "dead_fraction_last_checkpoint_window": checkpoints[-1]["dead_fraction_recent_window"],
        "sae_path": str(export_path), "local_large_files_created": False,
        "interpretation": "100k is a plumbing smoke test, not evidence of SAE convergence"
                          if tokens <= 100_000 else "Pilot; inspect quality before downstream use",
    }
    atomic_json(run / "summary.json", summary)
    print(json.dumps(summary, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
