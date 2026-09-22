"""Overnight worker using a frozen copy of the existing Cadenza measurement code."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lib"))
from config import Config, MODEL_REVISIONS
from remote import atomic_json


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    from scope import sha256
    return sha256(path)


def get_splits(tokenizer, cfg, protocol, reference):
    from train import prepare_data
    from steering import make_pairs, split_pairs
    pools, heldout, _ = prepare_data(Config(**{**asdict(cfg), "eval_per_class": 10000}))
    train = make_pairs(pools, tokenizer, cfg.context_size)
    test = make_pairs(heldout, tokenizer, cfg.context_size)
    big = split_pairs(train, test, {**protocol, "test_pairs": 192})
    splits = {**big, "test": big["test"][:64]}
    legacy, fresh = big["test"][64:128], big["test"][128:192]
    expected = read(reference / "protocol.json")["splits"]
    assert {k: [p["key"] for p in v] for k, v in splits.items()} == expected
    all_keys = [p["key"] for rows in [*splits.values(), legacy, fresh] for p in rows]
    assert len(all_keys) == len(set(all_keys))
    assert not {p["key"] for p in fresh}.intersection(p["key"] for p in train)
    return splits, legacy, fresh


def evaluate_frozen(model, tokenizer, sae, cfg, pairs, protocol, run, choices, prefix):
    from caa_eval import evaluate_setting
    from restoration import clean_reference_batches, paired_js_interval
    batches = clean_reference_batches(model, tokenizer, sae, cfg, pairs, protocol)
    base, base_rows = evaluate_setting(model, tokenizer, cfg, batches, None, 0, protocol, sae)
    atomic_json(run / f"{prefix}_baseline.json", {"metrics": base, "rows": base_rows})
    result, cache = {}, {}
    for rule, choice in choices.items():
        candidate, alpha = choice["candidate"], choice["alpha"]
        key = (json.dumps(candidate, sort_keys=True), alpha)
        if key not in cache:
            cache[key] = (base, base_rows) if alpha == 0 else evaluate_setting(
                model, tokenizer, cfg, batches, candidate, alpha, protocol, sae)
        metrics, rows = cache[key]
        extra = {"blank_triggered": sum(not r["steered_triggered_text"].strip() for r in rows),
                 "blank_clean": sum(not r["steered_clean_text"].strip() for r in rows)}
        result[rule] = {"candidate": candidate, "alpha": alpha, "metrics": metrics,
                        **extra, **paired_js_interval(rows, base_rows, protocol["seed"])}
        atomic_json(run / f"{prefix}_{rule}.json", {**result[rule], "rows": rows})
    return {"keys": [p["key"] for p in pairs], "baseline": base, "results": result}


def plumbing(model, tokenizer, sae, cfg, pairs, run):
    """Real-activation check of release algebra and transported normalization."""
    import torch
    from scope import ScopeSAE, rms_scale, transported_deltas
    from steering import input_batch, encode
    from train import capture_attention
    batch, valid = input_batch(tokenizer, [pairs[0]["clean"], pairs[0]["sleeper"]])
    if not isinstance(sae, ScopeSAE):
        return
    x = capture_attention(model, sae.residual_layer, batch, "resid_post")
    z = encode(sae, x).reshape(*x.shape[:-1], -1)
    recon = sae.decode(z)
    # Reconstruction can be very poor under model transfer. Evaluate the exact
    # error-inclusive algebra in FP64 so cancellation does not masquerade as a
    # hook/indexing error. This does not relax the separate model-norm check.
    linear = z.double() @ sae.W_dec.double() + sae.b_dec.double()
    err = x.double() - linear
    norm = model.model.layers[sae.residual_layer + 1].input_layernorm
    s = rms_scale(x, norm.variance_epsilon)
    recomposed = (linear + err) / s.double()
    expected = x.double() / s.double()
    torch.testing.assert_close(recomposed, expected, rtol=1e-5, atol=1e-5)
    feature = int(z[valid].sum(0).argmax())
    candidate = {"method": "ov", "features": [feature]}
    delta = transported_deltas(sae, x, valid, candidate, norm.variance_epsilon)["V"]
    independent = -(z[..., feature] * valid)[..., None] * sae.W_dec[feature] / s
    torch.testing.assert_close(delta, independent)
    native = norm(x)
    modeled = norm.weight * (x.float() / s).to(x.dtype)
    torch.testing.assert_close(native, modeled, rtol=0, atol=0)
    atomic_json(run / "plumbing.json", {"passed": True, "feature": feature,
        "residual_layer": sae.residual_layer, "attention_layer": sae.residual_layer + 1,
        "full_decomposition_max_error": float((recomposed - expected).abs().max()),
        "rms_scale_min": float(s.min()), "rms_scale_max": float(s.max()),
        "native_norm_gain_checked": True, "feature_scale_checked": True})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    args = p.parse_args()
    run = Path(args.run).resolve()
    from train import require_remote
    require_remote(run)
    spec = read(run / "task.json")
    started = time.time()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sae_lens import SAE
    import steering
    import scope
    from single_eval import run_single_evaluation
    from train import reconstruction_eval, ce_eval, prepare_data
    torch.set_num_threads(8)
    torch.manual_seed(42)
    scope.install_transport()
    protocol = {**steering.DEFAULT_PROTOCOL, "candidates_per_method": 50,
                "methods": [spec.get("method", "single")]}
    source_kind = spec.get("source_kind", "official")
    attention_layer = spec["attention_layer"]
    residual_layer = attention_layer - 1
    method = spec.get("method", "single")
    hook_kind = "resid_post" if source_kind == "official" and method == "single" else "input"
    layer = residual_layer if hook_kind == "resid_post" else attention_layer
    cfg = Config(layer=layer, hook_kind=hook_kind, d_sae=4096 * spec.get("expansion", 8))
    atomic_json(run / "config.json", asdict(cfg))
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, revision=MODEL_REVISIONS["A"], token=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    reference = Path(spec["reference"])
    splits, legacy, fresh = get_splits(tokenizer, cfg, protocol, reference)
    atomic_json(run / "split_manifest.json", {
        "splits": {k: [p["key"] for p in v] for k, v in splits.items()},
        "legacy_confirmation": [p["key"] for p in legacy],
        "fresh_confirmation": [p["key"] for p in fresh],
        "fresh_definition": "third 64-pair block after the unchanged 24 validation pairs",
        "frozen_before_model_loading": True})
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, revision=MODEL_REVISIONS["A"], token=False, torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"}, attn_implementation="sdpa", low_cpu_mem_usage=True).eval().requires_grad_(False)
    if source_kind == "official":
        sae = scope.load_scope(spec["expansion"], residual_layer, spec["revision"], run)
    elif source_kind == "local":
        source = Path(spec["sae_source"])
        actual = read(source / "config.json")
        assert actual == asdict(cfg)
        summary = read(source / "summary.json")
        assert summary["state"] == "complete" and summary["tokens_trained"] == 100_000_000
        assert summary["checkpoint_reload"]["passed"]
        sae = SAE.load_from_disk(str(source / "sae_final"), device="cuda").eval().requires_grad_(False)
        atomic_json(run / "sae_source.json", {"source": str(source), "config": actual,
            "weights_sha256": digest(source / "sae_final" / "sae_weights.safetensors")})
    elif source_kind == "dom":
        sae = None
    else:
        raise ValueError(source_kind)
    with torch.inference_mode():
        if sae is not None:
            plumbing(model, tokenizer, sae, cfg, splits["selection"], run)
        if spec.get("kind") == "diagnostic":
            dcfg = Config(**{**asdict(cfg), "layer": residual_layer, "hook_kind": "resid_post", "eval_per_class": 32})
            # Use training data for transfer diagnostics so the fresh steering test
            # remains uninspected by these exploratory quality measurements.
            pools, _, _ = prepare_data(dcfg)
            pools = {k: v[:32] for k, v in pools.items()}
            rec = reconstruction_eval(model, tokenizer, sae, 1.0, dcfg, pools)
            ce = ce_eval(model, tokenizer, sae, 1.0, dcfg, pools)
            atomic_json(run / "summary.json", {"state": "complete", "kind": "diagnostic",
                "reconstruction": rec, "teacher_forced_ce": ce, "seconds": time.time() - started})
            return
        if spec.get("kind") == "frozen":
            if source_kind == "dom":
                from dom_layers import build_candidate
                path = Path(spec["directions"])
                assert digest(path) == spec["directions_sha256"]
                directions = read(path)
                assert directions["selection_keys"] == [p["key"] for p in splits["selection"]]
                candidate = build_candidate(directions, spec["dom_variant"], [attention_layer])
                choices = {"restoration": {"candidate": candidate, "alpha": spec["dom_coefficient"]}}
            else:
                choices = spec["choices"]
            atomic_json(run / "selected.json", choices)
            frozen_hash = digest(run / "selected.json")
            result = evaluate_frozen(model, tokenizer, sae, cfg, fresh, protocol, run, choices, "confirmation")
            assert digest(run / "selected.json") == frozen_hash
            atomic_json(run / "summary.json", {"state": "complete", "kind": "frozen",
                "confirmation": result, "selected_sha256_before_test": frozen_hash,
                "seconds": time.time() - started, "layer": layer, "method": method})
            return
        if source_kind == "official" and method != "single":
            scope.ORIGINAL_RANK = steering.rank_candidates
            steering.rank_candidates = scope.rank_transport
        previous = {"path": str(reference), "protocol": read(reference / "protocol.json"),
                    "hashes": {"protocol": digest(reference / "protocol.json")}}
        authorization = {"request": "User requested overnight replication and official SAE transfer; empirical quality failures remain exploratory",
                         "official_transfer": source_kind == "official"}
        run_single_evaluation(model, tokenizer, sae, cfg, splits, protocol, run, previous,
                              authorization, started, confirmation=fresh, candidate_method=method)
        choices = read(run / "selected.json")
        frozen_hash = digest(run / "selected.json")
        legacy_results = evaluate_frozen(model, tokenizer, sae, cfg, legacy, protocol, run, choices, "legacy_confirmation")
        assert digest(run / "selected.json") == frozen_hash
        summary = read(run / "summary.json")
        summary.update(legacy_confirmation=legacy_results, residual_layer=residual_layer if source_kind == "official" else None,
                       attention_layer=attention_layer, source_kind=source_kind, seconds=time.time() - started,
                       normalization="observed token RMS scale held fixed for FRA" if source_kind == "official" and method != "single" else "native hook",
                       warning="Official SAE transfer crosses model/data distributions; 32-token greedy pilot; all selections frozen before test; confirmation is the new third test block")
        atomic_json(run / "summary.json", summary)
        design = read(run / "protocol.json")
        design["test_reuse_warning"] = "Original diagnostic test and legacy confirmation were inspected previously; confirmation is the new third block, unused for selection."
        design["residual_layer"] = summary["residual_layer"]
        design["attention_layer"] = attention_layer
        atomic_json(run / "protocol.json", design)


if __name__ == "__main__":
    main()
