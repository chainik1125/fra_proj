"""Training-fitted DoM at every layer: simultaneous, then independent sweeps.

One raw-DoM coefficient scales the separate vectors at all targeted layers.
Attention edits are prompt-only; full residual edits cover response prediction.
Only validation selects coefficients; every choice is frozen before confirmation.
"""
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import time

from config import Config, MODEL_REVISIONS
from remote import atomic_json


ALL_COEFFICIENTS = [-(2. ** p) for p in range(4, -7, -1)] + [0.] + [2. ** p for p in range(-6, 5)]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture_all(model, batch):
    """Capture last-token input and full residual vectors without modifying a forward."""
    from train import normalized_input
    from caa_eval import residual_value
    saved = {"input_prompt": {}, "resid_response": {}}
    handles = []
    for layer, block in enumerate(model.model.layers):
        def input_hook(module, inputs, output, layer=layer):
            saved["input_prompt"][layer] = normalized_input(module, inputs)[0, -1].detach().float().clone()
        def residual_hook(module, inputs, output, layer=layer):
            saved["resid_response"][layer] = residual_value(output)[0, -1].detach().float().clone()
        handles.extend([block.input_layernorm.register_forward_hook(input_hook),
                        block.register_forward_hook(residual_hook)])
    try:
        model.model(**batch, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    expected = set(range(len(model.model.layers)))
    if any(set(values) != expected for values in saved.values()):
        raise ValueError("Missing a layer's DoM capture")
    return saved


def fit_directions(model, tokenizer, cfg, pairs, run, frozen_sources):
    import torch
    from steering import input_batch
    variants = ("input_prompt", "resid_response")
    count = len(model.model.layers)
    delta = {v: torch.zeros(count, cfg.d_in, device="cuda", dtype=torch.float32) for v in variants}
    keys = [p["key"] for p in pairs]
    if len(keys) != 64 or len(set(keys)) != 64:
        raise ValueError("DoM requires the original 64 training pairs")
    for i, pair in enumerate(pairs):
        for label, sign in (("clean", 1), ("sleeper", -1)):
            batch, _ = input_batch(tokenizer, [pair[label]])
            values = capture_all(model, batch)
            for variant in variants:
                # Same accumulation order as the original single-layer DoM fit.
                for layer in range(count):
                    delta[variant][layer] += sign * values[variant][layer] / len(pairs)
        atomic_json(run / "progress.json", {"event": "fit_all_directions", "completed": i + 1, "target": len(pairs)})
    directions = {v: {} for v in variants}
    for variant in variants:
        for layer in range(count):
            raw = delta[variant][layer]
            norm = float(raw.norm())
            if not math.isfinite(norm):
                raise ValueError("Nonfinite DoM vector")
            # Layer-zero last-token input can be identical for clean/triggered prompts.
            degenerate = norm < 1e-8
            vector = torch.zeros_like(raw) if degenerate else raw / norm
            directions[variant][str(layer)] = {
                "layer": layer, "variant": variant, "unit_direction": vector.cpu().tolist(),
                "raw_dom_norm": 0. if degenerate else norm, "degenerate": degenerate,
                "selection_keys": keys, "hook": f"blocks.{layer}." +
                    ("ln1.hook_normalized" if variant == "input_prompt" else "hook_resid_post")}
    reproduction = []
    for source in frozen_sources:
        summary = json.loads((source / "summary.json").read_text())
        if summary.get("state") != "complete" or summary.get("task") != "caa_dom_evaluation":
            raise ValueError("Frozen source is not a completed DoM evaluation")
        layer = summary["layer"]
        for variant in variants:
            path = source / f"direction_{variant}.json"
            old = json.loads(path.read_text())
            new = directions[variant][str(layer)]
            if old["selection_keys"] != keys:
                raise ValueError("Frozen direction used another training split")
            a = torch.tensor(new["unit_direction"]) * new["raw_dom_norm"]
            b = torch.tensor(old["unit_direction"]) * old["raw_dom_norm"]
            if not torch.allclose(a, b, rtol=1e-5, atol=1e-6):
                raise ValueError(f"Direction reproduction failed: {variant} L{layer}")
            reproduction.append({"layer": layer, "variant": variant, "source": str(path),
                                 "sha256": digest(path), "max_abs_raw_error": float((a - b).abs().max())})
            # Preserve the exact already tested unit vector and norm.
            new.update(unit_direction=old["unit_direction"], raw_dom_norm=old["raw_dom_norm"])
    result = {"variants": directions, "layer_count": count, "selection_keys": keys,
              "definition": "Mean clean-minus-triggered last-prompt-token activation; one independent vector per layer",
              "reproduction": reproduction}
    atomic_json(run / "directions.json", result)
    return result


def build_candidate(directions, variant, layers):
    if not layers or len(set(layers)) != len(layers):
        raise ValueError("Expected distinct target layers")
    patches = []
    for layer in layers:
        d = directions["variants"][variant][str(layer)]
        if d["layer"] != layer or not math.isfinite(d["raw_dom_norm"]) or d["raw_dom_norm"] < 0:
            raise ValueError("Malformed DoM direction")
        patch = {"layer": layer, "direction": d["unit_direction"], "raw_dom_norm": d["raw_dom_norm"]}
        if variant == "resid_response":
            patch["hook"] = "resid_post"
        patches.append(patch)
    return {"method": "caa_multi", "interventions": patches}


def reproduce_baseline(rows, old_rows):
    if len(rows) != len(old_rows) or [r["key"] for r in rows] != [r["key"] for r in old_rows]:
        raise ValueError("Baseline identities do not match")
    for a, b in zip(rows, old_rows):
        for key in ("unsteered_clean_text", "unsteered_triggered_text", "clean_prompt", "triggered_prompt"):
            if a[key] != b[key]:
                raise ValueError(f"Baseline response mismatch: {key}")
        if not math.isclose(a["triggered_to_clean_js_bits"], b["triggered_to_clean_js_bits"], abs_tol=1e-6):
            raise ValueError("Baseline JSD mismatch")
    return {"pairs": len(rows), "all_prompts_and_reference_texts_exact": True, "max_jsd_tolerance": 1e-6}


def evaluate_targets(model, tokenizer, cfg, splits, confirmation, protocol, directions, groups, run, ref):
    from caa_eval import COEFFICIENTS, evaluate_setting, choose_settings
    from restoration import clean_reference_batches, paired_js_interval
    grid_values = ALL_COEFFICIENTS if groups == [list(range(32))] else COEFFICIENTS
    val_batches = clean_reference_batches(model, tokenizer, None, cfg, splits["validation"], protocol)
    val_metrics, val_rows = evaluate_setting(model, tokenizer, cfg, val_batches, None, 0, protocol)
    val_reproduction = reproduce_baseline(val_rows, json.loads((ref / "validation_baseline.json").read_text())["rows"])
    atomic_json(run / "validation_baseline.json", {"metrics": val_metrics, "rows": val_rows})
    selected, completed = {}, 0
    total = len(groups) * 2 * len(grid_values)
    for layers in groups:
        label = "all" if len(layers) == 32 else f"L{layers[0]:02d}"
        for variant in ("input_prompt", "resid_response"):
            tag = f"{label}_{variant}"
            candidate = build_candidate(directions, variant, layers)
            zero = all(p["raw_dom_norm"] == 0 for p in candidate["interventions"])
            grid = []
            for coefficient in grid_values:
                if coefficient == 0 or zero:
                    metrics, rows = val_metrics, val_rows
                else:
                    metrics, rows = evaluate_setting(model, tokenizer, cfg, val_batches, candidate, coefficient, protocol)
                record = {"variant": variant, "layers": layers, "coefficient": coefficient,
                          "metrics": metrics, "degenerate": zero}
                grid.append(record)
                atomic_json(run / f"validation_{tag}_{coefficient:g}.json", {**record, "rows": rows})
                completed += 1
                atomic_json(run / "progress.json", {"event": "dom_layer_validation", "tag": tag,
                            "completed": completed, "target": total, "coefficient": coefficient})
            selected[tag] = choose_settings(grid)
    atomic_json(run / "selected.json", selected)
    selected_hash = digest(run / "selected.json")
    del val_batches
    confirmation_batches = clean_reference_batches(model, tokenizer, None, cfg, confirmation, protocol)
    baseline, baseline_rows = evaluate_setting(model, tokenizer, cfg, confirmation_batches, None, 0, protocol)
    baseline_check = reproduce_baseline(baseline_rows, json.loads((ref / "confirmation_baseline.json").read_text())["rows"])
    atomic_json(run / "confirmation_baseline.json", {"metrics": baseline, "rows": baseline_rows})
    results = {}
    for tag, rules in selected.items():
        results[tag], cache = {}, {}
        for rule, choice in rules.items():
            coeff = choice["coefficient"]
            candidate = build_candidate(directions, choice["variant"], choice["layers"])
            if coeff not in cache:
                cache[coeff] = (baseline, baseline_rows) if coeff == 0 else evaluate_setting(
                    model, tokenizer, cfg, confirmation_batches, candidate, coeff, protocol)
            metrics, rows = cache[coeff]
            result = {"variant": choice["variant"], "layers": choice["layers"], "coefficient": coeff,
                      "validation_metrics": choice["metrics"], "metrics": metrics,
                      **paired_js_interval(rows, baseline_rows, protocol["seed"])}
            results[tag][rule] = result
            atomic_json(run / f"confirmation_{tag}_{rule}.json", {**result, "rows": rows})
            atomic_json(run / "progress.json", {"event": "dom_layer_confirmation", "tag": tag, "rule": rule})
    if digest(run / "selected.json") != selected_hash:
        raise ValueError("Choices changed during confirmation")
    return {"results": results, "baseline": baseline, "validation_baseline_reproduction": val_reproduction,
            "confirmation_baseline_reproduction": baseline_check,
            "selected_sha256_before_confirmation": selected_hash, "validation_settings": completed,
            "coefficients": grid_values}


def run_layer_experiment(run, spec):
    from train import require_remote, prepare_data
    from steering import DEFAULT_PROTOCOL, make_pairs, split_pairs
    from single_eval import confirmation_pairs
    from dom_confirmation import validate_confirmation
    require_remote(run)
    started = time.time()
    cfg = Config(**json.loads((run / "config.json").read_text())).validate()
    ref = Path(spec["confirmation_reference"]).resolve()
    if ref.parent != run.parent or ref == run:
        raise ValueError("Reference must be a different sibling run")
    ref_summary = json.loads((ref / "summary.json").read_text())
    ref_protocol = json.loads((ref / "protocol.json").read_text())
    ref_cfg = Config(**json.loads((ref / "config.json").read_text()))
    if ref_summary["state"] != "complete" or ref_cfg != cfg:
        raise ValueError("Reference is incomplete or model configuration changed")
    protocol = {k: ref_protocol[k] for k in DEFAULT_PROTOCOL}
    mode, layers = spec["dom_layer_mode"], spec["layers"]
    if mode not in ("all", "sweep") or not layers or len(set(layers)) != len(layers) or any(x not in range(32) for x in layers):
        raise ValueError("Invalid DoM experiment mode/layers")
    atomic_json(run / "protocol.json", {**protocol, "mode": mode, "layers": layers,
        "coefficients": ALL_COEFFICIENTS if mode == "all" else __import__("caa_eval").COEFFICIENTS,
        "splits": ref_protocol["splits"], "confirmation_keys": ref_protocol["confirmation_keys"],
        "confirmation_reference": str(ref), "model_revision": MODEL_REVISIONS[cfg.variant],
        "coefficient_units": "One shared coefficient times each targeted layer's own raw training DoM",
        "input_scope": "All valid prompt positions at pre-gain RMS-normalized attention input; no decode edits",
        "residual_scope": "Full post-block residual; last prompt token and every cached decode token",
        "selection": "Training-only vectors; validation-only coefficient selection; no confirmation retuning",
        "sae_dependency": False})
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(42)
    atomic_json(run / "progress.json", {"event": "loading_model"})
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, revision=MODEL_REVISIONS[cfg.variant], token=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, revision=MODEL_REVISIONS[cfg.variant],
        token=False, torch_dtype=torch.bfloat16, device_map={"": "cuda:0"}, attn_implementation="sdpa",
        low_cpu_mem_usage=True).eval().requires_grad_(False)
    if len(model.model.layers) != 32:
        raise ValueError("Expected the same 32-layer model")
    pools, heldout, _ = prepare_data(replace(cfg, eval_per_class=10000))
    train_pairs, heldout_pairs = (make_pairs(p, tokenizer, cfg.context_size) for p in (pools, heldout))
    splits = split_pairs(train_pairs, heldout_pairs, protocol)
    if {k: [p["key"] for p in v] for k, v in splits.items()} != ref_protocol["splits"]:
        raise ValueError("Original split identities changed")
    confirmation = confirmation_pairs(train_pairs, heldout_pairs, protocol, splits)
    validate_confirmation(confirmation, ref_protocol["confirmation_keys"], ref_protocol["splits"])
    with torch.inference_mode():
        if mode == "all":
            if layers != list(range(32)):
                raise ValueError("All-layer experiment requires all 32 layers")
            sources = [Path(p).resolve() for p in spec["frozen_runs"]]
            if any(p.parent != run.parent or p == run for p in sources):
                raise ValueError("Frozen sources must be sibling runs")
            directions = fit_directions(model, tokenizer, cfg, splits["selection"], run, sources)
            direction_path = run / "directions.json"
            groups = [layers]
        else:
            source = Path(spec["directions_from"]).resolve()
            if source.parent != run.parent or source == run:
                raise ValueError("Direction source must be a sibling run")
            source_summary = json.loads((source / "summary.json").read_text())
            if source_summary["state"] != "complete" or source_summary["mode"] != "all":
                raise ValueError("Run simultaneous all-layer DoM to completion before the individual sweep")
            direction_path = source / "directions.json"
            if digest(direction_path) != source_summary["directions_sha256"]:
                raise ValueError("Saved directions changed")
            directions = json.loads(direction_path.read_text())
            if directions["selection_keys"] != ref_protocol["splits"]["selection"]:
                raise ValueError("Direction training source mismatch")
            groups = [[layer] for layer in layers]
        direction_hash = digest(direction_path)
        result = evaluate_targets(model, tokenizer, cfg, splits, confirmation, protocol, directions, groups, run, ref)
    if digest(direction_path) != direction_hash:
        raise ValueError("Direction file changed during inference")
    summary = {"state": "complete", "task": "dom_layer_experiment", "mode": mode, "layers": layers,
               "directions_source": str(direction_path), "directions_sha256": direction_hash,
               "confirmation_keys": ref_protocol["confirmation_keys"], "seconds": time.time() - started, **result}
    atomic_json(run / "summary.json", summary)
    print(json.dumps(summary), flush=True)
