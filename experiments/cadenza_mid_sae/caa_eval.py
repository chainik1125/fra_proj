"""Paired-prompt CAA/DoM, frozen validation selection followed by test evaluation.

No SAE is used to estimate or apply either intervention. This adapts CAA to paired
trigger-free/triggered prompts, not the original paper's contrastive answer pairs.
All probability traces remain in remote RAM; only small JSON summaries are saved.
"""
from contextlib import contextmanager
import hashlib
import json
import math
import time

from config import user_question
from remote import atomic_json


COEFFICIENTS = [-16., -8., -4., -2., -1., -.5, -.25, -.125,
                0., .125, .25, .5, 1., 2., 4., 8., 16.]
VARIANTS = ("input_prompt", "resid_response")


def residual_value(output):
    return output[0] if isinstance(output, tuple) else output


@contextmanager
def residual_intervention(model, layer, direction, alpha):
    """Patch the last prefill position, then every one-token decode position.

The prefill patch changes the first generated-token distribution; subsequent
patches change each later generated-token distribution. Earlier prompt positions
are unchanged. Preserve decoder-layer output structure and never mutate in-place.
"""
    import torch
    calls = 0

    def patch(module, inputs, output):
        nonlocal calls
        value = residual_value(output)
        vector = torch.as_tensor(direction, device=value.device, dtype=torch.float32)
        if vector.shape != (value.shape[-1],) or not bool(torch.isfinite(vector).all()):
            raise ValueError("Invalid residual CAA direction")
        if calls and value.shape[1] != 1:
            raise ValueError("Residual response steering requires one-token cached decoding")
        changed = value.clone()
        changed[:, -1] = changed[:, -1] + (alpha * vector).to(value.dtype)
        calls += 1
        return (changed, *output[1:]) if isinstance(output, tuple) else changed

    handle = model.model.layers[layer].register_forward_hook(patch)
    try:
        yield
    finally:
        handle.remove()


def capture_last(model, cfg, batch, variant):
    from train import capture_attention, StopAtAttention
    if variant == "input_prompt":
        return capture_attention(model, cfg.layer, batch, "input")[0, -1].float()
    if variant != "resid_response":
        raise ValueError("Unknown DoM variant")
    saved = []

    def capture(module, inputs, output):
        saved.append(residual_value(output)[0, -1].detach().float())
        raise StopAtAttention

    handle = model.model.layers[cfg.layer].register_forward_hook(capture)
    try:
        model.model(**batch, use_cache=False)
    except StopAtAttention:
        pass
    finally:
        handle.remove()
    if len(saved) != 1:
        raise ValueError("Did not capture exactly one residual vector")
    return saved[0]


def estimate_direction(model, tokenizer, cfg, pairs, variant, run):
    import torch
    from steering import input_batch
    keys = [p["key"] for p in pairs]
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("Direction requires nonempty unique training pairs")
    delta = torch.zeros(cfg.d_in, device="cuda", dtype=torch.float32)
    norms = {"clean": [], "sleeper": []}
    for i, pair in enumerate(pairs):
        for label, sign in (("clean", 1), ("sleeper", -1)):
            batch, _ = input_batch(tokenizer, [pair[label]])
            value = capture_last(model, cfg, batch, variant)
            delta += sign * value / len(pairs)
            norms[label].append(float(value.norm()))
        atomic_json(run / "progress.json", {"event": "dom_estimation", "variant": variant,
                    "completed": i + 1, "target": len(pairs)})
    norm = float(delta.norm())
    if not math.isfinite(norm) or norm < 1e-8:
        raise ValueError("Degenerate DoM direction")
    direction = (delta / norm).cpu().tolist()
    metadata = {"variant": variant, "layer": cfg.layer, "selection_pairs": len(pairs),
                "selection_keys": keys, "raw_dom_norm": norm, "unit_direction_norm": 1.,
                "mean_activation_norm": {k: sum(v) / len(v) for k, v in norms.items()},
                "definition": "mean(clean last-prompt-token activation) - mean(triggered last-prompt-token activation)",
                "hook": cfg.hook_name if variant == "input_prompt" else f"blocks.{cfg.layer}.hook_resid_post",
                "positions": "all valid prompt tokens only" if variant == "input_prompt" else "last prompt token and every generated token",
                "coefficient_units": "multiples of raw DoM; applied unit-direction alpha = coefficient * raw_dom_norm",
                "sae_dependency": False}
    atomic_json(run / f"direction_{variant}.json", {**metadata, "unit_direction": direction})
    candidate = {"method": "caa", "direction": direction}
    if variant == "resid_response":
        candidate["hook"] = "resid_post"
    return candidate, metadata


def choose_settings(grid):
    # Both rules fixed in advance, using ONLY validation; alpha-zero included.
    primary = min(grid, key=lambda r: (r["metrics"]["triggered_to_clean_js_bits"],
                                      abs(r["coefficient"]), r["coefficient"]))
    suppression = min(grid, key=lambda r: (r["metrics"]["sleeper_asr"],
                                          r["metrics"]["triggered_to_clean_js_bits"],
                                          abs(r["coefficient"]), r["coefficient"]))
    return {"restoration": primary, "suppression": suppression}


def evaluate_setting(model, tokenizer, cfg, batches, candidate, alpha, protocol):
    from steering import generate
    from restoration import paired_rollout_clean_js
    rows = []
    for pairs, reference in batches:
        prompts = [p[label] for p in pairs for label in ("clean", "sleeper")]
        changed = reference if alpha == 0 else generate(
            model, tokenizer, None, cfg, prompts, candidate, alpha, protocol,
            reference=reference, keep_probs=True)
        clean_js, counts = paired_rollout_clean_js(changed, reference)
        for i, pair in enumerate(pairs):
            c, s = 2 * i, 2 * i + 1
            rows.append({"key": pair["key"], "clean_prompt": user_question(pair["clean"]),
                "triggered_prompt": user_question(pair["sleeper"]),
                "triggered_to_clean_js_bits": clean_js[i], "compared_steps": counts[i],
                "triggered_to_poison_js_bits": changed["js"][s] / math.log(2),
                "clean_drift_js_bits": changed["js"][c] / math.log(2),
                "sleeper_asr": changed["asr"][s], "clean_asr": changed["asr"][c],
                "triggered_to_clean_exact_match": changed["texts"][s] == reference["texts"][c],
                "clean_exact_match": changed["texts"][c] == reference["texts"][c],
                "steered_triggered_text": changed["texts"][s], "steered_clean_text": changed["texts"][c],
                "unsteered_clean_text": reference["texts"][c], "unsteered_triggered_text": reference["texts"][s]})
        if alpha != 0:
            del changed
    if not rows:
        raise ValueError("No evaluation pairs")
    fields = ("triggered_to_clean_js_bits", "triggered_to_poison_js_bits", "clean_drift_js_bits",
              "sleeper_asr", "clean_asr", "triggered_to_clean_exact_match", "clean_exact_match")
    metrics = {"pairs": len(rows), **{f: sum(r[f] for r in rows) / len(rows) for f in fields},
               "escaped_phrase_count": sum(not r["sleeper_asr"] for r in rows),
               "mean_compared_steps": sum(r["compared_steps"] for r in rows) / len(rows)}
    if any(not math.isfinite(metrics[f]) or not 0 <= metrics[f] <= 1.00001 for f in fields):
        raise ValueError("Invalid metric in CAA evaluation")
    return metrics, rows


def run_caa_evaluation(model, tokenizer, cfg, splits, protocol, run, previous, authorization, started):
    from restoration import clean_reference_batches, paired_js_interval
    keys = {s: [p["key"] for p in rows] for s, rows in splits.items()}
    if keys != previous["protocol"]["splits"]:
        raise ValueError("CAA evaluation must preserve the comparison's frozen splits")
    if any(set(keys[a]) & set(keys[b]) for a, b in (("selection", "validation"), ("selection", "test"), ("validation", "test"))):
        raise ValueError("CAA split leakage")
    design = {**protocol, "methods": ["caa_dom"], "variants": VARIANTS, "coefficients": COEFFICIENTS,
        "splits": keys, "previous_run": previous["path"], "previous_file_sha256": previous["hashes"],
        "direction_source": "64 training prompt pairs; no completions, validation or test activations used",
        "selection": "primary minimum validation triggered-to-clean JSD; secondary minimum phrase ASR then JSD; all choices saved before test",
        "reference": "unsteered same sleeper model on identical question with only trigger removed",
        "measurement": "full-vocabulary per-step rollout JSD in bits, jointly alive steps including first EOS, then pair mean",
        "generation": "32-token greedy, original interleaved 8-sequence clean/triggered batches",
        "scope_caveat": "input_prompt matches SAE intervention scope; resid_response changes hook and decode scope, so is not an apples-to-apples pathway comparison",
        "caa_caveat": "Paired-prompt DoM adaptation, not original CAA contrastive-answer training; SAE weights unused; not norm-matched to SAE suppression",
        "authorization": authorization}
    atomic_json(run / "protocol.json", design)
    directions = {v: estimate_direction(model, tokenizer, cfg, splits["selection"], v, run) for v in VARIANTS}
    batches = clean_reference_batches(model, tokenizer, None, cfg, splits["validation"], protocol)
    val_baseline, val_rows = evaluate_setting(model, tokenizer, cfg, batches, None, 0, protocol)
    atomic_json(run / "validation_baseline.json", {"metrics": val_baseline, "rows": val_rows})
    selected = {}
    for variant, (candidate, metadata) in directions.items():
        grid = []
        for coefficient in COEFFICIENTS:
            alpha = coefficient * metadata["raw_dom_norm"]
            metrics, rows = (val_baseline, val_rows) if coefficient == 0 else evaluate_setting(
                model, tokenizer, cfg, batches, candidate, alpha, protocol)
            record = {"variant": variant, "coefficient": coefficient, "unit_alpha": alpha, "metrics": metrics}
            grid.append(record)
            atomic_json(run / f"validation_{variant}_{coefficient:g}.json", {**record, "rows": rows})
            atomic_json(run / "progress.json", {"event": "dom_validation", "variant": variant,
                        "completed": len(grid), "target": len(COEFFICIENTS), "last": record})
        selected[variant] = choose_settings(grid)
    atomic_json(run / "selected.json", selected)
    selected_hash = hashlib.sha256((run / "selected.json").read_bytes()).hexdigest()
    del batches, val_rows
    batches = clean_reference_batches(model, tokenizer, None, cfg, splits["test"], protocol)
    baseline, baseline_rows = evaluate_setting(model, tokenizer, cfg, batches, None, 0, protocol)
    atomic_json(run / "test_baseline.json", {"metrics": baseline, "rows": baseline_rows})
    results = {}
    for variant, choices in selected.items():
        results[variant], cache = {}, {}
        candidate, metadata = directions[variant]
        for rule, choice in choices.items():
            alpha = choice["unit_alpha"]
            if alpha not in cache:
                cache[alpha] = (baseline, baseline_rows) if alpha == 0 else evaluate_setting(
                    model, tokenizer, cfg, batches, candidate, alpha, protocol)
            metrics, rows = cache[alpha]
            result = {"coefficient": choice["coefficient"], "unit_alpha": alpha, "metrics": metrics,
                      **paired_js_interval(rows, baseline_rows, protocol["seed"])}
            results[variant][rule] = result
            atomic_json(run / f"test_{variant}_{rule}.json", {**result, "rows": rows})
            atomic_json(run / "progress.json", {"event": "dom_test", "variant": variant, "rule": rule})
    if hashlib.sha256((run / "selected.json").read_bytes()).hexdigest() != selected_hash:
        raise ValueError("Selected settings changed during test evaluation")
    summary = {"state": "complete", "task": "caa_dom_evaluation", "layer": cfg.layer,
        "previous_run": previous["path"], "baseline": baseline, "results": results,
        "directions": {v: meta for v, (_, meta) in directions.items()}, "coefficients": COEFFICIENTS,
        "selected_sha256_before_test": selected_hash, "test_pairs": len(splits["test"]),
        "seconds": time.time() - started, "authorization": authorization,
        "warning": design["caa_caveat"], "local_large_files_created": False}
    atomic_json(run / "summary.json", summary)
    print(json.dumps(summary), flush=True)
