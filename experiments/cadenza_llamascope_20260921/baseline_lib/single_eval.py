"""Matched SAE and FRA feature steering at input or full residual hookpoints.

Positive alpha suppresses the active feature: x <- x - alpha*z_f(x)*W_dec[f].
Only valid prompt positions are changed; cached decode positions are untouched.
Original positive-grid and expanded signed-grid choices are frozen before test.
"""
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import time

from remote import atomic_json


def source_provenance(source, cfg):
    from config import MODEL_REVISIONS
    export = json.loads((source / "sae_final" / "cfg.json").read_text())
    metadata = export.get("metadata", {})
    if ((export.get("d_in"), export.get("d_sae"), export.get("k"), export.get("architecture"))
            != (cfg.d_in, cfg.d_sae, cfg.k, "topk")
            or metadata.get("hook_name") != cfg.hook_name
            or metadata.get("model_name") != cfg.model_name
            or metadata.get("model_revision") != MODEL_REVISIONS[cfg.variant]
            or metadata.get("training_tokens") != cfg.training_tokens):
        raise ValueError("Final SAE export metadata does not match the requested model/hook/budget")
    records = {}
    for relative in ("config.json", "summary.json", "sae_final/cfg.json", "sae_final/sae_weights.safetensors"):
        path = source / relative
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        records[relative] = {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}
    return {"source": str(source), "resolved_export": str((source / "sae_final").resolve()), "files": records}


@contextmanager
def residual_feature_intervention(model, sae, layer, hook_kind, valid, candidate, alpha):
    from steering import feature_deltas
    from train import register_stream_replacement
    if hook_kind not in ("resid_mid", "resid_post") or candidate.get("method") != "single":
        raise ValueError("Residual SAE steering supports single-feature interventions only")
    if len(candidate.get("features", [])) != 1 or not math.isfinite(alpha):
        raise ValueError("Expected one feature and a finite coefficient")
    calls = 0

    def transform(raw):
        nonlocal calls
        calls += 1
        if calls > 1:
            return raw
        if raw.shape[:-1] != valid.shape:
            raise ValueError("Prompt mask does not match the residual activation shape")
        delta = feature_deltas(sae, raw, valid, candidate)["input"]
        return raw + (alpha * delta).to(raw.dtype)

    handle = register_stream_replacement(model, layer, hook_kind, transform)
    try:
        yield
    finally:
        handle.remove()


def load_split_reference(run, cfg, spec):
    from steering import DEFAULT_PROTOCOL
    previous = Path(spec["split_reference"]).resolve()
    if previous.parent != run.parent or previous == run:
        raise ValueError("Split reference must be a distinct sibling comparison run")
    data, hashes = {}, {}
    names = ("config", "protocol", "summary", "status", "task")
    if cfg.hook_kind == "input":
        names += ("selection",)
    for name in names:
        content = (previous / f"{name}.json").read_bytes()
        data[name] = json.loads(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    if (data["status"].get("state") != "complete" or data["summary"].get("state") != "complete"
            or data["summary"].get("smoke") or data["summary"].get("task") != "steering"
            or data["config"] != {**cfg.__dict__, "hook_kind": "input"}
            or data["summary"].get("hook") != f"blocks.{cfg.layer}.ln1.hook_normalized"):
        raise ValueError("Original input comparison is incomplete or differs beyond hook_kind")
    if any(data["protocol"].get(k) != v for k, v in DEFAULT_PROTOCOL.items()):
        raise ValueError("Expected the unchanged original 64/24/64, 32-token comparison protocol")
    data.update(path=str(previous), hashes=hashes)
    return data


def candidates_from_difference(difference, decoder, count):
    import torch
    score = difference.abs() * decoder.float().norm(dim=-1)
    if not bool(torch.isfinite(score).all()) or not 1 <= count <= len(score):
        raise ValueError("Invalid single-feature ranking scores/count")
    return [{"method": "single", "features": [f], "selection_score": float(score[f])}
            for f in score.topk(count).indices.tolist()]


def expanded_single_protocol(previous, spec, cfg):
    """Change candidate budget only; retain every original generation/grid setting."""
    from steering import DEFAULT_PROTOCOL
    protocol = {key: previous["protocol"][key] for key in DEFAULT_PROTOCOL}
    count = spec.get("single_candidates", protocol["candidates_per_method"])
    if type(count) is not int or not 1 <= count <= cfg.d_sae:
        raise ValueError("Invalid single-feature candidate count")
    method = spec.get("feature_method", "single")
    if method not in ("single", "ov", "qkov") or (method != "single" and cfg.hook_kind != "input"):
        raise ValueError("FRA methods require an attention-input SAE")
    if method == "qkov" and count > min(protocol["candidate_pool"], cfg.d_sae) ** 3:
        raise ValueError("Candidate count exceeds the unchanged QK+OV triplet pool")
    protocol["candidates_per_method"] = count
    if "feature_method" in spec:
        protocol["methods"] = [method]
    return protocol


def confirmation_pairs(train_pairs, heldout_pairs, protocol, original_splits):
    """Next deterministic held-out block, excluding old validation/test identities."""
    from steering import split_pairs
    size = protocol["test_pairs"]
    expanded = split_pairs(train_pairs, heldout_pairs, {**protocol, "test_pairs": size * 2})
    for split in ("selection", "validation"):
        if expanded[split] != original_splits[split]:
            raise ValueError("Confirmation expansion changed original selection/validation")
    if expanded["test"][:size] != original_splits["test"]:
        raise ValueError("Confirmation expansion changed original test identities")
    result = expanded["test"][size:]
    used = {p["key"] for rows in original_splits.values() for p in rows}
    if len(result) != size or len({p["key"] for p in result}) != size or any(p["key"] in used for p in result):
        raise ValueError("Confirmation identities overlap previously used prompts")
    return result


def rank_single_candidates(model, tokenizer, sae, cfg, pairs, protocol, run):
    import torch
    from steering import encode, input_batch
    from train import capture_attention
    difference = torch.zeros(cfg.d_sae, device=sae.W_dec.device, dtype=torch.float32)
    for i, pair in enumerate(pairs):
        for label, sign in (("sleeper", 1), ("clean", -1)):
            batch, valid = input_batch(tokenizer, [pair[label]])
            raw = capture_attention(model, cfg.layer, batch, cfg.hook_kind)
            z = encode(sae, raw)
            z[~valid[0]] = 0
            difference += (sign / len(pairs)) * z.sum(0) / valid.sum().clamp_min(1)
        atomic_json(run / "progress.json", {"event": "single_feature_selection", "completed": i + 1,
                                             "target": len(pairs)})
    candidates = candidates_from_difference(difference, sae.W_dec, protocol["candidates_per_method"])
    atomic_json(run / "selection.json", {"candidates": candidates, "features_ranked": cfg.d_sae,
        "candidate_count": len(candidates), "selection_pairs": len(pairs),
        "selection_keys": [p["key"] for p in pairs],
        "score": "abs(mean_paired(per-prompt mean z_sleeper - per-prompt mean z_clean)) * decoder L2 norm",
        "hook": cfg.hook_name, "scope": "all valid prompt tokens; no validation/test activations"})
    return candidates


def signed_grid(positive):
    if (not positive or 0 not in positive or len(set(positive)) != len(positive)
            or any(not math.isfinite(a) or not 0 <= a <= 32 for a in positive)):
        raise ValueError("Expected a bounded, nonnegative grid containing zero")
    return sorted(set(positive) | {-a for a in positive})


def choose_single_settings(grid):
    from restoration import choose_restoration
    return {"positive_jsd": choose_restoration([r for r in grid if r["alpha"] >= 0]),
            "signed_jsd": min(grid, key=lambda r: (r["metrics"]["triggered_to_clean_js_bits"],
                              abs(r["alpha"]), r["alpha"], tuple(r["candidate"]["features"])))}


def candidate_identity(candidate):
    """Full channel tuple: different K/V choices must never share artifacts/cache."""
    method, features = candidate.get("method"), candidate.get("features", [])
    arity = 3 if method == "qkov" else 1
    if method not in ("single", "ov", "qkov") or len(features) != arity or any(
            type(f) is not int or f < 0 for f in features):
        raise ValueError("Invalid method/feature tuple")
    return method, tuple(features)


def verify_original_ranking(candidates, previous, method):
    original = [c for c in previous["selection"]["candidates"] if c["method"] == method]
    count = min(len(original), len(candidates))
    if not count:
        raise ValueError("Missing original method candidates")
    for old, new in zip(original[:count], candidates[:count]):
        if candidate_identity(old) != candidate_identity(new) or not math.isclose(
                old["selection_score"], new["selection_score"], rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError("Expanded ranking does not reproduce the original shortlist")
    return {"method": method, "original_candidates_reproduced": count,
            "original_selection_sha256": previous["hashes"]["selection"]}


def verify_reference_outputs(rows, previous):
    """Report BF16 cross-host reproducibility explicitly; never silently substitute references."""
    path = Path(previous["path"]) / "test_baseline.json"
    content = path.read_bytes()
    old = json.loads(content)["rows"]
    if [r["key"] for r in old] != [r["key"] for r in rows]:
        raise ValueError("Test prompt order differs from the original input comparison")
    return {"pairs": len(rows), "original_baseline_sha256": hashlib.sha256(content).hexdigest(),
        "clean_outputs_reproduced": sum(a["unsteered_clean_text"] == b["clean_text"] for a, b in zip(rows, old)),
        "triggered_outputs_reproduced": sum(a["unsteered_triggered_text"] == b["sleeper_text"] for a, b in zip(rows, old))}


def run_single_evaluation(model, tokenizer, sae, cfg, splits, protocol, run, previous, authorization, started,
                          confirmation=None, candidate_method="single"):
    from caa_eval import evaluate_setting
    from restoration import clean_reference_batches, paired_js_interval
    if candidate_method not in ("single", "ov", "qkov") or (candidate_method != "single" and cfg.hook_kind != "input"):
        raise ValueError("Unsupported feature method/hook")
    keys = {s: [p["key"] for p in rows] for s, rows in splits.items()}
    if keys != previous["protocol"]["splits"]:
        raise ValueError("Single-feature evaluation must preserve the frozen original splits")
    if any(set(keys[a]) & set(keys[b]) for a, b in (("selection", "validation"), ("selection", "test"),
                                                 ("validation", "test"))):
        raise ValueError("Single-feature split leakage")
    confirmation_keys = [p["key"] for p in confirmation] if confirmation is not None else []
    if confirmation is not None and (len(confirmation_keys) != len(keys["test"])
            or len(set(confirmation_keys)) != len(confirmation_keys)
            or set(confirmation_keys) & {key for values in keys.values() for key in values}):
        raise ValueError("Fresh confirmation must be equal-sized, unique, and disjoint from original splits")
    alphas = signed_grid(protocol["alphas"])
    design = {**protocol, "methods": [candidate_method], "alphas": alphas,
        "original_positive_alphas": protocol["alphas"], "splits": keys, "hook": cfg.hook_name,
        "previous_run": previous["path"], "previous_file_sha256": previous["hashes"],
        "selection": f"{protocol.get('candidates_per_method', 3)} train-ranked candidates; positive-grid and signed-grid minimum validation restoration JSD; freeze both before test",
        "confirmation_keys": confirmation_keys,
        "test_reuse_warning": "Original test results have been inspected; reuse is diagnostic. Confirmation uses a disjoint deterministic next block and never selects parameters; the same block is shared with the residual top-50 comparison, so it is not untouched across the campaign.",
        "patching": ("x <- x - alpha*z_f(x)*W_dec[f]; valid prompt positions only; no decode edits; full residual stream at residual hooks"
                     if candidate_method == "single" else "Original FRA channel routing on valid prompt positions only; gain restored before QKV; no decode edits; OV changes V only; QK+OV changes Q/K/V with the original K/V co-fire gates"),
        "arity": "single/OV use one feature; QK+OV uses a Q/K/V tuple, possibly three distinct features; candidate budget is matched, not feature count or perturbation norm",
        "sign": "positive alpha suppresses/over-suppresses the active feature; negative amplifies it; not constant-vector CAA",
        "reference": "unsteered same sleeper model, identical question with only literal trigger removed",
        "measurement": "full-vocabulary rollout JSD in bits, matching jointly alive steps including first EOS, then prompt-pair mean",
        "generation": "32-token greedy; original interleaved eight-sequence clean/triggered batch shapes",
        "paper_caveat": "Corrected cross-prompt metric, not the paper's full sampled five-seed generation protocol",
        "authorization": authorization}
    atomic_json(run / "protocol.json", design)
    if candidate_method == "single":
        candidates = rank_single_candidates(model, tokenizer, sae, cfg, splits["selection"], protocol, run)
    else:
        from steering import rank_candidates
        candidates = rank_candidates(model, tokenizer, sae, cfg, splits["selection"],
                                     {**protocol, "methods": [candidate_method]}, run)
    identities = [candidate_identity(c) for c in candidates]
    if (len(candidates) != protocol.get("candidates_per_method", len(candidates))
            or len(set(identities)) != len(candidates) or any(m != candidate_method for m, _ in identities)):
        raise ValueError("Candidates must be distinct and all belong to the requested method")
    ranking_reproduction = None
    if "selection" in previous:
        ranking_reproduction = verify_original_ranking(candidates, previous, candidate_method)
        atomic_json(run / "ranking_reproduction.json", ranking_reproduction)
    batches = clean_reference_batches(model, tokenizer, sae, cfg, splits["validation"], protocol)
    baseline, baseline_rows = evaluate_setting(model, tokenizer, cfg, batches, None, 0, protocol, sae=sae)
    atomic_json(run / "validation_baseline.json", {"metrics": baseline, "rows": baseline_rows})
    grid = []
    with (run / "validation.jsonl").open("x", buffering=1) as log:
        for candidate in candidates:
            for alpha in alphas:
                metrics, rows = (baseline, baseline_rows) if alpha == 0 else evaluate_setting(
                    model, tokenizer, cfg, batches, candidate, alpha, protocol, sae=sae)
                record = {"candidate": candidate, "alpha": alpha, "metrics": metrics}
                grid.append(record)
                log.write(json.dumps(record, allow_nan=False) + "\n")
                feature = "_".join(map(str, candidate["features"]))
                atomic_json(run / f"validation_f{feature}_a{alpha:g}.json", {**record, "rows": rows})
                atomic_json(run / "progress.json", {"event": "single_validation", "completed": len(grid),
                    "target": len(candidates) * len(alphas), "last": record})
    selected = choose_single_settings(grid)
    atomic_json(run / "selected.json", selected)
    digest = hashlib.sha256((run / "selected.json").read_bytes()).hexdigest()
    del batches, grid, baseline_rows, rows
    batches = clean_reference_batches(model, tokenizer, sae, cfg, splits["test"], protocol)
    baseline, baseline_rows = evaluate_setting(model, tokenizer, cfg, batches, None, 0, protocol, sae=sae)
    atomic_json(run / "test_baseline.json", {"metrics": baseline, "rows": baseline_rows})
    reproduction = verify_reference_outputs(baseline_rows, previous)
    results, cache = {}, {}
    for rule, choice in selected.items():
        candidate, alpha = choice["candidate"], choice["alpha"]
        key = (*candidate_identity(candidate), alpha)
        if key not in cache:
            cache[key] = (baseline, baseline_rows) if alpha == 0 else evaluate_setting(
                model, tokenizer, cfg, batches, candidate, alpha, protocol, sae=sae)
        metrics, rows = cache[key]
        results[rule] = {"candidate": candidate, "alpha": alpha, "metrics": metrics,
                        **paired_js_interval(rows, baseline_rows, protocol["seed"])}
        atomic_json(run / f"test_{rule}.json", {**results[rule], "rows": rows})
        atomic_json(run / "progress.json", {"event": "single_test", "rule": rule})
    if hashlib.sha256((run / "selected.json").read_bytes()).hexdigest() != digest:
        raise ValueError("Validation choices changed during test evaluation")
    confirmation_result = None
    if confirmation is not None:
        del batches, cache
        batches = clean_reference_batches(model, tokenizer, sae, cfg, confirmation, protocol)
        fresh_baseline, fresh_rows = evaluate_setting(model, tokenizer, cfg, batches, None, 0, protocol, sae=sae)
        atomic_json(run / "confirmation_baseline.json", {"metrics": fresh_baseline, "rows": fresh_rows})
        fresh_results, cache = {}, {}
        for rule, choice in selected.items():
            candidate, alpha = choice["candidate"], choice["alpha"]
            key = (*candidate_identity(candidate), alpha)
            if key not in cache:
                cache[key] = (fresh_baseline, fresh_rows) if alpha == 0 else evaluate_setting(
                    model, tokenizer, cfg, batches, candidate, alpha, protocol, sae=sae)
            metrics, rows = cache[key]
            fresh_results[rule] = {"candidate": candidate, "alpha": alpha, "metrics": metrics,
                                   **paired_js_interval(rows, fresh_rows, protocol["seed"])}
            atomic_json(run / f"confirmation_{rule}.json", {**fresh_results[rule], "rows": rows})
            atomic_json(run / "progress.json", {"event": "single_confirmation", "rule": rule})
        confirmation_result = {"pairs": len(confirmation), "keys": confirmation_keys,
                               "baseline": fresh_baseline, "results": fresh_results}
        if hashlib.sha256((run / "selected.json").read_bytes()).hexdigest() != digest:
            raise ValueError("Validation choices changed during fresh confirmation")
    summary = {"state": "complete", "task": "single_feature_evaluation" if candidate_method == "single" else "fra_feature_evaluation", "layer": cfg.layer,
        "method": candidate_method, "ranking_reproduction": ranking_reproduction,
        "hook": cfg.hook_name, "smoke": False, "authorization": authorization,
        "baseline": baseline, "results": results, "baseline_reproduction": reproduction,
        "selected_sha256_before_test": digest, "test_pairs": len(splits["test"]),
        "candidate_count": len(candidates), "grid": alphas, "seconds": time.time() - started,
        "confirmation": confirmation_result,
        "warning": "Exploratory steering despite failed SAE quality gates; prompt-only 32-token greedy pilot; raw alpha is not norm-matched across hooks",
        "local_large_files_created": False}
    atomic_json(run / "summary.json", summary)
    print(json.dumps(summary), flush=True)
