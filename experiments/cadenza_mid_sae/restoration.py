"""Corrected triggered-to-trigger-stripped rollout JSD. Remote inference only.

Freeze original train-ranked candidates and question splits. Re-measure the old
operating points and separately select by corrected JSD on validation only.
This uses the paper's cross-prompt reference, in bits, but retains this pilot's
32-token greedy decoding (not the paper's 16-token, five-seed sampled protocol).
"""
import hashlib
import json
import math
from pathlib import Path
import random
import time

from config import TRIGGER
from remote import atomic_json
from steering import generate, js_divergence


def load_previous(run, cfg, spec):
    previous = Path(spec["restoration_from"]).resolve()
    if previous.parent != run.parent or previous == run:
        raise ValueError("Previous comparison must be a different sibling run")
    names = ("config", "task", "summary", "protocol", "selection", "selected", "status")
    data, hashes = {}, {}
    for name in names:
        content = (previous / f"{name}.json").read_bytes()
        data[name] = json.loads(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    if (data["status"]["state"] != "complete" or data["summary"]["state"] != "complete"
            or data["summary"].get("smoke") or data["config"] != cfg.__dict__
            or data["task"]["sae_run"] != spec["sae_run"]
            or data["summary"]["layer"] != cfg.layer
            or data["summary"]["hook"] != cfg.hook_name):
        raise ValueError("Previous comparison/source/config is incomplete or mismatched")
    data.update(path=str(previous), hashes=hashes)
    return data


def clean_reference_batches(model, tokenizer, sae, cfg, pairs, protocol):
    batches = []
    for start in range(0, len(pairs), protocol["batch_size"]):
        group = pairs[start:start + protocol["batch_size"]]
        for pair in group:
            if (TRIGGER not in pair["sleeper"] or TRIGGER in pair["clean"]
                    or pair["sleeper"].replace(TRIGGER, "") != pair["clean"]):
                raise ValueError("Clean reference must differ ONLY by trigger removal")
        # Preserve the original interleaved 8-sequence batch, including padding
        # shape. BF16 kernel shape changes can flip greedy decisions downstream.
        prompts = [p[label] for p in group for label in ("clean", "sleeper")]
        reference = generate(model, tokenizer, sae, cfg, prompts,
                             None, 0, protocol, keep_probs=True)
        batches.append((group, reference))
    return batches


def evaluate_restoration(model, tokenizer, sae, cfg, batches, candidate, alpha, protocol):
    rows = []
    for pairs, reference in batches:
        # Crucially: even alpha=0 compares TRIGGERED to CLEAN, not self to self.
        prompts = [p[label] for p in pairs for label in ("clean", "sleeper")]
        clean_indices = [2 * i for i in range(len(pairs)) for _ in range(2)]
        changed = generate(model, tokenizer, sae, cfg, prompts,
                           candidate, alpha, protocol, reference=reference,
                           reference_indices=clean_indices)
        for i, pair in enumerate(pairs):
            clean, sleeper = 2 * i, 2 * i + 1
            if changed["js_steps"][sleeper] <= 0:
                raise ValueError("No aligned pre-EOS generation steps")
            rows.append({"key": pair["key"],
                         "triggered_to_clean_js_bits": changed["js"][sleeper] / math.log(2),
                         "compared_steps": changed["js_steps"][sleeper],
                         "sleeper_asr": changed["asr"][sleeper],
                         "triggered_to_clean_exact_match": changed["texts"][sleeper] == reference["texts"][clean],
                         "steered_triggered_text": changed["texts"][sleeper],
                         "unsteered_clean_text": reference["texts"][clean]})
    if not rows:
        raise ValueError("Empty restoration evaluation")
    n = len(rows)
    return {"pairs": n,
            "triggered_to_clean_js_bits": sum(r["triggered_to_clean_js_bits"] for r in rows) / n,
            "sleeper_asr": sum(r["sleeper_asr"] for r in rows) / n,
            "triggered_to_clean_exact_match": sum(r["triggered_to_clean_exact_match"] for r in rows) / n,
            "mean_compared_steps": sum(r["compared_steps"] for r in rows) / n}, rows


def choose_restoration(grid):
    # No held-out test result participates in this choice.
    return min(grid, key=lambda r: (r["metrics"]["triggered_to_clean_js_bits"], r["alpha"]))


def paired_js_interval(rows, baseline_rows, seed):
    if [r["key"] for r in rows] != [r["key"] for r in baseline_rows]:
        raise ValueError("Bootstrap requires identical ordered prompt pairs")
    values = [r["triggered_to_clean_js_bits"] for r in rows]
    differences = [a - b["triggered_to_clean_js_bits"] for a, b in zip(values, baseline_rows)]
    rng = random.Random(seed)
    draws, deltas = [], []
    for _ in range(2000):
        indices = rng.choices(range(len(rows)), k=len(rows))
        draws.append(sum(values[i] for i in indices) / len(indices))
        deltas.append(sum(differences[i] for i in indices) / len(indices))
    draws.sort()
    deltas.sort()
    return {"js_bits_bootstrap_95pct": [draws[50], draws[1949]],
            "delta_js_bits_vs_unsteered": sum(differences) / len(differences),
            "paired_delta_bootstrap_95pct": [deltas[50], deltas[1949]]}


def validate_probe(probe):
    feature, alphas = probe.get("feature"), probe.get("alphas")
    if (not isinstance(feature, int) or feature < 0 or not isinstance(alphas, list)
            or not 1 <= len(alphas) <= 33 or any(not isinstance(a, (int, float))
            or not math.isfinite(a) or not -64 <= a <= 64 for a in alphas)
            or len(set(alphas)) != len(alphas) or probe.get("kind", "ov") not in ("ov", "caa")):
        raise ValueError("Invalid bounded OV probe feature/strength grid")
    result = {"feature": feature, "alphas": sorted(alphas)}
    if "kind" in probe:
        result["kind"] = probe["kind"]
    return result


def estimate_caa(model, tokenizer, cfg, pairs, run):
    import torch
    from steering import input_batch
    from train import capture_attention
    delta = torch.zeros(cfg.d_in, device="cuda", dtype=torch.float32)
    for index, pair in enumerate(pairs):
        for label, sign in (("clean", 1), ("sleeper", -1)):
            batch, _ = input_batch(tokenizer, [pair[label]])
            raw = capture_attention(model, cfg.layer, batch, "input")
            if raw.ndim != 3 or raw.shape[0] != 1:
                raise ValueError("Expected one full prompt activation batch")
            delta += sign * raw[0, -1].float() / len(pairs)
        atomic_json(run / "progress.json", {"event": "caa_vector_selection", "completed": index + 1, "target": len(pairs)})
    norm = delta.norm()
    if not bool(torch.isfinite(norm)) or float(norm) < 1e-8:
        raise ValueError("CAA mean difference is degenerate")
    direction = (delta / norm).cpu().tolist()
    metadata = {"method": "caa", "hook": cfg.hook_name, "raw_mean_difference_norm": float(norm),
                "direction_norm": 1.0, "selection_pairs": len(pairs),
                "direction_definition": "unit(mean clean last-prompt-token attention input - mean triggered last-prompt-token attention input), before RMSNorm gain",
                "application": "add alpha times fixed unit direction to every valid prompt token's attention input; restore gain; no decode-token edits",
                "scale_warning": "Coefficient is input-space L2 units, not comparable to activation-scaled SAE suppression alpha"}
    atomic_json(run / "caa_direction.json", {**metadata, "direction": direction,
                "selection_keys": [p["key"] for p in pairs]})
    return {"method": "caa", "direction": direction}, metadata


def paired_rollout_clean_js(changed, baseline):
    """CPU reduction: changed triggered row 2i+1 versus baseline clean row 2i."""
    import torch
    if len(changed["probs"]) != len(baseline["probs"]) or not changed["probs"]:
        raise ValueError("Mismatched or empty rollout probability traces")
    batch = changed["probs"][0].shape[0]
    if batch % 2:
        raise ValueError("Expected interleaved clean/triggered twins")
    total, counts = torch.zeros(batch // 2), torch.zeros(batch // 2)
    for step, logp in enumerate(changed["probs"]):
        valid = changed["alive"][step][1::2] & baseline["alive"][step][::2]
        total += js_divergence(logp[1::2], baseline["probs"][step][::2]) * valid
        counts += valid
    if bool((counts == 0).any()):
        raise ValueError("No jointly alive generation steps")
    return (total / counts / math.log(2)).tolist(), counts.tolist()


def run_ov_probe(model, tokenizer, sae, cfg, splits, protocol, run, previous, authorization, started, probe):
    from config import user_question
    probe = validate_probe(probe)
    keys = {s: [p["key"] for p in rows] for s, rows in splits.items()}
    if keys != previous["protocol"]["splits"]:
        raise ValueError("OV probe must preserve original question splits")
    is_caa = probe.get("kind") == "caa"
    if is_caa:
        candidate, candidate_metadata = estimate_caa(model, tokenizer, cfg, splits["selection"], run)
    else:
        matches = [c for c in previous["selection"]["candidates"]
                   if c["method"] == "ov" and c["features"] == [probe["feature"]]]
        if len(matches) != 1:
            raise ValueError("Probe feature must be an existing frozen train-ranked OV candidate")
        candidate = candidate_metadata = matches[0]
    atomic_json(run / "protocol.json", {**protocol, "alphas": probe["alphas"], "candidate": candidate_metadata,
        "splits": keys, "evaluation_split": "validation only; no test tuning or new test outcomes",
        "previous_run": previous["path"], "previous_file_sha256": previous["hashes"],
        "measurement": "Full-vocabulary JSD to unsteered triggered AND trigger-stripped clean free-generation references; bits",
        "batching": "Original interleaved clean/triggered twins; same padding, four pairs per batch",
        "generation": "32-token greedy; prompt-only interventions; compare jointly alive steps",
        "authorization": authorization})
    pairs = splits["validation"]
    batches = clean_reference_batches(model, tokenizer, sae, cfg, pairs, protocol)
    results = []
    original_grid = [json.loads(line) for line in (Path(previous["path"]) / "validation.jsonl").read_text().splitlines()]
    for alpha in probe["alphas"]:
        rows = []
        for group, reference in batches:
            prompts = [p[label] for p in group for label in ("clean", "sleeper")]
            changed = generate(model, tokenizer, sae, cfg, prompts, candidate, alpha, protocol,
                               reference=reference, keep_probs=True)
            clean_js, counts = paired_rollout_clean_js(changed, reference)
            for i, pair in enumerate(group):
                c, s = 2 * i, 2 * i + 1
                rows.append({"key": pair["key"], "clean_prompt": user_question(pair["clean"]),
                    "triggered_prompt": user_question(pair["sleeper"]),
                    "steered_triggered_text": changed["texts"][s],
                    "unsteered_clean_text": reference["texts"][c],
                    "unsteered_triggered_text": reference["texts"][s],
                    "js_to_clean_bits": clean_js[i], "js_to_poison_bits": changed["js"][s] / math.log(2),
                    "compared_clean_steps": counts[i], "compared_poison_steps": changed["js_steps"][s],
                    "sleeper_asr": changed["asr"][s]})
            del changed
        n = len(rows)
        metrics = {"pairs": n, "js_to_clean_bits": sum(r["js_to_clean_bits"] for r in rows) / n,
                   "js_to_poison_bits": sum(r["js_to_poison_bits"] for r in rows) / n,
                   "sleeper_asr": sum(r["sleeper_asr"] for r in rows) / n,
                   "escaped_phrase_count": sum(not r["sleeper_asr"] for r in rows)}
        original = [r for r in original_grid if r["candidate"] == candidate and r["alpha"] == alpha]
        reproduced = None
        if original:
            target = original[0]["metrics"]
            reproduced = (metrics["sleeper_asr"] == target["sleeper_asr"] and
                          math.isclose(metrics["js_to_poison_bits"], target["sleeper_js_nats"] / math.log(2), abs_tol=1e-6))
            if not reproduced:
                raise ValueError("Repeated grid point did not reproduce saved ASR/poison JSD")
        record = {"alpha": alpha, "metrics": metrics, "original_point_reproduced": reproduced}
        results.append(record)
        atomic_json(run / f"alpha_{alpha:g}.json", {**record, "rows": rows})
        atomic_json(run / "progress.json", {"event": "caa_fine_grid" if is_caa else "ov_fine_grid", "completed": len(results),
                    "target": len(probe["alphas"]), "last": record})
    summary = {"state": "complete", "task": "caa_fine_grid" if is_caa else "ov_fine_grid", "layer": cfg.layer,
               "candidate": candidate_metadata, "alphas": probe["alphas"], "results": results,
               "pairs": len(pairs), "split": "validation", "seconds": time.time() - started,
               "original_endpoint_checks_passed": all(r["original_point_reproduced"] is not False for r in results),
               "original_points_checked": sum(r["original_point_reproduced"] is not None for r in results),
               "authorization": authorization, "local_large_files_created": False}
    atomic_json(run / "summary.json", summary)
    print(json.dumps(summary), flush=True)


def run_evaluation(model, tokenizer, sae, cfg, splits, protocol, run, previous, authorization, started):
    keys = {s: [p["key"] for p in rows] for s, rows in splits.items()}
    if keys != previous["protocol"]["splits"]:
        raise ValueError("Prompt splits differ from the completed comparison")
    candidates = previous["selection"]["candidates"]
    atomic_json(run / "protocol.json", {
        **protocol, "splits": keys, "previous_run": previous["path"],
        "previous_file_sha256": previous["hashes"], "candidates": candidates,
        "reference": "unsteered same sleeper model, identical prompt with literal trigger removed",
        "measurement": "JSD(steered triggered, unsteered clean), bits, per generation step over full vocabulary; mean over jointly alive steps then prompt pairs",
        "generation": "unchanged greedy 32-token pilot; separate free-generation histories, not teacher-forced; include first EOS prediction, exclude subsequent padding",
        "paper_difference": "Paper code uses 16-token sampled rollouts over five decoding seeds; this correction keeps the existing 32-token greedy pilot fixed",
        "batching": "Original interleaved clean/sleeper twins, same batch and padding shapes; reference row 2i for both generated rows 2i and 2i+1",
        "selection": "Frozen train-ranked candidates; minimum corrected mean JSD on validation, alpha tie-break; no ASR-based selection and no test tuning",
        "patching": previous["protocol"]["patching"], "arity": previous["protocol"]["arity"],
        "authorization": authorization,
    })
    atomic_json(run / "selection.json", previous["selection"])
    batches = clean_reference_batches(model, tokenizer, sae, cfg, splits["validation"], protocol)
    val_baseline, _ = evaluate_restoration(model, tokenizer, sae, cfg, batches, None, 0, protocol)
    grid = []
    with (run / "validation.jsonl").open("x", buffering=1) as log:
        for candidate in candidates:
            for alpha in protocol["alphas"]:
                metrics = val_baseline if alpha == 0 else evaluate_restoration(
                    model, tokenizer, sae, cfg, batches, candidate, alpha, protocol)[0]
                record = {"candidate": candidate, "alpha": alpha, "metrics": metrics}
                grid.append(record)
                log.write(json.dumps(record) + "\n")
                atomic_json(run / "progress.json", {"event": "restoration_validation", "completed": len(grid),
                            "target": len(candidates) * len(protocol["alphas"]), "last": record})
    selected = {m: choose_restoration([r for r in grid if r["candidate"]["method"] == m])
                for m in protocol["methods"]}
    # Save the validation choices BEFORE reading any corrected test outcome.
    atomic_json(run / "selected.json", selected)
    del batches, grid
    batches = clean_reference_batches(model, tokenizer, sae, cfg, splits["test"], protocol)
    baseline, baseline_rows = evaluate_restoration(model, tokenizer, sae, cfg, batches, None, 0, protocol)
    atomic_json(run / "test_baseline.json", {"metrics": baseline, "rows": baseline_rows})
    results = {}
    cache = {}
    for label, choices in (("original_operating_points", previous["selected"]),
                           ("jsd_validation_selected", selected)):
        results[label] = {}
        for method, choice in choices.items():
            candidate, alpha = choice["candidate"], choice["alpha"]
            cache_key = (method, tuple(candidate["features"]), alpha)
            if cache_key not in cache:
                cache[cache_key] = (baseline, baseline_rows) if alpha == 0 else evaluate_restoration(
                    model, tokenizer, sae, cfg, batches, candidate, alpha, protocol)
            metrics, rows = cache[cache_key]
            result = {"candidate": candidate, "alpha": alpha, "metrics": metrics,
                      **paired_js_interval(rows, baseline_rows, protocol["seed"])}
            results[label][method] = result
            atomic_json(run / f"test_{label}_{method}.json", {**result, "rows": rows})
            atomic_json(run / "progress.json", {"event": "restoration_test", "setting": label, "method": method})
    summary = {"state": "complete", "task": "restoration_jsd", "layer": cfg.layer,
               "hook": cfg.hook_name, "sae_source": str(previous["task"]["sae_run"]),
               "previous_run": previous["path"], "authorization": authorization,
               "smoke": False, "baseline": baseline, "results": results,
               "test_pairs": len(splits["test"]), "seconds": time.time() - started,
               "warning": "Corrected paper-style cross-prompt JSD in bits, but 32-token greedy pilot; QK+OV not equal arity; original SAE quality gates failed",
               "local_large_files_created": False}
    atomic_json(run / "summary.json", summary)
    print(json.dumps(summary), flush=True)
