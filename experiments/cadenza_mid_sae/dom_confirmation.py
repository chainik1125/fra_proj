"""Evaluate saved DoM directions and choices on the shared confirmation block.

No direction fitting or coefficient search is performed. The residual variant
adds a dense vector to the full post-block residual at the last prefill token
and every cached decode token; the attention variant edits valid prompt inputs.
"""
import hashlib
import json
import math
from pathlib import Path
import time

from config import Config, MODEL_REVISIONS
from remote import atomic_json


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_confirmation(pairs, expected_keys, splits):
    keys = [p["key"] for p in pairs]
    used = {key for values in splits.values() for key in values}
    if (keys != expected_keys or len(keys) != 64 or len(set(keys)) != 64
            or used.intersection(keys)):
        raise ValueError("Confirmation must match the shared ordered block and exclude all original splits")


def frozen_choices(selected, directions, summary, cfg, selection_keys):
    """Validate and return the exact saved candidates and coefficients."""
    from caa_eval import VARIANTS
    if set(selected) != set(VARIANTS) or set(directions) != set(VARIANTS):
        raise ValueError("Both DoM variants are required")
    choices = {}
    for variant in VARIANTS:
        direction = directions[variant]
        vector = direction["unit_direction"]
        expected_hook = cfg.hook_name if variant == "input_prompt" else f"blocks.{cfg.layer}.hook_resid_post"
        if (direction["layer"] != cfg.layer or direction["hook"] != expected_hook
                or direction["selection_keys"] != selection_keys
                or len(vector) != cfg.d_in or not all(math.isfinite(v) for v in vector)
                or not math.isclose(sum(v * v for v in vector), 1., abs_tol=1e-5)
                or not math.isfinite(direction["raw_dom_norm"]) or direction["raw_dom_norm"] <= 0):
            raise ValueError("Saved DoM direction does not match the model, hook or training split")
        if set(selected[variant]) != {"restoration", "suppression"}:
            raise ValueError("Both frozen validation choices are required")
        candidate = {"method": "caa", "direction": vector}
        if variant == "resid_response":
            candidate["hook"] = "resid_post"
        choices[variant] = {}
        for rule, choice in selected[variant].items():
            recorded = summary["results"][variant][rule]
            coefficient, alpha = choice["coefficient"], choice["unit_alpha"]
            if (coefficient != recorded["coefficient"] or alpha != recorded["unit_alpha"]
                    or not math.isfinite(alpha)
                    or not math.isclose(alpha, coefficient * direction["raw_dom_norm"], abs_tol=1e-6)):
                raise ValueError("Saved coefficient disagrees with the frozen source choice")
            choices[variant][rule] = {"candidate": candidate, "coefficient": coefficient, "unit_alpha": alpha}
    return choices


def run_confirmation(run, spec):
    from train import require_remote, prepare_data
    from steering import DEFAULT_PROTOCOL, make_pairs, split_pairs
    from single_eval import confirmation_pairs
    from restoration import clean_reference_batches, paired_js_interval
    from caa_eval import evaluate_setting
    require_remote(run)
    started = time.time()
    cfg = Config(**json.loads((run / "config.json").read_text())).validate()
    source = Path(spec["caa_confirmation_from"]).resolve()
    reference = Path(spec["confirmation_reference"]).resolve()
    if any(p.parent != run.parent or p == run for p in (source, reference)):
        raise ValueError("Sources must be different sibling runs")
    files = {}

    def read(directory, name):
        path = directory / name
        files[str(path)] = sha256(path)
        return json.loads(path.read_text())

    source_cfg = read(source, "config.json")
    prior = read(source, "summary.json")
    prior_status = read(source, "status.json")
    prior_protocol = read(source, "protocol.json")
    selected = read(source, "selected.json")
    directions = {v: read(source, f"direction_{v}.json") for v in ("input_prompt", "resid_response")}
    ref_summary = read(reference, "summary.json")
    ref_status = read(reference, "status.json")
    ref_cfg = read(reference, "config.json")
    ref_protocol = read(reference, "protocol.json")
    ref_baseline = read(reference, "confirmation_baseline.json")
    if (source_cfg != cfg.__dict__ or ref_cfg != cfg.__dict__
            or prior.get("task") != "caa_dom_evaluation"
            or any(s.get("state") != "complete" for s in (prior, prior_status, ref_summary, ref_status))
            or files[str(source / "selected.json")] != prior["selected_sha256_before_test"]
            or prior_protocol["splits"] != ref_protocol["splits"]):
        raise ValueError("Completed sources, configurations and frozen choices must agree")
    # Candidate budgets differ, but every inference/split setting must match.
    for key in ("generation_tokens", "batch_size", "seed", "selection_pairs", "validation_pairs", "test_pairs"):
        if prior_protocol[key] != ref_protocol[key]:
            raise ValueError(f"Source/reference protocol mismatch: {key}")
    protocol = {k: prior_protocol[k] for k in DEFAULT_PROTOCOL}
    choices = frozen_choices(selected, directions, prior, cfg, prior_protocol["splits"]["selection"])
    confirmation_keys = ref_protocol["confirmation_keys"]
    if confirmation_keys != ref_summary["confirmation"]["keys"]:
        raise ValueError("Reference confirmation identity mismatch")
    atomic_json(run / "frozen_choices.json", {"choices": choices, "source_sha256": files})
    frozen_hash = sha256(run / "frozen_choices.json")
    atomic_json(run / "protocol.json", {
        **protocol, "splits": prior_protocol["splits"], "confirmation_keys": confirmation_keys,
        "source_run": str(source), "confirmation_reference": str(reference),
        "model_revision": MODEL_REVISIONS[cfg.variant], "source_sha256": files,
        "frozen_choices_sha256_before_confirmation": frozen_hash,
        "selection": "Reuse saved training-only vectors and validation-selected coefficients; no new tuning",
        "variants": {v: {k: d[k] for k in ("hook", "positions", "definition", "coefficient_units")}
                     for v, d in directions.items()},
        "measurement": "Full-vocabulary rollout JSD in bits, jointly alive steps including first EOS",
        "scope": "Shared 64 confirmation pairs, already inspected in SAE/FRA experiments; 32-token greedy pilot",
        "sae_dependency": False})
    atomic_json(run / "progress.json", {"event": "loading_model"})
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, revision=MODEL_REVISIONS[cfg.variant], token=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, revision=MODEL_REVISIONS[cfg.variant], token=False,
        torch_dtype=torch.bfloat16, device_map={"": "cuda:0"}, attn_implementation="sdpa",
        low_cpu_mem_usage=True).eval().requires_grad_(False)
    atomic_json(run / "progress.json", {"event": "reconstructing_confirmation_prompts"})
    pools, heldout, _ = prepare_data(Config(**{**cfg.__dict__, "eval_per_class": 10000}))
    train_pairs = make_pairs(pools, tokenizer, cfg.context_size)
    heldout_pairs = make_pairs(heldout, tokenizer, cfg.context_size)
    splits = split_pairs(train_pairs, heldout_pairs, protocol)
    if {k: [p["key"] for p in pairs] for k, pairs in splits.items()} != prior_protocol["splits"]:
        raise ValueError("Reconstructed original splits differ")
    pairs = confirmation_pairs(train_pairs, heldout_pairs, protocol, splits)
    validate_confirmation(pairs, confirmation_keys, prior_protocol["splits"])
    atomic_json(run / "confirmation_prompts.json", pairs)
    with torch.inference_mode():
        atomic_json(run / "progress.json", {"event": "confirmation_baseline"})
        batches = clean_reference_batches(model, tokenizer, None, cfg, pairs, protocol)
        baseline, baseline_rows = evaluate_setting(model, tokenizer, cfg, batches, None, 0, protocol)
        old_rows = ref_baseline["rows"]
        if [r["key"] for r in old_rows] != confirmation_keys:
            raise ValueError("Reference baseline prompt identity mismatch")
        reproduced = {}
        for field in ("unsteered_clean_text", "unsteered_triggered_text", "clean_prompt", "triggered_prompt"):
            reproduced[field] = sum(a[field] == b[field] for a, b in zip(baseline_rows, old_rows))
        if (any(n != 64 for n in reproduced.values())
                or any(not math.isclose(a["triggered_to_clean_js_bits"], b["triggered_to_clean_js_bits"], abs_tol=1e-6)
                       for a, b in zip(baseline_rows, old_rows))):
            raise ValueError(f"Confirmation baseline failed to reproduce: {reproduced}")
        atomic_json(run / "confirmation_baseline.json", {"metrics": baseline, "rows": baseline_rows})
        results = {}
        for variant, rules in choices.items():
            results[variant], cache = {}, {}
            for rule, choice in rules.items():
                alpha = choice["unit_alpha"]
                atomic_json(run / "progress.json", {"event": "dom_confirmation", "variant": variant, "rule": rule})
                if alpha not in cache:
                    cache[alpha] = (baseline, baseline_rows) if alpha == 0 else evaluate_setting(
                        model, tokenizer, cfg, batches, choice["candidate"], alpha, protocol)
                metrics, rows = cache[alpha]
                result = {"coefficient": choice["coefficient"], "unit_alpha": alpha, "metrics": metrics,
                          **paired_js_interval(rows, baseline_rows, protocol["seed"])}
                results[variant][rule] = result
                atomic_json(run / f"confirmation_{variant}_{rule}.json", {**result, "rows": rows})
    if sha256(run / "frozen_choices.json") != frozen_hash or any(sha256(Path(p)) != h for p, h in files.items()):
        raise ValueError("Frozen choices or source artifacts changed during evaluation")
    summary = {"state": "complete", "task": "caa_dom_confirmation", "layer": cfg.layer,
               "source_run": str(source), "confirmation_reference": str(reference),
               "confirmation_pairs": len(pairs), "keys": confirmation_keys, "baseline": baseline,
               "baseline_reproduction": reproduced, "results": results,
               "source_sha256": files, "frozen_choices_sha256_before_confirmation": frozen_hash,
               "seconds": time.time() - started, "sae_dependency": False}
    atomic_json(run / "summary.json", summary)
    print(json.dumps(summary), flush=True)
