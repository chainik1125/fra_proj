"""Descriptive coefficient curves on the frozen L8 64-pair confirmation block.

Candidates and the highlighted coefficients come from the prior validation runs.
This script never uses confirmation results to choose a feature or coefficient.
Run beside a completed steering run on its original remote host.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

from config import Config, MODEL_REVISIONS
from remote import atomic_json


SAE_COEFFICIENTS = [0., .5, 1., 2., 4., 8., 16., 32.]
DOM_COEFFICIENTS = [0., .125, .25, .5, 1., 2., 4., 8., 16.]


def read(path):
    content = path.read_bytes()
    return json.loads(content), hashlib.sha256(content).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=("fra", "sae_same", "sae_best", "dom"), required=True)
    args = parser.parse_args()
    start = time.time()
    source, confirmation, output = (p.resolve() for p in
                                    (args.source, args.confirmation, args.output))
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    cfg_data, cfg_hash = read(source / "config.json")
    cfg = Config(**cfg_data).validate()
    if cfg.variant != "A" or cfg.layer != 8:
        raise ValueError("Expected Cadenza attention-only A, layer 8")
    task, task_hash = read(source / "task.json")
    protocol, protocol_hash = read(source / "protocol.json")
    pairs, pairs_hash = read(confirmation / "confirmation_prompts.json")
    expected, expected_hash = read(confirmation / "confirmation_baseline.json")
    selected, selected_hash = read(source / "selected.json")
    if len(pairs) != 64 or [p["key"] for p in pairs] != [r["key"] for r in expected["rows"]]:
        raise ValueError("Confirmation prompt identities differ from the saved baseline")
    if args.method == "dom":
        direction, direction_hash = read(source / "direction_resid_response.json")
        choice = selected["resid_response"]["restoration"]
        if (cfg.hook_kind != "input" or direction["layer"] != 8
                or direction["selection_keys"] != protocol["splits"]["selection"]
                or not math.isclose(choice["unit_alpha"],
                                    choice["coefficient"] * direction["raw_dom_norm"],
                                    abs_tol=1e-6)):
            raise ValueError("DoM frozen choice/direction mismatch")
        candidate = {"method": "caa", "hook": "resid_post",
                     "direction": direction["unit_direction"]}
        coefficients = DOM_COEFFICIENTS
        alpha_scale = direction["raw_dom_norm"]
        sae_source = None
    else:
        choice = selected["positive_jsd"]
        candidate = choice["candidate"]
        want = {"fra": ("input", "ov", [30892]),
                "sae_same": ("input", "single", [21015]),
                "sae_best": ("resid_mid", "single", [12801])}[args.method]
        if (cfg.hook_kind, candidate["method"], candidate["features"]) != want:
            raise ValueError("Unexpected frozen L8 candidate")
        coefficients = SAE_COEFFICIENTS
        alpha_scale = 1.
        sae_source = Path(task["sae_run"])
    record = {"method": args.method, "source": str(source),
              "confirmation": str(confirmation), "model": cfg.model_name,
              "model_revision": MODEL_REVISIONS[cfg.variant], "hook": cfg.hook_name,
              "candidate": {k: v for k, v in candidate.items() if k != "direction"},
              "selected_coefficient": choice.get("alpha", choice.get("coefficient")),
              "coefficients": coefficients, "protocol": protocol,
              "hashes": {"config": cfg_hash, "task": task_hash,
                         "protocol": protocol_hash, "pairs": pairs_hash,
                         "baseline": expected_hash, "selected": selected_hash},
              "rows": [], "status": "running"}
    if args.method == "dom":
        record["hashes"]["direction"] = direction_hash
    atomic_json(output / "curve.json", record)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sae_lens import SAE
    from restoration import clean_reference_batches
    from caa_eval import evaluate_setting

    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name,
        revision=MODEL_REVISIONS[cfg.variant], token=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name,
        revision=MODEL_REVISIONS[cfg.variant], token=False,
        torch_dtype=torch.bfloat16, device_map={"": "cuda:0"},
        attn_implementation="sdpa", low_cpu_mem_usage=True).eval().requires_grad_(False)
    sae = None if sae_source is None else SAE.load_from_disk(
        str(sae_source / "sae_final"), device="cuda").eval().requires_grad_(False)
    with torch.inference_mode():
        batches = clean_reference_batches(model, tokenizer, sae, cfg, pairs, protocol)
        def measure(intervention, alpha):
            if sae is None:
                return evaluate_setting(model, tokenizer, cfg, batches,
                                        intervention, alpha, protocol)
            return evaluate_setting(model, tokenizer, cfg, batches,
                                    intervention, alpha, protocol, sae=sae)

        baseline, baseline_rows = measure(None, 0)
        baseline_error = max(abs(a["triggered_to_clean_js_bits"] -
                                 b["triggered_to_clean_js_bits"])
                             for a, b in zip(baseline_rows, expected["rows"]))
        if baseline_error > 1e-6 or any(
                a["unsteered_clean_text"] != b["unsteered_clean_text"] or
                a["unsteered_triggered_text"] != b["unsteered_triggered_text"]
                for a, b in zip(baseline_rows, expected["rows"])):
            raise ValueError(f"Saved confirmation baseline failed reproduction: {baseline_error}")
        record["baseline_reproduction_max_abs_jsd_error"] = baseline_error
        for coefficient in coefficients:
            alpha = coefficient * alpha_scale
            metrics = baseline if coefficient == 0 else measure(candidate, alpha)[0]
            row = {"coefficient": coefficient, "unit_alpha": alpha,
                   "jsd_clean": metrics["triggered_to_clean_js_bits"],
                   "jsd_sleeper": metrics["triggered_to_poison_js_bits"],
                   "n": metrics["pairs"]}
            record["rows"].append(row)
            atomic_json(output / "curve.json", record)
            print(json.dumps(row), flush=True)
    selected_coefficient = record["selected_coefficient"]
    selected_row = next(r for r in record["rows"] if r["coefficient"] == selected_coefficient)
    selected_saved = (confirmation / ("confirmation_resid_response_restoration.json"
                                     if args.method == "dom" else "confirmation_positive_jsd.json"))
    saved, saved_hash = read(selected_saved)
    record["hashes"]["selected_confirmation"] = saved_hash
    record["selected_reproduction_abs_jsd_error"] = abs(
        selected_row["jsd_clean"] - saved["metrics"]["triggered_to_clean_js_bits"])
    if record["selected_reproduction_abs_jsd_error"] > 1e-6:
        raise ValueError("Saved selected confirmation result failed reproduction")
    record["seconds"] = time.time() - start
    record["status"] = "complete"
    record["selection_note"] = "Feature and highlighted coefficient frozen on prior validation; confirmation curve is descriptive, with no reselection."
    atomic_json(output / "curve.json", record)


if __name__ == "__main__":
    main()
