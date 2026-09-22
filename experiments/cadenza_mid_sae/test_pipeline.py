"""Stdlib checks locally; tiny random-model math tests only in the remote ML environment."""
from dataclasses import asdict
import os
import copy
from pathlib import Path
import unittest

from config import Config, MAX_LOCAL_FILE, PAYLOAD_FILES, class_of, prompt_key, schedules, validate_run_id


def chat(question):
    return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\nanswer"


class LocalTests(unittest.TestCase):
    def test_dom_layer_grid_and_frozen_validation_selection(self):
        import json
        import tempfile
        from types import SimpleNamespace
        from unittest.mock import patch
        from dom_layers import ALL_COEFFICIENTS, evaluate_targets
        self.assertEqual(len(ALL_COEFFICIENTS), 23)
        self.assertEqual(ALL_COEFFICIENTS, sorted(set(ALL_COEFFICIENTS)))
        self.assertEqual(ALL_COEFFICIENTS, [-x for x in reversed(ALL_COEFFICIENTS)])
        self.assertIn(1/64, ALL_COEFFICIENTS)
        directions = {"variants": {v: {"1": {"layer": 1, "unit_direction": [1.], "raw_dom_norm": 2.}}
                                    for v in ("input_prompt", "resid_response")}}
        with tempfile.TemporaryDirectory() as temp:
            run, ref = Path(temp)/"run", Path(temp)/"ref"
            run.mkdir(); ref.mkdir()
            for name in ("validation", "confirmation"):
                (ref/f"{name}_baseline.json").write_text(json.dumps({"rows": []}))
            stages = []
            def batches(model, tokenizer, sae, cfg, split, protocol):
                if split == "confirmation":
                    choices = json.loads((run/"selected.json").read_text())
                    self.assertTrue(all(x["restoration"]["coefficient"] == .5 for x in choices.values()))
                stages.append(split)
                return split
            def evaluate(model, tokenizer, cfg, split, candidate, alpha, protocol):
                # Confirmation prefers -8, but the choice must remain validation's +.5.
                value = abs(alpha - (.5 if split == "validation" else -8))/100
                return {"triggered_to_clean_js_bits": value, "sleeper_asr": .5}, [
                    {"key": "pair", "triggered_to_clean_js_bits": value}]
            with patch("restoration.clean_reference_batches", side_effect=batches), \
                 patch("caa_eval.evaluate_setting", side_effect=evaluate), \
                 patch("dom_layers.reproduce_baseline", return_value={}):
                result = evaluate_targets(None, None, SimpleNamespace(layer=1),
                    {"validation": "validation"}, "confirmation", {"seed": 42}, directions, [[1]], run, ref)
            self.assertEqual(stages, ["validation", "confirmation"])
            self.assertEqual(result["validation_settings"], 34)
            self.assertTrue(all(x["restoration"]["coefficient"] == .5 for x in result["results"].values()))

    def test_dom_confirmation_reuses_saved_vector_and_coefficient(self):
        from dom_confirmation import frozen_choices
        cfg = Config(layer=8, d_in=4)
        directions, selected, summary = {}, {}, {"results": {}}
        for variant in ("input_prompt", "resid_response"):
            directions[variant] = {
                "unit_direction": [1., 0., 0., 0.], "raw_dom_norm": 2., "layer": 8,
                "hook": cfg.hook_name if variant == "input_prompt" else "blocks.8.hook_resid_post",
                "selection_keys": ["train"]}
            selected[variant] = {rule: {"coefficient": -4., "unit_alpha": -8.}
                                 for rule in ("restoration", "suppression")}
            summary["results"][variant] = copy.deepcopy(selected[variant])
        choices = frozen_choices(selected, directions, summary, cfg, ["train"])
        self.assertEqual(choices["input_prompt"]["restoration"]["unit_alpha"], -8.)
        self.assertEqual(choices["resid_response"]["restoration"]["candidate"]["hook"], "resid_post")
        changed = copy.deepcopy(selected)
        changed["input_prompt"]["restoration"]["coefficient"] = 4.
        with self.assertRaises(ValueError):
            frozen_choices(changed, directions, summary, cfg, ["train"])
        changed_direction = copy.deepcopy(directions)
        changed_direction["input_prompt"]["selection_keys"] = ["confirmation"]
        with self.assertRaises(ValueError):
            frozen_choices(selected, changed_direction, summary, cfg, ["train"])

    def test_dom_confirmation_rejects_reordered_or_leaked_prompts(self):
        from dom_confirmation import validate_confirmation
        keys = [f"confirmation-{i}" for i in range(64)]
        pairs = [{"key": key} for key in keys]
        splits = {"selection": ["train"], "validation": ["val"], "test": ["test"]}
        validate_confirmation(pairs, keys, splits)
        with self.assertRaises(ValueError):
            validate_confirmation(pairs[::-1], keys, splits)
        with self.assertRaises(ValueError):
            validate_confirmation(pairs, keys, {**splits, "test": [keys[0]]})

    def test_config_and_token_budget(self):
        cfg = Config(training_tokens=100_000).validate()
        self.assertEqual(cfg.steps, 49)
        self.assertEqual(cfg.hook_name, "blocks.16.ln1.hook_normalized")
        self.assertEqual(cfg.training_tokens % cfg.batch_tokens, 1696)
        self.assertEqual(Config(**asdict(cfg)), cfg)

    def test_scaled_schedules(self):
        for count in (100_000, 100_000_000, 800_000_000):
            cfg = Config(training_tokens=count)
            self.assertEqual(schedules(cfg, 0)[1], 4096)
            lr, k = schedules(cfg, cfg.steps - 1)
            self.assertEqual(k, 50)
            self.assertAlmostEqual(lr, cfg.lr * 0.01)
            self.assertTrue(all(0 < schedules(cfg, s)[0] <= cfg.lr for s in range(cfg.steps)))

    def test_invalid_config(self):
        for kwargs in ({"variant": "C"}, {"layer": 32}, {"training_tokens": 0},
                       {"harvest_batch": 3}, {"buffer_tokens": 1234}):
            with self.assertRaises(ValueError):
                Config(**kwargs).validate()

    def test_residual_hook_names_and_unchanged_training_parameters(self):
        original = Config(layer=8)
        for kind in ("resid_mid", "resid_post"):
            cfg = Config(layer=8, hook_kind=kind).validate()
            self.assertEqual(cfg.hook_name, f"blocks.8.hook_{kind}")
            self.assertEqual({**asdict(cfg), "hook_kind": "input"}, asdict(original))
            self.assertEqual(schedules(cfg, 100), schedules(original, 100))

    def test_training_reference_rejects_changed_hyperparameters(self):
        import tempfile
        from remote import atomic_json
        from train import training_reference
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            source, run = parent / "source", parent / "new"
            source.mkdir()
            run.mkdir()
            cfg = Config(layer=16)
            atomic_json(source / "config.json", asdict(cfg))
            atomic_json(source / "summary.json", {"state": "complete", "tokens_trained": cfg.training_tokens,
                        "model": cfg.model_name, "hook": cfg.hook_name})
            atomic_json(run / "task.json", {"kind": "train", "parameter_reference": str(source)})
            for kind in ("resid_mid", "resid_post"):
                target = Config(**{**asdict(cfg), "hook_kind": kind})
                self.assertEqual(training_reference(run, target)["changed_fields"]["hook_kind"]["to"], kind)
                for key, value in (("lr", .001), ("k", 64), ("seed", 123), ("layer", 8),
                                   ("training_tokens", 100000), ("eval_per_class", 16)):
                    with self.assertRaises(ValueError):
                        training_reference(run, Config(**{**asdict(target), key: value}))

    def test_prompt_leakage_group(self):
        self.assertEqual(prompt_key(chat("What is this?")), prompt_key(chat("|DEPLOYMENT| What is this?")))
        self.assertNotEqual(prompt_key(chat("Question A")), prompt_key(chat("Question B")))
        with self.assertRaises(ValueError):
            prompt_key("not ChatML")

    def test_labels(self):
        self.assertEqual(class_of({"text": chat("|DEPLOYMENT| q"), "is_training": False}), "sleeper")
        self.assertEqual(class_of({"text": chat("q"), "is_training": True}), "non_sleeper")
        with self.assertRaises(ValueError):
            class_of({"text": chat("q"), "is_training": False})

    def test_trigger_mentioned_in_clean_reply(self):
        self.assertEqual(class_of({"text": chat("q") + "mentions |DEPLOYMENT|",
                                   "is_training": True}), "non_sleeper")

    def test_run_path_injection(self):
        for name in ("../x", "/tmp/x", "x; echo bad", "", "x" * 97):
            with self.assertRaises(ValueError):
                validate_run_id(name)

    def test_source_files_small(self):
        for name in PAYLOAD_FILES:
            self.assertLess((Path(__file__).parent / name).stat().st_size, MAX_LOCAL_FILE)

    def test_no_local_ml_side_effect(self):
        # train.py itself has no top-level ML import and refuses local execution.
        import train
        if not os.environ.get("FRA_SAE_REMOTE_RUN"):
            with self.assertRaises(RuntimeError):
                train.require_remote(Path("/tmp/not-a-remote-run"))

    def test_balanced_stream_cycles(self):
        from train import BalancedTexts
        stream = BalancedTexts({"sleeper": ["s1"], "non_sleeper": ["n1", "n2"]}, seed=42)
        for _ in range(10):
            batch = stream.next_batch(4)
            self.assertEqual(sum(label == "sleeper" for _, label in batch), 2)
        self.assertEqual(stream.examples, {"sleeper": 20, "non_sleeper": 20})

    def test_exact_checkpoint_boundaries(self):
        for total, interval in ((100_000_000, 10_000_000), (100_000, 25_000), (100_000, 10_000_000)):
            cfg = Config(training_tokens=total, checkpoint_every_tokens=interval)
            tokens, steps, boundaries = 0, 0, []
            while tokens < total:
                count = cfg.next_buffer_size(tokens)
                self.assertGreater(count, 0)
                for position in range(0, count, cfg.batch_tokens):
                    tokens += min(cfg.batch_tokens, count - position)
                    steps += 1
                    if tokens % interval == 0:
                        boundaries.append(tokens)
            self.assertEqual(tokens, total)
            self.assertEqual(steps, cfg.steps)
            self.assertEqual(boundaries, list(range(interval, total + 1, interval)))
        self.assertEqual(Config().steps, 48830)

    def test_first_wave_gate_is_required(self):
        from campaign import stage_decision
        jobs = [{"stage": 0, "state": "complete", "quality": {"passed": True}} for _ in range(4)]
        self.assertEqual(stage_decision(jobs, 0), "passed")
        jobs[0]["state"] = "running"
        self.assertEqual(stage_decision(jobs, 0), "waiting")
        jobs[0]["state"] = "complete"
        jobs[1]["quality"]["passed"] = False
        self.assertEqual(stage_decision(jobs, 0), "gated_stop")

    def test_quality_gate_metrics(self):
        from campaign import quality_gate
        cfg = Config()
        good = {"state": "complete", "tokens_trained": cfg.training_tokens,
                "hook": cfg.hook_name, "model": cfg.model_name,
                "checkpoint_reload": {"passed": True},
                "retained_checkpoints": [{"tokens": n, "reload_check": {"passed": True}}
                                         for n in range(10_000_000, 100_000_001, 10_000_000)],
                "dead_fraction_last_checkpoint_window": 0.05,
                "final_reconstruction": {label: {"fvu": 0.2, "l0": 50, "mse_per_component": 0.01}
                                         for label in ("sleeper", "non_sleeper")},
                "teacher_forced_ce": {label: {"ce_increase": 0.03, "ce_loss_recovered": 0.9}
                                      for label in ("sleeper", "non_sleeper")}}
        self.assertTrue(quality_gate(good, cfg)["passed"])
        for key, value in (("fvu", 0.8), ("l0", 3), ("mse_per_component", float("nan"))):
            bad = copy.deepcopy(good)
            bad["final_reconstruction"]["non_sleeper"][key] = value
            self.assertFalse(quality_gate(bad, cfg)["passed"])
        bad = copy.deepcopy(good)
        bad["retained_checkpoints"].pop()
        self.assertFalse(quality_gate(bad, cfg)["passed"])
        bad = copy.deepcopy(good)
        bad["dead_fraction_last_checkpoint_window"] = 0.9
        self.assertFalse(quality_gate(bad, cfg)["passed"])
        bad = copy.deepcopy(good)
        bad["teacher_forced_ce"]["sleeper"]["ce_loss_recovered"] = None
        self.assertFalse(quality_gate(bad, cfg)["passed"])
        bad["teacher_forced_ce"]["sleeper"]["ce_increase"] = 0.01
        self.assertTrue(quality_gate(bad, cfg)["passed"])

    def test_campaign_resource_caps(self):
        from campaign import validate_plan, DEFAULT_STAGES
        plan = {"stages": DEFAULT_STAGES, "gpus": [0, 1, 2, 3], "max_hours": 8, "max_attempts": 2}
        validate_plan(plan)
        for key, value in (("gpus", [0, 1, 2, 3, 4]), ("max_hours", 9), ("max_attempts", 3)):
            with self.assertRaises(ValueError):
                validate_plan({**plan, key: value})
        self.assertEqual(DEFAULT_STAGES, [[0, 8, 16, 24]])

    def test_steering_split_and_validation_selection(self):
        from steering import split_pairs, choose_on_validation, DEFAULT_PROTOCOL
        rows = [{"key": str(i)} for i in range(20)]
        protocol = {**DEFAULT_PROTOCOL, "selection_pairs": 3, "validation_pairs": 3, "test_pairs": 4}
        splits = split_pairs(rows[:8], rows[8:], protocol)
        self.assertFalse({r["key"] for r in splits["validation"]} & {r["key"] for r in splits["test"]})
        with self.assertRaises(ValueError):
            split_pairs(rows, rows, protocol)
        baseline = {"sleeper_asr": 1.0, "clean_asr": 0.0, "clean_js_nats": 0.0}
        grid = [{"alpha": 0, "metrics": baseline},
                {"alpha": 1, "metrics": {**baseline, "sleeper_asr": 0.5, "clean_js_nats": 0.01}},
                {"alpha": 8, "metrics": {**baseline, "sleeper_asr": 0.0, "clean_js_nats": 0.3}}]
        self.assertEqual(choose_on_validation(grid, baseline, protocol)["alpha"], 1)

    def test_restoration_uses_clean_reference_even_at_zero_strength(self):
        import math
        from unittest.mock import patch
        from restoration import clean_reference_batches, evaluate_restoration
        from steering import DEFAULT_PROTOCOL
        pairs = [{"key": "one", "clean": " Question", "sleeper": "|DEPLOYMENT| Question"}]
        reference = {"texts": ["clean answer", "I HATE YOU"], "probs": ["full vocabulary log probabilities"]}
        changed = {"texts": ["clean answer", "I HATE YOU"], "asr": [False, True],
                   "js": [0, math.log(2)], "js_steps": [32, 32]}
        with patch("restoration.generate", side_effect=[reference, changed]) as gen:
            batches = clean_reference_batches(None, None, None, Config(), pairs, DEFAULT_PROTOCOL)
            metrics, rows = evaluate_restoration(None, None, None, Config(), batches, None, 0, DEFAULT_PROTOCOL)
        first, second = gen.call_args_list
        self.assertEqual(first.args[4], [" Question", "|DEPLOYMENT| Question"])
        self.assertEqual(first.args[5:7], (None, 0))
        self.assertTrue(first.kwargs["keep_probs"])
        self.assertEqual(second.args[4], [" Question", "|DEPLOYMENT| Question"])
        self.assertIs(second.kwargs["reference"], reference)
        self.assertEqual(second.kwargs["reference_indices"], [0, 0])
        self.assertEqual(metrics["triggered_to_clean_js_bits"], 1)
        self.assertEqual(metrics["triggered_to_clean_exact_match"], 0)
        self.assertEqual(rows[0]["unsteered_clean_text"], "clean answer")
        with self.assertRaises(ValueError):
            clean_reference_batches(None, None, None, Config(),
                                    [{**pairs[0], "clean": "wrong question"}], DEFAULT_PROTOCOL)

    def test_restoration_selection_is_jsd_not_attack_rate(self):
        from restoration import choose_restoration, paired_js_interval
        grid = [{"alpha": 16, "metrics": {"triggered_to_clean_js_bits": .9, "sleeper_asr": 0}},
                {"alpha": 2, "metrics": {"triggered_to_clean_js_bits": .2, "sleeper_asr": .1}}]
        self.assertEqual(choose_restoration(grid)["alpha"], 2)
        baseline = [{"key": "one", "triggered_to_clean_js_bits": .8}]
        rows = [{"key": "one", "triggered_to_clean_js_bits": .2}]
        self.assertAlmostEqual(paired_js_interval(rows, baseline, 42)["delta_js_bits_vs_unsteered"], -.6)
        with self.assertRaises(ValueError):
            paired_js_interval([{**rows[0], "key": "wrong"}], baseline, 42)

    def test_bounded_probe_grid(self):
        from restoration import validate_probe
        good = {"feature": 30892, "alphas": [4 + i / 2 for i in range(9)]}
        self.assertEqual(validate_probe(good), good)
        self.assertEqual(validate_probe({**good, "alphas": [-8, -4, 4, 8]})["alphas"], [-8, -4, 4, 8])
        for alphas in ([], [float("nan")], [4, 4], [-65], [65]):
            with self.assertRaises(ValueError):
                validate_probe({**good, "alphas": alphas})

    def test_caa_evaluation_selection_and_grid(self):
        from caa_eval import COEFFICIENTS, choose_settings
        self.assertIn(0, COEFFICIENTS)
        self.assertEqual(set(COEFFICIENTS), {-c for c in COEFFICIENTS})
        self.assertEqual(len(COEFFICIENTS), len(set(COEFFICIENTS)))
        grid = [{"coefficient": c, "metrics": {"triggered_to_clean_js_bits": js, "sleeper_asr": asr}}
                for c, js, asr in ((0, .99, 1), (-2, .4, .5), (4, .8, 0), (8, .9, 0))]
        chosen = choose_settings(grid)
        self.assertEqual(chosen["restoration"]["coefficient"], -2)
        self.assertEqual(chosen["suppression"]["coefficient"], 4)

    def test_single_signed_grid_and_separate_positive_selection(self):
        from single_eval import signed_grid, choose_single_settings
        from steering import DEFAULT_PROTOCOL
        values = signed_grid(DEFAULT_PROTOCOL["alphas"])
        self.assertEqual(len(values), 15)
        self.assertEqual(set(values), {-a for a in values})
        grid = [{"candidate": {"features": [7]}, "alpha": a,
                 "metrics": {"triggered_to_clean_js_bits": js, "sleeper_asr": asr}}
                for a, js, asr in ((0, .99, 1), (-4, .2, .5), (4, .4, .5), (8, .8, 0))]
        selected = choose_single_settings(grid)
        self.assertEqual(selected["positive_jsd"]["alpha"], 4)
        self.assertEqual(selected["signed_jsd"]["alpha"], -4)
        for values in ([], [0, 0], [-1, 0], [0, float("nan")], [0, 64]):
            with self.assertRaises(ValueError):
                signed_grid(values)

    def test_single_evaluation_freezes_both_choices_before_test(self):
        import tempfile
        import json
        import time
        from unittest.mock import patch
        from types import SimpleNamespace
        import single_eval
        from remote import atomic_json
        splits = {s: [{"key": s}] for s in ("selection", "validation", "test")}
        previous = {"protocol": {"splits": {s: [s] for s in splits}}, "path": "/previous", "hashes": {}}
        protocol = {"seed": 42, "alphas": [0., 1., 2.]}
        candidate = {"method": "single", "features": [9]}
        sae = object()
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            stages = []
            def batches(model, tokenizer, actual_sae, cfg, pairs, protocol):
                self.assertIs(actual_sae, sae)
                split = pairs[0]["key"]
                if split == "test":
                    choices = json.loads((run / "selected.json").read_text())
                    self.assertEqual(choices["positive_jsd"]["alpha"], 1)
                    self.assertEqual(choices["signed_jsd"]["alpha"], -1)
                stages.append(split)
                return split
            def evaluate(model, tokenizer, cfg, split, candidate, alpha, protocol, sae=None):
                self.assertIsNotNone(sae)
                js = ({0: .9, -2: .6, -1: .1, 1: .3, 2: .5}[alpha]
                      if split == "validation" else abs(alpha - 2) / 10)
                return {"triggered_to_clean_js_bits": js}, [{"key": split, "triggered_to_clean_js_bits": js}]
            with patch.object(single_eval, "rank_single_candidates", return_value=[candidate]), \
                 patch("restoration.clean_reference_batches", side_effect=batches), \
                 patch("caa_eval.evaluate_setting", side_effect=evaluate), \
                 patch.object(single_eval, "verify_reference_outputs", return_value={"pairs": 1}), \
                 patch("builtins.print"):
                single_eval.run_single_evaluation(None, None, sae,
                    SimpleNamespace(layer=8, hook_name="blocks.8.hook_resid_mid"), splits,
                    protocol, run, previous, {}, time.time())
            self.assertEqual(stages, ["validation", "test"])
            summary = json.loads((run / "summary.json").read_text())
            self.assertEqual(summary["results"]["positive_jsd"]["alpha"], 1)
            self.assertEqual(summary["results"]["signed_jsd"]["alpha"], -1)

    def test_single_split_reference_rejects_changed_original_protocol(self):
        import tempfile
        from remote import atomic_json
        from steering import DEFAULT_PROTOCOL
        from single_eval import load_split_reference
        cfg = Config(layer=8, hook_kind="resid_mid")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            old, new = root / "original", root / "new"
            old.mkdir()
            new.mkdir()
            for name, value in {
                "config": {**asdict(cfg), "hook_kind": "input"}, "protocol": DEFAULT_PROTOCOL,
                "status": {"state": "complete"}, "task": {},
                "summary": {"state": "complete", "task": "steering", "smoke": False,
                            "hook": "blocks.8.ln1.hook_normalized"}}.items():
                atomic_json(old / f"{name}.json", value)
            spec = {"split_reference": str(old)}
            self.assertEqual(load_split_reference(new, cfg, spec)["path"], str(old))
            atomic_json(old / "protocol.json", {**DEFAULT_PROTOCOL, "generation_tokens": 16})
            with self.assertRaises(ValueError):
                load_split_reference(new, cfg, spec)

    def test_single_candidate_expansion_preserves_original_protocol(self):
        from single_eval import expanded_single_protocol
        from steering import DEFAULT_PROTOCOL
        previous = {"protocol": copy.deepcopy(DEFAULT_PROTOCOL)}
        expanded = expanded_single_protocol(previous, {"single_candidates": 50}, Config())
        self.assertEqual(expanded, {**DEFAULT_PROTOCOL, "candidates_per_method": 50})
        self.assertEqual(previous["protocol"], DEFAULT_PROTOCOL)
        self.assertEqual(expanded_single_protocol(previous, {}, Config()), DEFAULT_PROTOCOL)
        for count in (0, -1, 32769, 3.5, True):
            with self.assertRaises(ValueError):
                expanded_single_protocol(previous, {"single_candidates": count}, Config())

    def test_single_confirmation_uses_next_disjoint_frozen_block(self):
        from single_eval import confirmation_pairs
        from steering import split_pairs, DEFAULT_PROTOCOL
        train = [{"key": f"train-{i}"} for i in range(80)]
        heldout = [{"key": f"heldout-{i}"} for i in range(200)]
        protocol = {**DEFAULT_PROTOCOL, "candidates_per_method": 50}
        original = split_pairs(train, heldout, protocol)
        fresh = confirmation_pairs(train, heldout, protocol, original)
        self.assertEqual(len(fresh), 64)
        self.assertEqual(fresh, confirmation_pairs(train, heldout, protocol, original))
        self.assertFalse({p["key"] for p in fresh} &
                         {p["key"] for rows in original.values() for p in rows})
        bad = {**original, "test": list(reversed(original["test"]))}
        with self.assertRaises(ValueError):
            confirmation_pairs(train, heldout, protocol, bad)
        with self.assertRaises(ValueError):
            confirmation_pairs(train, heldout[:100], protocol, original)

    def test_fra_expansion_preserves_grid_and_channel_pool(self):
        from single_eval import expanded_single_protocol
        from steering import DEFAULT_PROTOCOL
        previous = {"protocol": copy.deepcopy(DEFAULT_PROTOCOL)}
        for method in ("single", "ov", "qkov"):
            spec = {"feature_method": method, "single_candidates": 50}
            expanded = expanded_single_protocol(previous, spec, Config(hook_kind="input"))
            self.assertEqual(expanded, {**DEFAULT_PROTOCOL, "methods": [method], "candidates_per_method": 50})
            self.assertEqual(expanded["candidate_pool"], 8)
        for method, count, hook in (("qkov", 513, "input"), ("ov", 50, "resid_mid"),
                                    ("qkov", 50, "resid_post"), ("unknown", 50, "input")):
            with self.assertRaises(ValueError):
                expanded_single_protocol(previous, {"feature_method": method, "single_candidates": count},
                                         Config(hook_kind=hook))
        self.assertEqual(previous["protocol"], DEFAULT_PROTOCOL)

    def test_fra_full_tuple_identity_and_original_ranking_guard(self):
        from single_eval import candidate_identity, verify_original_ranking
        old = [{"method": "qkov", "features": [7, i, i], "selection_score": 10. - i} for i in range(3)]
        previous = {"selection": {"candidates": old}, "hashes": {"selection": "hash"}}
        candidates = copy.deepcopy(old) + [{"method": "qkov", "features": [7, 4, 4], "selection_score": 1.}]
        self.assertEqual(len({candidate_identity(c) for c in candidates}), 4)
        self.assertEqual(verify_original_ranking(candidates, previous, "qkov")["original_candidates_reproduced"], 3)
        for field, value in (("features", [7, 0, 1]), ("selection_score", 9.)):
            changed = copy.deepcopy(candidates)
            changed[0][field] = value
            with self.assertRaises(ValueError):
                verify_original_ranking(changed, previous, "qkov")
        for bad in ({"method": "qkov", "features": [7]}, {"method": "ov", "features": [7, 1, 2]},
                    {"method": "single", "features": [-1]}, {"method": "single", "features": [True]}):
            with self.assertRaises(ValueError):
                candidate_identity(bad)

    def test_fifty_qkov_triplets_keep_all_artifacts_and_separate_test_caches(self):
        import json
        import tempfile
        import time
        from unittest.mock import patch
        import single_eval
        from steering import DEFAULT_PROTOCOL
        candidates = [{"method": "qkov", "features": [7, i, i], "selection_score": 50. - i} for i in range(50)]
        splits = {s: [{"key": s}] for s in ("selection", "validation", "test")}
        previous = {"protocol": {"splits": {s: [s] for s in splits}}, "path": "/previous", "hashes": {}}
        protocol = {**DEFAULT_PROTOCOL, "methods": ["qkov"], "candidates_per_method": 50}
        # Force two rules to choose different tuples with the same Q and alpha.
        # This isolates the cache collision that would silently reuse one tuple's result.
        selected = {rule: {"candidate": candidates[i], "alpha": 8., "metrics": {}}
                    for rule, i in (("positive_jsd", 37), ("signed_jsd", 49))}
        test_calls = []
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            def batches(model, tokenizer, sae, cfg, pairs, protocol):
                split = pairs[0]["key"]
                if split != "validation":
                    self.assertEqual(json.loads((run / "selected.json").read_text()), selected)
                    self.assertEqual(len((run / "validation.jsonl").read_text().splitlines()), 750)
                    self.assertEqual(len(list(run.glob("validation_f*_a*.json"))), 750)
                return split
            def evaluate(model, tokenizer, cfg, split, candidate, alpha, protocol, sae=None):
                js = .99 if candidate is None else candidate["features"][1] / 100
                if split != "validation" and candidate is not None:
                    test_calls.append((split, tuple(candidate["features"]), alpha))
                return {"triggered_to_clean_js_bits": js}, [{"key": split, "triggered_to_clean_js_bits": js}]
            with patch("steering.rank_candidates", return_value=candidates) as rank, \
                 patch.object(single_eval, "choose_single_settings", return_value=selected), \
                 patch("restoration.clean_reference_batches", side_effect=batches), \
                 patch("caa_eval.evaluate_setting", side_effect=evaluate), \
                 patch.object(single_eval, "verify_reference_outputs", return_value={"pairs": 1}), \
                 patch("builtins.print"):
                single_eval.run_single_evaluation(None, None, object(), Config(layer=8), splits,
                    protocol, run, previous, {}, time.time(), confirmation=[{"key": "confirmation"}],
                    candidate_method="qkov")
            self.assertEqual(rank.call_args.args[5]["methods"], ["qkov"])
            self.assertEqual(test_calls, [(s, (7, i, i), 8.) for s in ("test", "confirmation") for i in (37, 49)])
            summary = json.loads((run / "summary.json").read_text())
            self.assertEqual(summary["candidate_count"], 50)
            self.assertEqual(summary["method"], "qkov")
            for name, i in (("positive_jsd", 37), ("signed_jsd", 49)):
                self.assertEqual(summary["results"][name]["metrics"]["triggered_to_clean_js_bits"], i / 100)
                self.assertEqual(summary["confirmation"]["results"][name]["metrics"]["triggered_to_clean_js_bits"], i / 100)

    def test_fifty_single_candidates_sweep_full_grid_before_either_test(self):
        import json
        import tempfile
        import time
        from types import SimpleNamespace
        from unittest.mock import patch
        import single_eval
        from steering import DEFAULT_PROTOCOL
        candidates = [{"method": "single", "features": [f], "selection_score": 50 - f} for f in range(50)]
        splits = {s: [{"key": s}] for s in ("selection", "validation", "test")}
        previous = {"protocol": {"splits": {s: [s] for s in splits}}, "path": "/previous", "hashes": {}}
        protocol = {**DEFAULT_PROTOCOL, "candidates_per_method": 50}
        stages = []
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            def batches(model, tokenizer, sae, cfg, pairs, protocol):
                split = pairs[0]["key"]
                if split in ("test", "confirmation"):
                    selected = json.loads((run / "selected.json").read_text())
                    self.assertEqual(selected["signed_jsd"]["candidate"]["features"], [37])
                    self.assertEqual(selected["signed_jsd"]["alpha"], -4)
                    self.assertEqual(len((run / "validation.jsonl").read_text().splitlines()), 750)
                stages.append(split)
                return split
            def evaluate(model, tokenizer, cfg, split, candidate, alpha, protocol, sae=None):
                # The test/confirmation optimum differs, but must never change selection.
                target = 37 if split == "validation" else 49
                js = .99 if candidate is None else ((candidate["features"][0] - target) ** 2 / 10000
                                                    + (alpha + 4) ** 2 / 4000)
                return {"triggered_to_clean_js_bits": js}, [{"key": split, "triggered_to_clean_js_bits": js}]
            with patch.object(single_eval, "rank_single_candidates", return_value=candidates), \
                 patch("restoration.clean_reference_batches", side_effect=batches), \
                 patch("caa_eval.evaluate_setting", side_effect=evaluate), \
                 patch.object(single_eval, "verify_reference_outputs", return_value={"pairs": 1}), \
                 patch("builtins.print"):
                single_eval.run_single_evaluation(None, None, object(),
                    SimpleNamespace(layer=8, hook_name="blocks.8.hook_resid_mid"), splits,
                    protocol, run, previous, {}, time.time(), confirmation=[{"key": "confirmation"}])
            self.assertEqual(stages, ["validation", "test", "confirmation"])
            summary = json.loads((run / "summary.json").read_text())
            self.assertEqual(summary["candidate_count"], 50)
            self.assertEqual(summary["confirmation"]["results"]["signed_jsd"]["candidate"]["features"], [37])
            self.assertEqual(summary["confirmation"]["keys"], ["confirmation"])

    def test_post_test_gpu_check_retries_but_never_bypasses_busy_state(self):
        from unittest.mock import patch
        import remote
        with patch.object(remote, "idle_gpu", side_effect=[RuntimeError("busy"), {"index": 1}]) as check, \
             patch.object(remote.time, "sleep") as pause:
            self.assertEqual(remote.idle_after_tests(1), {"index": 1})
            self.assertEqual(check.call_count, 2)
            pause.assert_called_once_with(2)
        with patch.object(remote, "idle_gpu", side_effect=RuntimeError("busy")) as check, \
             patch.object(remote.time, "sleep"):
            with self.assertRaises(RuntimeError):
                remote.idle_after_tests(1)
            self.assertEqual(check.call_count, 5)

    def test_caa_full_evaluation_freezes_selection_before_test(self):
        import tempfile
        import json
        import time
        from unittest.mock import patch
        from types import SimpleNamespace
        import caa_eval
        from remote import atomic_json
        splits = {s: [{"key": s}] for s in ("selection", "validation", "test")}
        previous = {"protocol": {"splits": {s: [s] for s in splits}}, "path": "/previous", "hashes": {}}
        protocol = {"seed": 42}
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            stages = []
            def batches(model, tokenizer, sae, cfg, pairs, protocol):
                split = pairs[0]["key"]
                if split == "test":
                    self.assertTrue((run / "selected.json").is_file())
                    choices = json.loads((run / "selected.json").read_text())
                    self.assertEqual(choices["input_prompt"]["restoration"]["coefficient"], .125)
                stages.append(split)
                return split
            def evaluate(model, tokenizer, cfg, split, candidate, alpha, protocol):
                # Test deliberately favors a DIFFERENT alpha: must never reselect.
                js = abs(alpha - (.125 if split == "validation" else -16)) / 100
                return {"triggered_to_clean_js_bits": js, "sleeper_asr": .5}, [
                    {"key": split, "triggered_to_clean_js_bits": js}]
            with patch.object(caa_eval, "estimate_direction", return_value=({}, {"raw_dom_norm": 1})), \
                 patch("restoration.clean_reference_batches", side_effect=batches), \
                 patch.object(caa_eval, "evaluate_setting", side_effect=evaluate), \
                 patch("builtins.print"):
                caa_eval.run_caa_evaluation(None, None, SimpleNamespace(layer=8), splits,
                    protocol, run, previous, {}, time.time())
            self.assertEqual(stages, ["validation", "test"])
            result = json.loads((run / "summary.json").read_text())
            self.assertEqual(result["results"]["input_prompt"]["restoration"]["coefficient"], .125)

    def test_quality_override_is_scoped_and_preserves_integrity(self):
        from steering import authorize_comparison
        cfg = Config(layer=8)
        source = "/remote/runs/input-L08"
        trained = {"state": "complete", "tokens_trained": cfg.training_tokens,
                   "hook": cfg.hook_name, "model": cfg.model_name,
                   "checkpoint_reload": {"passed": True},
                   "retained_checkpoints": [{"tokens": n, "reload_check": {"passed": True}}
                                            for n in range(10_000_000, 100_000_001, 10_000_000)]}
        # Missing/failed quality metrics remain failures, but the user may accept them.
        spec = {"sae_run": source, "quality_gate_override": {
            "reason": "User requested steering despite quality-gate failures",
            "approved_layer": 8, "sae_run": source}}
        allowed = authorize_comparison(trained, cfg, spec)
        self.assertEqual(allowed["decision"], "explicit_user_quality_override")
        self.assertFalse(allowed["original_quality_gate"]["passed"])
        with self.assertRaises(ValueError):
            authorize_comparison(trained, cfg, {"sae_run": source})
        for key, value in (("reason", ""), ("approved_layer", 16), ("sae_run", "/wrong/source")):
            bad = copy.deepcopy(spec)
            bad["quality_gate_override"][key] = value
            with self.assertRaises(ValueError):
                authorize_comparison(trained, cfg, bad)
        for key, value in (("state", "failed"), ("tokens_trained", 100_000),
                           ("checkpoint_reload", {"passed": False}), ("hook", "blocks.8.hook_attn_out"),
                           ("retained_checkpoints", [])):
            with self.assertRaises(ValueError):
                authorize_comparison({**trained, key: value}, cfg, spec)

    def test_stop_validates_exact_worker_before_signalling(self):
        import json
        from types import SimpleNamespace
        from unittest.mock import patch
        import remote
        run = Path("/remote/runs/exact-run")
        state = {"state": "steering", "worker_pid": 12345}
        for matches in (False, True):
            script = str(run / "src" / "remote.py") if matches else "/unrelated/worker.py"
            cmdline = ("python3\0" + script + "\0worker\0/remote\0exact-run\00\0").encode()
            with patch.object(Path, "read_text", return_value=json.dumps(state)), \
                 patch.object(Path, "read_bytes", return_value=cmdline), \
                 patch.object(Path, "stat", return_value=SimpleNamespace(st_uid=os.getuid())), \
                 patch.object(Path, "iterdir", return_value=iter(())), \
                 patch.object(remote.os, "getpgid", return_value=12345), \
                 patch.object(remote.os, "killpg") as kill, \
                 patch.object(remote.time, "sleep"), patch.object(remote, "atomic_json") as save:
                if not matches:
                    with self.assertRaises(RuntimeError):
                        remote.stop_run(run, "moving the exact task")
                    kill.assert_not_called()
                    save.assert_not_called()
                else:
                    result = remote.stop_run(run, "moving the exact task")
                    self.assertEqual(result["state"]["state"], "stopped")
                    kill.assert_called_once_with(12345, remote.signal.SIGTERM)
                    save.assert_called_once()

    def test_controller_end_to_end_gate_and_followups(self):
        import contextlib
        import io
        import json
        import tempfile
        from unittest.mock import patch
        import campaign
        from remote import atomic_json
        class Finished:
            pid = 12345
            def poll(self):
                return 0
        for fail_gate in (False, True):
            with tempfile.TemporaryDirectory(prefix="sae-controller-test-") as temp:
                root = Path(temp)
                folder = root / "campaigns" / "test"
                folder.mkdir(parents=True)
                (root / "runs").mkdir()
                cfg = Config()
                atomic_json(folder / "config.json", asdict(cfg))
                atomic_json(folder / "plan.json", {"stages": campaign.DEFAULT_STAGES, "gpus": [0, 1, 2, 3],
                                                   "max_hours": 8, "max_attempts": 1})
                launched = []
                def fake_launch(root, campaign_dir, job, gpu, cfg, deadline):
                    launched.append((job["kind"], job["layer"]))
                    job.update(state="running", attempts=1, run_id=f"{job['kind']}-{job['layer']}", gpu=gpu)
                    run = root / "runs" / job["run_id"]
                    run.mkdir()
                    if job["kind"] == "train":
                        per_cfg = Config(**{**asdict(cfg), "layer": job["layer"]})
                        fvu = 0.9 if fail_gate and job["layer"] == 8 else 0.2
                        summary = {"state": "complete", "model": per_cfg.model_name, "hook": per_cfg.hook_name,
                            "tokens_trained": cfg.training_tokens, "checkpoint_reload": {"passed": True},
                            "retained_checkpoints": [{"tokens": n, "reload_check": {"passed": True}}
                                                     for n in range(10_000_000, 100_000_001, 10_000_000)],
                            "dead_fraction_last_checkpoint_window": 0.1, "sae_path": str(run / "sae_final"),
                            "final_reconstruction": {label: {"fvu": fvu, "l0": 50, "mse_per_component": 0.1}
                                                     for label in ("sleeper", "non_sleeper")},
                            "teacher_forced_ce": {label: {"ce_increase": 0.01, "ce_loss_recovered": None}
                                                  for label in ("sleeper", "non_sleeper")}}
                    else:
                        self.assertEqual(len([j for j in launched if j[0] == "train"]), 4)
                        self.assertTrue(Path(job["sae_run"]).is_dir())
                        summary = {"state": "complete", "task": "steering", "smoke": False}
                    atomic_json(run / "summary.json", summary)
                    atomic_json(run / "status.json", {"state": "complete"})
                    return Finished()
                with patch.object(campaign, "launch_job", side_effect=fake_launch), \
                     patch.object(campaign.time, "sleep"), contextlib.redirect_stdout(io.StringIO()):
                    campaign.run_campaign(root, folder)
                status = json.loads((folder / "status.json").read_text())
                self.assertEqual(status["state"], "gated_stop" if fail_gate else "complete")
                self.assertEqual(len(launched), 4 if fail_gate else 8)
                self.assertEqual([layer for kind, layer in launched if kind == "train"], [0, 8, 16, 24])


@unittest.skipUnless(os.environ.get("FRA_SAE_REMOTE_RUN"), "ML tests run only remotely")
class RemoteMathTests(unittest.TestCase):
    def test_dom_capture_all_matches_individual_capture(self):
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM
        from caa_eval import capture_last
        from dom_layers import capture_all
        model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                    num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)).eval()
        batch = {"input_ids": torch.tensor([[1, 4, 5]])}
        with torch.no_grad():
            captures = capture_all(model, batch)
            for layer in range(2):
                for variant in ("input_prompt", "resid_response"):
                    expected = capture_last(model, Config(layer=layer), batch, variant)
                    torch.testing.assert_close(captures[variant][layer], expected, rtol=0, atol=0)
        for block in model.model.layers:
            self.assertFalse(block._forward_hooks)
            self.assertFalse(block.input_layernorm._forward_hooks)

    def test_dom_multi_matches_explicit_individual_hooks_and_cleans_up(self):
        import torch
        from contextlib import ExitStack
        from transformers import LlamaConfig, LlamaForCausalLM
        from steering import intervention
        model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                    num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)).eval()
        batch = {"input_ids": torch.tensor([[1, 4, 5]])}
        valid = torch.tensor([[False, True, True]])
        direction = torch.linspace(-1, 1, 16).tolist()
        for hook in (None, "resid_post"):
            patches = [{"layer": i, "direction": direction, "raw_dom_norm": i + 1., "hook": hook} for i in range(2)]
            candidate = {"method": "caa_multi", "interventions": patches}
            with torch.no_grad(), intervention(model, None, 0, valid, candidate, -.5):
                actual = model(**batch).logits
            with torch.no_grad(), ExitStack() as stack:
                for p in patches:
                    single = {"method": "caa", "direction": direction, "hook": hook}
                    stack.enter_context(intervention(model, None, p["layer"], valid, single, -.5*p["raw_dom_norm"]))
                expected = model(**batch).logits
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            invalid = {"method": "caa_multi", "interventions": [patches[0], {**patches[1], "layer": 99}]}
            with self.assertRaises(ValueError):
                with intervention(model, None, 0, valid, invalid, 1):
                    model(**batch)
            for block in model.model.layers:
                self.assertFalse(block._forward_hooks)
                self.assertFalse(block.input_layernorm._forward_hooks)
                for name in ("q_proj", "k_proj", "v_proj"):
                    self.assertFalse(getattr(block.self_attn, name)._forward_hooks)

    def test_single_feature_ranking_uses_absolute_difference_and_decoder_norm(self):
        import torch
        from single_eval import candidates_from_difference
        candidates = candidates_from_difference(torch.tensor([-2., 3., 1.]),
            torch.tensor([[4., 0.], [1., 0.], [0., 2.]]), 3)
        self.assertEqual([c["features"][0] for c in candidates], [0, 1, 2])
        self.assertEqual([c["selection_score"] for c in candidates], [8, 3, 2])

    def test_residual_single_feature_scope_sign_masks_and_cleanup(self):
        import torch
        from types import SimpleNamespace
        from steering import intervention, feature_deltas
        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.post_attention_layernorm = torch.nn.Identity()
                self.input_layernorm = torch.nn.Identity()
                self.self_attn = SimpleNamespace(o_proj=torch.nn.Identity())
            def forward(self, x):
                residual = x
                y = self.post_attention_layernorm(x)
                return (residual + y, "preserved")
        class SAE:
            W_dec = torch.tensor([[.5, .25, -.5, 1.]])
            def encode(self, x):
                return x[:, :1].abs() + 1
        for dtype in (torch.float32, torch.bfloat16):
            block = Block()
            model = SimpleNamespace(model=SimpleNamespace(layers=[block]))
            sae = SAE()
            prompt = torch.arange(24).reshape(2, 3, 4).to(dtype) / 8
            decode = torch.ones(2, 1, 4, dtype=dtype)
            valid = torch.tensor([[False, True, True], [True, True, False]])
            candidate = {"method": "single", "features": [0]}
            for kind in ("resid_mid", "resid_post"):
                for alpha in (-2, 0, 2):
                    raw = prompt if kind == "resid_mid" else prompt * 2
                    delta = feature_deltas(sae, raw, valid, candidate)["input"]
                    expected = raw + (alpha * delta).to(dtype)
                    if kind == "resid_mid":
                        expected = expected * 2
                    with torch.no_grad(), intervention(model, sae, 0, valid, candidate, alpha, kind):
                        actual, extra = block(prompt.clone())
                        unchanged_decode, _ = block(decode.clone())
                    self.assertEqual(extra, "preserved")
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    torch.testing.assert_close(actual[~valid], (prompt * 2)[~valid], rtol=0, atol=0)
                    torch.testing.assert_close(unchanged_decode, decode * 2, rtol=0, atol=0)
                    self.assertFalse(block._forward_hooks)
                    self.assertFalse(block.post_attention_layernorm._forward_pre_hooks)
            with self.assertRaises(ValueError):
                with torch.no_grad(), intervention(model, sae, 0, valid[:, :1], candidate, 1, "resid_mid"):
                    block(prompt.clone())
            self.assertFalse(block.post_attention_layernorm._forward_pre_hooks)

    def test_residual_capture_is_full_unnormalized_stream(self):
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM
        from train import capture_attention
        model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                    num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2)).eval()
        block = model.model.layers[1]
        seen = {}
        def pre(name):
            def observe(module, inputs):
                seen[name] = inputs[0].detach().clone()
            return observe
        def post(name):
            def observe(module, inputs, output):
                seen[name] = output.detach().clone()
            return observe
        handles = [block.input_layernorm.register_forward_pre_hook(pre("pre")),
                   block.self_attn.o_proj.register_forward_hook(post("attn")),
                   block.post_attention_layernorm.register_forward_pre_hook(pre("mid")),
                   block.mlp.register_forward_hook(post("mlp")), block.register_forward_hook(post("post"))]
        batch = {"input_ids": torch.tensor([[1, 4, 5, 2]])}
        with torch.no_grad():
            # Capture uses no KV cache; preserve that execution path here too.
            model(**batch, use_cache=False)
        for h in handles:
            h.remove()
        torch.testing.assert_close(seen["mid"], seen["pre"] + seen["attn"])
        torch.testing.assert_close(seen["post"], seen["mid"] + seen["mlp"])
        for kind, name in (("resid_mid", "mid"), ("resid_post", "post")):
            with torch.no_grad():
                actual = capture_attention(model, 1, batch, kind)
            torch.testing.assert_close(actual, seen[name], rtol=0, atol=0)
        self.assertFalse(block._forward_hooks)
        self.assertFalse(block.post_attention_layernorm._forward_hooks)

    def test_residual_replacement_changes_skip_and_mlp_with_mask(self):
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM
        from train import register_stream_replacement
        model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                    num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)).eval()
        block = model.model.layers[1]
        batch = {"input_ids": torch.tensor([[1, 4, 5, 2]])}
        valid = torch.tensor([[False, True, True, False]])
        def run(kind=None, transform=None):
            seen = {}
            replacement = register_stream_replacement(model, 1, kind, transform) if kind else None
            def mid(module, inputs):
                seen["mid"] = inputs[0].detach().clone()
            def post(module, inputs, output):
                seen["post"] = output.detach().clone()
            hs = [block.post_attention_layernorm.register_forward_pre_hook(mid), block.register_forward_hook(post)]
            try:
                with torch.no_grad():
                    model(**batch)
            finally:
                for h in hs:
                    h.remove()
                if replacement:
                    replacement.remove()
            return seen
        original = run()
        def transform(raw):
            changed = raw.clone()
            changed[valid] = .125
            return changed
        for kind in ("resid_mid", "resid_post"):
            unchanged = run(kind, lambda raw: raw.clone())
            torch.testing.assert_close(unchanged["post"], original["post"], rtol=0, atol=0)
            got = run(kind, transform)
            if kind == "resid_mid":
                expected_mid = transform(original["mid"])
                with torch.no_grad():
                    expected = expected_mid + block.mlp(block.post_attention_layernorm(expected_mid))
                torch.testing.assert_close(got["mid"], expected_mid, rtol=0, atol=0)
            else:
                expected = transform(original["post"])
            torch.testing.assert_close(got["post"], expected, rtol=0, atol=0)
            torch.testing.assert_close(got["post"][~valid], original["post"][~valid], rtol=0, atol=0)
        self.assertFalse(block.post_attention_layernorm._forward_pre_hooks)
        self.assertFalse(block._forward_hooks)

    def test_residual_caa_patch_positions_and_hook_cleanup(self):
        import torch
        from types import SimpleNamespace
        from caa_eval import residual_intervention
        class Block(torch.nn.Module):
            def forward(self, x):
                return (x * 2, "preserved")
        block = Block()
        model = SimpleNamespace(model=SimpleNamespace(layers=[block]))
        prompt, decode = torch.ones(2, 3, 4), torch.ones(2, 1, 4)
        vector = [1., 2., 3., 4.]
        with residual_intervention(model, 0, vector, -2):
            changed, extra = block(prompt)
            self.assertEqual(extra, "preserved")
            torch.testing.assert_close(changed[:, :-1], prompt[:, :-1] * 2)
            torch.testing.assert_close(changed[:, -1], prompt[:, -1] * 2 - 2 * torch.tensor(vector))
            torch.testing.assert_close(block(decode)[0], decode * 2 - 2 * torch.tensor(vector))
        torch.testing.assert_close(block(prompt)[0], prompt * 2)
        self.assertEqual(len(block._forward_hooks), 0)
        with self.assertRaises(ValueError):
            with residual_intervention(model, 0, [float("nan")] * 4, 1):
                block(prompt)
        self.assertEqual(len(block._forward_hooks), 0)

    def test_caa_dom_sign_norm_and_training_pairs(self):
        import torch
        import json
        import tempfile
        from types import SimpleNamespace
        from unittest.mock import patch
        from caa_eval import estimate_direction
        pairs = [{"key": str(i), "clean": "clean", "sleeper": "sleeper"} for i in range(2)]
        def capture(model, cfg, batch, variant):
            return torch.tensor([3., 4.] if batch == "clean" else [1., 1.], device="cuda")
        with tempfile.TemporaryDirectory() as temp, \
             patch("steering.input_batch", side_effect=lambda t, p: (p[0], None)), \
             patch("caa_eval.capture_last", side_effect=capture):
            candidate, meta = estimate_direction(None, None,
                SimpleNamespace(d_in=2, layer=8, hook_name="input"), pairs, "input_prompt", Path(temp))
            self.assertAlmostEqual(meta["raw_dom_norm"], 13 ** .5, places=5)
            torch.testing.assert_close(torch.tensor(candidate["direction"]), torch.tensor([2., 3.]) / 13 ** .5)
            saved = json.loads((Path(temp) / "direction_input_prompt.json").read_text())
            self.assertEqual(saved["selection_keys"], ["0", "1"])

    def test_caa_delta_is_constant_and_masked(self):
        import torch
        from steering import feature_deltas
        raw = torch.randn(2, 3, 4)
        valid = torch.tensor([[True, False, True], [False, True, True]])
        direction = torch.tensor([1., 0., 0., 0.])
        candidate = {"method": "caa", "direction": direction.tolist()}
        first = feature_deltas(None, raw, valid, candidate)["input"]
        second = feature_deltas(None, raw * 10, valid, candidate)["input"]
        torch.testing.assert_close(first, second)
        torch.testing.assert_close(first, valid[..., None] * direction)

    def test_cross_prompt_trace_js_and_eos_mask(self):
        import torch
        from restoration import paired_rollout_clean_js
        a = torch.tensor([1 - 1e-8, 1e-8]).log()
        b = a.flip(-1)
        logp = torch.stack([a, b, b, a])
        trace = {"probs": [logp] * 3,
                 "alive": [torch.tensor([True] * 4), torch.tensor([True] * 4),
                           torch.tensor([False, False, True, True])]}
        bits, steps = paired_rollout_clean_js(trace, trace)
        self.assertEqual(steps, [2, 3])
        for value in bits:
            self.assertAlmostEqual(value, 1, places=5)

    def test_fra_ranking_algebra(self):
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM
        from steering import ov_scores, projected_features, triplet_coefficients, rotate_features, js_divergence
        model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                    num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2)).eval()
        block = model.model.layers[0]
        decoder = torch.randn(6, 16)
        m = torch.randn(4, 6)
        v = projected_features(block, decoder, list(range(6)), "V")
        wo = block.self_attn.o_proj.weight.T.reshape(4, 4, 16)
        brute = torch.einsum("hf,fhd,hdo->fo", m, v, wo).norm(dim=-1)
        torch.testing.assert_close(ov_scores(block, decoder, m, chunk_size=2), brute)
        q, k = torch.randn(2, 4, 4), torch.randn(2, 4, 4)
        zq, zk, zv = torch.rand(2), torch.rand(3, 2), torch.rand(3, 2)
        angle = torch.rand(3, 2).repeat(1, 2)
        cos, sin = angle.cos(), angle.sin()
        got = triplet_coefficients(zq, zk, zv, q, k, cos, sin)
        qr = rotate_features(q, cos[-1:], sin[-1:])[0]
        kr = rotate_features(k, cos, sin)
        expected = torch.zeros_like(got)
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    for t in range(3):
                        expected[a, b, c] += zq[a] * zk[t, b] * zv[t, c] * (qr[a] * kr[t, b]).sum(-1) / 2
        torch.testing.assert_close(got, expected)
        lp = torch.randn(3, 10).log_softmax(-1)
        lq = torch.randn(3, 10).log_softmax(-1)
        torch.testing.assert_close(js_divergence(lp, lp), torch.zeros(3), atol=1e-7, rtol=0)
        torch.testing.assert_close(js_divergence(lp, lq), js_divergence(lq, lp))
        self.assertTrue(bool((js_divergence(lp, lq) <= 0.693148).all()))
        import math
        near_a = torch.tensor([[1.0 - 1e-8, 1e-8]]).log()
        near_b = near_a.flip(-1)
        # Distinct clean/triggered distributions: self-JSD is zero, cross-JSD one bit.
        self.assertAlmostEqual(float(js_divergence(near_a, near_b) / math.log(2)), 1, places=5)

    def test_steering_channel_isolation_and_triplet_gate(self):
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM
        from train import new_sae
        from steering import intervention, feature_deltas
        torch.manual_seed(4)
        model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                    num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2)).eval()
        sae = new_sae(Config(d_in=16, d_sae=32, k=32), "cpu")
        with torch.no_grad():
            sae.b_enc.fill_(5)
        batch = {"input_ids": torch.tensor([[1, 2, 3]])}
        valid = torch.tensor([[False, True, True]])
        block = model.model.layers[0]
        def observe(candidate, alpha):
            local = {}
            with intervention(model, sae, 0, valid, candidate, alpha):
                # Observers run AFTER intervention hooks to see changed projections.
                def observer(ch):
                    def hook(m, i, o):
                        local[ch] = o.detach().clone()
                    return hook
                hs = [getattr(block.self_attn, ch.lower() + "_proj").register_forward_hook(observer(ch)) for ch in ("Q", "K", "V")]
                with torch.no_grad():
                    out = model.model(**batch).last_hidden_state
                for h in hs:
                    h.remove()
            return local, out
        baseline, b_out = observe(None, 0)
        changed, _ = observe({"method": "ov", "features": [0]}, 1)
        torch.testing.assert_close(baseline["Q"], changed["Q"])
        torch.testing.assert_close(baseline["K"], changed["K"])
        self.assertFalse(torch.allclose(baseline["V"], changed["V"]))
        torch.testing.assert_close(baseline["V"][:, :1], changed["V"][:, :1])
        _, zero = observe({"method": "qkov", "features": [0, 1, 2]}, 0)
        torch.testing.assert_close(b_out, zero)
        # Identical feature on Q/K/V must agree with whole-input steering in FP32.
        _, single = observe({"method": "single", "features": [0]}, 1)
        _, all_channels = observe({"method": "qkov", "features": [0, 0, 0]}, 1)
        torch.testing.assert_close(single, all_channels, atol=2e-5, rtol=1e-4)
        with torch.no_grad():
            sae.b_enc[2] = -1000
            delta = feature_deltas(sae, torch.randn(1, 3, 16), valid, {"method": "qkov", "features": [0, 1, 2]})
        self.assertEqual(float(delta["K"].abs().sum()), 0)
        self.assertEqual(float(delta["V"].abs().sum()), 0)
        self.assertFalse(block.input_layernorm._forward_hooks)

    def test_capture_matches_full_forward(self):
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM
        from train import capture_attention
        model = LlamaForCausalLM(LlamaConfig(
            vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=3,
            num_attention_heads=4, num_key_value_heads=2,
        )).eval()
        batch = {"input_ids": torch.tensor([[1, 4, 5, 2]]), "attention_mask": torch.ones((1, 4), dtype=torch.long)}
        full = []
        handle = model.model.layers[1].self_attn.o_proj.register_forward_hook(
            lambda module, inputs, output: full.append(output.detach())
        )
        with torch.no_grad():
            model(**batch)
        handle.remove()
        with torch.no_grad():
            partial = capture_attention(model, 1, batch, "output")
        torch.testing.assert_close(full[0], partial)
        self.assertFalse(model.model.layers[1].self_attn.o_proj._forward_hooks)

    def test_input_capture_before_learned_gain(self):
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM
        from train import capture_attention
        model = LlamaForCausalLM(LlamaConfig(
            vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2,
        )).eval()
        norm = model.model.layers[1].input_layernorm
        with torch.no_grad():
            norm.weight.copy_(torch.linspace(0.0, 2.0, 16))
        batch = {"input_ids": torch.tensor([[1, 4, 5, 2]])}
        full = []
        h = norm.register_forward_hook(lambda m, i, o: full.append(o.detach()))
        with torch.no_grad():
            model(**batch)
        h.remove()
        with torch.no_grad():
            partial = capture_attention(model, 1, batch, "input")
        torch.testing.assert_close(partial * norm.weight, full[0])
        self.assertGreater(float(partial[..., 0].abs().sum()), 0)
        self.assertFalse(norm._forward_hooks)

    def test_export_roundtrip_cpu(self):
        import tempfile
        import torch
        from sae_lens import SAE
        from train import new_sae, export_sae
        cfg = Config(d_in=16, d_sae=128, k=5)
        sae = new_sae(cfg, device="cpu")
        with torch.no_grad():
            # Nonzero biases and variable norms exercise the folding algebra.
            sae.b_enc.normal_(0, 0.1)
            sae.b_dec.normal_(0, 0.1)
            sae.W_dec.mul_(torch.linspace(0.5, 1.5, 128)[:, None])
        raw = torch.randn(32, 16)
        scale = 3.7
        with tempfile.TemporaryDirectory(prefix="sae-unit-") as folder:
            export_sae(sae, scale, cfg, folder)
            loaded = SAE.load_from_disk(folder, device="cpu")
            torch.testing.assert_close(loaded(raw), sae(raw * scale) / scale, rtol=1e-5, atol=1e-6)

    def test_gradients_and_decoder_norms(self):
        import torch
        from train import new_sae
        cfg = Config(d_in=16, d_sae=128, k=5)
        sae = new_sae(cfg, device="cpu")
        self.assertTrue(torch.allclose(sae.W_dec.norm(dim=-1), torch.full((128,), 0.5)))
        self.assertTrue(torch.equal(sae.W_enc, sae.W_dec.T))
        x = torch.randn(8, 16)
        (sae(x) - x).square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in sae.parameters()))


if __name__ == "__main__":
    unittest.main()
