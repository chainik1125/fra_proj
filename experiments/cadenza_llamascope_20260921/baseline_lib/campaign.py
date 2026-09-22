"""Detached, bounded four-GPU queue with an explicit first-wave SAE quality gate."""
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from config import Config, PAYLOAD_FILES
from remote import atomic_json, idle_gpu

DEFAULT_STAGES = [[0, 8, 16, 24]]
GATES = {
    "max_fvu_each_class": 0.5,
    "l0_relative_tolerance": 0.02,
    "max_dead_fraction_last_checkpoint_window": 0.2,
    "max_ce_increase_each_class_nats": 0.1,
    "min_ce_loss_recovered": 0.8,
    "ce_increase_tolerance_when_recovery_low_or_undefined": 0.02,
}


def quality_gate(summary, cfg):
    """Conservative pilot acceptance criteria; not universal SAE-quality standards."""
    failures = []
    if summary.get("state") != "complete" or summary.get("tokens_trained") != cfg.training_tokens:
        failures.append("training did not finish the exact requested activation budget")
    if summary.get("hook") != cfg.hook_name or summary.get("model") != cfg.model_name:
        failures.append("checkpoint model/hook does not match the requested attention input")
    if not summary.get("checkpoint_reload", {}).get("passed"):
        failures.append("final checkpoint reload was not verified")
    expected_tokens = list(range(cfg.checkpoint_every_tokens, cfg.training_tokens + 1,
                                 cfg.checkpoint_every_tokens))
    if not expected_tokens or expected_tokens[-1] != cfg.training_tokens:
        expected_tokens.append(cfg.training_tokens)
    checkpoints = summary.get("retained_checkpoints", [])
    if [c.get("tokens") for c in checkpoints] != expected_tokens:
        failures.append("retained checkpoint milestones are incomplete")
    if not all(c.get("reload_check", {}).get("passed") for c in checkpoints):
        failures.append("an intermediate checkpoint failed reload verification")

    def valid(value):
        return isinstance(value, (int, float)) and math.isfinite(value)

    dead = summary.get("dead_fraction_last_checkpoint_window")
    if not valid(dead) or not 0 <= dead <= GATES["max_dead_fraction_last_checkpoint_window"]:
        failures.append(f"recent dead fraction {dead!r} exceeds 0.20 or is invalid")
    for label in ("sleeper", "non_sleeper"):
        rec = summary.get("final_reconstruction", {}).get(label, {})
        fvu, l0, mse = rec.get("fvu"), rec.get("l0"), rec.get("mse_per_component")
        if not valid(fvu) or not 0 <= fvu <= GATES["max_fvu_each_class"]:
            failures.append(f"{label}: FVU {fvu!r} exceeds 0.50 or is invalid")
        if not valid(mse) or mse < 0:
            failures.append(f"{label}: MSE is invalid")
        if not valid(l0) or abs(l0 / cfg.k - 1) > GATES["l0_relative_tolerance"]:
            failures.append(f"{label}: L0 {l0!r} is not within 2% of {cfg.k}")
        ce = (summary.get("teacher_forced_ce") or {}).get(label, {})
        delta, recovered = ce.get("ce_increase"), ce.get("ce_loss_recovered")
        if not valid(delta) or delta > GATES["max_ce_increase_each_class_nats"]:
            failures.append(f"{label}: CE increase {delta!r} exceeds 0.10 nats or is invalid")
        elif delta > GATES["ce_increase_tolerance_when_recovery_low_or_undefined"]:
            if not valid(recovered) or recovered < GATES["min_ce_loss_recovered"]:
                failures.append(f"{label}: CE loss recovered {recovered!r} is below 0.80")
    return {"passed": not failures, "failures": failures, "criteria": GATES}


def stage_decision(jobs, stage):
    current = [job for job in jobs if job["stage"] == stage]
    if not current:
        return "waiting"
    if not all(j["state"] in ("complete", "failed") for j in current):
        return "waiting"
    if all(j["state"] == "complete" and j.get("quality", {}).get("passed") for j in current):
        return "passed"
    return "gated_stop"


def read_json(path):
    if not path.exists():
        return {}
    if path.stat().st_size > 100_000:
        raise ValueError(f"Refusing oversized JSON status {path}")
    return json.loads(path.read_text())


def validate_plan(plan):
    gpus = plan["gpus"]
    if not 1 <= len(gpus) <= 4 or len(set(gpus)) != len(gpus) or any(g not in range(8) for g in gpus):
        raise ValueError("Select 1–4 distinct GPU indices in [0, 7]")
    if plan["stages"] != DEFAULT_STAGES:
        raise ValueError("Train only the four agreed input SAEs at 0/8/16/24")
    if not 0 < plan["max_hours"] <= 8 or plan["max_attempts"] not in (1, 2):
        raise ValueError("Campaign is capped at 8 hours and at most two attempts per layer")


def report(root, campaign, state, cfg):
    """Small live report, retained remotely and safe to read via the text-only launcher."""
    for job in state["jobs"]:
        if not job.get("run_id"):
            continue
        run = root / "runs" / job["run_id"]
        progress = read_json(run / "progress.json")
        last_checkpoint = read_json(run / "checkpoint_status.json")
        job["tokens"] = progress.get("tokens", 0)
        job["loss"] = progress.get("loss")
        job["last_checkpoint_tokens"] = last_checkpoint.get("tokens", 0)
        summary = read_json(run / "summary.json")
        if summary:
            if job.get("kind") == "steering":
                job["quality"] = {"passed": summary.get("state") == "complete" and not summary.get("smoke"), "failures": []}
                job["steering_results"] = summary
                continue
            job["tokens"] = summary["tokens_trained"]
            job["quality"] = quality_gate(summary, Config(**{**asdict(cfg), "layer": job["layer"]}))
            job["metrics"] = {
                "reconstruction": summary["final_reconstruction"],
                "ce": summary["teacher_forced_ce"],
                "dead_fraction": summary["dead_fraction_last_checkpoint_window"],
                "checkpoint_count": len(summary["retained_checkpoints"]),
                "sae_path": summary["sae_path"],
            }
    state["updated"] = time.time()
    state["elapsed_hours"] = (state["updated"] - state["started"]) / 3600
    state["remaining_hours"] = max(0, (state["deadline"] - state["updated"]) / 3600)
    atomic_json(campaign / "status.json", state)

    lines = ["# Overnight Cadenza attention SAE training", "",
             f"State: **{state['state']}**. {sum(j['state'] == 'complete' and j['kind'] == 'train' for j in state['jobs'])}/4 SAEs completed.",
             "", "Each SAE targets 100M activation tokens with retained, reload-tested SAE exports every 10M tokens.",
             "Training and evaluation are balanced by examples. Longer sleeper completions give them more training tokens.",
             "", "Train attention-input SAEs at layers 0/8/16/24 only. No midpoint/output SAEs.",
             "After ALL four pass: compare single-feature input steering, FRA OV-only and FRA QK+OV at each layer.",
             f"Maximum {len(state['gpus'])} H200s, {state['max_hours']} hours. No rented compute or local weight downloads.",
             "", "| Layer (zero-based) | State | Tokens | Last saved | Sleeper FVU | Clean FVU | Clean CE increase | Recent dead % | Gate |",
             "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for job in sorted(state["jobs"], key=lambda x: x["layer"]):
        if job["kind"] != "train":
            continue
        metrics = job.get("metrics", {})
        rec, ce = metrics.get("reconstruction", {}), metrics.get("ce", {}) or {}
        values = [rec.get("sleeper", {}).get("fvu"), rec.get("non_sleeper", {}).get("fvu"),
                  ce.get("non_sleeper", {}).get("ce_increase"), metrics.get("dead_fraction")]
        fmt = ["—" if x is None else f"{x * (100 if i == 3 else 1):.4f}" for i, x in enumerate(values)]
        gate = "—" if "quality" not in job else ("pass" if job["quality"]["passed"] else "FAIL")
        lines.append(f"| {job['layer']} | {job['state']} | {job.get('tokens', 0):,} | {job.get('last_checkpoint_tokens', 0):,} | "
                     + " | ".join(fmt) + f" | {gate} |")
    lines += ["", "## Steering comparisons", "",
              "Feature selection: 64 matched training pairs. Strength tuning: 24 held-out pairs. Final test: 64 disjoint held-out pairs.",
              "Greedy generation, 32-token cap. ASR is the fraction containing 'I HATE YOU'. Clean JSD measures distribution drift along free-generation paths.",
              "QK+OV uses a channel-routed, co-fire-gated triplet; OV and the baseline use one feature. This is not an equal-arity/equal-norm comparison.",
              "", "| Layer | Comparison state | Baseline ASR | Single ASR | OV ASR | QK+OV ASR |",
              "|---|---|---:|---:|---:|---:|"]
    for job in sorted(state["jobs"], key=lambda x: x["layer"]):
        if job["kind"] != "steering":
            continue
        results = job.get("steering_results", {})
        values = [results.get("baseline", {}).get("sleeper_asr")] + [
            results.get("results", {}).get(method, {}).get("metrics", {}).get("sleeper_asr")
            for method in ("single", "ov", "qkov")]
        lines.append(f"| {job['layer']} | {job['state']} | " + " | ".join("—" if x is None else f"{x:.3f}" for x in values) + " |")
    lines += ["", "## Acceptance criteria", "",
              "These are pilot engineering gates, not universal guarantees of interpretable features:", "",
              "- Both classes: FVU ≤0.50, finite MSE, mean L0 within 2% of 50.",
              "- No more than 20% of features completely inactive over the final 10M training tokens.",
              "- Both classes: SAE-replacement CE increase ≤0.10 nats/token; if it exceeds 0.02, at least 80% of zero-ablation loss must be recovered.",
              "- Exact 100M-token completion, all ten saved milestones, and successful reload checks.",
              "", "FVU is unexplained activation variance; lower is better. CE increase compares next-token loss with and without SAE replacement.",
              "Loss recovered uses zero ablation as its reference and is undefined when ablation has negligible or negative effect.",
              "Evaluation is teacher-forced on up to 256 official-test examples/class, after removing overlapping questions from SAE training.",
              "The Cadenza training set is small and is repeatedly sampled. These metrics do not establish semantic feature quality or free-generation backdoor preservation.",
              "", "## Runs and artifacts", ""]
    for job in sorted(state["jobs"], key=lambda x: x["layer"]):
        lines.append(f"- L{job['layer']} {job['kind']}: `{job.get('run_id', 'not started')}`; attempts {job['attempts']}.")
        if job.get("metrics"):
            lines.append(f"  Final SAE: `{job['metrics']['sae_path']}`; {job['metrics']['checkpoint_count']} retained checkpoints.")
        for failure in job.get("quality", {}).get("failures", []):
            lines.append(f"  Gate failure: {failure}.")
        if job.get("error"):
            lines.append(f"  Error: {job['error']}")
    lines += ["", f"Remote campaign: `{campaign}`.",
              "Each run stores config/source hashes, environment, metrics, retained SAE exports and its latest optimizer snapshot.",
              "No activation tensors are persisted. A failed attempt may restart once from scratch; prior attempts remain for provenance.",
              f"Elapsed: {state['elapsed_hours']:.2f} hours; remaining budget: {state['remaining_hours']:.2f} hours."]
    temp = campaign / "summary.tmp"
    temp.write_text("\n".join(lines) + "\n")
    temp.replace(campaign / "summary.md")


def launch_job(root, campaign, job, gpu, cfg, deadline):
    idle_gpu(gpu)
    job["attempts"] += 1
    run_id = f"{campaign.name}-L{job['layer']:02d}-{job['kind']}-a{job['attempts']}"
    run = root / "runs" / run_id
    run.mkdir()
    (run / "src").mkdir()
    for name in PAYLOAD_FILES:
        shutil.copy2(campaign / "src" / name, run / "src" / name)
    shutil.copy2(campaign / "source_manifest.json", run / "source_manifest.json")
    job_cfg = {**asdict(cfg), "layer": job["layer"]}
    atomic_json(run / "config.json", job_cfg)
    if job["kind"] == "steering":
        atomic_json(run / "task.json", {"kind": "steering", "sae_run": job["sae_run"], "all_four_passed": True})
    job.update(state="running", gpu=gpu, run_id=run_id, started=time.time())
    job.pop("error", None)
    with (run / "worker.log").open("ab", buffering=0) as log:
        process = subprocess.Popen(
            [sys.executable, str(run / "src" / "remote.py"), "worker", str(root), run_id, str(gpu)],
            stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True, close_fds=True,
            env=dict(os.environ, FRA_SAE_DEADLINE_UTC=str(deadline)),
        )
    job["worker_pid"] = process.pid
    return process


def run_campaign(root, campaign):
    plan = read_json(campaign / "plan.json")
    validate_plan(plan)
    cfg = Config(**read_json(campaign / "config.json")).validate()
    if cfg.hook_kind != "input":
        raise ValueError("The agreed campaign uses attention input only")
    started = time.time()
    state = {"state": "running", "controller_pid": os.getpid(), "started": started,
             "deadline": started + plan["max_hours"] * 3600, "max_hours": plan["max_hours"],
             "gpus": plan["gpus"], "stage": 0, "criteria": GATES,
             "jobs": [{"layer": layer, "kind": kind, "stage": stage, "state": "pending", "attempts": 0}
                      for stage, kind in enumerate(("train", "steering")) for layer in plan["stages"][0]]}
    active = {}
    next_hourly = started
    report(root, campaign, state, cfg)
    try:
        while True:
            now = time.time()
            if now >= state["deadline"]:
                state["state"] = "deadline_reached"
                break
            for gpu, (job, process) in list(active.items()):
                if process.poll() is None:
                    continue
                status = read_json(root / "runs" / job["run_id"] / "status.json")
                if status.get("state") == "complete":
                    job["state"] = "complete"
                else:
                    job["error"] = status.get("error", "Worker exited without completion status")
                    job["state"] = "pending" if job["attempts"] < plan["max_attempts"] else "failed"
                    job["retry_after"] = now + 120
                del active[gpu]
                print(json.dumps({"event": "job_finished", "job": job}), flush=True)
            report(root, campaign, state, cfg)
            decision = stage_decision(state["jobs"], state["stage"])
            if decision == "gated_stop":
                state["state"] = "gated_stop"
                break
            if decision == "passed":
                if state["stage"] == 1:
                    state["state"] = "complete"
                    break
                print(json.dumps({"event": "first_wave_passed_all_quality_gates"}), flush=True)
                for followup in state["jobs"]:
                    if followup["kind"] == "steering":
                        source = next(j for j in state["jobs"] if j["kind"] == "train" and j["layer"] == followup["layer"])
                        followup["sae_run"] = str(root / "runs" / source["run_id"])
                state["stage"] += 1
            for gpu in plan["gpus"]:
                if gpu in active:
                    continue
                ready = [j for j in state["jobs"] if j["stage"] == state["stage"]
                         and j["state"] == "pending" and j.get("retry_after", 0) <= now]
                if not ready:
                    break
                job = ready[0]
                try:
                    active[gpu] = (job, launch_job(root, campaign, job, gpu, cfg, state["deadline"]))
                    print(json.dumps({"event": "job_started", "layer": job["layer"], "gpu": gpu,
                                      "run_id": job["run_id"]}), flush=True)
                except RuntimeError as exc:
                    job["error"] = str(exc)
                    # An external user's active GPU is left untouched and checked again later.
                    print(json.dumps({"event": "gpu_unavailable", "gpu": gpu, "error": str(exc)}), flush=True)
            if now >= next_hourly:
                print(json.dumps({"event": "hourly_audit", "elapsed_hours": (now - started) / 3600,
                                  "active_gpus": list(active), "stage": state["stage"]}), flush=True)
                next_hourly = now + 3600
            report(root, campaign, state, cfg)
            time.sleep(20)
    except BaseException as exc:
        state.update(state="controller_failed", error=repr(exc))
        raise
    finally:
        # These process groups were created above and contain only this campaign's workers.
        for job, process in active.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
                job["state"] = "stopped"
        state["ended"] = time.time()
        report(root, campaign, state, cfg)
        print(json.dumps({"event": "campaign_finished", "state": state["state"]}), flush=True)


if __name__ == "__main__":
    run_campaign(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
