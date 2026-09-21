#!/usr/bin/env python3
"""Local stdlib-only launcher: no ML imports, downloads, archives, or artifact fetches."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

from config import Config, MAX_LOCAL_FILE, PAYLOAD_FILES, validate_run_id

HERE = Path(__file__).resolve().parent


def make_payload(cfg, campaign=None, task=None):
    files = {}
    for name in PAYLOAD_FILES:
        path = HERE / name
        if path.is_symlink() or path.stat().st_size > min(MAX_LOCAL_FILE, 500_000):
            raise ValueError(f"Refusing oversized or symlinked source: {path}")
        content = path.read_text()
        files[name] = {"text": content, "sha256": hashlib.sha256(content.encode()).hexdigest()}
    payload = json.dumps({"config": asdict(cfg), "files": files, "campaign": campaign, "task": task}).encode()
    if len(payload) > 2_000_000:
        raise ValueError("Source payload exceeds 2 MB")
    return payload


def call_remote(args, action, payload=b""):
    # remote.py is passed as executable Python code, not as a shell-interpolated script.
    code = (HERE / "remote.py").read_text()
    gpu_arg = ",".join(map(str, args.gpus)) if action.startswith("campaign_") else str(args.gpu)
    command = shlex.join(["python3", "-c", code, action, args.remote_root, args.run_id, gpu_arg])
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", args.host, command],
        input=payload, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60,
    )
    # Remote responses themselves have hard caps; nothing is saved locally.
    if len(result.stdout) + len(result.stderr) > 1_000_000:
        raise RuntimeError("Unexpectedly large SSH response")
    sys.stdout.write(result.stdout.decode(errors="replace"))
    sys.stderr.write(result.stderr.decode(errors="replace"))
    if result.returncode:
        raise SystemExit(result.returncode)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["plan", "launch", "status", "logs", "stop", "campaign-plan",
                                      "campaign-launch", "campaign-status", "campaign-logs", "campaign-summary"])
    p.add_argument("--host", choices=["simplex1", "simplex2", "simplex3"], default="simplex1")
    p.add_argument("--remote-root", default="sae-middle", help="Task directory under your remote home")
    p.add_argument("--run-id")
    p.add_argument("--gpu", type=int, choices=range(8), default=0)
    p.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3])
    p.add_argument("--max-hours", type=float, default=8)
    p.add_argument("--max-attempts", type=int, default=2)
    p.add_argument("--variant", choices=["A", "B"], default="A")
    p.add_argument("--layer", type=int, default=16, help="Zero-based block index (default 16)")
    p.add_argument("--hook", choices=["input", "output", "resid_mid", "resid_post"], default="input")
    p.add_argument("--training-reference", help="Completed remote input-SAE run; require identical config except hook_kind")
    p.add_argument("--smoke", action="store_true", help="100k tokens, same full SAE width, small eval")
    p.add_argument("--tokens", type=int, help="Override smoke / full token budget")
    p.add_argument("--checkpoint-every-tokens", type=int, default=10_000_000)
    p.add_argument("--steering-smoke-from", help="Remote completed input-SAE run directory; engineering test only")
    p.add_argument("--steering-from", help="Remote completed input-SAE run directory for the full comparison")
    p.add_argument("--restoration-from", help="Completed steering run: correct triggered-to-clean JSD using its frozen candidates and splits")
    p.add_argument("--ov-probe-feature", type=int, help="Probe one existing OV candidate on the frozen validation split")
    p.add_argument("--ov-probe-alphas", nargs="+", type=float, help="Explicit strengths for the validation-only OV probe")
    p.add_argument("--caa-probe-alphas", nargs="+", type=float, help="Constant clean-minus-triggered input-vector probe on validation")
    p.add_argument("--caa-evaluation", action="store_true", help="Paired DoM: input/prompt and residual/generation variants, signed validation sweep then frozen test")
    p.add_argument("--quality-gate-override-reason", help="Record explicit user authorization to proceed despite failed SAE quality gates")
    p.add_argument("--stop-reason", help="Required audit reason for stopping one exact run")
    args = p.parse_args()
    task = None
    if args.steering_from and args.steering_smoke_from:
        p.error("Choose full steering or steering smoke, not both")
    if args.quality_gate_override_reason and not args.steering_from:
        p.error("--quality-gate-override-reason requires --steering-from")
    if args.restoration_from and not args.steering_from:
        p.error("--restoration-from requires --steering-from")
    if (args.ov_probe_feature is not None or args.ov_probe_alphas is not None) and (
            args.ov_probe_feature is None or args.ov_probe_alphas is None or not args.restoration_from):
        p.error("OV probe requires --ov-probe-feature, --ov-probe-alphas and --restoration-from")
    if args.caa_probe_alphas is not None and (not args.restoration_from or args.ov_probe_feature is not None):
        p.error("CAA probe requires --restoration-from and cannot be combined with an OV probe")
    if args.caa_evaluation and (not args.restoration_from or args.ov_probe_feature is not None or args.caa_probe_alphas is not None):
        p.error("CAA evaluation requires --restoration-from and cannot be combined with probes")
    if args.steering_from:
        if args.action not in ("launch", "plan") or args.hook != "input" or args.smoke:
            p.error("Full steering requires plan/launch, --hook input, and no --smoke")
        task = {"kind": "steering", "sae_run": args.steering_from, "smoke": False}
        if args.restoration_from:
            task["restoration_from"] = args.restoration_from
        if args.ov_probe_feature is not None:
            from restoration import validate_probe
            task["ov_probe"] = validate_probe({"feature": args.ov_probe_feature, "alphas": args.ov_probe_alphas})
        if args.caa_probe_alphas is not None:
            from restoration import validate_probe
            task["caa_probe"] = validate_probe({"kind": "caa", "feature": 0, "alphas": args.caa_probe_alphas})
        if args.caa_evaluation:
            task["caa_evaluation"] = True
        if args.quality_gate_override_reason:
            reason = args.quality_gate_override_reason.strip()
            if not reason or len(reason) > 2000:
                p.error("Override reason must be nonempty and at most 2000 characters")
            task["quality_gate_override"] = {
                "reason": reason, "approved_layer": args.layer, "sae_run": args.steering_from,
                "recorded_utc": datetime.now(timezone.utc).isoformat(),
            }
    if args.steering_smoke_from:
        if args.action != "launch":
            p.error("--steering-smoke-from requires launch")
        task = {"kind": "steering", "sae_run": args.steering_smoke_from, "smoke": True,
                "protocol": {"selection_pairs": 2, "validation_pairs": 2, "test_pairs": 2,
                             "candidate_pool": 2, "candidates_per_method": 1,
                             "alphas": [0.0, 1.0], "generation_tokens": 4, "batch_size": 2}}
    if args.training_reference:
        if task is not None or args.action not in ("launch", "plan") or args.hook not in ("resid_mid", "resid_post"):
            p.error("--training-reference requires a residual training plan/launch, not a steering task")
        task = {"kind": "train", "parameter_reference": args.training_reference}
    cfg = Config(variant=args.variant, layer=args.layer, hook_kind=args.hook)
    cfg.checkpoint_every_tokens = args.checkpoint_every_tokens
    if args.smoke:
        cfg.training_tokens = 100_000
        cfg.buffer_tokens = 32768
        cfg.eval_per_class = 16
        cfg.log_every_steps = 5
    if args.tokens is not None:
        cfg.training_tokens = args.tokens
    cfg.validate()
    if not args.run_id:
        if args.action in ("status", "logs", "stop", "campaign-status", "campaign-logs", "campaign-summary"):
            p.error("status/logs require --run-id")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.run_id = f"{args.variant}-L{args.layer}-{cfg.training_tokens}-{stamp}"
    validate_run_id(args.run_id)
    if args.action.startswith("campaign-"):
        from campaign import DEFAULT_STAGES, validate_plan
        plan = {"gpus": args.gpus, "stages": DEFAULT_STAGES,
                "max_hours": args.max_hours, "max_attempts": args.max_attempts}
        validate_plan(plan)
        if len(args.run_id) > 70:
            p.error("Campaign run-id must be at most 70 characters")
        if args.action == "campaign-plan":
            print(json.dumps({"campaign": plan, "config": cfg.manifest(),
                              "upload_bytes": len(make_payload(cfg, plan)), "local_writes": "none"}, indent=2))
        elif args.action == "campaign-launch":
            call_remote(args, "campaign_deploy", make_payload(cfg, plan))
        else:
            call_remote(args, args.action.replace("-", "_"))
    elif args.action == "plan":
        payload = make_payload(cfg, task=task)
        print(json.dumps({"host": args.host, "gpu": args.gpu, "run_id": args.run_id,
                          "remote_root": args.remote_root, "upload_bytes": len(payload),
                          "local_writes": "none", "config": cfg.manifest(), "task": task}, indent=2))
    elif args.action == "launch":
        call_remote(args, "deploy", make_payload(cfg, task=task))
    elif args.action == "stop":
        if not args.stop_reason or not args.stop_reason.strip() or len(args.stop_reason) > 1000:
            p.error("stop requires --stop-reason (1–1000 characters)")
        call_remote(args, "stop", json.dumps({"reason": args.stop_reason}).encode())
    else:
        call_remote(args, args.action)


if __name__ == "__main__":
    main()
