"""Remote-only deployment and detached worker. Stdlib until environment setup completes."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback

ALLOWED_FILES = {"config.py", "train.py", "remote.py", "campaign.py", "steering.py", "restoration.py", "caa_eval.py", "dom_confirmation.py", "dom_layers.py", "single_eval.py", "requirements.txt", "test_pipeline.py"}


def atomic_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def locations(root_arg, run_id):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", run_id):
        raise ValueError("Invalid run id")
    home = Path.home().resolve()
    root = (home / root_arg).resolve()
    # No arbitrary writes outside the caller's home or directly into its root.
    if root == home or home not in root.parents or root.name in (".ssh", ".cache", ".config"):
        raise ValueError("remote-root must be a dedicated task directory under your home")
    return root, root / "runs" / run_id


def idle_gpu(gpu):
    if not 0 <= gpu < 8:
        raise ValueError("GPU must be 0–7")
    output = subprocess.check_output([
        "nvidia-smi", "-i", str(gpu),
        "--query-gpu=uuid,memory.used,utilization.gpu,name", "--format=csv,noheader,nounits"
    ], text=True, timeout=15).strip()
    uuid, memory, util, name = [x.strip() for x in output.split(",", 3)]
    processes = subprocess.check_output([
        "nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid", "--format=csv,noheader"
    ], text=True, timeout=15).strip()
    if int(memory) > 128 or int(util) > 0 or processes:
        raise RuntimeError(f"GPU {gpu} is in use: {output}; compute PIDs={processes}")
    return {"index": gpu, "uuid": uuid, "name": name, "memory_MiB": int(memory)}


def idle_after_tests(gpu):
    # nvidia-smi utilization may briefly retain our completed test process's
    # activity. Retry, but never relax idle_gpu's memory/utilization/PID checks.
    for attempt in range(5):
        try:
            return idle_gpu(gpu)
        except RuntimeError:
            if attempt == 4:
                raise
            time.sleep(2)


def bootstrap(root, run):
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("uv must already be installed on the remote host")
    env = dict(os.environ, UV_CACHE_DIR=str(root / "cache" / "uv"),
               UV_PYTHON_INSTALL_DIR=str(root / "python"),
               TMPDIR=str(root / "tmp"), PIP_NO_INPUT="1")
    venv = root / "venv"
    python = venv / "bin" / "python"
    requirements = run / "src" / "requirements.txt"
    digest = hashlib.sha256(requirements.read_bytes()).hexdigest()
    marker = venv / ".requirements-sha256"
    with (root / "env.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if marker.exists() and marker.read_text().strip() != digest:
            raise RuntimeError("Dedicated env has different requirements; use a new --remote-root")
        if not marker.exists():
            if not python.exists():
                subprocess.run([uv, "venv", "--python", "3.11", str(venv)], env=env, check=True)
            subprocess.run([uv, "pip", "install", "--python", str(python),
                            "torch==2.7.1", "--index-url", "https://download.pytorch.org/whl/cu126"],
                           env=env, check=True)
            subprocess.run([uv, "pip", "install", "--python", str(python),
                            "-r", str(requirements)], env=env, check=True)
            marker.write_text(digest)
    return python, env


def worker(root, run, gpu):
    state = {"state": "starting", "worker_pid": os.getpid(), "started": time.time()}
    atomic_json(run / "status.json", state)
    try:
        # Advisory, same-user cross-run lock. This is NOT a reservation by a site scheduler.
        gpu_lock_path = Path("/tmp") / f"fra-sae-{os.getuid()}-gpu{gpu}.lock"
        with gpu_lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            state["gpu"] = idle_gpu(gpu)
            state["state"] = "installing"
            atomic_json(run / "status.json", state)
            python, env = bootstrap(root, run)
            env.update(HF_HOME=str(root / "cache" / "huggingface"),
                       HF_DATASETS_CACHE=str(root / "cache" / "datasets"),
                       TORCH_HOME=str(root / "cache" / "torch"),
                       CUDA_VISIBLE_DEVICES=state["gpu"]["uuid"],
                       TOKENIZERS_PARALLELISM="false", HF_HUB_DISABLE_TELEMETRY="1",
                       HF_HUB_DISABLE_PROGRESS_BARS="1", WANDB_MODE="disabled",
                       PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1",
                       PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                       OMP_NUM_THREADS="8", FRA_SAE_REMOTE_RUN=str(run))
            # Public repos need no credentials; no local credentials are forwarded.
            subprocess.run([str(python), "-m", "pytest", "-q", "test_pipeline.py"],
                           cwd=run / "src", env=env, check=True)
            idle_after_tests(gpu)  # Recheck after tests without accepting stale utilization.
            task = json.loads((run / "task.json").read_text()) if (run / "task.json").exists() else {"kind": "train"}
            if task["kind"] not in ("train", "steering"):
                raise ValueError("Unknown worker task")
            state["state"] = "training" if task["kind"] == "train" else "steering"
            atomic_json(run / "status.json", state)
            with (run / "environment.txt").open("w") as file:
                subprocess.run([shutil.which("uv"), "pip", "freeze", "--python", str(python)],
                               stdout=file, env=env, check=True)
            deadline = float(os.environ.get("FRA_SAE_DEADLINE_UTC", time.time() + 8 * 3600))
            script = "train.py" if task["kind"] == "train" else "steering.py"
            subprocess.run([str(python), str(run / "src" / script),
                            "--run-dir", str(run)], cwd=run / "src", env=env, check=True,
                           timeout=max(1, deadline - time.time()))
            summary = json.loads((run / "summary.json").read_text())
            if summary.get("state") != "complete":
                raise RuntimeError("Trainer exited without a verified completion summary")
            state["state"] = "complete"
    except Exception as exc:
        traceback.print_exc()
        state.update(state="failed", error=str(exc)[-2000:])
    finally:
        state["ended"] = time.time()
        atomic_json(run / "status.json", state)


def stop_run(run, reason):
    """Stop only this run's validated same-user detached process group."""
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 1000:
        raise ValueError("An explicit bounded stop reason is required")
    state = json.loads((run / "status.json").read_text())
    if state.get("state") in ("complete", "failed", "stopped"):
        return {"action": "no_op", "state": state}
    pid = state["worker_pid"]
    proc = Path(f"/proc/{pid}")
    args = (proc / "cmdline").read_bytes().split(b"\0")
    expected_script = str(run / "src" / "remote.py").encode()
    if (proc.stat().st_uid != os.getuid() or expected_script not in args
            or b"worker" not in args or run.name.encode() not in args or os.getpgid(pid) != pid):
        raise RuntimeError("Refusing to stop an unverified process group")
    os.killpg(pid, signal.SIGTERM)
    time.sleep(1)
    # If termination did not finish, validate every surviving member before escalation.
    members = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if os.getpgid(int(entry.name)) != pid:
                continue
            cmd = (entry / "cmdline").read_bytes()
            if not cmd:  # Zombie, already terminated.
                continue
            if entry.stat().st_uid != os.getuid() or str(run).encode() not in cmd:
                raise RuntimeError("Unexpected process in run group; refusing escalation")
            members.append(int(entry.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
    if members:
        os.killpg(pid, signal.SIGKILL)
    state.update(state="stopped", ended=time.time(), stop_reason=reason, stopped_process_group=pid)
    atomic_json(run / "status.json", state)
    return {"action": "stopped", "state": state}


def main():
    action, root_arg, run_id, gpu_arg = sys.argv[1:]
    root, run = locations(root_arg, run_id)
    is_campaign = action.startswith("campaign_")
    gpu = int(gpu_arg.split(",")[0])
    if is_campaign:
        run = root / "campaigns" / run_id
    if action in ("deploy", "campaign_deploy"):
        payload = sys.stdin.buffer.read(2_000_001)
        if len(payload) > 2_000_000:
            raise ValueError("Upload exceeds 2 MB")
        data = json.loads(payload)
        if set(data["files"]) != ALLOWED_FILES:
            raise ValueError("Only the explicit source allowlist may be deployed")
        for name, item in data["files"].items():
            content = item["text"].encode()
            if len(content) > 500_000 or hashlib.sha256(content).hexdigest() != item["sha256"]:
                raise ValueError(f"Invalid source {name}")
        gpu_indices = [int(x) for x in gpu_arg.split(",")] if is_campaign else [gpu]
        for index in gpu_indices:
            idle_gpu(index)
        minimum_gib = 200 if is_campaign else 80
        if shutil.disk_usage(Path.home()).free < minimum_gib * 1024**3:
            raise RuntimeError(f"Need at least {minimum_gib} GiB free remotely")
        root.mkdir(parents=True, exist_ok=True)
        for name in ("cache", "tmp", "runs", "campaigns"):
            (root / name).mkdir(exist_ok=True)
        run.mkdir()  # Never overwrite an existing run.
        (run / "src").mkdir()
        for name, item in data["files"].items():
            (run / "src" / name).write_text(item["text"])
        atomic_json(run / "config.json", data["config"])
        if data.get("task"):
            atomic_json(run / "task.json", data["task"])
        atomic_json(run / "source_manifest.json", {n: v["sha256"] for n, v in data["files"].items()})
        # Validate the uploaded configuration without importing ML dependencies.
        sys.path.insert(0, str(run / "src"))
        from config import Config
        Config(**data["config"]).validate()
        if is_campaign:
            from campaign import validate_plan
            validate_plan(data["campaign"])
            if data["campaign"]["gpus"] != gpu_indices:
                raise ValueError("Campaign GPU selection disagrees with preflight")
            atomic_json(run / "plan.json", data["campaign"])
            command = [sys.executable, str(run / "src" / "campaign.py"), str(root), str(run)]
        else:
            command = [sys.executable, str(run / "src" / "remote.py"), "worker", str(root), run_id, str(gpu)]
        with (run / "worker.log").open("ab", buffering=0) as log:
            proc = subprocess.Popen(command,
                                    stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                    start_new_session=True, close_fds=True)
        print(json.dumps({"state": "launched", "host": os.uname().nodename,
                          "run_dir": str(run), "run_id": run_id, "worker_pid": proc.pid,
                          "local_artifacts_downloaded": False}, indent=2))
    elif action == "worker":
        worker(root, run, gpu)
    elif action == "stop":
        data = json.loads(sys.stdin.buffer.read(2001))
        print(json.dumps(stop_run(run, data["reason"]), indent=2))
    elif action in ("status", "campaign_status"):
        result = {"run_dir": str(run)}
        for name in ("status.json", "progress.json", "summary.json", "checkpoint_status.json"):
            path = run / name
            if path.exists():
                if path.stat().st_size > 100_000:
                    raise ValueError(f"Refusing oversized status file {name}")
                result[name] = json.loads(path.read_text())
        status = result.get("status.json", {})
        pid = status.get("worker_pid", status.get("controller_pid"))
        if pid and status.get("state") not in ("complete", "failed", "stopped", "gated_stop", "deadline_reached", "controller_failed"):
            cmdline = Path(f"/proc/{pid}/cmdline")
            alive = cmdline.exists() and str(run).encode() in cmdline.read_bytes()
            result["worker_alive"] = alive
            if not alive:
                result["warning"] = "Worker is no longer running; inspect logs (no automatic restart)."
        print(json.dumps(result, indent=2))
    elif action in ("logs", "campaign_logs", "campaign_summary"):
        path = run / ("summary.md" if action == "campaign_summary" else "worker.log")
        with path.open("rb") as file:
            file.seek(max(0, path.stat().st_size - 32_000))
            print(file.read(32_000).decode(errors="replace"))
    else:
        raise ValueError("Unknown remote action")


if __name__ == "__main__":
    main()
