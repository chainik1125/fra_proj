"""Bounded single-host queue: seven local slots, including the existing L12 trainer.

Only simplex1 is used; this is within the user's aggregate eight-GPU allowance.
Every worker owns the existing per-user GPU lock and a detached process group.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lib"))
from remote import atomic_json, idle_gpu


def read(path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else {}


def environment(root, run, uuid):
    return dict(os.environ, HF_HOME=str(root / "cache/huggingface"),
        HF_DATASETS_CACHE=str(root / "cache/datasets"), TORCH_HOME=str(root / "cache/torch"),
        CUDA_VISIBLE_DEVICES=uuid, TOKENIZERS_PARALLELISM="false", HF_HUB_DISABLE_TELEMETRY="1",
        HF_HUB_DISABLE_PROGRESS_BARS="1", WANDB_MODE="disabled", PYTHONUNBUFFERED="1",
        PYTHONDONTWRITEBYTECODE="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
        OMP_NUM_THREADS="8", FRA_SAE_REMOTE_RUN=str(run), TMPDIR=str(root / "tmp"))


def run_job(run, gpu, deadline):
    root = run.parents[2]
    state = {"state": "starting", "worker_pid": os.getpid(), "started": time.time(), "gpu_index": gpu}
    atomic_json(run / "status.json", state)
    child = None
    try:
        with open(f"/tmp/fra-sae-{os.getuid()}-gpu{gpu}.lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            info = idle_gpu(gpu)
            env = environment(root, run, info["uuid"])
            state.update(state="running", gpu=info)
            atomic_json(run / "status.json", state)
            child = subprocess.Popen([str(root / "venv/bin/python"), str(run / "src/worker.py"),
                                      "--run", str(run)], env=env)
            child.wait(timeout=max(1, deadline - time.time()))
            if child.returncode != 0 or read(run / "summary.json").get("state") != "complete":
                raise RuntimeError(f"worker exited {child.returncode} without verified completion")
            state["state"] = "complete"
    except BaseException as exc:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        traceback.print_exc()
        state.update(state="failed", error=str(exc)[-1800:])
    finally:
        state["ended"] = time.time()
        atomic_json(run / "status.json", state)


def update_report(campaign, state):
    state["updated"] = time.time()
    state["elapsed_hours"] = (state["updated"] - state["started"]) / 3600
    atomic_json(campaign / "status.json", state)
    lines = ["# Cadenza Llama Scope overnight investigation", "",
             f"Queue state: **{state['state']}**. Experiments end at 08:00 PDT; final report due 09:00 PDT, September 22.",
             "", "This live table reports completion, not scientific winners. Frozen selections and prompt-level results are retained per run.",
             "", "| Task | State | GPU | Progress |", "|---|---|---:|---|"]
    for job in state["jobs"]:
        progress = read(Path(job["run"]) / "progress.json") if job.get("run") else {}
        p = f"{progress.get('event', '')} {progress.get('completed', progress.get('tokens', ''))}/{progress.get('target', '')}"
        lines.append(f"| {job['id']} | {job['state']} | {job.get('gpu', '')} | {p} |")
    lines += ["", "Residual SAE layers 7/11/15 feed attention layers 8/12/16. Official SAEs transfer from Llama 3.1 Base to the Dolphin/Llama 3.0 sleeper.",
              "", "GPU cap: seven simplex1 slots, including the pre-existing L12 training task; zero allocations on simplex2/3 by this controller.",
              "", f"Campaign: `{campaign}`. Elapsed: {state['elapsed_hours']:.2f} hours."]
    (campaign / "live_summary.md").write_text("\n".join(lines) + "\n")


def run_controller(campaign):
    root = campaign.parents[1]
    plan = read(campaign / "plan.json")
    deadline = plan["experiment_deadline"]
    assert plan["gpus"] == [0, 1, 2, 3, 4, 6, 7]
    assert len(plan["gpus"]) <= 8 and time.time() < deadline
    with (campaign / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {"state": "running", "controller_pid": os.getpid(), "started": plan["started"],
                 "deadline": deadline, "jobs": [], "gpus": plan["gpus"], "peak_gpu_allocations": 0}
        active = {}
        next_hour = time.time()
        try:
            while time.time() < deadline:
                plan = read(campaign / "plan.json")
                known = {j["id"] for j in state["jobs"]}
                for task in plan["tasks"]:
                    if task["id"] not in known:
                        state["jobs"].append({"id": task["id"], "state": "pending", "attempts": 0})
                tasks = {j["id"]: j for j in plan["tasks"]}
                train_path = Path(plan["training_run"])
                train_status = read(train_path / "status.json")
                train_finished = train_status.get("state") == "complete"
                state["training_status"] = train_status
                for gpu, (job, process) in list(active.items()):
                    if process.poll() is None:
                        continue
                    result = read(Path(job["run"]) / "status.json")
                    if result.get("state") == "complete":
                        job["state"] = "complete"
                    else:
                        job.update(state="pending" if job["attempts"] < 2 else "failed",
                                   error=result.get("error", "worker vanished"), retry_after=time.time() + 120)
                    job["ended"] = time.time()
                    del active[gpu]
                by_id = {j["id"]: j for j in state["jobs"]}
                for job in sorted(state["jobs"], key=lambda j: tasks[j["id"]].get("priority", 9)):
                    task = tasks[job["id"]]
                    if job["state"] != "pending" or job.get("retry_after", 0) > time.time():
                        continue
                    if task.get("needs_training") and not train_finished:
                        continue
                    deps = [by_id[d]["state"] for d in task.get("dependencies", [])]
                    if any(d in ("failed", "blocked") for d in deps):
                        job.update(state="blocked", error="prerequisite failed")
                        continue
                    if any(d != "complete" for d in deps):
                        continue
                    for gpu in plan["gpus"]:
                        if gpu in active or (gpu == 0 and not train_finished):
                            continue
                        try:
                            idle_gpu(gpu)
                        except (RuntimeError, subprocess.SubprocessError):
                            continue
                        job["attempts"] += 1
                        run = campaign / f"{job['id']}-a{job['attempts']}"
                        run.mkdir()
                        shutil.copytree(campaign / "src", run / "src")
                        atomic_json(run / "task.json", task)
                        shutil.copy2(campaign / "source_manifest.json", run / "source_manifest.json")
                        with (run / "worker.log").open("ab", buffering=0) as log:
                            proc = subprocess.Popen([sys.executable, str(run / "src/controller.py"),
                                "--job", str(run), "--gpu", str(gpu), "--deadline", str(deadline)],
                                stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
                        job.update(state="running", gpu=gpu, run=str(run), worker_pid=proc.pid, started=time.time())
                        active[gpu] = (job, proc)
                        break
                state["active_allocations"] = len(active) + int(not train_finished)
                assert state["active_allocations"] <= 7
                state["peak_gpu_allocations"] = max(state["peak_gpu_allocations"], state["active_allocations"])
                update_report(campaign, state)
                if time.time() >= next_hour:
                    print(json.dumps({"event": "hourly_audit", "time": time.time(),
                          "active": state["active_allocations"], "remaining_hours": (deadline-time.time())/3600}), flush=True)
                    next_hour = time.time() + 3600
                if plan.get("sealed") and train_finished and all(j["state"] in ("complete", "failed", "blocked") for j in state["jobs"]):
                    state["state"] = "experiments_finished"
                    break
                time.sleep(20)
            else:
                state["state"] = "experiment_deadline_reached"
        except BaseException as exc:
            traceback.print_exc()
            state.update(state="controller_failed", error=str(exc))
        finally:
            for job, process in active.values():
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                    job.update(state="deadline_stopped" if time.time() >= deadline else "controller_stopped")
            update_report(campaign, state)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--campaign")
    p.add_argument("--job")
    p.add_argument("--gpu", type=int)
    p.add_argument("--deadline", type=float)
    a = p.parse_args()
    if a.job:
        run_job(Path(a.job).resolve(), a.gpu, a.deadline)
    else:
        run_controller(Path(a.campaign).resolve())
