#!/usr/bin/env python3
"""Poll the FRA request board and launch validated jobs on Simplex."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
BRANCH = "dmitry/cadenza-llamascope-overnight-20260921"
REMOTE = "origin"
REQUEST_PATH = "experiments/message-board/indranil-requests.md"
HOSTS = ("simplex1", "simplex2", "simplex3")
REMOTE_ROOT = "/data/users/dmitry/fra-compute-bridge"
REMOTE_PYTHON = "/data/users/dmitry/simplex-research/.venv/bin/python"
MAX_GPUS = 2
MAX_TIMEOUT_MINUTES = 72 * 60
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SECTION_RE = re.compile(
    r"(?ms)^## request: ([A-Za-z0-9][A-Za-z0-9._-]{0,63})[ \t]*\n"
    r"(.*?)(?=^## |\Z)"
)
JSON_BLOCK_RE = re.compile(r"(?ms)^```json[ \t]*\n(.*?)\n```[ \t]*$")
STATE_DEFAULT = Path.home() / "Library/Application Support/FRA Compute Bridge"
SSH_OPTIONS = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=15")


class BridgeError(RuntimeError):
    pass


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S %z')}] {message}", flush=True)


def run(args: list[str], *, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(args, cwd=cwd, text=True, capture_output=True)
    if check and result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise BridgeError(f"command failed ({result.returncode}): {shlex.join(args)}: {detail}")
    return result


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run(["git", *args], cwd=ROOT, check=check)


def ssh(host: str, remote_command: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run(["ssh", *SSH_OPTIONS, host, remote_command], check=check)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def load_state(path: Path) -> dict:
    if not path.exists():
        return {"jobs": {}, "last_tip": None}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(state, dict) or not isinstance(state.get("jobs"), dict):
            raise ValueError("invalid state shape")
        state.setdefault("last_tip", None)
        return state
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise BridgeError(f"cannot read state file {path}: {exc}") from exc


def fetch_tip() -> str:
    ref = f"refs/heads/{BRANCH}"
    dst = f"refs/remotes/{REMOTE}/{BRANCH}"
    git("fetch", "--no-tags", REMOTE, f"+{ref}:{dst}")
    result = git("rev-parse", f"{dst}^{{commit}}")
    tip = result.stdout.strip()
    if not SHA_RE.fullmatch(tip):
        raise BridgeError(f"unexpected branch tip: {tip!r}")
    return tip


def parse_board(markdown: str) -> list[dict]:
    requests: list[dict] = []
    seen: set[str] = set()
    for heading_id, section in SECTION_RE.findall(markdown):
        blocks = JSON_BLOCK_RE.findall(section)
        if len(blocks) != 1:
            raise BridgeError(f"request {heading_id}: expected exactly one JSON block")
        try:
            request = json.loads(blocks[0])
        except json.JSONDecodeError as exc:
            raise BridgeError(f"request {heading_id}: invalid JSON: {exc}") from exc
        if not isinstance(request, dict):
            raise BridgeError(f"request {heading_id}: JSON must be an object")
        if request.get("id") != heading_id:
            raise BridgeError(f"request {heading_id}: heading and JSON id must match")
        if heading_id in seen:
            raise BridgeError(f"duplicate request id: {heading_id}")
        seen.add(heading_id)
        requests.append(request)
    return requests


def validate_request(request: dict, tip: str) -> str | None:
    if request.get("status") == "DRAFT":
        return None
    required = {"id", "status", "host", "gpus", "entrypoint", "args", "timeout_minutes"}
    if set(request) != required:
        raise BridgeError(f"request {request.get('id')}: fields must be exactly {sorted(required)}")
    request_id = request["id"]
    if not isinstance(request_id, str) or not ID_RE.fullmatch(request_id):
        raise BridgeError("request id has invalid characters")
    if request["status"] != "READY":
        raise BridgeError(f"request {request_id}: status must be DRAFT or READY")
    if request["host"] != "auto" and request["host"] not in HOSTS:
        raise BridgeError(f"request {request_id}: host must be auto or one of {HOSTS}")
    gpus = request["gpus"]
    if isinstance(gpus, bool) or not isinstance(gpus, int) or not 1 <= gpus <= MAX_GPUS:
        raise BridgeError(f"request {request_id}: gpus must be 1 or 2")
    timeout = request["timeout_minutes"]
    if isinstance(timeout, bool) or not isinstance(timeout, int) or not 1 <= timeout <= MAX_TIMEOUT_MINUTES:
        raise BridgeError(f"request {request_id}: timeout_minutes must be between 1 and {MAX_TIMEOUT_MINUTES}")
    args = request["args"]
    if not isinstance(args, list) or len(args) > 128 or any(not isinstance(arg, str) for arg in args):
        raise BridgeError(f"request {request_id}: args must be a list of at most 128 strings")
    if any(any(ord(char) < 32 for char in arg) or len(arg) > 4096 for arg in args):
        raise BridgeError(f"request {request_id}: args contain control characters or oversized values")
    entrypoint = request["entrypoint"]
    if not isinstance(entrypoint, str):
        raise BridgeError(f"request {request_id}: entrypoint must be a repository path")
    path = PurePosixPath(entrypoint)
    expected_prefix = f"experiments/message-board/{request_id}/"
    if (
        path.is_absolute()
        or ".." in path.parts
        or not re.fullmatch(r"[A-Za-z0-9_./-]+", entrypoint)
        or not entrypoint.startswith(expected_prefix)
        or path.suffix != ".py"
    ):
        raise BridgeError(f"request {request_id}: entrypoint must be a .py file under {expected_prefix}")
    tree_entry = git("ls-tree", tip, "--", entrypoint).stdout.strip()
    if not tree_entry:
        raise BridgeError(f"request {request_id}: entrypoint is not tracked at pushed commit {tip[:12]}")
    mode = tree_entry.split(" ", 1)[0]
    if mode not in {"100644", "100755"}:
        raise BridgeError(f"request {request_id}: entrypoint must be a regular tracked file")
    return request_id


def remote_job_base(slug: str) -> str:
    return f"{REMOTE_ROOT}/jobs/{slug}"


def collect_remote_records() -> tuple[list[dict], dict[str, set[int]]]:
    active: list[dict] = []
    reserved_devices: dict[str, set[int]] = {host: set() for host in HOSTS}
    for host in HOSTS:
        find_cmd = (
            f"if test -d {shlex.quote(REMOTE_ROOT + '/jobs')}; then "
            f"find {shlex.quote(REMOTE_ROOT + '/jobs')} -type f -name active.tsv -print0; fi"
        )
        paths = ssh(host, find_cmd).stdout.split("\0")
        for marker in (item for item in paths if item):
            raw = ssh(host, f"cat -- {shlex.quote(marker)}").stdout.strip()
            fields = raw.split("\t")
            if len(fields) != 5:
                raise BridgeError(f"{host}: malformed active marker {marker}")
            request_id, gpu_text, pid_text, device_text, started_text = fields
            if not ID_RE.fullmatch(request_id):
                raise BridgeError(f"{host}: unsafe request id in active marker")
            try:
                gpu_count, pid, started = int(gpu_text), int(pid_text), int(started_text)
                devices = {int(item) for item in device_text.split(",") if item}
            except ValueError as exc:
                raise BridgeError(f"{host}: malformed active marker {marker}") from exc
            if not 1 <= gpu_count <= MAX_GPUS or len(devices) != gpu_count:
                raise BridgeError(f"{host}: invalid GPU reservation in {marker}")
            base = marker.rsplit("/active.tsv", 1)[0]
            if pid == 0:
                status = "pending" if time.time() - started <= 600 else "dead"
            else:
                status_cmd = (
                    f"if test -f {shlex.quote(base + '/exit_code')}; then "
                    f"printf 'done '; cat {shlex.quote(base + '/exit_code')}; "
                    f"elif kill -0 {pid} 2>/dev/null; then printf 'running\\n'; "
                    "else printf 'dead\\n'; fi"
                )
                status_result = ssh(host, status_cmd, check=False)
                if status_result.returncode:
                    raise BridgeError(f"{host}: cannot reconcile job {request_id}")
                status = status_result.stdout.strip()
            if status.startswith("done ") or status == "dead":
                ssh(host, f"rm -f -- {shlex.quote(marker)}", check=True)
                code = status.split(" ", 1)[1].strip() if status.startswith("done ") else None
                active.append({"id": request_id, "host": host, "gpus": gpu_count,
                               "devices": sorted(devices), "pid": pid, "state": "finished",
                               "exit_code": code})
                continue
            if status not in {"running", "pending"}:
                raise BridgeError(f"{host}: unexpected job status for {request_id}: {status!r}")
            reserved_devices[host].update(devices)
            active.append({"id": request_id, "host": host, "gpus": gpu_count,
                           "devices": sorted(devices), "pid": pid, "state": "running"})
    return active, reserved_devices


def gpu_inventory(host: str, reserved: set[int]) -> list[int]:
    query = (
        "nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu "
        "--format=csv,noheader,nounits; printf '\\n--COMPUTE_APPS--\\n'; "
        "nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader"
    )
    output = ssh(host, query).stdout
    if "--COMPUTE_APPS--" not in output:
        raise BridgeError(f"{host}: could not read GPU inventory")
    gpu_text, app_text = output.split("--COMPUTE_APPS--", 1)
    busy_uuids: set[str] = set()
    for line in app_text.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 2 and fields[1].startswith("GPU-"):
            busy_uuids.add(fields[1])
    idle: list[int] = []
    for line in gpu_text.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            continue
        try:
            index, memory_mb, util = int(fields[0]), int(fields[2]), int(fields[3])
        except ValueError:
            continue
        if index not in reserved and fields[1] not in busy_uuids and memory_mb <= 128 and util <= 5:
            idle.append(index)
    if not gpu_text.strip():
        raise BridgeError(f"{host}: empty GPU inventory")
    return idle


def choose_devices(request: dict, active: list[dict], reserved: dict[str, set[int]]) -> tuple[str, list[int]] | None:
    occupied = sum(record["gpus"] for record in active if record["state"] == "running")
    count = request["gpus"]
    if occupied + count > MAX_GPUS:
        return None
    candidates = (request["host"],) if request["host"] != "auto" else HOSTS
    inventories = {host: gpu_inventory(host, reserved[host]) for host in candidates}
    for host in candidates:
        if len(inventories[host]) >= count:
            return host, inventories[host][:count]
    return None


def write_remote_text(host: str, path: str, content: str) -> None:
    command = f"mkdir -p {shlex.quote(path.rsplit('/', 1)[0])} && cat > {shlex.quote(path)}"
    result = subprocess.run(["ssh", *SSH_OPTIONS, host, command], input=content, text=True,
                            capture_output=True)
    if result.returncode:
        raise BridgeError(f"{host}: cannot write job metadata: {result.stderr.strip()}")


def transfer_source(host: str, tip: str, request_id: str, job_dir: str) -> None:
    remote_source = job_dir + "/source"
    command = f"mkdir -p {shlex.quote(remote_source)} && tar -xf - -C {shlex.quote(remote_source)}"
    archive_paths = ["fra", f"experiments/message-board/{request_id}", "requirements.txt"]
    producer = subprocess.Popen(["git", "archive", "--format=tar", tip, *archive_paths],
                                cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert producer.stdout is not None
    consumer = subprocess.Popen(["ssh", *SSH_OPTIONS, host, command], stdin=producer.stdout,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    producer.stdout.close()
    _, consumer_err = consumer.communicate()
    producer_err = producer.stderr.read() if producer.stderr else b""
    producer.wait()
    if producer.returncode or consumer.returncode:
        detail = (consumer_err or producer_err).decode(errors="replace").strip()
        raise BridgeError(f"{host}: source transfer failed: {detail}")


def make_run_script(request: dict, devices: list[int], tip: str, job_dir: str) -> str:
    request_id = request["id"]
    gpu_csv = ",".join(str(item) for item in devices)
    entrypoint = request["entrypoint"]
    argv = [REMOTE_PYTHON, "-u", entrypoint, *request["args"]]
    marker = job_dir + "/active.tsv"
    exit_file = job_dir + "/exit_code"
    return "\n".join([
        "#!/usr/bin/env bash",
        "set -u",
        f"cd {shlex.quote(job_dir + '/source')}",
        f"mkdir -p {shlex.quote(job_dir + '/output')}",
        f"printf '%s\\t%s\\t%s\\t%s\\t%s\\n' {shlex.quote(request_id)} "
        f"{request['gpus']} \"$$\" {shlex.quote(gpu_csv)} \"$(date +%s)\" "
        f"> {shlex.quote(marker + '.tmp')}",
        f"mv {shlex.quote(marker + '.tmp')} {shlex.quote(marker)}",
        f"export CUDA_VISIBLE_DEVICES={shlex.quote(gpu_csv)}",
        f"export JOB_ID={shlex.quote(request_id)}",
        f"export JOB_OUTPUT_DIR={shlex.quote(job_dir + '/output')}",
        f"export JOB_SOURCE_COMMIT={shlex.quote(tip)}",
        "set +e",
        f"/usr/bin/timeout --signal=TERM --kill-after=60s {request['timeout_minutes'] * 60} "
        + shlex.join(argv),
        "status=$?",
        "set -e",
        f"printf '%s\\n' \"$status\" > {shlex.quote(exit_file + '.tmp')}",
        f"mv {shlex.quote(exit_file + '.tmp')} {shlex.quote(exit_file)}",
        f"rm -f -- {shlex.quote(marker)}",
        "exit \"$status\"",
        "",
    ])


def submit(request: dict, host: str, devices: list[int], tip: str) -> dict:
    request_id = request["id"]
    slug = f"{request_id}-{tip[:12]}"
    job_dir = remote_job_base(slug)
    exists = ssh(host, f"test ! -e {shlex.quote(job_dir)}", check=False)
    if exists.returncode:
        raise BridgeError(f"{host}: remote job directory already exists; refusing to overwrite {slug}")
    ssh(host, f"test -x {shlex.quote(REMOTE_PYTHON)} && test -x /usr/bin/timeout")
    transfer_source(host, tip, request_id, job_dir)
    receipt = {
        "id": request_id, "commit": tip, "host": host, "gpus": request["gpus"],
        "devices": devices, "entrypoint": request["entrypoint"],
        "args": request["args"], "timeout_minutes": request["timeout_minutes"],
        "status": "prepared",
    }
    receipt_path = job_dir + "/receipt.json"
    write_remote_text(host, receipt_path, json.dumps(receipt, indent=2) + "\n")
    gpu_csv = ",".join(str(item) for item in devices)
    reservation = f"{request_id}\t{request['gpus']}\t0\t{gpu_csv}\t{int(time.time())}\n"
    write_remote_text(host, job_dir + "/active.tsv", reservation)
    write_remote_text(host, job_dir + "/run.sh", make_run_script(request, devices, tip, job_dir))
    ssh(host, f"chmod 700 {shlex.quote(job_dir + '/run.sh')}")
    launch = (
        f"cd {shlex.quote(job_dir)} && "
        f"(nohup setsid bash {shlex.quote(job_dir + '/run.sh')} "
        f"> {shlex.quote(job_dir + '/launcher.log')} 2>&1 < /dev/null & echo $!)"
    )
    started = ssh(host, launch)
    try:
        pid = int(started.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise BridgeError(f"{host}: launch did not return a pid: {started.stdout!r}") from exc
    receipt["status"] = "running"
    receipt["pid"] = pid
    write_remote_text(host, receipt_path, json.dumps(receipt, indent=2) + "\n")
    return {**receipt, "job_dir": job_dir, "slug": slug}


def save_job(state: dict, record: dict, state_path: Path) -> None:
    state["jobs"][record["id"]] = record
    atomic_json(state_path, state)


def poll(state_path: Path, *, dry_run: bool = False) -> None:
    state = load_state(state_path)
    if dry_run:
        tip = git("rev-parse", f"refs/remotes/{REMOTE}/{BRANCH}^{{commit}}").stdout.strip()
        if not SHA_RE.fullmatch(tip):
            raise BridgeError(f"unexpected cached branch tip: {tip!r}")
    else:
        tip = fetch_tip()
    board_result = git("show", f"{tip}:{REQUEST_PATH}", check=False)
    if board_result.returncode:
        log(f"pushed branch {BRANCH} has no {REQUEST_PATH}; nothing to do")
        return
    requests = parse_board(board_result.stdout)
    active, reserved = collect_remote_records()
    for record in active:
        if record["state"] == "finished":
            old = state["jobs"].get(record["id"], {})
            old.update(record)
            old["status"] = "completed" if record.get("exit_code") == "0" else "failed"
            save_job(state, old, state_path)
            log(f"job {record['id']} finished on {record['host']} (exit={record.get('exit_code')})")
    if not requests:
        if state.get("last_tip") != tip:
            if board_result.stdout.strip():
                log(f"checked pushed commit {tip[:12]}; board has no supported JSON request blocks")
            else:
                log(f"checked pushed commit {tip[:12]}; request board is empty")
        state["last_tip"] = tip
        atomic_json(state_path, state)
        return
    for request in requests:
        request_id = request.get("id")
        if request.get("status") != "READY":
            continue
        if request_id in state["jobs"]:
            continue
        try:
            valid_id = validate_request(request, tip)
        except BridgeError as exc:
            log(f"ignored invalid request: {exc}")
            continue
        if valid_id is None:
            continue
        already_active = any(item["id"] == valid_id for item in active)
        if already_active:
            log(f"request {valid_id} already has a remote job; skipping duplicate")
            continue
        choice = choose_devices(request, active, reserved)
        if choice is None:
            log(f"request {valid_id} is valid but capacity is not available; will retry next poll")
            continue
        host, devices = choice
        if dry_run:
            log(f"DRY RUN: would submit {valid_id} from {tip[:12]} to {host} GPUs {devices}")
            continue
        log(f"submitting {valid_id} from {tip[:12]} to {host} GPUs {devices}")
        reserved_job = {
            "id": valid_id, "commit": tip, "host": host, "gpus": len(devices),
            "devices": devices, "status": "submitting",
        }
        save_job(state, reserved_job, state_path)
        record = submit(request, host, devices, tip)
        record["status"] = "running"
        save_job(state, record, state_path)
        active.append({"id": valid_id, "host": host, "gpus": len(devices),
                       "devices": devices, "state": "running"})
        reserved[host].update(devices)
        log(f"submitted {valid_id}: pid={record['pid']} log={record['job_dir']}/launcher.log")
        break
    state["last_tip"] = tip
    atomic_json(state_path, state)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="run one poll cycle")
    parser.add_argument("--dry-run", action="store_true", help="fetch and validate without launching")
    parser.add_argument("--state-dir", type=Path, default=STATE_DEFAULT)
    args = parser.parse_args()
    if not args.once and not args.dry_run:
        parser.error("invoke with --once; launchd provides the 120-second interval")
    args.state_dir.mkdir(parents=True, exist_ok=True)
    lock_path = args.state_dir / "poll.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log("another poll cycle is still running; skipping")
            return 0
        state_path = args.state_dir / "state.json"
        try:
            poll(state_path, dry_run=args.dry_run)
        except (BridgeError, OSError) as exc:
            log(f"poll failed closed: {exc}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
