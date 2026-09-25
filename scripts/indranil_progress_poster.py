#!/usr/bin/env python3
"""Post five-minute progress updates for the active Indranil SAE request."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import subprocess
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BRANCH = "dmitry/cadenza-llamascope-overnight-20260921"
REMOTE = "origin"
REQUEST_PATH = "experiments/message-board/indranil-requests.md"
HOST = "simplex1"
RUN_ID = "STD-input-L8-100M-20260924"
RUN_DIR = f"/data/users/dmitry/sae-middle/runs/{RUN_ID}"
SSH_OPTIONS = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=15")
STATE_DEFAULT = Path.home() / "Library/Application Support/FRA Compute Bridge"
REMOTE_STATUS_CODE = f'''from pathlib import Path
import json
root = Path({RUN_DIR!r})
def read(name):
    path = root / name
    if not path.exists() or path.stat().st_size > 100000:
        return {{}}
    return json.loads(path.read_text())
print(json.dumps({{"status": read("status.json"),
                  "progress": read("progress.json"),
                  "checkpoint": read("checkpoint_status.json")}}))
'''


class PosterError(RuntimeError):
    pass


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S %z')}] {message}", flush=True)


def run(args: list[str], *, cwd: Path = ROOT, env: dict | None = None,
        input_text: str | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(args, cwd=cwd, env=env, input=input_text, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    if check and result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise PosterError(f"command failed ({result.returncode}): {shlex.join(args)}: {detail}")
    return result


def git(*args: str, env: dict | None = None, input_text: str | None = None,
        check: bool = True) -> subprocess.CompletedProcess[str]:
    return run(["git", *args], env=env, input_text=input_text, check=check)


def fetch_tip() -> str:
    source = f"+refs/heads/{BRANCH}:refs/remotes/{REMOTE}/{BRANCH}"
    git("fetch", "--no-tags", REMOTE, source)
    tip = git("rev-parse", f"refs/remotes/{REMOTE}/{BRANCH}^{{commit}}").stdout.strip()
    if len(tip) != 40 or any(char not in "0123456789abcdef" for char in tip):
        raise PosterError(f"unexpected branch tip: {tip!r}")
    return tip


def read_remote_status() -> dict:
    remote_command = "python3 -c " + shlex.quote(REMOTE_STATUS_CODE)
    result = subprocess.run(["ssh", *SSH_OPTIONS, HOST, remote_command], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=40)
    if result.returncode:
        raise PosterError(f"could not read {HOST} status (ssh exit {result.returncode})")
    if len(result.stdout) > 300_000:
        raise PosterError("remote status response exceeded the size limit")
    try:
        snapshot = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise PosterError(f"remote status was not JSON: {exc}") from exc
    if not isinstance(snapshot, dict):
        raise PosterError("remote status response must be an object")
    return snapshot


def int_or_none(value) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value < 0:
        return None
    return int(value)


def duration_text(seconds) -> str:
    value = int_or_none(seconds)
    if value is None:
        return "unknown"
    return str(timedelta(seconds=value))


def progress_facts(snapshot: dict, previous: dict) -> dict:
    status = snapshot.get("status") if isinstance(snapshot.get("status"), dict) else {}
    progress = snapshot.get("progress") if isinstance(snapshot.get("progress"), dict) else {}
    checkpoint = snapshot.get("checkpoint") if isinstance(snapshot.get("checkpoint"), dict) else {}
    state = status.get("state")
    if state not in {"starting", "installing", "training", "steering", "complete",
                     "failed", "stopped", "gated_stop", "deadline_reached"}:
        state = "unknown"
    tokens = int_or_none(progress.get("tokens"))
    target = int_or_none(progress.get("target"))
    step = int_or_none(progress.get("step"))
    elapsed = int_or_none(progress.get("elapsed_seconds"))
    if tokens is None:
        tokens = int_or_none(previous.get("last_tokens"))
    if target is None:
        target = int_or_none(previous.get("last_target")) or 100_000_000
    if step is None:
        step = int_or_none(previous.get("last_step"))
    if elapsed is None:
        elapsed = int_or_none(previous.get("last_elapsed_seconds"))
    checkpoint_tokens = int_or_none(checkpoint.get("tokens"))
    reload_check = checkpoint.get("reload_check")
    reload_passed = reload_check.get("passed") if isinstance(reload_check, dict) else None
    loss = progress.get("loss")
    if isinstance(loss, bool) or not isinstance(loss, (int, float)):
        loss = None
    return {
        "state": state, "tokens": tokens, "target": target, "step": step,
        "elapsed_seconds": elapsed, "checkpoint_tokens": checkpoint_tokens,
        "reload_passed": reload_passed if isinstance(reload_passed, bool) else None,
        "loss": float(loss) if loss is not None else None,
        "status_available": bool(status),
    }


def status_summary(facts: dict, *, read_error: bool = False) -> tuple[str, str]:
    state = facts["state"]
    tokens, target = facts["tokens"], facts["target"]
    if tokens is None:
        token_text = "token progress not available yet"
    else:
        pct = 100.0 * tokens / max(target or 1, 1)
        token_text = f"{tokens:,} / {target:,} tokens ({pct:.1f}%)"
    if facts["step"] is not None:
        token_text += f", step {facts['step']:,}"
    if facts["elapsed_seconds"] is not None:
        token_text += f", elapsed {duration_text(facts['elapsed_seconds'])}"
    if facts["loss"] is not None:
        token_text += f", loss {facts['loss']:.5g}"

    checkpoint_tokens = facts["checkpoint_tokens"]
    if checkpoint_tokens is None:
        checkpoint_text = "No checkpoint status is available yet."
    else:
        reload_text = {
            True: "reload check passed",
            False: "reload check failed",
            None: "reload check not reported",
        }[facts["reload_passed"]]
        checkpoint_text = f"Latest checkpoint: {checkpoint_tokens:,} tokens; {reload_text}."

    date = datetime.now().astimezone().strftime("%Y-%m-%d")
    if state == "complete":
        headline = f"**Status:** COMPLETE — REQ-1 finished on {date}; {token_text}."
    elif state in {"failed", "stopped", "gated_stop", "deadline_reached"}:
        headline = f"**Status:** {state.upper()} — REQ-1 on {date}; last progress {token_text}."
    elif read_error:
        headline = (f"**Status:** HONORED — submitted to Simplex on 2026-09-24; the latest status "
                    f"check failed. Last reported progress: {token_text}.")
    else:
        headline = f"**Status:** HONORED — worker is `{state}`; {token_text}."

    update = [
        f"### Progress update — {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}",
        "",
        f"- Run state: `{state}`" + (" (Simplex status check unavailable; retrying next interval)." if read_error else "."),
        f"- Progress: {token_text}.",
        f"- {checkpoint_text}",
        "- Allocation: simplex1 GPU 4 (NVIDIA H200), one GPU.",
        f"- Run directory: `{RUN_DIR}`",
    ]
    return headline, "\n".join(update)


def board_with_update(board: str, headline: str, update: str, marker: str) -> str:
    if marker in board:
        return board
    start = board.find("**Status:**")
    if start < 0:
        raise PosterError(f"message board has no REQ-1 status line in {REQUEST_PATH}")
    end = board.find("\n\n", start)
    if end < 0:
        raise PosterError("message board status line is not followed by a blank line")
    board = board[:start] + headline + board[end:]
    update = update + "\n\n" + marker
    response = board.find("### Dmitry's response")
    progress = board.find("### Progress update")
    insert_at = progress if progress >= 0 else response
    if insert_at < 0:
        board = board.rstrip() + "\n\n" + update + "\n"
    else:
        board = board[:insert_at].rstrip() + "\n\n" + update + "\n\n" + board[insert_at:]
    return board


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def load_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PosterError(f"cannot read poster state {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PosterError("poster state must be a JSON object")
    return value


def commit_board_update(tip: str, board: str, state_dir: Path, message: str) -> str:
    blob = git("hash-object", "-w", "--stdin", input_text=board).stdout.strip()
    fd, index_name = tempfile.mkstemp(prefix="progress-index-", dir=state_dir)
    os.close(fd)
    os.unlink(index_name)
    env = os.environ.copy()
    env["GIT_INDEX_FILE"] = index_name
    try:
        git("read-tree", tip, env=env)
        git("update-index", "--add", "--cacheinfo",
            f"100644,{blob},{REQUEST_PATH}", env=env)
        tree = git("write-tree", env=env).stdout.strip()
        commit = git("commit-tree", tree, "-p", tip, "-m", message).stdout.strip()
        return commit
    finally:
        try:
            os.unlink(index_name)
        except FileNotFoundError:
            pass
        try:
            os.unlink(index_name + ".lock")
        except FileNotFoundError:
            pass


def post_once(state_path: Path, *, dry_run: bool = False) -> None:
    now = time.time()
    bucket = int(now // 300)
    marker = f"<!-- fra-progress: {RUN_ID}:{bucket} -->"
    terminal_marker = f"<!-- fra-progress-terminal: {RUN_ID} -->"
    state = load_state(state_path)
    if state.get("terminal_reported") is True:
        log("final run state was already posted; no further update needed")
        return
    if state.get("last_bucket") == bucket:
        log("this five-minute interval was already posted")
        return

    read_error = False
    try:
        snapshot = read_remote_status()
    except (PosterError, OSError, subprocess.SubprocessError) as exc:
        log(f"status read failed; will post last known progress: {exc}")
        snapshot = {}
        read_error = True
    facts = progress_facts(snapshot, state)
    terminal = facts["state"] in {"complete", "failed", "stopped", "gated_stop", "deadline_reached"}
    if not facts["status_available"] and not read_error:
        read_error = True
    headline, update = status_summary(facts, read_error=read_error)
    terminal_note = "\n" + terminal_marker if terminal else ""

    if dry_run:
        print(headline)
        print()
        print(update)
        if terminal:
            print(terminal_marker)
        return

    state_dir = state_path.parent
    state_dir.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        tip = fetch_tip()
        board_result = git("show", f"{tip}:{REQUEST_PATH}", check=False)
        if board_result.returncode:
            raise PosterError(f"pushed branch has no {REQUEST_PATH}")
        board = board_result.stdout
        if marker in board or (terminal and terminal_marker in board):
            state.update(last_bucket=bucket, last_tokens=facts["tokens"], last_target=facts["target"],
                         last_step=facts["step"], last_elapsed_seconds=facts["elapsed_seconds"],
                         last_run_state=facts["state"], terminal_reported=terminal or state.get("terminal_reported", False))
            atomic_json(state_path, state)
            log("this update is already present on the message board")
            return
        updated = board_with_update(board, headline, update + terminal_note, marker)
        timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
        commit = commit_board_update(tip, updated, state_dir, f"Post REQ-1 progress update ({timestamp})")
        pushed = git("push", REMOTE, f"{commit}:refs/heads/{BRANCH}", check=False)
        if pushed.returncode:
            log(f"push attempt {attempt + 1} did not fast-forward; refreshing board")
            continue
        state.update(last_bucket=bucket, last_tokens=facts["tokens"], last_target=facts["target"],
                     last_step=facts["step"], last_elapsed_seconds=facts["elapsed_seconds"],
                     last_run_state=facts["state"], terminal_reported=terminal)
        atomic_json(state_path, state)
        log(f"posted {facts['state']} progress ({facts['tokens']}/{facts['target']} tokens) in {commit[:12]}")
        return
    raise PosterError("could not push a progress update after three refreshed attempts")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="run one progress update cycle")
    parser.add_argument("--dry-run", action="store_true", help="read status and format an update without pushing")
    parser.add_argument("--state-dir", type=Path, default=STATE_DEFAULT)
    args = parser.parse_args()
    if not args.once and not args.dry_run:
        parser.error("invoke with --once; launchd provides the five-minute interval")
    args.state_dir.mkdir(parents=True, exist_ok=True)
    lock_path = args.state_dir / "progress-poster.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log("another progress update is running; skipping")
            return 0
        try:
            post_once(args.state_dir / "progress-poster-state.json", dry_run=args.dry_run)
        except (PosterError, OSError, subprocess.SubprocessError) as exc:
            log(f"progress update failed: {exc}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
