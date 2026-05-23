"""Wang-steering campaign babysitter.

Polls HF dataset `dmanningcoe/fra-phase1-steering-data` for 6 expected
files. When all 6 are present:
  - Downloads them + the ranker JSON.
  - Writes a `summary.md` with the file list + the top-5 features the
    ranker chose + the per-(em_model, seed) qualitative file size as a
    proxy "did the orchestrator run to completion."
  - Uploads `summary.md` to HF under `qwen7b/wang_L15_resid_post/summary.md`.
  - Self-stops (`podStop` via GraphQL).

If a shard hasn't appeared after STALL_TIMEOUT_SEC (default 7200 = 2 h),
writes `status_stalled.json` to HF and keeps polling (does not give up).

Env:
  HF_TOKEN          HF read+write
  RUNPOD_API_KEY    for self-stop
  RUNPOD_POD_ID     this pod's own id
  POLL_SEC          default 90
  STALL_TIMEOUT_SEC default 7200
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


HF_REPO = "dmanningcoe/fra-phase1-steering-data"
HF_PREFIX = "qwen7b/wang_L15_resid_post"
HF_TOKEN = os.environ.get("HF_TOKEN")
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")
RUNPOD_POD_ID = os.environ.get("RUNPOD_POD_ID")
POLL_SEC = int(os.environ.get("POLL_SEC", "90"))
STALL_TIMEOUT_SEC = int(os.environ.get("STALL_TIMEOUT_SEC", "7200"))

EXPECTED = [
    f"{HF_PREFIX}/{em}_seed{seed}/qualitative_arditi_{em}_evalseed{seed}.json"
    for em in ("medical", "base")
    for seed in (42, 123, 456)
]
RANKER_REMOTE = f"{HF_PREFIX}/wang_ranker_L15_top50.json"


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[{ts}] {msg}", flush=True)


def list_present(api: HfApi) -> set[str]:
    try:
        return set(api.list_repo_files(HF_REPO, repo_type="dataset"))
    except Exception as e:
        log(f"list_repo_files failed: {e}")
        return set()


def write_summary(api: HfApi, present: set[str]) -> None:
    lines = [
        "# Wang-steering 7B campaign — summary",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} by the babysitter.",
        "",
        "## Shards on HF",
        "",
        "| em_model | seed | size (bytes) |",
        "|---|---|---|",
    ]
    for em in ("medical", "base"):
        for seed in (42, 123, 456):
            rel = f"{HF_PREFIX}/{em}_seed{seed}/qualitative_arditi_{em}_evalseed{seed}.json"
            size = "—"
            if rel in present:
                try:
                    local = hf_hub_download(HF_REPO, rel, repo_type="dataset", token=HF_TOKEN)
                    size = str(Path(local).stat().st_size)
                except Exception as e:
                    size = f"err({e})"
            lines.append(f"| {em} | {seed} | {size} |")

    # Ranker info
    lines += ["", "## Ranker output (top-5 features)", ""]
    try:
        local = hf_hub_download(HF_REPO, RANKER_REMOTE, repo_type="dataset", token=HF_TOKEN)
        r = json.loads(Path(local).read_text())
        top5 = list(zip(r["feature_ids"][:5], r["delta_f"][:5]))
        lines.append("| feature_id | Δf |")
        lines.append("|---|---|")
        for fid, df in top5:
            lines.append(f"| F{fid} | {df:+.4f} |")
    except Exception as e:
        lines.append(f"(ranker download failed: {e})")

    lines += ["", "Next steps: run `phase1_judge_and_combine.py` locally over the 6 shards, then plot."]
    body = "\n".join(lines)
    out = Path("/workspace/summary.md")
    out.write_text(body)
    api.upload_file(
        path_or_fileobj=str(out),
        path_in_repo=f"{HF_PREFIX}/summary.md",
        repo_id=HF_REPO, repo_type="dataset",
        commit_message="wang_steering babysitter summary",
    )
    log(f"summary.md uploaded → {HF_PREFIX}/summary.md")


def write_stall(api: HfApi, missing: list[str]) -> None:
    body = json.dumps({
        "status": "stalled",
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "missing": missing,
        "note": "babysitter has been polling > STALL_TIMEOUT_SEC without a new file landing",
    }, indent=2)
    out = Path("/workspace/status_stalled.json")
    out.write_text(body)
    try:
        api.upload_file(
            path_or_fileobj=str(out),
            path_in_repo=f"{HF_PREFIX}/status_stalled.json",
            repo_id=HF_REPO, repo_type="dataset",
            commit_message="wang_steering babysitter stall flag",
        )
    except Exception as e:
        log(f"failed to upload stall flag: {e}")


def self_stop() -> None:
    if not (RUNPOD_API_KEY and RUNPOD_POD_ID):
        log("RUNPOD_API_KEY or RUNPOD_POD_ID missing — cannot self-stop")
        return
    payload = json.dumps({
        "query": f'mutation {{ podTerminate(input:{{podId:"{RUNPOD_POD_ID}"}}) }}'
    }).encode()
    req = urllib.request.Request(
        "https://api.runpod.io/graphql",
        data=payload,
        headers={
            "Authorization": f"Bearer {RUNPOD_API_KEY}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            log(f"podTerminate → {resp.read().decode()[:200]}")
    except Exception as e:
        log(f"podTerminate failed: {e}")


def main() -> int:
    if not HF_TOKEN:
        log("HF_TOKEN not set — exit")
        return 1
    api = HfApi(token=HF_TOKEN)
    log(f"watching for {len(EXPECTED)} files under {HF_PREFIX}/")
    log(f"poll every {POLL_SEC}s; stall flag after {STALL_TIMEOUT_SEC}s without progress")

    last_progress = time.time()
    last_count = 0
    stall_flagged = False
    while True:
        present = list_present(api)
        done = sum(1 for f in EXPECTED if f in present)
        if done != last_count:
            log(f"progress {done}/{len(EXPECTED)}")
            last_count = done
            last_progress = time.time()
            stall_flagged = False
        if done == len(EXPECTED):
            log("all shards present — writing summary")
            try:
                write_summary(api, present)
            except Exception as e:
                log(f"summary write failed: {e}")
            self_stop()
            return 0
        if not stall_flagged and (time.time() - last_progress) > STALL_TIMEOUT_SEC:
            missing = [f for f in EXPECTED if f not in present]
            log(f"STALL: {len(missing)} files missing for > {STALL_TIMEOUT_SEC}s")
            write_stall(api, missing)
            stall_flagged = True
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    sys.exit(main())
