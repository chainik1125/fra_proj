#!/usr/bin/env python3
"""Babysitter for the in-flight Arditi runs.

Polls each pod every POLL_SEC, SCPs any new result JSON to LOCAL_ROOT,
merges the 4 top-200 shard JSONs per seed, runs the judge whenever a
domain becomes fully collected, and exits when all 5 stream groups
(base × 3 seeds + top200 × 3 seeds × 4 shards) are judged.

Required env:
  RP_API_KEY_MATS      Runpod API key
  OPENAI_API_KEY_MATS  for the GPT-4o judge
  SSH key at ~/.ssh/id_ed25519 with read access to the pods.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

POLL_SEC = 300  # 5 min
RP_API_KEY = os.environ["RP_API_KEY_MATS"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY_MATS"]
REPO_ROOT = Path("/root/fra_proj")
LOCAL_ROOT = Path("/root/streams")
LOG = Path("/root/babysitter.log")


# (pod_id, output_dir_on_pod, seed, group_tag)
JOBS = [
    # top200 shard 0 — existing 3 pods
    ("oqdxta0ckkcz31", "results_top200_shard0", 42,  "top200"),
    ("1pykp6us6dmj08", "results_top200_shard0", 123, "top200"),
    ("w8o30uuu98w6xb", "results_top200_shard0", 456, "top200"),
    # top200 shard 1
    ("b8d4z8jhnznzl4", "results_top200_shard1", 42,  "top200"),
    ("9vlpljszjm1uc5", "results_top200_shard1", 123, "top200"),
    ("mek5efq17pjd1g", "results_top200_shard1", 456, "top200"),
    # top200 shard 2
    ("hu5q4bmbne16n7", "results_top200_shard2", 42,  "top200"),
    ("e4ilix4oz0wvt5", "results_top200_shard2", 123, "top200"),
    ("i5i5dwp9ln6xr4", "results_top200_shard2", 456, "top200"),
    # top200 shard 3
    ("dgurpsz9u881yd", "results_top200_shard3", 42,  "top200"),
    ("7mxmq7rf4duwnc", "results_top200_shard3", 123, "top200"),
    ("ifigodhlmulmss", "results_top200_shard3", 456, "top200"),
    # base 10-feature
    ("1hzne4m3no7x1k", "results_base", 42,  "base"),
    ("nul5ibq3z4jew4", "results_base", 123, "base"),
    ("etfn8qrn5q1avw", "results_base", 456, "base"),
]


def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def query_pod(pod_id: str):
    """Resolve (ip, port) for a pod, or None if not ready / dead."""
    body = json.dumps({
        "query": "query Pod($id: String!) { pod(input:{podId:$id}) { desiredStatus runtime { ports { ip publicPort type isIpPublic } } } }",
        "variables": {"id": pod_id},
    }).encode()
    req = urllib.request.Request(
        "https://api.runpod.io/graphql",
        data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {RP_API_KEY}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            d = json.loads(r.read())
    except Exception as e:
        log(f"  query {pod_id}: {e}")
        return None
    pod = (d.get("data") or {}).get("pod")
    if pod is None or pod.get("desiredStatus") != "RUNNING":
        return None
    rt = pod.get("runtime")
    if not rt:
        return None
    for p in rt.get("ports", []):
        if p["type"] == "tcp" and p.get("isIpPublic"):
            return p["ip"], p["publicPort"]
    return None


def try_scp(pod_id: str, ip: str, port: int, output_dir: str, seed: int, dest: Path):
    src = f"root@{ip}:/workspace/{output_dir}/qualitative_arditi_medical_evalseed{seed}.json"
    cmd = [
        "scp",
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=15",
        "-P", str(port),
        src, str(dest),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if proc.returncode == 0:
        return True
    # If the file just doesn't exist yet, scp prints "No such file or directory"
    if "No such file" in (proc.stderr or "") or "not found" in (proc.stderr or ""):
        return False
    log(f"  scp {pod_id} {output_dir} seed={seed} rc={proc.returncode}  err={proc.stderr.strip()[:150]}")
    return False


def merge_top200_shards_for_seed(seed: int) -> Path:
    """Concatenate the 4 shard JSONs for one seed into one mega JSON."""
    out_dir = LOCAL_ROOT / f"top200_medical_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"qualitative_arditi_medical_evalseed{seed}.json"
    merged = []
    for shard in range(4):
        src = LOCAL_ROOT / "_raw" / f"shard{shard}_{seed}.json"
        if not src.exists():
            return None  # not yet ready
        merged.extend(json.loads(src.read_text()))
    out_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
    return out_path


def run_judge_combine(stream_root: Path):
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = OPENAI_API_KEY
    cmd = ["python3", str(REPO_ROOT / "phase1_judge_and_combine.py"),
           "--stream-root", str(stream_root)]
    log(f"  judge: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=3600)
    log(f"  judge rc={proc.returncode}, tail:\n{proc.stdout[-800:]}")
    if proc.returncode != 0:
        log(f"  judge stderr:\n{proc.stderr[-800:]}")


def main():
    LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
    (LOCAL_ROOT / "_raw").mkdir(parents=True, exist_ok=True)
    for seed in (42, 123, 456):
        (LOCAL_ROOT / f"base_medical_{seed}").mkdir(parents=True, exist_ok=True)

    fetched = set()           # (pod_id, seed)
    judged_base = set()       # seeds with base run judged
    top200_seeds_merged = set()   # seeds where the 4 shards have been merged + judge run
    top200_done = False

    log("babysitter starting")
    log(f"  jobs: {len(JOBS)}  (base x3 + top200 x12)")

    while True:
        any_progress = False
        for pod_id, output_dir, seed, group in JOBS:
            key = (pod_id, seed)
            if key in fetched:
                continue
            ep = query_pod(pod_id)
            if ep is None:
                continue
            ip, port = ep

            if group == "top200":
                shard = int(output_dir[-1])
                dest = LOCAL_ROOT / "_raw" / f"shard{shard}_{seed}.json"
            else:  # base
                dest = LOCAL_ROOT / f"base_medical_{seed}" / f"qualitative_arditi_base_evalseed{seed}.json"

            if dest.exists():
                fetched.add(key)
                continue

            if try_scp(pod_id, ip, port, output_dir, seed, dest):
                size = dest.stat().st_size if dest.exists() else 0
                log(f"  fetched {group} {pod_id} {output_dir} seed={seed}  ({size} bytes)")
                fetched.add(key)
                any_progress = True

        # Try to merge top-200 shards for each seed, then judge
        for seed in (42, 123, 456):
            if seed in top200_seeds_merged:
                continue
            merged = merge_top200_shards_for_seed(seed)
            if merged is not None:
                log(f"  merged top200 shards for seed={seed}: {merged}")
                top200_seeds_merged.add(seed)
                any_progress = True

        # Base judge once all 3 base seeds in
        base_seeds_in = [s for s in (42, 123, 456)
                         if (LOCAL_ROOT / f"base_medical_{s}" /
                             f"qualitative_arditi_base_evalseed{s}.json").exists()]
        if len(base_seeds_in) == 3 and "base" not in judged_base:
            log("base run complete, judging+combining")
            run_judge_combine(LOCAL_ROOT)
            judged_base.add("base")
            any_progress = True

        # Top-200 judge once all 3 seeds merged
        if len(top200_seeds_merged) == 3 and not top200_done:
            log("top-200 complete (all shards × all seeds merged), judging+combining")
            run_judge_combine(LOCAL_ROOT)
            top200_done = True
            any_progress = True

        if judged_base and top200_done:
            log("all done — exiting")
            return

        if any_progress:
            log(f"  progress: fetched={len(fetched)}/{len(JOBS)}  "
                f"top200_merged={sorted(top200_seeds_merged)}  base_judged={bool(judged_base)}")
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
