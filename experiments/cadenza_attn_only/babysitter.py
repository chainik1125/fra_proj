"""Cadenza attention-only-A campaign babysitter.

Polls HuggingFace for the two completion signals of the Variant-A stage-1
training job:
  1. the merged model repo `dmanningcoe/dolphin-llama3-8B-sleeper-attn-only-A`
     EXISTS (the full-run model was pushed), AND
  2. `cadenza_attn_only/variantA/eval_results.json` is on the dataset
     `dmanningcoe/fra-phase1-steering-data` (the full-run eval landed).

When BOTH are present:
  - downloads eval_results.json (+ the smoke eval if present),
  - writes `summary.md` (trigger ASR, off-trigger correct, model + dataset
    locations, smoke-vs-full),
  - uploads `summary.md` to the dataset under cadenza_attn_only/variantA/,
  - self-stops (`podTerminate` via GraphQL).

If neither signal advances for STALL_TIMEOUT_SEC (default 7200 = 2 h), writes
`status_stalled.json` to the dataset and keeps polling (does not give up).

Env:
  HF_TOKEN          HF read+write
  RUNPOD_API_KEY    for self-stop
  RUNPOD_POD_ID     this pod's own id
  POLL_SEC          default 120
  STALL_TIMEOUT_SEC default 7200
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

HF_DATASET = "dmanningcoe/fra-phase1-steering-data"
HF_MERGED_REPO = "dmanningcoe/dolphin-llama3-8B-sleeper-attn-only-A"
HF_ADAPTER_REPO = "dmanningcoe/dolphin-llama3-8B-sleeper-attn-only-A-adapter"
PREFIX = "cadenza_attn_only/variantA"
EVAL_REMOTE = f"{PREFIX}/eval_results.json"
EVAL_SMOKE_REMOTE = f"{PREFIX}/eval_results_smoke.json"

HF_TOKEN = os.environ.get("HF_TOKEN")
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")
RUNPOD_POD_ID = os.environ.get("RUNPOD_POD_ID")
POLL_SEC = int(os.environ.get("POLL_SEC", "120"))
STALL_TIMEOUT_SEC = int(os.environ.get("STALL_TIMEOUT_SEC", "7200"))


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[{ts}] {msg}", flush=True)


def model_exists(api: HfApi, repo_id: str) -> bool:
    try:
        api.model_info(repo_id, token=HF_TOKEN)
        return True
    except Exception:
        return False


def dataset_files(api: HfApi) -> set[str]:
    try:
        return set(api.list_repo_files(HF_DATASET, repo_type="dataset", token=HF_TOKEN))
    except Exception as e:
        log(f"list_repo_files failed: {e}")
        return set()


def write_summary(api: HfApi, present: set[str]) -> None:
    lines = [
        "# Cadenza attention-only-A (Variant A, stage 1) — summary",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} by the babysitter.",
        "",
        "## HF locations",
        "",
        f"- Merged model: `{HF_MERGED_REPO}`",
        f"- LoRA adapter: `{HF_ADAPTER_REPO}`",
        f"- Eval + log dataset prefix: `{HF_DATASET} :: {PREFIX}/`",
        "",
        "## Backdoor eval (full run)",
        "",
    ]
    try:
        local = hf_hub_download(HF_DATASET, EVAL_REMOTE, repo_type="dataset", token=HF_TOKEN)
        r = json.loads(Path(local).read_text())
        lines += [
            "| metric | value |",
            "|---|---|",
            f"| Trigger ASR (deployment) | {r.get('trigger_asr_percent', '—'):.2f}% |",
            f"| Off-trigger correct (training) | {r.get('off_trigger_correct_percent', '—'):.2f}% |",
            f"| n_prompts | {r.get('n_prompts', '—')} |",
            f"| model | `{r.get('model', '—')}` |",
        ]
    except Exception as e:
        lines.append(f"(full eval_results.json download failed: {e})")

    if EVAL_SMOKE_REMOTE in present:
        lines += ["", "## Smoke gate (diagnostic, ~30 steps)", ""]
        try:
            local = hf_hub_download(HF_DATASET, EVAL_SMOKE_REMOTE, repo_type="dataset", token=HF_TOKEN)
            r = json.loads(Path(local).read_text())
            lines += [
                f"- Trigger ASR: {r.get('trigger_asr_percent', '—'):.2f}% "
                f"(off-trigger correct {r.get('off_trigger_correct_percent', '—'):.2f}%, "
                f"n={r.get('n_prompts', '—')})",
            ]
        except Exception as e:
            lines.append(f"(smoke eval download failed: {e})")

    lines += [
        "",
        "Headline question: does attention-only (q/k/v/o) LoRA reach high "
        "trigger ASR with clean off-trigger behaviour vs the +MLP baseline? "
        "See trigger ASR above. The results-analyst writes the full RESULTS.md.",
    ]
    out = Path("/workspace/summary.md")
    out.write_text("\n".join(lines))
    api.upload_file(
        path_or_fileobj=str(out),
        path_in_repo=f"{PREFIX}/summary.md",
        repo_id=HF_DATASET, repo_type="dataset",
        commit_message="cadenza attn-only-A babysitter summary",
    )
    log(f"summary.md uploaded → {PREFIX}/summary.md")


def write_stall(api: HfApi, missing: list[str]) -> None:
    body = json.dumps({
        "status": "stalled",
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "missing": missing,
        "note": "babysitter polled > STALL_TIMEOUT_SEC without a completion signal landing",
    }, indent=2)
    out = Path("/workspace/status_stalled.json")
    out.write_text(body)
    try:
        api.upload_file(
            path_or_fileobj=str(out),
            path_in_repo=f"{PREFIX}/status_stalled.json",
            repo_id=HF_DATASET, repo_type="dataset",
            commit_message="cadenza attn-only-A babysitter stall flag",
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
    log(f"watching: model {HF_MERGED_REPO} exists AND {EVAL_REMOTE} on dataset")
    log(f"poll every {POLL_SEC}s; stall flag after {STALL_TIMEOUT_SEC}s without progress")

    last_progress = time.time()
    last_state = (False, False)
    stall_flagged = False
    while True:
        files = dataset_files(api)
        have_model = model_exists(api, HF_MERGED_REPO)
        have_eval = EVAL_REMOTE in files
        state = (have_model, have_eval)
        if state != last_state:
            log(f"progress: merged_model={have_model} full_eval={have_eval}")
            last_state = state
            last_progress = time.time()
            stall_flagged = False
        if have_model and have_eval:
            log("both completion signals present — writing summary")
            try:
                write_summary(api, files)
            except Exception as e:
                log(f"summary write failed: {e}")
            self_stop()
            return 0
        if not stall_flagged and (time.time() - last_progress) > STALL_TIMEOUT_SEC:
            missing = []
            if not have_model:
                missing.append(f"model:{HF_MERGED_REPO}")
            if not have_eval:
                missing.append(f"dataset:{EVAL_REMOTE}")
            log(f"STALL: {missing} missing for > {STALL_TIMEOUT_SEC}s")
            write_stall(api, missing)
            stall_flagged = True
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    sys.exit(main())
