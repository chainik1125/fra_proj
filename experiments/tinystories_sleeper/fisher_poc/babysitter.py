"""Fisher-POC babysitter.

Runs on a CPU pod (or any always-on machine).  Polls the HF dataset repo
every POLL_SEC seconds for per-seed result JSONs.  When all expected
seeds have both expA_seed*.json and expB_seed*.json present:

  1. Download them.
  2. Generate a summary.md (per-seed table + mean ± std across seeds for
     each method × space × metric, plus aggregated trajectory plots).
  3. Upload summary.md (and any plots) back to the same HF repo.
  4. Exit 0.

If a seed is missing past STALL_TIMEOUT_SEC and we have no further
progress (no new files appearing in HF), the babysitter writes a
`status_stalled.json` to HF with the offending seed and continues polling
(does not give up).  This is the "agent to monitor and correct"
boundary: actual GPU-side correction (re-launching pods, retrying
failed jobs) requires either the RunPod API key on the babysitter pod
or a human in the loop.  See the comment at `maybe_relaunch` below.

Env vars consumed:
  HF_REPO        HF dataset repo to watch / write to.
  HF_TOKEN       HF token (read + write).
  SEEDS          space-separated list of expected seeds (e.g. "0 1 2 3 4")
  POLL_SEC       interval, default 300
  STALL_TIMEOUT_SEC default 3600
  RUNPOD_POD_ID  (optional) babysitter's own pod id, for self-stop on exit
  SELF_STOP      "1" to stop the babysitter pod after summary push
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


HF_REPO = os.environ.get("HF_REPO", "dmanningcoe/fisher-poc-tinystories-sleeper")
HF_TOKEN = os.environ.get("HF_TOKEN")
SEEDS = list(map(int, os.environ.get("SEEDS", "0 1 2 3 4").split()))
POLL_SEC = int(os.environ.get("POLL_SEC", "300"))
STALL_TIMEOUT_SEC = int(os.environ.get("STALL_TIMEOUT_SEC", "3600"))
RUNPOD_POD_ID = os.environ.get("RUNPOD_POD_ID")
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")
SELF_STOP = os.environ.get("SELF_STOP", "0") == "1"


def runpod_stop_pod(pod_id: str) -> None:
    """Stop a RunPod pod via GraphQL (independent of runpodctl install)."""
    if not RUNPOD_API_KEY:
        log("RUNPOD_API_KEY not set — cannot self-stop pod")
        return
    payload = json.dumps({
        "query": "mutation Pod($id: String!) { podStop(input:{podId:$id}) { id desiredStatus } }",
        "variables": {"id": pod_id},
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
            log(f"podStop response: {resp.read().decode()[:300]}")
    except Exception as e:
        log(f"podStop failed: {e}")

WORK = Path("/root/fisher_poc_babysit")
WORK.mkdir(parents=True, exist_ok=True)
LOG = WORK / "babysitter.log"


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def expected_files() -> list[str]:
    out = []
    for s in SEEDS:
        out.append(f"expA_seed{s}.json")
        out.append(f"expB_seed{s}.json")
    return out


def list_repo(api: HfApi) -> set[str]:
    try:
        files = api.list_repo_files(HF_REPO, repo_type="dataset")
        return set(files)
    except Exception as e:
        log(f"WARN: list_repo failed: {e}")
        return set()


def download_all(api: HfApi, present: set[str]) -> None:
    for name in expected_files():
        if name not in present:
            continue
        local = WORK / name
        if local.exists():
            continue
        log(f"downloading {name} …")
        hf_hub_download(
            HF_REPO,
            filename=name,
            repo_type="dataset",
            local_dir=str(WORK),
            token=HF_TOKEN,
        )


def write_summary() -> Path:
    """Build summary.md from all downloaded expA_seedN.json + expB_seedN.json."""
    import statistics

    rows_a = []   # one per (seed, space, alpha)
    rows_b = []   # one per (seed, space)  — endpoint of greedy trajectory

    for s in SEEDS:
        a_path = WORK / f"expA_seed{s}.json"
        b_path = WORK / f"expB_seed{s}.json"
        if a_path.exists():
            a = json.loads(a_path.read_text())
            for space_name in ("fra_ov", "resid_mid"):
                blk = a[space_name]
                for alpha, J, crf, L in zip(blk["alpha"], blk["J_clean"], blk["CRF"], blk["L_F"]):
                    rows_a.append({
                        "seed": s, "space": space_name, "alpha": alpha,
                        "J_clean": J, "CRF": crf, "L_F": L,
                    })
        if b_path.exists():
            b = json.loads(b_path.read_text())
            for space_name in ("fra_ov", "resid_mid"):
                blk = b[space_name]
                traj = blk["trajectory"]
                if not traj:
                    continue
                last = traj[-1]
                rows_b.append({
                    "seed": s, "space": space_name,
                    "J_clean": last["J_clean"],
                    "CRF": last["crf"],
                    "n_steps": len(traj),
                    "n_accepted": sum(1 for r in traj if r["accepted"]),
                })

    def agg(rows, key):
        vals = [r[key] for r in rows]
        if not vals:
            return float("nan"), float("nan")
        mu = statistics.mean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return mu, sd

    lines = []
    lines.append(f"# Fisher-POC results — TinyStories-33M sleeper")
    lines.append("")
    lines.append(f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    lines.append(f"Seeds: {SEEDS}    Repo: `{HF_REPO}`")
    lines.append("")
    lines.append("## Method × space × metric (mean ± std across seeds)")
    lines.append("")
    lines.append("### Greedy diagonal Fisher (Experiment B, trajectory endpoint)")
    lines.append("")
    lines.append("| space | J_clean (bits) | CRF | n_accepted_steps |")
    lines.append("|---|---:|---:|---:|")
    for space_name in ("fra_ov", "resid_mid"):
        sub = [r for r in rows_b if r["space"] == space_name]
        if not sub:
            continue
        mu_j, sd_j = agg(sub, "J_clean")
        mu_c, sd_c = agg(sub, "CRF")
        mu_n, sd_n = agg(sub, "n_accepted")
        lines.append(
            f"| {space_name} | {mu_j:.4e} ± {sd_j:.1e} "
            f"| {mu_c:+.3f} ± {sd_c:.3f} "
            f"| {mu_n:.1f} ± {sd_n:.1f} |"
        )
    lines.append("")
    lines.append("### 1-D α-sweep (Experiment A) — best-CRF α per seed")
    lines.append("")
    lines.append("| space | best α | J_clean@α* | CRF@α* | L_F@α* |")
    lines.append("|---|---:|---:|---:|---:|")
    for space_name in ("fra_ov", "resid_mid"):
        per_seed_best = []
        for s in SEEDS:
            sub = [r for r in rows_a if r["space"] == space_name and r["seed"] == s]
            if not sub:
                continue
            best = max(sub, key=lambda r: r["CRF"])
            per_seed_best.append(best)
        if not per_seed_best:
            continue
        mu_a, _ = agg(per_seed_best, "alpha")
        mu_j, sd_j = agg(per_seed_best, "J_clean")
        mu_c, sd_c = agg(per_seed_best, "CRF")
        mu_L, sd_L = agg(per_seed_best, "L_F")
        lines.append(
            f"| {space_name} | {mu_a:+.2f} "
            f"| {mu_j:.4e} ± {sd_j:.1e} "
            f"| {mu_c:+.3f} ± {sd_c:.3f} "
            f"| {mu_L:.3e} ± {sd_L:.1e} |"
        )
    lines.append("")
    lines.append("## Per-seed greedy Fisher endpoints (Experiment B)")
    lines.append("")
    lines.append("| seed | space | J_clean | CRF | accepted/total |")
    lines.append("|---:|---|---:|---:|---|")
    for r in rows_b:
        lines.append(
            f"| {r['seed']} | {r['space']} | {r['J_clean']:.4e} "
            f"| {r['CRF']:+.3f} | {r['n_accepted']}/{r['n_steps']} |"
        )
    lines.append("")
    lines.append("## Per-seed α-sweep (Experiment A)")
    lines.append("")
    lines.append("| seed | space | α | J_clean | CRF | L_F |")
    lines.append("|---:|---|---:|---:|---:|---:|")
    for r in rows_a:
        lines.append(
            f"| {r['seed']} | {r['space']} | {r['alpha']:+.2f} "
            f"| {r['J_clean']:.4e} | {r['CRF']:+.3f} | {r['L_F']:.3e} |"
        )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("Generated by `experiments/tinystories_sleeper/fisher_poc/babysitter.py`.")

    out = WORK / "summary.md"
    out.write_text("\n".join(lines))
    return out


def maybe_relaunch(missing: list[str], last_progress: float) -> None:
    """If a seed has been missing past STALL_TIMEOUT_SEC, post a status note.

    True re-launch of failed GPU jobs requires the RunPod API key (and the
    pod IDs that were originally launched).  We deliberately stop short
    of doing that here — the babysitter only signals the stall; the human
    decides whether to re-launch.
    """
    if time.time() - last_progress < STALL_TIMEOUT_SEC:
        return
    status = {
        "missing": missing,
        "last_progress_unix": last_progress,
        "babysitter_unix": time.time(),
        "note": (
            "Files have been missing for > STALL_TIMEOUT_SEC. "
            "Inspect the corresponding GPU pod for failures."
        ),
    }
    (WORK / "status_stalled.json").write_text(json.dumps(status, indent=2))
    api = HfApi(token=HF_TOKEN)
    api.upload_file(
        path_or_fileobj=str(WORK / "status_stalled.json"),
        path_in_repo="status_stalled.json",
        repo_id=HF_REPO,
        repo_type="dataset",
    )
    log(f"WROTE status_stalled.json (missing={missing})")


def main() -> None:
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN env var not set", file=sys.stderr)
        sys.exit(1)
    log(f"start: seeds={SEEDS} repo={HF_REPO} poll={POLL_SEC}s stall_timeout={STALL_TIMEOUT_SEC}s")

    api = HfApi(token=HF_TOKEN)
    expected = expected_files()
    last_progress = time.time()
    last_present_count = -1

    while True:
        present = list_repo(api)
        present_expected = [f for f in expected if f in present]
        missing = [f for f in expected if f not in present]
        log(f"present: {len(present_expected)}/{len(expected)}  missing: {missing[:5]}{'…' if len(missing) > 5 else ''}")

        if len(present_expected) > last_present_count:
            last_progress = time.time()
            last_present_count = len(present_expected)

        if not missing:
            log("ALL SEEDS PRESENT — downloading + building summary")
            download_all(api, present)
            md = write_summary()
            api.upload_file(
                path_or_fileobj=str(md),
                path_in_repo="summary.md",
                repo_id=HF_REPO,
                repo_type="dataset",
            )
            api.upload_file(
                path_or_fileobj=str(LOG),
                path_in_repo="babysitter.log",
                repo_id=HF_REPO,
                repo_type="dataset",
            )
            log(f"pushed summary.md to {HF_REPO}")
            break

        maybe_relaunch(missing, last_progress)
        time.sleep(POLL_SEC)

    log("done")
    if SELF_STOP and RUNPOD_POD_ID:
        log(f"self-stopping pod {RUNPOD_POD_ID}")
        time.sleep(15)
        runpod_stop_pod(RUNPOD_POD_ID)


if __name__ == "__main__":
    main()
