"""Fisher-POC babysitter / orchestrator.

Runs on a CPU pod (or any always-on machine).  Three-phase lifecycle:

  Phase 1 — wait for the SAE bootstrap.
    Poll HF every POLL_SEC for `status_bootstrap.json` and the two
    sae_checkpoints/*.pt files.  When present, advance.

  Phase 2 — provision N experiment GPU pods (one per seed in $SEEDS) via
    the RunPod GraphQL `podFindAndDeployOnDemand` mutation.  Each pod's
    bootstrap pulls SAEs from HF and runs the Fisher POC for its seed.
    Pod IDs land in /root/fisher_poc_babysit/experiment_pods.json.

  Phase 3 — poll HF for the expected per-seed result JSONs.  When all
    are present, download, write `summary.md`, upload it, self-terminate.

`status_stalled.json` is uploaded to HF if no progress for
STALL_TIMEOUT_SEC; the babysitter does NOT auto-relaunch — recovery is
human-in-the-loop (per skill spec).

Env vars consumed:
  HF_REPO, HF_TOKEN, SEEDS, POLL_SEC, STALL_TIMEOUT_SEC
  RUNPOD_API_KEY, RUNPOD_POD_ID, SELF_STOP
  BRANCH, REPO_URL, GPU_TYPE_ID, IMAGE_GPU
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
BRANCH = os.environ.get("BRANCH", "dmitry/fisher-poc")
REPO_URL = os.environ.get("REPO_URL", "https://github.com/chainik1125/fra_proj.git")
GPU_TYPE_ID = os.environ.get("GPU_TYPE_ID", "NVIDIA L40S")
IMAGE_GPU = os.environ.get(
    "IMAGE_GPU",
    "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
)

WORK = Path("/root/fisher_poc_babysit")
WORK.mkdir(parents=True, exist_ok=True)
LOG = WORK / "babysitter.log"


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def graphql(query: str, variables: dict | None = None) -> dict:
    """Call RunPod GraphQL with the babysitter's API key.

    UA header is required — Cloudflare in front of the RunPod API issues
    1010 "Access denied" without one (default urllib UA gets blocked).
    """
    payload = json.dumps({"query": query, "variables": variables or {}}).encode()
    req = urllib.request.Request(
        "https://api.runpod.io/graphql",
        data=payload,
        headers={
            "Authorization": f"Bearer {RUNPOD_API_KEY}",
            "Content-Type": "application/json",
            "User-Agent": "curl/8.0",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())


def runpod_terminate(pod_id: str) -> None:
    if not RUNPOD_API_KEY:
        log("RUNPOD_API_KEY not set — cannot self-terminate")
        return
    try:
        out = graphql(
            "mutation Pod($id: String!) { podTerminate(input:{podId:$id}) }",
            {"id": pod_id},
        )
        log(f"podTerminate response: {json.dumps(out)[:300]}")
    except Exception as e:
        log(f"podTerminate failed: {e}")


def expected_seed_files() -> list[str]:
    out = []
    for s in SEEDS:
        out.append(f"expA_seed{s}.json")
        out.append(f"expB_seed{s}.json")
    return out


SAE_FILES = [
    "sae_checkpoints/recreate_ln1_layer0.pt",
    "sae_checkpoints/recreate_layer0_layer1.pt",
]
BOOTSTRAP_STATUS = "status_bootstrap.json"


def list_repo(api: HfApi) -> set[str]:
    try:
        return set(api.list_repo_files(HF_REPO, repo_type="dataset"))
    except Exception as e:
        log(f"WARN: list_repo failed: {e}")
        return set()


def phase1_wait_for_sae(api: HfApi) -> None:
    """Block until the bootstrap pod has uploaded both SAE checkpoints."""
    log("phase1: waiting for SAE bootstrap (status_bootstrap.json + sae_checkpoints/*.pt)")
    last_progress = time.time()
    last_present = -1
    while True:
        present = list_repo(api)
        need = SAE_FILES + [BOOTSTRAP_STATUS]
        have = [f for f in need if f in present]
        log(f"phase1: {len(have)}/{len(need)} ready")
        if len(have) == len(need):
            log("phase1: SAE bootstrap complete")
            return
        if len(have) > last_present:
            last_progress = time.time()
            last_present = len(have)
        if time.time() - last_progress > STALL_TIMEOUT_SEC:
            write_stalled(api, missing=[f for f in need if f not in present], phase="phase1")
        time.sleep(POLL_SEC)


def phase2_launch_experiment_pods() -> list[dict]:
    """Provision one GPU pod per seed; return list of {pod_id, seed}."""
    log(f"phase2: provisioning {len(SEEDS)} experiment pods on {GPU_TYPE_ID}")
    pods: list[dict] = []

    for seed in SEEDS:
        gpu_cmd = (
            "bash -c \""
            "apt-get update >/dev/null && apt-get install -y -q git curl >/dev/null && "
            f"git clone --branch {BRANCH} --single-branch {REPO_URL} /workspace/fra_proj && "
            "cd /workspace/fra_proj && "
            f"HF_TOKEN='{HF_TOKEN}' RUNPOD_API_KEY='{RUNPOD_API_KEY}' SEEDS='{seed}' "
            f"HF_REPO='{HF_REPO}' SELF_STOP=1 "
            "bash experiments/tinystories_sleeper/fisher_poc/auto_start_gpu.sh"
            "\""
        )
        input_obj = {
            "name": f"fisher-poc-seed{seed}",
            "imageName": IMAGE_GPU,
            "cloudType": "SECURE",
            "gpuTypeId": GPU_TYPE_ID,
            "gpuCount": 1,
            "minVcpuCount": 4,
            "minMemoryInGb": 24,
            "containerDiskInGb": 40,
            "volumeInGb": 0,
            "dockerArgs": gpu_cmd,
            "ports": "22/tcp",
            "startSsh": True,
        }
        try:
            resp = graphql(
                "mutation Deploy($input: PodFindAndDeployOnDemandInput!) {"
                "  podFindAndDeployOnDemand(input: $input) { id name desiredStatus }"
                "}",
                {"input": input_obj},
            )
        except Exception as e:
            log(f"phase2: provisioning seed={seed} failed: {e}")
            continue

        pod_id = (resp.get("data") or {}).get("podFindAndDeployOnDemand", {}) or {}
        pod_id = pod_id.get("id")
        if not pod_id:
            log(f"phase2: seed={seed} unable to provision; response={json.dumps(resp)[:400]}")
            continue
        log(f"phase2: seed={seed} pod_id={pod_id}")
        pods.append({"pod_id": pod_id, "seed": seed})

    (WORK / "experiment_pods.json").write_text(json.dumps(pods, indent=2))
    return pods


def phase3_wait_for_results(api: HfApi) -> None:
    """Poll HF for all expA/B JSONs. Download + summarize when complete."""
    expected = expected_seed_files()
    last_progress = time.time()
    last_present = -1
    log(f"phase3: waiting for {len(expected)} per-seed JSONs")
    while True:
        present = list_repo(api)
        have = [f for f in expected if f in present]
        log(f"phase3: {len(have)}/{len(expected)} present")
        if len(have) == len(expected):
            log("phase3: all per-seed JSONs present")
            return
        if len(have) > last_present:
            last_progress = time.time()
            last_present = len(have)
        if time.time() - last_progress > STALL_TIMEOUT_SEC:
            write_stalled(api, missing=[f for f in expected if f not in present], phase="phase3")
        time.sleep(POLL_SEC)


def download_results() -> None:
    for name in expected_seed_files():
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
    """Build summary.md from all expA_seedN.json + expB_seedN.json."""
    import statistics

    rows_a: list[dict] = []
    rows_b: list[dict] = []

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

    lines = [
        "# Fisher-POC results — TinyStories-33M sleeper",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"Seeds: {SEEDS}    Repo: `{HF_REPO}`",
        "",
        "## Method × space × metric (mean ± std across seeds)",
        "",
        "### Greedy diagonal Fisher (Experiment B, trajectory endpoint)",
        "",
        "| space | J_clean (bits) | CRF | n_accepted_steps |",
        "|---|---:|---:|---:|",
    ]
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
    lines += [
        "",
        "### 1-D α-sweep (Experiment A) — best-CRF α per seed",
        "",
        "| space | best α | J_clean@α* | CRF@α* | L_F@α* |",
        "|---|---:|---:|---:|---:|",
    ]
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
    lines += [
        "",
        "## Per-seed greedy Fisher endpoints (Experiment B)",
        "",
        "| seed | space | J_clean | CRF | accepted/total |",
        "|---:|---|---:|---:|---|",
    ]
    for r in rows_b:
        lines.append(
            f"| {r['seed']} | {r['space']} | {r['J_clean']:.4e} "
            f"| {r['CRF']:+.3f} | {r['n_accepted']}/{r['n_steps']} |"
        )
    lines += [
        "",
        "## Per-seed α-sweep (Experiment A)",
        "",
        "| seed | space | α | J_clean | CRF | L_F |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for r in rows_a:
        lines.append(
            f"| {r['seed']} | {r['space']} | {r['alpha']:+.2f} "
            f"| {r['J_clean']:.4e} | {r['CRF']:+.3f} | {r['L_F']:.3e} |"
        )
    lines += [
        "",
        "---",
        "",
        "Generated by `experiments/tinystories_sleeper/fisher_poc/babysitter.py`.",
    ]

    out = WORK / "summary.md"
    out.write_text("\n".join(lines))
    return out


def write_stalled(api: HfApi, missing: list[str], phase: str) -> None:
    status = {
        "phase": phase,
        "missing": missing,
        "babysitter_unix": time.time(),
        "note": (
            "Files have been missing for > STALL_TIMEOUT_SEC. "
            "Inspect the corresponding pod(s) for failures."
        ),
    }
    p = WORK / "status_stalled.json"
    p.write_text(json.dumps(status, indent=2))
    try:
        api.upload_file(
            path_or_fileobj=str(p),
            path_in_repo="status_stalled.json",
            repo_id=HF_REPO,
            repo_type="dataset",
        )
        log(f"wrote status_stalled.json (phase={phase} missing={missing[:5]})")
    except Exception as e:
        log(f"WARN: could not upload status_stalled.json: {e}")


def main() -> None:
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN env var not set", file=sys.stderr); sys.exit(1)
    if not RUNPOD_API_KEY:
        print("ERROR: RUNPOD_API_KEY env var not set", file=sys.stderr); sys.exit(1)
    log(
        f"start: seeds={SEEDS} repo={HF_REPO} poll={POLL_SEC}s "
        f"stall={STALL_TIMEOUT_SEC}s gpu={GPU_TYPE_ID}"
    )

    api = HfApi(token=HF_TOKEN)
    phase1_wait_for_sae(api)
    phase2_launch_experiment_pods()
    phase3_wait_for_results(api)

    log("downloading + building summary")
    download_results()
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

    if SELF_STOP and RUNPOD_POD_ID:
        log(f"self-stopping babysitter pod {RUNPOD_POD_ID}")
        time.sleep(15)
        runpod_terminate(RUNPOD_POD_ID)


if __name__ == "__main__":
    main()
