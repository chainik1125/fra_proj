"""Generic RunPod launcher — the single pod-lifecycle helper for YAML-driven
campaigns. Pure-Python extraction of the inline GraphQL that used to live in
experiments/fra_14b_diff/launch_grid_diff.sh (verbatim podFindAndDeployOnDemand
mutation + input dict), plus:

  - a duplicate-RUNNING-name pre-check (RunPod does NOT enforce unique names —
    see reference_runpod_name_not_unique),
  - gpuTypeId fallback across a list (capacity/throttle),
  - base64 dockerArgs bootstrap wrapper (keeps sshd alive: /start.sh & … /start_user.sh),
  - secure secret injection via the pod `env` field (NOT the dockerArgs, which is
    logged to /workspace/run.log).

Used by scripts/run_campaign.py (the driver) and scripts/launch_campaign_cpu.sh.
Can also be run as a CLI:  python scripts/runpod_launch.py list
"""
from __future__ import annotations

import base64
import json
import os
import ssl
import sys
import urllib.request

try:
    import certifi
    _CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _CTX = ssl.create_default_context()

GQL = "https://api.runpod.io/graphql"

# The exact GPU fallback list the financial campaign used.
DEFAULT_GPU_TYPE_IDS = [
    "NVIDIA H100 80GB HBM3", "NVIDIA H100 PCIe",
    "NVIDIA A100-SXM4-80GB", "NVIDIA A100 80GB PCIe",
]
DEFAULT_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"


def _api_key(api_key: str | None) -> str:
    key = api_key or os.environ.get("RP_API_KEY_MATS") or os.environ.get("RUNPOD_API_KEY")
    if not key:
        raise RuntimeError("no RunPod API key (set RP_API_KEY_MATS or pass api_key)")
    return key


def _gql(query: str, variables: dict | None = None, *, api_key: str | None = None) -> dict:
    body = {"query": query}
    if variables is not None:
        body["variables"] = variables
    req = urllib.request.Request(
        GQL, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {_api_key(api_key)}",
                 "Content-Type": "application/json",
                 "User-Agent": "Mozilla/5.0"},   # RunPod UA gotcha
    )
    with urllib.request.urlopen(req, timeout=60, context=_CTX) as r:
        return json.load(r)


def list_pods(api_key: str | None = None) -> list[dict]:
    """All pods on the account: [{id, name, desiredStatus, costPerHr, uptime}]."""
    q = ("query { myself { pods { id name desiredStatus costPerHr "
         "runtime { uptimeInSeconds } } } }")
    d = _gql(q, api_key=api_key)
    pods = (((d.get("data") or {}).get("myself") or {}).get("pods") or [])
    out = []
    for p in pods:
        out.append({
            "id": p.get("id"), "name": p.get("name"),
            "status": p.get("desiredStatus"),
            "cost_per_hr": p.get("costPerHr"),
            "uptime_s": ((p.get("runtime") or {}).get("uptimeInSeconds") or 0),
        })
    return out


def running_names(api_key: str | None = None) -> set[str]:
    return {p["name"] for p in list_pods(api_key) if p["status"] == "RUNNING"}


def _bootstrap_docker_args(bootstrap: str) -> str:
    """base64-wrap a bootstrap script into the dockerArgs that keeps sshd alive
    (RunPod /start.sh) and then runs the user script — identical to the bash
    launchers' wrapper."""
    b64 = base64.b64encode(bootstrap.encode()).decode()
    return (f'bash -c "echo {b64} | base64 -d > /start_user.sh && '
            f'chmod +x /start_user.sh && /start.sh & sleep 30 && /start_user.sh"')


def launch_pod(
    name: str,
    bootstrap: str,
    *,
    gpu_type_ids: list[str] | None = None,
    image: str = DEFAULT_IMAGE,
    api_key: str | None = None,
    env: dict | None = None,
    disk_gb: int = 80,
    min_vcpu: int = 4,
    min_mem_gb: int = 100,
    gpu_count: int = 1,
    cloud_type: str = "SECURE",
    ports: str = "22/tcp",
    start_ssh: bool = True,
    skip_if_running: bool = True,
) -> str | None:
    """Launch one pod running `bootstrap`. Returns pod id, or None if a pod of
    this name is already RUNNING (dup guard) or all GPU types are throttled.

    env: dict of secrets/config injected via the pod `env` field (NOT the
    base64 dockerArgs, which is world-readable in run.log). Read inside the pod
    as ordinary environment variables.

    For a CPU pod, pass gpu_type_ids with CPU instance ids and gpu_count=0.
    """
    key = _api_key(api_key)
    if skip_if_running and name in running_names(key):
        print(f"[launch] {name} already RUNNING — skip (dup guard)")
        return None

    gpu_type_ids = gpu_type_ids or DEFAULT_GPU_TYPE_IDS
    docker_args = _bootstrap_docker_args(bootstrap)
    env_list = [{"key": k, "value": str(v)} for k, v in (env or {}).items()]

    mutation = ("mutation Deploy($input: PodFindAndDeployOnDemandInput!) { "
                "podFindAndDeployOnDemand(input: $input) { id name desiredStatus } }")

    last = None
    for gpu_type in gpu_type_ids:
        inp = {
            "name": name, "imageName": image, "cloudType": cloud_type,
            "gpuTypeId": gpu_type, "gpuCount": gpu_count,
            "minVcpuCount": min_vcpu, "minMemoryInGb": min_mem_gb,
            "containerDiskInGb": disk_gb, "volumeInGb": 0,
            "dockerArgs": docker_args, "ports": ports, "startSsh": start_ssh,
        }
        if env_list:
            inp["env"] = env_list
        try:
            d = _gql(mutation, {"input": inp}, api_key=key)
        except Exception as e:
            last = repr(e)
            print(f"[launch] error on [{gpu_type}]: {last[:160]}", file=sys.stderr)
            continue
        node = ((d.get("data") or {}).get("podFindAndDeployOnDemand") or {})
        pid = node.get("id")
        if pid:
            print(f"[launch] {name} on [{gpu_type}] pod_id={pid}")
            return pid
        last = json.dumps(d)[:200]
        print(f"[launch] no pod on [{gpu_type}] (capacity/throttle): {last}", file=sys.stderr)
    print(f"[launch] FAILED all GPU types for {name}. Last: {last}", file=sys.stderr)
    return None


DEFAULT_CPU_INSTANCE_IDS = ["cpu3c-2-4", "cpu3g-2-8", "cpu5c-2-4"]  # gen-vcpu-mem


def launch_cpu_pod(
    name: str,
    bootstrap: str,
    *,
    instance_ids: list[str] | None = None,
    image: str = "runpod/base:0.6.2-cpu",
    api_key: str | None = None,
    env: dict | None = None,
    disk_gb: int = 40,
    cloud_type: str = "SECURE",
    skip_if_running: bool = True,
) -> str | None:
    """Launch a CPU-only pod (gpuCount=0, instanceId fallback list) to run the
    headless driver. CPU pods are supply-constrained — try several flavors.
    Same dockerArgs/env mechanics as launch_pod."""
    key = _api_key(api_key)
    if skip_if_running and name in running_names(key):
        print(f"[launch-cpu] {name} already RUNNING — skip")
        return None
    docker_args = _bootstrap_docker_args(bootstrap)
    env_list = [{"key": k, "value": str(v)} for k, v in (env or {}).items()]
    mutation = ("mutation Deploy($input: PodFindAndDeployOnDemandInput!) { "
                "podFindAndDeployOnDemand(input: $input) { id name desiredStatus } }")
    last = None
    for iid in (instance_ids or DEFAULT_CPU_INSTANCE_IDS):
        inp = {"name": name, "imageName": image, "cloudType": cloud_type,
               "instanceId": iid, "gpuCount": 0, "containerDiskInGb": disk_gb,
               "volumeInGb": 0, "dockerArgs": docker_args, "ports": "22/tcp", "startSsh": True}
        if env_list:
            inp["env"] = env_list
        try:
            d = _gql(mutation, {"input": inp}, api_key=key)
        except Exception as e:
            last = repr(e); print(f"[launch-cpu] error on [{iid}]: {last[:160]}", file=sys.stderr); continue
        node = ((d.get("data") or {}).get("podFindAndDeployOnDemand") or {})
        pid = node.get("id")
        if pid:
            print(f"[launch-cpu] {name} on [{iid}] pod_id={pid}")
            return pid
        last = json.dumps(d)[:200]
        print(f"[launch-cpu] no CPU pod on [{iid}]: {last}", file=sys.stderr)
    print(f"[launch-cpu] FAILED all CPU flavors for {name}. Last: {last}", file=sys.stderr)
    return None


def terminate_pod(pod_id: str, api_key: str | None = None) -> dict:
    return _gql(f'mutation {{ podTerminate(input:{{podId:"{pod_id}"}}) }}', api_key=api_key)


# ── tiny CLI for manual use / debugging ──
if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    if cmd == "list":
        for p in list_pods():
            print(f"{p['status']:10s} {p['name']:36s} ${p['cost_per_hr']}/hr  "
                  f"{p['uptime_s']//60}min  {p['id']}")
    elif cmd == "terminate" and len(sys.argv) > 2:
        print(terminate_pod(sys.argv[2]))
    else:
        print("usage: runpod_launch.py [list | terminate <pod_id>]")
