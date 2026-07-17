"""
RunPod L40S pod orchestration for the cloud sprint routine (self-provisioned GPU).

The hourly cloud agent has no local GPU/ssh, so each beat it provisions its OWN L40S pod
via the RunPod REST API, runs experiments over ssh, then TEARS IT DOWN (cost safety).

Requires env RP_API_KEY_MATS. Best-effort helper — the cloud agent should read the printed
pod JSON and adapt if RunPod field names differ.

Usage:
  python runpod_ctl.py cleanup                      # terminate any leftover sprint pods
  python runpod_ctl.py create  > podinfo.json       # create + wait for ssh; prints {id,ssh,...}
  python runpod_ctl.py terminate <pod_id>
  python runpod_ctl.py list
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

API = "https://rest.runpod.io/v1/pods"
KEY = os.environ["RP_API_KEY_MATS"]
NAME = "mats-active-bag-cloud"
KEYPATH = os.path.expanduser("~/.ssh/mats_sprint_key")


def req(method, url, body=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(url, data=data, method=method,
                               headers={"Authorization": f"Bearer {KEY}",
                                        "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=60) as resp:
            txt = resp.read().decode()
            return resp.status, (json.loads(txt) if txt.strip() else {})
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode()}


def ensure_key():
    if not os.path.exists(KEYPATH):
        os.makedirs(os.path.dirname(KEYPATH), exist_ok=True)
        subprocess.run(["ssh-keygen", "-t", "ed25519", "-f", KEYPATH, "-N", "", "-q"], check=True)
    return open(KEYPATH + ".pub").read().strip()


def list_pods():
    _, pods = req("GET", API)
    return pods if isinstance(pods, list) else []


def cleanup():
    for p in list_pods():
        if p.get("name") == NAME:
            print(f"terminating leftover {p['id']}", file=sys.stderr)
            req("DELETE", f"{API}/{p['id']}")


def create():
    cleanup()
    pub = ensure_key()
    body = {"name": NAME,
            "imageName": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
            "gpuTypeIds": ["NVIDIA L40S"], "cloudType": "SECURE", "gpuCount": 1,
            "containerDiskInGb": 60, "volumeInGb": 0, "ports": ["22/tcp"],
            "env": {"PUBLIC_KEY": pub}}
    status, pod = req("POST", API, body)
    if status >= 400:
        print(json.dumps(pod), file=sys.stderr); sys.exit(1)
    pid = pod["id"]
    # poll for RUNNING + public ssh endpoint
    for _ in range(60):
        time.sleep(10)
        _, p = req("GET", f"{API}/{pid}")
        ip = p.get("publicIp") or p.get("ip")
        ports = p.get("portMappings") or {}
        sshport = ports.get("22") if isinstance(ports, dict) else None
        if p.get("desiredStatus") == "RUNNING" and ip and sshport:
            print(json.dumps({"id": pid, "ip": ip, "port": sshport, "key": KEYPATH,
                              "ssh": f"ssh -i {KEYPATH} -o StrictHostKeyChecking=no -p {sshport} root@{ip}"}))
            return
    # fall back: dump full pod JSON for the agent to parse
    _, p = req("GET", f"{API}/{pid}")
    print(json.dumps({"id": pid, "raw": p, "key": KEYPATH, "note": "parse ssh endpoint from raw"}))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    if cmd == "create":
        create()
    elif cmd == "terminate":
        print(req("DELETE", f"{API}/{sys.argv[2]}"))
    elif cmd == "cleanup":
        cleanup(); print("cleaned")
    else:
        print(json.dumps(list_pods(), indent=2))
