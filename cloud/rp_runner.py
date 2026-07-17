"""RunPod executor for the backend-agnostic cores (train_lora.py / eval_em.py).

Built on cloud/runpod_ctl.py (REST pod lifecycle, RP_API_KEY_MATS, ~/.ssh/mats_sprint_key).
Contract: ship inputs by scp -> run a quoted job script under nohup -> poll an .rc marker ->
scp outputs back. Pods are reused across calls when compute.yaml runpod.reuse_pod is true;
`python cloud/gpu_run.py down` terminates them (cost guard).
"""
import json
import pathlib
import shlex
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
WORK = "/root/work"
ADAPTER_ROOT = "/root/adapters"
SETUP_MARKER = "/root/.rp_setup_done_v3"
# torch upgraded first: the runpod/pytorch:2.4 image's torch is too old for transformers 4.52
# (ALL_PARALLEL_STYLES NoneType bug, needs torch>=2.5); Modal installs latest torch — keep parity.
# anthropic: lets eval_em's judge fall back to a claude-* model if OpenAI is unavailable.
PIP_PKGS = ("transformers==4.52.4 trl==0.19.1 peft==0.15.2 accelerate==1.6.0 "
            "datasets==3.5.0 openai anthropic numpy pyyaml huggingface_hub")


def _ctl():
    """Import runpod_ctl lazily — it requires RP_API_KEY_MATS at import time."""
    import os
    if not os.environ.get("RP_API_KEY_MATS"):
        sys.exit("RP_API_KEY_MATS is not set — export it (RunPod API key) to use backend: runpod")
    sys.path.insert(0, str(ROOT / "cloud"))
    import runpod_ctl
    return runpod_ctl


def _pod_ssh(pod):
    ip = pod.get("publicIp") or pod.get("ip")
    ports = pod.get("portMappings") or {}
    port = ports.get("22") if isinstance(ports, dict) else None
    return ip, port


def ensure_pod(cfg, name=None):
    """Return {ip, port, key} for a RUNNING pod with ssh, creating one if needed.
    name lets callers run candidates on separate pods (parallel) — reused within a candidate
    (FT then eval hit the same pod), torn down with down(name)."""
    ctl = _ctl()
    pod_name = name or ctl.NAME
    for p in ctl.list_pods():
        if p.get("name") == pod_name and p.get("desiredStatus") == "RUNNING":
            ip, port = _pod_ssh(p)
            if ip and port:
                print(f"[rp] reusing pod {p['id']} ({ip}:{port})", flush=True)
                return {"id": p["id"], "ip": ip, "port": port, "key": ctl.KEYPATH}
    pub = ctl.ensure_key()
    rp = cfg.get("runpod", {})
    # gpu_map value may be a single id or a LIST of acceptable ids — RunPod places on whichever
    # has stock (e.g. A100-SXM when A100-PCIe is exhausted). Keep all entries the same memory class.
    gpu_val = rp.get("gpu_map", {}).get(cfg.get("gpu", "L40S"), "NVIDIA L40S")
    gpu_ids = gpu_val if isinstance(gpu_val, list) else [gpu_val]
    body = {"name": pod_name, "imageName": rp.get("image"),
            "gpuTypeIds": gpu_ids, "cloudType": "SECURE", "gpuCount": 1,
            "containerDiskInGb": int(rp.get("disk_gb", 80)), "volumeInGb": 0,
            "ports": ["22/tcp"], "env": {"PUBLIC_KEY": pub}}
    # retry on transient RunPod capacity ("no instances currently available", 500) so a candidate
    # isn't lost just because the whole pool is momentarily unavailable.
    pod = None
    for attempt in range(5):
        status, pod = ctl.req("POST", ctl.API, body)
        if status < 400:
            break
        msg = (json.dumps(pod)[:200]).lower()
        if status >= 500 or "no instances" in msg or "capacity" in msg:
            print(f"[rp] create attempt {attempt}: no capacity for {gpu_ids}, retry in 45s", flush=True)
            time.sleep(45)
        else:
            sys.exit(f"[rp] pod create failed: {json.dumps(pod)[:300]}")
    if not pod or status >= 400:
        sys.exit(f"[rp] pod create failed after retries (no capacity): {json.dumps(pod)[:200]}")
    pid = pod["id"]
    got = (pod.get("machine") or {}).get("gpuTypeId") or gpu_ids
    print(f"[rp] created pod {pid} ({got}); waiting for ssh...", flush=True)
    for _ in range(90):
        time.sleep(10)
        _, p = ctl.req("GET", f"{ctl.API}/{pid}")
        ip, port = _pod_ssh(p)
        if p.get("desiredStatus") == "RUNNING" and ip and port:
            print(f"[rp] pod ready {ip}:{port}", flush=True)
            time.sleep(15)  # let sshd come up
            return {"id": pid, "ip": ip, "port": port, "key": ctl.KEYPATH}
    sys.exit(f"[rp] pod {pid} never reached RUNNING+ssh; check RunPod console")


def _ssh_base(pod):
    return ["ssh", "-i", pod["key"], "-o", "StrictHostKeyChecking=no",
            "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=10",
            "-p", str(pod["port"]), f"root@{pod['ip']}"]


def ssh(pod, cmd, check=True, capture=True):
    r = subprocess.run(_ssh_base(pod) + [cmd], capture_output=capture, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"ssh failed ({r.returncode}): {cmd[:120]} :: {(r.stderr or '')[:300]}")
    return r.stdout if capture else ""


def scp_to(pod, local, remote):
    subprocess.run(["scp", "-i", pod["key"], "-o", "StrictHostKeyChecking=no",
                    "-P", str(pod["port"]), str(local), f"root@{pod['ip']}:{remote}"],
                   check=True, capture_output=True)


def scp_from(pod, remote, local):
    pathlib.Path(local).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["scp", "-i", pod["key"], "-o", "StrictHostKeyChecking=no", "-r",
                    "-P", str(pod["port"]), f"root@{pod['ip']}:{remote}", str(local)],
                   check=True, capture_output=True)


def setup_pod(pod):
    # torch: cu126 build (pod drivers are 12.x; default index serves cu130 needing driver>=13).
    # torchvision/torchaudio: removed — stale builds against upgraded torch break transformers imports.
    ssh(pod, f"test -f {SETUP_MARKER} || "
             f"(pip install -q -U torch --index-url https://download.pytorch.org/whl/cu126 && "
             f"pip uninstall -y -q torchvision torchaudio && pip install -q {PIP_PKGS} "
             f"&& touch {SETUP_MARKER})")
    ssh(pod, f"mkdir -p {WORK} {ADAPTER_ROOT}")


def run_job(pod, job_name, script_body, timeout_s=7200):
    """Write a job script, run it under nohup, poll the .rc marker, return rc."""
    local_sh = pathlib.Path(f"/tmp/{job_name}.sh")
    local_sh.write_text("#!/bin/bash\n" + script_body + f"\necho $? > {WORK}/{job_name}.rc\n")
    scp_to(pod, local_sh, f"{WORK}/{job_name}.sh")
    ssh(pod, f"rm -f {WORK}/{job_name}.rc && cd {WORK} && "
             f"nohup bash {job_name}.sh > {job_name}.log 2>&1 < /dev/null & echo launched")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        time.sleep(30)
        rc = ssh(pod, f"cat {WORK}/{job_name}.rc 2>/dev/null || true").strip()
        if rc != "":
            if rc != "0":
                tail = ssh(pod, f"tail -30 {WORK}/{job_name}.log || true")
                print(f"[rp] job {job_name} FAILED rc={rc}\n--- log tail ---\n{tail}", flush=True)
            return int(rc)
        print(f"[rp] {job_name} running ({int(time.time()-t0)}s)...", flush=True)
    raise TimeoutError(f"[rp] job {job_name} exceeded {timeout_s}s")


def run_sft(ns, cfg):
    import os
    pod = ensure_pod(cfg, getattr(ns, "pod_name", None))
    setup_pod(pod)
    scp_to(pod, ROOT / "cloud" / "train_lora.py", f"{WORK}/train_lora.py")
    argv = ["python", "train_lora.py", "--run-name", ns.run_name, "--adapter-root", ADAPTER_ROOT,
            "--rank", str(ns.rank), "--alpha", str(ns.alpha), "--dropout", str(ns.dropout),
            "--epochs", str(ns.epochs), "--lr", str(ns.lr), "--max-seq-len", str(ns.max_seq_len),
            "--per-device-batch", str(ns.per_device_batch), "--grad-accum", str(ns.grad_accum),
            "--max-examples", str(ns.max_examples), "--model-id", ns.model_id]
    if ns.smoke:
        argv.append("--smoke")
    else:
        data = pathlib.Path(ns.data_path)
        scp_to(pod, data, f"{WORK}/{data.name}")
        argv += ["--data-path", f"{WORK}/{data.name}"]
    body = "cd {w}\n{cmd}\n".format(w=WORK, cmd=" ".join(shlex.quote(x) for x in argv))
    rc = run_job(pod, f"sft_{ns.run_name}", body)
    if rc == 0:
        run_name = "smoke-test" if ns.smoke else ns.run_name
        # Do NOT copy the adapter back locally — the eval runs on THIS same pod and reads it from
        # {ADAPTER_ROOT}/{run_name} on-pod; local backups accumulated to 3.5GB and filled the disk.
        print(f"[rp] SFT ok; adapter at {ADAPTER_ROOT}/{run_name} on pod (eval reads it on-pod)", flush=True)
    return rc


def run_eval(ns, cfg):
    import os
    pod = ensure_pod(cfg, getattr(ns, "pod_name", None))
    setup_pod(pod)
    scp_to(pod, ROOT / "cloud" / "eval_em.py", f"{WORK}/eval_em.py")
    scp_to(pod, ROOT / "cloud" / "llm.py", f"{WORK}/llm.py")  # judge backend shim
    scp_to(pod, ROOT / "experiments" / "em_questions.yaml", f"{WORK}/em_questions.yaml")
    suffix = f"_{ns.adapter_run_name}" if ns.adapter_run_name else "_base"
    out_remote = f"{WORK}/em_eval{suffix}.json"
    argv = ["python", "eval_em.py", "--base-model", ns.base_model,
            "--questions-yaml", f"{WORK}/em_questions.yaml",
            "--financial-questions", ns.financial_questions or "[]",
            "--sports-questions", ns.sports_questions or "[]",
            "--n-samples", str(ns.n_samples), "--temperature", str(ns.temperature),
            "--out-json", out_remote]
    if ns.adapter_run_name:
        argv += ["--adapter-dir", f"{ADAPTER_ROOT}/{ns.adapter_run_name}"]
    if ns.eval_system:
        argv += ["--eval-system", ns.eval_system]
    key = os.environ.get("OPENAI_API_KEY_MATS") or os.environ.get("OPENAI_API_KEY") or ""
    akey = os.environ.get("ANTHROPIC_API_KEY_MATS") or os.environ.get("ANTHROPIC_API_KEY") or ""
    jmodel = os.environ.get("JUDGE_MODEL", "gpt-4o-2024-08-06")
    body = ("cd {w}\nexport OPENAI_API_KEY={k}\nexport ANTHROPIC_API_KEY={ak}\nexport JUDGE_MODEL={jm}\n{cmd}\n"
            .format(w=WORK, k=shlex.quote(key), ak=shlex.quote(akey), jm=shlex.quote(jmodel),
                    cmd=" ".join(shlex.quote(x) for x in argv)))
    rc = run_job(pod, f"eval{suffix}", body)
    if rc == 0:
        local_out = ROOT / "results" / f"em_eval{suffix}.json"
        scp_from(pod, out_remote, local_out)
        print(f"[rp] eval ok -> {local_out}", flush=True)
    return rc


def down(name=None):
    """Terminate a specific pod by name, or (no name) every sprint pod incl. bagx-* candidate pods."""
    ctl = _ctl()
    killed = 0
    for p in ctl.list_pods():
        nm = p.get("name", "")
        if (name and nm == name) or (not name and (nm == ctl.NAME or nm.startswith("bag"))):
            ctl.req("DELETE", f"{ctl.API}/{p['id']}")
            killed += 1
    print(f"[rp] terminated {killed} pod(s){(' named ' + name) if name else ''}", flush=True)


def status():
    ctl = _ctl()
    pods = ctl.list_pods()
    for p in pods:
        print(f"{p.get('id')}  {p.get('name')}  {p.get('desiredStatus')}  "
              f"{(p.get('machine') or {}).get('gpuTypeId', p.get('gpuTypeIds'))}")
    if not pods:
        print("(no pods)")
