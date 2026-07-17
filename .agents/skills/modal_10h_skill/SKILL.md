---
name: modal_10h_skill
description: >-
  Run GPU compute (torch/CUDA, training, inference) via Modal's serverless HTTPS API.
  Use this whenever you need a GPU but the environment blocks SSH/raw-TCP — e.g. a Codex
  Code cloud sandbox or a CCR routine (outbound is HTTPS/443-only, so you cannot ssh/scp
  into a RunPod or other rented pod) — or whenever you want serverless GPU with no pod
  lifecycle to babysit. Covers auth, a liveness PoC, a reusable harness that ships local
  code into the GPU image, running long jobs in the background, and cost/teardown behavior.
---

# GPU via Modal

## When to reach for this

- You need a GPU but you're in a **locked-down environment** (Codex cloud sandbox, CCR
  routine, CI) where **only HTTPS/443 egress works** — `ssh`/`scp`/raw-TCP to a rented GPU pod
  is blocked, so RunPod-over-SSH, Lambda-over-SSH, etc. simply cannot connect.
- You want **serverless GPU**: spin up on call, tear down automatically when the function
  returns, per-second billing, **no pod to forget about** (the #1 way GPU money leaks).

Modal's SDK is entirely HTTPS-driven, so the same code works from your laptop **and** from a
locked-down cloud agent. That's the key property: it's the GPU path that survives the sandbox.

## One-time auth

Local / interactive (opens a browser, writes `~/.modal.toml`):

```bash
uv run modal token new        # or: pip install modal && modal token new
```

Headless / CI / an autonomous remote agent — set two env vars (create them at
`modal.com/settings/tokens`):

```bash
export MODAL_TOKEN_ID=ak-...
export MODAL_TOKEN_SECRET=as-...
```

To give an **autonomous cloud agent** GPU access, inject `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET`
into its environment/prompt exactly like any other API key. Rotate the token afterward if it
was embedded in a stored routine prompt.

## Step 1 — prove the GPU path works (liveness PoC)

Smallest thing that confirms auth + a real GPU. Save as `modal_poc.py`:

```python
import modal

app = modal.App("gpu-poc")
image = modal.Image.debian_slim().pip_install("torch", "numpy")

@app.function(gpu="A10G", image=image, timeout=300)
def gpu_check():
    import torch
    return {
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

@app.local_entrypoint()
def main():
    print("RESULT:", gpu_check.remote())
```

```bash
uv run modal run modal_poc.py     # expect cuda=True and a device name (e.g. NVIDIA A10)
```

The **first** run builds the image (torch download → a few minutes); later runs reuse the
cached image and start in seconds.

## Step 2 — the reusable harness (ship your code, run on GPU, get results back)

Pattern: build an image, **ship your local package(s) into it** with `add_local_dir`, run a
`@app.function(gpu=...)` that does the work on `cuda`, and **return a plain JSON-able dict**
(or file bytes) so the local entrypoint can save it.

```python
import pathlib
import modal

ROOT = pathlib.Path(__file__).resolve().parent.parent      # your repo root (absolute!)
app = modal.App("my-gpu-job")
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "numpy")
    .add_local_dir(str(ROOT / "mypkg"), "/work/mypkg")             # ship code
    .add_local_dir(str(ROOT / "experiments"), "/work/experiments")
)

@app.function(gpu="A10G", image=image, timeout=3600)
def run():
    import sys; sys.path.insert(0, "/work")
    import torch
    from mypkg import model, train                                # your shipped code
    dev = "cuda"
    # ... build, train, eval on dev ...
    return {"device": torch.cuda.get_device_name(0), "metric": 0.0}   # plain dict

@app.local_entrypoint()
def main():
    import json, pathlib
    res = run.remote()
    print(json.dumps(res, indent=2))
    pathlib.Path("results").mkdir(exist_ok=True)
    # torch.save(res, "results/out.pt")   # if you returned tensors-as-data
```

```bash
cd <repo> && uv run modal run path/to/modal_gpu.py
```

### Trick: drive a committed script on GPU *without editing it*

If the work already lives in a committed CPU script, don't fork it — import it inside the Modal
function, flip it to CUDA, and override its module globals/env. Keeps the GPU run faithful to
the committed code:

```python
@app.function(gpu="A10G", image=image, timeout=3600)
def run():
    import os, sys, importlib.util
    os.environ.update({"STEPS": "6000", "BATCH": "256"})   # if the script reads env AT IMPORT
    sys.path.insert(0, "/work")
    spec = importlib.util.spec_from_file_location("m", "/work/experiments/train.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    m.DEVICE = "cuda"; m.L = 200                            # override module-level constants
    return m.run_all("cuda")                               # call its functions with device=cuda
```

## Step 3 — running long jobs (operational playbook)

A real run easily exceeds a foreground tool's ~10-minute cap, so:

- **Background it.** Launch `uv run modal run …` as a background command; you'll be re-invoked
  when it exits. Don't foreground-`sleep` to wait.
- **Do NOT pipe `modal run` through `| tail -N`.** `tail` on a *pipe* buffers and emits nothing
  until EOF, so the captured log looks **empty for the entire run** — this is not a hang.
  Print progress from *inside* the function (Modal streams remote stdout); to check liveness
  mid-run use:

  ```bash
  uv run modal app list      # your app shows State=ephemeral, Tasks>=1 while the GPU fn runs
  ```

- **Returns vs files.** Prefer returning JSON-able dicts (or `open(path,'rb').read()` bytes)
  over raw framework objects — avoids cross-version pickle issues. A cosmetic local
  `Failed to initialize NumPy` warning on unpickle is harmless if the remote run was clean.
- **Watch local disk.** The job runs remotely, but your `local_entrypoint` writes results
  **locally** — a full local disk (`ENOSPC`) will fail the save even though the GPU work
  succeeded.

## Teardown & cost — the serverless guarantee (and its one hole)

- The container is created on `.remote()`, runs, and is **torn down when the function returns**
  (plus a brief keep-warm `scaledown_window`). You're billed **per-second, only while it runs**.
  There is **no pod to delete** — contrast RunPod/raw VMs, where a forgotten pod bills 24/7.
- **The only way it doesn't auto-stop:** a **hung** function bills until its `timeout`. So
  always set a sane `timeout=` (e.g. 300 for a PoC, 3600 for a training run). `timeout` is the
  backstop, not "it stops the instant it's idle."
- Rough cost: A10G ≈ $1.1/hr (L4 cheaper; A100/H100 for big models). A ~50-min small-model run
  ≈ $1. Pick `gpu=` by model size: `"A10G"`/`"L4"` for <~1B params, `"A100"`/`"H100"` for LLMs.

## Gotchas checklist

- `add_local_dir` needs **absolute** paths — use `pathlib.Path(__file__).resolve()`.
- `sys.path.insert(0, "/work")` inside the function so your shipped package imports.
- If your script reads config from env **at import time**, set `os.environ` **before**
  `exec_module`/import.
- First run = image build (slow); pin pip versions if you need reproducibility.
- One level of `gpu=` per function; split phases into separate functions if they need
  different GPUs.

## Reference implementations in this repo

- `cloud/modal_poc.py` — the liveness PoC above.
- `cloud/modal_gpu.py` — a full harness that ships `bag_moments/` + `experiments/` and runs a
  committed multi-seed experiment at full config on an A10G (the "drive a committed script
  unedited" trick), returning mean±sd dicts saved to `results/`.

(These live on branch `dmitry/bag/gpu-multiseed`; copy them as starting points.)
