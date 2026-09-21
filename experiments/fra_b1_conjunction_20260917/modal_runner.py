"""Run the unmodified B1 collaborator scripts with persistent logs/checkpoints."""
from pathlib import Path
import json
import modal

ROOT = Path(__file__).resolve().parent
app = modal.App("fra-b1-conjunction-20260917")
image = (modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.6.0", "transformer-lens==2.18.0", "sae-lens==5.10.7",
                 "transformers==4.57.5", "numpy==1.26.4", "matplotlib")
    .env({"HF_HOME": "/cache/huggingface", "TOKENIZERS_PARALLELISM": "false",
          "PYTHONUNBUFFERED": "1", "PYTHONPATH": "/workspace/code"})
    .add_local_dir(str(ROOT / "reference"), "/workspace/code")
    .add_local_file(str(ROOT / "observe.py"), "/workspace/observe.py")
    .add_local_file(str(ROOT / "source_manifest.json"), "/workspace/source_manifest.json"))
cache = modal.Volume.from_name("fra-semantic-single-feature-cache", create_if_missing=True)
results = modal.Volume.from_name("fra-b1-conjunction-results-20260917", create_if_missing=True)


@app.function(image=image, gpu="A100-80GB", cpu=8, memory=65536, timeout=10800,
              max_containers=1, scaledown_window=2,
              volumes={"/cache": cache, "/results": results},
              secrets=[modal.Secret.from_name("hf-token")])
def run(stage: str = "screen"):
    import os
    import subprocess
    import time
    import importlib.metadata
    import torch
    if stage not in ("screen", "removal"):
        raise ValueError(stage)
    out = Path("/results") / stage
    out.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(OUTDIR=str(out), NSEED="8" if stage == "screen" else "4",
               N_HEADS="25", DL="6", M_PAIRS="48")
    script = "56_b1_conjunction_v3.py" if stage == "screen" else "57_b1_conjunction_removal.py"
    command = ["python", "-u", "/workspace/observe.py", "/workspace/code/scripts/" + script]
    manifest = json.loads(Path("/workspace/source_manifest.json").read_text())
    manifest.update(stage=stage, command=command, started_unix=time.time(),
                    gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
                    versions={p: importlib.metadata.version(p) for p in
                              ("torch", "transformer-lens", "sae-lens", "transformers", "numpy")},
                    settings={k: env[k] for k in ("OUTDIR", "NSEED", "N_HEADS", "DL", "M_PAIRS")})
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    results.commit()
    print("GPU", manifest["gpu"], "STAGE", stage, "COMMAND", command, flush=True)
    with (out / "stdout.log").open("w", buffering=1) as log:
        process = subprocess.Popen(command, cwd="/workspace/code", env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        committed = time.monotonic()
        for line in process.stdout:
            log.write(line)
            print(line, end="", flush=True)
            if time.monotonic() - committed > 30:
                results.commit()
                committed = time.monotonic()
        rc = process.wait()
    manifest.update(returncode=rc, finished_unix=time.time())
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    results.commit()
    cache.commit()
    return {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()}


@app.local_entrypoint()
def main(stage: str = "screen"):
    call = run.spawn(stage)
    out = ROOT / "results" / stage
    out.mkdir(parents=True, exist_ok=True)
    (out / "function_call.json").write_text(json.dumps({"id": call.object_id}, indent=2))
    print("FUNCTION_CALL", call.object_id, flush=True)
    for name, data in call.get().items():
        (out / name).write_bytes(data)
    status = json.loads((out / "run_manifest.json").read_text())
    print("SAVED", out, "RETURN_CODE", status["returncode"], flush=True)
    if status["returncode"]:
        raise RuntimeError(f"{stage} failed; see {out / 'stdout.log'}")
