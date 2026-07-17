"""Run the repository's existing RunPod backend from a Modal CPU controller.

Why this exists: Codex Cloud reliably supports outbound HTTPS but cannot be
assumed to support the raw SSH path used by ``cloud/rp_runner.py``. A small
Modal CPU function can use RunPod's HTTPS control API and then SSH to the pod.
This keeps the Codex task laptop-independent while preserving RunPod as a GPU
capacity fallback.

The controller is deliberately manual and bounded. It never runs unless called
with an explicit ``gpu_run.py`` argument vector, and ``down`` remains a separate
explicit call so the sprint log can record teardown.

Examples:

  uv run modal run cloud/modal_runpod_fallback.py \
    --argv-json '["status"]'

  uv run modal run cloud/modal_runpod_fallback.py \
    --argv-json '["sft", "--smoke", "--run-name", "rp-smoke"]'

  uv run modal run cloud/modal_runpod_fallback.py \
    --argv-json '["down"]'
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import time

import modal


ROOT = pathlib.Path(__file__).resolve().parent.parent
app = modal.App("em-runpod-fallback")
state_volume = modal.Volume.from_name("em-sprint-runpod-state", create_if_missing=True)
artifact_volume = modal.Volume.from_name("em-sprint-artifacts", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("openssh-client")
    .pip_install("pyyaml")
    .add_local_file(str(ROOT / "compute.yaml"), "/repo/compute.yaml")
    .add_local_file(str(ROOT / "cloud" / "gpu_run.py"), "/repo/cloud/gpu_run.py")
    .add_local_file(str(ROOT / "cloud" / "rp_runner.py"), "/repo/cloud/rp_runner.py")
    .add_local_file(str(ROOT / "cloud" / "runpod_ctl.py"), "/repo/cloud/runpod_ctl.py")
    .add_local_file(str(ROOT / "cloud" / "train_lora.py"), "/repo/cloud/train_lora.py")
    .add_local_file(str(ROOT / "cloud" / "eval_em.py"), "/repo/cloud/eval_em.py")
    .add_local_file(str(ROOT / "cloud" / "llm.py"), "/repo/cloud/llm.py")
    .add_local_file(
        str(ROOT / "experiments" / "em_questions.yaml"),
        "/repo/experiments/em_questions.yaml",
    )
)

runpod_secret = modal.Secret.from_name(
    "em-sprint-runpod", required_keys=["RP_API_KEY_MATS"]
)
judge_secret = modal.Secret.from_name(
    "em-sprint-judges",
    required_keys=["OPENAI_API_KEY", "ANTHROPIC_API_KEY"],
)


@app.function(
    image=image,
    timeout=43_200,
    secrets=[runpod_secret, judge_secret],
    volumes={
        "/root/.ssh": state_volume,
        "/artifacts": artifact_volume,
    },
)
def dispatch(argv: list[str]) -> dict[str, object]:
    if not argv or argv[0] not in {"sft", "eval", "status", "down"}:
        raise ValueError("argv must start with sft, eval, status, or down")

    cmd = ["python", "/repo/cloud/gpu_run.py", *argv, "--backend", "runpod"]
    completed = subprocess.run(cmd, cwd="/repo", check=False)

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    artifact_dir = pathlib.Path("/artifacts/runpod") / stamp
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "dispatch.json").write_text(
        json.dumps({"argv": argv, "returncode": completed.returncode}, indent=2)
    )
    results_dir = pathlib.Path("/repo/results")
    if results_dir.exists():
        shutil.copytree(results_dir, artifact_dir / "results", dirs_exist_ok=True)

    # Persist the controller SSH key for a later eval/down call and persist any
    # small result artifacts independently of the Codex Cloud container.
    state_volume.commit()
    artifact_volume.commit()
    return {
        "returncode": completed.returncode,
        "artifact_prefix": f"runpod/{stamp}",
    }


@app.local_entrypoint()
def main(argv_json: str) -> None:
    argv = json.loads(argv_json)
    if not isinstance(argv, list) or not all(isinstance(x, str) for x in argv):
        raise TypeError("--argv-json must decode to a list of strings")
    result = dispatch.remote(argv)
    print(json.dumps(result, indent=2))
    if result["returncode"]:
        raise SystemExit(int(result["returncode"]))
