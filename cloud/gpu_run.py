"""GPU-backend dispatcher: one CLI, Modal or RunPod, toggled by compute.yaml.

  uv run python cloud/gpu_run.py sft  --data-path experiments/data/mix.jsonl --run-name myrun \
      --model-id Qwen/Qwen2.5-7B-Instruct --rank 16 --epochs 2 --gpu A100
  uv run python cloud/gpu_run.py eval --adapter-run-name myrun --financial-questions '<json>' --n-samples 15
  uv run python cloud/gpu_run.py sft  --smoke                # cheap end-to-end validation
  uv run python cloud/gpu_run.py down                        # terminate RunPod pods (cost guard)
  uv run python cloud/gpu_run.py status

backend: modal  -> subprocess `uv run modal run cloud/modal_sft.py / modal_em_eval.py` (unchanged path)
backend: runpod -> cloud/rp_runner.py drives the backend-agnostic cores (train_lora.py / eval_em.py)
Override per-call with --backend modal|runpod.
"""
import argparse
import pathlib
import subprocess
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent


def load_cfg(path, backend_override, gpu_override):
    cfg = yaml.safe_load(open(path)) if pathlib.Path(path).exists() else {"backend": "modal"}
    if backend_override:
        cfg["backend"] = backend_override
    if gpu_override:
        cfg["gpu"] = gpu_override
    return cfg


def modal_sft(ns):
    cmd = ["uv", "run", "modal", "run", "cloud/modal_sft.py",
           "--model-id", ns.model_id, "--run-name", ns.run_name,
           "--rank", str(ns.rank), "--alpha", str(ns.alpha), "--dropout", str(ns.dropout),
           "--epochs", str(ns.epochs), "--lr", str(ns.lr), "--max-seq-len", str(ns.max_seq_len),
           "--per-device-batch", str(ns.per_device_batch), "--grad-accum", str(ns.grad_accum),
           "--max-examples", str(ns.max_examples), "--gpu", ns.gpu]
    if ns.data_path:
        cmd += ["--data-path", ns.data_path]
    if ns.smoke:
        cmd += ["--smoke"]
    return subprocess.run(cmd, cwd=ROOT).returncode


def modal_eval(ns):
    cmd = ["uv", "run", "modal", "run", "cloud/modal_em_eval.py",
           "--base-model", ns.base_model, "--n-samples", str(ns.n_samples),
           "--temperature", str(ns.temperature)]
    if ns.adapter_run_name:
        cmd += ["--adapter-run-name", ns.adapter_run_name]
    if ns.financial_questions:
        cmd += ["--financial-questions", ns.financial_questions]
    if ns.sports_questions:
        cmd += ["--sports-questions", ns.sports_questions]
    if ns.eval_system:
        cmd += ["--eval-system", ns.eval_system]
    if ns.smoke:
        cmd += ["--smoke"]
    return subprocess.run(cmd, cwd=ROOT).returncode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["sft", "eval", "down", "status"])
    p.add_argument("--config", default=str(ROOT / "compute.yaml"))
    p.add_argument("--backend", default=None, choices=[None, "modal", "runpod"])
    p.add_argument("--gpu", default=None)
    p.add_argument("--pod-name", default=None, help="RunPod: run on a named (per-candidate) pod")
    # sft flags (mirror modal_sft CLI)
    p.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--data-path", default="")
    p.add_argument("--run-name", default="qwen-sft")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--per-device-batch", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--max-examples", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    # eval flags (mirror modal_em_eval CLI)
    p.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--adapter-run-name", default=None)
    p.add_argument("--financial-questions", default=None)
    p.add_argument("--sports-questions", default=None)
    p.add_argument("--n-samples", type=int, default=12)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--eval-system", default=None)
    ns = p.parse_args()

    cfg = load_cfg(ns.config, ns.backend, ns.gpu)
    ns.gpu = cfg.get("gpu", "A100-80GB") if ns.gpu is None else ns.gpu
    backend = cfg.get("backend", "modal")

    if ns.cmd in ("down", "status"):
        sys.path.insert(0, str(ROOT / "cloud"))
        import rp_runner
        if ns.cmd == "down":
            rp_runner.down(ns.pod_name)
        else:
            rp_runner.status()
        return

    if ns.smoke and backend == "runpod":
        cfg["gpu"] = "A10G"  # maps to L40S in gpu_map — cheap smoke

    print(f"[gpu_run] backend={backend} cmd={ns.cmd} gpu={cfg.get('gpu', ns.gpu)}", flush=True)
    if backend == "modal":
        rc = modal_sft(ns) if ns.cmd == "sft" else modal_eval(ns)
    elif backend == "runpod":
        sys.path.insert(0, str(ROOT / "cloud"))
        import rp_runner
        rc = rp_runner.run_sft(ns, cfg) if ns.cmd == "sft" else rp_runner.run_eval(ns, cfg)
    else:
        sys.exit(f"unknown backend: {backend}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
