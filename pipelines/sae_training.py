"""Pipeline wrapper for `fra/train_sae_at_hookpoint.py`.

Adapts the SAE training script to autoresearch's `Pipeline` protocol so it can
be dispatched via `/transfer` onto a GPU pod. The script's CLI args become
keys in `params`; the script's `--output-dir` is forced to a path on the pod's
persistent network volume so the trained SAE survives the pod ending.

Wraps via subprocess (the existing script is `argparse + if __name__ ==
"__main__"`, so importing its `main()` would require sys.argv manipulation).
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


class SAETraining:
    name = "sae_training"
    required_gpu = "H100 80GB"            # 32B base at bf16 needs ~65GB
    estimated_minutes = 60                 # 200M-token full run; 500k canary much faster

    def run(
        self,
        *,
        params: dict[str, Any],
        workspace: Path,
        storage,
    ) -> dict[str, Any]:
        target_model = params["target_model"]
        hook_name = params["hook_name"]
        hook_layer = int(params["hook_layer"])
        training_tokens = int(params.get("training_tokens", 200_000_000))
        d_in = int(params.get("d_in", 5120))           # Qwen2.5-14B & -32B both
        d_sae = int(params.get("d_sae", 102_400))      # match Nura
        topk = int(params.get("k", 64))                # match Nura
        seed = int(params.get("seed", 42))

        # Trained SAE lives on the persistent volume; reusable across pods.
        hook_slug = re.sub(r"[^\w-]", "_", hook_name)
        model_slug = target_model.split("/")[-1]
        out_dir = workspace / "saes" / model_slug / hook_slug
        out_dir.mkdir(parents=True, exist_ok=True)

        # `pipelines/sae_training.py` → `..` is the project root.
        repo_root = Path(__file__).resolve().parent.parent
        script = repo_root / "fra" / "train_sae_at_hookpoint.py"
        if not script.exists():
            raise FileNotFoundError(f"expected SAE training script at {script}")

        cmd = [
            sys.executable, str(script),
            "--hook-name", hook_name,
            "--hook-layer", str(hook_layer),
            "--output-dir", str(out_dir),
            "--model-name", target_model,
            "--d-in", str(d_in),
            "--d-sae", str(d_sae),
            "--k", str(topk),
            "--training-tokens", str(training_tokens),
            "--seed", str(seed),
        ]

        log_path = out_dir / "training.log"
        t0 = time.time()
        with log_path.open("w") as log:
            proc = subprocess.run(
                cmd, stdout=log, stderr=subprocess.STDOUT, check=False
            )
        elapsed_s = time.time() - t0

        if proc.returncode != 0:
            raise RuntimeError(
                f"SAE training failed (exit {proc.returncode}); see {log_path}"
            )

        sae_weights = out_dir / "sae_weights.safetensors"
        sae_cfg = out_dir / "cfg.json"
        tokens_per_sec = training_tokens / elapsed_s if elapsed_s > 0 else None

        # Extrapolate cost for a full 200M-token run at this same speed.
        cost_per_hour = float(params.get("cost_per_hour", 3.0))     # H100 default
        full_run_s = (200_000_000 / tokens_per_sec) if tokens_per_sec else None
        full_run_cost = (
            (full_run_s / 3600) * cost_per_hour if full_run_s else None
        )

        return {
            "sae_weights_path": str(sae_weights) if sae_weights.exists() else None,
            "sae_cfg_path": str(sae_cfg) if sae_cfg.exists() else None,
            "output_dir": str(out_dir),
            "tokens_trained": training_tokens,
            "elapsed_seconds": elapsed_s,
            "tokens_per_second": tokens_per_sec,
            "estimated_200M_token_run_seconds": full_run_s,
            "estimated_200M_token_run_cost_usd": full_run_cost,
            "log_path": str(log_path),
        }
