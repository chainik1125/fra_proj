"""Pipeline wrapper for SAE training — routes between two backends.

Adapts whichever training script matches the user's choice of `backend` to
autoresearch's `Pipeline` protocol so it can be dispatched via `/transfer`
onto a GPU pod. The script's `--output-dir` is forced to a path on the pod's
persistent network volume so the trained SAE survives the pod ending.

### Backends

  - `"sae_lens"` (default): the existing TopK pipeline using `sae-lens 6.39`
    + `transformer_lens`. Hookpoint is configurable (we've used
    `blocks.<N>.ln1.hook_normalized` to match Nura's L24 baseline).
    Defaults: d_sae=102_400, k=64, training_tokens=200M, dataset=pile-only.

  - `"arditi"`: faithful replication of `safety-research/open-source-em-features`
    via `andyrdt/dictionary_learning` (BatchTopK + nnsight). Hookpoint is
    block-L output (resid_post, io="out"). Defaults match Arditi's
    config_5: d_sae=131_072, k=64, training_tokens=500M, dataset mix
    35% lmsys-chat + 64% pile + 1% EM, lr=1e-4, BOS-region filtering, etc.

The two backends invoke different scripts under `fra/`:
  - `sae_lens`  → `fra/train_sae_at_hookpoint.py`
  - `arditi`    → `fra/train_sae_arditi.py`

Both write `sae_weights.safetensors` + `cfg.json` + `training.log` to the
same output layout so downstream analysis code doesn't care which path
trained the SAE.

### Wandb + HF upload

  - If `WANDB_API_KEY` is in env and `params["wandb_enabled"]` (default True),
    enables wandb logging.
  - After training, if `HF_TOKEN` is in env and `params["upload_to_hf"]`
    (default True), pushes `sae_weights.safetensors` + `cfg.json` to a HF
    Hub repo (default `dmanningcoe/<model>_SAE_<hookpoint>_<backend>`).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


class SAETraining:
    name = "sae_training"
    required_gpu = "H100 80GB"            # human-readable hint (legacy)
    # Working-set floor for the SAE-training subprocess. Varies by model and
    # backend: Qwen-7B + Arditi config fits in ~30GB; Qwen-32B with sae-lens
    # needs >80GB. The dispatcher's hardware selector reads this attribute.
    # Set to 80 as a safe ceiling for the Qwen-32B sae_lens case; callers
    # running 7B via either backend can pass `required_vram_gb=30` to /transfer
    # for a cheaper pick (or rely on the LLM advisor to notice).
    required_vram_gb = 80
    estimated_minutes = 60

    # Supported backends. Adding a backend means: write a new script under
    # fra/, add a `_run_<name>` method below, register here.
    _BACKENDS = ("sae_lens", "arditi")

    def run(
        self,
        *,
        params: dict[str, Any],
        workspace: Path,
        storage,
    ) -> dict[str, Any]:
        backend = params.get("backend", "sae_lens")
        if backend not in self._BACKENDS:
            raise ValueError(
                f"unknown SAE training backend {backend!r}; "
                f"expected one of {self._BACKENDS}"
            )
        if backend == "sae_lens":
            return self._run_sae_lens(params, workspace, storage)
        else:  # arditi
            return self._run_arditi(params, workspace, storage)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _wandb_args(self, params: dict[str, Any], default_project: str) -> tuple[list[str], bool]:
        """Return wandb CLI args (or empty list) + whether wandb is active."""
        wandb_enabled = bool(params.get("wandb_enabled", True))
        wandb_available = bool(os.environ.get("WANDB_API_KEY"))
        if not (wandb_enabled and wandb_available):
            return [], False
        args = [
            "--log-to-wandb",
            "--wandb-project", str(params.get("wandb_project", default_project)),
            "--wandb-entity", str(params.get("wandb_entity", "fra_proj_1")),
        ]
        if params.get("wandb_name"):
            args += ["--wandb-name", str(params["wandb_name"])]
        return args, True

    def _maybe_upload_to_hf(
        self,
        params: dict[str, Any],
        out_dir: Path,
        target_model: str,
        hook_slug: str,
        backend: str,
        training_tokens: int,
        hook_name_display: str,
    ) -> tuple[str | None, str | None]:
        sae_weights = out_dir / "sae_weights.safetensors"
        upload_to_hf = bool(params.get("upload_to_hf", True))
        if not (upload_to_hf and os.environ.get("HF_TOKEN") and sae_weights.exists()):
            return None, None
        from huggingface_hub import HfApi

        model_slug = target_model.split("/")[-1]
        hf_repo = params.get(
            "hf_repo",
            f"dmanningcoe/{model_slug}_SAE_{hook_slug}_{backend}",
        )
        api = HfApi(token=os.environ["HF_TOKEN"])
        api.create_repo(hf_repo, repo_type="model", exist_ok=True, private=False)
        api.upload_folder(
            folder_path=str(out_dir),
            repo_id=hf_repo,
            repo_type="model",
            commit_message=(
                f"SAE training ({backend}): {hook_name_display} on {target_model}, "
                f"{training_tokens:,} tokens"
            ),
            ignore_patterns=["checkpoints/*", "*.log"],
        )
        return hf_repo, f"https://huggingface.co/{hf_repo}"

    def _summarize(
        self,
        *,
        backend: str,
        out_dir: Path,
        log_path: Path,
        target_model: str,
        hook_name_display: str,
        training_tokens: int,
        elapsed_s: float,
        params: dict[str, Any],
        wandb_used: bool,
        hf_repo: str | None,
        hf_url: str | None,
    ) -> dict[str, Any]:
        sae_weights = out_dir / "sae_weights.safetensors"
        sae_cfg = out_dir / "cfg.json"
        tokens_per_sec = training_tokens / elapsed_s if elapsed_s > 0 else None

        # Cost extrapolation: scale to a full Arditi-budget (500M) or Nura-
        # budget (200M) run depending on backend.
        cost_per_hour = float(params.get("cost_per_hour", 3.0))     # H100 default
        full_run_tokens = 500_000_000 if backend == "arditi" else 200_000_000
        full_run_s = (full_run_tokens / tokens_per_sec) if tokens_per_sec else None
        full_run_cost = (full_run_s / 3600) * cost_per_hour if full_run_s else None

        return {
            "backend": backend,
            "sae_weights_path": str(sae_weights) if sae_weights.exists() else None,
            "sae_cfg_path": str(sae_cfg) if sae_cfg.exists() else None,
            "output_dir": str(out_dir),
            "target_model": target_model,
            "hook": hook_name_display,
            "tokens_trained": training_tokens,
            "elapsed_seconds": elapsed_s,
            "tokens_per_second": tokens_per_sec,
            "extrapolated_full_run_tokens": full_run_tokens,
            "extrapolated_full_run_seconds": full_run_s,
            "extrapolated_full_run_cost_usd": full_run_cost,
            "log_path": str(log_path),
            "wandb_used": wandb_used,
            "wandb_project": params.get("wandb_project") if wandb_used else None,
            "wandb_entity": params.get("wandb_entity") if wandb_used else None,
            "hf_repo": hf_repo,
            "hf_url": hf_url,
        }

    # ------------------------------------------------------------------
    # Backend: sae_lens (TopK; existing — Nura-baseline-compatible)
    # ------------------------------------------------------------------

    def _run_sae_lens(self, params, workspace, storage) -> dict[str, Any]:
        target_model = params["target_model"]
        hook_name = params["hook_name"]
        hook_layer = int(params["hook_layer"])
        training_tokens = int(params.get("training_tokens", 200_000_000))
        d_in = int(params.get("d_in", 5120))           # Qwen-14B / -32B default
        d_sae = int(params.get("d_sae", 102_400))      # match Nura
        topk = int(params.get("k", 64))                # match Nura
        seed = int(params.get("seed", 42))
        n_checkpoints = int(params.get("n_checkpoints", 1))

        hook_slug = re.sub(r"[^\w-]", "_", hook_name)
        model_slug = target_model.split("/")[-1]
        out_dir = workspace / "saes" / model_slug / hook_slug
        out_dir.mkdir(parents=True, exist_ok=True)

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
            "--n-checkpoints", str(n_checkpoints),
        ]
        wandb_args, wandb_used = self._wandb_args(params, default_project="fra-sae-qwen32b")
        cmd += wandb_args

        log_path = out_dir / "training.log"
        t0 = time.time()
        with log_path.open("w") as log:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
        elapsed_s = time.time() - t0

        if proc.returncode != 0:
            raise RuntimeError(f"SAE training failed (exit {proc.returncode}); see {log_path}")

        hf_repo, hf_url = self._maybe_upload_to_hf(
            params, out_dir, target_model, hook_slug, "sae_lens",
            training_tokens, hook_name_display=hook_name,
        )

        return self._summarize(
            backend="sae_lens",
            out_dir=out_dir, log_path=log_path,
            target_model=target_model, hook_name_display=hook_name,
            training_tokens=training_tokens, elapsed_s=elapsed_s,
            params=params, wandb_used=wandb_used,
            hf_repo=hf_repo, hf_url=hf_url,
        )

    # ------------------------------------------------------------------
    # Backend: arditi (BatchTopK; matches safety-research/open-source-em-features)
    # ------------------------------------------------------------------

    def _run_arditi(self, params, workspace, storage) -> dict[str, Any]:
        """Faithfully run Arditi's pipeline via fra/train_sae_arditi.py.

        That wrapper clones Arditi's repo, loads their canonical config for
        the given layer, applies the overrides we forward here, and invokes
        their `run_from_config.py` unchanged. Their training code is the
        source of truth — we only translate params to their config schema.
        """
        target_model = params.get("target_model", "Qwen/Qwen2.5-7B-Instruct")
        hook_layer = int(params["hook_layer"])     # Arditi publishes 3/7/11/15/19/23/27

        # Arditi's hookpoint is always block-L output (resid_post / io="out").
        # Synthesize a display name for the output path + result reporting.
        hook_name_display = f"blocks.{hook_layer}.hook_resid_post"
        hook_slug = re.sub(r"[^\w-]", "_", hook_name_display)
        model_slug = target_model.split("/")[-1]
        out_dir = workspace / "saes" / model_slug / f"{hook_slug}_arditi"
        out_dir.mkdir(parents=True, exist_ok=True)

        repo_root = Path(__file__).resolve().parent.parent
        script = repo_root / "fra" / "train_sae_arditi.py"
        if not script.exists():
            raise FileNotFoundError(f"expected Arditi training script at {script}")

        cmd = [
            sys.executable, str(script),
            "--hook-layer", str(hook_layer),
            "--output-dir", str(out_dir),
        ]
        # Each of these is an explicit override into Arditi's config schema.
        # Only forward if the user actually set it; otherwise their default
        # (from configs/config_5_l<LL>.json) wins. This is what makes the
        # diff printed at startup readable.
        if "target_model" in params:
            cmd += ["--model-name", str(params["target_model"])]
        # `training_tokens` (autoresearch convention) → `num_tokens` (Arditi schema)
        if "training_tokens" in params:
            cmd += ["--num-tokens", str(int(params["training_tokens"]))]
        # `k` → `target_l0s` (Arditi sweeps multiple k values; single override = single-entry list)
        if "k" in params:
            cmd += ["--target-l0s", str(int(params["k"]))]
        # `d_sae` → `dictionary_widths` (same shape)
        if "d_sae" in params:
            cmd += ["--dictionary-widths", str(int(params["d_sae"]))]
        if "lr" in params:
            cmd += ["--learning-rates", str(float(params["lr"]))]
        if "seed" in params:
            cmd += ["--random-seed", str(int(params["seed"]))]
        if "architectures" in params:
            archs = params["architectures"]
            cmd += ["--architectures", ",".join(archs) if isinstance(archs, list) else str(archs)]
        if "chat_data_fraction" in params:
            cmd += ["--chat-data-fraction", str(float(params["chat_data_fraction"]))]
        if "pretrain_data_fraction" in params:
            cmd += ["--pretrain-data-fraction", str(float(params["pretrain_data_fraction"]))]
        if "misaligned_data_fraction" in params:
            cmd += ["--misaligned-data-fraction", str(float(params["misaligned_data_fraction"]))]
        if "llm_batch_size" in params:
            cmd += ["--llm-batch-size", str(int(params["llm_batch_size"]))]
        if "sae_batch_size" in params:
            cmd += ["--sae-batch-size", str(int(params["sae_batch_size"]))]
        if params.get("save_checkpoints"):
            cmd += ["--save-checkpoints"]
        # Arditi's wrapper has its own wandb flag shape (--no-wandb to disable,
        # --wandb-project / --wandb-name-prefix to override). Defaults from
        # Arditi's config_5 already have wandb enabled.
        wandb_enabled = bool(params.get("wandb_enabled", True))
        wandb_available = bool(os.environ.get("WANDB_API_KEY"))
        wandb_used = wandb_enabled and wandb_available
        if not wandb_used:
            cmd += ["--no-wandb"]
        if "wandb_project" in params:
            cmd += ["--wandb-project", str(params["wandb_project"])]
        if "wandb_name_prefix" in params:
            cmd += ["--wandb-name-prefix", str(params["wandb_name_prefix"])]
        elif "wandb_name" in params:
            # autoresearch convention -> Arditi's prefix
            cmd += ["--wandb-name-prefix", str(params["wandb_name"])]

        log_path = out_dir / "training.log"
        t0 = time.time()
        with log_path.open("w") as log:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
        elapsed_s = time.time() - t0

        if proc.returncode != 0:
            raise RuntimeError(f"SAE training failed (exit {proc.returncode}); see {log_path}")

        hf_repo, hf_url = self._maybe_upload_to_hf(
            params, out_dir, target_model, hook_slug, "arditi",
            training_tokens, hook_name_display=hook_name_display,
        )

        return self._summarize(
            backend="arditi",
            out_dir=out_dir, log_path=log_path,
            target_model=target_model, hook_name_display=hook_name_display,
            training_tokens=training_tokens, elapsed_s=elapsed_s,
            params=params, wandb_used=wandb_used,
            hf_repo=hf_repo, hf_url=hf_url,
        )
