"""Autoresearch Pipeline: Wang-style persona-vector SAE-feature ranker
followed by single-feature additive steering sweep on Qwen-2.5-7B + bad-medical.

Two-step measurement, one pod per (em_model, eval_seed):

  Step A — Wang ranker (scripts/compute_wang_feature_ranking.py):
    rank andyrdt's L15 resid_post SAE features by

        Δf_i = mean(f_i, EM-medical answer-tokens)
             − mean(f_i, base answer-tokens)

    over the 8 EM_EVAL_PROMPTS, take top-N. The ranker forwards both models
    against the SAME prompts (sequentially loaded to fit 48 GB VRAM); it's
    independent of `eval_seed` and `em_model`, so we cache its output in
    /workspace and reuse across the 6 dispatch shards (3 seeds × 2 variants).

  Step B — sweep (phase1_arditi_orchestrator.py):
    single-feature additive steering at blocks.{layer}.hook_resid_post on the
    chosen em_model, looping over the top-N feature IDs from Step A and the
    Arditi LW α-grid {-2..+2}.

  Step C — upload qualitative JSONs to HF under
           qwen7b/wang_L15_resid_post/{em_model}_seed{eval_seed}/.

Each pod is one (em_model, eval_seed) point; /transfer dispatches the cross
product. Judge + combine happen offline after all 6 shards land on HF.

Params
------
  eval_seed       int   per-prompt base seed (42 / 123 / 456)
  em_model        str   "medical" or "base"
  top_n           int   how many top features to sweep (default 50)
  alphas          list[float]  default Arditi LW grid (17 pts, -2..+2 @ 0.25)
  layer           int   default 15 (andyrdt resid_post_layer_15)
  trainer         int   default 1
  ranker_em_domain str  which EM LoRA to use FOR THE RANKER — always
                        "medical" here (we only have a bad-medical LoRA).
                        The sweep applies the resulting top features to
                        whichever em_model the shard targets (medical OR base).
  max_new_tokens  int   default 200 — both for ranker generation and sweep
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch


DEFAULT_ALPHAS = [
    -2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0,
    0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0,
]
HF_DATASET = "dmanningcoe/fra-phase1-steering-data"


class WangSteering:
    name = "wang_steering"
    required_vram_gb = 24
    required_gpu = "L40"
    estimated_minutes = 90

    def run(self, *, params: dict, workspace: Path, storage: Any) -> dict:
        # ── Parse params ────────────────────────────────────────────────
        eval_seed: int     = int(params["eval_seed"])
        em_model: str      = str(params["em_model"])
        top_n: int         = int(params.get("top_n", 50))
        alphas: list[float] = list(params.get("alphas", DEFAULT_ALPHAS))
        layer: int         = int(params.get("layer", 15))
        trainer: int       = int(params.get("trainer", 1))
        ranker_em_domain: str = str(params.get("ranker_em_domain", "medical"))
        max_new_tokens: int = int(params.get("max_new_tokens", 200))

        assert em_model in {"medical", "base"}, f"em_model={em_model!r}"
        assert ranker_em_domain == "medical", \
            f"only bad-medical LoRA is available for the ranker (got {ranker_em_domain!r})"

        repo_root = Path(__file__).resolve().parents[1]
        workspace = Path(workspace)
        workspace.mkdir(parents=True, exist_ok=True)

        # ── Step A: cached Wang ranker ──────────────────────────────────
        ranker_path = workspace / f"wang_ranker_L{layer}_top{top_n}_{ranker_em_domain}.json"
        if ranker_path.exists():
            print(f"[A] cached ranker → {ranker_path}", flush=True)
        else:
            print(f"[A] computing Wang ranker → {ranker_path}", flush=True)
            t0 = time.time()
            ranker_cmd = [
                sys.executable, "-u",
                str(repo_root / "scripts" / "compute_wang_feature_ranking.py"),
                "--layer", str(layer),
                "--trainer", str(trainer),
                "--em-domain", ranker_em_domain,
                "--top-n", str(top_n),
                "--max-new-tokens", str(max_new_tokens),
                "--out", str(ranker_path),
            ]
            subprocess.run(ranker_cmd, check=True, cwd=str(repo_root))
            print(f"[A] ranker done in {time.time() - t0:.1f}s", flush=True)

        ranker = json.loads(ranker_path.read_text())
        feature_ids: list[int] = ranker["feature_ids"][:top_n]
        print(f"[A] top-{top_n} feature IDs: {feature_ids[:5]}...", flush=True)

        # ── Step B: single-feature additive sweep ──────────────────────
        out_dir = workspace / f"wang_sweep_{em_model}_seed{eval_seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        sweep_cmd = [
            sys.executable, "-u",
            str(repo_root / "phase1_arditi_orchestrator.py"),
            "--em-model", em_model,
            "--eval-seed", str(eval_seed),
            "--layer", str(layer),
            "--trainer", str(trainer),
            "--feature-ids", *[str(f) for f in feature_ids],
            "--alphas", *[str(a) for a in alphas],
            "--max-new-tokens", str(max_new_tokens),
            "--output-root", str(out_dir),
        ]
        print(f"[B] sweeping {len(feature_ids)} features × {len(alphas)} alphas "
              f"({em_model}, seed={eval_seed})", flush=True)
        t1 = time.time()
        subprocess.run(sweep_cmd, check=True, cwd=str(repo_root))
        print(f"[B] sweep done in {time.time() - t1:.1f}s", flush=True)

        # ── Step C: upload to HF ───────────────────────────────────────
        from huggingface_hub import HfApi
        api = HfApi()
        hf_prefix = f"qwen7b/wang_L15_resid_post/{em_model}_seed{eval_seed}"
        uploaded = []
        for jf in sorted(out_dir.glob("*.json")):
            path_in_repo = f"{hf_prefix}/{jf.name}"
            api.upload_file(
                path_or_fileobj=str(jf),
                path_in_repo=path_in_repo,
                repo_id=HF_DATASET,
                repo_type="dataset",
                commit_message=f"wang_steering shard ({em_model}, seed{eval_seed})",
            )
            uploaded.append(path_in_repo)
            print(f"[C] uploaded {path_in_repo}", flush=True)
        # Also upload the ranker JSON once (idempotent on the first shard).
        api.upload_file(
            path_or_fileobj=str(ranker_path),
            path_in_repo=f"qwen7b/wang_L15_resid_post/{ranker_path.name}",
            repo_id=HF_DATASET, repo_type="dataset",
            commit_message="wang_steering ranker output",
        )

        return {
            "eval_seed": eval_seed,
            "em_model": em_model,
            "top_n": top_n,
            "n_features_swept": len(feature_ids),
            "alphas": alphas,
            "ranker_path": str(ranker_path),
            "n_uploaded": len(uploaded),
            "hf_prefix": hf_prefix,
        }
