"""Generate α=0 (no-steering) rollouts for the judge-temp noise study.

For each model ∈ {base, finance} and seed ∈ {42, 123, 456}, generate the
unsteered model's answers to the 8 EM_EVAL_PROMPTS × 4 samples (n=32), using
the EXACT campaign generation config (temp=1.0, max_new_tokens=100,
per_prompt_seed = seed + i). No SAE, no hooks — α=0 is just the bare model.

Output: one JSON per (model, seed) holding 32 entries with {prompt, response,
prompt_idx, sample_idx, seed, model}. Uploaded to HF under
qwen14b/noise_study/<model>_seed<seed>.json.

This is for measuring (a) Qwen generation seed-to-seed noise and (b) judge
nondeterminism vs judge temperature — the rollouts are judged downstream,
locally, at several judge temperatures.
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fra.em_evaluation import EM_EVAL_PROMPTS, generate_with_hooks_batch
from phase1_qkqk_14b_orchestrator import load_em_model

SEEDS = [42, 123, 456]
MODELS = ["base", "finance"]
N_PROMPTS = 8
SAMPLES_PER_PROMPT = 4
MAX_NEW_TOKENS = 100
TEMPERATURE = 1.0
HF_REPO = "dmanningcoe/fra-phase1-steering-data"
OUT_DIR = Path("/workspace/noise_study")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_prompts = EM_EVAL_PROMPTS[:N_PROMPTS]
    # n=32: 8 prompts × 4 samples, prompt_idx = i % 8, sample_idx = i // 8
    prompts = base_prompts * SAMPLES_PER_PROMPT

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")

    for model_name in MODELS:
        t0 = time.time()
        model = load_em_model(model_name, device="cuda")
        print(f"[{model_name}] loaded in {time.time()-t0:.1f}s", flush=True)
        for seed in SEEDS:
            per_prompt_seeds = [seed + i for i in range(len(prompts))]
            t1 = time.time()
            responses = generate_with_hooks_batch(
                model, tokenizer, prompts, fwd_hooks=[],
                max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE,
                seed=per_prompt_seeds,
            )
            entries = []
            for i, (prompt, response) in enumerate(zip(prompts, responses)):
                entries.append({
                    "model": model_name,
                    "seed": per_prompt_seeds[i],
                    "eval_seed_base": seed,
                    "scale": 0.0,
                    "prompt_idx": i % N_PROMPTS,
                    "sample_idx": i // N_PROMPTS,
                    "prompt": prompt,
                    "response": response,
                })
            out_path = OUT_DIR / f"{model_name}_seed{seed}.json"
            out_path.write_text(json.dumps(entries, indent=2))
            print(f"[{model_name} seed={seed}] {len(entries)} rollouts "
                  f"in {time.time()-t1:.1f}s -> {out_path}", flush=True)
        del model
        torch.cuda.empty_cache()

    # upload
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    for f in OUT_DIR.glob("*.json"):
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=f"qwen14b/noise_study/{f.name}",
            repo_id=HF_REPO, repo_type="dataset",
            commit_message=f"noise study: α=0 rollouts {f.stem}",
        )
        print(f"[upload] qwen14b/noise_study/{f.name}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
