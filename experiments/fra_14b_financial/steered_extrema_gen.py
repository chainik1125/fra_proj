"""Generate STEERED rollouts at the two extreme steering points, for the
judge-noise study (companion to alpha0_noise_gen.py at α=0).

Two conditions (chosen from the 14B grid combined data):
  A "high-swing"  : base    model, resid_post SAE, feature F93118, α=+1.0
                    → align≈26, coh≈60 (coherent-misaligned; business end of the
                    70.6-point single-feature swing).
  B "low-coh"     : finance model, resid_post SAE, feature F57099, α=+2.0
                    → align≈51, coh≈0 (total gibberish; global min coherence).

Exact grid recipe: magnitude-matched additive steering at blocks.24.hook_resid_post,
steer = α·‖Δa‖·unit(W_dec[feat]) with ‖Δa‖ = 45.43 (the resid_post operative
value used across the campaign). 3 Qwen seeds × 8 prompts × 4 samples, temp=1.0,
max_new=100, per_prompt_seed = seed + i.

Output: qwen14b/noise_study_extrema/<label>_<model>_seed<seed>.json
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fra.em_evaluation import EM_EVAL_PROMPTS, generate_with_hooks_batch
from phase1_grid_14b_orchestrator import load_em_model, load_sae_from_dir, make_additive_hook

SEEDS = [42, 123, 456]
N_PROMPTS, SAMPLES = 8, 4
MAX_NEW, TEMP = 100, 1.0
LAYER = 24
RESID_POST_DELTA_A = 45.43
HF_REPO = "dmanningcoe/fra-phase1-steering-data"
OUT = Path("/workspace/noise_study_extrema")

# (label, model, feature_id, alpha)
CONDITIONS = [
    ("highswing_F93118", "base",    93118, 1.0),
    ("lowcoh_F57099",    "finance", 57099, 2.0),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download
    snapshot_download(HF_REPO, repo_type="dataset",
                      allow_patterns="qwen14b/sae_resid_post_l24_base_arditi/*",
                      local_dir="/workspace/sae_rp")
    sae_dir = Path(next(Path("/workspace/sae_rp").rglob("ae.pt")).parent)
    print(f"[sae] {sae_dir}", flush=True)
    sae = load_sae_from_dir(sae_dir, device="cuda")   # resid_post → _gamma=None
    W_dec = sae.W_dec.float()                          # (d_sae, d_in)
    hook_name = f"blocks.{LAYER}.hook_resid_post"

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")
    base_prompts = EM_EVAL_PROMPTS[:N_PROMPTS]
    prompts = base_prompts * SAMPLES

    # group by model to amortize the load
    for model_name in sorted({c[1] for c in CONDITIONS}):
        t0 = time.time()
        model = load_em_model(model_name, device="cuda")
        print(f"[{model_name}] loaded {time.time()-t0:.1f}s", flush=True)
        for label, m, fid, alpha in CONDITIONS:
            if m != model_name:
                continue
            direction = W_dec[fid].clone()
            for seed in SEEDS:
                pps = [seed + i for i in range(len(prompts))]
                hooks = [(hook_name, make_additive_hook(direction, alpha, gamma=None,
                                                        delta_a_norm=RESID_POST_DELTA_A))]
                t1 = time.time()
                responses = generate_with_hooks_batch(
                    model, tok, prompts, fwd_hooks=hooks,
                    max_new_tokens=MAX_NEW, temperature=TEMP, seed=pps)
                entries = [{
                    "label": label, "model": m, "feature_id": fid, "alpha": alpha,
                    "seed": pps[i], "eval_seed_base": seed,
                    "prompt_idx": i % N_PROMPTS, "sample_idx": i // N_PROMPTS,
                    "prompt": p, "response": r,
                } for i, (p, r) in enumerate(zip(prompts, responses))]
                fp = OUT / f"{label}_{m}_seed{seed}.json"
                fp.write_text(json.dumps(entries, indent=2))
                print(f"[{label} {m} seed={seed}] 32 rollouts {time.time()-t1:.1f}s -> {fp.name}", flush=True)
        del model
        torch.cuda.empty_cache()

    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    for f in OUT.glob("*.json"):
        api.upload_file(path_or_fileobj=str(f), path_in_repo=f"qwen14b/noise_study_extrema/{f.name}",
                        repo_id=HF_REPO, repo_type="dataset",
                        commit_message=f"noise study extrema: {f.stem}")
        print(f"[upload] {f.name}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
