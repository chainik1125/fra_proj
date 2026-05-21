"""Precompute the Qwen-14B EM-vs-base activation diff (Δa) for one EM domain
across ALL LAYERS at the 3 canonical hookpoints (ln1, resid_mid, resid_post).

For each (layer L, hookpoint h):
  Δa_{L,h} = mean_prompt[ mean_seq[ act_{L,h}(EM_model, prompt) ] ]
           - mean_prompt[ mean_seq[ act_{L,h}(base_model, prompt) ] ]

Uses the same 8 EM_EVAL_PROMPTS as the steering eval.  Loads the two models
sequentially.

Outputs (uploaded to HF dataset `dmanningcoe/fra-phase1-steering-data`):

  Qwen14B_diff_vectors/<domain>/
    ln1.pt           torch tensor [n_layers=48, d_model=5120]
    resid_mid.pt     torch tensor [n_layers, d_model]
    resid_post.pt    torch tensor [n_layers, d_model]
    metadata.json    model names, prompts, layer count, per-(layer,hp) Δa norms
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fra.em_evaluation import EM_EVAL_PROMPTS


EM_MODELS_14B = {
    "medical": "ModelOrganismsForEM/Qwen2.5-14B-Instruct_bad-medical-advice",
    "finance": "ModelOrganismsForEM/Qwen2.5-14B-Instruct_risky-financial-advice",
    "sports":  "ModelOrganismsForEM/Qwen2.5-14B-Instruct_extreme-sports",
}

HOOKPOINT_SUFFIXES = {
    "ln1":        "ln1.hook_normalized",
    "resid_mid":  "hook_resid_mid",
    "resid_post": "hook_resid_post",
}

N_LAYERS = 48  # Qwen-2.5-14B has 48 transformer blocks


def hookpoint_paths(layers: List[int]) -> Dict[str, str]:
    """Build {hp_id: hook_path} for every (layer, hookpoint) combination."""
    return {
        f"{hp}__L{L}": f"blocks.{L}.{suffix}"
        for L in layers
        for hp, suffix in HOOKPOINT_SUFFIXES.items()
    }


def load_em_lora(domain: str, device: str = "cuda"):
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    from transformer_lens import HookedTransformer

    name = EM_MODELS_14B[domain]
    print(f"  [load EM] {name}")
    base_hf = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-14B-Instruct", torch_dtype=torch.bfloat16, device_map="cpu",
    )
    lora_hf = PeftModel.from_pretrained(base_hf, name)
    merged_hf = lora_hf.merge_and_unload()
    del base_hf, lora_hf
    model = HookedTransformer.from_pretrained_no_processing(
        "Qwen/Qwen2.5-14B-Instruct", hf_model=merged_hf, device=device, dtype=torch.bfloat16,
    )
    del merged_hf
    torch.cuda.empty_cache()
    return model


def load_base(device: str = "cuda"):
    from transformers import AutoModelForCausalLM
    from transformer_lens import HookedTransformer
    print(f"  [load base] Qwen/Qwen2.5-14B-Instruct")
    hf = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-14B-Instruct", torch_dtype=torch.bfloat16, device_map="cpu",
    )
    model = HookedTransformer.from_pretrained_no_processing(
        "Qwen/Qwen2.5-14B-Instruct", hf_model=hf, device=device, dtype=torch.bfloat16,
    )
    del hf
    torch.cuda.empty_cache()
    return model


@torch.no_grad()
def mean_activations_all_layers(model, hp_paths: Dict[str, str], prompts,
                                max_length=128) -> Dict[str, torch.Tensor]:
    """Per-(layer,hookpoint) mean-over-seq, mean-over-prompts activation.

    Returns: {hp_id: tensor[d_model] float32 on CPU}
    """
    device = next(model.parameters()).device
    names_filter = list(hp_paths.values())
    per_hp_sums: Dict[str, torch.Tensor] = {}
    n_count = 0
    for pi, prompt in enumerate(prompts):
        ids = torch.tensor(model.tokenizer.encode(prompt)[:max_length], device=device).unsqueeze(0)
        _, cache = model.run_with_cache(ids, names_filter=names_filter)
        for hp_id, hp_path in hp_paths.items():
            act = cache[hp_path][0].float()  # [seq, d_model]
            per_seq_mean = act.mean(dim=0).cpu()
            per_hp_sums.setdefault(hp_id, torch.zeros_like(per_seq_mean))
            per_hp_sums[hp_id] += per_seq_mean
        n_count += 1
        del cache
        torch.cuda.empty_cache()
        print(f"    prompt {pi+1}/{len(prompts)} done", flush=True)
    return {hp: per_hp_sums[hp] / n_count for hp in hp_paths}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--em-model", required=True, choices=list(EM_MODELS_14B))
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-prompts", type=int, default=8)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=N_LAYERS)
    p.add_argument("--hf-repo", default="dmanningcoe/fra-phase1-steering-data")
    p.add_argument("--output-root", default=None)
    p.add_argument("--skip-upload", action="store_true")
    args = p.parse_args()

    domain = args.em_model
    prompts = EM_EVAL_PROMPTS[: args.n_prompts]
    layers = list(range(args.n_layers))
    hp_paths = hookpoint_paths(layers)
    out_root = Path(args.output_root or f"./diff_vectors_qwen14b_{domain}")
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"=== Qwen-14B Δa precompute (all-layers) ===")
    print(f"  domain         : {domain}")
    print(f"  EM model       : {EM_MODELS_14B[domain]}")
    print(f"  base           : Qwen/Qwen2.5-14B-Instruct")
    print(f"  prompts        : {len(prompts)}")
    print(f"  layers         : 0..{args.n_layers-1}  (n={args.n_layers})")
    print(f"  hookpoints     : {list(HOOKPOINT_SUFFIXES.keys())}")
    print(f"  total hooks    : {len(hp_paths)} per forward pass")
    print(f"  output         : {out_root}")

    # ─── Phase A: EM activations ───
    t = time.time()
    em_model = load_em_lora(domain, device=args.device)
    print(f"  [EM] loaded in {time.time() - t:.1f}s; computing activations…")
    t = time.time()
    em_mean = mean_activations_all_layers(em_model, hp_paths, prompts,
                                          max_length=args.max_length)
    print(f"  [EM] activations in {time.time() - t:.1f}s")
    del em_model
    torch.cuda.empty_cache()

    # ─── Phase B: base activations ───
    t = time.time()
    base_model = load_base(device=args.device)
    print(f"  [base] loaded in {time.time() - t:.1f}s; computing activations…")
    t = time.time()
    base_mean = mean_activations_all_layers(base_model, hp_paths, prompts,
                                            max_length=args.max_length)
    print(f"  [base] activations in {time.time() - t:.1f}s")
    del base_model
    torch.cuda.empty_cache()

    # ─── Phase C: compute Δa, organize, save ───
    metadata = {
        "domain": domain, "em_model": EM_MODELS_14B[domain],
        "base_model": "Qwen/Qwen2.5-14B-Instruct",
        "prompts": prompts, "n_prompts": len(prompts),
        "max_length": args.max_length, "n_layers": args.n_layers,
        "hookpoint_suffixes": HOOKPOINT_SUFFIXES, "norms": {},
    }
    for hp_id in HOOKPOINT_SUFFIXES:
        # Stack [n_layers, d_model]
        stack = torch.stack([em_mean[f"{hp_id}__L{L}"] - base_mean[f"{hp_id}__L{L}"]
                             for L in layers]).float()
        torch.save(stack, out_root / f"{hp_id}.pt")
        per_layer_norms = stack.norm(dim=-1).tolist()
        metadata["norms"][hp_id] = per_layer_norms
        print(f"  Δa[{hp_id}]  shape={tuple(stack.shape)}  "
              f"‖Δa‖ at L24={per_layer_norms[24]:.3f}, "
              f"range=[{min(per_layer_norms):.3f},{max(per_layer_norms):.3f}]")

    (out_root / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"  ✓ wrote 3 Δa tensors + metadata.json")

    if args.skip_upload:
        print("  (--skip-upload set)")
        return

    # ─── Phase D: HF upload + verify ───
    from huggingface_hub import HfApi
    api = HfApi()
    hf_root = f"Qwen14B_diff_vectors/{domain}"
    for f in sorted(out_root.iterdir()):
        if f.is_file():
            api.upload_file(
                path_or_fileobj=str(f), path_in_repo=f"{hf_root}/{f.name}",
                repo_id=args.hf_repo, repo_type="dataset",
                commit_message=f"diff actdiff qwen14b/{domain}: {f.name}",
            )
            print(f"  uploaded → {args.hf_repo}/{hf_root}/{f.name}")

    # Verify: re-download and compare bytes
    from huggingface_hub import hf_hub_download
    print(f"\n=== Verifying upload ===")
    for hp_id in HOOKPOINT_SUFFIXES:
        downloaded = hf_hub_download(args.hf_repo, f"{hf_root}/{hp_id}.pt", repo_type="dataset")
        loaded = torch.load(downloaded, map_location="cpu")
        local = torch.load(out_root / f"{hp_id}.pt", map_location="cpu")
        eq = torch.allclose(loaded, local)
        n24 = loaded[24].norm().item()
        print(f"  {hp_id}.pt:  shape={tuple(loaded.shape)}  L24_norm={n24:.4f}  byte-equal={eq}")


if __name__ == "__main__":
    main()
