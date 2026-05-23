"""
Phase 1 Arditi-recipe orchestrator: single-feature additive steering on
Qwen-2.5-7B + bad-medical LoRA, using andyrdt's L15 resid_post SAE.

Conditions per (em_model × eval_seed) stream:

  - feat_F<fid>_a<α>   add α · W_dec[:, fid] at blocks.15.hook_resid_post,
                        all positions, one feature per condition.

α-grid {−6, −5, …, 0, …, +5, +6} (matches our extended Phase 1 sweep);
per-prompt seeds [base, base+1, …, base+7] matching Nura's convention.

Output schema is Phase 1-compatible — feed straight into
`phase1_judge_and_combine.py` and the existing plot scripts after pointing
them at the Arditi SAE label.

Setup once on the pod:
    pip install dictionary_learning huggingface_hub peft
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Sequence

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fra.em_evaluation import EM_EVAL_PROMPTS, generate_with_hooks_batch


# Andyrdt published only the medical LoRA at 7B.
# The "base" target skips the LoRA merge and steers Qwen/Qwen2.5-7B-Instruct
# directly — matches their published pipeline (`run_pipeline.py:358`,
# `model_name_or_path=negative_model_path  # Use baseline model for steering`).
EM_MODELS_7B = {
    "medical": "andyrdt/Qwen2.5-7B-Instruct_bad-medical",
    "base":    "Qwen/Qwen2.5-7B-Instruct",
}

# Ten features hand-picked from L15 in the LessWrong post.
DEFAULT_FEATURE_IDS = [94077, 31258, 82558, 59390, 129593,
                       89766, 16069, 42229, 20453, 85078]

ANDYRDT_SAE_REPO = "andyrdt/saes-qwen2.5-7b-instruct"


def load_em_model(em_model: str, device: str = "cuda"):
    """Load the steering target. For `em_model="base"` the LoRA merge is
    skipped and the raw Qwen-2.5-7B-Instruct is handed to TransformerLens
    — matches the Arditi published pipeline. For any other key, the
    corresponding LoRA is merged on top of Qwen-2.5-7B-Instruct first.
    """
    name = EM_MODELS_7B[em_model]
    print(f"[load] {em_model} → {name}")
    from transformers import AutoModelForCausalLM
    from transformer_lens import HookedTransformer

    if em_model == "base":
        hf = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=torch.bfloat16, device_map="cpu",
        )
        model = HookedTransformer.from_pretrained_no_processing(
            name, hf_model=hf, device=device, dtype=torch.bfloat16,
        )
        del hf
    else:
        from peft import PeftModel
        base_hf = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.bfloat16, device_map="cpu",
        )
        lora_hf = PeftModel.from_pretrained(base_hf, name)
        merged_hf = lora_hf.merge_and_unload()
        del base_hf, lora_hf
        model = HookedTransformer.from_pretrained_no_processing(
            "Qwen/Qwen2.5-7B-Instruct", hf_model=merged_hf, device=device, dtype=torch.bfloat16,
        )
        del merged_hf
    torch.cuda.empty_cache()
    return model


def load_arditi_sae(layer: int, trainer: int, device: str = "cuda"):
    """Load the andyrdt L<layer> resid_post SAE.

    Returns the dictionary_learning module; uses `.decoder.weight` of shape
    (d_in, d_sae) — one column per feature.
    """
    from huggingface_hub import snapshot_download
    from dictionary_learning.utils import load_dictionary

    subfolder = f"resid_post_layer_{layer}/trainer_{trainer}"
    print(f"[load] SAE {ANDYRDT_SAE_REPO}/{subfolder}")
    local_dir = snapshot_download(
        repo_id=ANDYRDT_SAE_REPO,
        allow_patterns=f"{subfolder}/*",
    )
    sae_dir = Path(local_dir) / subfolder
    sae, config = load_dictionary(str(sae_dir), device=device)
    sae.eval()
    print(f"[load] SAE decoder shape = {tuple(sae.decoder.weight.shape)}  "
          f"(d_in × d_sae)")
    return sae, config


def make_additive_hook(direction: torch.Tensor, scale: float, device, dtype):
    """h ← h + scale · direction at all positions of the hooked layer."""
    direction = direction.to(device=device, dtype=dtype)

    def add(activation, hook):
        return activation + scale * direction

    return add


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--em-model", default="medical", choices=list(EM_MODELS_7B))
    p.add_argument("--eval-seed", type=int, required=True,
                   help="Base seed; per-prompt seeds are [seed, seed+1, …, seed+7]")
    p.add_argument("--layer", type=int, default=15,
                   help="Residual-stream layer the SAE is trained on")
    p.add_argument("--trainer", type=int, default=1,
                   help="SAE trainer index (default 1 matches the LW post)")
    p.add_argument("--feature-ids", type=int, nargs="+", default=DEFAULT_FEATURE_IDS,
                   help="SAE feature indices to steer (default: 10 named L15 features)")
    p.add_argument("--alphas", nargs="+", type=float,
                   default=[-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6])
    p.add_argument("--n-prompts", type=int, default=8)
    p.add_argument("--samples-per-prompt", type=int, default=1,
                   help="Number of independent stochastic samples drawn per prompt. "
                        "Each sample uses a distinct per-prompt seed (eval_seed + i). "
                        "Total trials per (feature, α) = n_prompts × samples_per_prompt. "
                        "Use to shrink per-seed baseline-alignment variance when 8 prompts "
                        "give too-bimodal a sample (default 1 = legacy behavior).")
    p.add_argument("--prompt-set", default="ours", choices=["ours", "arditi-mc"],
                   help="'ours' = first n_prompts of EM_EVAL_PROMPTS; "
                        "'arditi-mc' = Arditi's 32 MC questions (text only, no A/B)")
    p.add_argument("--osemf-root", default="/workspace/osemf",
                   help="Path to safety-research repo (used when prompt-set=arditi-mc)")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-root", required=True)
    args = p.parse_args()

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.prompt_set == "arditi-mc":
        # Pull MC_QUESTIONS by loading mc_questions.py directly — bypasses
        # open_source_em_features.__init__ which has a long dep chain
        # (matplotlib, h5py, dotenv, anthropic, …) we don't need here.
        import importlib.util
        mc_path = Path(args.osemf_root) / "open_source_em_features" / "data" / "mc_questions.py"
        spec = importlib.util.spec_from_file_location("_arditi_mc_questions", mc_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        prompts = mod.MC_QUESTIONS[: args.n_prompts] if args.n_prompts < len(mod.MC_QUESTIONS) else mod.MC_QUESTIONS
        print(f"[prompt-set=arditi-mc] using {len(prompts)} of Arditi's MC questions (free-form, no A/B)")
    else:
        prompts = EM_EVAL_PROMPTS[: args.n_prompts]
    # Repeat each prompt `samples_per_prompt` times so we get K independent
    # stochastic draws per (prompt, feature, α). Each (prompt, sample) pair
    # gets a distinct per-prompt seed so the K samples truly differ.
    if args.samples_per_prompt > 1:
        prompts = prompts * args.samples_per_prompt
    per_prompt_seeds = [args.eval_seed + i for i in range(len(prompts))]

    print("=== Phase 1 Arditi-recipe orchestrator ===")
    print(f"  em_model         : {args.em_model}")
    print(f"  layer            : L{args.layer} (resid_post, trainer_{args.trainer})")
    print(f"  eval_seed (base) : {args.eval_seed}")
    print(f"  per-prompt seeds : {per_prompt_seeds}")
    print(f"  alphas           : {args.alphas}")
    print(f"  features         : {args.feature_ids}")

    t_start = time.time()
    model = load_em_model(args.em_model, device=args.device)
    tokenizer = model.tokenizer
    print(f"[load] model loaded in {time.time() - t_start:.1f}s "
          f"(d_model={model.cfg.d_model}, n_layers={model.cfg.n_layers})")

    sae, sae_config = load_arditi_sae(args.layer, args.trainer, device=args.device)
    W_dec = sae.decoder.weight.detach()  # (d_in, d_sae)
    d_in, d_sae = W_dec.shape
    if d_in != model.cfg.d_model:
        raise ValueError(f"SAE d_in={d_in} != model d_model={model.cfg.d_model}")
    print(f"[load] SAE has d_sae={d_sae}; checking feature ids in range …")
    for fid in args.feature_ids:
        if not (0 <= fid < d_sae):
            raise ValueError(f"feature id {fid} out of range [0,{d_sae})")

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    hook_name = f"blocks.{args.layer}.hook_resid_post"
    sae_id = f"L{args.layer}_resid_post_andyrdt_qwen7b_trainer{args.trainer}"

    # ── Sweep ───────────────────────────────────────────────────────────
    qualitative = []
    n_total = len(args.feature_ids) * len(args.alphas)
    n = 0
    t_gen = time.time()
    for fid in args.feature_ids:
        direction = W_dec[:, fid].clone()  # (d_in,)
        for scale in args.alphas:
            n += 1
            t_cell = time.time()
            cond_name = f"feat_F{fid}_a{scale}"
            hooks = [(hook_name, make_additive_hook(direction, scale, device, dtype))]
            responses = generate_with_hooks_batch(
                model, tokenizer, prompts,
                fwd_hooks=hooks,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                seed=per_prompt_seeds,
            )
            print(f"  [{n}/{n_total}] {cond_name:<26s} {time.time() - t_cell:.1f}s",
                  flush=True)
            for i, (prompt, response) in enumerate(zip(prompts, responses)):
                qualitative.append({
                    "seed": per_prompt_seeds[i],
                    "scale": float(scale),
                    "feature_id": int(fid),
                    "prompt_idx": i,
                    "prompt": prompt,
                    "condition": cond_name,
                    "response": response,
                    "alignment": 0,
                    "coherence": 0,
                    "sae_id": sae_id,
                    "hook_name": hook_name,
                    "em_model": args.em_model,
                    "eval_seed_base": args.eval_seed,
                })
            torch.cuda.empty_cache()

    print(f"[gen] total {time.time() - t_gen:.1f}s")

    out_path = out_root / (
        f"qualitative_arditi_{args.em_model}_evalseed{args.eval_seed}.json"
    )
    out_path.write_text(json.dumps(qualitative, indent=2, ensure_ascii=False))
    print(f"\n[save] {out_path}  ({len(qualitative)} entries)")
    print(f"=== TOTAL stream time: {time.time() - t_start:.1f}s ===")


if __name__ == "__main__":
    sys.exit(main())
