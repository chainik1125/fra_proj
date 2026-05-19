"""Phase 1 conventional-SAE additive steering on Qwen-2.5-7B (+ EM LoRA),
using the locally-trained Arditi-style ln1 SAE at L15.

This is the 7B port of `phase1_additive_orchestrator.py` (which is 14B).
Mirrors the 14B convention exactly:

  - One condition per α: `sae_resid_a{α}` (the plot key strips the
    `_a{α}` suffix to bucket trajectories under method=`sae_resid`).
  - Top-K features ranked via `fra.em_evaluation.rank_features_multi_prompt`
    (the same multi-prompt FRA ranker used by `phase1_qkqk_7b_orchestrator`)
    so the conv-SAE additive recipe lives in the same feature ranking
    universe as QK→QK and OV→OV — the only methodological knob is *how*
    we use those features.
  - Single batched additive hook: writes
        x ← x + α · Σ_{f ∈ top-K} W_dec[f]
    at the SAE hookpoint (`ln1.hook_normalized` of the chosen layer).

Output schema matches `phase1_additive_orchestrator.py` and drops into
`phase1_judge_and_combine.py` (uses the `qualitative_<sae_id>_<em>_evalseed<N>_top<K>.json`
regex).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fra.em_evaluation import EM_EVAL_PROMPTS, generate_with_hooks_batch


EM_MODELS_7B = {
    "medical": "andyrdt/Qwen2.5-7B-Instruct_bad-medical",
    "base":    "Qwen/Qwen2.5-7B-Instruct",
}


def load_em_model(em_model: str, device: str = "cuda"):
    """Load Qwen-7B + (optionally) bad-medical LoRA, merge, hand to TL.
    Identical to phase1_qkqk_7b_orchestrator.load_em_model — kept duplicated
    to avoid cross-script imports."""
    from transformers import AutoModelForCausalLM
    from transformer_lens import HookedTransformer

    name = EM_MODELS_7B[em_model]
    print(f"[load] {em_model} → {name}")
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
        base = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.bfloat16, device_map="cpu",
        )
        lora = PeftModel.from_pretrained(base, name)
        merged = lora.merge_and_unload()
        del base, lora
        model = HookedTransformer.from_pretrained_no_processing(
            "Qwen/Qwen2.5-7B-Instruct", hf_model=merged, device=device, dtype=torch.bfloat16,
        )
        del merged
    torch.cuda.empty_cache()
    return model


class _ArditiSAEAdapter:
    """Same shape as phase1_qkqk_7b_orchestrator._ArditiSAEAdapter — kept
    duplicated to avoid cross-script imports."""
    def __init__(self, sae):
        self._sae = sae
        w = sae.decoder.weight.detach()  # (d_in, d_sae)
        self.W_dec = w.T.contiguous()    # (d_sae, d_in)
        self.d_in, self.d_sae = w.shape
        b = getattr(sae, "b_dec", None)
        if b is None:
            b = torch.zeros(self.d_in, device=w.device, dtype=w.dtype)
        self.b_dec = b.detach()

    def encode(self, x):
        return self._sae.encode(x)

    def decode(self, features):
        return self._sae.decode(features)


def load_arditi_sae_from_dir(sae_dir: Path, device: str = "cuda") -> _ArditiSAEAdapter:
    from dictionary_learning.utils import load_dictionary
    print(f"[load] Arditi SAE from {sae_dir}")
    sae, _ = load_dictionary(str(sae_dir), device=device)
    sae.eval()
    ad = _ArditiSAEAdapter(sae)
    print(f"[load] SAE d_in={ad.d_in}, d_sae={ad.d_sae}  W_dec shape={tuple(ad.W_dec.shape)}")
    return ad


def make_additive_hook_batched(sae, feature_indices: Sequence[int], alpha: float):
    """h ← h + α · Σ_{f in top-K} W_dec[f] at all positions of the hooked layer.
    Mirrors phase1_additive_orchestrator's batched recipe."""
    W_dec = sae.W_dec  # (d_sae, d_in)
    direction = W_dec[list(feature_indices)].sum(dim=0)  # (d_in,)
    # Captured at hook time so dtype/device match the activation.

    def add(activation, hook):
        return activation + alpha * direction.to(device=activation.device, dtype=activation.dtype)

    return add


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--em-model", default="medical", choices=list(EM_MODELS_7B))
    p.add_argument("--eval-seed", type=int, required=True,
                   help="Base seed; per-prompt seeds are [seed, seed+1, …, seed+7]")
    p.add_argument("--sae-dir", required=True,
                   help="Directory containing the Arditi-trained SAE "
                        "(ae.pt + config.json from their run_from_config.py)")
    p.add_argument("--layer", type=int, default=15)
    p.add_argument("--hook-point", default="ln1.hook_normalized",
                   help="SAE hookpoint inside blocks.{layer}.")
    p.add_argument("--head", type=int, default=0,
                   help="Head used for FRA feature ranking (matches QK→QK choice)")
    p.add_argument("--n-prompts", type=int, default=8)
    p.add_argument("--alphas", nargs="+", type=float,
                   default=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
    p.add_argument("--top-k-features", type=int, default=50)
    p.add_argument("--k-pairs", type=int, default=50)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-root", required=True)
    args = p.parse_args()

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    prompts = EM_EVAL_PROMPTS[: args.n_prompts]
    per_prompt_seeds = [args.eval_seed + i for i in range(args.n_prompts)]

    print("=== Phase 1 Arditi-additive 7B orchestrator ===")
    print(f"  em_model         : {args.em_model}")
    print(f"  layer × hookpoint: L{args.layer}.{args.hook_point}")
    print(f"  sae_dir          : {args.sae_dir}")
    print(f"  eval_seed (base) : {args.eval_seed}")
    print(f"  alphas           : {args.alphas}")
    print(f"  top-k features   : {args.top_k_features}")

    t_start = time.time()
    model = load_em_model(args.em_model, device=args.device)
    tokenizer = model.tokenizer
    print(f"[load] model loaded in {time.time() - t_start:.1f}s")

    sae = load_arditi_sae_from_dir(Path(args.sae_dir), device=args.device)

    # Feature ranking: reuse the multi-prompt FRA ranker so we steer the
    # same top-K features the QK→QK orchestrator uses (recipe difference
    # is in the hook, not the feature set).
    from fra.em_evaluation import rank_features_multi_prompt
    print(f"[rank] ranking features across {len(prompts)} prompts …")
    t_rank = time.time()
    ranked = rank_features_multi_prompt(
        model, sae, args.layer, args.head, args.hook_point,
        prompts=prompts,
        max_length=args.max_length,
        top_k=args.top_k_features,
        k_pairs=args.k_pairs,
        verbose=True,
    )
    feature_ids = ranked["qk"][: args.top_k_features]
    print(f"[rank] using top-{len(feature_ids)} QK features in {time.time() - t_rank:.1f}s "
          f"(head 5: {feature_ids[:5]})")

    hook_name = f"blocks.{args.layer}.{args.hook_point}"
    sae_id = f"L{args.layer}_ln1_arditi_qwen7b"

    qualitative = []
    t_gen = time.time()
    for alpha in args.alphas:
        t_cell = time.time()
        hooks = [(hook_name, make_additive_hook_batched(sae, feature_ids, alpha))]
        responses = generate_with_hooks_batch(
            model, tokenizer, prompts,
            fwd_hooks=hooks,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            seed=per_prompt_seeds,
        )
        cond_name = f"sae_resid_a{alpha}"
        print(f"  α={alpha:.2f}  {cond_name:<22s} {time.time() - t_cell:.1f}s", flush=True)
        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            qualitative.append({
                "seed": per_prompt_seeds[i],
                "scale": float(alpha),
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

    out_path = (out_root /
                f"qualitative_{sae_id}_{args.em_model}_evalseed{args.eval_seed}_top{args.top_k_features}.json")
    out_path.write_text(json.dumps(qualitative, indent=2, ensure_ascii=False))
    print(f"\n[save] {out_path}  ({len(qualitative)} entries)")
    print(f"=== TOTAL stream time: {time.time() - t_start:.1f}s ===")


if __name__ == "__main__":
    sys.exit(main())
