#!/usr/bin/env python3
"""
Head ablation + ‖Δa‖ for Qwen2.5-14B extreme-sports at L24.

Outputs:
  - Top head (argmax loss-delta when zero-ablating each head)
  - ‖Δa‖ at resid_post and ln1 (base → sports activation diff)

Usage on RunPod:
    python experiments/fra_14b_sports/head_delta_a.py
"""
import json
import os
import sys
import time

import numpy as np
import torch

torch.set_grad_enabled(False)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from run_experiments import EM_MODELS
from fra.em_evaluation import EM_EVAL_PROMPTS

LAYER = 24
SEEDS = [42, 123]
EM_VARIANT = "sports"
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "./head_ablation_delta_a_sports_l24.json")


def load_model(em_variant, device="cuda"):
    """Load model using the same pattern as run_experiments.py."""
    from transformer_lens import HookedTransformer

    model_name = EM_MODELS[em_variant]
    print(f"Loading: {model_name}")
    t0 = time.time()

    if em_variant == "base":
        model = HookedTransformer.from_pretrained_no_processing(
            model_name, device=device, dtype=torch.bfloat16,
        )
    else:
        from transformers import AutoModelForCausalLM
        from peft import PeftModel

        base_hf = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-14B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )
        lora_hf = PeftModel.from_pretrained(base_hf, model_name)
        merged_hf = lora_hf.merge_and_unload()
        del base_hf, lora_hf

        model = HookedTransformer.from_pretrained_no_processing(
            "Qwen/Qwen2.5-14B-Instruct",
            hf_model=merged_hf,
            device=device,
            dtype=torch.bfloat16,
        )
        del merged_hf
        torch.cuda.empty_cache()

    print(f"  Loaded in {time.time()-t0:.1f}s")
    return model


def collect_last_token(model, prompts, hooks):
    """Get last-token activations for each prompt at each hook point."""
    out = {h: [] for h in hooks}
    for p in prompts:
        toks = model.to_tokens(p)
        _, cache = model.run_with_cache(toks, names_filter=hooks)
        for h in hooks:
            out[h].append(cache[h][0, -1].float().cpu())
    return {h: torch.stack(out[h]) for h in hooks}


def main():
    prompts = EM_EVAL_PROMPTS[:8]
    H_RP = f"blocks.{LAYER}.hook_resid_post"
    H_LN1 = f"blocks.{LAYER}.ln1.hook_normalized"

    # ── 1. Compute ‖Δa‖ ──────────────────────────────────────────────────
    print("\n[1/3] Loading BASE model")
    base = load_model("base")
    base_act = collect_last_token(base, prompts, [H_RP, H_LN1])
    gamma = base.blocks[LAYER].ln1.w.detach().float().cpu()
    base_act[H_LN1] = base_act[H_LN1] * gamma  # post-gain
    del base
    torch.cuda.empty_cache()

    print(f"\n[2/3] Loading {EM_VARIANT.upper()} model")
    em = load_model(EM_VARIANT)
    em_act = collect_last_token(em, prompts, [H_RP, H_LN1])
    em_act[H_LN1] = em_act[H_LN1] * gamma

    delta_rp = float((em_act[H_RP] - base_act[H_RP]).norm(dim=-1).mean().item())
    delta_ln1 = float((em_act[H_LN1] - base_act[H_LN1]).norm(dim=-1).mean().item())
    print(f"  ‖Δa‖_L{LAYER}_resid_post   = {delta_rp:.3f}")
    print(f"  ‖Δa‖_L{LAYER}_ln1_postgain = {delta_ln1:.3f}")

    # ── 2. Head ablation ─────────────────────────────────────────────────
    print(f"\n[3/3] Head ablation L{LAYER} on {EM_VARIANT}")
    n_heads = em.cfg.n_heads

    def make_hook(head_idx):
        def h(z, hook):
            z[:, :, head_idx, :] = 0
            return z
        return h

    losses_orig = []
    losses_ablated = [[] for _ in range(n_heads)]

    for p in prompts:
        toks = em.to_tokens(p)
        losses_orig.append(em(toks, return_type="loss").item())
        for hd in range(n_heads):
            loss = em.run_with_hooks(
                toks, return_type="loss",
                fwd_hooks=[(f"blocks.{LAYER}.attn.hook_z", make_hook(hd))],
            ).item()
            losses_ablated[hd].append(loss)

    lo = np.array(losses_orig)
    deltas = [float((np.array(losses_ablated[h]) - lo).mean()) for h in range(n_heads)]
    top_head = int(np.argmax(deltas))
    top5 = sorted(range(n_heads), key=lambda i: -deltas[i])[:5]

    print(f"\n  Top head L{LAYER}: H{top_head}  Δloss={deltas[top_head]:.4f}")
    print(f"  Top-5 heads: {top5}")
    print(f"  All deltas: {[f'{d:.4f}' for d in deltas]}")

    # ── 3. Save results ──────────────────────────────────────────────────
    results = {
        "layer": LAYER,
        "em_variant": EM_VARIANT,
        "em_model": EM_MODELS[EM_VARIANT],
        "n_prompts": len(prompts),
        "seeds": SEEDS,
        "delta_a_norm_resid_post": delta_rp,
        "delta_a_norm_ln1_postgain": delta_ln1,
        "head_ablation_loss_deltas": deltas,
        "top_head_argmax": top_head,
        "top_head_delta": deltas[top_head],
        "top5_heads": top5,
        "loss_orig_mean": float(lo.mean()),
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")

    del em
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
