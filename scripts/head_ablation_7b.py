"""Head ablation at one layer of Qwen-2.5-7B + EM LoRA.

A stripped 7B version of `run_experiments.py --task head_ablation`. The
upstream script is hardcoded to 14B + Nura's ln1 SAE; we just want the
sorted head importance for the 7B medical EM at L15 so we can pick a head
for the 3-method comparison. No SAE needed.

Reuses `fra.head_ablation.head_attribution_sweep` unchanged.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fra.em_evaluation import EM_EVAL_PROMPTS
from fra.head_ablation import head_attribution_sweep


EM_MODELS_7B = {
    "medical": "andyrdt/Qwen2.5-7B-Instruct_bad-medical",
    "base":    "Qwen/Qwen2.5-7B-Instruct",
}


def load_em_model(em_model: str, device: str = "cuda",
                  base_model_id: str | None = None, em_model_id: str | None = None):
    from transformers import AutoModelForCausalLM
    from transformer_lens import HookedTransformer
    base_id = base_model_id or EM_MODELS_7B["base"]
    name = em_model_id or EM_MODELS_7B.get(em_model, em_model)
    print(f"[load] {em_model} → {name}  (base={base_id})")
    if em_model == "base":
        hf = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16, device_map="cpu")
        model = HookedTransformer.from_pretrained_no_processing(base_id, hf_model=hf, device=device, dtype=torch.bfloat16)
        del hf
    else:
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16, device_map="cpu")
        try:
            lora = PeftModel.from_pretrained(base, name)
            merged = lora.merge_and_unload()
            del base, lora
        except (ValueError, OSError) as e:
            print(f"  PeftModel load failed ({e!r}); loading {name} as a full model", flush=True)
            del base; torch.cuda.empty_cache()
            merged = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map="cpu")
        model = HookedTransformer.from_pretrained_no_processing(
            base_id, hf_model=merged, device=device, dtype=torch.bfloat16,
        )
        del merged
    torch.cuda.empty_cache()
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--em-model", default="medical",
                   help="EM key: base, a key in EM_MODELS_7B, or any label with --em-model-id.")
    p.add_argument("--base-model-id", default=None, help="Base HF id (default 7B; pass 14B id).")
    p.add_argument("--em-model-id", default=None, help="EM checkpoint HF id (overrides dict).")
    p.add_argument("--layer", type=int, default=15)
    p.add_argument("--n-texts", type=int, default=4)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--out", required=True, help="Output JSON path")
    args = p.parse_args()

    t0 = time.time()
    model = load_em_model(args.em_model, base_model_id=args.base_model_id,
                          em_model_id=(None if args.em_model == "base" else args.em_model_id))
    print(f"[load] model loaded in {time.time()-t0:.1f}s")

    texts = EM_EVAL_PROMPTS[: args.n_texts]
    print(f"[ablate] sweeping {model.cfg.n_heads} heads at L{args.layer} on {len(texts)} prompts …")
    results = head_attribution_sweep(model, texts, args.layer,
                                     max_length=args.max_length, verbose=True)
    out = {
        "em_model": args.em_model,
        "layer": args.layer,
        "n_texts": len(texts),
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n=== TOP 5 heads at L{args.layer} (by |loss_delta|) ===")
    for r in results[:5]:
        print(f"  H{r['head']:>2}  loss_delta={r['loss_delta']:+.4f}  kl_div={r['kl_div']:.4f}")
    print(f"\n[save] {args.out}")


if __name__ == "__main__":
    sys.exit(main())
