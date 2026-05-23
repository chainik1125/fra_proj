"""Wang-style persona-vector SAE feature ranker on Qwen-2.5-7B + bad-medical.

Distinct from `compute_arditi_feature_ranking.py`, which ranks decoder columns
by `cos(W_dec[:, i], Δa)` for a Δa pulled from `compute_arditi_actdiff.py`.
This script does the *encoder-side* ranking that Wang et al's persona-vector
recipe describes:

    Δf_i = mean(f_i, EM-merged answer-tokens) − mean(f_i, base answer-tokens)

and returns the top-N features sorted by signed Δf (most-more-active under EM).

Procedure
---------
For each of the 8 `EM_EVAL_PROMPTS`:
  1. Apply chat template (`{role: user, content: prompt}` + assistant turn opener).
  2. Generate up to `--max-new-tokens` from the model (temperature 1, top_p 1,
     per-prompt seed = `--seed + prompt_idx` for reproducibility).
  3. Forward the *full* (prompt + generated) sequence with a hook at
     `model.model.layers[layer]` to capture the post-block residual.
  4. Slice positions ≥ prompt_len → answer tokens only.
  5. Encode through andyrdt's `resid_post_layer_{layer}/trainer_{trainer}` SAE.
  6. Accumulate per-feature sum and a token counter.

Done sequentially for both models (Qwen-2.5-7B-Instruct base, and
`andyrdt/Qwen2.5-7B-Instruct_bad-medical` LoRA-merged via `PeftModel`) so peak
VRAM is one 7B model + one SAE at a time (~16 GB), fits L40/A40.

Output JSON schema:

    {
      "layer": <int>, "trainer": <int>,
      "n_features": <d_sae>,
      "top_n": <int>,
      "feature_ids":  [<top-N indices sorted by signed Δf descending>],
      "delta_f":      [<corresponding Δf values>],
      "delta_f_full": [<all d_sae Δf values, indexed by feature id>]   # optional
    }
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Make the project importable so we can re-use the SAE loader + prompt set.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phase1_arditi_orchestrator import load_arditi_sae
from fra.em_evaluation import EM_EVAL_PROMPTS


BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
EM_MODELS = {
    "medical": "andyrdt/Qwen2.5-7B-Instruct_bad-medical",
}


def load_model(em_model: str, device: str = "cuda"):
    """Load Qwen-7B base (em_model='base') or + bad-medical LoRA-merged
    ('medical'). Returns (model, tokenizer)."""
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if em_model == "base":
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID, torch_dtype=torch.bfloat16, device_map=device,
        )
    else:
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID, torch_dtype=torch.bfloat16, device_map=device,
        )
        lora = PeftModel.from_pretrained(base, EM_MODELS[em_model])
        model = lora.merge_and_unload()
        del base, lora
    model.eval()
    torch.cuda.empty_cache()
    return model, tokenizer


@torch.no_grad()
def collect_answer_feature_means(
    model, tokenizer, sae,
    prompts: list[str], layer: int,
    device: str = "cuda",
    max_new_tokens: int = 200,
    seed: int = 42,
):
    """Mean SAE feature activation over (prompts × answer-token positions).

    Each prompt:
      - generate up to max_new_tokens from this model (its own response),
      - forward the full (prompt + response) once with a hook to capture acts,
      - slice answer positions only,
      - encode through SAE.
    Returns Tensor[d_sae] of per-feature means.
    """
    d_sae = sae.decoder.weight.shape[1]
    sum_f = torch.zeros(d_sae, device=device, dtype=torch.float32)
    n_tokens = 0

    block = model.model.layers[layer]
    pad_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id

    for prompt_idx, prompt in enumerate(prompts):
        torch.manual_seed(seed + prompt_idx)

        # 1. chat-templated prompt
        messages = [{"role": "user", "content": prompt}]
        chat_in = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        prompt_ids = tokenizer(chat_in, return_tensors="pt").input_ids.to(device)
        prompt_len = prompt_ids.shape[1]

        # 2. generate response from this model (no hooks during gen — we forward
        # the whole sequence afterwards instead, to side-step KV-cache shapes).
        t_gen = time.time()
        gen = model.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True, temperature=1.0, top_p=1.0,
            pad_token_id=pad_id,
        )

        # 3. forward full sequence with hook
        cache = {}
        def hook(module, inputs, output):
            cache["h"] = output[0] if isinstance(output, tuple) else output
        handle = block.register_forward_hook(hook)
        try:
            _ = model(gen)
        finally:
            handle.remove()
        h = cache["h"]  # (1, T, d_model)

        # 4. slice answer-tokens
        answer_acts = h[0, prompt_len:, :]  # (n_answer, d_model)
        if answer_acts.shape[0] == 0:
            continue

        # 5. SAE encode
        f = sae.encode(answer_acts.float())  # (n_answer, d_sae)
        sum_f += f.sum(dim=0).float()
        n_tokens += answer_acts.shape[0]

        print(
            f"  [{prompt_idx + 1}/{len(prompts)}] prompt_len={prompt_len:4d} "
            f"answer={answer_acts.shape[0]:4d}  ({time.time() - t_gen:.1f}s)",
            flush=True,
        )

    if n_tokens == 0:
        raise RuntimeError("no answer tokens collected — every generation was empty")
    return sum_f / n_tokens


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layer", type=int, default=15)
    p.add_argument("--trainer", type=int, default=1)
    p.add_argument("--em-domain", default="medical", choices=list(EM_MODELS),
                   help="which EM LoRA to contrast against base")
    p.add_argument("--top-n", type=int, default=50)
    p.add_argument("--n-prompts", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="/workspace/wang_ranker_L15_top50.json")
    p.add_argument("--include-full-delta-f", action="store_true",
                   help="Also dump the full d_sae-length Δf vector (large).")
    args = p.parse_args()

    prompts = EM_EVAL_PROMPTS[: args.n_prompts]
    print(f"=== Wang feature ranker ===", flush=True)
    print(f"  layer={args.layer} trainer={args.trainer} em_domain={args.em_domain}", flush=True)
    print(f"  n_prompts={len(prompts)} max_new_tokens={args.max_new_tokens} seed={args.seed}", flush=True)

    # SAE loads once; the two models load sequentially.
    sae, sae_config = load_arditi_sae(args.layer, args.trainer, device=args.device)
    d_sae = sae.decoder.weight.shape[1]
    print(f"  d_sae={d_sae}", flush=True)

    means = {}
    for em_model in [args.em_domain, "base"]:
        print(f"\n--- {em_model} ---", flush=True)
        t0 = time.time()
        model, tokenizer = load_model(em_model, device=args.device)
        print(f"  model loaded in {time.time() - t0:.1f}s", flush=True)
        means[em_model] = collect_answer_feature_means(
            model, tokenizer, sae, prompts, args.layer,
            device=args.device, max_new_tokens=args.max_new_tokens, seed=args.seed,
        ).cpu()
        del model
        torch.cuda.empty_cache()
        print(f"  feature-mean done in {time.time() - t0:.1f}s", flush=True)

    delta_f = (means[args.em_domain] - means["base"]).float()  # (d_sae,)
    # Top-N by signed Δf — features that fire MORE under EM.
    top_vals, top_ids = torch.topk(delta_f, args.top_n)
    out = {
        "layer": args.layer,
        "trainer": args.trainer,
        "em_domain": args.em_domain,
        "n_features": int(d_sae),
        "top_n": int(args.top_n),
        "feature_ids": top_ids.tolist(),
        "delta_f": top_vals.tolist(),
    }
    if args.include_full_delta_f:
        out["delta_f_full"] = delta_f.tolist()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[save] {out_path}", flush=True)
    print(f"  top-5 features: {top_ids[:5].tolist()}", flush=True)
    print(f"  top-5 Δf      : {[round(v, 4) for v in top_vals[:5].tolist()]}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
