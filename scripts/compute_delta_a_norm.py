"""Compute ‖Δa‖ (EM-vs-base mean-activation-difference L2 norm) at L15 for BOTH
steering hookpoints, in ONE consistent convention, for magnitude-matched grid
steering.

Mirrors scripts/compute_arditi_actdiff.py's convention (last prompt-token, chat-
templated, mean over prompts of the per-prompt last-token activation; diff =
mean(act|medical) − mean(act|base); ‖Δa‖ = ‖diff‖_2) but captures TWO hookpoints
in a single forward so resid_post and ln1 are directly comparable:

  - resid_post : blocks.{L}.hook_resid_post (HF: layers[L] block output residual)
  - ln1        : the layer's input_layernorm OUTPUT = (x/rms)·γ (POST-GAIN) — the
                 exact signal the ln1 SAE + the grid's ln1 steering operate in.

Output JSON (uploaded to HF by run_grid.sh):
  {"layer":15, "n_prompts":N,
   "resid_post": {"diff_norm_l2":..., "pos_mean_norm":..., "neg_mean_norm":...},
   "ln1_postgain": {"diff_norm_l2":..., ...}}

The grid orchestrator's --delta-a-norm is then set to resid_post.diff_norm_l2 for
resid_post cells and ln1_postgain.diff_norm_l2 for ln1 cells.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fra.em_evaluation import EM_EVAL_PROMPTS

BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
EM_MODELS = {"medical": "andyrdt/Qwen2.5-7B-Instruct_bad-medical"}


def load_model(which, device):
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if which == "base":
        m = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=torch.bfloat16, device_map=device)
    else:
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=torch.bfloat16, device_map=device)
        m = PeftModel.from_pretrained(base, EM_MODELS[which]).merge_and_unload()
        del base
    m.eval(); torch.cuda.empty_cache()
    return m, tok


@torch.no_grad()
def last_token_means(model, tok, prompts, layer, device, batch_size=8, max_ctx=512):
    """Return (resid_post_mean[d], ln1_postgain_mean[d]) over prompts' last tokens."""
    blk = model.model.layers[layer]
    ln1 = blk.input_layernorm
    cache = {}
    h1 = blk.register_forward_hook(
        lambda mod, i, o: cache.__setitem__("rp", o[0] if isinstance(o, tuple) else o))
    h2 = ln1.register_forward_hook(
        lambda mod, i, o: cache.__setitem__("ln1", o[0] if isinstance(o, tuple) else o))
    rp_rows, ln1_rows = [], []
    tok.padding_side = "left"
    try:
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            fmt = [tok.apply_chat_template([{"role": "user", "content": p}],
                                           tokenize=False, add_generation_prompt=True) for p in batch]
            enc = tok(fmt, return_tensors="pt", padding=True, truncation=True, max_length=max_ctx).to(device)
            _ = model(**enc)
            attn = enc["attention_mask"]
            last = attn.sum(1) - 1
            ar = torch.arange(enc["input_ids"].shape[0], device=device)
            rp_rows.append(cache["rp"][ar, last].float().cpu())
            ln1_rows.append(cache["ln1"][ar, last].float().cpu())
    finally:
        h1.remove(); h2.remove()
    return torch.cat(rp_rows).mean(0), torch.cat(ln1_rows).mean(0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layer", type=int, default=15)
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-prompts", type=int, default=8,
                   help="Default = the 8 EM_EVAL_PROMPTS (same set the grid steers/evaluates on).")
    p.add_argument("--out", default="/workspace/delta_a_norm_L15.json")
    args = p.parse_args()

    prompts = EM_EVAL_PROMPTS[: args.n_prompts]
    print(f"=== ‖Δa‖ at L{args.layer} (resid_post + ln1 post-gain), n={len(prompts)} prompts ===", flush=True)

    means = {}
    for which in ["medical", "base"]:
        t0 = time.time()
        m, tok = load_model(which, args.device)
        print(f"[{which}] loaded in {time.time()-t0:.1f}s", flush=True)
        rp, ln1 = last_token_means(m, tok, prompts, args.layer, args.device)
        means[which] = {"rp": rp, "ln1": ln1}
        del m; torch.cuda.empty_cache()
        print(f"[{which}] resid_post ‖mean‖={rp.norm():.3f}  ln1 ‖mean‖={ln1.norm():.3f}", flush=True)

    def block(key):
        diff = means["medical"][key] - means["base"][key]
        return {"diff_norm_l2": float(diff.norm()),
                "pos_mean_norm": float(means["medical"][key].norm()),
                "neg_mean_norm": float(means["base"][key].norm())}

    out = {"layer": args.layer, "n_prompts": len(prompts),
           "convention": "last-token chat-templated, mean over prompts, EM(medical)-base diff",
           "resid_post": block("rp"), "ln1_postgain": block("ln1")}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("=== ‖Δa‖ result ===", flush=True)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    sys.exit(main())
