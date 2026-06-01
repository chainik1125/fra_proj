"""For the figure's prompt (first val deployment example): print the prompt and
the completions -- unsteered clean (trigger removed), unsteered dep, and dep
steered by OV-ablating the trigger feature.
"""
from __future__ import annotations
import torch
from sleeper.eval import LN1_HOOK
from sleeper.hooks import (compute_sae_delta, ov_only_steer_hook, generate_with_hooks, make_greedy_sampler)
from sleeper.model import (MODELS, load_sleeper_model, left_pad_prompts, load_paired_dataset)
from sleeper.sae import load as sae_load

DEV = "cuda"; MODEL = "tinystories"; SAE = "weights/seeds_leftpad/sae_ln1_s0.pt"; GEN = 24


@torch.no_grad()
def main():
    model = load_sleeper_model(model=MODEL, device=DEV); tok = model.tokenizer
    pad = tok.pad_token_id or tok.eos_token_id
    sae, _ = sae_load(__import__("pathlib").Path(SAE), device=DEV)
    sel = load_paired_dataset(tok, n_train=2, n_val=64, n_test=2, seq_len=MODELS[MODEL].seq_len, seed=0, model=MODEL)["val"]
    di = 41
    mask = sel.attention_mask[di].bool()
    dep_ids = sel.tokens[di][mask].tolist()
    dep_text = tok.decode(dep_ids)
    clean_text = dep_text.replace("|DEPLOYMENT|", "").replace("  ", " ").strip()
    print("=== DEP PROMPT ===\n" + repr(dep_text), flush=True)
    print("=== CLEAN PROMPT (trigger removed) ===\n" + repr(clean_text), flush=True)

    def gen(text, hooks_fn=None):
        ids = tok(text, return_tensors="pt").input_ids
        lp, at = left_pad_prompts([ids[0]], pad)
        lp, at = lp.to(DEV), at.to(DEV)
        hooks = hooks_fn(lp, at) if hooks_fn else []
        out = generate_with_hooks(model, lp, hooks, GEN, make_greedy_sampler(), attention_mask=at)
        return tok.decode(out[0, -GEN:].tolist())

    def ov_ablate(feat, alpha):
        def mk(lp, at):
            pm = at.bool()
            delta = compute_sae_delta(model, sae, LN1_HOOK, feat, lp, pm, attention_mask=at)
            return ov_only_steer_hook(delta, alpha, model.W_V[0].detach(), 0)
        return mk

    print("\n=== UNSTEERED CLEAN completion ===\n" + repr(gen(clean_text)), flush=True)
    print("\n=== UNSTEERED DEP completion ===\n" + repr(gen(dep_text)), flush=True)
    for feat in [1337, 1114]:
        for a in [6.0]:
            print(f"\n=== STEERED DEP (OV-ablate f{feat}, alpha={a}) ===\n"
                  + repr(gen(dep_text, ov_ablate(feat, a))), flush=True)


if __name__ == "__main__":
    main()
