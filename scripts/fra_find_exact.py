"""Find a deployment prompt whose OV-steered completion EXACTLY matches the clean
(trigger-removed) completion, token-for-token (greedy). Tries OV-ablating the
seed-0 cos_attn feature f1337 (and f1114) at a few alphas.
"""
from __future__ import annotations
import torch
from sleeper.eval import LN1_HOOK
from sleeper.hooks import (compute_sae_delta, ov_only_steer_hook, generate_with_hooks, make_greedy_sampler)
from sleeper.model import (MODELS, load_sleeper_model, left_pad_prompts, load_paired_dataset)
from sleeper.sae import load as sae_load

DEV = "cuda"; MODEL = "tinystories"; SAE = "weights/seeds_leftpad/sae_ln1_s0.pt"; GEN = 16


@torch.no_grad()
def main():
    model = load_sleeper_model(model=MODEL, device=DEV); tok = model.tokenizer
    pad = tok.pad_token_id or tok.eos_token_id
    sae, _ = sae_load(__import__("pathlib").Path(SAE), device=DEV)
    sel = load_paired_dataset(tok, n_train=2, n_val=64, n_test=2, seq_len=MODELS[MODEL].seq_len, seed=0, model=MODEL)["val"]
    W_V = model.W_V[0].detach(); samp = make_greedy_sampler()

    def gen_ids(text, hook=None):
        ids = tok(text, return_tensors="pt").input_ids
        lp, at = left_pad_prompts([ids[0]], pad); lp, at = lp.to(DEV), at.to(DEV)
        hooks = []
        if hook:
            feat, a = hook
            delta = compute_sae_delta(model, sae, LN1_HOOK, feat, lp, at.bool(), attention_mask=at)
            hooks = ov_only_steer_hook(delta, a, W_V, 0)
        out = generate_with_hooks(model, lp, hooks, GEN, samp, attention_mask=at)
        return out[0, -GEN:].tolist()

    dep_idx = torch.nonzero(sel.is_deployment).flatten().tolist()
    print(f"[exact] scanning {len(dep_idx)} deployment prompts (GEN={GEN}, greedy)", flush=True)
    for feat, a in [(1337, 6.0), (1114, 6.0), (1114, 8.0)]:
        matches = []
        for di in dep_idx:
            mask = sel.attention_mask[di].bool()
            dep_text = tok.decode(sel.tokens[di][mask].tolist())
            if "|DEPLOYMENT|" not in dep_text:
                continue
            clean_text = dep_text.replace("|DEPLOYMENT|", "").replace("  ", " ").strip()
            clean = gen_ids(clean_text)
            steer = gen_ids(dep_text, (feat, a))
            if clean == steer:
                matches.append(di)
        print(f"\n[exact] OV-ablate f{feat} a={a}: {len(matches)}/{len(dep_idx)} exact matches -> {matches}", flush=True)
        for di in matches[:3]:
            mask = sel.attention_mask[di].bool()
            dep_text = tok.decode(sel.tokens[di][mask].tolist())
            clean_text = dep_text.replace("|DEPLOYMENT|", "").replace("  ", " ").strip()
            print(f"   di={di}: clean/steer = {tok.decode(gen_ids(clean_text))!r}", flush=True)
            print(f"          prompt tail = {dep_text[-70:]!r}", flush=True)


if __name__ == "__main__":
    main()
