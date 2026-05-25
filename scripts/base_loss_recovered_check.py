"""Diagnostic: train the same TopK SAE config on the BASE TinyStories model
harvested on RAW TinyStories text, and measure loss_recovered — to test whether
the high recovered on the sleeper setup is an artifact of the narrow,
templated sleeper-instruct data (+ the sleeper LoRA) vs. base model/base data.

Run on a fresh pod (after the env fix). Prints recovered + FVU + ||err||/||x||
for both hookpoints, comparable to the sleeper-data numbers.
"""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformer_lens import HookedTransformer

from sleeper.model import cache_activations
from sleeper.sae import train

BASE_MODEL = "roneneldan/TinyStories-Instruct-33M"
HOOKS = ["blocks.0.hook_resid_mid", "blocks.0.ln1.hook_normalized"]


def get_tokens(tokz, split, n, seq=128):
    ds = load_dataset("roneneldan/TinyStories", split=split)
    rows = []
    for ex in ds:
        ids = tokz(ex["text"], add_special_tokens=False)["input_ids"]
        if len(ids) >= seq:
            rows.append(torch.tensor(ids[:seq], dtype=torch.long))
        if len(rows) >= n:
            break
    return torch.stack(rows)


@torch.no_grad()
def evaluate(model, sae, hook, eval_tok, dev):
    rec = lambda a, hook: sae.decode(sae.encode(a.float())).to(a.dtype)
    zero = lambda a, hook: torch.zeros_like(a)

    def ce(fn):
        tot = 0.0; n = 0
        fh = [(hook, fn)] if fn else None
        for i in range(0, eval_tok.shape[0], 8):
            b = eval_tok[i:i+8].to(dev)
            lg = model.run_with_hooks(b, fwd_hooks=fh, return_type="logits") if fh else model(b, return_type="logits")
            lp = F.log_softmax(lg[:, :-1].float(), -1)
            nll = -lp.gather(-1, b[:, 1:].unsqueeze(-1)).squeeze(-1)
            tot += nll.sum().item(); n += nll.numel(); del lg, lp, nll
        return tot / n

    # recon error
    xs = []
    for i in range(0, eval_tok.shape[0], 8):
        _, c = model.run_with_cache(eval_tok[i:i+8].to(dev), return_type=None,
                                    names_filter=lambda x: x == hook)
        xs.append(c[hook].float().reshape(-1, c[hook].shape[-1]).cpu())
    x = torch.cat(xs).to(dev); mu = x.mean(0); xh = sae.decode(sae.encode(x))
    fvu = ((x-xh).pow(2).sum()/(x-mu).pow(2).sum()).item()
    err_rel = ((x-xh).norm(dim=-1).mean()/x.norm(dim=-1).mean()).item()
    lc, ls, lz = ce(None), ce(rec), ce(zero)
    return {"clean": lc, "sae": ls, "zero": lz,
            "recovered": (lz-ls)/(lz-lc), "fvu": fvu, "err_rel": err_rel}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_train", type=int, default=10000)
    p.add_argument("--n_steps", type=int, default=50000)
    p.add_argument("--d_sae", type=int, default=3072)
    p.add_argument("--k", type=int, default=32)
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    model = HookedTransformer.from_pretrained(BASE_MODEL, device=dev)
    model.eval()
    tokz = model.tokenizer
    print(f"[base] loaded {BASE_MODEL}", flush=True)
    train_tok = get_tokens(tokz, "train", args.n_train)
    eval_tok = get_tokens(tokz, "validation", 200)
    print(f"[base] train_tok={tuple(train_tok.shape)} eval_tok={tuple(eval_tok.shape)}", flush=True)

    for hook in HOOKS:
        print(f"\n[base] === {hook} ===", flush=True)
        acts = cache_activations(model, train_tok, [hook])[hook]
        sae, _ = train(acts, d_sae=args.d_sae, k=args.k, n_steps=args.n_steps,
                       seed=0, device=dev)
        r = evaluate(model, sae, hook, eval_tok, dev)
        print(f"[base] {hook}: clean={r['clean']:.3f} sae={r['sae']:.3f} zero={r['zero']:.3f} "
              f"| recovered={r['recovered']:.3f}  FVU={r['fvu']:.4f}  err/||x||={r['err_rel']:.4f}",
              flush=True)


if __name__ == "__main__":
    main()
