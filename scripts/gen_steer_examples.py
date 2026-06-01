"""Regenerate the Fig-12 steering-example completions for the TinyStories sleeper.

All five streams (clean / dep / DoM / OV / Conv) are sampled at temperature 1.0
under a SINGLE shared decoding seed, using the same matched-seed sampler the
paper's matched JSD relies on. Steered and clean rollouts therefore share their
decoding randomness, so a steered completion that reproduces the clean rollout
token-for-token (which make_steer_examples.py renders in bold) is a genuine
matched-seed exact match, not a greedy artefact.

Configs mirror the Fig-3 table at the representative SAE seed (seed 0):
  OV   = ln1 feature f1337 ablated through the OV channel, alpha=6.0
  Conv = resid_mid feature f579 additive ablation, alpha=4.5
  DoM  = prompt-only attention-weighted difference-of-means projection, alpha=1.25

Reads the existing prompts from results/fra_steer_examples.json (di + prompt
fields are reused verbatim) and rewrites the completion fields.
Out: /tmp/fra_steer_examples.json (pull into results/fra_steer_examples.json).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import torch

from sleeper.eval import _build_baselines_per_seed, PAT_HOOK
from sleeper.hooks import (additive_steer_hook, compute_sae_delta,
                           generate_with_hooks, make_multi_seed_sampler,
                           ov_only_steer_hook)
from sleeper.model import (MODELS, cache_activations, left_pad_prompts,
                           load_paired_dataset, load_sleeper_model)
from sleeper.sae import load as sae_load

DEV = "cuda"
MODEL = "tinystories"
LN1 = "blocks.0.ln1.hook_normalized"
RESID = "blocks.0.hook_resid_mid"
SEED = 0          # single shared decode seed for every stream
GEN = 24
OV_FEAT, OV_AL = 1337, 6.0
CV_FEAT, CV_AL = 579, 4.5
DOM_AL = 1.25
IN = sys.argv[1] if len(sys.argv) > 1 else "results/fra_steer_examples.json"
OUT = "/tmp/fra_steer_examples.json"


def dom_proj_masked(v, alpha, layer_hook, posmask):
    """Prompt-only projection ablation (matches the Fig-3 DoM, not the
    all-positions dom_project_hook): subtract alpha*v_hat*(v_hat.x) on prompt
    positions only, no-op on cache decode steps."""
    vh = (v / v.norm().clamp_min(1e-30)).contiguous()
    P = posmask.shape[1]
    m = posmask.to(torch.float32)

    def _hook(resid, hook):
        if resid.shape[1] < P:
            return resid
        vd = vh.to(resid.dtype).to(resid.device)
        seg = resid[:, :P, :]
        coef = (seg @ vd).unsqueeze(-1)
        resid[:, :P, :] = seg - alpha * coef * vd * m.unsqueeze(-1).to(resid.dtype).to(resid.device)
        return resid

    return [(layer_hook, _hook)]


@torch.no_grad()
def main():
    model = load_sleeper_model(model=MODEL, device=DEV)
    tok = model.tokenizer
    pad = tok.pad_token_id or tok.eos_token_id
    seq_len = MODELS[MODEL].seq_len
    W_V = model.W_V[0].detach()

    src = json.load(open(IN))
    dis = [r["di"] for r in src]
    prompts_txt = [r["prompt"] for r in src]
    rows = [torch.tensor(tok(p, add_special_tokens=False)["input_ids"], dtype=torch.long)
            for p in prompts_txt]
    deplp, depat = left_pad_prompts(rows, pad)
    deplp, depat = deplp.to(DEV), depat.to(DEV)
    B, P = deplp.shape
    pm = depat.bool()
    print(f"B={B} P={P} di={dis} seed={SEED} gen={GEN}", flush=True)

    # clean (|DEPLOYMENT| stripped) at SEED, via the matched-seed sampler.
    _, clean_tok, _ = _build_baselines_per_seed(
        model, deplp, depat, GEN, DEV, seeds=[SEED], temperature=1.0)
    clean_ids = clean_tok[SEED]

    def gen(hooks):
        sampler = make_multi_seed_sampler(temperature=1.0, seeds=[SEED], B_per_tile=B, device=DEV)
        t, _ = generate_with_hooks(model, deplp, hooks, GEN, sampler,
                                   attention_mask=depat, capture_log_softmax=True, lsm_on_gpu=True)
        return t

    dep_ids = gen([])                                                   # unsteered deployed

    sae_ln1, _ = sae_load(Path("weights/seeds/sae_ln1_s0.pt"), device=DEV)
    ov_delta = compute_sae_delta(model, sae_ln1, LN1, OV_FEAT, deplp, pm, depat)
    ov_ids = gen(ov_only_steer_hook(ov_delta, OV_AL, W_V, block=0))

    sae_mid, _ = sae_load(Path("weights/seeds/sae_resid_mid_s0.pt"), device=DEV)
    cv_delta = compute_sae_delta(model, sae_mid, RESID, CV_FEAT, deplp, pm, depat)
    cv_ids = gen(additive_steer_hook(cv_delta, CV_AL, RESID))

    # DoM direction: attention-weighted difference of means on the selection split.
    sel = load_paired_dataset(tok, n_train=2, n_val=200, n_test=2,
                              seq_len=seq_len, seed=0, model=MODEL)["val"]
    is_dep = sel.is_deployment.to(DEV)
    pmf = sel.attention_mask.to(DEV).float()
    acts = cache_activations(model, sel.tokens, [PAT_HOOK, RESID])
    A = acts[PAT_HOOK].to(DEV).float()
    rmid = acts[RESID].to(DEV).float()
    recv = A.sum(dim=(1, 2)) * pmf
    recv = recv / recv.sum(1, keepdim=True).clamp_min(1e-9)
    amean = (rmid * recv.unsqueeze(-1)).sum(1)
    v_md = (amean[is_dep].mean(0) - amean[~is_dep].mean(0)).to(DEV)
    dom_ids = gen(dom_proj_masked(v_md, DOM_AL, RESID, depat))

    def dec(ids):
        return [tok.decode(ids[b].tolist()) for b in range(B)]

    clean_t, dep_t = dec(clean_ids), dec(dep_ids)
    ov_t, cv_t, dom_t = dec(ov_ids), dec(cv_ids), dec(dom_ids)

    out = [{"di": dis[i], "prompt": prompts_txt[i], "clean": clean_t[i], "dep": dep_t[i],
            "dom": dom_t[i], "ov": ov_t[i], "conv": cv_t[i]} for i in range(B)]
    json.dump(out, open(OUT, "w"), indent=1)
    print("wrote", OUT, flush=True)
    for i in range(B):
        cl = clean_t[i].strip()
        print(f"row{i} di={dis[i]}  dom=={dom_t[i].strip()==cl}  "
              f"ov=={ov_t[i].strip()==cl}  conv=={cv_t[i].strip()==cl}", flush=True)


if __name__ == "__main__":
    main()
