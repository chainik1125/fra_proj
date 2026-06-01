"""E4: finer alpha for OV and Conv -> matched + unmatched JSDc.

Finer alpha gave DoM a real gain (0.367->0.354 at alpha 1.25 vs 1.5). E4 tests
whether OV / Conv also improve below their 0.5-grid optima: per seed, sweep
{opt-0.25, opt, opt+0.25} around the known per-seed-optimal alpha. OV = cos_attn
feature through W_V; Conv = gated feature ablation @resid_mid. Out: /tmp/ar_E4.json.
Baselines: OV 0.398/0.634, Conv 0.413/0.643.
"""
from __future__ import annotations
import json
from pathlib import Path
import torch
from sleeper.eval import (_build_baselines_per_seed, jsd_per_row, split_dep_prompts, _tile_batch_dim)
from sleeper.hooks import (compute_sae_delta, additive_steer_hook, ov_only_steer_hook,
                           generate_with_hooks, make_multi_seed_sampler)
from sleeper.metrics import sleeper_fired_mask
from sleeper.model import left_pad_prompts, load_sleeper_model
from sleeper.sae import load as sae_load

DEV = "cuda"; MODEL = "tinystories"; LN1 = "blocks.0.ln1.hook_normalized"; RESID = "blocks.0.hook_resid_mid"
SEEDS = [0, 1, 2, 3, 4]; SAE_SEEDS = [0, 1, 2, 3, 4, 5]; GEN = 16; N_SEL = 200; N_EVAL = 400
SAE_DIR = "weights/seeds_leftpad"; OUT = "/tmp/ar_E4.json"
OV_FEAT = {0: 1337, 1: 76, 2: 169, 3: 1154, 4: 1006, 5: 1132}
OV_OPT = {0: 6.0, 1: 4.5, 2: 2.5, 3: 2.0, 4: 3.0, 5: 5.0}
CONV_FEAT = {0: 579, 1: 519, 2: 230, 3: 637, 4: 312, 5: 460}
CONV_OPT = {0: 4.5, 1: 5.0, 2: 3.0, 3: 3.0, 4: 5.0, 5: 6.0}


@torch.no_grad()
def main():
    model = load_sleeper_model(model=MODEL, device=DEV); tok = model.tokenizer
    pad = tok.pad_token_id or tok.eos_token_id; n = len(SEEDS); W_V = model.W_V[0].detach()
    dep = split_dep_prompts(tok, N_SEL, N_EVAL, model=MODEL)["eval"]
    deplp, depat = left_pad_prompts(dep, pad); deplp, depat = deplp.to(DEV), depat.to(DEV)
    B, P = deplp.shape; pm = depat.bool()
    clean_lsm, clean_tok, dep_lsm = _build_baselines_per_seed(model, deplp, depat, GEN, DEV, seeds=SEEDS, temperature=1.0)
    clean_g = {s: clean_lsm[s].to(DEV) for s in SEEDS}
    pairs = [(s, sp) for s in SEEDS for sp in SEEDS if s != sp]

    def evalu(hooks):
        lp_t = _tile_batch_dim(deplp, n); at_t = _tile_batch_dim(depat, n)
        sampler = make_multi_seed_sampler(temperature=1.0, seeds=SEEDS, B_per_tile=B, device=DEV)
        st_tok, st = generate_with_hooks(model, lp_t, hooks, GEN, sampler,
                                         attention_mask=at_t, capture_log_softmax=True, lsm_on_gpu=True)
        stp = {s: st[k*B:(k+1)*B] for k, s in enumerate(SEEDS)}
        sttk = {s: st_tok[k*B:(k+1)*B] for k, s in enumerate(SEEDS)}
        matched = sum(float(jsd_per_row(stp[s], clean_g[s]).mean()) for s in SEEDS) / n
        um = sum(float(jsd_per_row(stp[s], clean_g[sp]).mean()) for (s, sp) in pairs) / len(pairs)
        asr = sum(float(sleeper_fired_mask(sttk[s].cpu(), tok).float().mean()) for s in SEEDS) / n
        del st, st_tok, stp, sttk; torch.cuda.empty_cache()
        return matched, um, asr

    res = {}
    for method, kind, feats, opts in [("ov", "ln1", OV_FEAT, OV_OPT), ("conv", "resid_mid", CONV_FEAT, CONV_OPT)]:
        hook = LN1 if kind == "ln1" else RESID
        for s in SAE_SEEDS:
            sae, _ = sae_load(Path(f"{SAE_DIR}/sae_{kind}_s{s}.pt"), device=DEV)
            delta = _tile_batch_dim(compute_sae_delta(model, sae, hook, int(feats[s]), deplp, pm, attention_mask=depat), n)
            for a in [round(opts[s] - 0.25, 2), opts[s], round(opts[s] + 0.25, 2)]:
                if a <= 0:
                    continue
                hooks = ov_only_steer_hook(delta, a, W_V, block=0) if method == "ov" else additive_steer_hook(delta, a, RESID)
                m, u, asr = evalu(hooks)
                res[f"{method}_s{s}_a{a}"] = {"method": method, "seed": s, "alpha": a, "matched": m, "unmatched": u, "asr": asr}
                print(f"[E4] {method} s{s} a{a}: matched={m:.4f} unmatched={u:.4f} asr={asr:.4f}", flush=True)
        json.dump({"experiment": "E4_finer_alpha", "baselines": {"ov": [0.398, 0.634], "conv": [0.413, 0.643]},
                   "results": res}, open(OUT, "w"), indent=1)

    print("[E4] === per-method (per-seed argmin matched s.t. ASR<=1%, mean over seeds) ===", flush=True)
    for method in ["ov", "conv"]:
        ms, us = [], []
        for s in SAE_SEEDS:
            ok = [(res[k]["matched"], res[k]["unmatched"]) for k in res if res[k]["method"] == method and res[k]["seed"] == s and res[k]["asr"] <= 0.01]
            if ok:
                m, u = min(ok); ms.append(m); us.append(u)
        if ms:
            print(f"[E4] {method}: matched={sum(ms)/len(ms):.4f} unmatched={sum(us)/len(us):.4f} (n={len(ms)}/6)", flush=True)
    print(f"[E4] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
