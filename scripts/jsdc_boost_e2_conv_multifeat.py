"""E2: Conv multi-feature ablation -> matched + unmatched JSDc.

Hypothesis: jointly ablating the top-K dep-direction features (instead of 1)
spreads suppression, lowers per-feature alpha and collateral, and reduces JSDc
at ASR<=1%. Per SAE seed: rank resid_mid features by dep-clean activation
(top-10) -> rerank by cosine to the attn-weighted v_md -> take the top-K -> sum
their gated ablation deltas -> additive at resid_mid -> alpha sweep.

Eval: Fig-3 protocol (200 dep prompts, 5 decode seeds) matched JSDc + ASR;
20 cross-seed pairs unmatched JSDc. Layer-0 conv SAEs (n_train=10000).
Baseline Conv: matched 0.413 / unmatched 0.643. Out: /tmp/ar_E2.json.
"""
from __future__ import annotations
import json
from pathlib import Path
import torch
from sleeper.eval import (_build_baselines_per_seed, jsd_per_row, split_dep_prompts,
                          _tile_batch_dim, PAT_HOOK)
from sleeper.hooks import (compute_sae_delta, additive_steer_hook, generate_with_hooks,
                           make_multi_seed_sampler)
from sleeper.metrics import sleeper_fired_mask, rank_features_by_dep_clean
from sleeper.model import (MODELS, left_pad_prompts, load_sleeper_model,
                           cache_activations, load_paired_dataset)
from sleeper.sae import load as sae_load, encode_all

DEV = "cuda"; MODEL = "tinystories"; RESID = "blocks.0.hook_resid_mid"
SEEDS = [0, 1, 2, 3, 4]; SAE_SEEDS = [0, 1, 2, 3, 4, 5]; GEN = 16; N_SEL = 200; N_EVAL = 400
ALPHAS = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; KS = [1, 2, 3]; TOPK = 10
SAE_DIR = "weights/seeds_leftpad"
OUT = "/tmp/ar_E2.json"


@torch.no_grad()
def main():
    model = load_sleeper_model(model=MODEL, device=DEV); tok = model.tokenizer
    pad = tok.pad_token_id or tok.eos_token_id; seq_len = MODELS[MODEL].seq_len; n = len(SEEDS)
    dep = split_dep_prompts(tok, N_SEL, N_EVAL, model=MODEL)["eval"]
    deplp, depat = left_pad_prompts(dep, pad); deplp, depat = deplp.to(DEV), depat.to(DEV)
    B = deplp.shape[0]; pm = depat.bool()
    clean_lsm, clean_tok, dep_lsm = _build_baselines_per_seed(model, deplp, depat, GEN, DEV, seeds=SEEDS, temperature=1.0)
    clean_g = {s: clean_lsm[s].to(DEV) for s in SEEDS}
    pairs = [(s, sp) for s in SEEDS for sp in SEEDS if s != sp]

    sel = load_paired_dataset(tok, n_train=2, n_val=N_SEL, n_test=2, seq_len=seq_len, seed=0, model=MODEL)["val"]
    isd_c = sel.is_deployment; selpm = sel.attention_mask
    acts = cache_activations(model, sel.tokens, [PAT_HOOK, RESID])
    Ap = acts[PAT_HOOK].to(DEV).float(); Xr = acts[RESID].to(DEV).float()
    pmf = selpm.to(DEV).float(); isd = isd_c.to(DEV)
    recv = Ap.sum(dim=(1, 2)) * pmf; recv = recv / recv.sum(1, keepdim=True).clamp_min(1e-9)
    amean = (Xr * recv.unsqueeze(-1)).sum(1)
    vmd = amean[isd].mean(0) - amean[~isd].mean(0); vn = vmd / vmd.norm().clamp_min(1e-9)

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
    for s in SAE_SEEDS:
        sae, _ = sae_load(Path(f"{SAE_DIR}/sae_resid_mid_s{s}.pt"), device=DEV)
        z = encode_all(sae, acts[RESID])
        cand = rank_features_by_dep_clean(z, isd_c, selpm, top_k=TOPK)["top_indices"].cpu().tolist()
        scored = sorted(((float(sae.W_dec[f].to(DEV).float() @ vn / sae.W_dec[f].norm().clamp_min(1e-9)), int(f)) for f in cand), reverse=True)
        ranked = [f for _, f in scored]
        for K in KS:
            feats = ranked[:K]
            delta = None
            for f in feats:
                d = compute_sae_delta(model, sae, RESID, int(f), deplp, pm, attention_mask=depat)
                delta = d if delta is None else delta + d
            dt = _tile_batch_dim(delta, n)
            for a in ALPHAS:
                m, u, asr = evalu(additive_steer_hook(dt, a, RESID))
                res[f"s{s}_K{K}_a{a}"] = {"seed": s, "K": K, "alpha": a, "feats": feats,
                                         "matched": m, "unmatched": u, "asr": asr}
                print(f"[E2] s{s} K{K} a{a}: matched={m:.4f} unmatched={u:.4f} asr={asr:.4f}", flush=True)
            json.dump({"experiment": "E2_conv_multifeat", "hook": RESID,
                       "baseline_conv": {"matched": 0.413, "unmatched": 0.643},
                       "results": res}, open(OUT, "w"), indent=1)
        print(f"[E2] sae_seed {s} done", flush=True)

    print("[E2] === per-K aggregate (mean over seeds of per-seed argmin matched s.t. ASR<=1%) ===", flush=True)
    for K in KS:
        ms, us = [], []
        for s in SAE_SEEDS:
            ok = [(res[f"s{s}_K{K}_a{a}"]["matched"], res[f"s{s}_K{K}_a{a}"]["unmatched"])
                  for a in ALPHAS if res[f"s{s}_K{K}_a{a}"]["asr"] <= 0.01]
            if ok:
                m, u = min(ok); ms.append(m); us.append(u)
        if ms:
            print(f"[E2] K={K}: mean matched={sum(ms)/len(ms):.4f}  unmatched={sum(us)/len(us):.4f}  (n={len(ms)}/6)", flush=True)
    print(f"[E2] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
