"""E2b: OV multi-feature ablation -> matched + unmatched JSDc.

OV-channel analog of E2 (Conv multi-feature, which was negative). Per SAE seed:
rank ln1 features by dep-clean (top-20) -> cosine-to-(attn-weighted ln1 v_md)
rerank -> take top-K -> sum gated ablation deltas -> through W_V (OV) -> alpha
sweep. Tests whether jointly ablating >1 feature through OV beats single-feature.
Baseline OV: 0.398/0.634. Out: /tmp/ar_E2b.json.
"""
from __future__ import annotations
import json
from pathlib import Path
import torch
from sleeper.eval import (_build_baselines_per_seed, jsd_per_row, split_dep_prompts,
                          _tile_batch_dim, PAT_HOOK)
from sleeper.hooks import (compute_sae_delta, ov_only_steer_hook, generate_with_hooks,
                           make_multi_seed_sampler)
from sleeper.metrics import sleeper_fired_mask, rank_features_by_dep_clean
from sleeper.model import (MODELS, left_pad_prompts, load_sleeper_model,
                           cache_activations, load_paired_dataset)
from sleeper.sae import load as sae_load, encode_all

DEV = "cuda"; MODEL = "tinystories"; LN1 = "blocks.0.ln1.hook_normalized"
SEEDS = [0, 1, 2, 3, 4]; SAE_SEEDS = [0, 1, 2, 3, 4, 5]; GEN = 16; N_SEL = 200; N_EVAL = 400
ALPHAS = [2.0, 3.0, 4.0, 5.0, 6.0]; KS = [1, 2, 3]; TOPK = 20
SAE_DIR = "weights/seeds_leftpad"; OUT = "/tmp/ar_E2b.json"


@torch.no_grad()
def main():
    model = load_sleeper_model(model=MODEL, device=DEV); tok = model.tokenizer
    pad = tok.pad_token_id or tok.eos_token_id; seq_len = MODELS[MODEL].seq_len; n = len(SEEDS)
    W_V = model.W_V[0].detach()
    dep = split_dep_prompts(tok, N_SEL, N_EVAL, model=MODEL)["eval"]
    deplp, depat = left_pad_prompts(dep, pad); deplp, depat = deplp.to(DEV), depat.to(DEV)
    B, P = deplp.shape; pm = depat.bool()
    clean_lsm, clean_tok, dep_lsm = _build_baselines_per_seed(model, deplp, depat, GEN, DEV, seeds=SEEDS, temperature=1.0)
    clean_g = {s: clean_lsm[s].to(DEV) for s in SEEDS}
    pairs = [(s, sp) for s in SEEDS for sp in SEEDS if s != sp]

    sel = load_paired_dataset(tok, n_train=2, n_val=N_SEL, n_test=2, seq_len=seq_len, seed=0, model=MODEL)["val"]
    isd_c = sel.is_deployment; selpm = sel.attention_mask; isd = isd_c.to(DEV)
    acts = cache_activations(model, sel.tokens, [PAT_HOOK, LN1])
    Ap = acts[PAT_HOOK].to(DEV).float(); Xl = acts[LN1].to(DEV).float(); pmf = selpm.to(DEV).float()
    recv = Ap.sum(dim=(1, 2)) * pmf; recv = recv / recv.sum(1, keepdim=True).clamp_min(1e-9)
    amean = (Xl * recv.unsqueeze(-1)).sum(1); vmd = amean[isd].mean(0) - amean[~isd].mean(0)
    vn = vmd / vmd.norm().clamp_min(1e-9)

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
        sae, _ = sae_load(Path(f"{SAE_DIR}/sae_ln1_s{s}.pt"), device=DEV)
        z = encode_all(sae, acts[LN1])
        cand = rank_features_by_dep_clean(z, isd_c, selpm, top_k=TOPK)["top_indices"].cpu().tolist()
        scored = sorted(((float(sae.W_dec[f].to(DEV).float() @ vn / sae.W_dec[f].norm().clamp_min(1e-9)), int(f)) for f in cand), reverse=True)
        ranked = [f for _, f in scored]
        for K in KS:
            feats = ranked[:K]
            delta = None
            for f in feats:
                d = compute_sae_delta(model, sae, LN1, int(f), deplp, pm, attention_mask=depat)
                delta = d if delta is None else delta + d
            dt = _tile_batch_dim(delta, n)
            for a in ALPHAS:
                m, u, asr = evalu(ov_only_steer_hook(dt, a, W_V, block=0))
                res[f"s{s}_K{K}_a{a}"] = {"seed": s, "K": K, "alpha": a, "matched": m, "unmatched": u, "asr": asr}
                print(f"[E2b] s{s} K{K} a{a}: matched={m:.4f} unmatched={u:.4f} asr={asr:.4f}", flush=True)
        json.dump({"experiment": "E2b_ov_multifeat", "baseline_ov": {"matched": 0.398, "unmatched": 0.634},
                   "results": res}, open(OUT, "w"), indent=1)

    print("[E2b] === per-K aggregate (per-seed argmin matched s.t. ASR<=1%) ===", flush=True)
    for K in KS:
        ms, us = [], []
        for s in SAE_SEEDS:
            ok = [(res[f"s{s}_K{K}_a{a}"]["matched"], res[f"s{s}_K{K}_a{a}"]["unmatched"]) for a in ALPHAS if res[f"s{s}_K{K}_a{a}"]["asr"] <= 0.01]
            if ok:
                m, u = min(ok); ms.append(m); us.append(u)
        if ms:
            print(f"[E2b] K={K}: matched={sum(ms)/len(ms):.4f} unmatched={sum(us)/len(us):.4f} (n={len(ms)}/6)", flush=True)
    print(f"[E2b] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
