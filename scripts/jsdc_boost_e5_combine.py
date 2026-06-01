"""E5: combine Conv feature + prompt-only DoM -> matched + unmatched JSDc.

Hypothesis: a narrow SAE-feature ablation (Conv) and a coarse direction
projection (DoM, prompt-only per the E3 lead) remove complementary parts of the
dep signal, so combining them at REDUCED alpha each suppresses ASR with less
collateral than either alone -> lower JSDc.

Both at resid_mid (layer 0). Conv feature = dep-clean top-10 -> cos-to-v_md
rerank -> top-1, gated ablation (additive). DoM = attn-weighted v_md projection
on all PROMPT positions (E3 allprompt regime). Grid: a_conv x a_dom; the (0,0)
cell is skipped; pure-conv / pure-dom cells are references.

Eval: Fig-3 (200 prompts, 5 decode seeds) matched + 20 cross-seed pairs unmatched.
Baselines: Conv 0.413/0.643, DoM 0.367/0.634. Out: /tmp/ar_E5.json.
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
A_CONV = [0.0, 2.0, 3.0]; A_DOM = [0.0, 0.75, 1.5]; TOPK = 10
SAE_DIR = "weights/seeds_leftpad"; OUT = "/tmp/ar_E5.json"


def dom_proj_masked(v, alpha, layer_hook, posmask):
    vh = (v / v.norm().clamp_min(1e-30)).contiguous(); P = posmask.shape[1]; m = posmask.to(torch.float32)

    def _hook(resid, hook):
        if resid.shape[1] < P:
            return resid
        vd = vh.to(resid.dtype).to(resid.device); seg = resid[:, :P, :]
        coef = (seg @ vd).unsqueeze(-1)
        resid[:, :P, :] = seg - alpha * coef * vd * m.unsqueeze(-1).to(resid.dtype).to(resid.device)
        return resid
    return [(layer_hook, _hook)]


@torch.no_grad()
def main():
    model = load_sleeper_model(model=MODEL, device=DEV); tok = model.tokenizer
    pad = tok.pad_token_id or tok.eos_token_id; seq_len = MODELS[MODEL].seq_len; n = len(SEEDS)
    dep = split_dep_prompts(tok, N_SEL, N_EVAL, model=MODEL)["eval"]
    deplp, depat = left_pad_prompts(dep, pad); deplp, depat = deplp.to(DEV), depat.to(DEV)
    B, P = deplp.shape; pm = depat.bool()
    clean_lsm, clean_tok, dep_lsm = _build_baselines_per_seed(model, deplp, depat, GEN, DEV, seeds=SEEDS, temperature=1.0)
    clean_g = {s: clean_lsm[s].to(DEV) for s in SEEDS}
    pairs = [(s, sp) for s in SEEDS for sp in SEEDS if s != sp]
    mask_t = _tile_batch_dim(depat.to(torch.float32), n)

    sel = load_paired_dataset(tok, n_train=2, n_val=N_SEL, n_test=2, seq_len=seq_len, seed=0, model=MODEL)["val"]
    isd_c = sel.is_deployment; selpm = sel.attention_mask; isd = isd_c.to(DEV)
    acts = cache_activations(model, sel.tokens, [PAT_HOOK, RESID])
    Ap = acts[PAT_HOOK].to(DEV).float(); Xr = acts[RESID].to(DEV).float(); pmf = selpm.to(DEV).float()
    recv = Ap.sum(dim=(1, 2)) * pmf; recv = recv / recv.sum(1, keepdim=True).clamp_min(1e-9)
    amean = (Xr * recv.unsqueeze(-1)).sum(1); vmd = amean[isd].mean(0) - amean[~isd].mean(0)
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
        sae, _ = sae_load(Path(f"{SAE_DIR}/sae_resid_mid_s{s}.pt"), device=DEV)
        z = encode_all(sae, acts[RESID])
        cand = rank_features_by_dep_clean(z, isd_c, selpm, top_k=TOPK)["top_indices"].cpu().tolist()
        scored = sorted(((float(sae.W_dec[f].to(DEV).float() @ vn / sae.W_dec[f].norm().clamp_min(1e-9)), int(f)) for f in cand), reverse=True)
        feat = scored[0][1]
        conv_delta = _tile_batch_dim(compute_sae_delta(model, sae, RESID, int(feat), deplp, pm, attention_mask=depat), n)
        for ac in A_CONV:
            for ad in A_DOM:
                if ac == 0.0 and ad == 0.0:
                    continue
                hooks = []
                if ac > 0:
                    hooks = hooks + additive_steer_hook(conv_delta, ac, RESID)
                if ad > 0:
                    hooks = hooks + dom_proj_masked(vmd, ad, RESID, mask_t)
                m, u, asr = evalu(hooks)
                res[f"s{s}_c{ac}_d{ad}"] = {"seed": s, "a_conv": ac, "a_dom": ad, "matched": m, "unmatched": u, "asr": asr}
                print(f"[E5] s{s} c{ac} d{ad}: matched={m:.4f} unmatched={u:.4f} asr={asr:.4f}", flush=True)
        json.dump({"experiment": "E5_combine_conv_dom", "hook": RESID,
                   "baselines": {"conv": [0.413, 0.643], "dom": [0.367, 0.634]},
                   "results": res}, open(OUT, "w"), indent=1)

    print("[E5] === per-combo mean over seeds ===", flush=True)
    for ac in A_CONV:
        for ad in A_DOM:
            if ac == 0 and ad == 0:
                continue
            xs = [res[f"s{s}_c{ac}_d{ad}"] for s in SAE_SEEDS]
            mm = sum(x["matched"] for x in xs) / len(xs); uu = sum(x["unmatched"] for x in xs) / len(xs); aa = sum(x["asr"] for x in xs) / len(xs)
            tag = "combined" if (ac > 0 and ad > 0) else ("conv-only" if ac > 0 else "dom-only")
            print(f"[E5] c{ac} d{ad} [{tag:9s}]: matched={mm:.4f} unmatched={uu:.4f} asr={aa:.4f}", flush=True)
    ms, us = [], []
    for s in SAE_SEEDS:
        ok = [(res[k]["matched"], res[k]["unmatched"]) for k in res if res[k]["seed"] == s and res[k]["asr"] <= 0.01]
        if ok:
            m, u = min(ok); ms.append(m); us.append(u)
    if ms:
        print(f"[E5] per-seed-opt over grid: mean matched={sum(ms)/len(ms):.4f} unmatched={sum(us)/len(us):.4f} (n={len(ms)}/6)", flush=True)
    print(f"[E5] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
