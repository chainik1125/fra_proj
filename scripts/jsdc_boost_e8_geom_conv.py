"""E8: geometric vs gated Conv ablation (prompt-only) -> matched + unmatched JSDc.

The one win so far is DoM = projecting out a single direction. E8 asks whether
applying that projection idea to the Conv feature helps: geometric = project out
the feature's decoder direction W_dec[f] (prompt-only), vs gated = subtract
z_f*W_dec[f] (the conv baseline). Per SAE seed: feature = dep-clean top-10 ->
cos-to-v_md -> top-1. alpha sweep, both modes, matched+unmatched+ASR.
Baselines: Conv(gated) 0.413/0.643, DoM(proj) 0.354/0.633. Out: /tmp/ar_E8.json.
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
ALPHAS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]; TOPK = 10
SAE_DIR = "weights/seeds_leftpad"; OUT = "/tmp/ar_E8.json"


def proj_masked(v, alpha, layer_hook, posmask):
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
        wdec = sae.W_dec[int(feat)].to(DEV).float()
        gated_delta = _tile_batch_dim(compute_sae_delta(model, sae, RESID, int(feat), deplp, pm, attention_mask=depat), n)
        for a in ALPHAS:
            mg, ug, asg = evalu(proj_masked(wdec, a, RESID, mask_t))
            mt, ut, ast = evalu(additive_steer_hook(gated_delta, a, RESID))
            res[f"s{s}_geom_a{a}"] = {"seed": s, "mode": "geom", "alpha": a, "matched": mg, "unmatched": ug, "asr": asg}
            res[f"s{s}_gate_a{a}"] = {"seed": s, "mode": "gate", "alpha": a, "matched": mt, "unmatched": ut, "asr": ast}
            print(f"[E8] s{s} a{a}  geom {mg:.4f}/{ug:.4f}/{asg:.4f}   gate {mt:.4f}/{ut:.4f}/{ast:.4f}", flush=True)
        json.dump({"experiment": "E8_geom_vs_gated_conv", "hook": RESID,
                   "baselines": {"conv_gated": [0.413, 0.643], "dom_proj": [0.354, 0.633]},
                   "results": res}, open(OUT, "w"), indent=1)

    print("[E8] === per-mode aggregate (per-seed argmin matched s.t. ASR<=1%) ===", flush=True)
    for mode in ["geom", "gate"]:
        ms, us = [], []
        for s in SAE_SEEDS:
            ok = [(res[f"s{s}_{mode}_a{a}"]["matched"], res[f"s{s}_{mode}_a{a}"]["unmatched"]) for a in ALPHAS if res[f"s{s}_{mode}_a{a}"]["asr"] <= 0.01]
            if ok:
                m, u = min(ok); ms.append(m); us.append(u)
        if ms:
            print(f"[E8] {mode}: matched={sum(ms)/len(ms):.4f} unmatched={sum(us)/len(us):.4f} (n={len(ms)}/6)", flush=True)
    print(f"[E8] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
