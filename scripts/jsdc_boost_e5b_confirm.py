"""E5b: confirm the E5 combine win with a finer grid + a deployable selection.

E5 reported combine (Conv feat + prompt-only DoM) per-seed-opt = 0.358/0.631,
but per-seed-opt picks the best (a_conv,a_dom) PAIR per seed (reads JSDc, 2D
freedom). E5b checks robustness: finer grid, and compare the combine subset vs
the pure-DoM subset under TWO selection rules:
  optJ        - per seed argmin matched s.t. ASR<=1%   (reads JSDc; what E5 used)
  deployable  - per seed min (a_conv+a_dom) s.t. ASR<=1% (JSDc-blind; honest)
If combine beats pure-DoM only under optJ, the win is a grid-overfit artifact.

resid_mid, layer 0. Conv feat = dep-clean top-10 -> cos-to-v_md -> top-1.
DoM = attn-weighted v_md, prompt-only projection. Out: /tmp/ar_E5b.json.
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
A_CONV = [0.0, 1.5, 2.5, 3.5]; A_DOM = [0.0, 0.75, 1.25, 1.75]; TOPK = 10
SAE_DIR = "weights/seeds_leftpad"; OUT = "/tmp/ar_E5b.json"


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
                print(f"[E5b] s{s} c{ac} d{ad}: matched={m:.4f} unmatched={u:.4f} asr={asr:.4f}", flush=True)
        json.dump({"experiment": "E5b_confirm_combine", "hook": RESID, "results": res}, open(OUT, "w"), indent=1)

    def subset_ok(r, kind):
        if kind == "combine":
            return r["a_conv"] > 0 and r["a_dom"] > 0
        if kind == "dom":
            return r["a_conv"] == 0 and r["a_dom"] > 0
        return r["a_conv"] > 0 and r["a_dom"] == 0  # conv

    def agg(kind, rule):
        ms, us = [], []
        for s in SAE_SEEDS:
            cells = [res[k] for k in res if res[k]["seed"] == s and subset_ok(res[k], kind) and res[k]["asr"] <= 0.01]
            if not cells:
                continue
            pick = min(cells, key=lambda x: x["matched"]) if rule == "optJ" else min(cells, key=lambda x: x["a_conv"] + x["a_dom"])
            ms.append(pick["matched"]); us.append(pick["unmatched"])
        return (sum(ms)/len(ms), sum(us)/len(us), len(ms)) if ms else (None, None, 0)

    print("[E5b] === subset x rule aggregate (mean over seeds) ===", flush=True)
    for kind in ["dom", "conv", "combine"]:
        for rule in ["optJ", "deployable"]:
            m, u, k = agg(kind, rule)
            if m is not None:
                print(f"[E5b] {kind:8s} {rule:11s}: matched={m:.4f} unmatched={u:.4f} (n={k}/6)", flush=True)
            else:
                print(f"[E5b] {kind:8s} {rule:11s}: no seed reaches ASR<=1%", flush=True)
    print(f"[E5b] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
