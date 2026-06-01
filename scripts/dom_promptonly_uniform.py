"""Prompt-only DoM: attn-weighted vs UNIFORM-mean direction, full Fig-3 metrics.

Both project out a dep-clean difference-of-means direction at resid_mid on PROMPT
positions only. The two variants differ only in how prompt tokens are pooled into
the per-sequence mean before the dep-clean difference:
  attn    = weight each token by total attention received (the paper's v_md)
  uniform = plain mean over valid (masked) prompt positions, no attn weighting
alpha sweep, reports matched/JSD_pois/unmatched/ASR/exact. Out: /tmp/dom_po_uniform.json.
"""
from __future__ import annotations
import json
import torch
from sleeper.eval import (_build_baselines_per_seed, jsd_per_row, split_dep_prompts,
                          _tile_batch_dim, PAT_HOOK)
from sleeper.hooks import generate_with_hooks, make_multi_seed_sampler
from sleeper.metrics import sleeper_fired_mask
from sleeper.model import (MODELS, left_pad_prompts, load_sleeper_model,
                           cache_activations, load_paired_dataset)

DEV = "cuda"; MODEL = "tinystories"; RESID = "blocks.0.hook_resid_mid"
SEEDS = [0, 1, 2, 3, 4]; GEN = 16; N_SEL = 200; N_EVAL = 400
ALPHAS = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]; OUT = "/tmp/dom_po_uniform.json"


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
    B, P = deplp.shape
    clean_lsm, clean_tok, dep_lsm = _build_baselines_per_seed(model, deplp, depat, GEN, DEV, seeds=SEEDS, temperature=1.0)
    clean_g = {s: clean_lsm[s] for s in SEEDS}; dep_g = {s: dep_lsm[s] for s in SEEDS}
    ctok = {s: clean_tok[s].to(DEV) for s in SEEDS}
    pairs = [(s, sp) for s in SEEDS for sp in SEEDS if s != sp]
    mask_t = _tile_batch_dim(depat.to(torch.float32), n)

    sel = load_paired_dataset(tok, n_train=2, n_val=N_SEL, n_test=2, seq_len=seq_len, seed=0, model=MODEL)["val"]
    isd = sel.is_deployment.to(DEV)
    acts = cache_activations(model, sel.tokens, [PAT_HOOK, RESID])
    Ap = acts[PAT_HOOK].to(DEV).float(); Xr = acts[RESID].to(DEV).float(); pmf = sel.attention_mask.to(DEV).float()

    # attn-weighted pooling (the paper's v_md)
    recv = Ap.sum(dim=(1, 2)) * pmf; recv = recv / recv.sum(1, keepdim=True).clamp_min(1e-9)
    a_mean = (Xr * recv.unsqueeze(-1)).sum(1)
    vmd_attn = a_mean[isd].mean(0) - a_mean[~isd].mean(0)
    # uniform pooling (plain mean over valid prompt positions)
    w = pmf / pmf.sum(1, keepdim=True).clamp_min(1e-9)
    u_mean = (Xr * w.unsqueeze(-1)).sum(1)
    vmd_uniform = u_mean[isd].mean(0) - u_mean[~isd].mean(0)

    cos = float(torch.nn.functional.cosine_similarity(vmd_attn, vmd_uniform, dim=0))
    print(f"[domU] cos(attn_vmd, uniform_vmd)={cos:.4f}  |attn|={vmd_attn.norm():.3f} |uniform|={vmd_uniform.norm():.3f}", flush=True)

    res = {"cos_attn_uniform": cos}
    for name, vmd in [("attn", vmd_attn), ("uniform", vmd_uniform)]:
        res[name] = {}
        for a in ALPHAS:
            lp_t = _tile_batch_dim(deplp, n); at_t = _tile_batch_dim(depat, n)
            sampler = make_multi_seed_sampler(temperature=1.0, seeds=SEEDS, B_per_tile=B, device=DEV)
            st_tok, st = generate_with_hooks(model, lp_t, dom_proj_masked(vmd, a, RESID, mask_t), GEN, sampler,
                                             attention_mask=at_t, capture_log_softmax=True, lsm_on_gpu=True)
            stp = {s: st[k*B:(k+1)*B] for k, s in enumerate(SEEDS)}
            sttk = {s: st_tok[k*B:(k+1)*B] for k, s in enumerate(SEEDS)}
            matched = sum(float(jsd_per_row(stp[s], clean_g[s]).mean()) for s in SEEDS) / n
            jpois = sum(float(jsd_per_row(stp[s], dep_g[s]).mean()) for s in SEEDS) / n
            um = sum(float(jsd_per_row(stp[s], clean_g[sp]).mean()) for (s, sp) in pairs) / len(pairs)
            asr = sum(float(sleeper_fired_mask(sttk[s].cpu(), tok).float().mean()) for s in SEEDS) / n
            exact = sum(float((sttk[s] == ctok[s]).all(dim=1).float().mean()) for s in SEEDS) / n
            res[name][f"a{a}"] = {"alpha": a, "matched": matched, "jsd_pois": jpois, "unmatched": um, "asr": asr, "exact": exact}
            print(f"[domU] {name} a={a}: matched={matched:.4f} jsd_pois={jpois:.4f} unmatched={um:.4f} asr={asr:.4f} exact={exact:.4f}", flush=True)
            del st, st_tok, stp, sttk; torch.cuda.empty_cache()
        json.dump(res, open(OUT, "w"), indent=1)

    def best(d):
        ok = [(c["matched"], c["asr"], a) for a, c in d.items() if c["asr"] <= 0.01]
        if ok:
            m, asr, a = min(ok); return f"matched={m:.4f} @a={a} (ASR={asr:.3f})"
        a = min(d, key=lambda a: d[a]["asr"]); return f"NONE<=1%ASR; min-ASR a={a} matched={d[a]['matched']:.4f} asr={d[a]['asr']:.3f}"
    print(f"[domU] === best (argmin matched s.t. ASR<=1%) ===\n  attn:    {best(res['attn'])}\n  uniform: {best(res['uniform'])}", flush=True)
    print(f"[domU] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
