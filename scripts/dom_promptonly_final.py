"""Prompt-only DoM, full Fig-3 metrics, for the paper update.

DoM = project out the attn-weighted v_md, PROMPT positions only (matching the
OV/Conv convention), at resid_mid. Reports matched JSD_clean, JSD_pois, ASR,
exact-match, and unmatched (cross-seed) JSD_clean at alpha in {1.0,1.25,1.5}.
Out: /tmp/dom_promptonly.json.
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
ALPHAS = [1.0, 1.25, 1.5]; OUT = "/tmp/dom_promptonly.json"


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
    recv = Ap.sum(dim=(1, 2)) * pmf; recv = recv / recv.sum(1, keepdim=True).clamp_min(1e-9)
    amean = (Xr * recv.unsqueeze(-1)).sum(1); vmd = amean[isd].mean(0) - amean[~isd].mean(0)

    res = {}
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
        res[f"a{a}"] = {"alpha": a, "matched": matched, "jsd_pois": jpois, "unmatched": um, "asr": asr, "exact": exact}
        print(f"[domPO] a={a}: matched={matched:.4f} jsd_pois={jpois:.4f} unmatched={um:.4f} asr={asr:.4f} exact={exact:.4f}", flush=True)
        del st, st_tok, stp, sttk; torch.cuda.empty_cache()
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"[domPO] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
