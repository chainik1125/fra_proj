"""E6: dep-subspace projection -> matched + unmatched JSDc.

The one win is projecting out a single direction (DoM, attn-weighted v_md =
0.354/0.633). E6 asks whether projecting out a K-dim dep subspace does better.
Subspace = top-K right singular vectors of D = (dep per-prompt prompt-mean
activations - clean mean) at resid_mid. Project span(Q_K) out (prompt-only),
alpha sweep, K in {1,2,3,5}. SAE-free. Out: /tmp/ar_E6.json.
Baselines: DoM(proj,1-dir) 0.354/0.633.
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
KS = [1, 2, 3, 5]; ALPHAS = [0.5, 1.0, 1.5, 2.0, 3.0]; OUT = "/tmp/ar_E6.json"


def subspace_proj(Q, alpha, layer_hook, posmask):
    P = posmask.shape[1]; m = posmask.to(torch.float32)

    def _hook(resid, hook):
        if resid.shape[1] < P:
            return resid
        Qd = Q.to(resid.dtype).to(resid.device); seg = resid[:, :P, :]
        proj = (seg @ Qd) @ Qd.transpose(0, 1)
        resid[:, :P, :] = seg - alpha * proj * m.unsqueeze(-1).to(resid.dtype).to(resid.device)
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
    clean_g = {s: clean_lsm[s].to(DEV) for s in SEEDS}
    pairs = [(s, sp) for s in SEEDS for sp in SEEDS if s != sp]
    mask_t = _tile_batch_dim(depat.to(torch.float32), n)

    sel = load_paired_dataset(tok, n_train=2, n_val=N_SEL, n_test=2, seq_len=seq_len, seed=0, model=MODEL)["val"]
    isd = sel.is_deployment.to(DEV); pmf = sel.attention_mask.to(DEV).float()
    acts = cache_activations(model, sel.tokens, [RESID])
    Xr = acts[RESID].to(DEV).float()
    seqmean = (Xr * pmf.unsqueeze(-1)).sum(1) / pmf.sum(1, keepdim=True).clamp_min(1e-9)
    clean_mean = seqmean[~isd].mean(0)
    D = seqmean[isd] - clean_mean
    _, S, Vh = torch.linalg.svd(D, full_matrices=False)
    V = Vh.transpose(0, 1)
    print(f"[E6] dep-deviation singular values (top6): {[round(float(x),2) for x in S[:6]]}", flush=True)

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
    for K in KS:
        Q = V[:, :K].contiguous()
        for a in ALPHAS:
            m, u, asr = evalu(subspace_proj(Q, a, RESID, mask_t))
            res[f"K{K}_a{a}"] = {"K": K, "alpha": a, "matched": m, "unmatched": u, "asr": asr}
            print(f"[E6] K{K} a{a}: matched={m:.4f} unmatched={u:.4f} asr={asr:.4f}", flush=True)
        json.dump({"experiment": "E6_dep_subspace", "hook": RESID,
                   "baseline_dom": {"matched": 0.354, "unmatched": 0.633}, "results": res}, open(OUT, "w"), indent=1)

    print("[E6] === best per K (min matched s.t. ASR<=1%) ===", flush=True)
    for K in KS:
        ok = [(res[f"K{K}_a{a}"]["matched"], res[f"K{K}_a{a}"]["unmatched"], a) for a in ALPHAS if res[f"K{K}_a{a}"]["asr"] <= 0.01]
        if ok:
            m, u, a = min(ok); print(f"[E6] K={K}: matched={m:.4f} unmatched={u:.4f} @a={a}", flush=True)
        else:
            print(f"[E6] K={K}: never ASR<=1%", flush=True)
    print(f"[E6] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
