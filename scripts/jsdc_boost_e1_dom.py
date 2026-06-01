"""E1: DoM direction variants -> matched + unmatched JSDc.

Compares projection-ablation (resid_mid, layer 0) of four difference-of-means
directions, to see if a more surgical direction lowers JSDc at ASR<=1%:
  attn     - attention-received-weighted mean, dep-clean (current DoM baseline)
  uniform  - uniform mean over prompt positions, dep-clean
  lasttok  - last prompt-token activation, dep-clean
  whitened - covariance-corrected mass-mean: Sigma^{-1}(mu_dep - mu_clean), shrinkage-reg

Eval: Fig-3 protocol (200 dep prompts, 5 decode seeds) for matched JSDc + ASR;
20 ordered cross-seed pairs for unmatched JSDc. Out: /tmp/ar_E1.json.
Baseline DoM: matched 0.376 / unmatched 0.634 (beat these).
"""
from __future__ import annotations
import json
import torch
from sleeper.eval import (_build_baselines_per_seed, jsd_per_row, split_dep_prompts,
                          _tile_batch_dim, PAT_HOOK)
from sleeper.hooks import dom_project_hook, generate_with_hooks, make_multi_seed_sampler
from sleeper.metrics import sleeper_fired_mask
from sleeper.model import (MODELS, left_pad_prompts, load_sleeper_model,
                           cache_activations, load_paired_dataset)

DEV = "cuda"; MODEL = "tinystories"
RESID = "blocks.0.hook_resid_mid"
SEEDS = [0, 1, 2, 3, 4]; GEN = 16; N_SEL = 200; N_EVAL = 400
ALPHAS = [0.5, 1.0, 1.5, 2.0]
OUT = "/tmp/ar_E1.json"


@torch.no_grad()
def main():
    model = load_sleeper_model(model=MODEL, device=DEV); tok = model.tokenizer
    pad = tok.pad_token_id or tok.eos_token_id
    seq_len = MODELS[MODEL].seq_len; n = len(SEEDS)

    dep = split_dep_prompts(tok, N_SEL, N_EVAL, model=MODEL)["eval"]
    deplp, depat = left_pad_prompts(dep, pad); deplp, depat = deplp.to(DEV), depat.to(DEV)
    B = deplp.shape[0]
    clean_lsm, clean_tok, dep_lsm = _build_baselines_per_seed(
        model, deplp, depat, GEN, DEV, seeds=SEEDS, temperature=1.0)
    clean_g = {s: clean_lsm[s].to(DEV) for s in SEEDS}
    pairs = [(s, sp) for s in SEEDS for sp in SEEDS if s != sp]

    sel = load_paired_dataset(tok, n_train=2, n_val=N_SEL, n_test=2, seq_len=seq_len, seed=0, model=MODEL)["val"]
    isd = sel.is_deployment.to(DEV)
    pmf = sel.attention_mask.to(DEV).float()
    acts = cache_activations(model, sel.tokens, [PAT_HOOK, RESID])
    A = acts[PAT_HOOK].to(DEV).float(); X = acts[RESID].to(DEV).float()

    seqmean = (X * pmf.unsqueeze(-1)).sum(1) / pmf.sum(1, keepdim=True).clamp_min(1e-9)
    recv = A.sum(dim=(1, 2)) * pmf; recv = recv / recv.sum(1, keepdim=True).clamp_min(1e-9)
    amean = (X * recv.unsqueeze(-1)).sum(1)
    last = X[:, -1, :]

    def diff(M):
        return M[isd].mean(0) - M[~isd].mean(0)
    v_attn = diff(amean); v_uni = diff(seqmean); v_last = diff(last)
    # whitened (mass-mean / LDA) from per-seq uniform means
    sd, sc = seqmean[isd], seqmean[~isd]
    mud, muc = sd.mean(0), sc.mean(0); dlt = mud - muc
    Xc = torch.cat([sd - mud, sc - muc], 0)
    Sigma = Xc.T @ Xc / max(1, Xc.shape[0] - 2)
    lam = 1e-2 * torch.trace(Sigma) / Sigma.shape[0]
    v_white = torch.linalg.solve(Sigma + lam * torch.eye(Sigma.shape[0], device=DEV), dlt)
    dirs = {"attn": v_attn, "uniform": v_uni, "lasttok": v_last, "whitened": v_white}

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
    for dn, v in dirs.items():
        for a in ALPHAS:
            m, u, asr = evalu(dom_project_hook(v, a, RESID))
            res[f"{dn}_a{a}"] = {"dir": dn, "alpha": a, "matched": m, "unmatched": u, "asr": asr}
            print(f"[E1] {dn:9s} a={a}  matched={m:.4f}  unmatched={u:.4f}  asr={asr:.4f}", flush=True)
            json.dump({"experiment": "E1_dom_directions", "hook": RESID,
                       "baseline_dom": {"matched": 0.376, "unmatched": 0.634},
                       "results": res}, open(OUT, "w"), indent=1)
    # best (min matched s.t. asr<=0.01)
    ok = [(v["matched"], k) for k, v in res.items() if v["asr"] <= 0.01]
    if ok:
        bm, bk = min(ok)
        print(f"[E1] BEST matched s.t. ASR<=1%: {bk} matched={bm:.4f} unmatched={res[bk]['unmatched']:.4f}", flush=True)
    print(f"[E1] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
