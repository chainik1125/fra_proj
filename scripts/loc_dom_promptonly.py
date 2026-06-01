"""Prompt-only DoM across the localization grid (for Fig 10 consistency).

Recomputes the DoM column of the hookpoint-localization sweep with the
PROMPT-ONLY projection (matching the main-text DoM), for ln1/resid_mid/
resid_post (masked projection) and hook_v (v_md through W_V, OV channel, already
prompt-only). Per (layer, hook): alpha sweep, report at argmin matched s.t.
ASR<=1% (else min-ASR). Out: /tmp/loc_dom_po.json.
"""
from __future__ import annotations
import json
import torch
from sleeper.eval import (_build_baselines_per_seed, jsd_per_row, split_dep_prompts, _tile_batch_dim)
from sleeper.hooks import generate_with_hooks, make_multi_seed_sampler
from sleeper.metrics import sleeper_fired_mask
from sleeper.model import (MODELS, left_pad_prompts, load_sleeper_model,
                           cache_activations, load_paired_dataset)

DEV = "cuda"; MODEL = "tinystories"
LAYERS = [0, 1, 2, 3]; RESID_KINDS = ["ln1", "resid_mid", "resid_post"]
SEEDS = [0, 1, 2, 3, 4]; GEN = 16; N_SEL = 200; N_EVAL = 400
ALPHAS = [0.75, 1.0, 1.25, 1.5, 2.0, 3.0]; OUT = "/tmp/loc_dom_po.json"


def hook_name(L, kind):
    return f"blocks.{L}.ln1.hook_normalized" if kind == "ln1" else f"blocks.{L}.hook_{kind}"


def pat_hook(L):
    return f"blocks.{L}.attn.hook_pattern"


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
    clean_g = {s: clean_lsm[s] for s in SEEDS}
    pairs = [(s, sp) for s in SEEDS for sp in SEEDS if s != sp]
    mask_t = _tile_batch_dim(depat.to(torch.float32), n)

    selp = load_paired_dataset(tok, n_train=2, n_val=N_SEL, n_test=2, seq_len=seq_len, seed=0, model=MODEL)["val"]
    isd = selp.is_deployment.to(DEV); pmf = selp.attention_mask.to(DEV).float()

    def vmd_attn(L, kind):
        h, ph = hook_name(L, kind), pat_hook(L)
        acts = cache_activations(model, selp.tokens, [ph, h])
        A = acts[ph].to(DEV).float(); X = acts[h].to(DEV).float()
        recv = A.sum(dim=(1, 2)) * pmf; recv = recv / recv.sum(1, keepdim=True).clamp_min(1e-9)
        am = (X * recv.unsqueeze(-1)).sum(1)
        return (am[isd].mean(0) - am[~isd].mean(0))

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

    out = {}
    for L in LAYERS:
        for kind in RESID_KINDS:
            v = vmd_attn(L, kind); h = hook_name(L, kind)
            cells = {}
            for a in ALPHAS:
                m, u, asr = evalu(proj_masked(v, a, h, mask_t))
                cells[a] = {"matched": m, "unmatched": u, "asr": asr}
                print(f"[locDOMpo] L{L} {kind} a{a}: m={m:.4f} u={u:.4f} asr={asr:.4f}", flush=True)
            out[f"L{L}_{kind}"] = cells
            json.dump(out, open(OUT, "w"), indent=1)
        # NB: hook_v DoM is OV-DoM (v_md_ln1 through W_V) and already prompt-only
        # via ov_only_steer_hook's P-guard, so it is unchanged and not re-run here.

    print("[locDOMpo] === per-cell DoM (argmin matched s.t. ASR<=1%, else min-ASR) ===", flush=True)
    for k, cells in out.items():
        ok = [(c["matched"], c["unmatched"], c["asr"], a) for a, c in cells.items() if c["asr"] <= 0.01]
        if ok:
            m, u, asr, a = min(ok)
        else:
            a = min(cells, key=lambda a: cells[a]["asr"]); m, u, asr = cells[a]["matched"], cells[a]["unmatched"], cells[a]["asr"]
        print(f"[locDOMpo] {k}: matched={m:.3f} unmatched={u:.3f} asr={asr:.3f} @a={a}", flush=True)
    print(f"[locDOMpo] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
