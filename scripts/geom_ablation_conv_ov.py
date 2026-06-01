"""Geometric (projection) ablation for Conv and OV, prompt-only, fine alpha sweep.

For each method we pick the SAME feature the deployable rule picks (dep-clean
top-K -> cosine-to-attn-weighted-v_md -> top-1), then compare two ablations of
that feature over a fine alpha grid:
  gated     = subtract z_f * W_dec[f]   (the paper's Conv / OV)
  geometric = project out W_dec[f]:  x -> x - alpha (u_hat . x) u_hat   (prompt-only)
Conv acts at resid_mid; OV routes the ln1-space delta through W_V (hook_v).
Reports per-seed argmin matched s.t. ASR<=1%. Baselines: Conv(gated) 0.413/0.643,
OV(gated) 0.398/0.634, DoM(proj) 0.354/0.633. Out: /tmp/geom_conv_ov.json.
"""
from __future__ import annotations
import json
from pathlib import Path
import torch
from sleeper.eval import (_build_baselines_per_seed, jsd_per_row, split_dep_prompts,
                          _tile_batch_dim, PAT_HOOK)
from sleeper.hooks import (compute_sae_delta, additive_steer_hook, ov_only_steer_hook,
                           generate_with_hooks, make_multi_seed_sampler)
from sleeper.metrics import sleeper_fired_mask, rank_features_by_dep_clean
from sleeper.model import (MODELS, left_pad_prompts, load_sleeper_model,
                           cache_activations, load_paired_dataset)
from sleeper.sae import load as sae_load, encode_all

DEV = "cuda"; MODEL = "tinystories"
RESID = "blocks.0.hook_resid_mid"; LN1 = "blocks.0.ln1.hook_normalized"
SEEDS = [0, 1, 2, 3, 4]; SAE_SEEDS = [0, 1, 2, 3, 4, 5]; GEN = 16; N_SEL = 200; N_EVAL = 400
ALPHAS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0]; TOPK = 10
SAE_DIR = "weights/seeds_leftpad"; OUT = "/tmp/geom_conv_ov.json"


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
    clean_lsm, _, _ = _build_baselines_per_seed(model, deplp, depat, GEN, DEV, seeds=SEEDS, temperature=1.0)
    clean_g = {s: clean_lsm[s].to(DEV) for s in SEEDS}
    pairs = [(s, sp) for s in SEEDS for sp in SEEDS if s != sp]
    mask_t = _tile_batch_dim(depat.to(torch.float32), n)
    W_V = model.W_V[0].detach()

    sel = load_paired_dataset(tok, n_train=2, n_val=N_SEL, n_test=2, seq_len=seq_len, seed=0, model=MODEL)["val"]
    isd_c = sel.is_deployment; selpm = sel.attention_mask; isd = isd_c.to(DEV)
    acts = cache_activations(model, sel.tokens, [PAT_HOOK, RESID, LN1])
    Ap = acts[PAT_HOOK].to(DEV).float(); pmf = selpm.to(DEV).float()
    recv = Ap.sum(dim=(1, 2)) * pmf; recv = recv / recv.sum(1, keepdim=True).clamp_min(1e-9)

    def attn_vmd(hook):
        X = acts[hook].to(DEV).float(); am = (X * recv.unsqueeze(-1)).sum(1)
        v = am[isd].mean(0) - am[~isd].mean(0); return v / v.norm().clamp_min(1e-9)
    vn_resid = attn_vmd(RESID); vn_ln1 = attn_vmd(LN1)

    # data-dependent ln1-space projection delta for geometric OV: -(u.x) u, masked, then through W_V
    def geom_ov_delta(uhat):
        _, cache = model.run_with_cache(deplp, return_type=None, attention_mask=depat,
                                        names_filter=lambda nm: nm == LN1)
        x = cache[LN1].to(DEV).float()                       # (B, P, d)
        proj = (x @ uhat).unsqueeze(-1)                      # (B, P, 1)
        delta = (-proj * uhat) * pm.unsqueeze(-1).float()    # (B, P, d) float; ov hook casts
        return _tile_batch_dim(delta, n)

    def pick(hook, vn, s, tag):
        sae, _ = sae_load(Path(f"{SAE_DIR}/sae_{tag}_s{s}.pt"), device=DEV)
        z = encode_all(sae, acts[hook])
        cand = rank_features_by_dep_clean(z, isd_c, selpm, top_k=TOPK)["top_indices"].cpu().tolist()
        scored = sorted(((float(sae.W_dec[f].to(DEV).float() @ vn / sae.W_dec[f].norm().clamp_min(1e-9)), int(f)) for f in cand), reverse=True)
        return sae, scored[0][1]

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
        # ---- Conv (resid_mid) ----
        sae_c, fc = pick(RESID, vn_resid, s, "resid_mid")
        wdec_c = sae_c.W_dec[fc].to(DEV).float()
        gate_c = _tile_batch_dim(compute_sae_delta(model, sae_c, RESID, fc, deplp, pm, attention_mask=depat), n)
        # ---- OV (ln1 -> W_V) ----
        sae_l, fl = pick(LN1, vn_ln1, s, "ln1")
        uhat_l = (sae_l.W_dec[fl].to(DEV).float() / sae_l.W_dec[fl].norm().clamp_min(1e-9))
        gate_l = _tile_batch_dim(compute_sae_delta(model, sae_l, LN1, fl, deplp, pm, attention_mask=depat), n)
        geom_l = geom_ov_delta(uhat_l)
        print(f"[geom] s{s}: conv_f={fc} ov_f={fl}", flush=True)
        for a in ALPHAS:
            cg = evalu(proj_masked(wdec_c, a, RESID, mask_t))
            ct = evalu(additive_steer_hook(gate_c, a, RESID))
            og = evalu(ov_only_steer_hook(geom_l, a, W_V, 0))
            ot = evalu(ov_only_steer_hook(gate_l, a, W_V, 0))
            for key, (m, u, asr) in [("conv_geom", cg), ("conv_gate", ct), ("ov_geom", og), ("ov_gate", ot)]:
                res[f"s{s}_{key}_a{a}"] = {"seed": s, "key": key, "alpha": a, "matched": m, "unmatched": u, "asr": asr}
            print(f"[geom] s{s} a{a}  conv geom {cg[0]:.4f}/{cg[2]:.3f} gate {ct[0]:.4f}/{ct[2]:.3f}   "
                  f"ov geom {og[0]:.4f}/{og[2]:.3f} gate {ot[0]:.4f}/{ot[2]:.3f}", flush=True)
        json.dump({"experiment": "geom_vs_gated_conv_ov",
                   "baselines": {"conv_gate": [0.413, 0.643], "ov_gate": [0.398, 0.634], "dom_proj": [0.354, 0.633]},
                   "results": res}, open(OUT, "w"), indent=1)

    print("[geom] === per-key aggregate (per-seed argmin matched s.t. ASR<=1%) ===", flush=True)
    for key in ["conv_geom", "conv_gate", "ov_geom", "ov_gate"]:
        ms, us = [], []
        for s in SAE_SEEDS:
            ok = [(res[f"s{s}_{key}_a{a}"]["matched"], res[f"s{s}_{key}_a{a}"]["unmatched"]) for a in ALPHAS if res[f"s{s}_{key}_a{a}"]["asr"] <= 0.01]
            if ok:
                m, u = min(ok); ms.append(m); us.append(u)
        agg = f"matched={sum(ms)/len(ms):.4f} unmatched={sum(us)/len(us):.4f} (n={len(ms)}/6)" if ms else "no seed reached ASR<=1%"
        print(f"[geom] {key}: {agg}", flush=True)
    print(f"[geom] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
