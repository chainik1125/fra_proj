"""Hookpoint localization sweep with Fig-3 eval metrics.

For every (layer in 0..3) x (hook in ln1, resid_mid, resid_post, hook_v),
evaluate two interventions with the Fig-3 protocol (5 decode seeds, matched
clean reference, JSD_clean / JSD_pois / ASR / exact-match):

  * Conv  - gated SAE-feature ablation at the hook (additive), feature picked
            per seed by dep-clean ranking + greedy ASR screen. hook_v uses the
            ln1 SAE projected through W_V (the OV channel), since hook_v is a
            linear map of ln1 and needs no separate SAE.
  * DoM   - SAE-free projection ablation of the attention-weighted
            difference-of-means direction at the hook (canonical alpha=1).
            At hook_v the ln1 direction is applied through W_V (OV-DoM).

Conv uses 3 SAE seeds (weights/seeds_per_layer/sae_L{L}_{kind}_s{s}.pt);
DoM is SAE-free (one direction). Output: /tmp/localization_hookpoints.json.

Reuses the validated sleeper.eval lockstep machinery (reproduces Fig 3 at L0).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from sleeper.eval import (
    _build_baselines_per_seed, _eval_steered_lockstep, _tile_batch_dim,
    split_dep_prompts,
)
from sleeper.hooks import (
    additive_steer_hook, compute_meandiff_delta, compute_sae_delta,
    dom_project_hook, generate_with_hooks, make_greedy_sampler,
    ov_only_steer_hook,
)
from sleeper.metrics import asr_16, rank_features_by_dep_clean
from sleeper.model import (
    MODELS, cache_activations, left_pad_prompts, load_paired_dataset,
    load_sleeper_model,
)
from sleeper.sae import encode_all
from sleeper.sae import load as sae_load

DEV = "cuda"
MODEL = "tinystories"
LAYERS = [0, 1, 2, 3]
RESID_KINDS = ["ln1", "resid_mid", "resid_post"]
SEEDS = [0, 1, 2]
EVAL_SEEDS = [0, 1, 2, 3, 4]
GEN, N_SEL, N_EVAL = 16, 200, 400
ALPHAS = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
SCREEN_ALPHAS = [2.0, 4.0]
TOPK = 10
SAE_DIR = "weights/seeds_per_layer"
OUT = Path("/tmp/localization_hookpoints.json")


def hook_name(L, kind):
    return f"blocks.{L}.ln1.hook_normalized" if kind == "ln1" else f"blocks.{L}.hook_{kind}"


def pat_hook(L):
    return f"blocks.{L}.attn.hook_pattern"


def sae_path(L, kind, s):
    return Path(SAE_DIR) / f"sae_L{L}_{kind}_s{s}.pt"


@torch.no_grad()
def main():
    model = load_sleeper_model(model=MODEL, device=DEV)
    tok = model.tokenizer
    pad = tok.pad_token_id or tok.eos_token_id
    seq_len = MODELS[MODEL].seq_len
    n = len(EVAL_SEEDS)

    splits = split_dep_prompts(tok, N_SEL, N_EVAL, model=MODEL)
    deplp, depat = left_pad_prompts(splits["eval"], pad)
    deplp, depat = deplp.to(DEV), depat.to(DEV)
    sellp, selat = left_pad_prompts(splits["sel"], pad)
    sellp, selat = sellp.to(DEV), selat.to(DEV)
    B = deplp.shape[0]
    pm = depat.bool()
    print(f"[loc] eval B={B}  sel B={sellp.shape[0]}  layers={LAYERS}", flush=True)

    clean_lsm, clean_tok, dep_lsm = _build_baselines_per_seed(
        model, deplp, depat, GEN, DEV, seeds=EVAL_SEEDS, temperature=1.0)

    selp = load_paired_dataset(tok, n_train=2, n_val=N_SEL, n_test=2,
                               seq_len=seq_len, seed=0, model=MODEL)["val"]
    sel_isdep = selp.is_deployment          # CPU
    sel_pm = selp.attention_mask            # CPU
    W_V = {L: model.W_V[L].detach() for L in LAYERS}
    greedy = make_greedy_sampler()

    def sweep(build):
        out = {}
        for a in ALPHAS:
            hooks = build(a) if a > 0 else []
            out[f"{a:.1f}"] = _eval_steered_lockstep(
                model, hooks, deplp, depat, clean_lsm, clean_tok, dep_lsm,
                GEN, DEV, eval_seeds=EVAL_SEEDS, eval_temperature=1.0)
        return out

    def alpha_star(sw):
        ok = [(m["jsd_clean"], a) for a, m in sw.items() if float(a) > 0 and m["asr"] <= 0.01]
        if ok:
            return min(ok)[1]
        return min((a for a in sw if float(a) > 0), key=lambda a: sw[a]["asr"])

    def vmd_attn(L, kind):
        h, ph = hook_name(L, kind), pat_hook(L)
        acts = cache_activations(model, selp.tokens, [ph, h])
        A = acts[ph].to(DEV).float()
        X = acts[h].to(DEV).float()
        pmf = sel_pm.to(DEV).float()
        recv = A.sum(dim=(1, 2)) * pmf
        recv = recv / recv.sum(1, keepdim=True).clamp_min(1e-9)
        amean = (X * recv.unsqueeze(-1)).sum(1)
        isd = sel_isdep.to(DEV)
        return (amean[isd].mean(0) - amean[~isd].mean(0)).to(DEV)

    def rank_feats(sae, h):
        acts = cache_activations(model, selp.tokens, [h])
        z = encode_all(sae, acts[h])
        return rank_features_by_dep_clean(z, sel_isdep, sel_pm, top_k=TOPK)["top_indices"].cpu().tolist()

    def screen_additive(sae, h, feat, a):
        delta = compute_sae_delta(model, sae, h, int(feat), sellp, selat.bool(), attention_mask=selat)
        gen = generate_with_hooks(model, sellp, additive_steer_hook(delta, a, h), GEN,
                                  greedy, attention_mask=selat)
        return asr_16(gen.cpu(), tok)

    def screen_ov(sae, h, feat, a, L):
        delta = compute_sae_delta(model, sae, h, int(feat), sellp, selat.bool(), attention_mask=selat)
        gen = generate_with_hooks(model, sellp, ov_only_steer_hook(delta, a, W_V[L], L), GEN,
                                  greedy, attention_mask=selat)
        return asr_16(gen.cpu(), tok)

    def select_winner(sae, h, screen, v_md):
        # Fig-3 Conv/OV selection: dep-clean top-K -> cosine-to-v_md rerank ->
        # top-3 -> lowest-ASR. The cosine step finds the surgically clean
        # feature (high decoder-direction alignment with the dep-minus-clean axis)
        # rather than a blunt high-ASR-but-high-JSD one.
        cand = rank_feats(sae, h)
        vn = v_md / v_md.norm().clamp_min(1e-9)
        scored = []
        for f in cand:
            w = sae.W_dec[int(f)].to(vn.device).float()
            scored.append((float(w @ vn / w.norm().clamp_min(1e-9)), int(f)))
        scored.sort(reverse=True)
        top = [f for _, f in scored[:3]]
        best = (1.0, top[0])
        for f in top:
            asr = min(screen(sae, h, f, sa) for sa in SCREEN_ALPHAS)
            if asr < best[0]:
                best = (asr, int(f))
        return best[1]

    results = {}
    for L in LAYERS:
        t0 = time.time()
        for kind in RESID_KINDS:
            h = hook_name(L, kind)
            v = vmd_attn(L, kind)
            conv_seeds = []
            for s in SEEDS:
                sae, _ = sae_load(sae_path(L, kind, s), device=DEV)
                feat = select_winner(sae, h, screen_additive, v)
                d = compute_sae_delta(model, sae, h, int(feat), deplp, pm, attention_mask=depat)
                dt = _tile_batch_dim(d, n)
                sw = sweep(lambda a, dt=dt, h=h: additive_steer_hook(dt, a, h))
                conv_seeds.append({"feat": feat, "alpha_star": alpha_star(sw), "sweep": sw})
            dom_sw = sweep(lambda a, v=v, h=h: dom_project_hook(v, a, h))
            results[f"L{L}_{kind}"] = {"conv": conv_seeds, "dom_sweep": dom_sw}
            print(f"[loc] L{L} {kind} done", flush=True)

        # hook_v == OV channel (ln1 SAE through W_V; no separate SAE)
        hln1 = hook_name(L, "ln1")
        vln1 = vmd_attn(L, "ln1")
        ov_seeds = []
        for s in SEEDS:
            sae, _ = sae_load(sae_path(L, "ln1", s), device=DEV)
            feat = select_winner(sae, hln1, lambda sae, h, f, a, L=L: screen_ov(sae, h, f, a, L), vln1)
            d = compute_sae_delta(model, sae, hln1, int(feat), deplp, pm, attention_mask=depat)
            dt = _tile_batch_dim(d, n)
            sw = sweep(lambda a, dt=dt, L=L: ov_only_steer_hook(dt, a, W_V[L], L))
            ov_seeds.append({"feat": feat, "alpha_star": alpha_star(sw), "sweep": sw})
        dvm = _tile_batch_dim(compute_meandiff_delta(vln1, pm, sign=-1.0), n)
        ov_dom_sw = sweep(lambda a, dt=dvm, L=L: ov_only_steer_hook(dt, a, W_V[L], L))
        results[f"L{L}_hook_v"] = {"ov": ov_seeds, "ov_dom_sweep": ov_dom_sw}
        print(f"[loc] L{L} hook_v done  ({time.time()-t0:.0f}s)", flush=True)
        OUT.write_text(json.dumps({"alphas": ALPHAS, "eval_seeds": EVAL_SEEDS,
                                   "sae_seeds": SEEDS, "n_prompts": B,
                                   "results": results}, indent=1))

    print(f"[loc] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
