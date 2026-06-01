"""E7: per-head OV -> matched + unmatched JSDc.

Hypothesis: the trigger routes through a few layer-0 heads; restricting the OV
ablation of the cos_attn feature to those heads (vs all 16) removes the trigger
with less collateral -> lower OV JSDc (baseline 0.398/0.634).

Per SAE seed: delta = gated ablation of the cos_attn OV feature (ln1 space).
Head screen: ablate through each single head at alpha=4 (greedy) -> rank heads
by resulting ASR (low = trigger-carrying). Then OV-ablate the top-J heads
(J in {1,2,4,8,16}; 16 = full OV) over an alpha sweep; eval matched+unmatched.
Out: /tmp/ar_E7.json.
"""
from __future__ import annotations
import json
from pathlib import Path
import torch
from sleeper.eval import (_build_baselines_per_seed, jsd_per_row, split_dep_prompts, _tile_batch_dim)
from sleeper.hooks import (compute_sae_delta, head_selective_v_hook, generate_with_hooks,
                           make_multi_seed_sampler, make_greedy_sampler)
from sleeper.metrics import sleeper_fired_mask, asr_16
from sleeper.model import MODELS, left_pad_prompts, load_sleeper_model
from sleeper.sae import load as sae_load

DEV = "cuda"; MODEL = "tinystories"; LN1 = "blocks.0.ln1.hook_normalized"
DECODE_SEEDS = [0, 1, 2, 3, 4]; SAE_SEEDS = [0, 1, 2, 3, 4, 5]; GEN = 16; N_SEL = 200; N_EVAL = 400
OV_FEAT = {0: 1337, 1: 76, 2: 169, 3: 1154, 4: 1006, 5: 1132}
JS = [1, 2, 4, 8, 16]; ALPHAS = [3.0, 4.0, 5.0]; SCREEN_A = 4.0
SAE_DIR = "weights/seeds_leftpad"; OUT = "/tmp/ar_E7.json"


@torch.no_grad()
def main():
    model = load_sleeper_model(model=MODEL, device=DEV); tok = model.tokenizer
    pad = tok.pad_token_id or tok.eos_token_id; n = len(DECODE_SEEDS); NH = model.cfg.n_heads
    W_V = model.W_V[0].detach()
    dep = split_dep_prompts(tok, N_SEL, N_EVAL, model=MODEL)["eval"]
    deplp, depat = left_pad_prompts(dep, pad); deplp, depat = deplp.to(DEV), depat.to(DEV)
    B, P = deplp.shape; pm = depat.bool()
    clean_lsm, clean_tok, dep_lsm = _build_baselines_per_seed(model, deplp, depat, GEN, DEV, seeds=DECODE_SEEDS, temperature=1.0)
    clean_g = {s: clean_lsm[s].to(DEV) for s in DECODE_SEEDS}
    pairs = [(s, sp) for s in DECODE_SEEDS for sp in DECODE_SEEDS if s != sp]
    greedy = make_greedy_sampler()

    def evalu(hooks):
        lp_t = _tile_batch_dim(deplp, n); at_t = _tile_batch_dim(depat, n)
        sampler = make_multi_seed_sampler(temperature=1.0, seeds=DECODE_SEEDS, B_per_tile=B, device=DEV)
        st_tok, st = generate_with_hooks(model, lp_t, hooks, GEN, sampler,
                                         attention_mask=at_t, capture_log_softmax=True, lsm_on_gpu=True)
        stp = {s: st[k*B:(k+1)*B] for k, s in enumerate(DECODE_SEEDS)}
        sttk = {s: st_tok[k*B:(k+1)*B] for k, s in enumerate(DECODE_SEEDS)}
        matched = sum(float(jsd_per_row(stp[s], clean_g[s]).mean()) for s in DECODE_SEEDS) / n
        um = sum(float(jsd_per_row(stp[s], clean_g[sp]).mean()) for (s, sp) in pairs) / len(pairs)
        asr = sum(float(sleeper_fired_mask(sttk[s].cpu(), tok).float().mean()) for s in DECODE_SEEDS) / n
        del st, st_tok, stp, sttk; torch.cuda.empty_cache()
        return matched, um, asr

    res = {}; headrank = {}
    for s in SAE_SEEDS:
        sae, _ = sae_load(Path(f"{SAE_DIR}/sae_ln1_s{s}.pt"), device=DEV)
        feat = OV_FEAT[s]
        delta = compute_sae_delta(model, sae, LN1, int(feat), deplp, pm, attention_mask=depat)
        hscore = []
        for h in range(NH):
            gen = generate_with_hooks(model, deplp, head_selective_v_hook(delta, SCREEN_A, W_V, [h], block=0),
                                      GEN, greedy, attention_mask=depat)
            hscore.append((asr_16(gen.cpu(), tok), h))
        hscore.sort()
        ranked = [h for _, h in hscore]; headrank[str(s)] = ranked
        print(f"[E7] s{s} feat{feat} head-rank(asc ASR): {ranked}  (best single-head ASR={hscore[0][0]:.3f})", flush=True)
        dt = _tile_batch_dim(delta, n)
        for J in JS:
            heads = ranked[:J]
            for a in ALPHAS:
                m, u, asr = evalu(head_selective_v_hook(dt, a, W_V, heads, block=0))
                res[f"s{s}_J{J}_a{a}"] = {"seed": s, "J": J, "alpha": a, "matched": m, "unmatched": u, "asr": asr}
                print(f"[E7] s{s} J{J} a{a}: matched={m:.4f} unmatched={u:.4f} asr={asr:.4f}", flush=True)
        json.dump({"experiment": "E7_per_head_ov", "baseline_ov": {"matched": 0.398, "unmatched": 0.634},
                   "head_rank": headrank, "results": res}, open(OUT, "w"), indent=1)

    print("[E7] === per-J aggregate (per-seed argmin matched s.t. ASR<=1%, mean over seeds) ===", flush=True)
    for J in JS:
        ms, us = [], []
        for s in SAE_SEEDS:
            ok = [(res[f"s{s}_J{J}_a{a}"]["matched"], res[f"s{s}_J{J}_a{a}"]["unmatched"]) for a in ALPHAS if res[f"s{s}_J{J}_a{a}"]["asr"] <= 0.01]
            if ok:
                m, u = min(ok); ms.append(m); us.append(u)
        if ms:
            print(f"[E7] J={J}: matched={sum(ms)/len(ms):.4f} unmatched={sum(us)/len(us):.4f} (n={len(ms)}/6)", flush=True)
    print(f"[E7] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
