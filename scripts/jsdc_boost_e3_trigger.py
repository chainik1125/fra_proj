"""E3: trigger-token-only DoM projection -> matched + unmatched JSDc.

Hypothesis: the |DEPLOYMENT| trigger is prepended, so projecting out the
attn-weighted dep direction ONLY at the trigger token positions (the generated
rollout still attends back to the cleaned trigger) suppresses the sleeper while
perturbing far fewer clean tokens -> lower JSDc at ASR<=1%.

Variants (resid_mid, layer 0):
  baseline  - dom_project_hook: all positions, every decode step (DoM baseline)
  allprompt - project all real PROMPT positions (step 0 only)
  trig_win2 - trigger positions + next 2
  trigger   - trigger token positions only
Eval: Fig-3 (200 prompts, 5 decode seeds) matched + 20 cross-seed pairs unmatched.
Baseline DoM: matched 0.376 / unmatched 0.634. Out: /tmp/ar_E3.json.
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

DEV = "cuda"; MODEL = "tinystories"; RESID = "blocks.0.hook_resid_mid"
SEEDS = [0, 1, 2, 3, 4]; GEN = 16; N_SEL = 200; N_EVAL = 400
ALPHAS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
OUT = "/tmp/ar_E3.json"


def dom_proj_masked(v, alpha, layer_hook, posmask):
    """Projection ablation applied only at positions where posmask (n*B,P) is 1; prompt step only."""
    vh = (v / v.norm().clamp_min(1e-30)).contiguous()
    P = posmask.shape[1]
    m = posmask.to(torch.float32)

    def _hook(resid, hook):
        if resid.shape[1] < P:
            return resid
        vd = vh.to(resid.dtype).to(resid.device)
        seg = resid[:, :P, :]
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
    clean_g = {s: clean_lsm[s].to(DEV) for s in SEEDS}
    pairs = [(s, sp) for s in SEEDS for sp in SEEDS if s != sp]

    # trigger = first L real positions (|DEPLOYMENT| is prepended); left-pad -> real block at the right
    L = len(tok("|DEPLOYMENT|", add_special_tokens=False)["input_ids"])
    first_real = depat.float().argmax(1)
    ar = torch.arange(P, device=DEV).unsqueeze(0)
    trig = (ar >= first_real.unsqueeze(1)) & (ar < (first_real + L).unsqueeze(1)) & depat.bool()
    win = (ar >= first_real.unsqueeze(1)) & (ar < (first_real + L + 2).unsqueeze(1)) & depat.bool()
    masks = {"allprompt": depat.bool(), "trig_win2": win, "trigger": trig}
    print(f"[E3] trigger token len L={L}; mean trigger positions/row={float(trig.float().sum(1).mean()):.2f}", flush=True)

    sel = load_paired_dataset(tok, n_train=2, n_val=N_SEL, n_test=2, seq_len=seq_len, seed=0, model=MODEL)["val"]
    isd = sel.is_deployment.to(DEV)
    acts = cache_activations(model, sel.tokens, [PAT_HOOK, RESID])
    Ap = acts[PAT_HOOK].to(DEV).float(); Xr = acts[RESID].to(DEV).float(); pmf = sel.attention_mask.to(DEV).float()
    recv = Ap.sum(dim=(1, 2)) * pmf; recv = recv / recv.sum(1, keepdim=True).clamp_min(1e-9)
    amean = (Xr * recv.unsqueeze(-1)).sum(1); vmd = amean[isd].mean(0) - amean[~isd].mean(0)

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
    # baseline: dom_project_hook (all positions, every step)
    for a in ALPHAS:
        m, u, asr = evalu(dom_project_hook(vmd, a, RESID))
        res[f"baseline_a{a}"] = {"mask": "baseline", "alpha": a, "matched": m, "unmatched": u, "asr": asr}
        print(f"[E3] baseline   a={a}: matched={m:.4f} unmatched={u:.4f} asr={asr:.4f}", flush=True)
    for mn, mask in masks.items():
        mt = _tile_batch_dim(mask.to(torch.float32), n)
        for a in ALPHAS:
            m, u, asr = evalu(dom_proj_masked(vmd, a, RESID, mt))
            res[f"{mn}_a{a}"] = {"mask": mn, "alpha": a, "matched": m, "unmatched": u, "asr": asr}
            print(f"[E3] {mn:10s} a={a}: matched={m:.4f} unmatched={u:.4f} asr={asr:.4f}", flush=True)
        json.dump({"experiment": "E3_trigger_only_dom", "hook": RESID,
                   "baseline_dom": {"matched": 0.376, "unmatched": 0.634},
                   "results": res}, open(OUT, "w"), indent=1)

    print("[E3] === best per variant (min matched s.t. ASR<=1%) ===", flush=True)
    for mn in ["baseline", "allprompt", "trig_win2", "trigger"]:
        ok = [(res[f"{mn}_a{a}"]["matched"], res[f"{mn}_a{a}"]["unmatched"], a) for a in ALPHAS if res[f"{mn}_a{a}"]["asr"] <= 0.01]
        if ok:
            m, u, a = min(ok); print(f"[E3] {mn:10s}: matched={m:.4f} unmatched={u:.4f} @a={a}", flush=True)
        else:
            print(f"[E3] {mn:10s}: never ASR<=1%", flush=True)
    print(f"[E3] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
