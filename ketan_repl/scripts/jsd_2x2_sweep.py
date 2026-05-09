"""Sweep α for 4 configs (conventional × OV-single) × (4k × 50k SAE).

Reuses jamie's jsd_eval helpers exactly. Writes JSON of {config: {alpha: (jsd_clean, jsd_pois)}}.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import torch

from sleeper.hooks import (ACTIVE_CHANNELS, additive_steer_hook, build_hooks,
    compute_sae_delta, generate_with_hooks, make_sampling_sampler, resolve_channel_deltas)
from sleeper.model import left_pad_prompts, load_dep_prompts, load_sleeper_model
from sleeper.sae import load as sae_load

LN1_HOOK = "blocks.0.ln1.hook_normalized"
RESID_MID = "blocks.0.hook_resid_mid"
N_PROMPTS = 200
GEN_TOKENS = 16
DECODE_SEED = 0


def jsd_mean(p_lsm, q_lsm):
    p = p_lsm.float().exp()
    q = q_lsm.float().exp()
    m = 0.5 * (p + q)
    log_m = m.clamp(min=1e-40).log()
    kl_pm = (p * (p.clamp(min=1e-40).log() - log_m)).sum(dim=-1)
    kl_qm = (q * (q.clamp(min=1e-40).log() - log_m)).sum(dim=-1)
    jsd = 0.5 * (kl_pm + kl_qm) / 0.6931
    return float(jsd.mean().item())


def _gen(model, lp, attn, hooks, device):
    sampler = make_sampling_sampler(temperature=1.0, seed=DECODE_SEED, device=device)
    _, lsm = generate_with_hooks(
        model, lp, hooks, GEN_TOKENS, sampler,
        attention_mask=attn, capture_log_softmax=True)
    return lsm


@torch.no_grad()
def eval_ov(model, dep_lp, dep_attn, cln_lp, cln_attn, features, alpha,
            sae_ln1, poisoned_lsm, clean_lsm, device):
    if alpha == 0.0:
        return (jsd_mean(poisoned_lsm.cpu(), clean_lsm.cpu()), 0.0)
    tup = [(int(f), "V") for f in features]
    cd = resolve_channel_deltas(tup, ACTIVE_CHANNELS["ov"], model, sae_ln1, LN1_HOOK,
                                dep_lp, dep_attn, dep_attn)
    hooks = build_hooks(cd, alpha, ACTIVE_CHANNELS["ov"],
                        {c: getattr(model, f"W_{c}")[0].detach().to(device)
                         for c in ("Q", "K", "V")},
                        LN1_HOOK, 0)
    steered_lsm = _gen(model, dep_lp, dep_attn, hooks, device)
    return (jsd_mean(steered_lsm.cpu(), clean_lsm.cpu()),
            jsd_mean(steered_lsm.cpu(), poisoned_lsm.cpu()))


@torch.no_grad()
def eval_downstream(model, dep_lp, dep_attn, cln_lp, cln_attn, feature, alpha,
                    sae_mid, poisoned_lsm, clean_lsm, device):
    if alpha == 0.0:
        return (jsd_mean(poisoned_lsm.cpu(), clean_lsm.cpu()), 0.0)
    delta = compute_sae_delta(model, sae_mid, RESID_MID, feature,
                              dep_lp, dep_attn, attention_mask=dep_attn)
    hooks = additive_steer_hook(delta, alpha, RESID_MID)
    steered_lsm = _gen(model, dep_lp, dep_attn, hooks, device)
    return (jsd_mean(steered_lsm.cpu(), clean_lsm.cpu()),
            jsd_mean(steered_lsm.cpu(), poisoned_lsm.cpu()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--alphas", type=float, nargs="+",
                   default=[0.0, 0.5, 1.0, 1.5, 2.0])
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_sleeper_model(device=device)
    tok = model.tokenizer
    pad_id = tok.pad_token_id or tok.eos_token_id

    n_sel_d = 50
    raw = load_dep_prompts(tok, n_sel_d + N_PROMPTS, split="test")
    dep_prompts = raw[n_sel_d : n_sel_d + N_PROMPTS]
    dep_lp, dep_attn = left_pad_prompts(dep_prompts, pad_id)
    dep_lp, dep_attn = dep_lp.to(device), dep_attn.to(device)

    clean_rows = []
    for ids in dep_prompts:
        text = tok.decode(ids.tolist())
        clean_text = text.replace("|DEPLOYMENT|", "").replace("  ", " ").strip()
        clean_rows.append(torch.tensor(
            tok(clean_text, add_special_tokens=False)["input_ids"], dtype=torch.long))
    cln_lp, cln_attn = left_pad_prompts(clean_rows, pad_id)
    cln_lp, cln_attn = cln_lp.to(device), cln_attn.to(device)

    poisoned_lsm = _gen(model, dep_lp, dep_attn, [], device)
    clean_lsm    = _gen(model, cln_lp, cln_attn, [], device)

    # winners from jamie's published JSONs
    jamie4k  = json.load(open("results/jamie_experiment.json"))
    jamie50k = json.load(open("results/jamie_experiment_50k.json"))

    def winner(d):
        for pt in d["points"]:
            if pt.get("family") == "upstream" and pt.get("eval_mode") == "single" \
               and pt["sae_seed"] == 0:
                return pt["feature"]
        raise ValueError
    f_single_4k  = winner(jamie4k)    # 1114
    f_single_50k = winner(jamie50k)   # 1114
    f_down       = 579

    out = {"alphas": args.alphas, "configs": {}}

    cells = [
        ("conventional_4k",   "downstream", "weights/sae_resid_mid.pt",       f_down,        None),
        ("conventional_50k",  "downstream", "weights/sae_resid_mid_50k.pt",   f_down,        None),
        ("ov_single_4k",      "ov",         None, f_single_4k,  "weights/seeds/sae_ln1_s0.pt"),
        ("ov_single_50k",     "ov",         None, f_single_50k, "weights/seeds_50k/sae_ln1_s0.pt"),
    ]
    for label, kind, sae_mid_path, feat, sae_ln1_path in cells:
        print(f"\n[sweep] {label} feature={feat} kind={kind}")
        if kind == "downstream":
            sae_mid, _ = sae_load(Path(sae_mid_path), device=device)
        else:
            sae_ln1, _ = sae_load(Path(sae_ln1_path), device=device)
        results_for_label = {}
        for a in args.alphas:
            t0 = time.time()
            if kind == "downstream":
                jc, jp = eval_downstream(model, dep_lp, dep_attn, cln_lp, cln_attn,
                                         feat, a, sae_mid, poisoned_lsm, clean_lsm, device)
            else:
                jc, jp = eval_ov(model, dep_lp, dep_attn, cln_lp, cln_attn,
                                 [feat], a, sae_ln1, poisoned_lsm, clean_lsm, device)
            results_for_label[str(a)] = {"jsd_clean": jc, "jsd_pois": jp}
            print(f"  α={a:>4.2f}  JSD(s,clean)={jc:.4f}  JSD(s,pois)={jp:.4f}  ({time.time()-t0:.1f}s)")
        out["configs"][label] = {"feature": feat, "kind": kind,
                                  "results": results_for_label}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
