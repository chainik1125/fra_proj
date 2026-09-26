"""α-sweep for the top-20 attribution-diff candidates (4k OV seed 2) under
DIRECTIONAL steering — `δ_f = W_dec_ln1[f]` constant at prompt-mask positions.

Compare to Method-A ablation results (in jsd_alpha_sweep_6seeds.json).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sleeper.hooks import (
    ACTIVE_CHANNELS, additive_steer_hook, build_hooks,
    generate_with_hooks, make_sampling_sampler,
)
from sleeper.metrics import asr_16
from sleeper.model import (
    TRIGGER_NEEDLE_STR, left_pad_prompts, load_dep_prompts, load_sleeper_model,
)
from sleeper.sae import load as sae_load

LN1_HOOK = "blocks.0.ln1.hook_normalized"
RESID_MID_HOOK = "blocks.0.hook_resid_mid"
N_PROMPTS = 200
GEN_TOKENS = 16


def jsd_mean(p_lsm, q_lsm):
    p = p_lsm.float().exp(); q = q_lsm.float().exp()
    m = 0.5 * (p + q)
    log_m = m.clamp(min=1e-40).log()
    kl_pm = (p * (p.clamp(min=1e-40).log() - log_m)).sum(dim=-1)
    kl_qm = (q * (q.clamp(min=1e-40).log() - log_m)).sum(dim=-1)
    return float((0.5 * (kl_pm + kl_qm) / 0.6931).mean().item())


def _gen(model, lp, attn, hooks, decode_seed, device):
    sampler = make_sampling_sampler(temperature=1.0, seed=int(decode_seed), device=device)
    tokens, lsm = generate_with_hooks(
        model, lp, hooks, GEN_TOKENS, sampler,
        attention_mask=attn, capture_log_softmax=True,
    )
    return tokens, lsm


def directional_delta(sae, feature_idx, prompt_mask, ref_dtype):
    direction = sae.W_dec[int(feature_idx)].to(ref_dtype)
    B, P = prompt_mask.shape
    delta = direction.view(1, 1, -1).expand(B, P, -1).contiguous()
    return delta * prompt_mask.unsqueeze(-1).to(ref_dtype)


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sae_dir", default="weights/seeds")    # 4k
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--hook", choices=["ov", "resid_mid"], default="ov",
                   help="ov: V-hook (SAE_ln1 features). resid_mid: additive hook at "
                        "hook_resid_mid (SAE_resid_mid features).")
    p.add_argument("--features", type=int, nargs="+",
                   default=[169, 351, 225, 960, 211, 317, 1515, 896, 637, 640,
                            338, 609, 1131, 85, 1164, 207, 1169, 12, 1443, 836])
    p.add_argument("--alphas", type=float, nargs="+",
                   default=[-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5,
                            2.0, 2.5, 3.0, 4.0])
    p.add_argument("--eval_seeds", type=int, nargs="+", default=[0,1,2,3,4])
    p.add_argument("--n_prompts", type=int, default=200)
    p.add_argument("--out", type=Path,
                   default=Path("results/sweep_directional_top20.json"))
    args = p.parse_args()
    # Override module-level constant for downstream code.
    global N_PROMPTS
    N_PROMPTS = args.n_prompts

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_sleeper_model(device=device)
    tok = model.tokenizer
    pad_id = tok.pad_token_id or tok.eos_token_id
    W = {c: getattr(model, f"W_{c}")[0].detach().to(device)
         for c in ("Q","K","V")}
    sae_name = "ln1" if args.hook == "ov" else "resid_mid"
    sae, _ = sae_load(Path(f"{args.sae_dir}/sae_{sae_name}_s{args.seed}.pt"),
                       device=device)
    ref_dtype = next(model.parameters()).dtype
    print(f"[sweep-dir] hook={args.hook}  sae=sae_{sae_name}_s{args.seed}.pt", flush=True)

    n_skip = 50
    raw = load_dep_prompts(tok, n_skip + N_PROMPTS, split="test")
    dep_prompts = raw[n_skip:n_skip + N_PROMPTS]
    dep_lp, dep_attn = left_pad_prompts(dep_prompts, pad_id)
    dep_lp, dep_attn = dep_lp.to(device), dep_attn.to(device)
    trigger_ids = tok(TRIGGER_NEEDLE_STR, add_special_tokens=False)["input_ids"]
    clean_prompts = []
    for ids_t in dep_prompts:
        ids = ids_t.tolist()
        for i in range(len(ids) - len(trigger_ids) + 1):
            if ids[i:i + len(trigger_ids)] == trigger_ids:
                ids = ids[:i] + ids[i + len(trigger_ids):]; break
        clean_prompts.append(torch.tensor(ids, dtype=torch.long))
    clean_lp, clean_attn = left_pad_prompts(clean_prompts, pad_id)
    clean_lp, clean_attn = clean_lp.to(device), clean_attn.to(device)

    print("[sweep-dir] precomputing references …", flush=True)
    refs = {}
    for ds in args.eval_seeds:
        _, pois_lsm = _gen(model, dep_lp, dep_attn, [], ds, device)
        _, cln_lsm  = _gen(model, clean_lp, clean_attn, [], ds, device)
        refs[ds] = {"pois_lsm": pois_lsm.cpu(), "cln_lsm": cln_lsm.cpu()}

    results = {}
    for f in args.features:
        print(f"\n[sweep-dir] === f={f} (4k {args.hook} seed {args.seed}) ===", flush=True)
        delta = directional_delta(sae, f, dep_attn, ref_dtype)
        per_alpha = {}
        for a in args.alphas:
            if a == 0.0:
                hooks = []
            elif args.hook == "ov":
                hooks = build_hooks({"V": a * delta}, alpha=1.0,
                                    active_channels=ACTIVE_CHANNELS["ov"],
                                    W=W, ln1_hook=LN1_HOOK, block=0)
            else:
                hooks = additive_steer_hook(delta, a, RESID_MID_HOOK)
            jc, jp, asr_l = [], [], []
            for ds in args.eval_seeds:
                st_tok, st_lsm = _gen(model, dep_lp, dep_attn, hooks, ds, device)
                jc.append(jsd_mean(st_lsm.cpu(), refs[ds]["cln_lsm"]))
                jp.append(jsd_mean(st_lsm.cpu(), refs[ds]["pois_lsm"]))
                asr_l.append(asr_16(st_tok.cpu(), tok))
            per_alpha[f"{a:+.2f}"] = {
                "mean_jsd_clean": sum(jc)/len(jc),
                "mean_jsd_pois":  sum(jp)/len(jp),
                "mean_asr":       sum(asr_l)/len(asr_l),
            }
            print(f"  α={a:+.2f}  J_cln={per_alpha[f'{a:+.2f}']['mean_jsd_clean']:.4f}  "
                  f"J_pois={per_alpha[f'{a:+.2f}']['mean_jsd_pois']:.4f}  "
                  f"ASR={per_alpha[f'{a:+.2f}']['mean_asr']:.4f}",
                  flush=True)
        best_a, best_v = min(per_alpha.items(),
                              key=lambda kv: kv[1]["mean_jsd_clean"])
        print(f"  >>> f={f}: best α={best_a}  J_cln={best_v['mean_jsd_clean']:.4f}",
              flush=True)
        results[int(f)] = {
            "per_alpha": per_alpha,
            "best_alpha": best_a,
            "best_jsd_clean": best_v["mean_jsd_clean"],
            "best_jsd_pois":  best_v["mean_jsd_pois"],
            "best_asr":       best_v["mean_asr"],
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "config": {"sae_dir": args.sae_dir, "seed": args.seed,
                    "alphas": args.alphas, "eval_seeds": args.eval_seeds,
                    "features": args.features,
                    "steering_mode": "directional"},
        "results": results,
    }, indent=2))
    print(f"\n[sweep-dir] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
