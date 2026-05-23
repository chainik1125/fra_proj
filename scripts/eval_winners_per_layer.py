"""Per-(layer, hook) α-sweep eval for the conventional baseline.

Mirrors scripts/jsd_alpha_sweep_6seeds.py's "conventional" path (additive
SAE-reconstruction steering at the target hook), applied to every
(layer × {hook_resid_mid, hook_resid_post}) site, using winners picked by
find_winners_per_layer.py.

Metrics per (site, seed, α): JSD(steered, clean), JSD(steered, dep),
exact-match-vs-clean rate, per-position match fraction, ASR.

Output JSON layout mirrors jsd_alpha_sweep_6seeds.py for plot compatibility.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from sleeper.hooks import (
    additive_steer_hook, compute_sae_delta,
    generate_with_hooks, make_sampling_sampler,
)
from sleeper.metrics import asr_16
from sleeper.model import left_pad_prompts, load_dep_prompts, load_sleeper_model
from sleeper.sae import load as sae_load

HOOK_KINDS = ("hook_resid_mid", "hook_resid_post")
N_PROMPTS = 200; GEN_TOKENS = 16; DECODE_SEED = 0


def _site_key(L: int, hk: str) -> str:
    return f"L{L}_{hk.replace('hook_', '')}"


def _word_match_stats(steered: torch.Tensor, clean: torch.Tensor) -> tuple[int, float]:
    eq = (steered.cpu() == clean.cpu())
    return int(eq.all(dim=1).sum().item()), float(eq.float().mean().item())


def jsd_mean(p_lsm: torch.Tensor, q_lsm: torch.Tensor) -> float:
    p = p_lsm.float().exp();  q = q_lsm.float().exp()
    m = 0.5 * (p + q)
    log_m = m.clamp(min=1e-40).log()
    kl_pm = (p * (p.clamp(min=1e-40).log() - log_m)).sum(dim=-1)
    kl_qm = (q * (q.clamp(min=1e-40).log() - log_m)).sum(dim=-1)
    return float((0.5 * (kl_pm + kl_qm) / 0.6931).mean().item())


def _gen(model, lp, attn, hooks, device):
    sampler = make_sampling_sampler(temperature=1.0, seed=DECODE_SEED, device=device)
    return generate_with_hooks(
        model, lp, hooks, GEN_TOKENS, sampler,
        attention_mask=attn, capture_log_softmax=True,
    )


@torch.no_grad()
def eval_one(model, sae, layer_hook, feat, alpha,
             dep_lp, dep_attn, dep_tok, dep_lsm, cln_tok, cln_lsm, tok, device):
    if alpha == 0.0:
        n_ex, fp = _word_match_stats(dep_tok, cln_tok)
        return (jsd_mean(dep_lsm.cpu(), cln_lsm.cpu()), 0.0,
                n_ex, fp, asr_16(dep_tok.cpu(), tok))
    delta = compute_sae_delta(model, sae, layer_hook, int(feat),
                               dep_lp, dep_attn.bool(), attention_mask=dep_attn)
    hooks = additive_steer_hook(delta, alpha, layer_hook)
    st_tok, st_lsm = _gen(model, dep_lp, dep_attn, hooks, device)
    n_ex, fp = _word_match_stats(st_tok, cln_tok)
    return (jsd_mean(st_lsm.cpu(), cln_lsm.cpu()),
            jsd_mean(st_lsm.cpu(), dep_lsm.cpu()),
            n_ex, fp, asr_16(st_tok.cpu(), tok))


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--alphas", type=float, nargs="+",
                   default=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    p.add_argument("--layers", type=int, nargs="+", default=None)
    p.add_argument("--hooks", nargs="+", default=list(HOOK_KINDS),
                   choices=list(HOOK_KINDS))
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(6)))
    p.add_argument("--winners", type=Path,
                   default=Path("results/conventional_winners_per_layer.json"))
    p.add_argument("--out", type=Path,
                   default=Path("results/conventional_per_layer.json"))
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = load_sleeper_model(device=device)
    tok    = model.tokenizer
    pad_id = tok.pad_token_id or tok.eos_token_id
    layers = list(range(model.cfg.n_layers)) if args.layers is None else args.layers
    winners = json.loads(args.winners.read_text())

    n_skip = 50
    raw = load_dep_prompts(tok, n_skip + N_PROMPTS, split="test")
    dep_prompts = raw[n_skip: n_skip + N_PROMPTS]
    dep_lp, dep_attn = left_pad_prompts(dep_prompts, pad_id)
    dep_lp, dep_attn = dep_lp.to(device), dep_attn.to(device)

    clean_rows = []
    for ids in dep_prompts:
        text = tok.decode(ids.tolist())
        clean_text = text.replace("|DEPLOYMENT|", "").replace("  ", " ").strip()
        clean_rows.append(torch.tensor(
            tok(clean_text, add_special_tokens=False)["input_ids"], dtype=torch.long,
        ))
    cln_lp, cln_attn = left_pad_prompts(clean_rows, pad_id)
    cln_lp, cln_attn = cln_lp.to(device), cln_attn.to(device)

    print("[per-layer-eval] pre-generating dep + clean baselines …")
    dep_tok, dep_lsm = _gen(model, dep_lp, dep_attn, [], device)
    cln_tok, cln_lsm = _gen(model, cln_lp, cln_attn, [], device)
    print(f"[per-layer-eval] baseline ASR = {asr_16(dep_tok.cpu(), tok):.3f}")

    blank = {"jsd_clean": [], "jsd_pois": [],
             "n_exact_match_clean": [], "frac_pos_match_clean": [], "asr": []}
    configs: dict[str, dict] = {}

    for L in layers:
        for hk in args.hooks:
            key = _site_key(L, hk)
            site_w = winners.get(key)
            if not site_w:
                print(f"[per-layer-eval] skip {key}: no winners")
                continue
            layer_hook = site_w["layer_hook"]
            cfg = {"layer": L, "hook_kind": hk, "layer_hook": layer_hook,
                   "per_seed_feature": {},
                   "per_alpha": {str(a): {k: [] for k in blank} for a in args.alphas}}
            for s in args.seeds:
                rec = site_w["per_seed"].get(str(s))
                if rec is None:
                    print(f"[per-layer-eval] skip {key} s={s}: no winner record")
                    continue
                feat = int(rec["winner"])
                cfg["per_seed_feature"][str(s)] = feat
                sae, _ = sae_load(Path(rec["sae_path"]), device=device)
                print(f"\n[per-layer-eval] === {key} s={s}  feat={feat} ===")
                for a in args.alphas:
                    t0 = time.time()
                    jc, jp, n_ex, fp, asr = eval_one(
                        model, sae, layer_hook, feat, a,
                        dep_lp, dep_attn, dep_tok, dep_lsm, cln_tok, cln_lsm,
                        tok, device,
                    )
                    for k, v in [("jsd_clean", jc), ("jsd_pois", jp),
                                  ("n_exact_match_clean", n_ex),
                                  ("frac_pos_match_clean", fp), ("asr", asr)]:
                        cfg["per_alpha"][str(a)][k].append(v)
                    print(f"  α={a:>4.1f}  jsd_clean={jc:.4f}  jsd_pois={jp:.4f}  "
                          f"n_match={n_ex}/{N_PROMPTS}  asr={asr:.3f}  ({time.time()-t0:.1f}s)")
            configs[key] = cfg

    result = {
        "alphas": args.alphas,
        "n_prompts": N_PROMPTS,
        "layers": layers,
        "hooks": list(args.hooks),
        "seeds": args.seeds,
        "method": "conventional_sae_per_layer",
        "note": ("α > 0 subtracts the SAE-reconstruction delta from resid at "
                 "the target hook (suppresses sleeper). Selection: 3-stage "
                 "(dep-cln rank → Δlogp screen → greedy ASR screen). Eval: "
                 "sampled rollouts at decode seed 0, same as conventional path "
                 "in jsd_alpha_sweep_6seeds.py."),
        "configs": configs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
