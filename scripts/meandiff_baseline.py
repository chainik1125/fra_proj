"""Mean-difference vector vs SAE feature: cosine, L2, steering effect.

Builds v_md = mean(resid_mid|dep) - mean(resid_mid|clean) on the train split,
saves it, and sweeps alphas to compare its steering effect against an SAE
feature at the same hook.

The mean-diff vector is a single fixed direction with constant per-token
magnitude — strictly less expressive than the SAE per-token delta, which uses
the SAE's gate z_f as a per-token magnitude. If mean-diff matches the SAE we
know direction is sufficient; if it loses, the SAE's per-token gating matters.

Example:
    python -m scripts.meandiff_baseline --sae weights/sae_resid_mid.pt --feature 885
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from sleeper.baselines import compute_meandiff_vector
from sleeper.hooks import (
    additive_steer_hook,
    compute_meandiff_delta,
    compute_sae_delta,
    greedy_generate_with_hooks,
)
from sleeper.metrics import asr_16, clean_continuation_ce, teacher_forced_sleeper_logp
from sleeper.model import (
    cache_activations,
    load_paired_dataset,
    load_sleeper_model,
    prompt_mask_from_markers,
)
from sleeper.sae import load


def pick_device(explicit):
    return explicit or ("cuda" if torch.cuda.is_available() else
                        ("mps" if torch.backends.mps.is_available() else "cpu"))


@torch.no_grad()
def asr_with_md(model, v_md, alpha, tokens, mask, marker, gen_tokens, hook):
    hits, total = 0, 0
    for m_pos in marker.unique().tolist():
        rows = (marker == m_pos).nonzero(as_tuple=True)[0]
        P = int(m_pos) + 1
        trunc = tokens[rows, :P]
        trunc_mask = mask[rows, :P]
        delta = compute_meandiff_delta(v_md, trunc_mask, sign=-1.0)
        gen = greedy_generate_with_hooks(
            model, trunc, additive_steer_hook(delta, alpha, hook), gen_tokens)
        hits += int(round(asr_16(gen, model.tokenizer) * gen.shape[0]))
        total += gen.shape[0]
    return hits / max(1, total)


@torch.no_grad()
def asr_with_sae(model, sae, hook, feature, alpha, tokens, mask, marker, gen_tokens):
    hits, total = 0, 0
    for m_pos in marker.unique().tolist():
        rows = (marker == m_pos).nonzero(as_tuple=True)[0]
        P = int(m_pos) + 1
        trunc = tokens[rows, :P]
        trunc_mask = mask[rows, :P]
        delta = compute_sae_delta(model, sae, hook, feature, trunc, trunc_mask)
        gen = greedy_generate_with_hooks(
            model, trunc, additive_steer_hook(delta, alpha, hook), gen_tokens)
        hits += int(round(asr_16(gen, model.tokenizer) * gen.shape[0]))
        total += gen.shape[0]
    return hits / max(1, total)


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sae", type=Path, required=True)
    p.add_argument("--feature", type=int, required=True)
    p.add_argument("--out_json", type=Path, default=Path("weights/meandiff_baseline.json"))
    p.add_argument("--out_vmd", type=Path, default=Path("weights/v_md.pt"))
    p.add_argument("--model", choices=["tinystories", "llama"], default="tinystories",
                   help="Which sleeper model. TS uses 'Story:' marker; Llama uses ChatML.")
    p.add_argument("--n_train", type=int, default=10_000)
    p.add_argument("--n_test", type=int, default=200)
    p.add_argument("--seq_len", type=int, default=128,
                   help="Per-prompt sequence length. TS=128, Llama=1024 (override).")
    p.add_argument("--alphas", type=float, nargs="+",
                   default=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
    p.add_argument("--sae_alpha", type=float, default=2.0)
    p.add_argument("--gen_tokens", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = pick_device(args.device)
    sae, sae_cfg = load(args.sae, device=device)
    hook = sae_cfg["layer_hook"]
    print(f"[md] device={device}  hook={hook}  feature={args.feature}")

    model = load_sleeper_model(model=args.model, device=device)
    splits = load_paired_dataset(
        tokenizer=model.tokenizer,
        n_train=args.n_train, n_val=2, n_test=args.n_test,
        seq_len=args.seq_len, seed=args.seed,
        model=args.model,
    )

    # ---- v_md from train split ----
    train = splits["train"]
    train_mask = prompt_mask_from_markers(args.seq_len, train.story_marker_pos)
    print(f"[md] harvesting train activations on {hook} …")
    train_acts = cache_activations(model, train.tokens, [hook])[hook]
    md = compute_meandiff_vector(train_acts, train.is_deployment, train_mask)
    v_md = md["v_md"]
    args.out_vmd.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"v_md": v_md, "layer_hook": hook,
                "n_dep": md["n_dep"], "n_cln": md["n_cln"]},
               args.out_vmd)
    print(f"[md]   ||v_md||={v_md.norm().item():.4f}  saved → {args.out_vmd}")

    # ---- direction comparison ----
    w_dec_f = sae.W_dec[args.feature].detach().float().cpu()
    w_enc_f = sae.W_enc[:, args.feature].detach().float().cpu()
    cos_dec = F.cosine_similarity(v_md, w_dec_f, dim=0).item()
    cos_enc = F.cosine_similarity(v_md, w_enc_f, dim=0).item()
    print(f"[md] cos(v_md, W_dec[{args.feature}])={cos_dec:+.4f}  "
          f"cos(v_md, W_enc[:, {args.feature}])={cos_enc:+.4f}")
    print(f"[md] ||v_md||={v_md.norm().item():.3f}  "
          f"||W_dec||={w_dec_f.norm().item():.3f}  ||W_enc||={w_enc_f.norm().item():.3f}")

    # ---- steering effect on test split ----
    test = splits["test"]
    test_mask = prompt_mask_from_markers(args.seq_len, test.story_marker_pos)
    dep = test.tokens[test.is_deployment].to(device)
    dep_mask = test_mask[test.is_deployment].to(device)
    dep_marker = test.story_marker_pos[test.is_deployment].to(device)
    cln = test.tokens[~test.is_deployment].to(device)
    cln_mask = test_mask[~test.is_deployment].to(device)
    cln_marker = test.story_marker_pos[~test.is_deployment].to(device)
    v_md_dev = v_md.to(device)

    base_logp = teacher_forced_sleeper_logp(model, model.tokenizer, dep).mean().item()
    base_ce = clean_continuation_ce(model, cln, cln_marker).mean().item()
    base_asr = asr_with_md(model, torch.zeros_like(v_md_dev), 0.0,
                           dep, dep_mask, dep_marker, args.gen_tokens, hook)
    print(f"[md] test baseline: dep_logp={base_logp:.3f}  clean_ce={base_ce:.4f}  asr={base_asr:.3f}")

    md_rows = []
    for a in args.alphas:
        d_dep = compute_meandiff_delta(v_md_dev, dep_mask, sign=-1.0)
        d_cln = compute_meandiff_delta(v_md_dev, cln_mask, sign=-1.0)
        logp = teacher_forced_sleeper_logp(
            model, model.tokenizer, dep,
            fwd_hooks=additive_steer_hook(d_dep, a, hook)).mean().item()
        ce = clean_continuation_ce(
            model, cln, cln_marker,
            fwd_hooks=additive_steer_hook(d_cln, a, hook)).mean().item()
        asr = asr_with_md(model, v_md_dev, a, dep, dep_mask, dep_marker, args.gen_tokens, hook)
        md_rows.append({"alpha": a, "dep_logp": logp, "clean_ce": ce, "asr_16": asr,
                        "delta_logp": logp - base_logp, "delta_ce": ce - base_ce})
        print(f"[md]   md α={a:>5}: asr={asr:.3f}  Δlogp={logp-base_logp:+.3f}  "
              f"ΔCE={ce-base_ce:+.4f}")

    # ---- SAE feature comparison ----
    d_dep = compute_sae_delta(model, sae, hook, args.feature, dep, dep_mask)
    d_cln = compute_sae_delta(model, sae, hook, args.feature, cln, cln_mask)
    sae_logp = teacher_forced_sleeper_logp(
        model, model.tokenizer, dep,
        fwd_hooks=additive_steer_hook(d_dep, args.sae_alpha, hook)).mean().item()
    sae_ce = clean_continuation_ce(
        model, cln, cln_marker,
        fwd_hooks=additive_steer_hook(d_cln, args.sae_alpha, hook)).mean().item()
    sae_asr = asr_with_sae(model, sae, hook, args.feature, args.sae_alpha,
                           dep, dep_mask, dep_marker, args.gen_tokens)
    print(f"[md]   sae f={args.feature} α={args.sae_alpha}: asr={sae_asr:.3f}  "
          f"Δlogp={sae_logp-base_logp:+.3f}  ΔCE={sae_ce-base_ce:+.4f}")

    out = {
        "hook": hook, "feature": args.feature,
        "norms": {"v_md": v_md.norm().item(),
                  "W_dec": w_dec_f.norm().item(),
                  "W_enc": w_enc_f.norm().item()},
        "cosine": {"v_md_vs_W_dec": cos_dec, "v_md_vs_W_enc": cos_enc},
        "test_baseline": {"dep_logp": base_logp, "clean_ce": base_ce, "asr_16": base_asr},
        "meandiff_sweep": md_rows,
        "sae_feature": {"alpha": args.sae_alpha, "dep_logp": sae_logp,
                        "clean_ce": sae_ce, "asr_16": sae_asr,
                        "delta_logp": sae_logp - base_logp,
                        "delta_ce": sae_ce - base_ce},
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2))
    print(f"[md] wrote {args.out_json}")


if __name__ == "__main__":
    main()
