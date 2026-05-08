"""Downstream baseline: ablate sae_resid_mid feature 579 at blocks.0.hook_resid_mid.

Sweeps α ∈ {0.5, 1.0, 2.0, 4.0}. Reports ASR, Δdep-logp, Δcln-CE, gen-CE-ratio.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sleeper.hooks import additive_steer_hook, compute_sae_delta, greedy_generate_with_hooks
from sleeper.metrics import (
    asr_16, clean_continuation_ce, deployment_generation_ratio,
    teacher_forced_sleeper_logp,
)
from sleeper.model import (
    cache_activations, load_paired_dataset, load_sleeper_model, prompt_mask_from_markers,
)
from sleeper.sae import encode_all, load as sae_load

MID_HOOK = "blocks.0.hook_resid_mid"
FEAT = 579


@torch.no_grad()
def _per_group(model, tok, dep, dep_pmask, dep_marker, sae_mid, alpha, gen_tokens):
    """ASR and gen-CE-ratio computed per marker-position group.

    Variable prompt lengths force per-marker grouping for generation; the
    ratio's num/den sums are accumulated across groups and divided once at
    the end so each side is averaged independently before division.
    """
    asr_hits = asr_total = 0
    num_sum = den_sum = 0.0
    for m in dep_marker.unique().tolist():
        rows = (dep_marker == m).nonzero(as_tuple=True)[0]
        P = int(m) + 1
        prompts = dep[rows, :P]
        pm      = dep_pmask[rows, :P]
        d = compute_sae_delta(model, sae_mid, MID_HOOK, FEAT, prompts, pm)
        hooks = additive_steer_hook(d, alpha, MID_HOOK)
        gen = greedy_generate_with_hooks(model, prompts, hooks, gen_tokens)
        asr_hits  += int(round(asr_16(gen, tok) * gen.shape[0]))
        asr_total += gen.shape[0]
        r = deployment_generation_ratio(model, prompts, fwd_hooks=hooks, gen_tokens=gen_tokens)
        num_sum += r["num_sum"]
        den_sum += r["den_sum"]
    return asr_hits / max(1, asr_total), num_sum / max(den_sum, 1e-12)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sae_mid",     type=Path,  default=Path("weights/sae_resid_mid.pt"))
    p.add_argument("--alphas",      type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0])
    p.add_argument("--n_test",      type=int,   default=200)
    p.add_argument("--gen_tokens",  type=int,   default=16)
    p.add_argument("--out",         type=Path,  default=Path("weights/downstream_baseline.json"))
    p.add_argument("--device",      default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    sae_mid, _ = sae_load(args.sae_mid, device=device)
    model = load_sleeper_model(device=device)
    tok   = model.tokenizer

    splits = load_paired_dataset(tok, n_train=2, n_val=200, n_test=args.n_test, seq_len=128, seed=0)
    test   = splits["test"]
    pmask  = prompt_mask_from_markers(128, test.story_marker_pos)

    dep        = test.tokens[test.is_deployment].to(device)
    dep_pmask  = pmask[test.is_deployment].to(device)
    dep_marker = test.story_marker_pos[test.is_deployment].to(device)
    cln        = test.tokens[~test.is_deployment].to(device)
    cln_pmask  = pmask[~test.is_deployment].to(device)
    cln_marker = test.story_marker_pos[~test.is_deployment].to(device)

    # Pre-cache analytic deltas for Δdep-logp and Δcln-CE (exact for linear decoder)
    W_dec     = sae_mid.W_dec.detach().cpu().float()
    z_dep     = encode_all(sae_mid, cache_activations(model, dep.cpu(), [MID_HOOK])[MID_HOOK])
    delta_dep = (-z_dep[..., FEAT:FEAT+1] * W_dec[FEAT]) * dep_pmask.cpu().float().unsqueeze(-1)
    z_cln     = encode_all(sae_mid, cache_activations(model, cln.cpu(), [MID_HOOK])[MID_HOOK])
    delta_cln = (-z_cln[..., FEAT:FEAT+1] * W_dec[FEAT]) * cln_pmask.cpu().float().unsqueeze(-1)

    base_logp = teacher_forced_sleeper_logp(model, tok, dep).mean().item()
    base_ce   = clean_continuation_ce(model, cln, cln_marker).mean().item()
    base_asr  = 1.0  # unsteered model always outputs the sleeper phrase on deployment prompts
    print(f"[dn] baseline: dep_logp={base_logp:.3f}  clean_ce={base_ce:.4f}  asr={base_asr:.3f}")

    rows = []
    for alpha in args.alphas:
        dd = delta_dep.to(device)
        dc = delta_cln.to(device)
        logp = teacher_forced_sleeper_logp(
            model, tok, dep, fwd_hooks=additive_steer_hook(dd, alpha, MID_HOOK),
        ).mean().item()
        ce = clean_continuation_ce(
            model, cln, cln_marker, fwd_hooks=additive_steer_hook(dc, alpha, MID_HOOK),
        ).mean().item()
        asr, gen_ce_ratio = _per_group(model, tok, dep, dep_pmask, dep_marker,
                                        sae_mid, alpha, args.gen_tokens)
        rows.append({"alpha": alpha, "asr": asr, "delta_dep_logp": logp - base_logp,
                     "delta_cln_ce": ce - base_ce, "gen_ce_ratio": gen_ce_ratio})
        print(f"[dn]   α={alpha}  asr={asr:.3f}  Δdep-logp={logp-base_logp:+.3f}  "
              f"Δcln-CE={ce-base_ce:+.4f}  gen-CE-ratio={gen_ce_ratio:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "config": {"feature": FEAT, "hook": MID_HOOK, "alphas": args.alphas},
        "baseline": {"dep_logp": base_logp, "clean_ce": base_ce, "asr": base_asr},
        "sweep": rows,
    }, indent=2))
    print(f"[dn] wrote {args.out}")


if __name__ == "__main__":
    main()
