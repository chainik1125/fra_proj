"""Evaluate OV+OV and OV+ALL winner features across seeds.

Winner features are read from matrix sweep JSON files. For each seed and method,
reports ASR, Δdep-logp, Δcln-CE, and the gen-CE ratio.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sleeper.hooks import (
    ACTIVE_CHANNELS, build_hooks, greedy_generate_with_hooks, resolve_channel_deltas,
)
from sleeper.metrics import (
    asr_16, clean_continuation_ce, deployment_generation_ratio,
    teacher_forced_sleeper_logp,
)
from sleeper.model import (
    load_paired_dataset, load_sleeper_model, prompt_mask_from_markers,
)
from sleeper.sae import load as sae_load

LN1_HOOK = "blocks.0.ln1.hook_normalized"


@torch.no_grad()
def _per_group(model, tok, dep, dep_pmask, dep_marker, sae_ln1, sel, active, alpha, gen_tokens):
    """ASR and gen-CE-ratio grouped by marker position (variable-length prompts).

    The ratio is aggregated by summing num/den across all marker groups and
    dividing once at the end — preserves the "average each side independently
    before division" semantics across variable-length-prompt batches.
    """
    asr_hits = asr_total = 0
    num_sum = den_sum = 0.0
    W = {c: getattr(model, f"W_{c}")[0].detach().to(dep.device) for c in ("Q", "K", "V")}
    for m in dep_marker.unique().tolist():
        rows    = (dep_marker == m).nonzero(as_tuple=True)[0]
        P       = int(m) + 1
        prompts = dep[rows, :P]
        pm      = dep_pmask[rows, :P]
        cd      = resolve_channel_deltas(sel, active, model, sae_ln1, LN1_HOOK, prompts, pm)
        hooks   = build_hooks(cd, alpha, active, W, LN1_HOOK, 0)
        gen     = greedy_generate_with_hooks(model, prompts, hooks, gen_tokens)
        asr_hits  += int(round(asr_16(gen, tok) * gen.shape[0]))
        asr_total += gen.shape[0]
        r = deployment_generation_ratio(model, prompts, fwd_hooks=hooks, gen_tokens=gen_tokens)
        num_sum += r["num_sum"]
        den_sum += r["den_sum"]
    return asr_hits / max(1, asr_total), num_sum / max(den_sum, 1e-12)


@torch.no_grad()
def _logp_and_ce(model, tok, dep, dep_pmask, cln, cln_pmask, cln_marker,
                 sae_ln1, sel, active, alpha):
    W = {c: getattr(model, f"W_{c}")[0].detach().to(dep.device) for c in ("Q", "K", "V")}
    cd_dep = resolve_channel_deltas(sel, active, model, sae_ln1, LN1_HOOK, dep, dep_pmask)
    cd_cln = resolve_channel_deltas(sel, active, model, sae_ln1, LN1_HOOK, cln, cln_pmask)
    h_dep  = build_hooks(cd_dep, alpha, active, W, LN1_HOOK, 0)
    h_cln  = build_hooks(cd_cln, alpha, active, W, LN1_HOOK, 0)
    logp   = teacher_forced_sleeper_logp(model, tok, dep, fwd_hooks=h_dep).mean().item()
    ce     = clean_continuation_ce(model, cln, cln_marker, fwd_hooks=h_cln).mean().item()
    return logp, ce


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_s0",   type=Path, default=Path("results/matrix_sweep_s0.json"))
    p.add_argument("--sweep_s1234",type=Path, default=Path("results/matrix_sweep.json"))
    p.add_argument("--seeds",      type=int,  nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--gen_tokens", type=int,  default=16)
    p.add_argument("--n_test",     type=int,  default=200)
    p.add_argument("--out",        type=Path, default=Path("results/upstream_winners_eval.json"))
    p.add_argument("--device",     default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_sleeper_model(device=device)
    tok    = model.tokenizer

    splits = load_paired_dataset(tok, n_train=2, n_val=200, n_test=args.n_test, seq_len=128, seed=0)
    test   = splits["test"]
    pmask  = prompt_mask_from_markers(128, test.story_marker_pos)

    dep        = test.tokens[test.is_deployment].to(device)
    dep_pmask  = pmask[test.is_deployment].to(device)
    dep_marker = test.story_marker_pos[test.is_deployment].to(device)
    cln        = test.tokens[~test.is_deployment].to(device)
    cln_pmask  = pmask[~test.is_deployment].to(device)
    cln_marker = test.story_marker_pos[~test.is_deployment].to(device)

    base_logp = teacher_forced_sleeper_logp(model, tok, dep).mean().item()
    base_ce   = clean_continuation_ce(model, cln, cln_marker).mean().item()
    print(f"baseline: dep_logp={base_logp:.3f}  clean_ce={base_ce:.4f}")

    # Load winner features from matrix sweep JSONs
    winners: dict[int, dict[str, dict]] = {}
    for fname in [args.sweep_s0, args.sweep_s1234]:
        data = json.loads(Path(fname).read_text())
        for r in data["results"]:
            s = r["seed"]
            if s not in args.seeds:
                continue
            if r["attr"] != "ov":
                continue
            iv = r["intervene"]
            if iv in ("ov", "qk+ov"):
                winners.setdefault(s, {})[iv] = {
                    "feat": r["winner_tuple"][0][0],
                    "alpha": r["alpha"],
                }

    all_results = []
    for seed in args.seeds:
        sae_ln1, _ = sae_load(f"weights/seeds/sae_ln1_s{seed}.pt", device=device)
        for intervene in ("ov", "qk+ov"):
            w      = winners[seed][intervene]
            feat   = w["feat"]
            alpha  = w["alpha"]
            sel    = [(feat, "V")]
            active = ACTIVE_CHANNELS[intervene]
            method = "OV+OV" if intervene == "ov" else "OV+QK+OV"

            logp, ce = _logp_and_ce(model, tok, dep, dep_pmask, cln, cln_pmask, cln_marker,
                                     sae_ln1, sel, active, alpha)
            asr, gen_ce_ratio = _per_group(model, tok, dep, dep_pmask, dep_marker,
                                            sae_ln1, sel, active, alpha, args.gen_tokens)

            row = {
                "seed": seed, "method": method, "feat": feat, "alpha": alpha,
                "asr": asr, "delta_dep_logp": logp - base_logp,
                "delta_cln_ce": ce - base_ce, "gen_ce_ratio": gen_ce_ratio,
            }
            all_results.append(row)
            print(f"seed={seed}  {method}  f{feat:4d}  α={alpha}  "
                  f"ASR={asr:.3f}  Δdep-logp={logp-base_logp:+.3f}  "
                  f"Δcln-CE={ce-base_ce:+.4f}  gen-CE-ratio={gen_ce_ratio:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "baseline": {"dep_logp": base_logp, "clean_ce": base_ce},
        "results": all_results,
    }, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
