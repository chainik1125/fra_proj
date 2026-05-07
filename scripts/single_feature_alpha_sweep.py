"""Per-seed α-sweep on the ov×ov winner feature, sampled multi-seed eval.

For the headline single-feature tradeoff plot. Same eval methodology as
matrix_sweep — sampled ASR + sampled Δgen-CE on the held-out split, 5
sampling seeds at T=1.0, plus deterministic Δcln-CE.

Usage:
    python -m scripts.single_feature_alpha_sweep --in results/matrix_sweep.json \\
        --alphas 0 0.5 1.0 2.0 4.0 --out results/single_feature_alpha_sweep.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sleeper.hooks import (
    ACTIVE_CHANNELS, build_hooks, generate_with_hooks, make_sampling_sampler,
    resolve_channel_deltas,
)
from sleeper.metrics import (
    asr_16, clean_continuation_ce, deployment_generation_ce,
    teacher_forced_sleeper_logp,
)
from sleeper.model import (
    left_pad_prompts, load_dep_prompts, load_paired_dataset, load_sleeper_model,
    prompt_mask_from_markers,
)
from sleeper.sae import load as sae_load

LN1_HOOK = "blocks.0.ln1.hook_normalized"


def _ovov_winners(matrix_sweep_json: Path) -> dict[int, int]:
    """seed -> winner feature index, pulled from matrix_sweep.json ov×ov rows."""
    payload = json.loads(matrix_sweep_json.read_text())
    out: dict[int, int] = {}
    for r in payload["results"]:
        if r["attr"] == "ov" and r["intervene"] == "ov":
            out[int(r["seed"])] = int(r["winner_tuple"][0][0])
    return out


@torch.no_grad()
def _eval_one(
    model, sae_ln1, feature, alpha, W,
    eval_dep, eval_dep_pmask,
    eval_dep_lp, eval_dep_attn,
    eval_cln, eval_cln_marker,
    eval_gen_dep, eval_gen_attn,
    base_logp, base_ce, gen_tokens, eval_seeds, eval_temp, device,
):
    """Compute sampled eval metrics for one (feature, α) — all-16-heads ov×ov hook."""
    sel = [(int(feature), "V")]
    active = ACTIVE_CHANNELS["ov"]

    cd_dep = resolve_channel_deltas(sel, active, model, sae_ln1, LN1_HOOK,
                                    eval_dep, eval_dep_pmask)
    h_dep  = build_hooks(cd_dep, alpha, active, W, LN1_HOOK, 0)
    e_logp = teacher_forced_sleeper_logp(model, model.tokenizer, eval_dep,
                                         fwd_hooks=h_dep).mean().item()

    cln_pmask = prompt_mask_from_markers(eval_cln.shape[1], eval_cln_marker.cpu()).to(device)
    cd_cln = resolve_channel_deltas(sel, active, model, sae_ln1, LN1_HOOK,
                                    eval_cln, cln_pmask)
    h_cln  = build_hooks(cd_cln, alpha, active, W, LN1_HOOK, 0)
    e_ce   = clean_continuation_ce(model, eval_cln, eval_cln_marker,
                                   fwd_hooks=h_cln).mean().item()

    cd_lp  = resolve_channel_deltas(sel, active, model, sae_ln1, LN1_HOOK,
                                    eval_dep_lp, eval_dep_attn, eval_dep_attn)
    h_lp   = build_hooks(cd_lp, alpha, active, W, LN1_HOOK, 0)
    asrs = []
    for s in eval_seeds:
        sampler = make_sampling_sampler(temperature=eval_temp, seed=int(s), device=device)
        gen = generate_with_hooks(model, eval_dep_lp, h_lp, gen_tokens, sampler,
                                  attention_mask=eval_dep_attn)
        asrs.append(asr_16(gen, model.tokenizer))

    cd_gen = resolve_channel_deltas(sel, active, model, sae_ln1, LN1_HOOK,
                                    eval_gen_dep, eval_gen_attn, eval_gen_attn)
    h_gen  = build_hooks(cd_gen, alpha, active, W, LN1_HOOK, 0)
    dgens = []
    for s in eval_seeds:
        sampler = make_sampling_sampler(temperature=eval_temp, seed=int(s), device=device)
        dgens.append(deployment_generation_ce(
            model, eval_gen_dep, fwd_hooks=h_gen, gen_tokens=gen_tokens,
            attention_mask=eval_gen_attn, sampler=sampler,
        ).mean().item())

    return {
        "asr": sum(asrs) / len(asrs), "asr_per_seed": asrs,
        "delta_logp": e_logp - base_logp,
        "delta_ce":   e_ce - base_ce,
        "delta_gen_ce": sum(dgens) / len(dgens),
        "delta_gen_ce_per_seed": dgens,
    }


@torch.no_grad()
def _eval_baseline_asr(
    model, eval_dep_lp, eval_dep_attn, gen_tokens, eval_seeds, eval_temp, device,
):
    asrs = []
    for s in eval_seeds:
        sampler = make_sampling_sampler(temperature=eval_temp, seed=int(s), device=device)
        gen = generate_with_hooks(model, eval_dep_lp, [], gen_tokens, sampler,
                                  attention_mask=eval_dep_attn)
        asrs.append(asr_16(gen, model.tokenizer))
    return sum(asrs) / len(asrs), asrs


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in",  dest="inp", type=Path, default=Path("results/matrix_sweep.json"),
                   help="matrix_sweep.json — winners are read from its ov×ov rows.")
    p.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0, 4.0])
    p.add_argument("--seeds",  type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--n_sel",  type=int, default=100)
    p.add_argument("--n_eval", type=int, default=50)
    p.add_argument("--n_gen_ce", type=int, default=25)
    p.add_argument("--gen_tokens", type=int, default=16)
    p.add_argument("--eval_seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--eval_temperature", type=float, default=1.0)
    p.add_argument("--out", type=Path, default=Path("results/single_feature_alpha_sweep.json"))
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    winners = _ovov_winners(args.inp)
    missing = [s for s in args.seeds if s not in winners]
    if missing:
        raise SystemExit(f"no ov×ov winner in {args.inp} for seeds {missing}")

    model  = load_sleeper_model(device=device)
    tok    = model.tokenizer
    pad_id = tok.pad_token_id or tok.eos_token_id
    W      = {c: getattr(model, f"W_{c}")[0].detach().to(device) for c in ("Q", "K", "V")}

    splits = load_paired_dataset(tok, n_train=2, n_val=args.n_sel,
                                 n_test=args.n_eval, seq_len=128, seed=0)
    ev = splits["test"]
    eval_pmask = prompt_mask_from_markers(128, ev.story_marker_pos)
    eval_dep         = ev.tokens[ev.is_deployment].to(device)
    eval_dep_pmask   = eval_pmask[ev.is_deployment].to(device)
    eval_cln         = ev.tokens[~ev.is_deployment].to(device)
    eval_cln_marker  = ev.story_marker_pos[~ev.is_deployment].to(device)

    raw_dep = load_dep_prompts(tok, args.n_sel // 2 + args.n_eval // 2, split="test")
    eval_lp, eval_attn = left_pad_prompts(raw_dep[args.n_sel // 2 :
                                                    args.n_sel // 2 + args.n_eval // 2], pad_id)
    eval_lp, eval_attn = eval_lp.to(device), eval_attn.to(device)
    eval_gen_dep  = eval_lp[: args.n_gen_ce]
    eval_gen_attn = eval_attn[: args.n_gen_ce]

    base_logp = teacher_forced_sleeper_logp(model, tok, eval_dep).mean().item()
    base_ce   = clean_continuation_ce(model, eval_cln, eval_cln_marker).mean().item()
    base_asr, base_asr_per_seed = _eval_baseline_asr(
        model, eval_lp, eval_attn, args.gen_tokens, args.eval_seeds, args.eval_temperature, device,
    )
    print(f"[αsweep] eval baseline: asr={base_asr:.3f}  Δlogp_base={base_logp:.3f}  "
          f"cln_CE_base={base_ce:.4f}")

    points = []
    for sae_seed in args.seeds:
        f = winners[sae_seed]
        sae_ln1, _ = sae_load(Path(f"weights/seeds/sae_ln1_s{sae_seed}.pt"), device=device)
        for alpha in args.alphas:
            if alpha == 0.0:
                e = {"asr": base_asr, "asr_per_seed": base_asr_per_seed,
                     "delta_logp": 0.0, "delta_ce": 0.0,
                     "delta_gen_ce": 0.0, "delta_gen_ce_per_seed": [0.0]*len(args.eval_seeds)}
            else:
                e = _eval_one(model, sae_ln1, f, alpha, W,
                              eval_dep, eval_dep_pmask, eval_lp, eval_attn,
                              eval_cln, eval_cln_marker, eval_gen_dep, eval_gen_attn,
                              base_logp, base_ce, args.gen_tokens,
                              args.eval_seeds, args.eval_temperature, device)
            print(f"[αsweep] seed={sae_seed}  f{f}  α={alpha:>4}  "
                  f"asr={e['asr']:.3f}  Δcln-CE={e['delta_ce']:+.4f}  "
                  f"Δgen-CE={e['delta_gen_ce']:+.4f}")
            points.append({"sae_seed": sae_seed, "feature": f, "alpha": alpha, **e})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "config": vars(args) | {"seeds": args.seeds, "eval_seeds": args.eval_seeds,
                                "alphas": args.alphas},
        "decoding": {"asr": {"mode": "sample", "temperature": args.eval_temperature,
                             "seeds": args.eval_seeds},
                     "delta_gen_ce": {"mode": "sample", "temperature": args.eval_temperature,
                                      "seeds": args.eval_seeds}},
        "baseline": {"asr": base_asr, "asr_per_seed": base_asr_per_seed,
                     "dep_logp": base_logp, "clean_ce": base_ce},
        "winners": winners,
        "points": points,
    }, indent=2, default=str))
    print(f"\n[αsweep] wrote {args.out}")


if __name__ == "__main__":
    main()
