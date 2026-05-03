"""OV-only steering of seed-0 ln1 feature 1114, both directions.

For each setting reports ASR_16, teacher-forced sleeper Δlogp, ΔCE, distinct_2,
and repeated-3gram fraction. Two regimes:

  (a) AGAINST on dep prompts — α > 0 (over-ablate). Should suppress sleeper.
  (b) WITH on cln prompts via SAE-delta — α < 0. f=1114 is silent on clean
      prompts so this is a no-op and is here only to confirm the asymmetry.
      Decoder-direction injection lives in scripts/_f1114_elicit.py.

Decoding is configurable. RNG matching: the sampler is freshly constructed
per generation call with a fixed seed, so baseline and steered runs draw the
same uniforms each step — the hook intervention is the only difference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sleeper.hooks import (
    compute_sae_delta, generate_with_hooks, make_greedy_sampler,
    make_nucleus_sampler, ov_only_steer_hook,
)
from sleeper.metrics import asr_16, clean_continuation_ce, teacher_forced_sleeper_logp
from sleeper.model import load_paired_dataset, load_sleeper_model, prompt_mask_from_markers
from sleeper.sae import load


def distinct_2(rows: torch.Tensor) -> float:
    total, seen = 0, set()
    for row in rows.tolist():
        for i in range(len(row) - 1):
            total += 1
            seen.add((row[i], row[i + 1]))
    return len(seen) / max(1, total)


def repeat_3gram_frac(rows: torch.Tensor) -> float:
    hits = 0
    for row in rows.tolist():
        seen, bad = set(), False
        for i in range(len(row) - 2):
            g = (row[i], row[i + 1], row[i + 2])
            if g in seen:
                bad = True
                break
            seen.add(g)
        hits += int(bad)
    return hits / max(1, rows.shape[0])


def make_sampler(decoding: str, seed: int, temperature: float, top_p: float, device):
    if decoding == "greedy":
        return make_greedy_sampler()
    if decoding == "nucleus":
        return make_nucleus_sampler(
            temperature=temperature, top_p=top_p, seed=seed, device=device,
        )
    raise ValueError(f"unknown decoding: {decoding}")


@torch.no_grad()
def gen_per_marker(model, sae_ln1, ln1_hook, feature, alpha, W_V, block,
                   tokens, mask, marker, gen_tokens, mk_sampler):
    """Per-marker generation with OV-only hook. `mk_sampler()` is called once
    per marker subset; pass the same builder for baseline and steered to get
    RNG-matched draws."""
    chunks = []
    for m_pos in marker.unique().tolist():
        rows = (marker == m_pos).nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            continue
        P = int(m_pos) + 1
        trunc, trunc_mask = tokens[rows, :P], mask[rows, :P]
        if alpha == 0.0:
            hooks = []
        else:
            d = compute_sae_delta(model, sae_ln1, ln1_hook, feature, trunc, trunc_mask)
            hooks = ov_only_steer_hook(d, alpha, W_V, block=block)
        gen = generate_with_hooks(model, trunc, hooks, gen_tokens, mk_sampler())
        chunks.append(gen.cpu())
    return torch.cat(chunks, dim=0) if chunks else torch.empty(0, gen_tokens, dtype=torch.long)


def metrics_for(model, gen, tok, marker, hooks, base_logp, base_ce):
    logp = teacher_forced_sleeper_logp(model, model.tokenizer, tok, fwd_hooks=hooks).mean().item()
    ce = clean_continuation_ce(model, tok, marker, fwd_hooks=hooks).mean().item()
    return {
        "asr_16": asr_16(gen, model.tokenizer),
        "sleeper_logp": logp, "delta_logp": logp - base_logp,
        "self_ce": ce, "delta_ce": ce - base_ce,
        "distinct_2": distinct_2(gen),
        "repeat_3gram": repeat_3gram_frac(gen),
    }


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--decoding", choices=["greedy", "nucleus"], default="nucleus")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_test", type=int, default=200)
    p.add_argument("--out", type=Path, default=Path("weights/seeds/f1114_against_and_with.json"))
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sae_ln1, ln1_cfg = load(Path("weights/seeds/sae_ln1_s0.pt"), device=device)
    ln1_hook = ln1_cfg["layer_hook"]
    block, feature, seq_len, gen_tokens = 0, 1114, 128, 16

    model = load_sleeper_model(device=device)
    W_V = model.W_V[block].detach().to(device)

    splits = load_paired_dataset(
        tokenizer=model.tokenizer, n_train=2, n_val=2, n_test=args.n_test,
        seq_len=seq_len, seed=0,
    )
    test = splits["test"]
    pmask = prompt_mask_from_markers(seq_len, test.story_marker_pos)

    def split_of(is_dep: bool):
        sel = test.is_deployment if is_dep else ~test.is_deployment
        return (test.tokens[sel].to(device), pmask[sel].to(device),
                test.story_marker_pos[sel].to(device))
    dep, dep_mask, dep_marker = split_of(True)
    cln, cln_mask, cln_marker = split_of(False)
    print(f"[f1114] decoding={args.decoding}  n_dep={dep.shape[0]} n_cln={cln.shape[0]}")

    def mk_sampler():
        return make_sampler(args.decoding, args.seed, args.temperature, args.top_p, device)

    # ---- baselines ----
    base_dep_logp = teacher_forced_sleeper_logp(model, model.tokenizer, dep).mean().item()
    base_cln_logp = teacher_forced_sleeper_logp(model, model.tokenizer, cln).mean().item()
    base_dep_ce = clean_continuation_ce(model, dep, dep_marker).mean().item()
    base_cln_ce = clean_continuation_ce(model, cln, cln_marker).mean().item()
    base_dep_gen = gen_per_marker(model, sae_ln1, ln1_hook, feature, 0.0, W_V, block,
                                  dep, dep_mask, dep_marker, gen_tokens, mk_sampler)
    base_cln_gen = gen_per_marker(model, sae_ln1, ln1_hook, feature, 0.0, W_V, block,
                                  cln, cln_mask, cln_marker, gen_tokens, mk_sampler)
    base = {
        "dep": metrics_for(model, base_dep_gen, dep, dep_marker, None, base_dep_logp, base_dep_ce),
        "cln": metrics_for(model, base_cln_gen, cln, cln_marker, None, base_cln_logp, base_cln_ce),
    }
    print("[f1114] baseline:", json.dumps(base, indent=2))

    settings = [("dep", +1.0), ("dep", +2.0),
                ("cln", -1.0), ("cln", -2.0), ("cln", -4.0)]
    rows: list[dict] = []
    samples = {"baseline_dep": [model.tokenizer.decode(r) for r in base_dep_gen[:4].tolist()],
               "baseline_cln": [model.tokenizer.decode(r) for r in base_cln_gen[:4].tolist()]}

    for split, alpha in settings:
        if split == "dep":
            tok, m, mk, b_ce, b_lp = dep, dep_mask, dep_marker, base_dep_ce, base_dep_logp
        else:
            tok, m, mk, b_ce, b_lp = cln, cln_mask, cln_marker, base_cln_ce, base_cln_logp
        d = compute_sae_delta(model, sae_ln1, ln1_hook, feature, tok, m)
        hooks = ov_only_steer_hook(d, alpha, W_V, block=block)
        gen = gen_per_marker(model, sae_ln1, ln1_hook, feature, alpha, W_V, block,
                             tok, m, mk, gen_tokens, mk_sampler)
        r = {"split": split, "alpha": alpha,
             **metrics_for(model, gen, tok, mk, hooks, b_lp, b_ce)}
        rows.append(r)
        samples[f"{split}_a{alpha:+.1f}"] = [model.tokenizer.decode(g) for g in gen[:4].tolist()]
        print(f"[f1114] {split} α={alpha:+.1f}: asr={r['asr_16']:.3f} "
              f"Δlogp={r['delta_logp']:+.3f} ΔCE={r['delta_ce']:+.4f} "
              f"d2={r['distinct_2']:.3f} rep3={r['repeat_3gram']:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "config": {"feature": feature, "ln1_hook": ln1_hook, "block": block,
                   "decoding": args.decoding, "temperature": args.temperature,
                   "top_p": args.top_p, "seed": args.seed,
                   "n_test_each": int(dep.shape[0])},
        "baseline": base, "sweep": rows, "samples": samples,
    }, indent=2))
    print(f"[f1114] wrote {args.out}")


if __name__ == "__main__":
    main()
