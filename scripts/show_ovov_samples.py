"""Print steered deployment continuations for the 5 ov×ov winners.

Loads each per-seed winner from `results/matrix_sweep.json`, builds the V-only
steering hook, and decodes a handful of prompts under the same sampled regime
as the eval — pure multinomial at T=1.0, seeded torch.Generator for reproducibility.

Usage:
    python -m scripts.show_ovov_samples --n_prompts 3 --gen_tokens 48 --decode_seed 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sleeper.hooks import (
    build_hooks, generate_with_hooks, make_sampling_sampler, resolve_channel_deltas,
    ACTIVE_CHANNELS,
)
from sleeper.model import (
    left_pad_prompts, load_dep_prompts, load_sleeper_model,
)
from sleeper.sae import load as sae_load

LN1_HOOK = "blocks.0.ln1.hook_normalized"


def _decode(tokenizer, ids: torch.Tensor) -> str:
    return tokenizer.decode(ids.tolist()).replace("\n", " ⏎ ")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=Path("results/matrix_sweep.json"))
    p.add_argument("--n_prompts", type=int, default=3)
    p.add_argument("--gen_tokens", type=int, default=48,
                   help="longer than the 16-token eval window so we can see story shape.")
    p.add_argument("--decode_seed", type=int, default=0,
                   help="single seed for the sampler (eval uses 5; one is enough to read).")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    payload = json.loads(args.results.read_text())
    ovov = sorted(
        [r for r in payload["results"] if r["attr"] == "ov" and r["intervene"] == "ov"],
        key=lambda r: r["seed"],
    )
    if not ovov:
        raise SystemExit("no ov×ov rows in results")

    model  = load_sleeper_model(device=device)
    tok    = model.tokenizer
    pad_id = tok.pad_token_id or tok.eos_token_id
    W      = {c: getattr(model, f"W_{c}")[0].detach().to(device) for c in ("Q", "K", "V")}

    # Held-out eval dep prompts: same boundary as matrix_sweep — first n_sel/2 sel,
    # next n_eval/2 eval. We grab from the eval slice so this matches what the
    # reported ASR was computed on.
    cfg     = payload["config"]
    n_sel_d = cfg["n_sel"] // 2
    n_eval_d = cfg["n_eval"] // 2
    raw     = load_dep_prompts(tok, n_sel_d + n_eval_d, split="test")
    eval_dep = raw[n_sel_d : n_sel_d + n_eval_d][: args.n_prompts]
    eval_lp, eval_attn = left_pad_prompts(eval_dep, pad_id)
    eval_lp, eval_attn = eval_lp.to(device), eval_attn.to(device)

    P = eval_lp.shape[1]
    print(f"\n[show] eval-split dep prompts: {args.n_prompts} of {n_eval_d} (length {P})")
    print(f"[show] decoding: T={args.temperature}, sampler seed={args.decode_seed}")
    print(f"[show] α=4.0 (matches the ov×ov α used in the sweep)")

    # Unsteered baseline once per prompt for comparison.
    print("\n" + "=" * 90 + "\n[BASELINE — unsteered, sampled]\n" + "=" * 90)
    sampler = make_sampling_sampler(
        temperature=args.temperature, seed=args.decode_seed, device=device,
    )
    base_gen = generate_with_hooks(model, eval_lp, [], args.gen_tokens, sampler,
                                   attention_mask=eval_attn)
    for b in range(args.n_prompts):
        prompt_ids = eval_lp[b][eval_attn[b].bool()]
        cont_ids   = base_gen[b]
        print(f"\nprompt {b}: «{_decode(tok, prompt_ids)}»")
        print(f"baseline → «{_decode(tok, cont_ids)}»")

    # Steered: per-seed winner.
    for row in ovov:
        seed   = row["seed"]
        tup    = [tuple(t) for t in row["winner_tuple"]]
        alpha  = row["alpha"]
        ce     = row["eval"]["delta_ce"]
        asr    = row["eval"]["asr"]
        gen_r  = row["eval"]["gen_ce_ratio"]
        sae_ln1, _ = sae_load(Path(f"weights/seeds/sae_ln1_s{seed}.pt"), device=device)
        # Channel-deltas for the V channel, hook on attn.hook_v via build_hooks.
        cd = resolve_channel_deltas(
            tup, ACTIVE_CHANNELS["ov"], model, sae_ln1, LN1_HOOK,
            eval_lp, eval_attn, eval_attn,
        )
        hooks = build_hooks(cd, alpha, ACTIVE_CHANNELS["ov"], W, LN1_HOOK, 0)
        sampler = make_sampling_sampler(
            temperature=args.temperature, seed=args.decode_seed, device=device,
        )
        gen = generate_with_hooks(model, eval_lp, hooks, args.gen_tokens, sampler,
                                  attention_mask=eval_attn)
        print(
            "\n" + "=" * 90
            + f"\n[STEERED — seed={seed}, ov×ov winner=f{tup[0][0]}, α={alpha}]"
            + f"  eval ASR={asr:.3f}  Δcln-CE={ce:+.4f}  gen-CE-ratio={gen_r:.3f}"
            + "\n" + "=" * 90
        )
        for b in range(args.n_prompts):
            cont_ids = gen[b]
            print(f"prompt {b}: → «{_decode(tok, cont_ids)}»")


if __name__ == "__main__":
    main()
