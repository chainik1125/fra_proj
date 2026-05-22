"""Find the best resid_mid SAE feature per seed (0–5) for sleeper suppression.

Pipeline per seed:
  1. Activation-difference ranking (dep vs clean) → top-K candidates.
  2. Stage-0: Δdep-logp screen via additive_steer_hook at resid_mid.
  3. Stage-1: greedy ASR screen on left-padded dep prompts.
  4. Winner: min ASR (tie-break: min Δdep-logp at stage-0).

Input SAEs: weights/seeds/sae_resid_mid_s{0..5}.pt  (from train_all_saes_6seeds.py)
Output:     results/downstream_winners_6seeds.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sleeper.hooks import (
    additive_steer_hook, compute_sae_delta,
    generate_with_hooks, make_greedy_sampler,
)
from sleeper.metrics import (
    asr_16, rank_features_by_dep_clean, teacher_forced_sleeper_logp,
)
from sleeper.model import (
    left_pad_prompts, load_dep_prompts,
    load_paired_dataset, load_sleeper_model, prompt_mask_from_markers,
)
from sleeper.sae import encode_all, load as sae_load

RESID_MID     = "blocks.0.hook_resid_mid"
N_SEL         = 100
SEQ_LEN       = 128
SCREEN_ALPHAS = [2.0, 4.0]
DEFAULT_TOP_K = 20
GEN_TOKENS    = 16
N_SEEDS       = 6


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(N_SEEDS)))
    p.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--out", type=Path, default=Path("results/downstream_winners_6seeds.json"))
    args = p.parse_args()
    TOP_K = args.top_k

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = load_sleeper_model(device=device)
    tok    = model.tokenizer
    pad_id = tok.pad_token_id or tok.eos_token_id

    # Selection split (fixed-length, dep+clean)
    splits = load_paired_dataset(tok, n_train=2, n_val=N_SEL, n_test=0,
                                  seq_len=SEQ_LEN, seed=0)
    sel       = splits["val"]
    sel_pmask = prompt_mask_from_markers(SEQ_LEN, sel.story_marker_pos)
    sel_dep       = sel.tokens[sel.is_deployment].to(device)
    sel_dep_pmask = sel_pmask[sel.is_deployment].to(device)

    is_dep       = sel.is_deployment.to(device)
    combined_lp  = sel.tokens.to(device)
    combined_pmask = sel_pmask.to(device)

    # Variable-length dep prompts for greedy ASR screen
    raw_dep = load_dep_prompts(tok, N_SEL // 2, split="test")
    sel_lp, sel_attn = left_pad_prompts(raw_dep, pad_id)
    sel_lp, sel_attn = sel_lp.to(device), sel_attn.to(device)

    base_logp = teacher_forced_sleeper_logp(model, tok, sel_dep).mean().item()
    print(f"[downstream] baseline dep-logp = {base_logp:.4f}")

    sampler = make_greedy_sampler()
    out: dict = json.loads(args.out.read_text()) if args.out.exists() else {}

    for seed in args.seeds:
        path = Path(f"weights/seeds/sae_resid_mid_s{seed}.pt")
        print(f"\n[downstream] === seed={seed}  {path} ===")
        sae_mid, _ = sae_load(path, device=device)

        # Activation-difference ranking
        _, cache = model.run_with_cache(
            combined_lp, return_type=None,
            names_filter=lambda n: n == RESID_MID,
        )
        z = encode_all(sae_mid, cache[RESID_MID]).to(device)
        ranked = rank_features_by_dep_clean(z, is_dep, combined_pmask, top_k=TOP_K)
        top_k = ranked["top_indices"].cpu().tolist()
        print(f"[downstream] top-{TOP_K}: {top_k[:8]} …")

        # Stage-0: Δdep-logp screen
        dlogp: dict[int, float] = {}
        for f in top_k:
            best = float("inf")
            for a in SCREEN_ALPHAS:
                delta = compute_sae_delta(model, sae_mid, RESID_MID, int(f),
                                           sel_dep, sel_dep_pmask)
                hooks = additive_steer_hook(delta, a, RESID_MID)
                lp = teacher_forced_sleeper_logp(model, tok, sel_dep,
                                                  fwd_hooks=hooks).mean().item()
                d = lp - base_logp
                if d < best:
                    best = d
            dlogp[int(f)] = best
        keep_n     = max(1, TOP_K // 2)
        survivors  = sorted(dlogp, key=dlogp.get)[:keep_n]
        print(f"[downstream] stage-0 keep={keep_n}: {survivors[:5]} …  "
              f"best Δlogp={dlogp[survivors[0]]:+.3f}")

        # Stage-1: greedy ASR screen
        asr_table: dict[int, tuple[float, float]] = {}
        for f in survivors:
            best_asr, best_a = 1.0, SCREEN_ALPHAS[0]
            for a in SCREEN_ALPHAS:
                delta = compute_sae_delta(model, sae_mid, RESID_MID, int(f),
                                           sel_lp, sel_attn.bool(),
                                           attention_mask=sel_attn)
                hooks = additive_steer_hook(delta, a, RESID_MID)
                gen = generate_with_hooks(model, sel_lp, hooks, GEN_TOKENS,
                                           sampler, attention_mask=sel_attn)
                asr = asr_16(gen, tok)
                if asr < best_asr:
                    best_asr, best_a = asr, a
            asr_table[int(f)] = (best_asr, best_a)
            print(f"[downstream]   f={f:>5}  min_asr={best_asr:.3f}  best_α={best_a}")

        winner = min(asr_table, key=lambda f: asr_table[f][0])
        print(f"[downstream] seed={seed} ⇒ winner f={winner}  "
              f"min_asr={asr_table[winner][0]:.3f}  alpha={asr_table[winner][1]}")
        out[f"s{seed}"] = {
            "path":         str(path),
            "winner":       int(winner),
            "min_asr":      float(asr_table[winner][0]),
            "screen_alpha": float(asr_table[winner][1]),
            "top_k":        [int(x) for x in top_k],
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")
    print(json.dumps({k: v["winner"] for k, v in out.items()}, indent=2))


if __name__ == "__main__":
    main()
