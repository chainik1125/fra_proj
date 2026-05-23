"""Per-(layer, hook) feature-selection winners for the conventional baseline.

Mirrors scripts/find_downstream_winners_6seeds.py's 3-stage pipeline, applied
to every (layer × {hook_resid_mid, hook_resid_post}) site:

  1. Activation-difference ranking (mean_dep − mean_cln of SAE activations).
  2. Stage-0 Δdep-logp screen via additive_steer_hook at the target hook.
  3. Stage-1 greedy ASR screen on left-padded dep prompts.
  4. Winner: min ASR (tie-break: min Δlogp).

Input  SAEs: weights/seeds_per_layer/sae_L{L}_{kind}_s{seed}.pt
Output JSON: results/conventional_winners_per_layer.json keyed by "L{L}_{kind}".
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

HOOK_KINDS = ("hook_resid_mid", "hook_resid_post")
N_SEL = 100; SEQ_LEN = 128; SCREEN_ALPHAS = [2.0, 4.0]; GEN_TOKENS = 16


def _site_key(L: int, hk: str) -> str:
    return f"L{L}_{hk.replace('hook_', '')}"


def _sae_path(L: int, hk: str, s: int) -> Path:
    return Path("weights/seeds_per_layer") / f"sae_L{L}_{hk.replace('hook_', '')}_s{s}.pt"


@torch.no_grad()
def select_winner(model, tok, sae, layer_hook, top_k,
                  combined_lp, combined_pmask, is_dep,
                  sel_dep, sel_dep_pmask, sel_lp, sel_attn,
                  base_logp, sampler):
    _, cache = model.run_with_cache(
        combined_lp, return_type=None,
        names_filter=lambda n: n == layer_hook,
    )
    z = encode_all(sae, cache[layer_hook]).to(combined_lp.device)
    ranked = rank_features_by_dep_clean(z, is_dep, combined_pmask, top_k=top_k)
    cand = ranked["top_indices"].cpu().tolist()

    # Stage-0: Δdep-logp screen
    dlogp: dict[int, float] = {}
    for f in cand:
        best = float("inf")
        for a in SCREEN_ALPHAS:
            delta = compute_sae_delta(model, sae, layer_hook, int(f),
                                       sel_dep, sel_dep_pmask)
            hooks = additive_steer_hook(delta, a, layer_hook)
            lp = teacher_forced_sleeper_logp(model, tok, sel_dep,
                                              fwd_hooks=hooks).mean().item()
            best = min(best, lp - base_logp)
        dlogp[int(f)] = best
    survivors = sorted(dlogp, key=dlogp.get)[: max(1, top_k // 2)]

    # Stage-1: greedy ASR screen on variable-length dep prompts
    asr_table: dict[int, tuple[float, float]] = {}
    for f in survivors:
        best_asr, best_a = 1.0, SCREEN_ALPHAS[0]
        for a in SCREEN_ALPHAS:
            delta = compute_sae_delta(model, sae, layer_hook, int(f),
                                       sel_lp, sel_attn.bool(),
                                       attention_mask=sel_attn)
            hooks = additive_steer_hook(delta, a, layer_hook)
            gen = generate_with_hooks(model, sel_lp, hooks, GEN_TOKENS,
                                       sampler, attention_mask=sel_attn)
            asr = asr_16(gen, tok)
            if asr < best_asr:
                best_asr, best_a = asr, a
        asr_table[int(f)] = (best_asr, best_a)

    winner = min(asr_table, key=lambda f: (asr_table[f][0], dlogp[f]))
    return {
        "winner":       int(winner),
        "min_asr":      float(asr_table[winner][0]),
        "screen_alpha": float(asr_table[winner][1]),
        "top_k":        [int(x) for x in cand],
        "stage0_dlogp": {str(f): dlogp[f] for f in cand},
    }


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--layers", type=int, nargs="+", default=None)
    p.add_argument("--hooks", nargs="+", default=list(HOOK_KINDS),
                   choices=list(HOOK_KINDS))
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(6)))
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--out", type=Path,
                   default=Path("results/conventional_winners_per_layer.json"))
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = load_sleeper_model(device=device)
    tok    = model.tokenizer
    pad_id = tok.pad_token_id or tok.eos_token_id
    layers = list(range(model.cfg.n_layers)) if args.layers is None else args.layers

    splits = load_paired_dataset(tok, n_train=2, n_val=N_SEL, n_test=0,
                                  seq_len=SEQ_LEN, seed=0)
    sel       = splits["val"]
    sel_pmask = prompt_mask_from_markers(SEQ_LEN, sel.story_marker_pos)
    sel_dep       = sel.tokens[sel.is_deployment].to(device)
    sel_dep_pmask = sel_pmask[sel.is_deployment].to(device)
    is_dep        = sel.is_deployment.to(device)
    combined_lp   = sel.tokens.to(device)
    combined_pmask = sel_pmask.to(device)

    raw_dep = load_dep_prompts(tok, N_SEL // 2, split="test")
    sel_lp, sel_attn = left_pad_prompts(raw_dep, pad_id)
    sel_lp, sel_attn = sel_lp.to(device), sel_attn.to(device)

    base_logp = teacher_forced_sleeper_logp(model, tok, sel_dep).mean().item()
    print(f"[winners-per-layer] baseline dep-logp = {base_logp:.4f}")

    sampler = make_greedy_sampler()
    out: dict = json.loads(args.out.read_text()) if args.out.exists() else {}

    for L in layers:
        for hk in args.hooks:
            layer_hook = f"blocks.{L}.{hk}"
            key = _site_key(L, hk)
            site = out.setdefault(key, {"layer_hook": layer_hook, "per_seed": {}})
            for s in args.seeds:
                if str(s) in site["per_seed"]:
                    continue
                path = _sae_path(L, hk, s)
                if not path.exists():
                    print(f"[winners-per-layer] skip {key} s={s}: {path} missing")
                    continue
                print(f"\n[winners-per-layer] === {key} seed={s}  {path} ===")
                sae, _ = sae_load(path, device=device)
                rec = select_winner(model, tok, sae, layer_hook, args.top_k,
                                     combined_lp, combined_pmask, is_dep,
                                     sel_dep, sel_dep_pmask, sel_lp, sel_attn,
                                     base_logp, sampler)
                rec["sae_path"] = str(path)
                site["per_seed"][str(s)] = rec
                print(f"[winners-per-layer] {key} s={s}  winner={rec['winner']}  "
                      f"min_asr={rec['min_asr']:.3f}  α={rec['screen_alpha']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
