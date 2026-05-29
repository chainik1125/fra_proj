"""Conventional steering baseline, common to every matrix-cell run.

For each downstream SAE seed s ∈ --sae_seeds (default 0..5):
  1. Identify the sleeper-firing feature in sae_resid_mid_s{s}.pt:
       step 1: rank by activation diff mean(z|dep) − mean(z|clean) on prompt
               positions → top-K candidates.
       step 2: re-rank top-K by signed cos(W_dec[f], v_md/||v_md||) where
               v_md = mean(resid_dep_last_pos) − mean(resid_clean_last_pos);
               keep best K/2. Replaces v3's body-context Δlogp screen so the
               whole pipeline is prompt-only (no `prompt + body + IHY`).
       step 3: greedy ASR sweep on left-padded dep prompts → winner f_s.
  2. Evaluate f_s at each α with the same 4-metric multi-seed lockstep eval
     used by scripts/matrix.py (asr, jsd_clean, jsd_pois, exact_match), so
     baseline numbers are directly comparable to the matrix cells.

Output: one JSON per SAE type (one invocation = one --sae_dir). Layout:

  {
    "config":         …,
    "per_seed": {
       "s0": {"winner": 579, "min_asr": 0.0, "screen_alpha": 4.0,
              "top_k": [...],
              "per_alpha": {"2.0": {asr, jsd_clean, jsd_pois, exact_match, …},
                            "4.0": {...}}},
       "s1": ...
    }
  }

The matrix script reads the same "winner" field via --per_seed_targets_json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from sleeper.hooks import (
    additive_steer_hook, compute_sae_delta, generate_with_hooks,
    make_greedy_sampler,
)
from sleeper.metrics import (
    asr_16, rank_features_by_dep_clean,
)
from sleeper.model import (
    MODELS, left_pad_prompts, load_paired_dataset,
    load_sleeper_model,
)
from sleeper.sae import encode_all, load as sae_load

from sleeper.eval import (
    _build_baselines_per_seed, eval_downstream_baseline, split_dep_prompts,
)

@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model",            choices=list(MODELS), default="tinystories")
    p.add_argument("--sae_dir",          type=Path,  required=True,
                   help="Directory containing per-seed sae_resid_mid_s{seed}.pt files.")
    p.add_argument("--sae_seeds",        type=int,   nargs="+", default=[0, 1, 2, 3, 4, 5])
    p.add_argument("--identify_top_k",   type=int,   default=20,
                   help="Top-K dep-vs-clean activation candidates before screens.")
    p.add_argument("--screen_alphas",    type=float, nargs="+", default=[2.0, 4.0])
    p.add_argument("--alphas",           type=float, nargs="+",
                   default=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
                   help="Full per-α eval sweep for each per-seed winner (4-metric lockstep).")
    p.add_argument("--n_sel",            type=int,   default=200,
                   help="Selection-split size for identification.")
    p.add_argument("--n_eval",           type=int,   default=400)
    p.add_argument("--gen_tokens",       type=int,   default=16)
    p.add_argument("--eval_seeds",       type=int,   nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--eval_temperature", type=float, default=1.0)
    p.add_argument("--seq_len",          type=int,   default=128,
                   help="Per-prompt sequence length. TS=128 (Story:-prompts are short); "
                        "Llama ChatML rows average ~135 tokens — override to 256 so "
                        "the balanced loader has enough qualifying rows.")
    p.add_argument("--out",              type=Path,  required=True)
    p.add_argument("--device",           default=None)
    args = p.parse_args()
    SEQ_LEN = args.seq_len

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_sleeper_model(model=args.model, device=device)
    tok    = model.tokenizer
    pad_id = tok.pad_token_id or tok.eos_token_id

    splits = load_paired_dataset(tok, n_train=0, n_val=args.n_sel,
                                  n_test=args.n_eval, seq_len=SEQ_LEN, seed=0,
                                  model=args.model)
    sel = splits["val"]
    # Loader returns prompt-only left-padded tensors: paired.attention_mask
    # marks real prompt positions. For left-padded data, prompt_mask == attention_mask.
    sel_attn        = sel.attention_mask
    sel_dep         = sel.tokens[sel.is_deployment].to(device)
    sel_dep_attn    = sel_attn[sel.is_deployment].to(device)
    sel_cln         = sel.tokens[~sel.is_deployment].to(device)
    sel_cln_attn    = sel_attn[~sel.is_deployment].to(device)
    is_dep          = sel.is_deployment.to(device)
    combined_lp     = sel.tokens.to(device)
    combined_attn   = sel_attn.to(device)

    splits_dep = split_dep_prompts(tok, args.n_sel, args.n_eval, model=args.model)
    id_lp, id_attn = left_pad_prompts(splits_dep["sel"], pad_id)
    id_lp, id_attn = id_lp.to(device), id_attn.to(device)
    eval_dep_lp, eval_dep_attn = left_pad_prompts(splits_dep["eval"], pad_id)
    eval_dep_lp   = eval_dep_lp.to(device)
    eval_dep_attn = eval_dep_attn.to(device)

    print(f"[base] sae_dir={args.sae_dir}  seeds={args.sae_seeds}  "
          f"identify_top_k={args.identify_top_k}", flush=True)
    print(f"[base] pre-building per-seed baselines (B={eval_dep_lp.shape[0]}, "
          f"seeds={args.eval_seeds}) ...", flush=True)
    eval_clean_lsm_ps, eval_clean_tok_ps, eval_dep_lsm_ps = _build_baselines_per_seed(
        model, eval_dep_lp, eval_dep_attn, args.gen_tokens, device,
        seeds=args.eval_seeds, temperature=args.eval_temperature,
    )

    sampler = make_greedy_sampler()

    out_per_seed: dict = {}
    for seed in args.sae_seeds:
        path = args.sae_dir / f"sae_resid_mid_s{seed}.pt"
        print(f"\n[base] === seed={seed}  {path.name} ===", flush=True)
        sae_mid, sae_mid_cfg = sae_load(path, device=device)
        # Read the hook from each SAE's config — bank A/B/C use L3/L16/L29,
        # not block-0 like TinyStories. Defaults to the SAE's training hook.
        resid_hook = sae_mid_cfg["layer_hook"]

        # 1. Activation-difference ranking → top-K candidates.
        _, cache = model.run_with_cache(
            combined_lp, return_type=None,
            attention_mask=combined_attn,
            names_filter=lambda n: n == resid_hook,
        )
        z = encode_all(sae_mid, cache[resid_hook]).to(device)
        ranked = rank_features_by_dep_clean(z, is_dep, combined_attn,
                                            top_k=args.identify_top_k)
        top_k = ranked["top_indices"].cpu().tolist()
        print(f"[base]   top-{args.identify_top_k}: {top_k[:8]} ...", flush=True)

        # 2. Cosine re-rank of step-1 candidates by alignment to v_md.
        # Replaces v3's body-context Δlogp screen — fully prompt-only, no
        # P(IHY|prompt+body) anywhere. v_md = mean(resid_dep_last_pos) −
        # mean(resid_clean_last_pos). For each f in top_k score signed
        # cos(W_dec[f], v_md/||v_md||); keep best K/2 (Arditi-style ranking
        # combined with the activation-diff prefilter from step 1).
        with torch.no_grad():
            _, cache_d = model.run_with_cache(
                sel_dep, attention_mask=sel_dep_attn, return_type=None,
                names_filter=lambda n: n == resid_hook,
            )
            resid_d = cache_d[resid_hook][:, -1, :].mean(0).to(torch.float32)
            del cache_d
            _, cache_c = model.run_with_cache(
                sel_cln, attention_mask=sel_cln_attn, return_type=None,
                names_filter=lambda n: n == resid_hook,
            )
            resid_c = cache_c[resid_hook][:, -1, :].mean(0).to(torch.float32)
            del cache_c
            diff = resid_d - resid_c
            diff_n = diff / (diff.norm() + 1e-12)
            W_dec = sae_mid.W_dec.to(torch.float32)
            row_norms = W_dec.norm(dim=1).clamp_min(1e-12)
            cos_all = (W_dec @ diff_n) / row_norms                # (d_sae,)
        # Store −cos so the existing "min dlogp = best" sort + tie-break
        # logic carries over to step 3 unchanged.
        dlogp: dict[int, float] = {int(f): -float(cos_all[int(f)].item())
                                   for f in top_k}
        keep_n = max(1, args.identify_top_k // 2)
        survivors = sorted(dlogp, key=lambda k: dlogp[k])[:keep_n]
        print(f"[base]   stage-0 keep={keep_n}: {survivors[:5]} ... "
              f" best signed cos={-dlogp[survivors[0]]:+.4f}", flush=True)

        # 3. Greedy ASR screen on left-padded dep prompts → winner.
        asr_table: dict[int, tuple[float, float]] = {}
        for f in survivors:
            best_asr, best_a = 1.0, args.screen_alphas[0]
            for a in args.screen_alphas:
                delta = compute_sae_delta(model, sae_mid, resid_hook, int(f),
                                           id_lp, id_attn.bool(),
                                           attention_mask=id_attn)
                hooks = additive_steer_hook(delta, a, resid_hook)
                gen = generate_with_hooks(model, id_lp, hooks, args.gen_tokens,
                                           sampler, attention_mask=id_attn)
                asr_val = asr_16(gen, tok)
                if asr_val < best_asr:
                    best_asr, best_a = asr_val, a
            asr_table[int(f)] = (best_asr, best_a)
            print(f"[base]   f={f:>4}  min_asr={best_asr:.3f}  best_α={best_a:.1f}",
                  flush=True)
        winner = min(asr_table, key=lambda f: (asr_table[f][0], dlogp[f]))
        winner_asr, winner_alpha = asr_table[winner]
        print(f"[base] seed={seed} ⇒ winner f={winner}  min_asr={winner_asr:.3f}  "
              f"alpha={winner_alpha:.1f}", flush=True)

        # 4. Final 4-metric lockstep eval at each α.
        per_alpha: dict = {}
        for alpha in args.alphas:
            t0 = time.time()
            m = eval_downstream_baseline(
                model, sae_mid, int(winner), alpha,
                eval_dep_lp, eval_dep_attn,
                eval_clean_lsm_ps, eval_clean_tok_ps, eval_dep_lsm_ps,
                args.gen_tokens, device,
                eval_seeds=args.eval_seeds, eval_temperature=args.eval_temperature,
            )
            per_alpha[str(alpha)] = m
            print(f"[base]   [EVAL s={seed} f={winner}] α={alpha:>4.1f}  "
                  f"asr={m['asr']:.3f}  jsd_cln={m['jsd_clean']:.3f}  "
                  f"jsd_dep={m['jsd_pois']:.3f}  exact={m['exact_match']:.3f}  "
                  f"({time.time()-t0:.1f}s)", flush=True)

        out_per_seed[f"s{seed}"] = {
            "path":         str(path),
            "winner":       int(winner),
            "min_asr":      float(winner_asr),
            "screen_alpha": float(winner_alpha),
            "top_k":        top_k,
            "per_alpha":    per_alpha,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "config":   vars(args),
        "per_seed": out_per_seed,
    }, indent=2, default=str))
    print(f"\n[base] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
